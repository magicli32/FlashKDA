import os
import time
import statistics
import torch
import torch.distributed as dist
import flash_kda

torch.set_grad_enabled(False)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group("nccl", device_id=device)

rank = dist.get_rank()
world = dist.get_world_size()

assert world == 2

# ============================================================
# Configuration
# ============================================================
B = 1
T = 8192
SPLIT = 4096
H = 96
D = 128

PIPE_DEPTH = int(os.environ.get("PIPE_DEPTH", "2"))
WARMUP_REQS = 6
TIMED_REQS = int(os.environ.get("TIMED_REQS", "30"))

scale = D ** -0.5
lower_bound = -5.0

STATE_MIB = (
    B * H * D * D * 2
    / 1024 / 1024
)

# ============================================================
# Same request payload on both ranks.
# We reuse it deliberately: benchmark pipeline scheduling,
# not input allocation.
# ============================================================
gen = torch.Generator(device=device)
gen.manual_seed(20260819)

def randn(shape):
    return torch.randn(
        shape,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )

q = randn((B,T,H,D))
k = randn((B,T,H,D))
v = randn((B,T,H,D))
g = randn((B,T,H,D))
beta = randn((B,T,H))

A_log = torch.zeros(
    H,
    device=device,
    dtype=torch.float32,
)

dt_bias = torch.zeros(
    H,D,
    device=device,
    dtype=torch.float32,
)

q0 = q[:,:SPLIT].contiguous()
k0 = k[:,:SPLIT].contiguous()
v0 = v[:,:SPLIT].contiguous()
g0 = g[:,:SPLIT].contiguous()
b0 = beta[:,:SPLIT].contiguous()

q1 = q[:,SPLIT:].contiguous()
k1 = k[:,SPLIT:].contiguous()
v1 = v[:,SPLIT:].contiguous()
g1 = g[:,SPLIT:].contiguous()
b1 = beta[:,SPLIT:].contiguous()

# ============================================================
# Buffers
# ============================================================
out0 = torch.empty_like(v0)
out1 = torch.empty_like(v1)

state_shape = (B,H,D,D)

send_states = [
    torch.empty(
        state_shape,
        device=device,
        dtype=torch.bfloat16,
    )
    for _ in range(PIPE_DEPTH)
]

recv_states = [
    torch.empty(
        state_shape,
        device=device,
        dtype=torch.bfloat16,
    )
    for _ in range(PIPE_DEPTH)
]

final_state = torch.empty(
    state_shape,
    device=device,
    dtype=torch.bfloat16,
)

# rank0: protect state buffer until send finishes.
send_works = [None] * PIPE_DEPTH

# rank1: protect receive buffer until KDA consuming it finishes.
consume_events = [
    torch.cuda.Event()
    for _ in range(PIPE_DEPTH)
]

consume_valid = [False] * PIPE_DEPTH


def run_chunk0(state_out):
    flash_kda.fwd(
        q0,k0,v0,g0,b0,
        scale,
        out0,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=None,
        final_state=state_out,
    )


def run_chunk1(state_in):
    flash_kda.fwd(
        q1,k1,v1,g1,b1,
        scale,
        out1,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=state_in,
        final_state=final_state,
    )


def batch_send(tensor):
    op = dist.P2POp(
        dist.isend,
        tensor,
        1,
    )

    works = dist.batch_isend_irecv([op])
    return works[0]


def batch_recv(tensor):
    op = dist.P2POp(
        dist.irecv,
        tensor,
        0,
    )

    works = dist.batch_isend_irecv([op])
    return works[0]


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()


# ============================================================
# Pipeline execution
# ============================================================
def run_pipeline(num_reqs):

    # Reset local tracking state.
    for i in range(PIPE_DEPTH):
        send_works[i] = None
        consume_valid[i] = False

    sync_all()

    t0 = time.perf_counter()

    if rank == 0:

        for i in range(num_reqs):

            slot = i % PIPE_DEPTH

            # Do not overwrite state while previous send
            # from this slot is still using it.
            if send_works[slot] is not None:
                send_works[slot].wait()

            # Stage 0 compute.
            run_chunk0(
                send_states[slot]
            )

            # NCCL stream will observe dependency on
            # state produced by current CUDA stream.
            send_works[slot] = batch_send(
                send_states[slot]
            )

        # Pipeline drain.
        for w in send_works:
            if w is not None:
                w.wait()

    else:

        for i in range(num_reqs):

            slot = i % PIPE_DEPTH

            # Before receiving into a reused slot,
            # make sure previous KDA has finished
            # reading that state buffer.
            if consume_valid[slot]:
                consume_events[slot].synchronize()

            # Post receive as early as possible.
            recv_work = batch_recv(
                recv_states[slot]
            )

            # Establish receive -> KDA dependency.
            recv_work.wait()

            # Stage 1 compute.
            run_chunk1(
                recv_states[slot]
            )

            # Record when this slot is safe to reuse.
            consume_events[slot].record()
            consume_valid[slot] = True

        torch.cuda.synchronize()

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000

    # Overall pipeline makespan = slower rank.
    result = torch.tensor(
        [elapsed_ms],
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        result,
        op=dist.ReduceOp.MAX,
    )

    sync_all()

    return result.item()


# ============================================================
# Warmup
# ============================================================
run_pipeline(WARMUP_REQS)

# ============================================================
# Timed repeats
# ============================================================
times = []

for _ in range(5):
    times.append(
        run_pipeline(TIMED_REQS)
    )

# ============================================================
# Correctness of final request
# ============================================================
out_full = torch.empty_like(v)

state_full = torch.empty(
    state_shape,
    device=device,
    dtype=torch.bfloat16,
)

flash_kda.fwd(
    q,k,v,g,beta,
    scale,
    out_full,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=state_full,
)

torch.cuda.synchronize()

if rank == 1:

    out_equal = torch.equal(
        out1,
        out_full[:,SPLIT:]
    )

    state_equal = torch.equal(
        final_state,
        state_full,
    )

    out_diff = (
        out1.float()
        - out_full[:,SPLIT:].float()
    ).abs().max()

    state_diff = (
        final_state.float()
        - state_full.float()
    ).abs().max()

else:

    out_equal = True
    state_equal = True

    out_diff = torch.tensor(
        0.0,
        device=device,
    )

    state_diff = torch.tensor(
        0.0,
        device=device,
    )

flags = torch.tensor(
    [
        int(out_equal),
        int(state_equal),
    ],
    device=device,
    dtype=torch.int32,
)

dist.all_reduce(
    flags,
    op=dist.ReduceOp.MIN,
)

dist.all_reduce(
    out_diff,
    op=dist.ReduceOp.MAX,
)

dist.all_reduce(
    state_diff,
    op=dist.ReduceOp.MAX,
)

# ============================================================
# Print
# ============================================================
if rank == 0:

    median_total = statistics.median(times)
    mean_total = statistics.mean(times)

    amortized_median = (
        median_total / TIMED_REQS
    )

    amortized_mean = (
        mean_total / TIMED_REQS
    )

    req_per_s = (
        1000.0 / amortized_median
    )

    print()
    print("===== FlashKDA SP2 Throughput Pipeline =====")
    print("pipeline depth        :", PIPE_DEPTH)
    print("timed requests        :", TIMED_REQS)
    print(
        "state payload         :",
        f"{STATE_MIB:.3f} MiB"
    )

    print()
    print("total times ms        :", [
        round(x,4)
        for x in times
    ])

    print(
        "median total          :",
        f"{median_total:.4f} ms"
    )

    print(
        "mean total            :",
        f"{mean_total:.4f} ms"
    )

    print(
        "amortized / request   :",
        f"{amortized_median:.4f} ms"
    )

    print(
        "throughput            :",
        f"{req_per_s:.2f} req/s"
    )

    print()
    print(
        "last output bitwise   :",
        bool(flags[0].item())
    )

    print(
        "last state bitwise    :",
        bool(flags[1].item())
    )

    print(
        "last output max diff  :",
        out_diff.item()
    )

    print(
        "last state max diff   :",
        state_diff.item()
    )

dist.destroy_process_group()
