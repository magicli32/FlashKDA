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

WARMUP = 5
ITERS = 20

scale = D ** -0.5
lower_bound = -5.0

STATE_MIB = (
    B * H * D * D * 2
    / 1024 / 1024
)

# ============================================================
# Identical global input on both ranks
# ============================================================
gen = torch.Generator(device=device)
gen.manual_seed(20260819)

def randn(shape, dtype=torch.bfloat16):
    return torch.randn(
        shape,
        device=device,
        dtype=dtype,
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
# Helper
# ============================================================
def sync_all():
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()


def run_full(out, final_state):
    flash_kda.fwd(
        q,k,v,g,beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=None,
        final_state=final_state,
    )


def run_chunk0(out, final_state):
    flash_kda.fwd(
        q0,k0,v0,g0,b0,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=None,
        final_state=final_state,
    )


def run_chunk1(out, initial_state, final_state):
    flash_kda.fwd(
        q1,k1,v1,g1,b1,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
    )


# ============================================================
# 1. Full reference
# ============================================================
out_full = torch.empty_like(v)

state_full = torch.empty(
    B,H,D,D,
    device=device,
    dtype=torch.bfloat16,
)

run_full(out_full, state_full)
torch.cuda.synchronize()

# ============================================================
# 2. Local chunk0 reference on both GPUs
#    Used to verify state transfer itself.
# ============================================================
out0_ref = torch.empty_like(v0)

state0_ref = torch.empty(
    B,H,D,D,
    device=device,
    dtype=torch.bfloat16,
)

run_chunk0(out0_ref, state0_ref)
torch.cuda.synchronize()

# ============================================================
# 3. Real GPU0 -> GPU1 state transfer
# ============================================================
state_recv = torch.empty_like(state0_ref)

sync_all()

if rank == 0:
    work = dist.isend(
        state0_ref,
        dst=1,
    )
    work.wait()
else:
    work = dist.irecv(
        state_recv,
        src=0,
    )
    work.wait()

torch.cuda.synchronize()
dist.barrier()

# Verify transport bit-for-bit.
if rank == 1:
    transfer_equal_local = torch.equal(
        state_recv,
        state0_ref,
    )

    transfer_diff_local = (
        state_recv.float()
        - state0_ref.float()
    ).abs().max()
else:
    transfer_equal_local = True
    transfer_diff_local = torch.tensor(
        0.0,
        device=device,
    )

transfer_equal = torch.tensor(
    [int(transfer_equal_local)],
    device=device,
    dtype=torch.int32,
)

dist.all_reduce(
    transfer_equal,
    op=dist.ReduceOp.MIN,
)

dist.all_reduce(
    transfer_diff_local,
    op=dist.ReduceOp.MAX,
)

# ============================================================
# 4. GPU1 executes second half using received state
# ============================================================
out1_sp = torch.empty_like(v1)

state1_sp = torch.empty(
    B,H,D,D,
    device=device,
    dtype=torch.bfloat16,
)

if rank == 1:
    run_chunk1(
        out1_sp,
        state_recv,
        state1_sp,
    )
    torch.cuda.synchronize()

dist.barrier()

# ============================================================
# 5. Cross-GPU numerical correctness
# ============================================================

# GPU0 checks first half.
if rank == 0:
    out0_equal_local = torch.equal(
        out0_ref,
        out_full[:,:SPLIT],
    )

    out0_diff = (
        out0_ref.float()
        - out_full[:,:SPLIT].float()
    ).abs().max()
else:
    out0_equal_local = True
    out0_diff = torch.tensor(
        0.0,
        device=device,
    )

# GPU1 checks second half + final state.
if rank == 1:
    out1_equal_local = torch.equal(
        out1_sp,
        out_full[:,SPLIT:],
    )

    out1_diff = (
        out1_sp.float()
        - out_full[:,SPLIT:].float()
    ).abs().max()

    state_equal_local = torch.equal(
        state1_sp,
        state_full,
    )

    state_diff = (
        state1_sp.float()
        - state_full.float()
    ).abs().max()
else:
    out1_equal_local = True
    state_equal_local = True

    out1_diff = torch.tensor(
        0.0,
        device=device,
    )

    state_diff = torch.tensor(
        0.0,
        device=device,
    )

flags = torch.tensor(
    [
        int(out0_equal_local),
        int(out1_equal_local),
        int(state_equal_local),
    ],
    device=device,
    dtype=torch.int32,
)

dist.all_reduce(
    flags,
    op=dist.ReduceOp.MIN,
)

for x in [
    out0_diff,
    out1_diff,
    state_diff,
]:
    dist.all_reduce(
        x,
        op=dist.ReduceOp.MAX,
    )

# ============================================================
# 6. Full T8192 single-GPU-equivalent baseline
# ============================================================
tmp_full = torch.empty_like(v)
tmp_full_state = torch.empty_like(state_full)

for _ in range(WARMUP):
    run_full(
        tmp_full,
        tmp_full_state,
    )

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(ITERS):
    run_full(
        tmp_full,
        tmp_full_state,
    )

end.record()
torch.cuda.synchronize()

full_ms = start.elapsed_time(end) / ITERS

full_tensor = torch.tensor(
    [full_ms],
    device=device,
    dtype=torch.float64,
)

dist.all_reduce(
    full_tensor,
    op=dist.ReduceOp.MAX,
)

full_ms = full_tensor.item()

# ============================================================
# 7. State-transfer-only latency
# ============================================================
transfer_times = []

for _ in range(WARMUP):
    sync_all()

    if rank == 0:
        w = dist.isend(
            state0_ref,
            dst=1,
        )
    else:
        w = dist.irecv(
            state_recv,
            src=0,
        )

    w.wait()
    torch.cuda.synchronize()

sync_all()

for _ in range(ITERS):

    sync_all()

    t0 = time.perf_counter()

    if rank == 0:
        w = dist.isend(
            state0_ref,
            dst=1,
        )
    else:
        w = dist.irecv(
            state_recv,
            src=0,
        )

    w.wait()
    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - t0
    ) * 1000

    if rank == 1:
        transfer_times.append(elapsed)

    dist.barrier()

# ============================================================
# 8. Single-request SP chain
#
# GPU0:
#   chunk0 -> send state
#
# GPU1:
#   recv state -> chunk1
#
# No cross-request overlap here.
# ============================================================
chain_times = []

out0_tmp = torch.empty_like(v0)
out1_tmp = torch.empty_like(v1)

state_send = torch.empty_like(state0_ref)
state_recv2 = torch.empty_like(state0_ref)
state_end = torch.empty_like(state0_ref)

# Warmup
for _ in range(WARMUP):

    sync_all()

    if rank == 0:

        run_chunk0(
            out0_tmp,
            state_send,
        )

        w = dist.isend(
            state_send,
            dst=1,
        )
        w.wait()

    else:

        w = dist.irecv(
            state_recv2,
            src=0,
        )
        w.wait()

        run_chunk1(
            out1_tmp,
            state_recv2,
            state_end,
        )

    torch.cuda.synchronize()
    dist.barrier()

# Timed
for _ in range(ITERS):

    sync_all()

    t0 = time.perf_counter()

    if rank == 0:

        run_chunk0(
            out0_tmp,
            state_send,
        )

        w = dist.isend(
            state_send,
            dst=1,
        )
        w.wait()

    else:

        w = dist.irecv(
            state_recv2,
            src=0,
        )
        w.wait()

        run_chunk1(
            out1_tmp,
            state_recv2,
            state_end,
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - t0
    ) * 1000

    # Rank1 observes the entire dependency chain:
    # wait for GPU0 compute -> recv -> GPU1 compute.
    if rank == 1:
        chain_times.append(elapsed)

    dist.barrier()

# ============================================================
# Results
# ============================================================
if rank == 1:

    transfer_mean = statistics.mean(
        transfer_times
    )

    transfer_median = statistics.median(
        transfer_times
    )

    chain_mean = statistics.mean(
        chain_times
    )

    chain_median = statistics.median(
        chain_times
    )

    gib_per_s = (
        STATE_MIB / 1024
    ) / (
        transfer_median / 1000
    )

    speedup = (
        full_ms / chain_median
    )

    print()
    print("===== FlashKDA SP2 Feasibility =====")
    print("global T              :", T)
    print("split                 :", f"{SPLIT} + {T-SPLIT}")
    print("H                     :", H)
    print(
        "state payload         :",
        f"{STATE_MIB:.3f} MiB"
    )

    print()
    print("----- Correctness -----")
    print(
        "state transfer bitwise:",
        bool(transfer_equal.item())
    )
    print(
        "state transfer maxdiff:",
        transfer_diff_local.item()
    )
    print(
        "chunk0 output bitwise :",
        bool(flags[0].item())
    )
    print(
        "chunk1 output bitwise :",
        bool(flags[1].item())
    )
    print(
        "final state bitwise   :",
        bool(flags[2].item())
    )
    print(
        "chunk0 max diff       :",
        out0_diff.item()
    )
    print(
        "chunk1 max diff       :",
        out1_diff.item()
    )
    print(
        "final state max diff  :",
        state_diff.item()
    )

    print()
    print("----- Performance -----")
    print(
        "single GPU full T8192 :",
        f"{full_ms:.4f} ms"
    )
    print(
        "state transfer median :",
        f"{transfer_median:.4f} ms"
    )
    print(
        "state transfer mean   :",
        f"{transfer_mean:.4f} ms"
    )
    print(
        "effective state BW    :",
        f"{gib_per_s:.2f} GiB/s"
    )
    print(
        "SP2 single-req median :",
        f"{chain_median:.4f} ms"
    )
    print(
        "SP2 single-req mean   :",
        f"{chain_mean:.4f} ms"
    )
    print(
        "SP2 / single GPU      :",
        f"{speedup:.3f}x"
    )

dist.destroy_process_group()
