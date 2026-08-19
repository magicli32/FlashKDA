import os
import time
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

# --------------------------------------------------
# TP configuration
# --------------------------------------------------
B = 1
T = 8192
H_GLOBAL = 96
D = 128

assert H_GLOBAL % world == 0
H = H_GLOBAL // world       # 24 on TP4

K_LOCAL = H * D             # 3072
OUT_DIM = 1024              # AR tensor = 8192*1024*2 = 16 MiB

WARMUP = 5
ITERS = 30

torch.manual_seed(1234 + rank)

# --------------------------------------------------
# FlashKDA inputs
# --------------------------------------------------
q = torch.randn(B,T,H,D,device=device,dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn_like(q)

beta = torch.randn(
    B,T,H,
    device=device,
    dtype=torch.bfloat16,
)

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

kda_out = torch.empty_like(v)

scale = D ** -0.5
lower_bound = -5.0

# Row-parallel output projection:
# each rank owns [H_local * D, OUT_DIM]
W = torch.randn(
    K_LOCAL,
    OUT_DIM,
    device=device,
    dtype=torch.bfloat16,
) * 0.01


def run_kda():
    flash_kda.fwd(
        q,k,v,g,beta,
        scale,
        kda_out,
        A_log,
        dt_bias,
        lower_bound,
    )


def project(y):
    x = kda_out.view(T, K_LOCAL)

    torch.mm(
        x,
        W,
        out=y,
    )


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def max_rank(value):
    t = torch.tensor(
        [value],
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        t,
        op=dist.ReduceOp.MAX,
    )

    return t.item()


# --------------------------------------------------
# Warmup
# --------------------------------------------------
warm_y = torch.empty(
    T, OUT_DIM,
    device=device,
    dtype=torch.bfloat16,
)

for _ in range(WARMUP):
    run_kda()
    project(warm_y)

    w = dist.all_reduce(
        warm_y,
        async_op=True,
    )
    w.wait()

sync_all()


# --------------------------------------------------
# KDA only
# --------------------------------------------------
sync_all()
t0 = time.perf_counter()

for _ in range(ITERS):
    run_kda()

torch.cuda.synchronize()

kda_ms = (
    (time.perf_counter() - t0)
    * 1000 / ITERS
)

kda_ms = max_rank(kda_ms)


# --------------------------------------------------
# Projection only
# --------------------------------------------------
run_kda()
torch.cuda.synchronize()

sync_all()
t0 = time.perf_counter()

for _ in range(ITERS):
    project(warm_y)

torch.cuda.synchronize()

proj_ms = (
    (time.perf_counter() - t0)
    * 1000 / ITERS
)

proj_ms = max_rank(proj_ms)


# --------------------------------------------------
# AllReduce only
# --------------------------------------------------
comm_y = torch.zeros(
    T,OUT_DIM,
    device=device,
    dtype=torch.bfloat16,
)

sync_all()
t0 = time.perf_counter()

for _ in range(ITERS):
    w = dist.all_reduce(
        comm_y,
        async_op=True,
    )
    w.wait()

torch.cuda.synchronize()

comm_ms = (
    (time.perf_counter() - t0)
    * 1000 / ITERS
)

comm_ms = max_rank(comm_ms)


# --------------------------------------------------
# SERIAL
#
# KDA -> projection -> AllReduce
# wait before next KDA
# --------------------------------------------------
serial_y = torch.empty(
    T,OUT_DIM,
    device=device,
    dtype=torch.bfloat16,
)

sync_all()
t0 = time.perf_counter()

for _ in range(ITERS):

    run_kda()

    project(serial_y)

    w = dist.all_reduce(
        serial_y,
        async_op=True,
    )

    # forces next compute to wait for this collective
    w.wait()

torch.cuda.synchronize()

serial_ms = (
    (time.perf_counter() - t0)
    * 1000 / ITERS
)

serial_ms = max_rank(serial_ms)

serial_ref = serial_y.clone()


# --------------------------------------------------
# PIPELINED OVERLAP
#
# Each communication output has its own buffer.
# Do not wait immediately.
# PGNCCL communication can overlap next KDA.
# --------------------------------------------------
ys = [
    torch.empty(
        T,OUT_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    for _ in range(ITERS)
]

works = []

sync_all()
t0 = time.perf_counter()

for i in range(ITERS):

    run_kda()

    project(ys[i])

    w = dist.all_reduce(
        ys[i],
        async_op=True,
    )

    works.append(w)

for w in works:
    w.wait()

torch.cuda.synchronize()

overlap_ms = (
    (time.perf_counter() - t0)
    * 1000 / ITERS
)

overlap_ms = max_rank(overlap_ms)


# --------------------------------------------------
# Correctness
# --------------------------------------------------
diff = (
    ys[-1].float()
    - serial_ref.float()
).abs()

max_diff = diff.max()

dist.all_reduce(
    max_diff,
    op=dist.ReduceOp.MAX,
)

same = torch.tensor(
    [
        int(torch.equal(
            ys[-1],
            serial_ref,
        ))
    ],
    device=device,
    dtype=torch.int32,
)

dist.all_reduce(
    same,
    op=dist.ReduceOp.MIN,
)


# --------------------------------------------------
# Results
# --------------------------------------------------
if rank == 0:

    saved = serial_ms - overlap_ms
    reduction = saved / serial_ms * 100
    speedup = serial_ms / overlap_ms

    theoretical_sum = (
        kda_ms
        + proj_ms
        + comm_ms
    )

    print()
    print("===== FlashKDA TP4 Compute/Comm Overlap =====")
    print(f"world            : {world}")
    print(f"global H         : {H_GLOBAL}")
    print(f"local H          : {H}")
    print(f"K_local          : {K_LOCAL}")
    print(f"projection       : {K_LOCAL} -> {OUT_DIM}")
    print(f"AR payload       : 16 MiB BF16")
    print()
    print(f"KDA only         : {kda_ms:.4f} ms")
    print(f"projection only  : {proj_ms:.4f} ms")
    print(f"AllReduce only   : {comm_ms:.4f} ms")
    print(f"component sum    : {theoretical_sum:.4f} ms")
    print()
    print(f"serial           : {serial_ms:.4f} ms")
    print(f"overlap          : {overlap_ms:.4f} ms")
    print(f"saved            : {saved:.4f} ms")
    print(f"reduction        : {reduction:.2f}%")
    print(f"speedup          : {speedup:.3f}x")
    print()
    print(f"bitwise equal    : {bool(same.item())}")
    print(f"max abs diff     : {max_diff.item()}")
    print("RESULT           :",
          "PASS" if max_diff.item() == 0 else "CHECK")

dist.destroy_process_group()
