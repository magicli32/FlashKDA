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
# Configuration
# --------------------------------------------------
B = 1
T = 8192
H_GLOBAL = 96
D = 128
OUT_DIM = 1024

WARMUP = 5
ITERS = 30

assert H_GLOBAL % world == 0

H = H_GLOBAL // world
K_LOCAL = H * D

scale = D ** -0.5
lower_bound = -5.0

torch.manual_seed(1234 + rank)

# --------------------------------------------------
# FlashKDA inputs
# --------------------------------------------------
q = torch.randn(
    B,T,H,D,
    device=device,
    dtype=torch.bfloat16,
)

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
    torch.mm(
        kda_out.view(T,K_LOCAL),
        W,
        out=y,
    )


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def max_rank(x):
    t = torch.tensor(
        [x],
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
tmp = torch.empty(
    T,OUT_DIM,
    device=device,
    dtype=torch.bfloat16,
)

for _ in range(WARMUP):
    run_kda()
    project(tmp)

    w = dist.all_reduce(
        tmp,
        async_op=True,
    )
    w.wait()

sync_all()


# --------------------------------------------------
# Reference result
# --------------------------------------------------
run_kda()
project(tmp)

w = dist.all_reduce(
    tmp,
    async_op=True,
)
w.wait()

torch.cuda.synchronize()

reference = tmp.clone()


# --------------------------------------------------
# Unbounded benchmark
# Original implementation: one buffer / iteration
# --------------------------------------------------
def bench_unbounded():

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

        works.append(
            dist.all_reduce(
                ys[i],
                async_op=True,
            )
        )

    for w in works:
        w.wait()

    torch.cuda.synchronize()

    elapsed = (
        (time.perf_counter() - t0)
        * 1000 / ITERS
    )

    elapsed = max_rank(elapsed)

    diff = (
        ys[-1].float()
        - reference.float()
    ).abs().max()

    dist.all_reduce(
        diff,
        op=dist.ReduceOp.MAX,
    )

    return elapsed, diff.item()


# --------------------------------------------------
# Ring-buffer benchmark
#
# Important:
# We run NEXT KDA first.
#
# Only before projection overwrites a reused buffer
# do we wait for its previous AllReduce.
#
# Therefore previous communication overlaps next KDA.
# --------------------------------------------------
def bench_ring(depth):

    ys = [
        torch.empty(
            T,OUT_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(depth)
    ]

    works = [None] * depth

    sync_all()

    t0 = time.perf_counter()

    last_slot = None

    for i in range(ITERS):

        # ------------------------------
        # This computation overlaps the
        # previous collective.
        # ------------------------------
        run_kda()

        slot = i % depth

        # Before overwriting this slot,
        # make sure old collective that
        # owns it has completed.
        if works[slot] is not None:
            works[slot].wait()

        project(ys[slot])

        works[slot] = dist.all_reduce(
            ys[slot],
            async_op=True,
        )

        last_slot = slot

    # Drain remaining communication.
    for w in works:
        if w is not None:
            w.wait()

    torch.cuda.synchronize()

    elapsed = (
        (time.perf_counter() - t0)
        * 1000 / ITERS
    )

    elapsed = max_rank(elapsed)

    diff = (
        ys[last_slot].float()
        - reference.float()
    ).abs().max()

    dist.all_reduce(
        diff,
        op=dist.ReduceOp.MAX,
    )

    return elapsed, diff.item()


# --------------------------------------------------
# Run
# --------------------------------------------------
unbounded_ms, unbounded_diff = bench_unbounded()

results = []

for depth in [1,2,4]:

    ms, diff = bench_ring(depth)

    results.append(
        (depth,ms,diff)
    )


if rank == 0:

    payload_mib = (
        T * OUT_DIM * 2
        / 1024 / 1024
    )

    print()
    print("===== FlashKDA TP Ring Buffer =====")
    print("world             :", world)
    print("global H          :", H_GLOBAL)
    print("local H           :", H)
    print("payload / buffer  :", f"{payload_mib:.1f} MiB")
    print()

    print(
        f"unbounded ({ITERS:2d} buffers): "
        f"{unbounded_ms:.4f} ms  "
        f"memory={ITERS*payload_mib:.0f} MiB  "
        f"diff={unbounded_diff}"
    )

    for depth,ms,diff in results:

        slowdown = (
            ms / unbounded_ms - 1
        ) * 100

        print(
            f"ring depth={depth}: "
            f"{ms:.4f} ms  "
            f"memory={depth*payload_mib:.0f} MiB  "
            f"vs_unbounded={slowdown:+.2f}%  "
            f"diff={diff}"
        )

dist.destroy_process_group()
