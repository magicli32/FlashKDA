import os
import argparse
import statistics
import torch

from flash_kda import fwd_cp


parser = argparse.ArgumentParser()
parser.add_argument("--direct-initial", type=int, choices=[0, 1], required=True)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()


# ------------------------------------------------------------
# Exact target: Mixed Varlen H96 / P2
# ------------------------------------------------------------
seq_lens = [1300, 547, 2048, 963, 271, 3063]

B = 1
T = sum(seq_lens)
H = 96
D = 128
N = len(seq_lens)

assert T == 8192
assert N == 6


# ------------------------------------------------------------
# Current M10/M11 winner switches
# ------------------------------------------------------------
os.environ["FLASHKDA_FORCE_CP_PIECES"] = "2"

os.environ["FLASHKDA_CP_STATIC_WARMUP"] = "1"
os.environ["FLASHKDA_CP_MIXED_DIRECT_BF16"] = "1"
os.environ["FLASHKDA_CP_BF16_RESIDENT"] = "1"
os.environ["FLASHKDA_CP_DIRECT_FINAL"] = "1"
os.environ["FLASHKDA_CP_COMPACT_WARMUP_K1"] = "1"

os.environ["FLASHKDA_CP_DIRECT_INITIAL"] = str(args.direct_initial)


# ------------------------------------------------------------
# Same deterministic inputs for OFF / ON
# ------------------------------------------------------------
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

device = "cuda"

cu = [0]
for x in seq_lens:
    cu.append(cu[-1] + x)

cu_seqlens = torch.tensor(
    cu,
    dtype=torch.int64,
    device=device,
)

q = torch.randn(
    B, T, H, D,
    dtype=torch.bfloat16,
    device=device,
)

k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn_like(q)

beta = torch.randn(
    B, T, H,
    dtype=torch.bfloat16,
    device=device,
)

A_log = (
    -torch.ones(
        H,
        dtype=torch.float32,
        device=device,
    ) * 0.1
)

dt_bias = torch.randn(
    H, D,
    dtype=torch.float32,
    device=device,
)

scale = D ** -0.5
lower_bound = -5.0

initial_state = (
    torch.randn(
        N, H, D, D,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.01
)

out = torch.empty_like(q)

final_state = torch.empty(
    N, H, D, D,
    dtype=torch.bfloat16,
    device=device,
)


def run_once():
    fwd_cp(
        q, k, v, g, beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        auto_cp=True,
    )


print("=" * 72)
print(
    f"M11 DIRECT_INITIAL={args.direct_initial} "
    f"shape=[{T},{H},{D}] "
    f"seq_lens={seq_lens}"
)
print(
    f"warmup={args.warmup} "
    f"iters={args.iters} "
    f"repeats={args.repeats}"
)
print("=" * 72)


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------
for _ in range(args.warmup):
    run_once()

torch.cuda.synchronize()


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
repeat_means = []
all_times = []

for r in range(args.repeats):
    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters)
    ]

    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters)
    ]

    for i in range(args.iters):
        starts[i].record()
        run_once()
        ends[i].record()

    torch.cuda.synchronize()

    times = [
        starts[i].elapsed_time(ends[i])
        for i in range(args.iters)
    ]

    all_times.extend(times)

    mean_ms = statistics.mean(times)
    min_ms = min(times)
    max_ms = max(times)

    repeat_means.append(mean_ms)

    print(
        f"repeat {r + 1}: "
        f"mean={mean_ms:.4f} ms, "
        f"min={min_ms:.4f} ms, "
        f"max={max_ms:.4f} ms"
    )


print()
print("===== SUMMARY =====")
print(
    f"DIRECT_INITIAL={args.direct_initial}: "
    f"mean={statistics.mean(repeat_means):.4f} ms, "
    f"min={min(all_times):.4f} ms, "
    f"max={max(all_times):.4f} ms"
)
