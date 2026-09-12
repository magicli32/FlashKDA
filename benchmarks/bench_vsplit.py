"""Correctness and full-forward A/B benchmark for K2 V parallelism."""

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

import flash_kda


def make_inputs(t, h, d, seed):
    torch.manual_seed(seed)
    shape = (1, t, h, d)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, t, h), dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(h, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(h, d, dtype=torch.float32, device="cuda")
    initial_state = torch.randn((1, h, d, d), dtype=torch.bfloat16, device="cuda")
    return q, k, v, g, beta, A_log, dt_bias, initial_state


def make_runner(inputs, v_split):
    q, k, v, g, beta, A_log, dt_bias, initial_state = inputs
    out = torch.empty_like(q)
    final_state = torch.empty_like(initial_state)

    def run():
        flash_kda.fwd(
            q, k, v, g, beta, 1.0 / math.sqrt(q.shape[-1]), out,
            A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
            initial_state=initial_state, final_state=final_state,
            v_split=v_split,
        )

    return run, out, final_state


def measure(run, warmup, iters):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=96)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.D != 128:
        raise ValueError("FlashKDA currently requires D=128")

    inputs = make_inputs(args.T, args.H, args.D, args.seed)
    baseline, out1, state1 = make_runner(inputs, 1)
    split2, out2, state2 = make_runner(inputs, 2)

    baseline()
    split2()
    torch.cuda.synchronize()
    if not torch.equal(out1, out2):
        diff = (out1.float() - out2.float()).abs()
        raise AssertionError(
            f"output mismatch: count={torch.count_nonzero(diff).item()} "
            f"max_abs={diff.max().item()}"
        )
    if not torch.equal(state1, state2):
        diff = (state1.float() - state2.float()).abs()
        raise AssertionError(
            f"final-state mismatch: count={torch.count_nonzero(diff).item()} "
            f"max_abs={diff.max().item()}"
        )
    print("correctness: bitwise equal")

    baseline_ms = []
    split2_ms = []
    # Alternate order so clock/cache drift does not always favor one variant.
    for repeat in range(args.repeats):
        order = ((baseline, baseline_ms), (split2, split2_ms))
        if repeat % 2:
            order = tuple(reversed(order))
        for run, results in order:
            results.append(measure(run, args.warmup, args.iters))

    base_median = statistics.median(baseline_ms)
    split_median = statistics.median(split2_ms)
    print(f"baseline repeats_ms: {baseline_ms}")
    print(f"v_split=2 repeats_ms: {split2_ms}")
    print(f"baseline median_ms: {base_median:.6f}")
    print(f"v_split=2 median_ms: {split_median:.6f}")
    print(f"speedup: {base_median / split_median:.4f}x")
    print(f"latency improvement: {(base_median - split_median) / base_median * 100.0:.2f}%")


if __name__ == "__main__":
    main()
