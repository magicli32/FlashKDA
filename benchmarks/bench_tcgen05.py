"""Correctness and full-forward A/B benchmark for TCGen05 K2 Phase 1.

The optimized path uses 16-column-wide TMEM loads for each 128x16 Phase-1
accumulator and a warp-local handoff to the legacy register path.  Keeping
this label in the output makes the experiment stages distinguishable without
changing the public dispatch interface.
"""

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


def make_runner(inputs, tcgen05_k2):
    q, k, v, g, beta, A_log, dt_bias, initial_state = inputs
    out = torch.empty_like(q)
    final_state = torch.empty_like(initial_state)

    def run():
        flash_kda.fwd(
            q, k, v, g, beta, 1.0 / math.sqrt(q.shape[-1]), out,
            A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
            initial_state=initial_state, final_state=final_state,
            v_split=1, tcgen05_k2=tcgen05_k2,
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
    baseline, out_base, state_base = make_runner(inputs, False)
    tcgen, out_tc, state_tc = make_runner(inputs, True)

    baseline()
    tcgen()
    torch.cuda.synchronize()
    if not torch.equal(out_base, out_tc):
        diff = (out_base.float() - out_tc.float()).abs()
        raise AssertionError(
            f"output mismatch: count={torch.count_nonzero(diff).item()} "
            f"max_abs={diff.max().item()}"
        )
    if not torch.equal(state_base, state_tc):
        diff = (state_base.float() - state_tc.float()).abs()
        raise AssertionError(
            f"final-state mismatch: count={torch.count_nonzero(diff).item()} "
            f"max_abs={diff.max().item()}"
        )
    print("correctness: bitwise equal")
    print("tcgen05 variant: phase1_wide_load_warp_local_handoff")

    baseline_ms = []
    tcgen_ms = []
    for repeat in range(args.repeats):
        order = ((baseline, baseline_ms), (tcgen, tcgen_ms))
        if repeat % 2:
            order = tuple(reversed(order))
        for run, results in order:
            results.append(measure(run, args.warmup, args.iters))

    base_median = statistics.median(baseline_ms)
    tcgen_median = statistics.median(tcgen_ms)
    print(f"baseline repeats_ms: {baseline_ms}")
    print(f"tcgen05 repeats_ms: {tcgen_ms}")
    print(f"baseline median_ms: {base_median:.6f}")
    print(f"tcgen05 median_ms: {tcgen_median:.6f}")
    print(f"speedup: {base_median / tcgen_median:.4f}x")
    print(
        "latency improvement: "
        f"{(base_median - tcgen_median) / base_median * 100.0:.2f}%"
    )


if __name__ == "__main__":
    main()
