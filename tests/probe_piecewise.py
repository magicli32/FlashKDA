import math
import os

import torch
import torch.nn.functional as F

import flash_kda


B = 1
T = 8192
H = int(os.environ.get("PROBE_H", "1"))
D = 128
P = 3

LOWER_BOUND = -5.0
SCALE = 1.0 / math.sqrt(D)

# 三段长度分别为 2736、2736、2720，
# 所有边界均与 FlashKDA 的 CHUNK=16 对齐。
BOUNDARIES = [0, 2736, 5472, 8192]


def error_stats(name, actual, reference):
    exact_ratio = (
        (actual == reference)
        .to(torch.float32)
        .mean()
        .item()
    )

    actual = actual.float()
    reference = reference.float()
    diff = actual - reference

    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    rmse = diff.square().mean().sqrt()
    ref_rms = reference.square().mean().sqrt().clamp_min(1e-12)
    rel_rmse = (rmse / ref_rms).item()

    finite = torch.isfinite(actual).all().item()

    print(
        f"{name:32s} "
        f"max_abs={max_abs:.6e} "
        f"mean_abs={mean_abs:.6e} "
        f"rel_rmse={rel_rmse:.6e} "
        f"exact_ratio={exact_ratio:.6f} "
        f"finite={finite}"
    )


@torch.inference_mode()
def run_flash(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    initial_state,
    cu_seqlens=None,
):
    out = torch.empty_like(q)
    state_in = initial_state.contiguous().clone()
    final_state = torch.empty_like(state_in)

    kwargs = {}
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens

    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        SCALE,
        out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=state_in,
        final_state=final_state,
        **kwargs,
    )

    return out, final_state


def build_piece_starts(h0, piece_R, piece_C, round_between):
    """
    使用：
        S_out = S_in @ R + C

    round_between=False:
        前缀状态一直保留FP32，传给K2时才转BF16。

    round_between=True:
        每经过一个piece边界都舍入为BF16。
    """
    state = h0[0].float()
    starts = []

    for p in range(P):
        starts.append(state.to(torch.bfloat16))

        if p + 1 < P:
            state = (
                torch.matmul(state, piece_R[p].float())
                + piece_C[p].float()
            )

            if round_between:
                state = state.to(torch.bfloat16).float()

    return torch.stack(starts).contiguous()


@torch.inference_mode()
def main():
    torch.manual_seed(0)

    # 防止FP32矩阵乘法自动使用TF32，先测更精确的FP32前缀。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = "cuda"
    dtype = torch.bfloat16

    print(f"B={B}, T={T}, H={H}, D={D}, P={P}")
    print(f"GPU={torch.cuda.get_device_name(0)}")
    print(f"boundaries={BOUNDARIES}")

    q = F.normalize(
        torch.randn(B, T, H, D, device=device, dtype=torch.float32),
        p=2,
        dim=-1,
    ).to(dtype)

    k = F.normalize(
        torch.randn(B, T, H, D, device=device, dtype=torch.float32),
        p=2,
        dim=-1,
    ).to(dtype)

    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.randn(B, T, H, device=device, dtype=dtype)

    A_log = torch.rand(H, device=device, dtype=torch.float32)
    dt_bias = torch.rand(H, D, device=device, dtype=torch.float32)
    h0 = (
        0.1 * torch.randn(
            B, H, D, D,
            device=device,
            dtype=torch.float32,
         )
     ).to(dtype)

    # ----------------------------------------------------------
    # 1. 原始完整序列 P=1
    # ----------------------------------------------------------
    out_base, state_base = run_flash(
        q, k, v, g, beta, A_log, dt_bias, h0
    )

    # ----------------------------------------------------------
    # 2. 把同一条序列声明为三个独立piece
    # ----------------------------------------------------------
    cu_seqlens = torch.tensor(
        BOUNDARIES,
        device=device,
        dtype=torch.int64,
    )

    zero_states = torch.zeros(
        P, H, D, D, device=device, dtype=dtype
    )

    identity_states = (
        torch.eye(D, device=device, dtype=dtype)
        .reshape(1, 1, D, D)
        .expand(P, H, D, D)
        .contiguous()
    )

    # C[p] = F_piece(0, v)
    unused_out, piece_C = run_flash(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        zero_states,
        cu_seqlens,
    )
    del unused_out

    # R[p] = F_piece(I, 0)
    zero_v = torch.zeros_like(v)

    unused_out, piece_R = run_flash(
        q,
        k,
        zero_v,
        g,
        beta,
        A_log,
        dt_bias,
        identity_states,
        cu_seqlens,
    )
    del unused_out, zero_v

    # ----------------------------------------------------------
    # 3. 用随机状态检查 S@R+C 是否能重建piece最终状态
    # ----------------------------------------------------------
    probe_state = (
        0.1
        * torch.randn(
            P, H, D, D,
            device=device,
            dtype=torch.float32,
        )
    ).to(dtype)

    unused_out, probe_actual = run_flash(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        probe_state,
        cu_seqlens,
    )
    del unused_out

    probe_predicted = (
        torch.matmul(probe_state.float(), piece_R.float())
        + piece_C.float()
    ).to(dtype)

    print("\n[1] Piece map reconstruction")
    error_stats(
        "predicted S@R+C vs actual",
        probe_predicted,
        probe_actual,
    )

    # ----------------------------------------------------------
    # 4. 两种前缀精度策略
    # ----------------------------------------------------------
    starts_keep_fp32 = build_piece_starts(
        h0, piece_R, piece_C, round_between=False
    )

    starts_round_bf16 = build_piece_starts(
        h0, piece_R, piece_C, round_between=True
    )

    out_keep, state_keep = run_flash(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        starts_keep_fp32,
        cu_seqlens,
    )

    out_round, state_round = run_flash(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        starts_round_bf16,
        cu_seqlens,
    )

    torch.cuda.synchronize()

    print("\n[2] Overall P=1 versus P=3")
    error_stats("output keep-FP32-prefix", out_keep, out_base)
    error_stats(
        "state keep-FP32-prefix",
        state_keep[-1],
        state_base[0],
    )

    error_stats("output round-BF16-prefix", out_round, out_base)
    error_stats(
        "state round-BF16-prefix",
        state_round[-1],
        state_base[0],
    )

    print("\n[3] Error inside each piece: keep-FP32-prefix")
    for p, (begin, end) in enumerate(
        zip(BOUNDARIES[:-1], BOUNDARIES[1:])
    ):
        error_stats(
            f"piece {p} output[{begin}:{end}]",
            out_keep[:, begin:end],
            out_base[:, begin:end],
        )

    print("\n[4] Error inside each piece: round-BF16-prefix")
    for p, (begin, end) in enumerate(
        zip(BOUNDARIES[:-1], BOUNDARIES[1:])
    ):
        error_stats(
            f"piece {p} output[{begin}:{end}]",
            out_round[:, begin:end],
            out_base[:, begin:end],
        )


if __name__ == "__main__":
    main()
