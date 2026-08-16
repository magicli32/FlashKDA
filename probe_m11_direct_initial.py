import os
import torch

from flash_kda import fwd_cp
from flash_kda.cp import _calc_cp_seqs

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

device = "cuda"

# Exact Mixed H96 target
seq_lens = [1300, 547, 2048, 963, 271, 3063]
B = 1
T = sum(seq_lens)
H = 96
D = 128
N = len(seq_lens)

assert T == 8192
assert N == 6

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
    device=device,
    dtype=torch.bfloat16,
)

k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn_like(q)

beta = torch.randn(
    B, T, H,
    device=device,
    dtype=torch.bfloat16,
)

A_log = (
    -torch.ones(
        H,
        device=device,
        dtype=torch.float32,
    ) * 0.1
)

dt_bias = torch.randn(
    H, D,
    device=device,
    dtype=torch.float32,
)

scale = D ** -0.5
lower_bound = -5.0

# M11 requires raw BF16 initial state.
initial_state = (
    torch.randn(
        N, H, D, D,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.01
)

# Force the exact P2 split:
# raw 6 sequences -> CP 8 segments.
os.environ["FLASHKDA_FORCE_CP_PIECES"] = "2"

# Current M10/M11 winner switches.
os.environ["FLASHKDA_CP_STATIC_WARMUP"] = "1"
os.environ["FLASHKDA_CP_MIXED_DIRECT_BF16"] = "1"
os.environ["FLASHKDA_CP_BF16_RESIDENT"] = "1"
os.environ["FLASHKDA_CP_DIRECT_FINAL"] = "1"
os.environ["FLASHKDA_CP_COMPACT_WARMUP_K1"] = "1"

# Verify that the CP partition is exactly the one M11 expects.
use_cp, cp_cu, seq_map_r2c, seq_map_c2r = _calc_cp_seqs(
    cu_seqlens,
    H,
    16,
)

assert use_cp
assert cp_cu is not None

cp_cu_cpu = cp_cu.cpu().tolist()

print("===== CP SHAPE CHECK =====")
print("raw seq_lens :", seq_lens)
print("raw N        :", N)
print("T            :", T)
print("H            :", H)
print("D            :", D)
print("cp_cu        :", cp_cu_cpu)
print("cp_N         :", len(cp_cu_cpu) - 1)
print("r2c          :", seq_map_r2c.cpu().tolist())
print("c2r          :", seq_map_c2r.cpu().tolist())

assert len(cp_cu_cpu) - 1 == 8, (
    f"expected cp_N=8, got {len(cp_cu_cpu) - 1}"
)

# ------------------------------------------------------------
# A: old M10 path
# ------------------------------------------------------------
os.environ["FLASHKDA_CP_DIRECT_INITIAL"] = "0"

out_old = torch.empty_like(q)

fs_old = torch.empty(
    N, H, D, D,
    device=device,
    dtype=torch.bfloat16,
)

torch.cuda.synchronize()

fwd_cp(
    q, k, v, g, beta,
    scale,
    out_old,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=initial_state,
    final_state=fs_old,
    cu_seqlens=cu_seqlens,
    auto_cp=True,
)

torch.cuda.synchronize()

# ------------------------------------------------------------
# B: M11 direct initial routing
# ------------------------------------------------------------
os.environ["FLASHKDA_CP_DIRECT_INITIAL"] = "1"

out_new = torch.empty_like(q)

fs_new = torch.empty(
    N, H, D, D,
    device=device,
    dtype=torch.bfloat16,
)

torch.cuda.synchronize()

fwd_cp(
    q, k, v, g, beta,
    scale,
    out_new,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=initial_state,
    final_state=fs_new,
    cu_seqlens=cu_seqlens,
    auto_cp=True,
)

torch.cuda.synchronize()


def report(name, a, b):
    af = a.float()
    bf = b.float()

    diff = (af - bf).abs()

    exact = torch.equal(a, b)
    allclose = torch.allclose(
        af,
        bf,
        atol=0.0,
        rtol=0.0,
    )

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    mismatch = torch.count_nonzero(a != b).item()

    print()
    print(f"===== {name} =====")
    print(f"exact_equal : {exact}")
    print(f"allclose_0  : {allclose}")
    print(f"max_abs     : {max_abs:.9f}")
    print(f"mean_abs    : {mean_abs:.9f}")
    print(f"mismatch    : {mismatch}")

    return exact


out_ok = report(
    "OUTPUT: OLD vs M11",
    out_old,
    out_new,
)

fs_ok = report(
    "FINAL_STATE: OLD vs M11",
    fs_old,
    fs_new,
)

print()
print("===== FINAL RESULT =====")

if out_ok and fs_ok:
    print("M11 DIRECT INITIAL: BITWISE PASS")
else:
    print("M11 DIRECT INITIAL: FAIL")
    raise SystemExit(1)
