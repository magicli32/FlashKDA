import os
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

B = 1
T = 8192
H_GLOBAL = 96
D = 128
OUT_DIM = 1024

assert H_GLOBAL % world == 0

H_LOCAL = H_GLOBAL // world

h0 = rank * H_LOCAL
h1 = h0 + H_LOCAL

K_GLOBAL = H_GLOBAL * D
K_LOCAL = H_LOCAL * D

scale = D ** -0.5
lower_bound = -5.0

# --------------------------------------------------
# Every rank generates IDENTICAL global tensors.
# Same seed + same shapes + same generation order.
# --------------------------------------------------
gen = torch.Generator(device=device)
gen.manual_seed(20260819)

def randn(shape, dtype=torch.bfloat16):
    return torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=gen,
    )

q_full = randn((B,T,H_GLOBAL,D))
k_full = randn((B,T,H_GLOBAL,D))
v_full = randn((B,T,H_GLOBAL,D))
g_full = randn((B,T,H_GLOBAL,D))

beta_full = randn((B,T,H_GLOBAL))

W_full = randn(
    (K_GLOBAL, OUT_DIM)
) * 0.01

A_log_full = torch.zeros(
    H_GLOBAL,
    device=device,
    dtype=torch.float32,
)

dt_bias_full = torch.zeros(
    H_GLOBAL,D,
    device=device,
    dtype=torch.float32,
)

# --------------------------------------------------
# 1. Full single-GPU-equivalent reference
# Every rank computes the same reference locally.
# --------------------------------------------------
full_kda = torch.empty_like(v_full)

flash_kda.fwd(
    q_full,
    k_full,
    v_full,
    g_full,
    beta_full,
    scale,
    full_kda,
    A_log_full,
    dt_bias_full,
    lower_bound,
)

torch.cuda.synchronize()

full_proj = torch.mm(
    full_kda.view(T, K_GLOBAL),
    W_full,
)

# --------------------------------------------------
# 2. TP local shard
# --------------------------------------------------
q = q_full[:,:,h0:h1,:].contiguous()
k = k_full[:,:,h0:h1,:].contiguous()
v = v_full[:,:,h0:h1,:].contiguous()
g = g_full[:,:,h0:h1,:].contiguous()

beta = beta_full[:,:,h0:h1].contiguous()

A_log = A_log_full[h0:h1].contiguous()
dt_bias = dt_bias_full[h0:h1,:].contiguous()

W = W_full[
    h0 * D : h1 * D,
    :
].contiguous()

local_kda = torch.empty_like(v)

flash_kda.fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    local_kda,
    A_log,
    dt_bias,
    lower_bound,
)

torch.cuda.synchronize()

# --------------------------------------------------
# Check KDA head sharding itself
# --------------------------------------------------
full_slice = full_kda[
    :,:,h0:h1,:
].contiguous()

kda_diff = (
    local_kda.float()
    - full_slice.float()
).abs()

kda_max = kda_diff.max()

# --------------------------------------------------
# Local row-parallel projection
# --------------------------------------------------
partial = torch.mm(
    local_kda.view(T, K_LOCAL),
    W,
)

# Sum TP partial results.
dist.all_reduce(
    partial,
    op=dist.ReduceOp.SUM,
)

torch.cuda.synchronize()

# --------------------------------------------------
# Compare TP layer result vs full reference
# --------------------------------------------------
diff = (
    partial.float()
    - full_proj.float()
)

abs_diff = diff.abs()

max_abs = abs_diff.max()
mean_abs = abs_diff.mean()

rmse = torch.sqrt(
    torch.mean(diff * diff)
)

ref_norm = torch.linalg.vector_norm(
    full_proj.float()
)

err_norm = torch.linalg.vector_norm(
    diff
)

rel_l2 = err_norm / ref_norm

# Worst rank values.
for x in [
    kda_max,
    max_abs,
    mean_abs,
    rmse,
    rel_l2,
]:
    dist.all_reduce(
        x,
        op=dist.ReduceOp.MAX,
    )

# Local KDA is expected to be extremely close;
# also explicitly test bitwise.
kda_equal = torch.tensor(
    [
        int(torch.equal(
            local_kda,
            full_slice,
        ))
    ],
    device=device,
    dtype=torch.int32,
)

dist.all_reduce(
    kda_equal,
    op=dist.ReduceOp.MIN,
)

if rank == 0:

    print()
    print("===== FlashKDA Cross-TP Correctness =====")
    print("world              :", world)
    print("global H           :", H_GLOBAL)
    print("local H            :", H_LOCAL)
    print()

    print("----- KDA shard correctness -----")
    print(
        "KDA bitwise equal  :",
        bool(kda_equal.item())
    )
    print(
        "KDA max abs diff   :",
        kda_max.item()
    )

    print()
    print("----- Full TP layer correctness -----")
    print(
        "projection max abs :",
        max_abs.item()
    )
    print(
        "projection mean abs:",
        mean_abs.item()
    )
    print(
        "projection RMSE    :",
        rmse.item()
    )
    print(
        "projection rel L2  :",
        rel_l2.item()
    )

dist.destroy_process_group()
