import torch
import flash_kda

torch.set_grad_enabled(False)

device = torch.device("cuda:0")
torch.cuda.set_device(device)

B = 1
T = 8192
SPLIT = 4096
H = 96
D = 128

scale = D ** -0.5
lower_bound = -5.0

torch.manual_seed(20260819)

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

# ==================================================
# Full T=8192 reference
# ==================================================
out_full = torch.empty_like(v)

state_full = torch.empty(
    B,H,D,D,
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

# ==================================================
# Chunk 0: [0,4096)
# ==================================================
q0 = q[:,:SPLIT].contiguous()
k0 = k[:,:SPLIT].contiguous()
v0 = v[:,:SPLIT].contiguous()
g0 = g[:,:SPLIT].contiguous()
b0 = beta[:,:SPLIT].contiguous()

out0 = torch.empty_like(v0)

state0 = torch.empty(
    B,H,D,D,
    device=device,
    dtype=torch.bfloat16,
)

flash_kda.fwd(
    q0,k0,v0,g0,b0,
    scale,
    out0,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=state0,
)

torch.cuda.synchronize()

# ==================================================
# Chunk 1: [4096,8192)
# state0 becomes initial_state
# ==================================================
q1 = q[:,SPLIT:].contiguous()
k1 = k[:,SPLIT:].contiguous()
v1 = v[:,SPLIT:].contiguous()
g1 = g[:,SPLIT:].contiguous()
b1 = beta[:,SPLIT:].contiguous()

out1 = torch.empty_like(v1)

state1 = torch.empty_like(state0)

flash_kda.fwd(
    q1,k1,v1,g1,b1,
    scale,
    out1,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=state0,
    final_state=state1,
)

torch.cuda.synchronize()

# ==================================================
# Compare
# ==================================================
out_split = torch.cat([out0,out1], dim=1)

out_diff = (
    out_split.float()
    - out_full.float()
).abs()

state_diff = (
    state1.float()
    - state_full.float()
).abs()

print()
print("===== FlashKDA Chunk-Chain Correctness =====")
print("full T             :", T)
print("split              :", SPLIT, "+", T-SPLIT)
print("state shape        :", list(state0.shape))
print(
    "state size MiB     :",
    state0.numel() * state0.element_size() / 1024 / 1024
)
print()

print("output bitwise     :", torch.equal(out_split, out_full))
print("output max abs     :", out_diff.max().item())
print("output mean abs    :", out_diff.mean().item())

print()
print("final state bitwise:", torch.equal(state1, state_full))
print("state max abs      :", state_diff.max().item())
print("state mean abs     :", state_diff.mean().item())
