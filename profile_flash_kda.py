import math
import torch
import torch.nn.functional as F
import flash_kda

torch.manual_seed(0)

B, T, H, D = 1, 8192, 96, 128
device = "cuda"
dtype = torch.bfloat16

q = F.normalize(
    torch.randn(B, T, H, D, device=device, dtype=torch.float32),
    dim=-1
).to(dtype)

k = F.normalize(
    torch.randn(B, T, H, D, device=device, dtype=torch.float32),
    dim=-1
).to(dtype)

v = torch.randn(B, T, H, D, device=device, dtype=dtype)
g = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.randn(B, T, H, device=device, dtype=dtype)

A_log = torch.rand(H, device=device, dtype=torch.float32)
dt_bias = torch.rand(H, D, device=device, dtype=torch.float32)

initial_state = torch.zeros(
    B, H, D, D, device=device, dtype=dtype
)
final_state = torch.zeros_like(initial_state)
out = torch.zeros_like(q)

@torch.inference_mode()
def run():
    flash_kda.fwd(
        q, k, v, g, beta,
        1.0 / math.sqrt(D),
        out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state,
        final_state=final_state,
    )

# 预热不采集
for _ in range(5):
    run()
torch.cuda.synchronize()

# 只采集这一次 FlashKDA
torch.cuda.cudart().cudaProfilerStart()
run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

print("Profiled one FlashKDA forward call")
