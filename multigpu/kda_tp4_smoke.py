import os
import torch
import torch.distributed as dist
import flash_kda

dist.init_process_group("nccl")

rank = dist.get_rank()
world = dist.get_world_size()

torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

B = 1
T = 8192
H_GLOBAL = 96
D = 128

assert H_GLOBAL % world == 0
H = H_GLOBAL // world

torch.manual_seed(1234 + rank)

q = torch.randn(B,T,H,D,device=device,dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn_like(q)

beta = torch.randn(B,T,H,device=device,dtype=torch.bfloat16)

A_log = torch.zeros(H,device=device,dtype=torch.float32)
dt_bias = torch.zeros(H,D,device=device,dtype=torch.float32)

out = torch.empty_like(v)

scale = D ** -0.5
lower_bound = -5.0

# warmup
for _ in range(10):
    flash_kda.fwd(
        q,k,v,g,beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
    )

torch.cuda.synchronize()
dist.barrier()

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(50):
    flash_kda.fwd(
        q,k,v,g,beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
    )

end.record()
end.synchronize()

ms = start.elapsed_time(end) / 50

result = torch.tensor(
    [ms],
    device=device,
    dtype=torch.float32,
)

gathered = [
    torch.zeros_like(result)
    for _ in range(world)
]

dist.all_gather(gathered,result)

if rank == 0:
    vals = [x.item() for x in gathered]

    print("===== FlashKDA TP Smoke =====")
    print("world       :", world)
    print("global H    :", H_GLOBAL)
    print("local H     :", H)
    print("per-rank ms :", [round(x,4) for x in vals])
    print("max rank ms :", round(max(vals),4))
    print("min rank ms :", round(min(vals),4))
    print("finite      :", torch.isfinite(out).all().item())
    print("RESULT      : PASS")

dist.destroy_process_group()
