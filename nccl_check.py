"""8 GPU の通信を本番の前に数十秒で確かめる(all_reduce の結果が world_size と一致するか)。"""
import os
import torch
import torch.distributed as dist
dist.init_process_group("nccl")
r, w = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(64 * 1024 * 1024, device="cuda")   # 256MB
for _ in range(5):
    y = x.clone()
    dist.all_reduce(y)
    assert torch.allclose(y, torch.full_like(y, w)), "all_reduce の値が違う"
torch.cuda.synchronize()
if r == 0:
    print(f"nccl ok world={w} {torch.cuda.get_device_name()}", flush=True)
dist.destroy_process_group()
