#存储模拟帧数据的函数
from pathlib import Path
import torch

def save_gaussian_frame(sim_dir: str, frame_id: int, pack: dict):
    """
    Save one frame of gaussian params to sim_dir/frame_XXX.pt
    pack: dict[str, torch.Tensor or python scalar]
    """
    sim_dir = Path(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    cpu_pack = {}
    for k, v in pack.items():
        if torch.is_tensor(v):
            cpu_pack[k] = v.detach().cpu()
        else:
            cpu_pack[k] = v
    torch.save(cpu_pack, sim_dir / f"frame_{frame_id:03d}.pt")
