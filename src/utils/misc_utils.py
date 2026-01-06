from typing import *

import os
import torch
import cv2
from torch.nn import Module
from torch.optim import Optimizer
from omegaconf import DictConfig
from matplotlib import pyplot as plt

from src.constitutive_models import get_elasticity, get_plasticity
from .render_utils import *

def export_static_gaussian_rendering(
    trans_pos, trans_cov, trans_shs,trans_opacity,
    unselected_params, rotation_matrices, scale_origin, original_mean_pos,
    model_path, pipeline, render_params, viewpoint_center_worldspace, observant_coordinates, gaussians, background, screen_points,
    export_path
):
    
    gs_num = trans_pos.shape[0]
    F = torch.eye(3, device=trans_pos.device).unsqueeze(0).repeat(gs_num, 1, 1)
    
    (
        render_pos,
        render_cov,
        render_shs,
        render_opacity,
        render_rot
    ) = get_mpm_gaussian_params(
        pos=trans_pos, cov=trans_cov, shs=trans_shs, opacity=trans_opacity,
        F=F,
        unselected_params=unselected_params,
        rotation_matrices=rotation_matrices,
        scale_origin=scale_origin,
        original_mean_pos=original_mean_pos
    )
    
    rendering = render_mpm_gaussian(
        model_path=model_path,
        pipeline=pipeline,
        render_params=render_params,
        step=0,
        viewpoint_center_worldspace=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        gaussians=gaussians,
        background=background,
        pos=render_pos,
        cov=render_cov,
        shs=render_shs,
        opacity=render_opacity,
        rot=render_rot,
        screen_points=screen_points
    )
    
    export_rendering(rendering, 'static', folder=export_path)


def load_reference_images(
    folder: str,
    export_path: str,
    params: DictConfig
):
    """Load reference images from the given directory."""
    print(f"Load reference images from {folder}\n")
    reference_images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(folder, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            reference_images.append(image)
            
    assert len(reference_images) >= params.num_frames, f"Expected {params.num_frames} reference images, but got {len(reference_images)}"
    save_video(folder, os.path.join(export_path, "reference.mp4"))
    return reference_images


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def save_model_architecture(model: Module, directory: str, model_name='model') -> None:
    """Save the model architecture to a `.txt` file."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f'Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n' + str(model)

    with open(os.path.join(directory, f'{model_name}.txt'), 'w') as f:
        f.write(message)

def save_model_dict_arch(model: Dict[str, Module], directory: str) -> None:
    for key, model in model.items():
        save_model_architecture(model, directory, model_name=key)
        
def save_checkpoints(
    elasticity_model: Module,
    plasticity_model: Module,
    elasticity_optimizer: Optimizer,
    plasticity_optimizer: Optimizer, 
    ckpt_dir: str, 
    epoch: int
) -> None:
    """Save checkpoint to the given experiment directory."""
    save_dict = {
        "elasticity": elasticity_model.state_dict(),
        "plasticity": plasticity_model.state_dict(),
        "elasticity_optimizer": elasticity_optimizer.state_dict(),
        "plasticity_optimizer": plasticity_optimizer.state_dict(),
        "epoch": epoch,
    }
    save_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
    torch.save(save_dict, save_path)        

def save_material_checkpoint(
    material_model: Module,
    material_optimizer: Optimizer,
    ckpt_dir: str,
    epoch: int
) -> None:
    """Save checkpoint to the given experiment directory."""
    save_dict = {
        "material": material_model.state_dict(),
        "material_optimizer": material_optimizer.state_dict(),
        "epoch": epoch,
    }
    save_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
    torch.save(save_dict, save_path)

def load_material_checkpoint(
    material_model: Module,
    ckpt_dir: str,
    material_optimizer: Optional[Optimizer]=None,
    epoch: Optional[int]=None,
    device='cuda',
    eval = False
):
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    if epoch is not None and epoch < 0:
        epoch = None

    model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".pth")]

    if len(model_files) == 0:  # no checkpoints found
        if eval:
            raise ValueError(f"Checkpoint file not found")
        print(f"No checkpoint found in {ckpt_dir}, starting from scratch")
        return 0

    epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
    checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
    if not os.path.exists(checkpoint_path):  # checkpoint file not found
        if eval:
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        print(f"Checkpoint file {checkpoint_path} not found, starting from scratch")
        return 0

    print(f"Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    material_model.load_state_dict(checkpoint["material"])
    if material_optimizer is not None:
        material_optimizer.load_state_dict(checkpoint["material_optimizer"])

    return checkpoint["epoch"] + 1


def load_checkpoints(
    elasticity_model: Module,
    plasticity_model: Module,
    ckpt_dir: str,
    elasticity_optimizer: Optional[Optimizer]=None,
    plasticity_optimizer: Optional[Optimizer]=None,
    epoch: Optional[int]=None,
    device='cuda',
    eval = False
):
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    if epoch is not None and epoch < 0:
        epoch = None

    model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".pth")]

    if len(model_files) == 0:  # no checkpoints found
        if eval:
            raise ValueError(f"Checkpoint file not found")
        print(f"No checkpoint found in {ckpt_dir}, starting from scratch")
        return 0

    epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
    checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
    if not os.path.exists(checkpoint_path):  # checkpoint file not found
        if eval:
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        print(f"Checkpoint file {checkpoint_path} not found, starting from scratch")
        return 0

    print(f"Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    elasticity_model.load_state_dict(checkpoint["elasticity"])
    plasticity_model.load_state_dict(checkpoint["plasticity"])
    
    elasticity_optimizer.load_state_dict(checkpoint["elasticity_optimizer"])
    plasticity_optimizer.load_state_dict(checkpoint["plasticity_optimizer"])

    return checkpoint["epoch"] + 1

def init_constitute(
    elasticity_name: str,
    plasticity_name: str,
    elasticity_physicals: Optional[List[str]]=None,
    plasticity_physicals: Optional[List[str]]=None,
    requires_grad: bool=True,
    device='cuda'
):
    elasticity = get_elasticity(
        elasticity_name,
        physicals=elasticity_physicals,
        device=device
    )
    elasticity.to(device)
    elasticity.requires_grad_(requires_grad)
    
    plasticity = get_plasticity(
        plasticity_name,
        physicals=plasticity_physicals,
        device=device
    )
    plasticity.to(device)
    plasticity.requires_grad_(requires_grad)
    
    return elasticity, plasticity


def plot_3d(xyz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()