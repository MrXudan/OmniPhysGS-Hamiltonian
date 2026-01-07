"""
main.py - 带详细中文注释的主训练脚本（已清理并恢复为单一、正确的模块结构）

功能概述：
 - 使用 Gaussian Splatting 表示的场景（dataset/bear）进行渲染。
 - 用 MPM（基于 warp）进行物理推进并用材料网络预测材料参数/类别。
 - 使用基于扩散的 SDS（Score Distillation Sampling）对渲染输出与文本提示进行对齐，优化材料网络。

本文件保留原始程序逻辑，添加注释以便理解各个步骤。
"""

import random
import time
import os
import shutil
import argparse
import math

import numpy as np
import warp as wp
import taichi as ti
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter


from src.mpm_core import MPMModel, set_boundary_conditions
from src.physics_guided_network import PhysicsNetwork
from src.video_distillation.ms_guidance import ModelscopeGuidance
from src.video_distillation.prompt_processors import ModelscopePromptProcessor
from src.utils.render_utils import *
from src.utils.misc_utils import *
from src.utils.camera_view_utils import load_camera_params
from src.utils.frame_io import save_gaussian_frame # 存储模拟帧数据的函数

# 初始化 warp
wp.init()  # Warp 已经绑定到 GPU


def init_training(cfg, args=None):
    """初始化训练环境并返回 (torch_device, export_path, writer)

    包含：创建导出目录、固定随机数种子、初始化 warp/taichi、设置 PyTorch 设备、导出配置文件、创建 TensorBoard writer。
    """

    # 准备输出路径
    export_path = cfg.train.export_path if cfg.train.export_path else './outputs'
    if cfg.train.train_tag is None:
        cfg.train.train_tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_path = os.path.join(export_path, cfg.train.train_tag)

    # 若目录已存在且未覆盖，则询问用户
    if os.path.exists(export_path):
        if args is not None and not args.overwrite:
            overwrite = input(f'Warning: export path {export_path} already exists. Exit?(y/n)')
            if overwrite.lower() == 'y':
                exit()
    else:
        os.makedirs(export_path)
        os.makedirs(os.path.join(export_path, 'images'))
        os.makedirs(os.path.join(export_path, 'videos'))
        os.makedirs(os.path.join(export_path, 'checkpoints'))

    # 固定随机种子，SEED代表同一组随机数，保证实验可复现
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 初始化 warp 配置
    device = f'cuda:{cfg.train.gpu}'
    wp.set_module_options({'fast_math': False})
    wp.ScopedTimer.enabled = False

    # 如果配置中启用了 particle_filling（粒子填充），初始化 taichi
    if cfg.preprocessing.particle_filling is not None:
        ti.init(arch=ti.cuda, device_memory_GB=8.0)

    # 设置 PyTorch 设备
    torch_device = torch.device(device)
    torch.cuda.set_device(cfg.train.gpu)
    torch.backends.cudnn.benchmark = False
    print(f'\nusing device: {device}\n')

    # 导出训练配置到输出目录
    print(f'exporting to: {export_path}\n')
    with open(os.path.join(export_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # 初始化 TensorBoard writer
    writer_path = os.path.join(export_path, 'writers', 'writer_' + time.strftime("%Y%m%d_%H_%M_%S"))
    os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)

    return torch_device, export_path, writer


def main(cfg, args=None):
    """主训练/渲染函数。

    逻辑大致如下：
    1. 初始化环境
    2. 加载 guidance（SDS）与 prompt processor（若训练）
    3. 加载 gaussian 场景并预处理
    4. 构建 MPM 仿真器与材料网络
    5. 跳过初始帧（warm-up）
    6. 进入 epoch/stage/internal_epoch 循环，在每个 internal_epoch 内渲染若干帧并计算 SDS 损失进行优化
    """

    train_params = cfg.train
    preprocessing_params = cfg.preprocessing
    render_params = cfg.render
    material_params = cfg.material
    model_params = cfg.model
    sim_params = cfg.sim
    guidance_params = cfg.guidance
    prompt_params = cfg.prompt_processor
    prompt_params.prompt = train_params.prompt

    torch_device, export_path, writer = init_training(cfg, args)

    # 如果启用训练，则加载 guidance（SDS）模块和 prompt processor
    if train_params.enable_train:
        print("Loading guidance and prompt processor...")
        guidance = ModelscopeGuidance(guidance_params)
        prompt_utils = ModelscopePromptProcessor(prompt_params)()
        print()

    # 加载 gaussian 场景并做必要预处理
    print("Initializing gaussian scene and pre-processing...")
    model_path = train_params.model_path
    gaussians = load_gaussian_ckpt(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if render_params.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    (
        mpm_params,
        init_e_cat, init_p_cat,
        unselected_params,
        translate_params,
        screen_points
    ) = load_params(
        gaussians, pipeline,
        preprocessing_params, material_params, model_params,
        export_path=export_path
    )

    trans_pos = mpm_params['pos']
    trans_cov = mpm_params['cov']
    trans_opacity = mpm_params['opacity']
    trans_shs = mpm_params['shs']
    trans_features = mpm_params['features']

    rotation_matrices = translate_params['rotation_matrices']
    scale_origin = translate_params['scale_origin']
    original_mean_pos = translate_params['original_mean_pos']

    gs_num = trans_pos.shape[0]
    print(f'Built gaussian particle number: {gs_num}\n')

    (viewpoint_center_worldspace, observant_coordinates) = load_camera_params(render_params, rotation_matrices, scale_origin, original_mean_pos)

    # 导出一张静态渲染图用于检查
    print(f'Exporting static gaussian rendering to {export_path}/static.png\n')
    export_static_gaussian_rendering(
        trans_pos, trans_cov, trans_shs, trans_opacity,
        unselected_params, rotation_matrices, scale_origin, original_mean_pos,
        model_path, pipeline, render_params, viewpoint_center_worldspace, observant_coordinates, gaussians, background, screen_points,
        export_path
    )

    # 构建 MPM 模型并设置边界条件
    print('Building MPM simulator and setting boundary conditions\n')
    mpm_model = MPMModel(
        sim_params,
        material_params,
        init_pos=trans_pos,
        enable_train=train_params.enable_train,
        device=torch_device
    )
    set_boundary_conditions(mpm_model, sim_params.boundary_conditions)

    # 构建材料网络（PhysicsNetwork 会根据 cfg.model.network 选择 KNN/MLP/Naive）
    material = PhysicsNetwork(
        elasticity_physicals=material_params.elasticity_physicals,
        plasticity_physicals=material_params.plasticity_physicals,
        in_channels=trans_features.shape[1],
        params=model_params,
        n_particles=gs_num,
        export_path=export_path
    ).to(torch_device).requires_grad_(train_params.enable_train)

    # 初始化具体本构（elasticity/plasticity），支持 'neural'（神经网络）或解析物理模型
    print('Loading neural constitutive model\n')
    elasticity, plasticity = init_constitute(
        material_params.elasticity,
        material_params.plasticity,
        elasticity_physicals=material_params.elasticity_physicals,
        plasticity_physicals=material_params.plasticity_physicals,
        requires_grad=train_params.enable_train,
        device=torch_device
    )

    save_model_dict_arch({'material': material, 'elasticity': elasticity, 'plasticity': plasticity}, export_path)

    # 训练相关参数
    start_epoch = 0
    epochs = train_params.epochs
    internal_epochs = train_params.internal_epochs
    num_skip_frames = sim_params.num_skip_frames
    num_frames = sim_params.num_frames
    frames_per_stage = sim_params.frames_per_stage
    assert (num_frames - num_skip_frames) % frames_per_stage == 0
    num_stages = (num_frames - num_skip_frames) // frames_per_stage
    steps_per_frame = sim_params.steps_per_frame

    # 优化器与 checkpoint 恢复
    if train_params.enable_train:
        material_opt = torch.optim.Adam(material.parameters(), lr=train_params.learning_rate)
        start_epoch = load_material_checkpoint(
            material, material_optimizer=material_opt,
            ckpt_dir=os.path.join(export_path, 'checkpoints'), epoch=train_params.ckpt_epoch, device=torch_device
        )
        print(f'\nStart training with\n{elasticity.name()}\n{plasticity.name()}')
        print(f"The prompt is: {train_params.prompt}\n")
    else:
        epochs = 1
        internal_epochs = 1
        load_material_checkpoint(material, ckpt_dir=os.path.join(export_path, 'checkpoints'), epoch=train_params.ckpt_epoch, device=torch_device)
        print('\nTraining is disabled.')
        print(f"Setting epochs to 1 and start rendering with\n{elasticity.name()}\n{plasticity.name()}\n")

    # 初始化仿真状态
    requires_grad = train_params.enable_train
    x = trans_pos.detach()
    v = torch.stack([torch.tensor([0.0, 0.0, -0.3], device=torch_device) for _ in range(gs_num)])
    C = torch.zeros((gs_num, 3, 3), device=torch_device)
    F = torch.eye(3, device=torch_device).unsqueeze(0).repeat(gs_num, 1, 1)

    x = x.requires_grad_(False)
    v = v.requires_grad_(False)
    C = C.requires_grad_(False)
    F = F.requires_grad_(False)

    # warm-up：跳过一段时间的仿真以便场景进入有意义的物理交互
    with torch.no_grad():
        mpm_model.reset()
        if material_params.elasticity == 'neural' or material_params.plasticity == 'neural':
            e_cat, p_cat = material(trans_pos, trans_features)
            e_cat = e_cat.detach().requires_grad_(False)
            p_cat = p_cat.detach().requires_grad_(False)
        else:
            e_cat, p_cat = init_e_cat, init_p_cat

        for frame in tqdm(range(num_skip_frames), desc='Skip Frames'):
            frame_id = frame
            render_pos, render_cov, render_shs, render_opacity, render_rot = get_mpm_gaussian_params(
                pos=x, cov=trans_cov, shs=trans_shs, opacity=trans_opacity,
                F=F, unselected_params=unselected_params, rotation_matrices=rotation_matrices,
                scale_origin=scale_origin, original_mean_pos=original_mean_pos
            )

            rendering = render_mpm_gaussian(
                model_path=model_path, pipeline=pipeline, render_params=render_params, step=frame_id,
                viewpoint_center_worldspace=viewpoint_center_worldspace, observant_coordinates=observant_coordinates,
                gaussians=gaussians, background=background, pos=render_pos, cov=render_cov, shs=render_shs,
                opacity=render_opacity, rot=render_rot, screen_points=screen_points
            )
            if train_params.export_video:
                export_rendering(rendering, frame_id, folder=os.path.join(export_path, 'images'))
            for step in range(steps_per_frame):
                stress = checkpoint(elasticity, F, e_cat)
                assert torch.all(torch.isfinite(stress))

                x, v, C, F = checkpoint(mpm_model, x, v, C, F, stress)
                assert torch.all(torch.isfinite(x))
                assert torch.all(torch.isfinite(F))

                F = checkpoint(plasticity, F, p_cat)
                assert torch.all(torch.isfinite(F))

    # 将 warm-up 后的状态保存为 checkpoint
    x_skip = x.detach()
    v_skip = v.detach()
    C_skip = C.detach()
    F_skip = F.detach()
    time_skip = mpm_model.time

    # 训练主循环
    for epoch in range(start_epoch, epochs):
        x_ckpt = x_skip.detach()
        v_ckpt = v_skip.detach()
        C_ckpt = C_skip.detach()
        F_ckpt = F_skip.detach()
        time_ckpt = time_skip

        if train_params.enable_train:
            epoch_lr = material_opt.param_groups[0]['lr']
            os.makedirs(os.path.join(export_path, 'videos', f'epoch_{epoch:04d}'), exist_ok=True)
            os.makedirs(os.path.join(export_path, 'checkpoints', f'epoch_{epoch:04d}'), exist_ok=True)

        for stage in tqdm(range(num_stages), desc=f'Epoch {epoch}'):
            if train_params.enable_train:
                stage_folder = os.path.join(export_path, 'videos', f'epoch_{epoch:04d}', f'stage_{stage:04d}')
                os.makedirs(stage_folder, exist_ok=True)
                if args.save_internal:
                    os.makedirs(os.path.join(export_path, 'checkpoints', f'epoch_{epoch:04d}', f'stage_{stage:04d}'), exist_ok=True)

            for internal_epoch in tqdm(range(internal_epochs), leave=False, desc=f'Internal'):
                if train_params.enable_train:
                    material_opt.zero_grad()

                x = x_ckpt.detach().requires_grad_(requires_grad)
                v = v_ckpt.detach().requires_grad_(requires_grad)
                C = C_ckpt.detach().requires_grad_(requires_grad)
                F = F_ckpt.detach().requires_grad_(requires_grad)
                mpm_model.time = time_ckpt

                trans_pos = trans_pos.detach().requires_grad_(requires_grad)
                trans_cov = trans_cov.detach().requires_grad_(requires_grad)
                trans_shs = trans_shs.detach().requires_grad_(requires_grad)
                trans_opacity = trans_opacity.detach().requires_grad_(requires_grad)
                trans_features = trans_features.detach().requires_grad_(requires_grad)

                for i in unselected_params:
                    if unselected_params[i] is not None:
                        unselected_params[i] = unselected_params[i].detach()
                scale_origin = scale_origin.detach()
                original_mean_pos = original_mean_pos.detach()
                screen_points = screen_points.detach()
                assert x.requires_grad == requires_grad

                if material_params.elasticity == 'neural' or material_params.plasticity == 'neural':
                    e_cat, p_cat = material(trans_pos, trans_features)
                else:
                    e_cat, p_cat = init_e_cat, init_p_cat

                frames = []
                for frame in tqdm(range(frames_per_stage), leave=False, desc=f'Stage {stage}'):
                    frame_id = stage * frames_per_stage + frame + num_skip_frames
                    render_pos, render_cov, render_shs, render_opacity, render_rot = get_mpm_gaussian_params(
                        pos=x, cov=trans_cov, shs=trans_shs, opacity=trans_opacity,
                        F=F, unselected_params=unselected_params, rotation_matrices=rotation_matrices,
                        scale_origin=scale_origin, original_mean_pos=original_mean_pos
                    )
                    render_scale = scale_origin[selected_mask]# 用于保存当前渲染的高斯缩放比例

                    rendering = render_mpm_gaussian(
                        model_path=model_path, pipeline=pipeline, render_params=render_params, step=frame_id,
                        viewpoint_center_worldspace=viewpoint_center_worldspace, observant_coordinates=observant_coordinates,
                        gaussians=gaussians, background=background, pos=render_pos, cov=render_cov, shs=render_shs,
                        opacity=render_opacity, rot=render_rot, screen_points=screen_points
                    )
                    
                    # 根据开关保存高斯参数帧
                    if cfg.train.dump_sim and frame_id >= cfg.train.dump_sim_start and (frame_id - cfg.train.dump_sim_start) % cfg.train.dump_sim_every == 0:
                        sim_dir = os.path.join(export_path, 'sim')
                        pack = {
                            'pos': render_pos,
                            'rot': render_rot,
                            'scale': render_scale,
                            'opacity': render_opacity,
                            'shs': render_shs
                        }
                        save_gaussian_frame(sim_dir, frame_id, pack)
                    
                    frames.append(rendering)

                # 计算 guidance（SDS）损失并进行反向传播
                if train_params.enable_train:
                    loss, guidance_loss = 0.0, 0.0
                    frames = torch.stack(frames)
                    guidance_out = guidance(
                        frames,
                        prompt_utils,
                        torch.Tensor([render_params['init_elevation']]),
                        torch.Tensor([render_params['init_azimuthm']]),
                        torch.Tensor([render_params['init_radius']]),
                        rgb_as_latents=False,
                        num_frames=frames_per_stage,
                        train_dynamic_camera=False
                    )

                    for name, value in guidance_out.items():
                        if name.startswith('loss_'):
                            guidance_loss += value * 1e-4
                    guidance_loss = guidance_loss / frames_per_stage

                    loss = guidance_loss
                    loss.backward()

                    for p in material.parameters():
                        if p is not None and p.grad is not None:
                            torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
                    torch.nn.utils.clip_grad_norm_(material.parameters(), 1.0)
                    material_opt.step()
                    with open(os.path.join(export_path, 'log.txt'), 'a', encoding='utf-8') as f:
                        lr = material_opt.param_groups[0]['lr']
                        f.write(f'epoch: {epoch}, stage: {stage}, internal_epoch: {internal_epoch}, loss: {loss}, lr: {lr}, e_cat: {e_cat.mean(dim=0).tolist()}, p_cat: {p_cat.mean(dim=0).tolist()}\n')

                    writer.add_scalar('loss', loss.item(), epoch * num_stages + stage)
                    writer.add_scalar('lr', material_opt.param_groups[0]['lr'], epoch * num_stages + stage)

                    if train_params.export_video and epoch % train_params.video_interval == 0:
                        save_video(f'{export_path}/images', os.path.join(stage_folder, f'internal_{internal_epoch:04d}.mp4'), start=stage * frames_per_stage + num_skip_frames, end=(stage + 1) * frames_per_stage + num_skip_frames)
                        if args.save_internal:
                            save_dict = {"material": material.state_dict(), "material_optimizer": material_opt.state_dict(), "epoch": epoch}
                            save_path = os.path.join(export_path, 'checkpoints', f'epoch_{epoch:04d}', f'stage_{stage:04d}', f'internal_{internal_epoch:04d}.pth')
                            torch.save(save_dict, save_path)

            # 保存 stage 结束时状态
            x_ckpt = x.detach()
            v_ckpt = v.detach()
            C_ckpt = C.detach()
            F_ckpt = F.detach()
            time_ckpt = mpm_model.time

        # 保存 checkpoint 与合成视频（按配置间隔）
        if train_params.enable_train and epoch % train_params.ckpt_interval == 0:
            save_material_checkpoint(material, material_opt, ckpt_dir=os.path.join(export_path, 'checkpoints'), epoch=epoch)

        if train_params.export_video and epoch % train_params.video_interval == 0:
            save_video(f'{export_path}/images', f'{export_path}/videos/video_{epoch:04d}.mp4')

    print(f'Training or rendering finished. The result has been exported to {export_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--guidance_config", type=str, default='./configs/ms_guidance.yaml', help="Path to the SDS guidance config file.")
    parser.add_argument("--test", action='store_true', help="Test mode.")
    parser.add_argument("--gpu", type=int, help="GPU index.")
    parser.add_argument("--tag", type=str, help="Training tag.")
    parser.add_argument("--overwrite", '-o', action='store_true', help="Overwrite the existing export folder.")
    parser.add_argument("--output", type=str, help="Output folder.")
    parser.add_argument("--save_internal", action='store_true', help="Save internal checkpoints.")

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    guidance_cfg = OmegaConf.load(args.guidance_config)
    cfg = OmegaConf.merge(cfg, guidance_cfg)

    if args.gpu is not None:
        cfg.train.gpu = args.gpu
    if args.test:
        cfg.train.enable_train = False
    if args.tag is not None:
        cfg.train.train_tag = args.tag
    if args.output is not None:
        cfg.train.export_path = args.output

    main(cfg, args)
