#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from scene.smplx_gaussian_model import SMPLXGaussianModel
from mesh_renderer import NVDiffRenderer


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)

def render_motion_sequence(dataset: ModelParams, iteration: int, pipeline: PipelineParams, motion_npz_path: str,
                          camera_id: int = 0, output_dir: str = "motion_renders", render_mesh: bool = False):
    """
    渲染SMPLX运动序列

    Args:
        dataset: 模型参数
        iteration: 迭代次数
        pipeline: 渲染管道参数
        motion_npz_path: NPZ运动序列文件路径
        camera_id: 使用的相机ID（默认0）
        output_dir: 输出目录名称
        render_mesh: 是否渲染mesh
    """
    with torch.no_grad():
        print(f"🎬 开始渲染运动序列: {motion_npz_path}")

        # 创建SMPLX高斯模型
        if dataset.bind_to_mesh:
            smplx_flag = os.path.exists(os.path.join(dataset.source_path, "canonical_smplx_param.json"))
            if smplx_flag:
                gaussians = SMPLXGaussianModel(dataset.sh_degree)
            else:
                gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            print("❌ 错误：运动序列渲染需要绑定到mesh的模型")
            return

        # 加载场景和模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 加载运动序列
        print(f"正在加载运动序列: {motion_npz_path}")
        motion_data = np.load(motion_npz_path)
        motion_params = {}
        for k, v in motion_data.items():
            if v.dtype in [np.float32, np.float64]:
                motion_params[k] = torch.from_numpy(v.astype(np.float32)).cuda()
                print(f"  加载参数 {k}: {v.shape} -> {motion_params[k].shape}")
            else:
                print(f"  跳过参数 {k}: dtype={v.dtype}")

        print(f"  总共加载了 {len(motion_params)} 个参数")

        # 检测模型类型并获取帧数
        if hasattr(gaussians, 'smplx_param') and gaussians.smplx_param is not None:
            model_type = "smplx"
            num_frames = motion_params['expression'].shape[0]
            print(f"✓ 检测到SMPLX模型，运动序列包含{num_frames}帧")
        elif hasattr(gaussians, 'flame_param') and gaussians.flame_param is not None:
            model_type = "flame"
            num_frames = motion_params['expr'].shape[0]
            print(f"✓ 检测到FLAME模型，运动序列包含{num_frames}帧")
        else:
            print("❌ 错误：无法检测模型类型或模型参数未加载")
            return

        # 获取相机
        train_cameras = scene.getTrainCameras()
        if len(train_cameras) == 0:
            print("❌ 错误：没有找到训练相机")
            return

        # 选择相机
        if camera_id >= len(train_cameras):
            camera_id = 0
            print(f"⚠ 警告：相机ID超出范围，使用相机0")

        camera = train_cameras[camera_id]
        print(f"✓ 使用相机 {camera_id}")

        # 设置输出路径
        iter_path = Path(dataset.model_path) / output_dir / f"ours_{iteration}"
        render_path = iter_path / "renders"
        if render_mesh:
            render_mesh_path = iter_path / "renders_mesh"
            makedirs(render_mesh_path, exist_ok=True)
        makedirs(render_path, exist_ok=True)

        # 设置背景
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(f"正在渲染{num_frames}帧...")

        # 渲染每一帧
        for timestep in tqdm(range(num_frames), desc="渲染进度"):
            try:
                # 应用运动参数到模型
                if model_type == "smplx":
                    # 直接构建当前帧的参数字典，不依赖训练后参数的帧数
                    frame_param = {}

                    # 从训练后参数中获取静态参数
                    if gaussians.smplx_param is not None:
                        for key, value in gaussians.smplx_param.items():
                            if key == 'betas':
                                # betas是静态的，直接使用训练后的值
                                frame_param[key] = value
                            # 其他参数先不处理，等NPZ覆盖

                    # 用NPZ中的动态参数构建当前帧参数
                    for key, value in motion_params.items():
                        if key == 'betas':
                            # 如果NPZ中有betas但训练后参数中也有，优先使用训练后的
                            if 'betas' not in frame_param:
                                frame_param[key] = value
                        else:
                            # 动态参数：使用NPZ中当前时间步的值
                            if len(value.shape) > 1:
                                frame_param[key] = value[[timestep]]  # 保持batch维度
                            else:
                                frame_param[key] = value  # 静态参数

                    # 确保所有必要的参数都存在
                    if 'betas' not in frame_param:
                        # 如果没有betas，创建默认值
                        frame_param['betas'] = torch.zeros(100, device='cuda')

                    # 使用SMPLX前向传播
                    verts, verts_cano = gaussians._smplx_forward(frame_param)
                    gaussians.update_mesh_properties(verts, verts_cano)

                elif model_type == "flame":
                    # 直接构建当前帧的FLAME参数
                    frame_param = {}

                    # 从训练后参数中获取静态参数
                    if gaussians.flame_param is not None:
                        for key, value in gaussians.flame_param.items():
                            if key in ['shape', 'static_offset']:
                                # 静态参数，直接使用训练后的值
                                frame_param[key] = value

                    # 用NPZ中的动态参数构建当前帧参数
                    for key, value in motion_params.items():
                        if key in ['shape', 'static_offset']:
                            # 如果NPZ中有静态参数但训练后参数中也有，优先使用训练后的
                            if key not in frame_param:
                                frame_param[key] = value
                        else:
                            # 动态参数：使用NPZ中当前时间步的值
                            if len(value.shape) > 1:
                                frame_param[key] = value[[timestep]]  # 保持batch维度
                            else:
                                frame_param[key] = value  # 静态参数

                    # 使用FLAME模型的前向传播
                    verts, verts_cano = gaussians.flame_model(
                        frame_param['shape'][None, ...],
                        frame_param['expr'],
                        frame_param['rotation'],
                        frame_param['neck_pose'],
                        frame_param['jaw_pose'],
                        frame_param['eyes_pose'],
                        frame_param['translation'],
                        zero_centered_at_root_node=False,
                        return_landmarks=False,
                        return_verts_cano=True,
                        static_offset=frame_param['static_offset'],
                        dynamic_offset=frame_param['dynamic_offset'],
                    )
                    gaussians.update_mesh_properties(verts, verts_cano)

                # 渲染当前帧
                rendering = render(camera, gaussians, pipeline, background)["render"]

                # 保存渲染结果
                render_image = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                Image.fromarray(render_image).save(render_path / f"frame_{timestep:06d}.png")

                # 如果需要，渲染mesh
                if render_mesh and hasattr(gaussians, 'verts') and hasattr(gaussians, 'faces'):
                    out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, camera)
                    rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)
                    rgb_mesh = rgba_mesh[:3, :, :]
                    alpha_mesh = rgba_mesh[3:, :, :]
                    mesh_opacity = 0.5

                    # 创建一个黑色背景用于mesh渲染
                    black_bg = torch.zeros_like(rendering)
                    rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + black_bg * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    mesh_image = rendering_mesh.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    Image.fromarray(mesh_image).save(render_mesh_path / f"frame_{timestep:06d}.png")

            except Exception as e:
                print(f"⚠ 警告：渲染第{timestep}帧时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 生成视频
        try:
            print("正在生成视频...")
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/motion_sequence.mp4")
            if render_mesh:
                os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/motion_sequence_mesh.mp4")
            print(f"✅ 视频已保存到: {iter_path}/motion_sequence.mp4")
        except Exception as e:
            print(f"⚠ 警告：生成视频时出错: {e}")

        print(f"✅ 运动序列渲染完成！输出目录: {iter_path}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            smplx_flag = os.path.exists(os.path.join(dataset.source_path, "canonical_smplx_param.json"))
            if smplx_flag:
                gaussians = SMPLXGaussianModel(dataset.sh_degree)
            else:
                gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh)

            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-5000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")

    # 新增：运动序列渲染参数
    parser.add_argument("--motion_npz", type=str, help="NPZ运动序列文件路径（用于渲染运动序列）")
    parser.add_argument("--camera_id", type=int, default=0, help="用于渲染的相机ID（默认0）")
    parser.add_argument("--output_dir", type=str, default="motion_renders", help="运动序列渲染输出目录名称")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 检查是否是运动序列渲染模式
    if args.motion_npz:
        print("🎬 运动序列渲染模式")
        if not os.path.exists(args.motion_npz):
            print(f"❌ 错误：运动序列文件不存在: {args.motion_npz}")
            exit(1)

        render_motion_sequence(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args.motion_npz,
            args.camera_id,
            args.output_dir,
            args.render_mesh
        )
    else:
        print("📷 标准渲染模式")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh)