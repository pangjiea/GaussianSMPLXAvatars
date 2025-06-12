#!/usr/bin/env python3
"""
PLY坐标转换脚本：将局部坐标的PLY文件转换为全局坐标的PLY文件

适用于绑定到SMPLX/FLAME网格的高斯点，将其从局部坐标系转换为全局坐标系。
支持输入npz运动序列，生成对应的PLY文件序列。

使用方法:
    # 单个时间步转换
    python ply_local_to_global.py --input input.ply --output output_global.ply
    python ply_local_to_global.py --input input.ply --output output_global.ply --model smplx --timestep 0

    # 使用npz运动序列生成PLY序列
    python ply_local_to_global.py --input input.ply --motion motion.npz --output_dir output_plys/
    python ply_local_to_global.py --input input.ply --motion motion.npz --output_dir output_plys/ --model smplx
"""

import torch
import numpy as np
import sys
import os
import argparse
from plyfile import PlyData, PlyElement
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.append('.')

def detect_model_type(ply_path):
    """根据PLY文件路径和相关文件自动检测模型类型"""
    path_lower = ply_path.lower()

    # 首先检查文件名
    if "smplx" in path_lower:
        return "smplx"
    elif "flame" in path_lower:
        return "flame"

    # 检查同目录下是否有参数文件
    import os
    ply_dir = os.path.dirname(ply_path)

    if os.path.exists(os.path.join(ply_dir, "smplx_param.npz")):
        return "smplx"
    elif os.path.exists(os.path.join(ply_dir, "flame_param.npz")):
        return "flame"
    else:
        return "gaussian"

def convert_with_motion_sequence(input_ply_path, motion_npz_path, output_dir, model_type="smplx", sh_degree=3, use_original_mesh=False, step_size=5, no_interpolate=False):
    """
    使用npz运动序列生成PLY文件序列

    Args:
        input_ply_path: 输入PLY文件路径（局部坐标）
        motion_npz_path: npz运动序列文件路径
        output_dir: 输出目录
        model_type: 模型类型 ("auto", "smplx", "flame", "gaussian")
        sh_degree: 球谐度数（默认3）
        use_original_mesh: 是否使用原始mesh参数而不是训练后的参数（默认False）
    """
    print(f"=== 使用运动序列生成PLY序列 ===")
    print(f"输入PLY: {input_ply_path}")
    print(f"运动序列: {motion_npz_path}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 自动检测模型类型
        if model_type == "auto":
            model_type = detect_model_type(input_ply_path)
            print(f"✓ 自动检测模型类型: {model_type}")

        # 根据模型类型导入相应的类
        print(f"正在导入{model_type.upper()}模型类...")
        if model_type.lower() == "smplx":
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            GaussianModelClass = SMPLXGaussianModel
        elif model_type.lower() == "flame":
            from scene.flame_gaussian_model import FlameGaussianModel
            GaussianModelClass = FlameGaussianModel
        else:
            print("❌ 错误：npz运动序列只支持SMPLX或FLAME模型")
            return False
        print("✓ 模型类导入成功")

        # 创建模型实例
        print(f"正在创建{model_type.upper()}模型实例...")
        gaussians = GaussianModelClass(sh_degree=sh_degree)
        print(f"✓ 创建{model_type.upper()}模型")

        # 先只加载PLY文件（不加载motion）
        print(f"正在加载PLY文件...")
        gaussians.load_ply(input_ply_path, has_target=False)
        print(f"✓ 加载PLY文件，包含{gaussians._xyz.shape[0]}个高斯点")

        # 检查是否有绑定信息
        if gaussians.binding is None:
            print("❌ 错误：PLY文件没有绑定信息，无法应用运动序列")
            return False

        print(f"✓ 发现绑定信息，绑定到{gaussians.binding.max().item() + 1}个面")

        # 加载运动序列数据
        print(f"正在加载运动序列: {motion_npz_path}")
        motion_data = np.load(motion_npz_path)

        # 改进的参数加载，支持更多数据类型
        motion_params = {}
        for k, v in motion_data.items():
            if v.dtype in [np.float32, np.float64]:
                motion_params[k] = torch.from_numpy(v.astype(np.float32)).cuda()
                print(f"  加载参数 {k}: {v.shape} -> {motion_params[k].shape}")
            else:
                print(f"  跳过参数 {k}: dtype={v.dtype}")

        # 检测原始帧数
        if model_type.lower() == "smplx":
            original_frames = motion_params['expression'].shape[0] if 'expression' in motion_params else 0
            print(f"✓ 加载SMPLX运动序列，包含{original_frames}帧原始数据")
        elif model_type.lower() == "flame":
            original_frames = motion_params['expr'].shape[0] if 'expr' in motion_params else 0
            print(f"✓ 加载FLAME运动序列，包含{original_frames}帧原始数据")
        else:
            print(f"❌ 不支持的模型类型: {model_type}")
            return False

        if original_frames == 0:
            print(f"❌ 无法检测运动序列帧数")
            return False

        # 检测是否需要跳帧插值
        auto_interpolate = not no_interpolate and step_size > 1

        if auto_interpolate and step_size > 1:
            target_frames = original_frames * step_size
            print(f"🔄 自动检测到跳帧数据，执行插值：{original_frames} 帧 -> {target_frames} 帧（步长 {step_size}）")

            # 对所有动态参数进行插值
            interpolated_params = {}
            for key, value in motion_params.items():
                if key == 'betas' or (model_type.lower() == "flame" and key in ['shape', 'static_offset']) or len(value.shape) == 1:
                    # 静态参数，直接复制
                    interpolated_params[key] = value
                    print(f"    {key}: 静态参数，保持 {value.shape}")
                else:
                    # 动态参数，进行线性插值
                    interpolated_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).transpose(1, 2),  # (1, dim, original_frames)
                        size=target_frames,
                        mode='linear',
                        align_corners=True
                    ).transpose(1, 2).squeeze(0)  # (target_frames, dim)

                    interpolated_params[key] = interpolated_value
                    print(f"    {key}: 插值 {value.shape} -> {interpolated_value.shape}")

            motion_params = interpolated_params
            num_frames = target_frames
            print(f"✓ 插值完成，最终处理 {num_frames} 帧数据")
        else:
            num_frames = original_frames
            print(f"✓ 使用原始数据，处理 {num_frames} 帧")

        # 为每一帧生成PLY文件
        print(f"正在生成{num_frames}个PLY文件...")
        for timestep in tqdm(range(num_frames), desc="生成PLY文件"):
            try:
                # 使用与select_mesh_by_timestep相同的逻辑，但应用NPZ运动序列
                if model_type.lower() == "smplx":
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

                elif model_type.lower() == "flame":
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

                # 获取全局坐标
                global_xyz = gaussians.get_xyz.detach().cpu().numpy()
                global_scaling = gaussians.get_scaling.detach().cpu().numpy()
                global_rotation = gaussians.get_rotation.detach().cpu().numpy()

                # 调试信息：对比关键帧
                if timestep == 0 or timestep % 100 == 0:
                    local_xyz = gaussians._xyz.detach().cpu().numpy()
                    print(f"第{timestep}帧调试信息:")
                    print(f"  局部坐标范围: [{local_xyz.min():.3f}, {local_xyz.max():.3f}]")
                    print(f"  全局坐标范围: [{global_xyz.min():.3f}, {global_xyz.max():.3f}]")
                    print(f"  坐标中心偏移: {(global_xyz.mean(axis=0) - local_xyz.mean(axis=0))}")

                    # 检查是否有变换
                    if np.allclose(local_xyz, global_xyz, atol=1e-6):
                        print("  ⚠ 警告：局部和全局坐标相同，可能没有变换")
                    else:
                        print("  ✓ 检测到坐标变换")

                # 生成输出文件名（使用连续序号）
                output_ply_path = output_path / f"frame_{timestep:06d}.ply"

                # 保存PLY文件
                save_global_ply(gaussians, str(output_ply_path), global_xyz, global_scaling, global_rotation)

            except Exception as e:
                print(f"⚠ 警告：处理第{timestep}帧时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"✅ 成功生成{num_frames}个PLY文件到目录: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_local_to_global_ply(input_ply_path, output_ply_path, model_type="smplx", timestep=0, sh_degree=3, use_original_mesh=False):
    """
    将局部坐标的PLY文件转换为全局坐标的PLY文件

    Args:
        input_ply_path: 输入PLY文件路径（局部坐标）
        output_ply_path: 输出PLY文件路径（全局坐标）
        model_type: 模型类型 ("auto", "smplx", "flame", "gaussian")
        timestep: 时间步（默认0）
        sh_degree: 球谐度数（默认3）
        use_original_mesh: 是否使用原始mesh参数而不是训练后的参数（默认False）
    """
    print(f"=== 转换PLY文件：{input_ply_path} -> {output_ply_path} ===")

    try:
        # 自动检测模型类型
        if model_type == "auto":
            model_type = detect_model_type(input_ply_path)
            print(f"✓ 自动检测模型类型: {model_type}")

        # 根据模型类型导入相应的类
        print(f"正在导入{model_type.upper()}模型类...")
        if model_type.lower() == "smplx":
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            GaussianModelClass = SMPLXGaussianModel
        elif model_type.lower() == "flame":
            from scene.flame_gaussian_model import FlameGaussianModel
            GaussianModelClass = FlameGaussianModel
        else:
            # 使用基础GaussianModel
            from scene.gaussian_model import GaussianModel
            GaussianModelClass = GaussianModel
        print("✓ 模型类导入成功")

        # 创建模型实例
        print(f"正在创建{model_type.upper()}模型实例...")
        gaussians = GaussianModelClass(sh_degree=sh_degree)
        print(f"✓ 创建{model_type.upper()}模型")

        # 加载PLY文件
        gaussians.load_ply(input_ply_path, has_target=False)
        print(f"✓ 加载PLY文件，包含{gaussians._xyz.shape[0]}个高斯点")

        # 检查是否有绑定信息
        if gaussians.binding is None:
            print("⚠ 警告：PLY文件没有绑定信息，坐标可能已经是全局的")
            # 直接复制文件
            import shutil
            shutil.copy2(input_ply_path, output_ply_path)
            print(f"✓ 直接复制文件到: {output_ply_path}")
            return True

        print(f"✓ 发现绑定信息，绑定到{gaussians.binding.max().item() + 1}个面")

        # 如果是SMPLX或FLAME模型，需要加载mesh数据
        if hasattr(gaussians, 'select_mesh_by_timestep'):
            try:
                # 检查是否有训练后的参数
                if hasattr(gaussians, 'smplx_param') and gaussians.smplx_param is not None:
                    if use_original_mesh:
                        print("✓ 使用原始SMPLX参数")
                        gaussians.select_mesh_by_timestep(timestep, original=True)
                    else:
                        print("✓ 使用训练后的SMPLX参数")
                        gaussians.select_mesh_by_timestep(timestep, original=False)
                elif hasattr(gaussians, 'flame_param') and gaussians.flame_param is not None:
                    if use_original_mesh:
                        print("✓ 使用原始FLAME参数")
                        gaussians.select_mesh_by_timestep(timestep, original=True)
                    else:
                        print("✓ 使用训练后的FLAME参数")
                        gaussians.select_mesh_by_timestep(timestep, original=False)
                else:
                    print("⚠ 警告：没有找到mesh参数，尝试使用默认参数")
                    gaussians.select_mesh_by_timestep(timestep)
                print(f"✓ 选择时间步: {timestep} ({'原始参数' if use_original_mesh else '训练后参数'})")
            except NotImplementedError:
                print("❌ 错误：select_mesh_by_timestep未实现，需要手动设置face_*属性")
                print("提示：请确保模型已正确加载mesh数据")
                return False
            except Exception as e:
                print(f"❌ 选择时间步时出错: {e}")
                print("提示：可能mesh参数文件损坏或不兼容")
                return False

        # 获取局部和全局坐标
        local_xyz = gaussians._xyz.detach().cpu().numpy()
        global_xyz = gaussians.get_xyz.detach().cpu().numpy()
        global_scaling = gaussians.get_scaling.detach().cpu().numpy()
        global_rotation = gaussians.get_rotation.detach().cpu().numpy()

        print(f"局部坐标范围: [{local_xyz.min():.3f}, {local_xyz.max():.3f}]")
        print(f"全局坐标范围: [{global_xyz.min():.3f}, {global_xyz.max():.3f}]")
        print(f"坐标中心偏移: {(global_xyz.mean(axis=0) - local_xyz.mean(axis=0))}")

        # 检查是否有变换
        if np.allclose(local_xyz, global_xyz, atol=1e-6):
            print("⚠ 警告：局部和全局坐标相同，可能没有变换")
        else:
            print("✓ 检测到坐标变换")

        # 保存全局坐标的PLY文件
        save_global_ply(gaussians, output_ply_path, global_xyz, global_scaling, global_rotation)
        print(f"✓ 成功保存全局坐标PLY文件: {output_ply_path}")

        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_global_ply(gaussians, path, xyz_global, scaling_global, rotation_global):
    """
    保存全局坐标的PLY文件
    """
    from utils.system_utils import mkdir_p

    mkdir_p(os.path.dirname(path))

    # 获取其他属性（不需要坐标变换）
    normals = np.zeros_like(xyz_global)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()

    # 将scaling转换回log空间（PLY格式要求）
    scale_log = gaussians.scaling_inverse_activation(torch.tensor(scaling_global)).cpu().numpy()

    # 构建属性列表
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(f_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale_log.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation_global.shape[1]):
        l.append('rot_{}'.format(i))

    # 注意：不包含binding信息，因为坐标已经是全局的

    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(xyz_global.shape[0], dtype=dtype_full)

    # 连接所有属性
    attributes = np.concatenate((
        xyz_global,
        normals,
        f_dc,
        f_rest,
        opacities,
        scale_log,
        rotation_global
    ), axis=1)

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="转换局部坐标PLY文件为全局坐标PLY文件，支持npz运动序列",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个时间步转换
  python ply_local_to_global.py --input local.ply --output global.ply
  python ply_local_to_global.py --input smplx_local.ply --output smplx_global.ply --model smplx

  # 使用原始mesh参数（如果训练后的结果不理想）
  python ply_local_to_global.py --input smplx_local.ply --output smplx_global.ply --model smplx --use_original_mesh

  # 指定特定时间步
  python ply_local_to_global.py --input flame_local.ply --output flame_global.ply --model flame --timestep 5

  # 使用npz运动序列生成PLY序列
  python ply_local_to_global.py --input local.ply --motion motion.npz --output_dir output_plys/
  python ply_local_to_global.py --input smplx_local.ply --motion smplx_motion.npz --output_dir smplx_plys/ --model smplx

  # 处理multiprocess步长5的跳帧数据（自动插值）
  python ply_local_to_global.py --input output/sc01-5/point_cloud/iteration_300000/point_cloud.ply --motion output/sc01-5/point_cloud/iteration_300000/smplx_param.npz --output_dir motionply/ --model smplx

  # 禁用插值或自定义步长
  python ply_local_to_global.py --input local.ply --motion motion.npz --output_dir output_plys/ --no_interpolate
  python ply_local_to_global.py --input local.ply --motion motion.npz --output_dir output_plys/ --step_size 10
        """
    )

    parser.add_argument("--input", "-i", required=True, help="输入PLY文件路径")
    parser.add_argument("--output", "-o", help="输出PLY文件路径（单个文件模式）")
    parser.add_argument("--motion", help="npz运动序列文件路径（序列模式）")
    parser.add_argument("--output_dir", help="输出目录路径（序列模式）")
    parser.add_argument("--model", "-m", default="auto",
                       choices=["auto", "smplx", "flame", "gaussian"],
                       help="模型类型 (默认: auto - 自动检测)")
    parser.add_argument("--timestep", "-t", type=int, default=0,
                       help="时间步 (默认: 0，仅单个文件模式)")
    parser.add_argument("--sh_degree", type=int, default=3,
                       help="球谐度数 (默认: 3)")
    parser.add_argument("--use_original_mesh", action="store_true",
                       help="使用原始mesh参数而不是训练后的参数")
    parser.add_argument("--step_size", type=int, default=5,
                       help="跳帧步长，用于插值处理（默认: 5，设为1禁用插值）")
    parser.add_argument("--no_interpolate", action="store_true",
                       help="禁用自动插值功能")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        return 1

    # 检查模式：序列模式 vs 单个文件模式
    if args.motion and args.output_dir:
        # 序列模式：使用npz运动序列
        if not os.path.exists(args.motion):
            print(f"错误：运动序列文件不存在: {args.motion}")
            return 1

        print("🎬 序列模式：使用npz运动序列生成PLY文件序列")
        success = convert_with_motion_sequence(
            args.input,
            args.motion,
            args.output_dir,
            args.model,
            args.sh_degree,
            args.use_original_mesh,
            args.step_size,
            args.no_interpolate
        )

        if success:
            print("✅ 序列转换成功完成！")
            print(f"PLY文件序列已保存到目录: {args.output_dir}")
            return 0
        else:
            print("❌ 序列转换失败！")
            return 1

    elif args.output:
        # 单个文件模式
        print("📄 单个文件模式：转换指定时间步的PLY文件")
        success = convert_local_to_global_ply(
            args.input,
            args.output,
            args.model,
            args.timestep,
            args.sh_degree,
            args.use_original_mesh
        )

        if success:
            print("✅ 转换成功完成！")
            print(f"全局坐标PLY文件已保存到: {args.output}")
            return 0
        else:
            print("❌ 转换失败！")
            return 1
    else:
        print("❌ 错误：请指定输出模式")
        print("  单个文件模式：使用 --output 参数")
        print("  序列模式：使用 --motion 和 --output_dir 参数")
        return 1

if __name__ == "__main__":
    sys.exit(main())
