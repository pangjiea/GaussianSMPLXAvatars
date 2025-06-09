#!/usr/bin/env python3
"""
PLY坐标转换脚本：将局部坐标的PLY文件转换为全局坐标的PLY文件

适用于绑定到SMPLX/FLAME网格的高斯点，将其从局部坐标系转换为全局坐标系。

使用方法:
    python ply_local_to_global.py --input input.ply --output output_global.ply
    python ply_local_to_global.py --input input.ply --output output_global.ply --model smplx --timestep 0
"""

import torch
import numpy as np
import sys
import os
import argparse
from plyfile import PlyData, PlyElement

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

        # 创建模型实例
        gaussians = GaussianModelClass(sh_degree=sh_degree)
        print(f"✓ 创建{model_type.upper()}模型")

        # 加载PLY文件
        gaussians.load_ply(input_ply_path)
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
        description="转换局部坐标PLY文件为全局坐标PLY文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用训练后的参数（推荐）
  python ply_local_to_global.py --input local.ply --output global.ply
  python ply_local_to_global.py --input smplx_local.ply --output smplx_global.ply --model smplx

  # 使用原始mesh参数（如果训练后的结果不理想）
  python ply_local_to_global.py --input smplx_local.ply --output smplx_global.ply --model smplx --use_original_mesh

  # 指定特定时间步
  python ply_local_to_global.py --input flame_local.ply --output flame_global.ply --model flame --timestep 5
        """
    )

    parser.add_argument("--input", "-i", required=True, help="输入PLY文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出PLY文件路径")
    parser.add_argument("--model", "-m", default="auto",
                       choices=["auto", "smplx", "flame", "gaussian"],
                       help="模型类型 (默认: auto - 自动检测)")
    parser.add_argument("--timestep", "-t", type=int, default=0,
                       help="时间步 (默认: 0)")
    parser.add_argument("--sh_degree", type=int, default=3,
                       help="球谐度数 (默认: 3)")
    parser.add_argument("--use_original_mesh", action="store_true",
                       help="使用原始mesh参数而不是训练后的参数")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        return 1

    # 执行转换
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

if __name__ == "__main__":
    sys.exit(main())
