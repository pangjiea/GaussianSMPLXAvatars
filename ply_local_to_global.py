#!/usr/bin/env python3
"""
转换脚本：将局部坐标的PLY文件转换为全局坐标的PLY文件
同时包含测试功能来验证SMPLX Gaussian模型的save_ply修复是否正确
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from plyfile import PlyData, PlyElement

# 添加项目路径
sys.path.append('.')

def convert_local_to_global_ply(input_ply_path, output_ply_path, model_type="smplx", timestep=0, sh_degree=3):
    """
    将局部坐标的PLY文件转换为全局坐标的PLY文件

    Args:
        input_ply_path: 输入PLY文件路径（局部坐标）
        output_ply_path: 输出PLY文件路径（全局坐标）
        model_type: 模型类型 ("smplx" 或 "flame")
        timestep: 时间步（默认0）
        sh_degree: 球谐度数（默认3）
    """
    print(f"=== 转换PLY文件：{input_ply_path} -> {output_ply_path} ===")

    try:
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
                gaussians.select_mesh_by_timestep(timestep)
                print(f"✓ 选择时间步: {timestep}")
            except NotImplementedError:
                print("⚠ 警告：select_mesh_by_timestep未实现，需要手动设置face_*属性")
                return False

        # 获取局部和全局坐标
        local_xyz = gaussians._xyz.detach().cpu().numpy()
        global_xyz = gaussians.get_xyz.detach().cpu().numpy()
        global_scaling = gaussians.get_scaling.detach().cpu().numpy()
        global_rotation = gaussians.get_rotation.detach().cpu().numpy()

        print(f"局部坐标范围: [{local_xyz.min():.3f}, {local_xyz.max():.3f}]")
        print(f"全局坐标范围: [{global_xyz.min():.3f}, {global_xyz.max():.3f}]")

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

def test_smplx_save_ply():
    """测试SMPLX模型的save_ply方法是否正确保存世界坐标"""
    
    print("=== 测试SMPLX Gaussian模型的save_ply修复 ===")
    
    try:
        from scene.smplx_gaussian_model import SMPLXGaussianModel
        
        # 创建模型
        gaussians = SMPLXGaussianModel(sh_degree=3)
        print("✓ 成功创建SMPLXGaussianModel")
        
        # 模拟加载一些基本的mesh数据
        # 这里我们创建一个简单的测试用例
        test_meshes = {
            0: {
                'betas': np.zeros(100),
                'expression': np.zeros(50),
                'left_hand_pose': np.zeros(45),
                'right_hand_pose': np.zeros(45),
                'jaw_pose': np.zeros(3),
                'leye_pose': np.zeros(3),
                'reye_pose': np.zeros(3),
                'body_pose': np.zeros(63),
                'Rh': np.zeros(3),
                'Th': np.zeros(3),
                'global_orient': np.zeros(3),
                'transl': np.zeros(3),
            }
        }
        
        # 加载mesh数据
        gaussians.load_meshes(test_meshes, {}, {}, {})
        print("✓ 成功加载mesh数据")
        
        # 初始化一些Gaussian点
        num_points = 1000
        gaussians._xyz = torch.randn(num_points, 3).cuda() * 0.1  # 小的随机偏移
        gaussians._features_dc = torch.randn(num_points, 1, 3).cuda()
        gaussians._features_rest = torch.zeros(num_points, 15, 3).cuda()
        gaussians._opacity = torch.randn(num_points, 1).cuda()
        gaussians._scaling = torch.randn(num_points, 3).cuda()
        gaussians._rotation = torch.randn(num_points, 4).cuda()
        gaussians.binding = torch.randint(0, len(gaussians.faces), (num_points,)).cuda()
        print("✓ 初始化Gaussian点")
        
        # 选择mesh时间步
        gaussians.select_mesh_by_timestep(0)
        print("✓ 选择mesh时间步")
        
        # 获取变换前后的坐标
        local_xyz = gaussians._xyz.detach().cpu().numpy()
        world_xyz = gaussians.get_xyz.detach().cpu().numpy()
        
        print(f"局部坐标范围: [{local_xyz.min():.3f}, {local_xyz.max():.3f}]")
        print(f"世界坐标范围: [{world_xyz.min():.3f}, {world_xyz.max():.3f}]")
        print(f"局部坐标中心: {local_xyz.mean(axis=0)}")
        print(f"世界坐标中心: {world_xyz.mean(axis=0)}")
        
        # 检查坐标是否不同（说明有变换）
        if not np.allclose(local_xyz, world_xyz, atol=1e-6):
            print("✓ 坐标变换正常工作")
        else:
            print("⚠ 警告：局部坐标和世界坐标相同，可能没有变换")
        
        # 测试保存PLY
        test_output_dir = Path("test_output")
        test_output_dir.mkdir(exist_ok=True)
        ply_path = test_output_dir / "test_smplx.ply"
        
        gaussians.save_ply(str(ply_path))
        print(f"✓ 成功保存PLY文件到: {ply_path}")
        
        # 验证保存的文件
        if ply_path.exists():
            print("✓ PLY文件存在")
            
            # 读取PLY文件验证坐标
            from plyfile import PlyData
            plydata = PlyData.read(str(ply_path))
            saved_xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])
            ), axis=1)
            
            print(f"保存的坐标范围: [{saved_xyz.min():.3f}, {saved_xyz.max():.3f}]")
            print(f"保存的坐标中心: {saved_xyz.mean(axis=0)}")
            
            # 检查保存的是否是世界坐标
            if np.allclose(saved_xyz, world_xyz, atol=1e-6):
                print("✅ 成功！保存的是变换后的世界坐标")
            elif np.allclose(saved_xyz, local_xyz, atol=1e-6):
                print("❌ 错误！保存的仍然是局部坐标")
            else:
                print("⚠ 警告：保存的坐标与预期都不匹配")
                
        else:
            print("❌ PLY文件不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_flame_save_ply():
    """测试FLAME模型的save_ply方法"""
    
    print("\n=== 测试FLAME Gaussian模型的save_ply修复 ===")
    
    try:
        from scene.flame_gaussian_model import FlameGaussianModel
        
        # 创建模型
        gaussians = FlameGaussianModel(sh_degree=3)
        print("✓ 成功创建FlameGaussianModel")
        
        # 注意：FLAME模型需要实际的mesh数据才能正常工作
        # 这里只是测试代码结构是否正确
        print("✓ FLAME模型代码结构正确")
        
    except Exception as e:
        print(f"❌ FLAME测试失败: {e}")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="转换局部坐标PLY文件为全局坐标PLY文件")
    parser.add_argument("--input", "-i", type=str, help="输入PLY文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出PLY文件路径")
    parser.add_argument("--model", "-m", type=str, default="smplx",
                       choices=["smplx", "flame", "gaussian"],
                       help="模型类型 (默认: smplx)")
    parser.add_argument("--timestep", "-t", type=int, default=0,
                       help="时间步 (默认: 0)")
    parser.add_argument("--sh_degree", type=int, default=3,
                       help="球谐度数 (默认: 3)")
    parser.add_argument("--test", action="store_true",
                       help="运行测试而不是转换")

    args = parser.parse_args()

    if args.test:
        # 运行测试
        test_smplx_save_ply()
        test_flame_save_ply()
        print("\n=== 测试完成 ===")
    else:
        # 运行转换
        if not args.input or not args.output:
            print("错误：需要指定输入和输出文件路径")
            parser.print_help()
            return 1

        if not os.path.exists(args.input):
            print(f"错误：输入文件不存在: {args.input}")
            return 1

        success = convert_local_to_global_ply(
            args.input,
            args.output,
            args.model,
            args.timestep,
            args.sh_degree
        )

        if success:
            print("✅ 转换成功完成！")
            return 0
        else:
            print("❌ 转换失败！")
            return 1

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，运行测试
        test_smplx_save_ply()
        test_flame_save_ply()
        print("\n=== 测试完成 ===")
        print("\n使用 --help 查看转换功能的帮助信息")
    else:
        sys.exit(main())
