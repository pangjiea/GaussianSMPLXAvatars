#!/usr/bin/env python3
"""
比较脚本：比较使用原始mesh参数和训练后mesh参数的差异

这个脚本可以帮助你理解为什么转换出的全局坐标和训练投影有差异。
"""

import torch
import numpy as np
import sys
import os
import argparse

# 添加项目路径
sys.path.append('.')

def compare_mesh_parameters(ply_path, timestep=0):
    """
    比较原始mesh参数和训练后mesh参数的差异
    """
    print(f"=== 比较mesh参数差异：{ply_path} ===")
    
    try:
        # 检测模型类型
        if "smplx" in ply_path.lower():
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            gaussians = SMPLXGaussianModel(sh_degree=3)
            model_type = "SMPLX"
        elif "flame" in ply_path.lower():
            from scene.flame_gaussian_model import FlameGaussianModel
            gaussians = FlameGaussianModel(sh_degree=3)
            model_type = "FLAME"
        else:
            print("❌ 无法确定模型类型，请确保文件名包含'smplx'或'flame'")
            return False
        
        print(f"✓ 创建{model_type}模型")
        
        # 加载PLY文件
        gaussians.load_ply(ply_path)
        print(f"✓ 加载PLY文件，包含{gaussians._xyz.shape[0]}个高斯点")
        
        if gaussians.binding is None:
            print("⚠ 警告：PLY文件没有绑定信息")
            return False
        
        # 检查是否有mesh参数
        if model_type == "SMPLX":
            if gaussians.smplx_param is None:
                print("❌ 没有找到SMPLX参数文件")
                return False
            
            print("✓ 找到SMPLX参数文件")
            
            # 比较原始参数和当前参数
            if gaussians.smplx_param_orig is not None:
                print("\n=== 参数差异分析 ===")
                
                for key in ['global_orient', 'body_pose', 'jaw_pose', 'expression', 'transl']:
                    if key in gaussians.smplx_param and key in gaussians.smplx_param_orig:
                        current = gaussians.smplx_param[key]
                        original = gaussians.smplx_param_orig[key]
                        
                        if current.shape != original.shape:
                            print(f"{key}: 形状不匹配 {current.shape} vs {original.shape}")
                            continue
                        
                        diff = torch.abs(current - original).mean().item()
                        max_diff = torch.abs(current - original).max().item()
                        
                        print(f"{key}: 平均差异={diff:.6f}, 最大差异={max_diff:.6f}")
                        
                        if diff > 1e-6:
                            print(f"  ⚠ {key} 参数已被训练优化")
                        else:
                            print(f"  ✓ {key} 参数未改变")
            else:
                print("⚠ 没有找到原始参数备份")
            
            # 比较使用不同参数时的mesh差异
            print("\n=== Mesh形状差异分析 ===")
            
            # 使用训练后参数
            gaussians.select_mesh_by_timestep(timestep, original=False)
            trained_center = gaussians.face_center.clone()
            trained_scaling = gaussians.face_scaling.clone()
            trained_xyz = gaussians.get_xyz.detach().cpu().numpy()
            
            # 使用原始参数
            if gaussians.smplx_param_orig is not None:
                gaussians.select_mesh_by_timestep(timestep, original=True)
                original_center = gaussians.face_center.clone()
                original_scaling = gaussians.face_scaling.clone()
                original_xyz = gaussians.get_xyz.detach().cpu().numpy()
                
                # 计算差异
                center_diff = torch.abs(trained_center - original_center).mean().item()
                scaling_diff = torch.abs(trained_scaling - original_scaling).mean().item()
                xyz_diff = np.abs(trained_xyz - original_xyz).mean()
                
                print(f"面心位置差异: {center_diff:.6f}")
                print(f"面缩放差异: {scaling_diff:.6f}")
                print(f"高斯点全局坐标差异: {xyz_diff:.6f}")
                
                print(f"\n训练后坐标范围: [{trained_xyz.min():.3f}, {trained_xyz.max():.3f}]")
                print(f"原始参数坐标范围: [{original_xyz.min():.3f}, {original_xyz.max():.3f}]")
                print(f"训练后坐标中心: {trained_xyz.mean(axis=0)}")
                print(f"原始参数坐标中心: {original_xyz.mean(axis=0)}")
                
                if xyz_diff > 1e-3:
                    print("\n🔍 发现显著差异！")
                    print("这解释了为什么转换脚本的结果与训练投影不同：")
                    print("- 训练过程优化了mesh参数")
                    print("- 转换脚本默认使用训练后的参数")
                    print("- 如果想要原始形状，请使用 --use_original_mesh 参数")
                else:
                    print("\n✓ 参数差异很小，可能训练没有显著改变mesh形状")
            else:
                print("⚠ 无法比较，因为没有原始参数备份")
        
        elif model_type == "FLAME":
            # 类似的FLAME参数比较逻辑
            if gaussians.flame_param is None:
                print("❌ 没有找到FLAME参数文件")
                return False
            
            print("✓ 找到FLAME参数文件")
            print("📝 FLAME参数比较功能待实现")
        
        return True
        
    except Exception as e:
        print(f"❌ 比较失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较原始mesh参数和训练后mesh参数的差异")
    parser.add_argument("--input", "-i", required=True, help="输入PLY文件路径")
    parser.add_argument("--timestep", "-t", type=int, default=0, help="时间步 (默认: 0)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        return 1
    
    # 执行比较
    success = compare_mesh_parameters(args.input, args.timestep)
    
    if success:
        print("\n✅ 比较完成！")
        return 0
    else:
        print("\n❌ 比较失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
