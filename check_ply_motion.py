#!/usr/bin/env python3
"""
检测PLY文件序列中相邻帧是否完全一样
用于验证运动序列是否正确生成
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path
from plyfile import PlyData
from tqdm import tqdm

def load_ply_vertices(ply_path):
    """加载PLY文件的顶点坐标"""
    try:
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        
        # 提取xyz坐标
        x = np.array(vertices['x'])
        y = np.array(vertices['y'])
        z = np.array(vertices['z'])
        xyz = np.column_stack((x, y, z))
        
        return xyz
    except Exception as e:
        print(f"❌ 无法加载PLY文件 {ply_path}: {e}")
        return None

def load_ply_all_attributes(ply_path):
    """加载PLY文件的所有属性"""
    try:
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        
        # 提取所有数值属性
        attributes = {}
        for prop in vertices.dtype.names:
            attributes[prop] = np.array(vertices[prop])
        
        return attributes
    except Exception as e:
        print(f"❌ 无法加载PLY文件 {ply_path}: {e}")
        return None

def compare_ply_files(ply1_path, ply2_path, check_all_attributes=False, tolerance=1e-6):
    """比较两个PLY文件是否相同"""
    if check_all_attributes:
        attrs1 = load_ply_all_attributes(ply1_path)
        attrs2 = load_ply_all_attributes(ply2_path)
        
        if attrs1 is None or attrs2 is None:
            return False, "加载失败"
        
        if set(attrs1.keys()) != set(attrs2.keys()):
            return False, f"属性不匹配: {set(attrs1.keys())} vs {set(attrs2.keys())}"
        
        differences = {}
        for attr in attrs1.keys():
            if not np.allclose(attrs1[attr], attrs2[attr], atol=tolerance):
                diff = np.abs(attrs1[attr] - attrs2[attr]).max()
                differences[attr] = diff
        
        if differences:
            return False, f"属性差异: {differences}"
        else:
            return True, "完全相同"
    else:
        xyz1 = load_ply_vertices(ply1_path)
        xyz2 = load_ply_vertices(ply2_path)
        
        if xyz1 is None or xyz2 is None:
            return False, "加载失败"
        
        if xyz1.shape != xyz2.shape:
            return False, f"顶点数量不匹配: {xyz1.shape[0]} vs {xyz2.shape[0]}"
        
        if np.allclose(xyz1, xyz2, atol=tolerance):
            return True, "坐标完全相同"
        else:
            max_diff = np.abs(xyz1 - xyz2).max()
            mean_diff = np.abs(xyz1 - xyz2).mean()
            center_diff = np.linalg.norm(xyz1.mean(axis=0) - xyz2.mean(axis=0))
            return False, f"坐标不同 - 最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}, 中心偏移: {center_diff:.6f}"

def check_ply_sequence_motion(ply_dir, check_all_attributes=False, tolerance=1e-6, max_files=None):
    """检查PLY序列中的运动"""
    print(f"=== 检查PLY序列运动 ===")
    print(f"目录: {ply_dir}")
    print(f"检查所有属性: {check_all_attributes}")
    print(f"容差: {tolerance}")
    
    # 查找所有PLY文件
    ply_files = sorted(list(Path(ply_dir).glob("*.ply")))
    
    if len(ply_files) == 0:
        print("❌ 未找到PLY文件")
        return False
    
    if len(ply_files) < 2:
        print("❌ PLY文件数量少于2个，无法比较")
        return False
    
    if max_files and len(ply_files) > max_files:
        ply_files = ply_files[:max_files]
        print(f"⚠ 限制检查前{max_files}个文件")
    
    print(f"✓ 找到{len(ply_files)}个PLY文件")
    
    # 统计结果
    identical_pairs = 0
    different_pairs = 0
    failed_pairs = 0
    
    # 检查相邻文件
    print("\n--- 检查相邻文件差异 ---")
    for i in tqdm(range(len(ply_files) - 1), desc="比较文件"):
        file1 = ply_files[i]
        file2 = ply_files[i + 1]
        
        is_same, message = compare_ply_files(file1, file2, check_all_attributes, tolerance)
        
        if "加载失败" in message:
            failed_pairs += 1
            print(f"❌ {file1.name} vs {file2.name}: {message}")
        elif is_same:
            identical_pairs += 1
            if identical_pairs <= 5:  # 只显示前5个相同的
                print(f"⚠ {file1.name} vs {file2.name}: {message}")
        else:
            different_pairs += 1
            if different_pairs <= 5:  # 只显示前5个不同的
                print(f"✓ {file1.name} vs {file2.name}: {message}")
    
    # 输出统计结果
    print(f"\n--- 统计结果 ---")
    print(f"总比较对数: {len(ply_files) - 1}")
    print(f"完全相同: {identical_pairs} ({identical_pairs/(len(ply_files)-1)*100:.1f}%)")
    print(f"有差异: {different_pairs} ({different_pairs/(len(ply_files)-1)*100:.1f}%)")
    print(f"加载失败: {failed_pairs}")
    
    # 判断结果
    if identical_pairs > different_pairs:
        print(f"\n❌ 问题：大部分相邻帧完全相同，模型可能没有运动")
        return False
    elif different_pairs > 0:
        print(f"\n✅ 正常：检测到帧间差异，模型有运动")
        return True
    else:
        print(f"\n⚠ 无法确定：所有比较都失败了")
        return False

def check_specific_frames(ply_dir, frame_indices, check_all_attributes=False, tolerance=1e-6):
    """检查特定帧的差异"""
    print(f"\n=== 检查特定帧 ===")
    
    ply_files = sorted(list(Path(ply_dir).glob("*.ply")))
    
    if len(ply_files) == 0:
        print("❌ 未找到PLY文件")
        return
    
    print(f"可用帧数: {len(ply_files)}")
    
    for i in range(len(frame_indices) - 1):
        idx1 = frame_indices[i]
        idx2 = frame_indices[i + 1]
        
        if idx1 >= len(ply_files) or idx2 >= len(ply_files):
            print(f"⚠ 跳过无效索引: {idx1}, {idx2}")
            continue
        
        file1 = ply_files[idx1]
        file2 = ply_files[idx2]
        
        is_same, message = compare_ply_files(file1, file2, check_all_attributes, tolerance)
        
        print(f"帧 {idx1} vs 帧 {idx2}: {message}")

def main():
    parser = argparse.ArgumentParser(
        description="检测PLY文件序列中相邻帧是否完全一样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查目录中所有PLY文件的坐标差异
  python check_ply_motion.py --dir output_plys/
  
  # 检查所有属性（包括颜色、透明度等）
  python check_ply_motion.py --dir output_plys/ --check_all
  
  # 设置更严格的容差
  python check_ply_motion.py --dir output_plys/ --tolerance 1e-8
  
  # 只检查前10个文件
  python check_ply_motion.py --dir output_plys/ --max_files 10
  
  # 检查特定帧
  python check_ply_motion.py --dir output_plys/ --frames 0 10 20 30
        """
    )
    
    parser.add_argument("--dir", "-d", required=True, help="PLY文件目录")
    parser.add_argument("--check_all", action="store_true", help="检查所有属性而不仅仅是坐标")
    parser.add_argument("--tolerance", "-t", type=float, default=1e-6, help="数值比较容差 (默认: 1e-6)")
    parser.add_argument("--max_files", type=int, help="最大检查文件数量")
    parser.add_argument("--frames", nargs="+", type=int, help="检查特定帧索引")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"❌ 目录不存在: {args.dir}")
        return 1
    
    if args.frames:
        check_specific_frames(args.dir, args.frames, args.check_all, args.tolerance)
    else:
        success = check_ply_sequence_motion(args.dir, args.check_all, args.tolerance, args.max_files)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
