#!/usr/bin/env python3
"""
详细检查NPZ文件的脚本
"""

import numpy as np
import argparse
from pathlib import Path

def detailed_check_npz(model_path):
    """详细检查NPZ文件"""
    model_path = Path(model_path)
    npz_path = model_path / 'smplx_param.npz'
    
    if not npz_path.exists():
        print(f"NPZ文件不存在: {npz_path}")
        return
    
    print(f"检查NPZ文件: {npz_path}")
    data = np.load(str(npz_path))
    
    for key, value in data.items():
        print(f"\n=== {key} ===")
        print(f"Shape: {value.shape}, dtype: {value.dtype}")
        
        if len(value.shape) > 1 and value.shape[0] > 1:
            # 检查前10帧是否都相同
            num_frames_to_check = min(10, value.shape[0])
            all_same = True
            
            for i in range(1, num_frames_to_check):
                if not np.allclose(value[0], value[i], atol=1e-6):
                    all_same = False
                    break
            
            if all_same:
                print(f"  ⚠ 警告: 前{num_frames_to_check}帧都相同！")
                
                # 检查是否所有帧都相同
                if np.allclose(value[0], value[1:], atol=1e-6):
                    print(f"  ❌ 严重问题: 所有{value.shape[0]}帧都相同！")
                else:
                    # 找到第一个不同的帧
                    for i in range(1, value.shape[0]):
                        if not np.allclose(value[0], value[i], atol=1e-6):
                            print(f"  ✓ 第{i}帧开始有差异")
                            break
            else:
                print(f"  ✓ 前{num_frames_to_check}帧包含不同数据")
                
            # 显示一些统计信息
            frame_diffs = []
            for i in range(1, min(5, value.shape[0])):
                diff = np.abs(value[0] - value[i]).mean()
                frame_diffs.append(diff)
                print(f"  帧0-{i}差异: {diff:.6f}")
                
            # 检查中间和末尾的帧
            if value.shape[0] > 10:
                mid_idx = value.shape[0] // 2
                end_idx = value.shape[0] - 1
                mid_diff = np.abs(value[0] - value[mid_idx]).mean()
                end_diff = np.abs(value[0] - value[end_idx]).mean()
                print(f"  帧0-{mid_idx}(中间)差异: {mid_diff:.6f}")
                print(f"  帧0-{end_idx}(末尾)差异: {end_diff:.6f}")
                
        elif key == 'betas':
            print(f"  ✓ {key} 是静态参数")
        else:
            print(f"  单帧参数")

def main():
    parser = argparse.ArgumentParser(description="详细检查NPZ文件")
    parser.add_argument("model_path", help="模型路径")
    args = parser.parse_args()
    
    detailed_check_npz(args.model_path)

if __name__ == "__main__":
    main()
