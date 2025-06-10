#!/usr/bin/env python3
"""
测试NPZ保存修复的脚本
检查保存的NPZ文件是否包含完整的多帧序列
"""

import numpy as np
import argparse
from pathlib import Path

def test_npz_files(model_path):
    """测试NPZ文件是否包含完整的多帧序列"""
    model_path = Path(model_path)
    
    # 检查文件是否存在
    npz_path = model_path / 'smplx_param.npz'
    
    print(f"检查模型路径: {model_path}")
    print(f"SMPLX参数文件: {npz_path} - {'存在' if npz_path.exists() else '不存在'}")
    
    if npz_path.exists():
        print("\n=== SMPLX参数 (smplx_param.npz) ===")
        data = np.load(str(npz_path))
        for key, value in data.items():
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if len(value.shape) > 1 and value.shape[0] > 1:
                # 检查是否所有帧都相同
                if np.allclose(value[0], value[1:], atol=1e-6):
                    print(f"  ⚠ 警告: {key} 的所有帧都相同！")
                else:
                    print(f"  ✓ {key} 包含不同的帧数据")
                    # 显示前几帧的差异
                    if value.shape[0] >= 3:
                        diff_01 = np.abs(value[0] - value[1]).mean()
                        diff_02 = np.abs(value[0] - value[2]).mean()
                        print(f"    帧0-1差异: {diff_01:.6f}, 帧0-2差异: {diff_02:.6f}")
            elif key == 'betas':
                print(f"  ✓ {key} 是静态参数")
    else:
        print("NPZ文件不存在，无法检查")

def main():
    parser = argparse.ArgumentParser(description="测试NPZ保存修复")
    parser.add_argument("model_path", help="模型路径")
    args = parser.parse_args()
    
    test_npz_files(args.model_path)

if __name__ == "__main__":
    main()
