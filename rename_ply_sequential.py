#!/usr/bin/env python3
"""
重新命名PLY文件为连续序号
将现有的PLY文件重命名为 frame_000000.ply, frame_000001.ply, frame_000002.ply...
"""

import os
import re
import shutil
from pathlib import Path

def rename_ply_sequential(directory, dry_run=True, backup=True):
    """
    重新命名PLY文件为连续序号
    
    Args:
        directory: 包含PLY文件的目录
        dry_run: 是否只是预览，不实际重命名（默认True）
        backup: 是否创建备份（默认True）
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ 目录不存在: {directory}")
        return
    
    print(f"🔍 扫描目录: {directory}")
    print(f"🔒 预览模式: {'是' if dry_run else '否'}")
    print(f"💾 创建备份: {'是' if backup else '否'}")
    
    # 查找所有frame_*.ply文件
    frame_files = list(directory.glob("frame_*.ply"))
    if not frame_files:
        print("❌ 没有找到frame_*.ply文件")
        return
    
    print(f"📁 找到 {len(frame_files)} 个frame_*.ply文件")
    
    # 解析文件名中的帧号并排序
    frame_data = []
    pattern = re.compile(r'frame_(\d+)\.ply$')
    
    for ply_file in frame_files:
        match = pattern.match(ply_file.name)
        if match:
            frame_num = int(match.group(1))
            frame_data.append((frame_num, ply_file))
        else:
            print(f"⚠ 无法解析帧号: {ply_file.name}")
    
    if not frame_data:
        print("❌ 没有找到可解析的帧文件")
        return
    
    # 按帧号排序
    frame_data.sort(key=lambda x: x[0])
    
    print(f"📊 原始帧号范围: {frame_data[0][0]} - {frame_data[-1][0]}")
    print(f"📊 将重命名为: 000000 - {len(frame_data)-1:06d}")
    
    # 显示重命名计划
    print(f"\n📋 重命名计划（前10个）:")
    for i, (original_frame, file_path) in enumerate(frame_data[:10]):
        new_name = f"frame_{i:06d}.ply"
        print(f"  {file_path.name} -> {new_name}")
    
    if len(frame_data) > 10:
        print(f"  ... 还有 {len(frame_data) - 10} 个文件")
    
    # 创建备份目录
    if backup and not dry_run:
        backup_dir = directory / "backup_original_names"
        backup_dir.mkdir(exist_ok=True)
        print(f"💾 备份目录: {backup_dir}")
    
    # 执行重命名
    if not dry_run:
        print(f"\n🔄 开始重命名...")
        
        # 第一步：重命名为临时名称（避免冲突）
        temp_files = []
        for i, (original_frame, file_path) in enumerate(frame_data):
            temp_name = f"temp_{i:06d}.ply"
            temp_path = directory / temp_name
            
            try:
                # 创建备份
                if backup:
                    backup_path = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                
                # 重命名为临时名称
                file_path.rename(temp_path)
                temp_files.append((i, temp_path))
                
                if (i + 1) % 50 == 0:
                    print(f"   第一步: 已处理 {i + 1}/{len(frame_data)} 个文件")
                    
            except Exception as e:
                print(f"   ❌ 重命名失败 {file_path}: {e}")
                return
        
        # 第二步：重命名为最终名称
        success_count = 0
        for i, temp_path in temp_files:
            final_name = f"frame_{i:06d}.ply"
            final_path = directory / final_name
            
            try:
                temp_path.rename(final_path)
                success_count += 1
                
                if (success_count) % 50 == 0:
                    print(f"   第二步: 已完成 {success_count}/{len(temp_files)} 个文件")
                    
            except Exception as e:
                print(f"   ❌ 最终重命名失败 {temp_path}: {e}")
        
        print(f"✅ 重命名完成: 成功处理 {success_count}/{len(frame_data)} 个文件")
        
        if backup:
            print(f"💾 原始文件已备份到: {backup_dir}")
            
    else:
        print(f"\n🔍 预览模式 - 没有实际重命名文件")
        print(f"💡 要执行实际重命名，请使用 --execute 参数")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="重新命名PLY文件为连续序号")
    parser.add_argument("directory", help="包含PLY文件的目录路径")
    parser.add_argument("--execute", action="store_true", help="执行实际重命名（默认只预览）")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份")
    
    args = parser.parse_args()
    
    rename_ply_sequential(args.directory, dry_run=not args.execute, backup=not args.no_backup)

if __name__ == "__main__":
    main()
