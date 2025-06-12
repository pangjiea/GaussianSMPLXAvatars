#!/usr/bin/env python3
"""
清理PLY文件，只保留步长为5的帧，删除中间帧
模拟multiprocess步长5的情况
"""

import os
import re
from pathlib import Path

def cleanup_ply_files(directory, step_size=5, dry_run=True):
    """
    清理PLY文件，只保留步长为step_size的帧
    
    Args:
        directory: 包含PLY文件的目录
        step_size: 步长（默认5）
        dry_run: 是否只是预览，不实际删除（默认True）
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ 目录不存在: {directory}")
        return
    
    print(f"🔍 扫描目录: {directory}")
    print(f"📏 步长: {step_size}")
    print(f"🔒 预览模式: {'是' if dry_run else '否'}")
    
    # 查找所有PLY文件
    ply_files = list(directory.glob("*.ply"))
    if not ply_files:
        print("❌ 没有找到PLY文件")
        return
    
    print(f"📁 找到 {len(ply_files)} 个PLY文件")
    
    # 解析文件名中的帧号
    frame_files = {}
    pattern = re.compile(r'.*?(\d+)\.ply$')
    
    for ply_file in ply_files:
        match = pattern.match(ply_file.name)
        if match:
            frame_num = int(match.group(1))
            frame_files[frame_num] = ply_file
        else:
            print(f"⚠ 无法解析帧号: {ply_file.name}")
    
    if not frame_files:
        print("❌ 没有找到可解析的帧文件")
        return
    
    # 排序帧号
    sorted_frames = sorted(frame_files.keys())
    print(f"📊 帧号范围: {sorted_frames[0]} - {sorted_frames[-1]}")
    print(f"📊 总帧数: {len(sorted_frames)}")
    
    # 确定要保留和删除的帧
    keep_frames = set()
    delete_frames = set()
    
    for frame_num in sorted_frames:
        if frame_num % step_size == 0:
            keep_frames.add(frame_num)
        else:
            delete_frames.add(frame_num)
    
    print(f"\n📋 处理计划:")
    print(f"✅ 保留帧数: {len(keep_frames)}")
    print(f"❌ 删除帧数: {len(delete_frames)}")
    
    # 显示前几个要保留和删除的帧
    keep_list = sorted(list(keep_frames))[:10]
    delete_list = sorted(list(delete_frames))[:20]
    
    print(f"✅ 保留的帧（前10个）: {keep_list}")
    if len(keep_frames) > 10:
        print(f"   ... 还有 {len(keep_frames) - 10} 个")
    
    print(f"❌ 删除的帧（前20个）: {delete_list}")
    if len(delete_frames) > 20:
        print(f"   ... 还有 {len(delete_frames) - 20} 个")
    
    # 执行删除操作
    if not dry_run:
        print(f"\n🗑️ 开始删除文件...")
        deleted_count = 0
        failed_count = 0
        
        for frame_num in delete_frames:
            try:
                file_path = frame_files[frame_num]
                file_path.unlink()
                deleted_count += 1
                if deleted_count % 50 == 0:
                    print(f"   已删除 {deleted_count}/{len(delete_frames)} 个文件")
            except Exception as e:
                print(f"   ❌ 删除失败 {frame_files[frame_num]}: {e}")
                failed_count += 1
        
        print(f"✅ 删除完成: 成功删除 {deleted_count} 个文件")
        if failed_count > 0:
            print(f"❌ 删除失败: {failed_count} 个文件")
    else:
        print(f"\n🔍 预览模式 - 没有实际删除文件")
        print(f"💡 要执行实际删除，请使用 --execute 参数")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="清理PLY文件，只保留步长为5的帧")
    parser.add_argument("directory", help="包含PLY文件的目录路径")
    parser.add_argument("--step", type=int, default=5, help="步长（默认5）")
    parser.add_argument("--execute", action="store_true", help="执行实际删除（默认只预览）")
    
    args = parser.parse_args()
    
    cleanup_ply_files(args.directory, args.step, dry_run=not args.execute)

if __name__ == "__main__":
    main()
