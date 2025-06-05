#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

from plyfile import PlyData

def check_ply(path: Path) -> bool:
    """
    尝试用 plyfile 打开给定路径的 PLY，捕捕获 header 解析错误。
    返回 True 表示文件格式合法；False 则表示解析时出错（例如 stream finished before end header）。
    """
    try:
        # 以 binary 或者 ascii 模式自动探测并解析全文
        PlyData.read(str(path))
        return True
    except Exception as e:
        # 捕获所有异常，包括 plyfile 相关的错误
        # 捕获其他意外异常
        print(f"[ERROR] 无法加载 PLY：{e}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) != 2:
        print("用法: python check_ply_format.py <your_file.ply>", file=sys.stderr)
        sys.exit(1)

    ply_path = Path(sys.argv[1])
    if not ply_path.is_file():
        print(f"[ERROR] 找不到文件: {ply_path}", file=sys.stderr)
        sys.exit(1)

    valid = check_ply(ply_path)
    if valid:
        print(f"[OK] PLY 文件格式检测通过：{ply_path}")
        sys.exit(0)
    else:
        print(f"[FAIL] PLY 文件格式检测失败：{ply_path}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
