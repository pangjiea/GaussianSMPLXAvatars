#!/usr/bin/env python3
import numpy as np

# 创建简单的跳帧数据（模拟步长5的情况）
num_frames = 10  # 模拟只有10帧数据
motion_params = {}

# 静态参数
motion_params['betas'] = np.random.randn(100).astype(np.float32) * 0.1

# 动态参数 - 创建明显的变化
expression = np.zeros((num_frames, 50), dtype=np.float32)
for i in range(num_frames):
    expression[i, 0] = 0.1 * i  # 线性变化
    expression[i, 1] = 0.05 * (i % 3)  # 循环变化
motion_params['expression'] = expression

# 位移参数
translation = np.zeros((num_frames, 3), dtype=np.float32)
for i in range(num_frames):
    translation[i, 0] = 0.01 * i
    translation[i, 1] = 0.005 * (i % 2)
motion_params['transl'] = translation

# 其他参数设为零
for param_name in ['body_pose', 'global_orient', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'Rh', 'Th']:
    if param_name in ['body_pose']:
        motion_params[param_name] = np.zeros((num_frames, 63), dtype=np.float32)
    elif param_name in ['left_hand_pose', 'right_hand_pose']:
        motion_params[param_name] = np.zeros((num_frames, 45), dtype=np.float32)
    else:
        motion_params[param_name] = np.zeros((num_frames, 3), dtype=np.float32)

# 保存NPZ文件
np.savez('test_skip_motion.npz', **motion_params)
print('✅ 跳帧测试文件已创建: test_skip_motion.npz')
print(f'包含{num_frames}帧数据，模拟步长5的跳帧情况')
