# alidata_process/process.py
# ============================
import json
import os
import shutil
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import cv2  # 用于处理Rodrigues向量和图像去畸变
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# --- 配置区域 ---
alidata_path = Path("/home/hello/data")
subject_name = "SC_01"

MASK_BASE_DIR = Path("/home/hello/remote/server2/data1/sapiens_processing/final_head_segments_multigpu/raw_videos/masks")
IMAGE_BASE_DIR = Path("/home/hello/remote/server2/data1/sapiens_processing/final_head_segments_multigpu/raw_videos/masked_images")  # 原始图像目录
UNDISTORTED_IMAGE_DIR = Path("/home/hello/remote/server2/data1/sapiens_processing/alidata_process/undistorted_images") / subject_name
SMPLX_FITTING_DIR = alidata_path / subject_name / "smplx_fitting"
CALIBRATION_FILE = alidata_path / subject_name / "calibration.json"
OUTPUT_DATASET_DIR = Path("/home/hello/data/SC_01/export_all")

NUM_FRAMES = 1280  # 总帧数
# 生成001到053的相机ID列表，排除指定的数字
excluded_numbers = {"002", "006", "012", "021", "024", "025", "030", "033", "043", "044", "051"}
USED_CAMERA_ID_STR_LIST = [f"{i:03d}" for i in range(1, 54) if f"{i:03d}" not in excluded_numbers]
TEST_CAMERA_ID_STR_LIST = ["046"]  # 测试集相机ID
TRAIN_RATIO = 0.7  # 训练集时间步比例
TRAIN_VAL_SUBJECT_SEED = "SC_01"

# 性能优化配置 - 针对16核系统优化
MAX_WORKERS_MULTIPLIER = 6  # I/O密集型任务的线程倍数（16核 × 6 = 96线程）
BATCH_SIZE = 100  # 批处理大小，16核系统可以处理更大批次
PROGRESS_UPDATE_INTERVAL = 50  # 进度更新间隔，减少频繁更新的开销

# 全局变量，用于并行访问（重命名以反映OpenCV坐标系）
all_camera_params_opencv = {}
cam_id_str_to_int_idx = {}

# --- 辅助函数 ---
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json_to_file(data, filepath: Path, indent=4):
    print(f"正在写入 JSON 到: {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_smplx_params(json_path: Path):
    with open(json_path, 'r') as f:
        return json.load(f)


def save_smplx_params(params, json_path: Path):
    ensure_dir(json_path.parent)
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            params[key] = value.tolist()
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)


def create_canonical_smplx_params(first_frame_smplx_path: Path, output_path: Path):
    print(f"正在从 {first_frame_smplx_path} 创建标准SMPL-X参数 (canonical_smplx_param.json)")
    params = load_smplx_params(first_frame_smplx_path)
    canonical_params = {}
    # betas or shape
    if 'betas' in params:
        canonical_params['betas'] = params['betas']
    elif 'shape' in params:
        canonical_params['betas'] = params['shape']
    else:
        print("警告: 未找到 'betas' 或 'shape'，使用零向量。")
        canonical_params['betas'] = np.zeros(100).tolist()
    # 其他参数置零
    param_zeros_configs = {
        'global_orient': 3, 'body_pose': 63, 'jaw_pose': 3, 'leye_pose': 3,
        'reye_pose': 3, 'expression': 50, 'transl': 3,
        'left_hand_pose': 45, 'right_hand_pose': 45,
        'Rh': 3, 'Th': 3
    }
    for key, default_size in param_zeros_configs.items():
        if key in params:
            arr = np.array(params[key])
            # 如果嵌套列表
            if arr.ndim > 1:
                canonical_params[key] = [np.zeros_like(sub).tolist() for sub in arr]
            else:
                canonical_params[key] = np.zeros_like(arr).tolist()
        else:
            canonical_params[key] = np.zeros(default_size).tolist()
            print(f"信息: 参数 '{key}' 未在源文件中找到，已添加零值。")
    canonical_params['gender'] = params.get('gender', 'neutral')
    save_smplx_params(canonical_params, output_path)
    print(f"标准SMPL-X参数已保存到: {output_path}")
    return canonical_params


def process_calibration_data(calib_file: Path, specified_camera_ids: list = None):
    data = json.load(open(calib_file, 'r'))
    cameras = data.get('cameras', {})
    poses = data.get('camera_poses', {})
    
    if specified_camera_ids:
        ids = [cid for cid in specified_camera_ids if cid in cameras and cid in poses]
        missing = set(specified_camera_ids) - set(ids)
        if missing:
            print(f"警告: 相机ID 未找到: {missing}")
    else:
        ids = [cid for cid in cameras if cid in poses]
    
    if not ids:
        raise ValueError("没有可用的相机数据")
    
    out = {}
    for cid in tqdm(ids, desc="处理相机标定"):
        intr = cameras[cid]
        w, h = intr['image_size']
        K = np.array(intr['K'])
        dist = np.array(intr.get('dist', np.zeros(5))).flatten()
        
        fx, fy = K[0,0], K[1,1]
        angle_x = 2 * math.atan(w / (2 * fx))
        angle_y = 2 * math.atan(h / (2 * fy))
        
        p = poses[cid]
        R = np.array(p['R'])
        T = np.array(p['T']).flatten()
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3] = T
        c2w = np.linalg.inv(w2c)   
        c2w_final = c2w

        cx, cy = K[0,2], K[1,2]
        out[cid] = {
            'id': int(cid),
            'width': w,
            'height': h,
            'camera_angle_x': angle_x,
            'camera_angle_y': angle_y,
            'fl_x': fx,
            'fl_y': fy,
            'cx': cx,
            'cy': cy,
            'transform_matrix': c2w_final.tolist(),
            
            # 原始参数（用于去畸变处理）
            'original_K_cv': K.tolist(),
            'original_dist_cv': dist.tolist(),
            
            # 去畸变后的参数（用于后续渲染）
            'undistorted_K_cv': K.tolist(),  # K矩阵保持不变
            'undistorted_dist_cv': [0.0, 0.0, 0.0, 0.0, 0.0],  # 畸变系数置零
            
            # 标记图像已去畸变和主点信息
            'image_undistorted': True,
        }
    
    print(f"✅ 成功处理 {len(out)} 个相机，使用OpenCV坐标系")
    return out


def process_single_frame_cam(frame_idx, cam_id_str):
    global all_camera_params_opencv, cam_id_str_to_int_idx

    try:
        # 快速检查：确保相机参数存在
        if cam_id_str not in all_camera_params_opencv:
            return {"error": f"相机参数不存在: {cam_id_str}"}

        # 快速检查：SMPL-X 参数文件是否存在
        src = SMPLX_FITTING_DIR/f"{frame_idx:06d}.json"
        if not src.exists():
            return {"error": f"SMPLX文件不存在: {src}"}

        # 快速检查：图像和mask文件是否存在
        stem = f"{frame_idx:06d}_{cam_id_str}"
        img_p = IMAGE_BASE_DIR/f"{stem}.png"
        if not img_p.exists():
            img_p = IMAGE_BASE_DIR/f"{stem}.jpg"
        mask_p = MASK_BASE_DIR/f"{stem}.png"

        if not img_p.exists():
            return {"error": f"图像文件不存在: {img_p} 和 {IMAGE_BASE_DIR/f'{stem}.jpg'}"}
        if not mask_p.exists():
            return {"error": f"掩码文件不存在: {mask_p}"}

        # 获取相机参数
        cam = all_camera_params_opencv[cam_id_str]
        idx = cam_id_str_to_int_idx[cam_id_str]

        # 预先计算输出路径
        rel = Path('smplx_param')/src.name
        name = f"{frame_idx:06d}_{idx:02d}.png"
        output_img_path = OUTPUT_DATASET_DIR/'images'/name
        output_mask_path = OUTPUT_DATASET_DIR/'masks_images'/name
        output_smplx_path = OUTPUT_DATASET_DIR/rel

        # 复制SMPL-X参数（只在第一次处理该帧时复制）
        if not output_smplx_path.exists():
            shutil.copyfile(str(src), str(output_smplx_path))

        # 读取图像和mask
        img = cv2.imread(str(img_p))
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            return None

        # 去畸变处理（使用预先转换的numpy数组）
        K_original = np.array(cam['original_K_cv'], dtype=np.float64)
        dist_original = np.array(cam['original_dist_cv'], dtype=np.float64)

        # 并行去畸变处理
        img_undistorted = cv2.undistort(img, K_original, dist_original, None, K_original)
        mask_undistorted = cv2.undistort(mask, K_original, dist_original, None, K_original)

        # 保存处理后的图像和mask
        cv2.imwrite(str(output_img_path), img_undistorted)
        cv2.imwrite(str(output_mask_path), mask_undistorted)

        # 返回元数据（预先构建字典以减少重复计算）
        return {
            "timestep_index": frame_idx,
            "timestep_index_original": frame_idx,
            "camera_index": idx,
            "cx": cam['cx'],
            "cy": cam['cy'],
            "fl_x": cam['fl_x'],
            "fl_y": cam['fl_y'],
            "h": cam['height'],
            "w": cam['width'],
            "camera_angle_x": cam['camera_angle_x'],
            "camera_angle_y": cam['camera_angle_y'],
            "transform_matrix": cam['transform_matrix'],
            "file_path": str(Path('images')/name),
            "fg_mask_path": str(Path('masks_images')/name),
            "smplx_param_path": str(rel),
        }

    except Exception as e:
        # 返回错误信息用于诊断
        return {"error": f"处理异常: {str(e)}"}


def setup_directories():
    """创建所有必要的输出目录。"""
    print("📂 创建输出目录...")
    for p in [OUTPUT_DATASET_DIR, OUTPUT_DATASET_DIR/'images', OUTPUT_DATASET_DIR/'masks_images', OUTPUT_DATASET_DIR/'smplx_param', UNDISTORTED_IMAGE_DIR]:
        ensure_dir(p)
    print("✅ 目录创建完成。")

def load_and_process_calibration():
    """加载并处理相机标定数据。"""
    try:
        camera_params = process_calibration_data(CALIBRATION_FILE, USED_CAMERA_ID_STR_LIST)
        sorted_ids = sorted(camera_params.keys(), key=int)
        cam_id_map = {cid: i for i, cid in enumerate(sorted_ids)}
        return camera_params, cam_id_map, sorted_ids
    except Exception as e:
        print(f"❌ 标定处理失败: {e}")
        return None, None, None

def create_canonical_smplx_parameters():
    """创建标准SMPL-X参数。"""
    first_frame_smplx_path = SMPLX_FITTING_DIR / '000009.json'
    if not first_frame_smplx_path.exists():
        print("❌ 缺少第一帧SMPL-X参数文件: 000009.json")
        return False
    create_canonical_smplx_params(first_frame_smplx_path, OUTPUT_DATASET_DIR / 'canonical_smplx_param.json')
    return True

def process_all_frames_and_cameras(camera_params, cam_id_map, sorted_camera_ids):
    """并行处理所有帧和相机，包括SMPL-X参数复制、图像去畸变和掩码处理。"""
    global all_camera_params_opencv, cam_id_str_to_int_idx
    all_camera_params_opencv = camera_params
    cam_id_str_to_int_idx = cam_id_map

    # 计算跳跃读取的帧数和步长
    total_frames = NUM_FRAMES
    num_frames_to_read = max(1, int(total_frames * 0.2))  # 均分读取其中的20%
    step = max(1, int(total_frames / num_frames_to_read))

    print(f"总帧数: {total_frames}, 将跳跃读取 {num_frames_to_read} 帧，步长为 {step}")

    # 生成所有任务前先检查文件存在情况
    available_frames = [f for f in range(0, total_frames, step) if (SMPLX_FITTING_DIR / f"{f:06d}.json").exists()]
    print(f"可用帧数: {len(available_frames)}/{num_frames_to_read}")

    # 检查图像文件存在情况
    sample_frame = available_frames[0] if available_frames else 0
    sample_cam = sorted_camera_ids[0] if sorted_camera_ids else "001"
    sample_stem = f"{sample_frame:06d}_{sample_cam}"
    sample_img = IMAGE_BASE_DIR / f"{sample_stem}.png"
    sample_mask = MASK_BASE_DIR / f"{sample_stem}.png"

    print(f"📁 检查文件路径:")
    print(f"   SMPLX目录: {SMPLX_FITTING_DIR}")
    print(f"   图像目录: {IMAGE_BASE_DIR}")
    print(f"   掩码目录: {MASK_BASE_DIR}")
    print(f"   样例文件: {sample_img} (存在: {sample_img.exists()})")
    print(f"   样例掩码: {sample_mask} (存在: {sample_mask.exists()})")

    tasks = [(f, c) for f in available_frames for c in sorted_camera_ids]
    print(f"总任务数: {len(tasks)} (帧数 × 相机数)")

    # 优化并行处理：根据实际CPU核心数调整
    actual_cores = os.cpu_count()
    # 如果检测到的核心数异常，使用保守估计
    if actual_cores > 20:  # 可能是超线程导致的
        effective_cores = actual_cores // 2
        print(f"⚠️  检测到 {actual_cores} 个逻辑核心，可能包含超线程，使用 {effective_cores} 个物理核心计算")
    else:
        effective_cores = actual_cores

    max_workers = min(len(tasks), effective_cores * MAX_WORKERS_MULTIPLIER)
    print(f"🚀 使用 {max_workers} 个工作线程进行并行处理")
    print(f"   逻辑核心数: {actual_cores}, 有效核心数: {effective_cores}, 线程倍数: {MAX_WORKERS_MULTIPLIER}")

    frames_all = []
    error_stats = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 批量提交任务以减少开销
        futures = {executor.submit(process_single_frame_cam, f, c): (f, c) for f, c in tasks}

        # 使用更高效的结果收集方式
        completed_count = 0

        with tqdm(total=len(futures), desc="并行导出帧数据", unit="任务") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        if "error" in result:
                            # 收集错误统计
                            error_type = result["error"].split(":")[0]
                            error_stats[error_type] = error_stats.get(error_type, 0) + 1
                        else:
                            frames_all.append(result)
                    completed_count += 1
                    pbar.update(1)

                    # 定期更新进度信息和性能统计
                    if completed_count % PROGRESS_UPDATE_INTERVAL == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        speed = completed_count / elapsed if elapsed > 0 else 0
                        success_rate = len(frames_all) / completed_count * 100 if completed_count > 0 else 0
                        eta = (len(futures) - completed_count) / speed if speed > 0 else 0

                        pbar.set_postfix({
                            "成功": len(frames_all),
                            "成功率": f"{success_rate:.1f}%",
                            "速度": f"{speed:.1f}任务/秒",
                            "预计剩余": f"{eta/60:.1f}分钟"
                        })

                        # 如果成功率太低，显示错误统计
                        if completed_count >= 100 and success_rate < 10:
                            print(f"\n⚠️  成功率过低 ({success_rate:.1f}%)，错误统计:")
                            for error_type, count in error_stats.items():
                                print(f"   {error_type}: {count} 次")

                except Exception as e:
                    print(f"任务执行失败: {e}")
                    pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    avg_speed = len(tasks) / total_time if total_time > 0 else 0

    print(f"✅ 并行处理完成: 成功处理 {len(frames_all)}/{len(tasks)} 个任务")
    print(f"   总耗时: {total_time/60:.2f} 分钟")
    print(f"   平均速度: {avg_speed:.2f} 任务/秒")
    print(f"   成功率: {len(frames_all)/len(tasks)*100:.1f}%")

    # 显示详细错误统计
    if error_stats:
        print(f"\n📊 错误统计:")
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(tasks) * 100
            print(f"   {error_type}: {count} 次 ({percentage:.1f}%)")

    return frames_all

def generate_transforms_metadata(frames_data, camera_params, sorted_camera_ids):
    """生成transforms.json文件的元数据。"""
    meta = {
        'camera_angle_x': 0, 'camera_angle_y': 0, 'fl_x': 0, 'fl_y': 0, 'cx': 0, 'cy': 0, 'w': 0, 'h': 0,
        'frames': [],'applied_transform': np.eye(4).tolist(),
        'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0, 'k3': 0
    }

    if sorted_camera_ids:
        first_cam_id = sorted_camera_ids[0]
        first_cam_data = camera_params[first_cam_id]
        meta.update({
            'camera_angle_x': first_cam_data['camera_angle_x'],
            'camera_angle_y': first_cam_data['camera_angle_y'],
            'fl_x': first_cam_data['fl_x'],
            'fl_y': first_cam_data['fl_y'],
            'cx': first_cam_data['cx'],
            'cy': first_cam_data['cy'],
            'w': first_cam_data['width'],
            'h': first_cam_data['height'],
            'k1': 0.0, 'k2': 0.0, 'k3': 0.0,
            'p1': 0.0, 'p2': 0.0,
            'images_undistorted': True,
            'distortion_model': 'none',
        })
    else:
        print("警告：没有可用的相机参数来初始化meta字典。")
        
    meta['frames'] = frames_data
    if frames_data:
        meta['timestep_indices'] = sorted(list(set(f['timestep_index'] for f in frames_data)))
        meta['camera_indices'] = sorted(list(set(f['camera_index'] for f in frames_data)))
    else:
        meta['timestep_indices'] = []
        meta['camera_indices'] = []
    
    write_json_to_file(meta, OUTPUT_DATASET_DIR / 'transforms.json')
    return meta

def split_dataset_and_save_metadata(frames_data, base_meta, cam_id_map):
    """划分训练/验证/测试集并保存相应的transforms_*.json文件。采用原版逻辑。"""
    print("📊 划分 train/val/test 数据集...")

    # 获取所有时间步并按原版逻辑划分
    all_timesteps = sorted(list(set(f['timestep_index'] for f in frames_data)))
    nt = len(all_timesteps)
    assert 0 < TRAIN_RATIO <= 1
    nt_train = int(np.ceil(nt * TRAIN_RATIO))

    # 时间步划分：前70%用于训练+验证，后30%用于测试
    train_val_timesteps = all_timesteps[:nt_train]
    test_timesteps = all_timesteps[nt_train:]

    # 相机划分
    all_camera_indices = sorted(list(set(f['camera_index'] for f in frames_data)))
    test_camera_int_ids = {cam_id_map[c] for c in TEST_CAMERA_ID_STR_LIST if c in cam_id_map}

    if test_camera_int_ids:
        # 有指定测试相机的情况
        train_camera_indices = [c for c in all_camera_indices if c not in test_camera_int_ids]
        val_camera_indices = list(test_camera_int_ids)  # 验证集使用测试相机
        test_camera_indices = all_camera_indices  # 测试集使用所有相机
    else:
        # 没有指定测试相机，使用最后一个相机作为验证相机
        train_camera_indices = all_camera_indices[:-1] if len(all_camera_indices) > 1 else all_camera_indices
        val_camera_indices = [all_camera_indices[-1]] if len(all_camera_indices) > 1 else []
        test_camera_indices = all_camera_indices

    # 按原版逻辑分配帧数据
    train_set = []
    val_set = []
    test_set = []

    for frame in frames_data:
        timestep = frame['timestep_index']
        camera_idx = frame['camera_index']

        if timestep in train_val_timesteps:
            # 训练+验证时间段
            if camera_idx in train_camera_indices:
                train_set.append(frame)
            elif camera_idx in val_camera_indices:
                val_set.append(frame)
        elif timestep in test_timesteps:
            # 测试时间段，所有相机都进入测试集
            test_set.append(frame)

    def save_split_metadata(name, data_subset, timestep_list, camera_list):
        split_meta = deepcopy(base_meta)
        split_meta['frames'] = data_subset
        split_meta['timestep_indices'] = timestep_list
        split_meta['camera_indices'] = camera_list
        write_json_to_file(split_meta, OUTPUT_DATASET_DIR / f'transforms_{name}.json')

    save_split_metadata('train', train_set, train_val_timesteps, train_camera_indices)
    save_split_metadata('val', val_set, train_val_timesteps, val_camera_indices)  # 验证集共享训练集时间步
    save_split_metadata('test', test_set, test_timesteps, test_camera_indices)

    print(f"✅ 数据集划分完成: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")
    print(f"   时间步划分: 训练+验证={len(train_val_timesteps)}, 测试={len(test_timesteps)}")
    print(f"   相机划分: 训练={train_camera_indices}, 验证={val_camera_indices}, 测试={test_camera_indices}")

def main():
    setup_directories()

    camera_params, cam_id_map, sorted_camera_ids = load_and_process_calibration()
    if camera_params is None:
        return

    if not create_canonical_smplx_parameters():
        return

    frames_data = process_all_frames_and_cameras(camera_params, cam_id_map, sorted_camera_ids)
    
    base_meta = generate_transforms_metadata(frames_data, camera_params, sorted_camera_ids)
    
    split_dataset_and_save_metadata(frames_data, base_meta, cam_id_map)

if __name__ == '__main__':
    main()
