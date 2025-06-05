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
import random
import cv2  # 用于处理Rodrigues向量和图像去畸变
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置区域 ---
alidata_path = Path("/home/hello/data")
subject_name = "SC_01"

MASK_BASE_DIR = Path("/home/hello/remote/server2/data1/sapiens_processing/final_head_segments_multigpu/raw_videos/masks")
IMAGE_BASE_DIR = Path("/home/hello/remote/server2/data1/sapiens_processing/final_head_segments_multigpu/raw_videos/masked_images")  # 原始图像目录
UNDISTORTED_IMAGE_DIR = Path("/home/hello/remote/server2/data1/sapiens_processing/alidata_process/undistorted_images") / subject_name
SMPLX_FITTING_DIR = alidata_path / subject_name / "smplx_fitting"
CALIBRATION_FILE = alidata_path / subject_name / "calibration.json"
OUTPUT_DATASET_DIR = Path("/home/hello/data/SC_01/export")

NUM_FRAMES = 1280  # 总帧数
USED_CAMERA_ID_STR_LIST = ["019","028","001","046","003","004","005","009","010","017","022","031","032","041"]  # 使用的相机ID列表
TEST_CAMERA_ID_STR_LIST = []  # 测试集相机ID
TEST_FRAMES_RATIO = 0.1
TRAIN_VAL_SUBJECT_SEED = "SC_01"

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
    
    # SMPL-X 参数
    src = SMPLX_FITTING_DIR/f"{frame_idx:06d}.json"
    if not src.exists():
        return None
    rel = Path('smplx_param')/src.name
    shutil.copyfile(str(src), str(OUTPUT_DATASET_DIR/rel))
    
    # 确保每个相机使用正确的参数
    if cam_id_str not in all_camera_params_opencv:
        print(f"错误: 相机ID {cam_id_str} 不存在于相机参数中")
        return None
    
    cam = all_camera_params_opencv[cam_id_str]
    
    # 使用原始参数进行去畸变
    K_original = np.array(cam['original_K_cv'])
    dist_original = np.array(cam['original_dist_cv'])
    
    stem = f"{frame_idx:06d}_{cam_id_str}"
    img_p = IMAGE_BASE_DIR/f"{stem}.png"
    if not img_p.exists(): 
        img_p = IMAGE_BASE_DIR/f"{stem}.jpg"
    mask_p = MASK_BASE_DIR/f"{stem}.png"
    
    if not img_p.exists() or not mask_p.exists(): 
        return None
    
    img = cv2.imread(str(img_p))
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None: 
        return None
    
    # 去畸变处理
    img_undistorted = cv2.undistort(img, K_original, dist_original, None, K_original)
    
    # 可选：也对mask进行去畸变（如果mask是在原始图像上生成的）
    mask_undistorted = cv2.undistort(mask, K_original, dist_original, None, K_original)
    
    idx = cam_id_str_to_int_idx[cam_id_str]
    name = f"{frame_idx:06d}_{idx:02d}.png"

    # 保存去畸变后的图像和mask到输出数据集目录
    cv2.imwrite(str(OUTPUT_DATASET_DIR/'images'/name), img_undistorted)
    cv2.imwrite(str(OUTPUT_DATASET_DIR/'masks_images'/name), mask_undistorted)
    
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
    num_frames_to_read = max(1, int(total_frames * 0.01))  # 均分读取其中的20%
    step = max(1, int(total_frames / num_frames_to_read))

    print(f"总帧数: {total_frames}, 将跳跃读取 {num_frames_to_read} 帧，步长为 {step}")

    tasks = [(f, c) for f in range(0, total_frames, step) if (SMPLX_FITTING_DIR / f"{f:06d}.json").exists() for c in sorted_camera_ids]
    frames_all = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_frame_cam, f, c): (f, c) for f, c in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="并行导出帧数据"):
            result = future.result()
            if result:
                frames_all.append(result)
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
    """划分训练/验证/测试集并保存相应的transforms_*.json文件。"""
    print("📊 划分 train/val/test 数据集...")

    test_camera_int_ids = {cam_id_map[c] for c in TEST_CAMERA_ID_STR_LIST if c in cam_id_map}
    test_set = [f for f in frames_data if f['camera_index'] in test_camera_int_ids]
    
    train_val_set = [f for f in frames_data if f not in test_set]
    
    train_val_timesteps = sorted({f['timestep_index'] for f in train_val_set})
    num_val_frames = int(len(train_val_timesteps) * TEST_FRAMES_RATIO) or 1
    
    rng = random.Random(TRAIN_VAL_SUBJECT_SEED)
    rng.shuffle(train_val_timesteps)
    
    val_timesteps = set(train_val_timesteps[:num_val_frames])
    train_timesteps = set(train_val_timesteps[num_val_frames:])
    
    train_set = [f for f in train_val_set if f['timestep_index'] in train_timesteps]
    val_set = [f for f in train_val_set if f['timestep_index'] in val_timesteps]

    def save_split_metadata(name, data_subset):
        split_meta = deepcopy(base_meta)
        split_meta['frames'] = data_subset
        split_meta['timestep_indices'] = sorted(list(set(f['timestep_index'] for f in data_subset)))
        split_meta['camera_indices'] = sorted(list(set(f['camera_index'] for f in data_subset)))
        write_json_to_file(split_meta, OUTPUT_DATASET_DIR / f'transforms_{name}.json')

    save_split_metadata('train', train_set)
    save_split_metadata('val', val_set)
    save_split_metadata('test', test_set)
    
    print(f"✅ 数据集划分完成: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")

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
