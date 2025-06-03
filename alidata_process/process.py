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
import cv2 # 用于处理Rodrigues向量和图像去畸变

# --- 配置区域 ---
# 输入路径: 这些是您已有的原始数据路径
alidata_path = Path("/home/plm/GaussianAvatars/alidata/")
subject_name = "SC_01"

MASK_BASE_DIR = Path("/mnt/sda/plm/sapiens_processing/final_head_segments_multigpu/raw_videos/masks")
IMAGE_BASE_DIR = Path("/mnt/sda/plm/sapiens_processing/final_head_segments_multigpu/raw_videos/masked_images") # 假设这里已经是去畸变后的图像，如果不是，需要修改
UNDISTORTED_IMAGE_DIR = Path("/mnt/sda/plm/sapiens_processing/alidata_process/undistorted_images") / subject_name # 新增：去畸变图像的输出路径
SMPLX_FITTING_DIR = alidata_path / subject_name / "smplx_fitting"
CALIBRATION_FILE = alidata_path / subject_name / "calibration.json"

# 输出路径: 这是您要"导出"生成的数据集的目标位置
OUTPUT_DATASET_DIR = Path("/mnt/sda/plm/sapiens_processing/alidata_process/export_opengl") / subject_name # 修改输出目录名以区分

# 数据集规格
NUM_FRAMES = 1280 # 总帧数

# --- 新增配置：指定要使用的相机ID列表 ---
USED_CAMERA_ID_STR_LIST = ["046","019"] # <--- 请将这里替换为您实际的相机ID字符串

# 训练/验证/测试集划分配置
TEST_CAMERA_ID_STR_LIST = ["019"]
TEST_FRAMES_RATIO = 0.1
TRAIN_VAL_SUBJECT_SEED = "SC_01" # 用于随机种子，确保可复现

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
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)

def create_canonical_smplx_params(first_frame_smplx_path: Path, output_path: Path):
    print(f"正在从 {first_frame_smplx_path} 创建标准SMPL-X参数 (canonical_smplx_param.json)")
    params = load_smplx_params(first_frame_smplx_path)
    canonical_params = {}
    if 'betas' in params:
        canonical_params['betas'] = params['betas']
    elif 'shape' in params:
         canonical_params['betas'] = params['shape']
    else:
        print("警告: 在SMPL-X参数中未找到 'betas' 或 'shape'。将为标准形状使用零向量。")
        canonical_params['betas'] = np.zeros(100).tolist() # 假设SMPL-X betas是100维
    param_zeros_configs = {
        'global_orient': 3, 'body_pose': 63, 'jaw_pose': 3, 'leye_pose': 3,
        'reye_pose': 3, 'expression': 50, 'transl': 3, # 假设expression是50维
        'left_hand_pose': 45, 'right_hand_pose': 45,
        'Rh': 3, 'Th': 3 # 添加Rh, Th
    }
    for key, default_size in param_zeros_configs.items():
        if key in params and isinstance(params[key], list):
            if params[key] and isinstance(params[key][0], list): # 检查是否为嵌套列表
                canonical_params[key] = [np.zeros_like(np.array(sublist)).tolist() for sublist in params[key]]
            else:
                 canonical_params[key] = np.zeros_like(np.array(params[key])).tolist()
        elif key in params and isinstance(params[key], (int, float)): # 处理单个数值的情况
             canonical_params[key] = 0.0 if isinstance(params[key], float) else 0
        else: # 如果参数不存在或类型不匹配，则创建零向量
            canonical_params[key] = np.zeros(default_size).tolist()
            if key not in params:
                 print(f"信息: 参数 '{key}' 未在源SMPL-X文件 {first_frame_smplx_path.name} 中找到，已为其在标准参数中添加零值。")

    if 'gender' in params:
        canonical_params['gender'] = params['gender']
    else:
        canonical_params['gender'] = "neutral" # 或其他合适的默认值
    save_smplx_params(canonical_params, output_path)
    print(f"标准SMPL-X参数已保存到: {output_path}")
    return canonical_params

def opencv_c2w_to_opengl_c2w(c2w_cv):
    """
    Converts a camera-to-world matrix from OpenCV convention to OpenGL convention.
    OpenCV:  X right, Y down, Z forward
    OpenGL: X right, Y up,   Z backward (camera looks along -Z)

    The conversion effectively means:
    1. The world viewed by the OpenCV camera is (X_cv, Y_cv, Z_cv).
    2. We want to define an OpenGL camera such that it views an OpenGL world (X_gl, Y_gl, Z_gl).
    3. The relationship between these worlds is: X_gl = X_cv, Y_gl = -Y_cv, Z_gl = -Z_cv.
       This is achieved by post-multiplying C2W_cv by a conversion matrix.
    """
    if not isinstance(c2w_cv, np.ndarray):
        c2w_cv = np.array(c2w_cv)
    
    # Conversion matrix to change the coordinate system of the world
    # that the camera is looking at.
    # If P_cv is a point in OpenCV world, P_gl = conversion_world @ P_cv
    # X_gl = X_cv, Y_gl = -Y_cv, Z_gl = -Z_cv
    conversion_world = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    
    # C2W_gl = C2W_cv @ conversion_world
    # This means if a point P_gl is in the new (OpenGL) world,
    # its coordinates in the OpenCV world are P_cv = inv(conversion_world) @ P_gl.
    # The camera's pose C2W_cv transforms P_cv to camera space.
    # So, C2W_gl transforms P_gl to the same camera space, but the camera's
    # local axes need to be flipped to match OpenGL's convention (Y up, Z backward view).
    
    c2w_gl = c2w_cv.copy()
    # Flip Y and Z axes of the camera's orientation and position in the new world system
    c2w_gl[0:3, 1] *= -1  # Flip Y axis of camera orientation
    c2w_gl[0:3, 2] *= -1  # Flip Z axis of camera orientation
    c2w_gl[0:3, 3] *= np.array([1, -1, -1]) # Flip Y and Z of camera position
    
    return c2w_gl

def process_calibration_data(calib_file: Path, specified_camera_ids: list = None):
    """
    Extracts camera parameters from calibration.json, converts to OpenGL C2W,
    and prepares data for transforms.json.
    Outputs OpenGL C2W matrices.
    """
    print(f"正在处理相机标定数据 (转换为OpenGL C2W): {calib_file}")
    
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    cameras_data_in_calib = calib_data.get('cameras', {})
    camera_poses_data = calib_data.get('camera_poses', {}) # These are OpenCV W2C
    
    if specified_camera_ids:
        print(f"使用指定的相机ID列表: {specified_camera_ids}")
        filtered_camera_ids = [cam_id for cam_id in specified_camera_ids if cam_id in cameras_data_in_calib and cam_id in camera_poses_data]
        if len(filtered_camera_ids) != len(specified_camera_ids):
            missing_ids = set(specified_camera_ids) - set(filtered_camera_ids)
            print(f"警告: 以下相机ID在标定数据中未找到或不完整: {missing_ids}")
    else:
        filtered_camera_ids = [cam_id for cam_id in cameras_data_in_calib.keys() if cam_id in camera_poses_data]
        print(f"使用所有可用的相机ID: {filtered_camera_ids}")
    
    if not filtered_camera_ids:
        raise ValueError("没有找到可用的相机数据")
    
    processed_cameras_data = {}
    
    for cam_id_str in tqdm(filtered_camera_ids, desc="处理相机标定"):
        cam_id_int = int(cam_id_str)
        
        camera_intrinsics = cameras_data_in_calib[cam_id_str]
        image_size = camera_intrinsics['image_size']
        width, height = image_size
        K_cv = np.array(camera_intrinsics['K'])
        dist_cv = np.array(camera_intrinsics.get('dist', np.zeros(5))).flatten() # Default to no distortion

        fx_cv = K_cv[0, 0]
        fy_cv = K_cv[1, 1]
        cx_cv = K_cv[0, 2]
        cy_cv = K_cv[1, 2]
        
        # Calculate FoV for OpenGL (Y is typically primary)
        # fov = 2 * arctan( sensor_dim / (2 * focal_length) )
        camera_angle_y_rad = 2 * np.arctan(height / (2 * fy_cv)) # FoV Y in radians
        camera_angle_x_rad = 2 * np.arctan(width / (2 * fx_cv))  # FoV X in radians
        
        pose_data = camera_poses_data[cam_id_str]
        R_w2c_cv = np.array(pose_data['R'])
        T_w2c_cv = np.array(pose_data['T']).flatten()
        
        # OpenCV W2C matrix
        w2c_cv = np.eye(4)
        w2c_cv[:3, :3] = R_w2c_cv
        w2c_cv[:3, 3] = T_w2c_cv
        
        # OpenCV C2W matrix
        c2w_cv = np.linalg.inv(w2c_cv)
        
        # Convert OpenCV C2W to OpenGL C2W
        c2w_gl = opencv_c2w_to_opengl_c2w(c2w_cv)
        
        processed_cameras_data[cam_id_str] = {
            "id": cam_id_int,
            "img_name": f"{cam_id_str}", # Will be updated later with frame index
            "width": width,
            "height": height,
            "camera_angle_x": camera_angle_x_rad, # Store in radians for NeRF convention
            "camera_angle_y": camera_angle_y_rad, # Store in radians
            "fl_x": fx_cv, # Store original focal lengths for reference if needed
            "fl_y": fy_cv,
            "cx": cx_cv,
            "cy": cy_cv,
            "transform_matrix": c2w_gl.tolist(), # This is now OpenGL C2W
            "coordinate_system": "opengl", # Explicitly mark as OpenGL
            "original_K_cv": K_cv.tolist(), # Store for undistortion
            "original_dist_cv": dist_cv.tolist() # Store for undistortion
        }
    
    print(f"成功处理了 {len(processed_cameras_data)} 个相机的标定数据 (已转换为OpenGL C2W)")
    return processed_cameras_data

def main():
    ensure_dir(OUTPUT_DATASET_DIR)
    ensure_dir(OUTPUT_DATASET_DIR / "images") # Will store undistorted images
    ensure_dir(OUTPUT_DATASET_DIR / "masks")
    ensure_dir(OUTPUT_DATASET_DIR / "smplx_params")
    ensure_dir(UNDISTORTED_IMAGE_DIR) # For undistorted images

    print("正在处理相机标定文件...")
    try:
        all_camera_params_opengl = process_calibration_data(CALIBRATION_FILE, USED_CAMERA_ID_STR_LIST)
    except Exception as e:
        print(f"处理标定文件 {CALIBRATION_FILE} 出错: {e}")
        return

    if not all_camera_params_opengl:
        print("未能加载任何相机参数。程序退出。")
        return
    
    sorted_camera_id_strs = sorted(all_camera_params_opengl.keys(), key=lambda x: int(x) if x.isdigit() else x)
    cam_id_str_to_int_idx = {cam_id_str: i for i, cam_id_str in enumerate(sorted_camera_id_strs)}
    all_camera_int_indices = sorted(list(cam_id_str_to_int_idx.values()))

    print(f"将处理 {len(sorted_camera_id_strs)} 个相机: {sorted_camera_id_strs}")
    print(f"相机ID到整数索引的映射: {cam_id_str_to_int_idx}")

    first_frame_smplx_file = SMPLX_FITTING_DIR / "000000.json"
    if not first_frame_smplx_file.exists():
        print(f"错误: 第一帧的SMPL-X参数文件 {first_frame_smplx_file} 未找到。")
        return
    canonical_smplx_output_path = OUTPUT_DATASET_DIR / "canonical_smplx_param.json"
    create_canonical_smplx_params(first_frame_smplx_file, canonical_smplx_output_path)

    # Initialize transforms_data for NeRF (OpenGL convention)
    # The global camera_angle_x, fl_x etc. are often taken from the first camera
    # or averaged, but for simplicity, we'll use the first processed camera.
    transforms_data = {
        "camera_angle_x": 0.0, # This will be filled from the first camera
        # "camera_angle_y": 0.0, # NeRF typically uses camera_angle_x and computes y from aspect
        "fl_x": 0.0,
        "fl_y": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "w": 0,
        "h": 0,
        "frames": [],
        "coordinate_system": "opengl", # Explicitly mark as OpenGL
        # NeRF often uses "applied_transform" for scene normalization,
        # but if your C2W matrices are already in a good world system,
        # this can be identity. For now, let's assume identity.
        "applied_transform": np.eye(4).tolist(),
        # Add k1, k2, p1, p2, k3 and set to 0 as images will be undistorted
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    }

    if sorted_camera_id_strs:
        first_cam_id_for_default = sorted_camera_id_strs[0]
        first_cam_params_for_default = all_camera_params_opengl[first_cam_id_for_default]
        transforms_data["camera_angle_x"] = first_cam_params_for_default["camera_angle_x"]
        # NeRF usually only needs camera_angle_x, and computes fov_y from aspect ratio and fov_x
        # Or, it might use fl_x and fl_y directly if provided.
        # Let's provide all for maximum compatibility, though some renderers might ignore some.
        transforms_data["fl_x"] = first_cam_params_for_default["fl_x"]
        transforms_data["fl_y"] = first_cam_params_for_default["fl_y"]
        transforms_data["cx"] = first_cam_params_for_default["cx"]
        transforms_data["cy"] = first_cam_params_for_default["cy"]
        transforms_data["w"] = first_cam_params_for_default["width"]
        transforms_data["h"] = first_cam_params_for_default["height"]

    all_frame_data_items = []
    timestep_indices_set = set()

    for frame_idx in tqdm(range(NUM_FRAMES), desc="导出帧数据"):
        frame_id_str_padded = f"{frame_idx:06d}"
        smplx_src_path = SMPLX_FITTING_DIR / f"{frame_id_str_padded}.json"
        if not smplx_src_path.exists():
            continue
        
        smplx_dst_filename_in_dataset = f"{frame_id_str_padded}.json"
        smplx_dst_rel_path_in_dataset = Path("smplx_params") / smplx_dst_filename_in_dataset
        shutil.copyfile(smplx_src_path, OUTPUT_DATASET_DIR / smplx_dst_rel_path_in_dataset)
        timestep_indices_set.add(frame_idx)

        for cam_id_str in sorted_camera_id_strs:
            current_cam_calib_params = all_camera_params_opengl[cam_id_str]
            K_cv = np.array(current_cam_calib_params["original_K_cv"])
            dist_cv = np.array(current_cam_calib_params["original_dist_cv"])

            img_filename_original_stem = f"{frame_id_str_padded}_{cam_id_str}"
            # Try .png first, then .jpg for original images
            original_img_src_path = IMAGE_BASE_DIR / f"{img_filename_original_stem}.png"
            if not original_img_src_path.exists():
                original_img_src_path = IMAGE_BASE_DIR / f"{img_filename_original_stem}.jpg"
            
            mask_src_path = MASK_BASE_DIR / f"{img_filename_original_stem}.png" # Assuming masks are always png

            if not original_img_src_path.exists():
                # print(f"警告: 帧 {frame_idx}, 相机 {cam_id_str}: 原始图像 {original_img_src_path} 未找到。")
                continue
            if not mask_src_path.exists():
                # print(f"警告: 帧 {frame_idx}, 相机 {cam_id_str}: 掩膜 {mask_src_path} 未找到。")
                continue
            
            # --- Image Undistortion ---
            img_original_cv = cv2.imread(str(original_img_src_path))
            if img_original_cv is None:
                print(f"警告: 无法读取图像 {original_img_src_path}")
                continue
            
            # Use original K for undistortion, new K will be the same for pinhole model
            # unless getOptimalNewCameraMatrix is used. For simplicity, assume K remains same.
            img_undistorted_cv = cv2.undistort(img_original_cv, K_cv, dist_cv, None, K_cv)
            
            # Also undistort mask if necessary (usually not, but depends on mask generation)
            # For now, assume mask corresponds to undistorted or distortion is negligible for mask
            mask_cv = cv2.imread(str(mask_src_path), cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                print(f"警告: 无法读取掩膜 {mask_src_path}")
                continue
            # mask_undistorted_cv = cv2.undistort(mask_cv, K_cv, dist_cv, None, K_cv) # If mask needs undistortion

            cam_int_idx = cam_id_str_to_int_idx[cam_id_str]
            
            # Save undistorted image and mask to the new dataset structure
            # Filename convention for NeRF datasets is often simpler (e.g., r_0.png, r_1.png per frame)
            # Or frame_XXXX_cam_YY.png. Let's use the latter for clarity.
            new_img_filename_in_dataset = f"{frame_idx:06d}_{cam_int_idx:02d}.png" # Save as png
            new_mask_filename_in_dataset = new_img_filename_in_dataset # Mask uses same name

            img_dst_rel_path_in_dataset = Path("images") / new_img_filename_in_dataset
            mask_dst_rel_path_in_dataset = Path("masks") / new_mask_filename_in_dataset

            cv2.imwrite(str(OUTPUT_DATASET_DIR / img_dst_rel_path_in_dataset), img_undistorted_cv)
            cv2.imwrite(str(OUTPUT_DATASET_DIR / mask_dst_rel_path_in_dataset), mask_cv) # Save original mask for now

            frame_item_metadata = {
                "file_path": str(img_dst_rel_path_in_dataset), # Relative path to image
                "mask_path": str(mask_dst_rel_path_in_dataset), # Relative path to mask
                "smplx_param_path": str(smplx_dst_rel_path_in_dataset),
                "transform_matrix": current_cam_calib_params["transform_matrix"],  # OpenGL C2W
                "timestep_id": frame_idx, 
                "camera_id": cam_int_idx,
                # Optional: include intrinsics per frame if they can vary,
                # otherwise, global intrinsics in transforms_data are used.
                # For NeRF, usually global intrinsics (fov) are used.
                # "fl_x": current_cam_calib_params["fl_x"],
                # "fl_y": current_cam_calib_params["fl_y"],
                # "cx": current_cam_calib_params["cx"],
                # "cy": current_cam_calib_params["cy"],
                # "w": current_cam_calib_params["width"],
                # "h": current_cam_calib_params["height"],
                # "camera_angle_x": current_cam_calib_params["camera_angle_x"],
            }
            all_frame_data_items.append(frame_item_metadata)

    transforms_data["frames"] = all_frame_data_items
    # Add these for compatibility with some loaders, though not strictly NeRF standard
    transforms_data["timestep_indices"] = sorted(list(timestep_indices_set))
    transforms_data["camera_indices"] = all_camera_int_indices

    main_transforms_path = OUTPUT_DATASET_DIR / "transforms.json" # For NeRF, often just one file
    write_json_to_file(transforms_data, main_transforms_path)
    print(f"主 transforms.json 文件已生成 (OpenGL C2W): {main_transforms_path}")

    # --- Dataset Splitting (NeRF Style) ---
    # NeRF typically has transforms_train.json, transforms_val.json, transforms_test.json
    # We will create these by filtering the `frames` list.
    
    print("正在按NeRF风格划分数据集为 train/val/test...")
    
    # Determine test camera integer indices
    test_camera_int_indices_set = set()
    for test_cam_str in TEST_CAMERA_ID_STR_LIST:
        if test_cam_str in cam_id_str_to_int_idx:
            test_camera_int_indices_set.add(cam_id_str_to_int_idx[test_cam_str])
        else:
            print(f"警告: 配置的测试相机ID '{test_cam_str}' 不在使用的相机列表中。")
    
    if not test_camera_int_indices_set and all_camera_int_indices:
        print(f"警告: 未指定有效测试相机，将使用第一个可用相机 (索引 {all_camera_int_indices[0]}) 作为测试相机。")
        test_camera_int_indices_set.add(all_camera_int_indices[0])
    
    print(f"测试集将使用相机整数索引: {list(test_camera_int_indices_set)}")

    # Determine test frames based on ratio (e.g., last TEST_FRAMES_RATIO of timesteps)
    num_total_timesteps_processed = len(transforms_data["timestep_indices"])
    if num_total_timesteps_processed == 0:
        print("错误：没有有效的时间步（帧）被处理，无法进行划分。")
        return

    num_test_timesteps = int(num_total_timesteps_processed * TEST_FRAMES_RATIO)
    if num_test_timesteps == 0 and num_total_timesteps_processed > 0 and TEST_FRAMES_RATIO > 0:
        num_test_timesteps = 1 # Ensure at least one test frame if ratio > 0

    # Option 1: Test frames are specific timesteps for specific cameras
    # Option 2: Test frames are specific timesteps for ALL cameras (common for novel view synthesis)
    # Option 3: Test cameras are held out entirely (common for novel camera synthesis)
    # Let's implement Option 3 for test set (held-out cameras)
    # And Option 1 for validation set (specific frames from non-test cameras)

    train_frames, val_frames, test_frames = [], [], []
    
    # Use a seed for reproducible train/val split of frames
    # We need to split the *timesteps* for validation, not individual frame items yet
    all_processed_timesteps = sorted(list(timestep_indices_set))
    rng = random.Random(TRAIN_VAL_SUBJECT_SEED)
    rng.shuffle(all_processed_timesteps) # Shuffle timesteps

    # For validation, pick some frames from non-test cameras
    # Let's say val is also TEST_FRAMES_RATIO of the *remaining* timesteps
    
    # First, separate frames by camera type (test vs non-test)
    frames_for_test_cameras = []
    frames_for_train_val_cameras = []

    for frame_data in transforms_data["frames"]:
        if frame_data["camera_id"] in test_camera_int_indices_set:
            frames_for_test_cameras.append(frame_data)
        else:
            frames_for_train_val_cameras.append(frame_data)
    
    test_frames.extend(frames_for_test_cameras) # All frames from test cameras go to test set

    # Now, from frames_for_train_val_cameras, split into train and val based on timesteps
    # Get unique timesteps present in frames_for_train_val_cameras
    train_val_timesteps = sorted(list(set(f["timestep_id"] for f in frames_for_train_val_cameras)))
    
    num_val_timesteps_for_split = int(len(train_val_timesteps) * TEST_FRAMES_RATIO) # Using same ratio for val
    if num_val_timesteps_for_split == 0 and len(train_val_timesteps) > 0 and TEST_FRAMES_RATIO > 0:
        num_val_timesteps_for_split = 1

    # Shuffle these timesteps for splitting
    rng_val_split = random.Random(TRAIN_VAL_SUBJECT_SEED + "_val_split") # Different seed for this part
    rng_val_split.shuffle(train_val_timesteps)

    val_timestep_set = set(train_val_timesteps[:num_val_timesteps_for_split])
    train_timestep_set = set(train_val_timesteps[num_val_timesteps_for_split:])
    
    for frame_data in frames_for_train_val_cameras:
        if frame_data["timestep_id"] in val_timestep_set:
            val_frames.append(frame_data)
        elif frame_data["timestep_id"] in train_timestep_set: # Should be all remaining
            train_frames.append(frame_data)
        # else: # Should not happen if sets are complementary
            # print(f"Warning: Frame with timestep {frame_data['timestep_id']} not in val or train set.")


    def create_split_json_dict(base_dict_meta, frame_list_for_split, split_name):
        split_dict = deepcopy(base_dict_meta)
        split_dict["frames"] = frame_list_for_split
        # Update timestep_indices and camera_indices for the split if needed by loader
        current_split_timestep_indices = sorted(list(set(f["timestep_id"] for f in frame_list_for_split))) if frame_list_for_split else []
        current_split_camera_indices = sorted(list(set(f["camera_id"] for f in frame_list_for_split))) if frame_list_for_split else []
        split_dict["timestep_indices"] = current_split_timestep_indices
        split_dict["camera_indices"] = current_split_camera_indices
        if not frame_list_for_split:
            print(f"警告: '{split_name}' 集合为空。")
        return split_dict

    base_meta_dict_for_splits = {k: v for k, v in transforms_data.items() if k not in ["frames", "timestep_indices", "camera_indices"]}
    
    # Update global intrinsics for each split if they were derived from the first camera of the *full* set
    # For simplicity, we'll keep the global ones from the full set.
    # If a split happens to be empty, its specific camera_angle_x etc. might be misleading if derived from an empty frame list.

    db_train = create_split_json_dict(base_meta_dict_for_splits, train_frames, "train")
    db_val = create_split_json_dict(base_meta_dict_for_splits, val_frames, "val")
    db_test = create_split_json_dict(base_meta_dict_for_splits, test_frames, "test")

    write_json_to_file(db_train, OUTPUT_DATASET_DIR / "transforms_train.json")
    write_json_to_file(db_val, OUTPUT_DATASET_DIR / "transforms_val.json")
    write_json_to_file(db_test, OUTPUT_DATASET_DIR / "transforms_test.json")
    print("训练、验证、测试集的 transforms 文件已按NeRF风格生成。")

    print(f"数据集导出完成! 数据集位于: {OUTPUT_DATASET_DIR.resolve()}")
    print(f"  处理的相机总数: {len(all_camera_int_indices)}")
    print(f"  训练集帧数: {len(db_train['frames'])}")
    print(f"  验证集帧数: {len(db_val['frames'])}")
    print(f"  测试集帧数: {len(db_test['frames'])}")

    if not db_train['frames']: print("重要警告: 训练集为空！")
    if not db_val['frames']: print("重要警告: 验证集为空！")
    if not db_test['frames'] and (TEST_FRAMES_RATIO > 0 or TEST_CAMERA_ID_STR_LIST): print("重要警告: 测试集为空！")


if __name__ == "__main__":
    main()