# MoveNet 17 keypoint index map (COCO order)
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Mapping from keypoint name to its MoveNet index
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

# 瑜伽动作的标准角度配置 (角度值为度数)

angle_config = {
    "mountain_pose": {
        "left_knee": 180,      # 左膝完全伸直
        "right_knee": 180,     # 右膝完全伸直
        "left_elbow": 180,     # 左肘完全伸直
        "right_elbow": 180,    # 右肘完全伸直
        "left_hip": 180,       # 左髋关节伸直
        "right_hip": 180,      # 右髋关节伸直
        "left_shoulder": 0,    # 双臂自然下垂
        "right_shoulder": 0    # 双臂自然下垂
    },
    
    "warrior_pose": {
        "left_knee": 90,       # 前腿膝盖90度弯曲
        "right_knee": 170,     # 后腿接近伸直
        "left_elbow": 180,     # 手臂伸直
        "right_elbow": 180,    # 手臂伸直
        "left_hip": 130,       # 髋部打开
        "right_hip": 160,      # 髋部打开
        "left_shoulder": 90,   # 手臂平举
        "right_shoulder": 90   # 手臂平举
    },
    
    "tree_pose": {
        "left_knee": 180,      # 支撑腿伸直
        "right_knee": 45,      # 抬起腿弯曲
        "left_elbow": 45,      # 双手合十在胸前
        "right_elbow": 45,     # 双手合十在胸前
        "left_hip": 180,       # 支撑腿髋部伸直
        "right_hip": 90,       # 抬起腿髋部弯曲
        "left_shoulder": 60,   # 手臂上举
        "right_shoulder": 60   # 手臂上举
    },
    
    "downward_dog": {
        "left_knee": 175,      # 腿部接近伸直
        "right_knee": 175,     # 腿部接近伸直
        "left_elbow": 175,     # 手臂接近伸直
        "right_elbow": 175,    # 手臂接近伸直
        "left_hip": 90,        # 髋部成直角
        "right_hip": 90,       # 髋部成直角
        "left_shoulder": 160,  # 肩部打开
        "right_shoulder": 160  # 肩部打开
    },
    
    "plank_pose": {
        "left_knee": 180,      # 腿部完全伸直
        "right_knee": 180,     # 腿部完全伸直
        "left_elbow": 180,     # 手臂完全伸直
        "right_elbow": 180,    # 手臂完全伸直
        "left_hip": 180,       # 身体成一条直线
        "right_hip": 180,      # 身体成一条直线
        "left_shoulder": 90,   # 肩部垂直于地面
        "right_shoulder": 90   # 肩部垂直于地面
    }
}

# Define Chinese names for poses
pose_name_mapping = {
    "mountain_pose": "山式",
    "warrior_pose": "战士式",
    "tree_pose": "树式",
    "downward_dog": "下犬式",
    "plank_pose": "平板支撑"
}

SUPPORTED_POSES = {}
for pose_id, _ in angle_config.items():
    name = pose_name_mapping.get(pose_id, f"{pose_id.replace('_', ' ').title()}式")
    SUPPORTED_POSES[pose_id] = {
        "id": pose_id,
        "name": name,
        "description": f"{name}的简要描述。",
        "difficulty": "初级"
    }

def validate_supported_poses(supported_poses_data):
    errors = []
    if not isinstance(supported_poses_data, dict):
        errors.append("SUPPORTED_POSES is not a dictionary.")
        return errors  # Early exit if the main structure is wrong

    for pose_key, pose_value in supported_poses_data.items():
        if not isinstance(pose_key, str):
            errors.append(f"Pose key '{pose_key}' is not a string.")

        if not isinstance(pose_value, dict):
            errors.append(f"Pose '{pose_key}': Value is not a dictionary.")
            continue  # Skip further checks for this item if value is not a dict

        # Check for 'id'
        if "id" not in pose_value:
            errors.append(f"Pose '{pose_key}': Missing 'id' field.")
        elif pose_value["id"] != pose_key:
            errors.append(f"Pose '{pose_key}': 'id' field ('{pose_value['id']}') does not match the pose key.")

        # Check for 'name'
        if "name" not in pose_value:
            errors.append(f"Pose '{pose_key}': Missing 'name' field.")
        elif not isinstance(pose_value.get("name"), str) or not pose_value.get("name"):
            errors.append(f"Pose '{pose_key}': 'name' field must be a non-empty string.")

        # Check for 'description'
        if "description" not in pose_value:
            errors.append(f"Pose '{pose_key}': Missing 'description' field.")
        elif not isinstance(pose_value.get("description"), str) or not pose_value.get("description"):
            errors.append(f"Pose '{pose_key}': 'description' field must be a non-empty string.")

        # Check for 'difficulty'
        if "difficulty" not in pose_value:
            errors.append(f"Pose '{pose_key}': Missing 'difficulty' field.")
        elif not isinstance(pose_value.get("difficulty"), str) or not pose_value.get("difficulty"):
            errors.append(f"Pose '{pose_key}': 'difficulty' field must be a non-empty string.")

    return errors

# Validate the SUPPORTED_POSES configuration
validation_errors = validate_supported_poses(SUPPORTED_POSES)

if validation_errors:
    print("SUPPORTED_POSES Configuration Errors:")
    for error in validation_errors:
        print(error)
else:
    print("SUPPORTED_POSES configuration is valid.")
