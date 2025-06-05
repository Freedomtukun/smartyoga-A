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

# 获取所有支持的动作ID
SUPPORTED_POSES = list(angle_config.keys())
