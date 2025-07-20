"""姿势检测模块
实现图片姿势检测、评分和骨架图生成，所有数据均在内存中处理。"""

import json
import logging
import math
import os
import time
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
import threading
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 导入骨架绘制模块
try:
    from utils.draw import draw_skeleton, draw_coco_skeleton, SkeletonFormat, DrawConfig
except ImportError:
    # 如果导入失败，尝试其他路径
    try:
        from draw import draw_skeleton, draw_coco_skeleton, SkeletonFormat, DrawConfig
    except ImportError:
        logger.error("无法导入骨架绘制模块 (utils.draw 或 draw)，骨架图生成功能将不可用。")
        # 定义占位函数以避免 NameError
        def draw_coco_skeleton(*args, **kwargs):
            raise ImportError("骨架绘制模块未正确安装")

# 模拟的姿势模型推理函数 - 请确保实际导入路径正确
try:
    from pose_model import infer_keypoints
except ImportError:
    logger.warning("Mocking 'infer_keypoints' function as 'pose_model' module was not found.")
    def infer_keypoints(image_bytes: bytes) -> Dict[str, List[float]]:
        # 返回一个模拟的关键点字典，用于测试
        return {
            "nose": [0.5, 0.1], "left_eye": [0.52, 0.08], "right_eye": [0.48, 0.08],
            "left_ear": [0.55, 0.09], "right_ear": [0.45, 0.09],
            "left_shoulder": [0.6, 0.2], "right_shoulder": [0.4, 0.2],
            "left_elbow": [0.65, 0.35], "right_elbow": [0.35, 0.35],
            "left_wrist": [0.7, 0.5], "right_wrist": [0.3, 0.5],
            "left_hip": [0.55, 0.5], "right_hip": [0.45, 0.5],
            "left_knee": [0.58, 0.7], "right_knee": [0.42, 0.7],
            "left_ankle": [0.6, 0.9], "right_ankle": [0.4, 0.9]
        }

# 从 angle_config.py 导入角度配置（若可用）
try:
    from angle_config import angle_config as ANGLE_CONFIG_DATA
except ImportError:
    logger.critical(
        "无法导入 angle_config.py 中的角度配置(angle_config)。将使用空的默认值。"
    )
    ANGLE_CONFIG_DATA = {}

# poses.json 路径，可通过环境变量覆盖
_DEFAULT_POSES_PATH = os.path.join(os.path.dirname(__file__), "poses.json")
POSES_FILE_PATH = os.environ.get("POSES_FILE_PATH", _DEFAULT_POSES_PATH)

# ======================== 模型路径配置 ========================
SCORE_MODEL_PATH = os.getenv("SCORE_MODEL_PATH", "models/score/latest_model.h5")
CLASSIFY_MODEL_PATH = os.getenv("CLASSIFY_MODEL_PATH", "models/classify/latest_model.h5")

# ======================== 全局模型管理 ========================
_model_lock = threading.Lock()
_score_model = None
_classify_model = None

# 全局变量，存储处理后的、保证为字典类型的支持姿势配置
_SUPPORTED_POSES_REGISTRY: Dict[str, Dict[str, Any]] = {}

# ======================== 模型懒加载和热重载 ========================
def _lazy_load():
    """线程安全的延迟加载最新模型权重"""
    global _score_model, _classify_model
    with _model_lock:
        if _score_model is None and os.path.exists(SCORE_MODEL_PATH):
            logger.info(f"🔄 加载评分模型: {SCORE_MODEL_PATH}")
            try:
                _score_model = keras.models.load_model(
                    SCORE_MODEL_PATH, custom_objects={"mse": MeanSquaredError()}
                )
            except Exception as e:
                logger.error(f"加载评分模型失败: {e}")
        
        if _classify_model is None and os.path.exists(CLASSIFY_MODEL_PATH):
            logger.info(f"🔄 加载分类模型: {CLASSIFY_MODEL_PATH}")
            try:
                _classify_model = keras.models.load_model(CLASSIFY_MODEL_PATH)
            except Exception as e:
                logger.error(f"加载分类模型失败: {e}")


def reload_models():
    """供外部热更新调用"""
    global _score_model, _classify_model
    results = {"status": "success", "details": {}}
    
    with _model_lock:
        # 重载评分模型
        if os.path.exists(SCORE_MODEL_PATH):
            try:
                logger.info(f"♻️ 重载评分模型: {SCORE_MODEL_PATH}")
                _score_model = keras.models.load_model(
                    SCORE_MODEL_PATH, custom_objects={"mse": MeanSquaredError()}
                )
                results["details"]["score_model"] = "reloaded"
            except Exception as e:
                logger.error(f"重载评分模型失败: {e}")
                results["details"]["score_model"] = f"error: {str(e)}"
                results["status"] = "partial_failure"
        else:
            results["details"]["score_model"] = "file_not_found"
        
        # 重载分类模型
        if os.path.exists(CLASSIFY_MODEL_PATH):
            try:
                logger.info(f"♻️ 重载分类模型: {CLASSIFY_MODEL_PATH}")
                _classify_model = keras.models.load_model(CLASSIFY_MODEL_PATH)
                results["details"]["classify_model"] = "reloaded"
            except Exception as e:
                logger.error(f"重载分类模型失败: {e}")
                results["details"]["classify_model"] = f"error: {str(e)}"
                results["status"] = "partial_failure"
        else:
            results["details"]["classify_model"] = "file_not_found"
    
    logger.info(f"✅ 模型热更新完成: {results}")
    return results


# ======================== 姿势元数据加载 ========================
def _load_pose_definitions(file_path: str) -> Dict[str, Dict[str, Any]]:
    """从 JSON 文件加载姿势定义并转换为以 pose_id 为键的字典。"""
    if not os.path.exists(file_path):
        logger.error(f"Pose definition file not found: {file_path}")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - log and fallback
        logger.error(f"Failed to load pose definitions: {exc}", exc_info=True)
        return {}

    processed: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                pid = str(item.get("id") or item.get("pose_id") or "").strip()
                if pid:
                    processed[pid] = item
    elif isinstance(data, dict):
        for pid, item in data.items():
            if isinstance(item, dict):
                processed[str(pid)] = item
    else:
        logger.error(f"Invalid pose definition format: {type(data)}")
    return processed


def _initialize_supported_poses():
    """
    初始化 _SUPPORTED_POSES_REGISTRY，确保其为字典格式。
    这个函数应该在模块加载时被调用一次。
    """
    global _SUPPORTED_POSES_REGISTRY
    if _SUPPORTED_POSES_REGISTRY:
        logger.debug("_SUPPORTED_POSES_REGISTRY 已初始化，跳过。")
        return

    data_path = POSES_FILE_PATH if os.path.exists(POSES_FILE_PATH) else _DEFAULT_POSES_PATH
    processed_poses = _load_pose_definitions(data_path)

    _SUPPORTED_POSES_REGISTRY = processed_poses
    if not _SUPPORTED_POSES_REGISTRY:
        logger.warning(
            f"_SUPPORTED_POSES_REGISTRY 初始化后为空。请检查姿势定义文件 '{data_path}' 是否存在且内容有效。"
        )
    else:
        logger.info(
            f"_SUPPORTED_POSES_REGISTRY 初始化完成，加载了 {len(_SUPPORTED_POSES_REGISTRY)} 个姿势的元数据。"
        )

    if not isinstance(ANGLE_CONFIG_DATA, dict):
        logger.error(f"angle_config.angle_config (即 ANGLE_CONFIG_DATA) 不是字典类型，而是 {type(ANGLE_CONFIG_DATA)}。这将影响目标角度的获取。")


# 在模块加载时执行初始化
_initialize_supported_poses()

print("【后端 ANGLE_CONFIG_DATA 支持体式数量】: ", len(ANGLE_CONFIG_DATA))
print("【后端 ANGLE_CONFIG_DATA 支持体式 key】: ", list(ANGLE_CONFIG_DATA.keys()))
print("【后端 _SUPPORTED_POSES_REGISTRY 支持体式数量】: ", len(_SUPPORTED_POSES_REGISTRY))
print("【后端 _SUPPORTED_POSES_REGISTRY 支持体式 key】: ", list(_SUPPORTED_POSES_REGISTRY.keys()))


# ======================== 错误定义 ========================
class ErrorCode(Enum):
    """错误码枚举"""
    SUCCESS = "SUCCESS"
    NO_KEYPOINT = "NO_KEYPOINT"
    INVALID_POSE = "INVALID_POSE"
    INVALID_INPUT = "INVALID_INPUT"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    SKELETON_ERROR = "SKELETON_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class PoseDetectionError(Exception):
    """姿势检测异常基类"""
    def __init__(self, message: str, code: ErrorCode = ErrorCode.UNKNOWN_ERROR, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
    def __str__(self):
        return f"[{self.code.value}] {super().__str__()} Details: {self.details}"


class NoKeypointError(PoseDetectionError):
    """无关键点异常"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.NO_KEYPOINT, details)


class InvalidPoseError(PoseDetectionError):
    """无效姿势ID异常"""
    def __init__(self, message: str, invalid_pose_id: str, available_pose_ids: List[str]):
        details = {
            "invalid_pose_id": invalid_pose_id,
            "supported_pose_ids": available_pose_ids
        }
        super().__init__(message, ErrorCode.INVALID_POSE, details)


# ======================== 数据结构 ========================
@dataclass
class DetectionConfig:
    """检测配置类"""
    angle_tolerance_excellent: float = 10.0
    angle_tolerance_good: float = 20.0
    angle_tolerance_pass: float = 30.0
    angle_tolerance_poor: float = 45.0
    min_detection_rate: float = 0.6
    detection_rate_penalty: float = 0.8
    joint_weights: Dict[str, float] = field(default_factory=lambda: {
        "left_shoulder": 1.0, "right_shoulder": 1.0, "left_elbow": 1.0,
        "right_elbow": 1.0, "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 1.0, "right_knee": 1.0
    })
    # 骨架绘制配置
    skeleton_format: Optional[Any] = None  # 将在运行时设置为 SkeletonFormat.COCO_17
    draw_config: Optional[Any] = None  # 将在运行时设置为 DrawConfig 实例
    enable_multi_person: bool = False
    max_persons: int = 1
    person_selection_strategy: str = "largest"


@dataclass
class PoseDetectionResult:
    """姿势检测结果的标准化结构"""
    score: float
    skeleton_image_bytes: BytesIO
    joint_angles_calculated: Dict[str, float]
    scoring_details: Dict[str, Any]
    timing_stats_ms: Dict[str, float]


# ======================== 核心计算函数 ========================
def calculate_angle(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """计算由三个点p1-p2-p3形成的以p2为顶点的角度 (0-180度)。"""
    if not (isinstance(p1, (list, tuple)) and len(p1) >= 2 and
            isinstance(p2, (list, tuple)) and len(p2) >= 2 and
            isinstance(p3, (list, tuple)) and len(p3) >= 2):
        raise ValueError(f"输入点坐标格式或数量错误。p1: {p1}, p2: {p2}, p3: {p3}")
    try:
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]

        len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if len_v1 < 1e-9 or len_v2 < 1e-9:
            return 0.0

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot_product / (len_v1 * len_v2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    except Exception as e:
        logger.error(f"计算角度时发生错误: p1={p1}, p2={p2}, p3={p3}, 错误: {e}", exc_info=True)
        raise ValueError(f"角度计算失败: {str(e)}")


def calculate_joint_angles(keypoints: Dict[str, List[float]]) -> Dict[str, float]:
    """根据提供的关键点计算预定义的各个关节的角度。"""
    if not isinstance(keypoints, dict):
        logger.error(f"calculate_joint_angles 期望关键点为字典，但收到 {type(keypoints)}")
        return {}

    angles: Dict[str, float] = {}
    joint_definitions = {
        "left_shoulder": ["left_elbow", "left_shoulder", "left_hip"],
        "right_shoulder": ["right_elbow", "right_shoulder", "right_hip"],
        "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_hip": ["left_shoulder", "left_hip", "left_knee"],
        "right_hip": ["right_shoulder", "right_hip", "right_knee"],
        "left_knee": ["left_hip", "left_knee", "left_ankle"],
        "right_knee": ["right_hip", "right_knee", "right_ankle"]
    }

    for joint_name, point_keys in joint_definitions.items():
        p1_key, p2_key, p3_key = point_keys
        p1_coords = keypoints.get(p1_key)
        p2_coords = keypoints.get(p2_key)
        p3_coords = keypoints.get(p3_key)

        if p1_coords and p2_coords and p3_coords:
            try:
                angle = calculate_angle(p1_coords, p2_coords, p3_coords)
                angles[joint_name] = angle
            except ValueError as e:
                logger.warning(f"计算关节 '{joint_name}' 角度失败: {e}")
            except Exception as e_calc:
                 logger.error(f"计算关节 '{joint_name}' 角度时发生意外错误: {e_calc}", exc_info=True)
        else:
            missing = [k for k,v in [(p1_key,p1_coords), (p2_key,p2_coords), (p3_key,p3_coords)] if v is None]
            logger.debug(f"计算关节 '{joint_name}' 角度时缺少关键点: {missing}")
    return angles


def score_pose(
    keypoints: Dict[str, List[float]],
    target_angles_for_pose: Dict[str, float],
    config: DetectionConfig
) -> Tuple[float, Dict[str, Any]]:
    """基于关节角度的姿势评分算法。"""
    if not keypoints:
        logger.warning("评分时关键点数据为空。")
        return 0.0, {"error": "no_keypoints_for_scoring", "message": "未提供关键点数据进行评分。"}
    if not target_angles_for_pose:
        logger.warning("评分时目标角度配置为空。")
        return 0.0, {"error": "no_target_angles_for_scoring", "message": "当前姿势无目标角度配置。"}

    current_joint_angles = calculate_joint_angles(keypoints)
    if not current_joint_angles:
        logger.warning("评分时未能根据关键点计算出任何当前关节角度。")
        return 0.0, {"error": "no_current_angles_calculated", "message": "无法计算当前关节角度。"}

    joint_scores_weighted: Dict[str, float] = {}
    joint_analysis_details: Dict[str, Dict[str, Any]] = {}
    total_score_sum = 0.0
    total_applicable_weight_sum = 0.0

    for joint_name, target_angle_value in target_angles_for_pose.items():
        current_angle_value = current_joint_angles.get(joint_name)
        if current_angle_value is None:
            logger.debug(f"目标关节 '{joint_name}' 在当前姿势中未计算出角度，跳过评分。")
            joint_analysis_details[joint_name] = {"status": "missing_current_angle", "target": target_angle_value}
            continue

        angle_difference = abs(current_angle_value - target_angle_value)
        joint_score_unweighted = 0.0
        grade = "fail"

        if angle_difference <= config.angle_tolerance_excellent:
            joint_score_unweighted = 100.0
            grade = "excellent"
        elif angle_difference <= config.angle_tolerance_good:
            joint_score_unweighted = 100.0 - (angle_difference - config.angle_tolerance_excellent) * \
                                   (20.0 / (config.angle_tolerance_good - config.angle_tolerance_excellent))
            grade = "good"
        elif angle_difference <= config.angle_tolerance_pass:
            joint_score_unweighted = 80.0 - (angle_difference - config.angle_tolerance_good) * \
                                   (20.0 / (config.angle_tolerance_pass - config.angle_tolerance_good))
            grade = "pass"
        elif angle_difference <= config.angle_tolerance_poor:
            joint_score_unweighted = 60.0 - (angle_difference - config.angle_tolerance_pass) * \
                                   (20.0 / (config.angle_tolerance_poor - config.angle_tolerance_pass))
            grade = "poor"
        else:
            joint_score_unweighted = max(0.0, 40.0 - (angle_difference - config.angle_tolerance_poor) * 1.0)
            grade = "fail"

        joint_score_unweighted = max(0.0, min(100.0, joint_score_unweighted))

        joint_weight = config.joint_weights.get(joint_name, 1.0)
        weighted_score_for_joint = joint_score_unweighted * joint_weight

        joint_scores_weighted[joint_name] = weighted_score_for_joint
        total_score_sum += weighted_score_for_joint
        total_applicable_weight_sum += joint_weight

        joint_analysis_details[joint_name] = {
            "status": "scored",
            "current_angle": round(current_angle_value, 2),
            "target_angle": round(target_angle_value, 2),
            "difference": round(angle_difference, 2),
            "score_unweighted": round(joint_score_unweighted, 2),
            "weight": joint_weight,
            "score_weighted": round(weighted_score_for_joint, 2),
            "grade": grade
        }

    average_score = (total_score_sum / total_applicable_weight_sum) if total_applicable_weight_sum > 0 else 0.0

    num_target_joints = len(target_angles_for_pose)
    num_scored_joints = len(joint_scores_weighted)
    detection_rate = (num_scored_joints / num_target_joints) if num_target_joints > 0 else 0.0
    penalty_info = {"applied": False, "factor": 1.0}

    if detection_rate < config.min_detection_rate:
        penalty_factor = detection_rate * config.detection_rate_penalty
        average_score *= penalty_factor
        penalty_info = {"applied": True, "factor": round(penalty_factor, 3), "original_detection_rate": round(detection_rate,3)}
        logger.info(f"应用检测率惩罚: 原始检测率={detection_rate:.2f}, 惩罚因子={penalty_factor:.2f}, 惩罚后平均分暂为={average_score:.2f}")

    final_score = round(max(0.0, min(100.0, average_score)), 2)

    scoring_summary_details = {
        "final_score": final_score,
        "average_score_before_penalty": round((total_score_sum / total_applicable_weight_sum) if total_applicable_weight_sum > 0 else 0.0, 2),
        "detection_rate_info": {
            "scored_joints": num_scored_joints,
            "target_joints": num_target_joints,
            "rate": round(detection_rate, 3),
            "min_threshold": config.min_detection_rate
        },
        "penalty_info": penalty_info,
        "joint_analysis": joint_analysis_details,
        "grades_summary": {
            grade: sum(1 for d in joint_analysis_details.values() if d.get("status") == "scored" and d.get("grade") == grade)
            for grade in ["excellent", "good", "pass", "poor", "fail"]
        }
    }
    logger.info(f"姿势评分完成: 最终分数={final_score}, 检测率={detection_rate:.2f}")
    return final_score, scoring_summary_details


def validate_keypoints(keypoints_data: Any) -> Dict[str, List[float]]:
    """验证和规范化输入关键点数据。"""
    if not isinstance(keypoints_data, dict):
        raise ValueError(f"关键点数据必须是字典类型，实际为: {type(keypoints_data)}")

    validated_keypoints: Dict[str, List[float]] = {}
    for name, coords in keypoints_data.items():
        if not isinstance(name, str):
            logger.warning(f"关键点名称非字符串，已跳过: '{name}' (类型: {type(name)})")
            continue
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            logger.warning(f"关键点 '{name}' 的坐标格式无效，已跳过: {coords}")
            continue
        try:
            x, y = float(coords[0]), float(coords[1])
            validated_keypoints[name] = [x, y]
        except (ValueError, TypeError) as e:
            logger.warning(f"无法解析关键点 '{name}' 的坐标: {coords}，错误: {e}")
            continue
    return validated_keypoints


# ======================== 主要检测函数 ========================
def detect_pose(
    image_bytes: bytes,
    pose_id: str,
    config: Optional[DetectionConfig] = None
) -> Tuple[float, BytesIO]:
    """
    核心姿势检测函数。
    接收图片字节流和姿势ID，返回评分和生成的骨架图（内存中）。
    """
    # 确保模型已加载（懒加载）
    _lazy_load()
    
    overall_start_time = time.monotonic()
    timing_stats: Dict[str, float] = {}

    current_config = config or DetectionConfig()

    # 1. 输入参数验证
    if not image_bytes or not isinstance(image_bytes, bytes):
        logger.error(f"无效的图片输入: image_bytes 为空或类型错误 ({type(image_bytes)})")
        raise PoseDetectionError("图片数据无效或为空。", ErrorCode.INVALID_INPUT, {"input_type": str(type(image_bytes))})

    if not isinstance(pose_id, str) or not pose_id.strip():
        logger.error(f"无效的 pose_id 输入: 类型为 {type(pose_id)}，值为空或仅包含空白。")
        raise InvalidPoseError("pose_id 必须为有效的非空字符串。", str(pose_id), get_supported_pose_ids())

    # 1.1 检查 pose_id 是否受支持
    if pose_id not in _SUPPORTED_POSES_REGISTRY:
        logger.warning(f"请求的 pose_id '{pose_id}' 不在支持的姿势元数据 (_SUPPORTED_POSES_REGISTRY) 中。")
        raise InvalidPoseError(
            f"不支持的动作ID: '{pose_id}'。",
            pose_id,
            get_supported_pose_ids()
        )
    logger.info(f"开始姿势检测流程: pose_id='{pose_id}', 图片大小={len(image_bytes)} bytes")

    try:
        # 2. 从图片中推理关键点
        inference_start_time = time.monotonic()
        raw_keypoints_data = infer_keypoints(image_bytes)
        logger.info(f"【DEBUG: 推理关键点数据】{raw_keypoints_data}")
        timing_stats["keypoint_inference_ms"] = (time.monotonic() - inference_start_time) * 1000
        logger.info(f"关键点推理完成，用时: {timing_stats['keypoint_inference_ms']:.2f}ms")

        if not raw_keypoints_data or not isinstance(raw_keypoints_data, dict):
            logger.warning(f"关键点推理结果为空或格式不正确: {type(raw_keypoints_data)}")
            raise NoKeypointError("未能从图片中检测到任何关键点数据。", {"inference_output_type": str(type(raw_keypoints_data))})

        # 3. 验证和规范化关键点数据
        validation_start_time = time.monotonic()
        try:
            validated_keypoints = validate_keypoints(raw_keypoints_data)
        except ValueError as ve:
            logger.warning(f"关键点数据验证失败: {ve}")
            raise NoKeypointError(f"检测到的关键点数据格式无效: {ve}", {"validation_error": str(ve)})
        timing_stats["keypoint_validation_ms"] = (time.monotonic() - validation_start_time) * 1000

        if not validated_keypoints:
             logger.warning("所有检测到的关键点均未能通过验证。")
             raise NoKeypointError("所有检测到的关键点均未能通过验证。", {"raw_keypoints_count": len(raw_keypoints_data)})
        logger.info(f"已验证 {len(validated_keypoints)} 个关键点。")

        # 4. 获取当前姿势的目标角度配置
        if not isinstance(ANGLE_CONFIG_DATA, dict):
            logger.critical(f"内部配置错误: ANGLE_CONFIG_DATA 不是字典类型 (实际为: {type(ANGLE_CONFIG_DATA)})。无法进行评分。")
            raise PoseDetectionError("服务器内部角度配置错误。", ErrorCode.CONFIG_ERROR)

        target_angles_for_current_pose = ANGLE_CONFIG_DATA.get(pose_id)
        if not target_angles_for_current_pose or not isinstance(target_angles_for_current_pose, dict):
            logger.error(f"未找到动作 '{pose_id}' 的有效目标角度配置。 ANGLE_CONFIG_DATA 中对应的值为: {target_angles_for_current_pose}")
            raise PoseDetectionError(
                f"内部错误：动作 '{pose_id}' 的目标角度配置缺失或无效。",
                ErrorCode.CONFIG_ERROR,
                {"details": f"Target angles for '{pose_id}' in ANGLE_CONFIG_DATA is missing or not a dict."}
            )

        # 5. 计算姿势分数
        scoring_start_time = time.monotonic()
        final_score, _ = score_pose(
            validated_keypoints,
            target_angles_for_current_pose,
            current_config,
        )
        timing_stats["pose_scoring_ms"] = (time.monotonic() - scoring_start_time) * 1000
        logger.info(f"姿势评分完成: 最终分数={final_score}, 用时: {timing_stats['pose_scoring_ms']:.2f}ms")

        # 6. 生成真实的骨架图
        skeleton_draw_start_time = time.monotonic()
        skeleton_image_bytes_io = BytesIO()
        
        try:
            skeleton_format_to_use = current_config.skeleton_format
            if skeleton_format_to_use is None:
                try:
                    skeleton_format_to_use = SkeletonFormat.COCO_17
                except NameError:
                    skeleton_format_to_use = None

            draw_config_to_use = current_config.draw_config
            if draw_config_to_use is None:
                try:
                    draw_config_to_use = DrawConfig(
                        line_color=(0, 255, 0, 255),
                        line_width=3,
                        point_color=(255, 0, 0, 255),
                        point_radius=5,
                        use_original_image=True,
                    )
                except NameError:
                    draw_config_to_use = None
            
            # 优先使用 draw_skeleton 函数（更灵活）
            if skeleton_format_to_use is not None and draw_config_to_use is not None:
                logger.debug("使用 draw_skeleton 函数生成骨架图")
                skeleton_image_bytes_io = draw_skeleton(
                    image_bytes=image_bytes,
                    keypoints=validated_keypoints,
                    config=draw_config_to_use,
                    skeleton_format=skeleton_format_to_use
                )
            else:
                # 备选方案：使用 draw_coco_skeleton 函数
                logger.debug("使用 draw_coco_skeleton 函数生成骨架图")
                skeleton_image_bytes_io = draw_coco_skeleton(validated_keypoints, size=256)
            
            # 验证生成的图片数据
            if skeleton_image_bytes_io.getbuffer().nbytes == 0:
                logger.error("骨架图生成后，缓冲区为空。")
                raise PoseDetectionError("骨架图生成失败：返回了空数据。", ErrorCode.SKELETON_ERROR)
            
            # 重置指针到开头
            skeleton_image_bytes_io.seek(0)
            logger.info(f"骨架图生成成功，大小: {skeleton_image_bytes_io.getbuffer().nbytes} bytes")
            
        except ImportError as e_import:
            logger.error(f"骨架绘制模块导入失败: {e_import}")
            raise PoseDetectionError(
                "骨架图生成失败：绘图模块未正确安装。",
                ErrorCode.SKELETON_ERROR,
                {"import_error": str(e_import)}
            )
        except Exception as e_draw:
            logger.error(f"生成骨架图时发生错误: {e_draw}", exc_info=True)
            raise PoseDetectionError(
                f"骨架图生成时发生内部错误: {str(e_draw)}",
                ErrorCode.SKELETON_ERROR,
                {"error_type": type(e_draw).__name__, "error_detail": str(e_draw)}
            )
        finally:
            timing_stats["skeleton_drawing_ms"] = (time.monotonic() - skeleton_draw_start_time) * 1000
            logger.info(f"骨架图生成/处理完成，用时: {timing_stats['skeleton_drawing_ms']:.2f}ms")

        timing_stats["total_detect_pose_ms"] = (time.monotonic() - overall_start_time) * 1000
        logger.info(f"姿势检测流程成功完成: pose_id='{pose_id}', 最终分数={final_score}, 总用时={timing_stats['total_detect_pose_ms']:.2f}ms")
        logger.debug(f"详细耗时 (ms): {timing_stats}")

        return final_score, skeleton_image_bytes_io

    except (NoKeypointError, InvalidPoseError, PoseDetectionError) as e_custom:
        logger.warning(f"姿势检测中发生已知业务逻辑错误 (pose_id='{pose_id}'): Code={e_custom.code.value}, Msg='{e_custom}'")
        raise
    except Exception as e_unexpected:
        logger.error(f"姿势检测过程中发生未预期系统错误 (pose_id='{pose_id}'): {e_unexpected}", exc_info=True)
        raise PoseDetectionError(
            f"姿势检测时发生内部系统错误: {str(e_unexpected)}",
            ErrorCode.UNKNOWN_ERROR,
            {"original_error_type": type(e_unexpected).__name__}
        ) from e_unexpected


def select_best_person(
    multi_person_keypoints_result: Dict[str, Any],
    strategy: str = "largest_bbox"
) -> Dict[str, List[float]]:
    """
    （占位）从多人关键点检测结果中根据指定策略选择一个"最佳"人体。
    """
    logger.warning(f"select_best_person 功能 (策略: '{strategy}') 尚未完全实现。将尝试返回第一个检测到的人（如果适用）。")
    if isinstance(multi_person_keypoints_result, list) and multi_person_keypoints_result:
        if isinstance(multi_person_keypoints_result[0], dict) and "keypoints" in multi_person_keypoints_result[0]:
             return multi_person_keypoints_result[0]["keypoints"]
    elif isinstance(multi_person_keypoints_result, dict) and "persons" in multi_person_keypoints_result:
        persons = multi_person_keypoints_result["persons"]
        if isinstance(persons, list) and persons and isinstance(persons[0], dict) and "keypoints" in persons[0]:
            return persons[0]["keypoints"]

    logger.error("select_best_person: 无法从输入数据中提取单人关键点。")
    raise NotImplementedError("多人检测中选择最佳人体的逻辑尚未完全实现或输入格式不兼容。")


def get_supported_poses() -> Dict[str, Dict[str, Any]]:
    """
    获取所有支持的姿势及其详细元数据配置。
    返回的是一个字典，键为 pose_id，值为该姿势的配置字典。
    """
    if not isinstance(_SUPPORTED_POSES_REGISTRY, dict):
        logger.critical(f"_SUPPORTED_POSES_REGISTRY 意外地不是字典类型 (实际为: {type(_SUPPORTED_POSES_REGISTRY)})。模块初始化可能存在问题。")
        return {}
    return _SUPPORTED_POSES_REGISTRY.copy()


def get_supported_pose_ids() -> List[str]:
    """获取所有支持的姿势ID的列表。"""
    if not isinstance(_SUPPORTED_POSES_REGISTRY, dict):
        logger.critical(f"_SUPPORTED_POSES_REGISTRY 意外地不是字典类型 (实际为: {type(_SUPPORTED_POSES_REGISTRY)})，无法获取姿势ID列表。")
        return []
    return list(_SUPPORTED_POSES_REGISTRY.keys())


def get_pose_info(pose_id: str) -> Optional[Dict[str, Any]]:
    """
    获取指定姿势ID的聚合信息，包括元数据和目标角度。
    如果姿势ID无效或其任何配置数据缺失/无效，则返回 None。
    """
    if not isinstance(pose_id, str) or not pose_id.strip():
        logger.warning(f"请求姿势信息时，提供的 pose_id 无效 (非字符串或为空): '{pose_id}'")
        return None

    pose_meta_config = _SUPPORTED_POSES_REGISTRY.get(pose_id)
    if pose_meta_config is None:
        logger.warning(f"未在 _SUPPORTED_POSES_REGISTRY 中找到 pose_id '{pose_id}' 的元数据配置。")
        return None
    if not isinstance(pose_meta_config, dict):
        logger.error(f"pose_id '{pose_id}' 在 _SUPPORTED_POSES_REGISTRY 中的元数据配置不是字典类型 (实际为: {type(pose_meta_config)})。")
        return None

    if not isinstance(ANGLE_CONFIG_DATA, dict):
        logger.error(f"全局 ANGLE_CONFIG_DATA 不是字典类型 (实际为: {type(ANGLE_CONFIG_DATA)})。无法获取 '{pose_id}' 的目标角度。")
        target_angles_for_pose = {"error": "Angle configuration data is not a dictionary."}
    else:
        target_angles_for_pose = ANGLE_CONFIG_DATA.get(pose_id)
        if target_angles_for_pose is None:
            logger.warning(f"未在 ANGLE_CONFIG_DATA 中找到 pose_id '{pose_id}' 的目标角度配置。")
            target_angles_for_pose = {"warning": "Target angles not defined for this pose."}
        elif not isinstance(target_angles_for_pose, dict):
            logger.error(f"pose_id '{pose_id}' 在 ANGLE_CONFIG_DATA 中的目标角度配置不是字典类型 (实际为: {type(target_angles_for_pose)})。")
            target_angles_for_pose = {"error": "Target angle configuration for this pose is not a dictionary."}
        
    return {
        "pose_id": pose_id,
        "name": pose_meta_config.get("name", pose_id),
        "description": pose_meta_config.get("description", "无可用描述。"),
        "difficulty": pose_meta_config.get("difficulty", "未知"),
        **{k: v for k, v in pose_meta_config.items() if k not in ["id", "pose_id", "name", "description", "difficulty"]},
        "target_angles": target_angles_for_pose
    }


# ================================================================
#  分类模型推理与统一分析接口（新增功能，保持向下兼容）
# ================================================================

_CLASS_LABELS: List[str] = []


def _load_classify_resources() -> None:
    """内部函数：延迟加载分类模型及标签。"""
    global _classify_model, _CLASS_LABELS

    # 模型加载已由 _lazy_load() 处理
    if _classify_model is None:
        _lazy_load()
        if _classify_model is None:
            return

    if not _CLASS_LABELS:
        labels_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "classify", "class_labels.txt"
        )
        if os.path.exists(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    _CLASS_LABELS = [line.strip() for line in f if line.strip()]
                logger.info(f"已加载 {len(_CLASS_LABELS)} 条姿势标签")
            except Exception as exc:
                logger.error(f"读取分类标签文件失败: {exc}")
        else:
            logger.warning(f"未找到分类标签文件: {labels_path}")


def predict_pose_class(image_bgr: "np.ndarray") -> str:
    """根据输入BGR图片预测姿势ID，置信度低于0.5时返回 'unknown'。"""
    # 确保模型已加载
    _lazy_load()
    _load_classify_resources()

    if _classify_model is None or not _CLASS_LABELS:
        logger.warning("分类模型或标签未就绪，返回 'unknown'")
        return "unknown"

    if image_bgr is None:
        logger.error("predict_pose_class 收到无效的图像输入")
        return "unknown"

    try:
        import cv2  # 延迟导入避免无此依赖时影响其他功能
        import numpy as np

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        input_data = img_resized.astype("float32") / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        predictions = _classify_model.predict(input_data)
        probs = predictions[0] if predictions.ndim > 1 else predictions
        best_index = int(np.argmax(probs))
        confidence = float(probs[best_index])

        if confidence < 0.5 or best_index >= len(_CLASS_LABELS):
            return "unknown"
        return _CLASS_LABELS[best_index]
    except Exception as exc:  # pragma: no cover - 运行时可能出现异常
        logger.error(f"分类模型推理失败: {exc}", exc_info=True)
        return "unknown"


def analyze(image_bgr: "np.ndarray") -> Dict[str, Any]:
    """可选的统一分析接口，组合分类与评分结果。"""
    # 确保模型已加载
    _lazy_load()
    
    result = {
        "pose_id": "unknown",
        "score": 0.0,
        "feedback_type": "unknown",
        "audio_url": "unknown",
    }

    pose_id = predict_pose_class(image_bgr)
    result["pose_id"] = pose_id

    if pose_id != "unknown":
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", image_bgr)
            if ok:
                score, _ = detect_pose(buf.tobytes(), pose_id)
                result["score"] = score
        except Exception as exc:  # pragma: no cover - 捕获推理异常
            logger.error(f"分析评分失败: {exc}")

    return result

def get_pose_keypoints(image: "np.ndarray") -> Optional[List[Dict[str, float]]]:
    """Detect pose keypoints using MediaPipe.

    Args:
        image (np.ndarray): BGR image read by cv2.

    Returns:
        Optional[List[Dict[str, float]]]: first 17 COCO keypoints or None.
    """
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:
        logger.error(f"依赖加载失败: {exc}")
        return None

    if image is None:
        return None

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark[:17]
        return [{"x": float(lm.x), "y": float(lm.y)} for lm in landmarks]
