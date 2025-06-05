"""
姿势检测模块
实现图片姿势检测、评分和骨架图生成
不生成任何本地文件，所有数据在内存中处理
"""

import math
import time
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO

# 模拟的姿势模型推理函数 - 请确保实际导入路径正确
try:
    from pose_model import infer_keypoints
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger is available
    logger.warning("Mocking 'infer_keypoints' function as 'pose_model' module was not found.")
    def infer_keypoints(image_bytes: bytes) -> Dict[str, List[float]]:
        # 返回一个模拟的关键点字典，用于测试
        # In a real scenario, this would come from your ML model
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

# 从 angle_config.py 导入原始姿势数据
# 我们假设 angle_config.py 中的 SUPPORTED_POSES_DATA 和 angle_config_data
# 是最原始的姿势定义来源。
try:
    # 假设 SUPPORTED_POSES_DATA 是姿势元数据 (e.g., name, description, difficulty)
    # 并且 angle_config 是目标角度配置 (e.g., {"squat": {"left_knee": 90, ...}})
    from angle_config import (
        SUPPORTED_POSES as SUPPORTED_POSES_DATA, # 重命名导入以示区分原始数据
        angle_config as ANGLE_CONFIG_DATA
    )
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger is available for this block
    logger.critical("无法导入 angle_config.py 中的姿势数据。请检查该文件是否存在且路径正确。将使用空的默认值。")
    SUPPORTED_POSES_DATA = {} # 默认为空字典，如果原始是列表则用 []
    ANGLE_CONFIG_DATA = {}    # 默认为空字典

logger = logging.getLogger(__name__)

# 全局变量，存储处理后的、保证为字典类型的支持姿势配置
# 类型: Dict[str, Dict[str, Any]]  (例如: {"squat": {"name": "深蹲", "description": "...", ...}})
_SUPPORTED_POSES_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _initialize_supported_poses():
    """
    初始化 _SUPPORTED_POSES_REGISTRY，确保其为字典格式。
    这个函数应该在模块加载时被调用一次。
    """
    global _SUPPORTED_POSES_REGISTRY
    if _SUPPORTED_POSES_REGISTRY: # 防止重复初始化
        logger.debug("_SUPPORTED_POSES_REGISTRY 已初始化，跳过。")
        return

    processed_poses: Dict[str, Dict[str, Any]] = {}
    source_data_for_registry = SUPPORTED_POSES_DATA # 这是姿势的元数据

    if isinstance(source_data_for_registry, dict):
        # 如果源数据已经是期望的字典格式 {pose_id: config_dict}
        for pose_id, pose_config in source_data_for_registry.items():
            if not isinstance(pose_id, str):
                logger.warning(f"在 SUPPORTED_POSES_DATA 中发现非字符串类型的 pose_id: {pose_id}，已跳过。")
                continue
            if not isinstance(pose_config, dict):
                logger.warning(f"姿势 '{pose_id}' 在 SUPPORTED_POSES_DATA 中的配置不是字典类型，已跳过。配置: {pose_config}")
                continue
            processed_poses[pose_id] = pose_config
    elif isinstance(source_data_for_registry, list):
        # 如果源数据是列表格式, e.g., [{"id": "squat", "name": "深蹲", ...}, ...]
        for item in source_data_for_registry:
            if not isinstance(item, dict):
                logger.warning(f"SUPPORTED_POSES_DATA 列表中的项目不是字典类型，已跳过。项目: {item}")
                continue
            # 尝试从常见的键名获取 pose_id
            pose_id = item.get("id") or item.get("pose_id") or item.get("key")
            if not pose_id or not isinstance(pose_id, str):
                logger.warning(f"SUPPORTED_POSES_DATA 项目缺少有效的 'id'/'pose_id'/'key' 或其非字符串，已跳过。项目: {item}")
                continue
            if pose_id in processed_poses:
                 logger.warning(f"在 SUPPORTED_POSES_DATA 中发现重复的 pose_id: '{pose_id}'，后一个将覆盖前一个。")
            processed_poses[pose_id] = item # 存储整个姿势配置字典
    else:
        logger.error(f"从 angle_config.py 加载的 SUPPORTED_POSES_DATA 类型不受支持: {type(source_data_for_registry)}。期望为 dict 或 list。")

    _SUPPORTED_POSES_REGISTRY = processed_poses
    if not _SUPPORTED_POSES_REGISTRY:
        logger.warning("_SUPPORTED_POSES_REGISTRY 初始化后为空。请检查 angle_config.py 中的 SUPPORTED_POSES_DATA 定义。")
    else:
        logger.info(f"_SUPPORTED_POSES_REGISTRY 初始化完成，加载了 {len(_SUPPORTED_POSES_REGISTRY)} 个姿势的元数据。")

    # 确保 ANGLE_CONFIG_DATA 也是字典
    if not isinstance(ANGLE_CONFIG_DATA, dict):
        logger.error(f"angle_config.angle_config (即 ANGLE_CONFIG_DATA) 不是字典类型，而是 {type(ANGLE_CONFIG_DATA)}。这将影响目标角度的获取。")
        # 可以选择在这里将 ANGLE_CONFIG_DATA 也置为空字典，或者让后续使用它的代码处理
        # global ANGLE_CONFIG_DATA
        # ANGLE_CONFIG_DATA = {} # 如果希望强制它为空字典


# 在模块加载时执行初始化
_initialize_supported_poses()


class ErrorCode(Enum):
    """错误码枚举"""
    SUCCESS = "SUCCESS"
    NO_KEYPOINT = "NO_KEYPOINT"
    INVALID_POSE = "INVALID_POSE"
    INVALID_INPUT = "INVALID_INPUT"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    SKELETON_ERROR = "SKELETON_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR" # 新增配置错误
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
            "supported_pose_ids": available_pose_ids # 使用更明确的键名
        }
        super().__init__(message, ErrorCode.INVALID_POSE, details)


@dataclass
class DetectionConfig:
    """检测配置类"""
    angle_tolerance_excellent: float = 10.0
    angle_tolerance_good: float = 20.0
    angle_tolerance_pass: float = 30.0
    angle_tolerance_poor: float = 45.0
    min_detection_rate: float = 0.6
    detection_rate_penalty: float = 0.8 # 如果检测到的关节数少于阈值，对最终分数应用此惩罚
    joint_weights: Dict[str, float] = field(default_factory=lambda: {
        "left_shoulder": 1.0, "right_shoulder": 1.0, "left_elbow": 1.0,
        "right_elbow": 1.0, "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 1.0, "right_knee": 1.0
        # 可以添加更多关节及其默认权重
    })
    # 假设 SkeletonFormat 和 DrawConfig 定义在 utils.draw 模块
    # from utils.draw import SkeletonFormat, DrawConfig # 移到模块顶部或按需导入
    # skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17
    # draw_config: Optional[DrawConfig] = None
    enable_multi_person: bool = False
    max_persons: int = 1
    person_selection_strategy: str = "largest" # "largest", "closest_to_center", "highest_confidence"


@dataclass
class PoseDetectionResult:
    """姿势检测结果的标准化结构"""
    score: float
    skeleton_image_bytes: BytesIO # 更明确的命名
    joint_angles_calculated: Dict[str, float] # 更明确的命名
    scoring_details: Dict[str, Any] # 从 score_pose 返回的详细信息
    timing_stats_ms: Dict[str, float] # 更明确的命名，单位为毫秒


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

        if len_v1 < 1e-9 or len_v2 < 1e-9: # 避免除以零或非常小的值导致数值不稳定
            return 0.0 # 或根据业务逻辑返回特定值/抛出异常

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot_product / (len_v1 * len_v2)
        # 限制cos_angle在[-1, 1]范围内，防止math.acos的DomainError
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    except Exception as e:
        logger.error(f"计算角度时发生错误: p1={p1}, p2={p2}, p3={p3}, 错误: {e}", exc_info=True)
        # 重新抛出，或者根据需要包装成自定义异常
        raise ValueError(f"角度计算失败: {str(e)}")


def calculate_joint_angles(keypoints: Dict[str, List[float]]) -> Dict[str, float]:
    """根据提供的关键点计算预定义的各个关节的角度。"""
    if not isinstance(keypoints, dict):
        logger.error(f"calculate_joint_angles 期望关键点为字典，但收到 {type(keypoints)}")
        return {} # 或者抛出 TypeError

    angles: Dict[str, float] = {}
    # 定义关节及其构成点 [远端点, 关节点(顶点), 近端点]
    # 这些点名需要与 keypoints 字典中的键名一致
    joint_definitions = {
        "left_shoulder": ["left_elbow", "left_shoulder", "left_hip"],
        "right_shoulder": ["right_elbow", "right_shoulder", "right_hip"],
        "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_hip": ["left_shoulder", "left_hip", "left_knee"], # 或 "neck", "left_hip", "left_knee"
        "right_hip": ["right_shoulder", "right_hip", "right_knee"],# 或 "neck", "right_hip", "right_knee"
        "left_knee": ["left_hip", "left_knee", "left_ankle"],
        "right_knee": ["right_hip", "right_knee", "right_ankle"]
    }

    for joint_name, point_keys in joint_definitions.items():
        p1_key, p2_key, p3_key = point_keys
        p1_coords = keypoints.get(p1_key)
        p2_coords = keypoints.get(p2_key) # 关节点（顶点）
        p3_coords = keypoints.get(p3_key)

        if p1_coords and p2_coords and p3_coords: # 确保所有三个点都存在
            try:
                angle = calculate_angle(p1_coords, p2_coords, p3_coords)
                angles[joint_name] = angle
            except ValueError as e: # calculate_angle 可能抛出 ValueError
                logger.warning(f"计算关节 '{joint_name}' 角度失败: {e}")
            except Exception as e_calc: # 捕获其他意外错误
                 logger.error(f"计算关节 '{joint_name}' 角度时发生意外错误: {e_calc}", exc_info=True)
        else:
            missing = [k for k,v in [(p1_key,p1_coords), (p2_key,p2_coords), (p3_key,p3_coords)] if v is None]
            logger.debug(f"计算关节 '{joint_name}' 角度时缺少关键点: {missing}")
    return angles


def score_pose(
    keypoints: Dict[str, List[float]],
    target_angles_for_pose: Dict[str, float], # 特定姿势的目标角度
    config: DetectionConfig
) -> Tuple[float, Dict[str, Any]]:
    """基于关节角度的姿势评分算法。"""
    if not keypoints:
        logger.warning("评分时关键点数据为空。")
        return 0.0, {"error": "no_keypoints_for_scoring", "message": "未提供关键点数据进行评分。"}
    if not target_angles_for_pose:
        logger.warning("评分时目标角度配置为空。")
        return 0.0, {"error": "no_target_angles_for_scoring", "message": "当前姿势无目标角度配置。"}

    # 1. 计算当前姿势的实际关节角度
    current_joint_angles = calculate_joint_angles(keypoints)
    if not current_joint_angles:
        logger.warning("评分时未能根据关键点计算出任何当前关节角度。")
        return 0.0, {"error": "no_current_angles_calculated", "message": "无法计算当前关节角度。"}

    # 2. 计算每个目标关节的得分
    joint_scores_weighted: Dict[str, float] = {}
    joint_analysis_details: Dict[str, Dict[str, Any]] = {}
    total_score_sum = 0.0
    total_applicable_weight_sum = 0.0 # 只计算实际参与评分的关节的总权重

    for joint_name, target_angle_value in target_angles_for_pose.items():
        current_angle_value = current_joint_angles.get(joint_name)
        if current_angle_value is None:
            logger.debug(f"目标关节 '{joint_name}' 在当前姿势中未计算出角度，跳过评分。")
            joint_analysis_details[joint_name] = {"status": "missing_current_angle", "target": target_angle_value}
            continue # 跳过此关节的评分

        angle_difference = abs(current_angle_value - target_angle_value)
        joint_score_unweighted = 0.0 # 0-100
        grade = "fail"

        if angle_difference <= config.angle_tolerance_excellent:
            joint_score_unweighted = 100.0
            grade = "excellent"
        elif angle_difference <= config.angle_tolerance_good:
            # 线性递减: 从100到 (比如) 80
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
        else: # angle_difference > config.angle_tolerance_poor
            # 可以设计一个更陡峭的下降或直接为0
            joint_score_unweighted = max(0.0, 40.0 - (angle_difference - config.angle_tolerance_poor) * 1.0) # 示例
            grade = "fail"

        joint_score_unweighted = max(0.0, min(100.0, joint_score_unweighted)) # 确保在0-100之间

        joint_weight = config.joint_weights.get(joint_name, 1.0) # 获取权重，默认为1
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

    # 3. 计算加权平均分
    average_score = (total_score_sum / total_applicable_weight_sum) if total_applicable_weight_sum > 0 else 0.0

    # 4. 根据检测到的关节数量应用惩罚（可选）
    num_target_joints = len(target_angles_for_pose)
    num_scored_joints = len(joint_scores_weighted) # 实际参与评分的关节数
    detection_rate = (num_scored_joints / num_target_joints) if num_target_joints > 0 else 0.0
    penalty_info = {"applied": False, "factor": 1.0}

    if detection_rate < config.min_detection_rate:
        # 示例惩罚：如果检测率低于阈值，按比例降低分数，并应用额外惩罚系数
        # penalty_factor = (detection_rate / config.min_detection_rate) * config.detection_rate_penalty
        # 更简单的惩罚：直接乘以检测率和惩罚系数
        penalty_factor = detection_rate * config.detection_rate_penalty
        average_score *= penalty_factor
        penalty_info = {"applied": True, "factor": round(penalty_factor, 3), "original_detection_rate": round(detection_rate,3)}
        logger.info(f"应用检测率惩罚: 原始检测率={detection_rate:.2f}, 惩罚因子={penalty_factor:.2f}, 惩罚后平均分暂为={average_score:.2f}")

    final_score = round(max(0.0, min(100.0, average_score)), 2) # 最终分数也约束在0-100

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
            # 假设坐标是归一化的 (0-1范围)。可以添加范围校验。
            # if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            #     logger.warning(f"关键点 '{name}' 的坐标 ({x:.3f}, {y:.3f}) 超出期望的归一化范围 [0,1]。")
            #     # 根据业务需求决定是否跳过或继续处理
            validated_keypoints[name] = [x, y] # 只取x, y，忽略可能的置信度等
        except (ValueError, TypeError) as e:
            logger.warning(f"无法解析关键点 '{name}' 的坐标: {coords}，错误: {e}")
            continue # 跳过无法解析的关键点
    return validated_keypoints


def detect_pose(
    image_bytes: bytes,
    pose_id: str,
    config: Optional[DetectionConfig] = None
) -> Tuple[float, BytesIO]: # 返回 (评分, 骨架图BytesIO对象)
    """
    核心姿势检测函数。
    接收图片字节流和姿势ID，返回评分和生成的骨架图（内存中）。
    """
    overall_start_time = time.monotonic()
    timing_stats: Dict[str, float] = {} # 用于存储各阶段耗时

    current_config = config or DetectionConfig() # 使用传入配置或默认配置

    # 1. 输入参数验证
    if not image_bytes or not isinstance(image_bytes, bytes):
        logger.error(f"无效的图片输入: image_bytes 为空或类型错误 ({type(image_bytes)})")
        raise PoseDetectionError("图片数据无效或为空。", ErrorCode.INVALID_INPUT, {"input_type": str(type(image_bytes))})

    if not isinstance(pose_id, str) or not pose_id.strip(): # 确保 pose_id 是非空字符串
        logger.error(f"无效的 pose_id 输入: 类型为 {type(pose_id)}，值为空或仅包含空白。")
        raise InvalidPoseError("pose_id 必须为有效的非空字符串。", str(pose_id), get_supported_pose_ids())

    # 1.1 检查 pose_id 是否受支持 (使用 _SUPPORTED_POSES_REGISTRY)
    # _SUPPORTED_POSES_REGISTRY 保证是字典
    if pose_id not in _SUPPORTED_POSES_REGISTRY:
        logger.warning(f"请求的 pose_id '{pose_id}' 不在支持的姿势元数据 (_SUPPORTED_POSES_REGISTRY) 中。")
        raise InvalidPoseError(
            f"不支持的动作ID: '{pose_id}'。",
            pose_id,
            get_supported_pose_ids() # 传递ID列表给错误详情
        )
    logger.info(f"开始姿势检测流程: pose_id='{pose_id}', 图片大小={len(image_bytes)} bytes")

    try:
        # 2. 从图片中推理关键点
        inference_start_time = time.monotonic()
        raw_keypoints_data = infer_keypoints(image_bytes) # 假设 infer_keypoints 返回 Dict[str, List[float]] 或类似结构
        timing_stats["keypoint_inference_ms"] = (time.monotonic() - inference_start_time) * 1000
        logger.info(f"关键点推理完成，用时: {timing_stats['keypoint_inference_ms']:.2f}ms")

        if not raw_keypoints_data or not isinstance(raw_keypoints_data, dict):
            logger.warning(f"关键点推理结果为空或格式不正确: {type(raw_keypoints_data)}")
            raise NoKeypointError("未能从图片中检测到任何关键点数据。", {"inference_output_type": str(type(raw_keypoints_data))})

        # 3. 验证和规范化关键点数据
        validation_start_time = time.monotonic()
        try:
            validated_keypoints = validate_keypoints(raw_keypoints_data)
        except ValueError as ve: # validate_keypoints 抛出 ValueError
            logger.warning(f"关键点数据验证失败: {ve}")
            raise NoKeypointError(f"检测到的关键点数据格式无效: {ve}", {"validation_error": str(ve)})
        timing_stats["keypoint_validation_ms"] = (time.monotonic() - validation_start_time) * 1000

        if not validated_keypoints:
             logger.warning("所有检测到的关键点均未能通过验证。")
             raise NoKeypointError("所有检测到的关键点均未能通过验证。", {"raw_keypoints_count": len(raw_keypoints_data)})
        logger.info(f"已验证 {len(validated_keypoints)} 个关键点。")


        # 4. 获取当前姿势的目标角度配置 (从 ANGLE_CONFIG_DATA)
        # ANGLE_CONFIG_DATA 应该也是一个字典: {pose_id: {joint_name: angle, ...}}
        if not isinstance(ANGLE_CONFIG_DATA, dict):
            logger.critical(f"内部配置错误: ANGLE_CONFIG_DATA 不是字典类型 (实际为: {type(ANGLE_CONFIG_DATA)})。无法进行评分。")
            raise PoseDetectionError("服务器内部角度配置错误。", ErrorCode.CONFIG_ERROR)

        target_angles_for_current_pose = ANGLE_CONFIG_DATA.get(pose_id)
        if not target_angles_for_current_pose or not isinstance(target_angles_for_current_pose, dict):
            logger.error(f"未找到动作 '{pose_id}' 的有效目标角度配置。 ANGLE_CONFIG_DATA 中对应的值为: {target_angles_for_current_pose}")
            raise PoseDetectionError(
                f"内部错误：动作 '{pose_id}' 的目标角度配置缺失或无效。",
                ErrorCode.CONFIG_ERROR, # 更明确的错误码
                {"details": f"Target angles for '{pose_id}' in ANGLE_CONFIG_DATA is missing or not a dict."}
            )

        # 5. 计算姿势分数
        scoring_start_time = time.monotonic()
        final_score, scoring_details_map = score_pose(validated_keypoints, target_angles_for_current_pose, current_config)
        timing_stats["pose_scoring_ms"] = (time.monotonic() - scoring_start_time) * 1000
        logger.info(f"姿势评分完成: 最终分数={final_score}, 用时: {timing_stats['pose_scoring_ms']:.2f}ms")

        # 6. 生成骨架图 (这是一个占位/模拟实现)
        skeleton_draw_start_time = time.monotonic()
        skeleton_image_bytes_io = BytesIO()
        try:
            # 模拟实际的骨架图绘制逻辑
            # from utils.draw import draw_skeleton, DrawConfig, SkeletonFormat # 假设的导入
            # draw_config_to_use = current_config.draw_config or DrawConfig()
            # skeleton_format_to_use = current_config.skeleton_format or SkeletonFormat.COCO_17
            # skeleton_image_bytes_io = draw_skeleton(
            #     image_bytes, validated_keypoints, config=draw_config_to_use, skeleton_format=skeleton_format_to_use
            # )
            # 简单地写入一些模拟数据
            skeleton_image_bytes_io.write(f"Skeleton for {pose_id} - Score: {final_score}".encode('utf-8'))
            skeleton_image_bytes_io.seek(0) # 重置指针到开头，以便读取
            
            if skeleton_image_bytes_io.getbuffer().nbytes == 0: # 检查缓冲区是否为空
                 logger.error("骨架图生成后，缓冲区为空。")
                 raise PoseDetectionError("骨架图生成失败：返回了空数据。", ErrorCode.SKELETON_ERROR)

        except ImportError: # 如果 utils.draw 导入失败
            logger.warning("绘图模块 (utils.draw) 未能导入，将返回空的骨架图。")
            # skeleton_image_bytes_io 会是一个空的 BytesIO
        except Exception as e_draw:
            logger.error(f"生成骨架图时发生错误: {e_draw}", exc_info=True)
            raise PoseDetectionError(f"骨架图生成时发生内部错误: {str(e_draw)}", ErrorCode.SKELETON_ERROR)
        finally:
            timing_stats["skeleton_drawing_ms"] = (time.monotonic() - skeleton_draw_start_time) * 1000
            logger.info(f"骨架图生成/处理完成，用时: {timing_stats['skeleton_drawing_ms']:.2f}ms")


        timing_stats["total_detect_pose_ms"] = (time.monotonic() - overall_start_time) * 1000
        logger.info(f"姿势检测流程成功完成: pose_id='{pose_id}', 最终分数={final_score}, 总用时={timing_stats['total_detect_pose_ms']:.2f}ms")
        logger.debug(f"详细耗时 (ms): {timing_stats}")

        # 构建并返回标准化的结果对象 (如果定义了 PoseDetectionResult)
        # current_joint_angles = scoring_details_map.get("joint_analysis", {}) # 假设 score_pose 返回的细节中有这个
        # detection_result = PoseDetectionResult(
        #     score=final_score,
        #     skeleton_image_bytes=skeleton_image_bytes_io,
        #     joint_angles_calculated=current_joint_angles, # 需要从 scoring_details_map 中获取
        #     scoring_details=scoring_details_map,
        #     timing_stats_ms=timing_stats
        # )
        # return detection_result.score, detection_result.skeleton_image_bytes
        return final_score, skeleton_image_bytes_io # 直接返回元组

    except (NoKeypointError, InvalidPoseError, PoseDetectionError) as e_custom: # 捕获我们自定义的异常
        logger.warning(f"姿势检测中发生已知业务逻辑错误 (pose_id='{pose_id}'): Code={e_custom.code.value}, Msg='{e_custom}'")
        raise # 直接重新抛出，它们已经包含了足够的上下文
    except Exception as e_unexpected: # 捕获所有其他未预期的Python原生异常
        logger.error(f"姿势检测过程中发生未预期系统错误 (pose_id='{pose_id}'): {e_unexpected}", exc_info=True)
        # 将原生异常包装成我们的 PoseDetectionError 以便上层统一处理
        raise PoseDetectionError(
            f"姿势检测时发生内部系统错误: {str(e_unexpected)}",
            ErrorCode.UNKNOWN_ERROR,
            {"original_error_type": type(e_unexpected).__name__}
        ) from e_unexpected # 保留原始异常链，便于调试


def select_best_person(
    multi_person_keypoints_result: Dict[str, Any], # 假设这是模型输出的原始多人结果
    strategy: str = "largest_bbox" # 例如: "largest_bbox", "highest_total_score", "closest_to_center"
) -> Dict[str, List[float]]:
    """
    （占位）从多人关键点检测结果中根据指定策略选择一个“最佳”人体。
    """
    # 实际实现会复杂得多，需要解析 multi_person_keypoints_result 的结构，
    # 可能包含每个人的边界框、置信度、关键点列表等。
    logger.warning(f"select_best_person 功能 (策略: '{strategy}') 尚未完全实现。将尝试返回第一个检测到的人（如果适用）。")
    if isinstance(multi_person_keypoints_result, list) and multi_person_keypoints_result:
        # 如果结果是列表，每个元素代表一个人
        if isinstance(multi_person_keypoints_result[0], dict) and "keypoints" in multi_person_keypoints_result[0]:
             return multi_person_keypoints_result[0]["keypoints"] # 假设结构
    elif isinstance(multi_person_keypoints_result, dict) and "persons" in multi_person_keypoints_result:
        # 如果结果是字典，包含一个 "persons" 列表
        persons = multi_person_keypoints_result["persons"]
        if isinstance(persons, list) and persons and isinstance(persons[0], dict) and "keypoints" in persons[0]:
            return persons[0]["keypoints"] # 假设结构

    # 如果无法按简单方式提取，则返回空或抛异常
    logger.error("select_best_person: 无法从输入数据中提取单人关键点。")
    raise NotImplementedError("多人检测中选择最佳人体的逻辑尚未完全实现或输入格式不兼容。")


def get_supported_poses() -> Dict[str, Dict[str, Any]]:
    """
    获取所有支持的姿势及其详细元数据配置。
    返回的是一个字典，键为 pose_id，值为该姿势的配置字典。
    这个配置字典通常包含如 "name", "description", "difficulty" 等信息。
    不包含目标角度数据，目标角度数据由 ANGLE_CONFIG_DATA 提供。
    """
    if not isinstance(_SUPPORTED_POSES_REGISTRY, dict):
        # 这种情况理论上不应发生，因为 _initialize_supported_poses 会确保它是字典
        logger.critical(f"_SUPPORTED_POSES_REGISTRY 意外地不是字典类型 (实际为: {type(_SUPPORTED_POSES_REGISTRY)})。模块初始化可能存在问题。")
        return {} # 返回空字典以避免下游因类型错误而崩溃
    return _SUPPORTED_POSES_REGISTRY.copy() # 返回副本以防止外部修改


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

    # 1. 获取姿势元数据
    # _SUPPORTED_POSES_REGISTRY 保证是字典
    pose_meta_config = _SUPPORTED_POSES_REGISTRY.get(pose_id)
    if pose_meta_config is None:
        logger.warning(f"未在 _SUPPORTED_POSES_REGISTRY 中找到 pose_id '{pose_id}' 的元数据配置。")
        return None
    if not isinstance(pose_meta_config, dict):
        logger.error(f"pose_id '{pose_id}' 在 _SUPPORTED_POSES_REGISTRY 中的元数据配置不是字典类型 (实际为: {type(pose_meta_config)})。")
        return None # 配置数据损坏

    # 2. 获取目标角度数据
    # ANGLE_CONFIG_DATA 也应该是一个字典
    if not isinstance(ANGLE_CONFIG_DATA, dict):
        logger.error(f"全局 ANGLE_CONFIG_DATA 不是字典类型 (实际为: {type(ANGLE_CONFIG_DATA)})。无法获取 '{pose_id}' 的目标角度。")
        # 根据策略，可以返回部分信息（不含角度）或直接返回 None
        target_angles_for_pose = {"error": "Angle configuration data is not a dictionary."}
    else:
        target_angles_for_pose = ANGLE_CONFIG_DATA.get(pose_id)
        if target_angles_for_pose is None:
            logger.warning(f"未在 ANGLE_CONFIG_DATA 中找到 pose_id '{pose_id}' 的目标角度配置。")
            # 即使没有目标角度，也可能希望返回其他元数据
            target_angles_for_pose = {"warning": "Target angles not defined for this pose."}
        elif not isinstance(target_angles_for_pose, dict):
            logger.error(f"pose_id '{pose_id}' 在 ANGLE_CONFIG_DATA 中的目标角度配置不是字典类型 (实际为: {type(target_angles_for_pose)})。")
            target_angles_for_pose = {"error": "Target angle configuration for this pose is not a dictionary."}
        
    # 3. 构建并返回聚合信息
    # 使用 pose_meta_config 中的信息，并加入 target_angles
    # 可以选择性地从 pose_meta_config 提取字段，或直接将其作为基础
    return {
        "pose_id": pose_id,
        "name": pose_meta_config.get("name", pose_id), # 如果name缺失，使用pose_id
        "description": pose_meta_config.get("description", "无可用描述。"),
        "difficulty": pose_meta_config.get("difficulty", "未知"),
        # 可以添加其他存储在 pose_meta_config 中的字段
        **{k: v for k, v in pose_meta_config.items() if k not in ["id", "pose_id", "name", "description", "difficulty"]},
        "target_angles": target_angles_for_pose # 加入目标角度信息
    }

