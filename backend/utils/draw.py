"""
骨架绘制模块
用于 API 端到端图片流转，不生成任何磁盘文件，不保证物理存储
支持多种骨架格式和高分辨率图像处理
"""

import io
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw
from io import BytesIO

logger = logging.getLogger(__name__)


class SkeletonDrawError(Exception):
    """骨架绘制专用异常类"""
    pass


class SkeletonFormat(Enum):
    """支持的骨架格式枚举"""
    COCO_17 = "coco_17"
    MOVENET_17 = "movenet_17"
    # 预留更多格式
    OPENPOSE_25 = "openpose_25"
    MEDIAPIPE_33 = "mediapipe_33"


@dataclass
class DrawConfig:
    """绘制配置类"""
    # 骨架连接线配置
    line_color: Tuple[int, int, int, int] = (0, 255, 0, 255)  # 绿色 RGBA
    line_width: int = 3
    
    # 关键点配置
    point_color: Tuple[int, int, int, int] = (255, 0, 0, 255)  # 红色 RGBA
    point_radius: int = 5
    
    # 性能配置
    max_image_size: Tuple[int, int] = (4096, 4096)  # 最大支持的图像尺寸
    enable_antialiasing: bool = True  # 是否启用抗锯齿
    
    # 验证配置
    strict_validation: bool = False  # 是否严格验证所有关键点
    min_keypoints_ratio: float = 0.3  # 最少需要的有效关键点比例


# 骨架连接定义
SKELETON_CONNECTIONS = {
    SkeletonFormat.COCO_17: [
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ],
    SkeletonFormat.MOVENET_17: [
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ],
}


def _validate_image_size(width: int, height: int, max_size: Tuple[int, int]) -> None:
    """验证图像尺寸是否在允许范围内"""
    max_width, max_height = max_size
    if width > max_width or height > max_height:
        raise SkeletonDrawError(
            f"图像尺寸 ({width}x{height}) 超出最大限制 ({max_width}x{max_height})"
        )


def _normalize_keypoint(x: float, y: float, width: int, height: int) -> Optional[Tuple[int, int]]:
    """
    将归一化坐标转换为像素坐标
    
    Args:
        x, y: 归一化坐标 [0, 1]
        width, height: 图像尺寸
        
    Returns:
        像素坐标元组，无效坐标返回 None
    """
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    
    pixel_x = int(x * width)
    pixel_y = int(y * height)
    
    # 确保坐标在图片范围内
    pixel_x = max(0, min(pixel_x, width - 1))
    pixel_y = max(0, min(pixel_y, height - 1))
    
    return (pixel_x, pixel_y)


def draw_skeleton(
    image_bytes: bytes, 
    keypoints: Dict[str, List[float]], 
    config: Optional[DrawConfig] = None,
    skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17
) -> BytesIO:
    """
    在透明背景上绘制人体骨架
    
    此函数仅用于 API 端到端图片流转，不生成任何磁盘文件，不保证物理存储
    
    Args:
        image_bytes: 原始图片数据
        keypoints: 关键点字典，格式为 {关键点名: [归一化x, 归一化y]}
                  归一化坐标范围为 [0, 1]
        config: 绘制配置，为 None 时使用默认配置
        skeleton_format: 骨架格式，默认为 COCO_17
    
    Returns:
        BytesIO: 包含骨架图PNG数据的缓冲区
        
    Raises:
        SkeletonDrawError: 图片处理或绘制失败时抛出
    """
    if config is None:
        config = DrawConfig()
    
    try:
        # 打开原始图片获取尺寸
        try:
            original_img = Image.open(io.BytesIO(image_bytes))
            width, height = original_img.size
            logger.debug(f"Original image size: {width}x{height}")
        except Exception as e:
            raise SkeletonDrawError(f"无法打开输入图像: {str(e)}")
        
        # 验证图像尺寸
        _validate_image_size(width, height, config.max_image_size)
        
        # 获取对应的骨架连接
        if skeleton_format not in SKELETON_CONNECTIONS:
            raise SkeletonDrawError(f"不支持的骨架格式: {skeleton_format}")
        
        skeleton_connections = SKELETON_CONNECTIONS[skeleton_format]
        
        # 创建透明背景的图片
        skeleton_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(skeleton_img, 'RGBA')
        
        # 将归一化坐标转换为像素坐标
        pixel_keypoints = {}
        invalid_keypoints = []
        
        for name, coords in keypoints.items():
            if len(coords) < 2:
                logger.warning(f"关键点 {name} 坐标数据不完整: {coords}")
                invalid_keypoints.append(name)
                continue
                
            pixel_coords = _normalize_keypoint(coords[0], coords[1], width, height)
            if pixel_coords:
                pixel_keypoints[name] = pixel_coords
            else:
                logger.warning(f"关键点 {name} 坐标超出范围: ({coords[0]:.3f}, {coords[1]:.3f})")
                invalid_keypoints.append(name)
        
        # 验证有效关键点数量
        total_required = len(set().union(*[[s, e] for s, e in skeleton_connections]))
        valid_ratio = len(pixel_keypoints) / total_required if total_required > 0 else 0
        
        if config.strict_validation and valid_ratio < config.min_keypoints_ratio:
            raise SkeletonDrawError(
                f"有效关键点比例过低: {valid_ratio:.2f} < {config.min_keypoints_ratio}"
            )
        
        logger.info(f"关键点转换完成: {len(pixel_keypoints)}/{len(keypoints)} 有效")
        
        # 绘制骨架连接线
        connections_drawn = 0
        for start, end in skeleton_connections:
            if start in pixel_keypoints and end in pixel_keypoints:
                try:
                    draw.line(
                        [pixel_keypoints[start], pixel_keypoints[end]], 
                        fill=config.line_color, 
                        width=config.line_width
                    )
                    connections_drawn += 1
                except Exception as e:
                    logger.warning(f"绘制连接线失败 {start}->{end}: {e}")
        
        logger.debug(f"绘制了 {connections_drawn}/{len(skeleton_connections)} 条连接线")
        
        # 绘制关键点
        points_drawn = 0
        for name, (x, y) in pixel_keypoints.items():
            try:
                bbox = [
                    x - config.point_radius, 
                    y - config.point_radius, 
                    x + config.point_radius, 
                    y + config.point_radius
                ]
                draw.ellipse(bbox, fill=config.point_color)
                points_drawn += 1
            except Exception as e:
                logger.warning(f"绘制关键点失败 {name} at ({x}, {y}): {e}")
        
        logger.debug(f"绘制了 {points_drawn} 个关键点")
        
        # 保存为PNG到内存缓冲区
        buffer = BytesIO()
        try:
            # 使用优化参数保存PNG
            skeleton_img.save(
                buffer, 
                format="PNG", 
                optimize=True,
                compress_level=6  # 平衡压缩率和速度
            )
            buffer.seek(0)
        except Exception as e:
            raise SkeletonDrawError(f"保存骨架图到缓冲区失败: {str(e)}")
        
        # 验证生成的图片数据
        buffer_size = len(buffer.getvalue())
        if buffer_size == 0:
            raise SkeletonDrawError("生成的骨架图数据为空")
        
        # 验证PNG格式
        buffer.seek(0)
        try:
            test_img = Image.open(buffer)
            if test_img.format != 'PNG':
                raise SkeletonDrawError(f"生成的图片格式错误: {test_img.format}")
            test_img.close()  # 显式关闭避免内存泄漏
        except Exception as e:
            raise SkeletonDrawError(f"生成的骨架图格式验证失败: {str(e)}")
        
        buffer.seek(0)  # 重置指针到开始位置
        logger.info(
            f"骨架图生成成功: {buffer_size} bytes, "
            f"{connections_drawn} 连接线, {points_drawn} 关键点"
        )
        
        return buffer
        
    except SkeletonDrawError:
        raise
    except Exception as e:
        logger.error(f"骨架绘制出现未预期错误: {e}")
        raise SkeletonDrawError(f"骨架图绘制失败: {str(e)}")


def validate_keypoints(
    keypoints: Dict[str, List[float]], 
    skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17
) -> Dict[str, List[str]]:
    """
    验证关键点数据的完整性
    
    Args:
        keypoints: 关键点字典
        skeleton_format: 骨架格式
        
    Returns:
        Dict[str, List[str]]: 验证结果，包含 valid, invalid, missing 三个列表
    """
    result = {
        "valid": [],
        "invalid": [],
        "missing": []
    }
    
    if skeleton_format not in SKELETON_CONNECTIONS:
        raise SkeletonDrawError(f"不支持的骨架格式: {skeleton_format}")
    
    # 获取所有可能的关键点名称
    all_keypoints = set()
    for start, end in SKELETON_CONNECTIONS[skeleton_format]:
        all_keypoints.add(start)
        all_keypoints.add(end)
    
    for name in all_keypoints:
        if name not in keypoints:
            result["missing"].append(name)
        else:
            coords = keypoints[name]
            if len(coords) >= 2 and 0 <= coords[0] <= 1 and 0 <= coords[1] <= 1:
                result["valid"].append(name)
            else:
                result["invalid"].append(name)
    
    return result


def get_skeleton_stats(
    keypoints: Dict[str, List[float]], 
    skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17
) -> Dict[str, Any]:
    """
    获取骨架绘制的统计信息
    
    Args:
        keypoints: 关键点字典
        skeleton_format: 骨架格式
        
    Returns:
        Dict: 统计信息
    """
    validation = validate_keypoints(keypoints, skeleton_format)
    
    if skeleton_format not in SKELETON_CONNECTIONS:
        raise SkeletonDrawError(f"不支持的骨架格式: {skeleton_format}")
    
    skeleton_connections = SKELETON_CONNECTIONS[skeleton_format]
    total_possible_connections = len(skeleton_connections)
    drawable_connections = 0
    
    for start, end in skeleton_connections:
        if start in validation["valid"] and end in validation["valid"]:
            drawable_connections += 1
    
    all_keypoints = set().union(*[[s, e] for s, e in skeleton_connections])
    total_keypoints_in_format = len(all_keypoints)
    
    return {
        "skeleton_format": skeleton_format.value,
        "total_keypoints": len(keypoints),
        "valid_keypoints": len(validation["valid"]),
        "invalid_keypoints": len(validation["invalid"]),
        "missing_keypoints": len(validation["missing"]),
        "total_keypoints_in_format": total_keypoints_in_format,
        "total_possible_connections": total_possible_connections,
        "drawable_connections": drawable_connections,
        "completeness_rate": len(validation["valid"]) / total_keypoints_in_format if total_keypoints_in_format > 0 else 0,
        "connection_rate": drawable_connections / total_possible_connections if total_possible_connections > 0 else 0,
        "validation_details": validation
    }


# 向后兼容的别名
MOVENET_SKELETON = SKELETON_CONNECTIONS[SkeletonFormat.MOVENET_17]