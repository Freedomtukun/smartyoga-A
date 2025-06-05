"""
骨架绘制模块
用于 API 端到端图片流转，不生成任何磁盘文件，不保证物理存储
支持多种骨架格式和高分辨率图像处理
"""

import io
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)


class SkeletonDrawError(Exception):
    """骨架绘制专用异常类"""
    pass


class SkeletonFormat(Enum):
    """支持的骨架格式枚举"""
    COCO_17 = "coco_17"
    MOVENET_17 = "movenet_17"
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
    
    # 背景配置
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)  # 透明背景
    use_original_image: bool = True  # 是否使用原图作为背景


# 骨架连接定义
SKELETON_CONNECTIONS = {
    SkeletonFormat.COCO_17: [
        # 头部连接
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        # 躯干连接
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        # 左臂连接
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        # 右臂连接
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        # 左腿连接
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        # 右腿连接
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ],
    SkeletonFormat.MOVENET_17: [
        # 与COCO_17相同
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
    image_bytes: Optional[bytes] = None,
    keypoints: Dict[str, List[float]] = None,
    config: Optional[DrawConfig] = None,
    skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17,
    image_size: Optional[Tuple[int, int]] = None
) -> BytesIO:
    """
    在透明背景或原图上绘制人体骨架
    
    此函数仅用于 API 端到端图片流转，不生成任何磁盘文件
    
    Args:
        image_bytes: 原始图片数据（可选）
        keypoints: 关键点字典，格式为 {关键点名: [归一化x, 归一化y]}
        config: 绘制配置，为 None 时使用默认配置
        skeleton_format: 骨架格式
        image_size: 当不提供image_bytes时，指定输出图片尺寸 (width, height)
    
    Returns:
        BytesIO: 包含骨架图PNG数据的缓冲区
        
    Raises:
        SkeletonDrawError: 图片处理或绘制失败时抛出
    """
    if config is None:
        config = DrawConfig()
    
    if keypoints is None:
        raise SkeletonDrawError("关键点数据不能为空")
    
    try:
        # 确定图片尺寸
        if image_bytes and config.use_original_image:
            # 使用原图作为背景
            try:
                original_img = Image.open(io.BytesIO(image_bytes))
                if original_img.mode != 'RGBA':
                    original_img = original_img.convert('RGBA')
                width, height = original_img.size
                skeleton_img = original_img.copy()
                logger.debug(f"Using original image as background: {width}x{height}")
            except Exception as e:
                raise SkeletonDrawError(f"无法打开输入图像: {str(e)}")
        else:
            # 创建新的透明背景图片
            if image_size:
                width, height = image_size
            elif image_bytes:
                # 从image_bytes获取尺寸但不使用原图
                try:
                    temp_img = Image.open(io.BytesIO(image_bytes))
                    width, height = temp_img.size
                    temp_img.close()
                except Exception as e:
                    raise SkeletonDrawError(f"无法获取图像尺寸: {str(e)}")
            else:
                # 默认尺寸
                width, height = 256, 256
                
            skeleton_img = Image.new("RGBA", (width, height), config.background_color)
            logger.debug(f"Created new image with size: {width}x{height}")
        
        # 验证图像尺寸
        _validate_image_size(width, height, config.max_image_size)
        
        # 获取对应的骨架连接
        if skeleton_format not in SKELETON_CONNECTIONS:
            raise SkeletonDrawError(f"不支持的骨架格式: {skeleton_format}")
        
        skeleton_connections = SKELETON_CONNECTIONS[skeleton_format]
        
        # 创建绘图对象
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
        
        # 绘制骨架连接线（先画线，后画点，避免线覆盖点）
        connections_drawn = 0
        for start, end in skeleton_connections:
            if start in pixel_keypoints and end in pixel_keypoints:
                try:
                    start_point = pixel_keypoints[start]
                    end_point = pixel_keypoints[end]
                    
                    # 如果启用抗锯齿，使用多次绘制实现
                    if config.enable_antialiasing and config.line_width > 1:
                        # 绘制多条稍微偏移的线来模拟抗锯齿
                        for offset in [-0.5, 0, 0.5]:
                            draw.line(
                                [(start_point[0] + offset, start_point[1]), 
                                 (end_point[0] + offset, end_point[1])],
                                fill=config.line_color,
                                width=config.line_width
                            )
                    else:
                        draw.line(
                            [start_point, end_point],
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
                if config.enable_antialiasing:
                    # 使用渐变圆实现抗锯齿效果
                    for r in range(config.point_radius, 0, -1):
                        alpha = int(255 * (r / config.point_radius))
                        color = (config.point_color[0], config.point_color[1], 
                                config.point_color[2], alpha)
                        bbox = [x - r, y - r, x + r, y + r]
                        draw.ellipse(bbox, fill=color)
                else:
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
        
        buffer.seek(0)
        logger.info(
            f"骨架图生成成功: {buffer_size} bytes, "
            f"{connections_drawn} 连接线, {points_drawn} 关键点"
        )
        
        return buffer
        
    except SkeletonDrawError:
        raise
    except Exception as e:
        logger.error(f"骨架绘制出现未预期错误: {e}", exc_info=True)
        raise SkeletonDrawError(f"骨架图绘制失败: {str(e)}")


def draw_coco_skeleton(keypoints: Dict[str, List[float]], size: int = 256) -> BytesIO:
    """
    便捷函数：绘制COCO格式骨架图
    
    Args:
        keypoints: 归一化坐标的关键点字典
        size: 输出图片尺寸（正方形）
        
    Returns:
        BytesIO: 包含骨架PNG图片的内存缓冲区
    """
    config = DrawConfig(
        line_color=(0, 255, 0, 255),  # 绿色连接线
        line_width=3,
        point_color=(255, 0, 0, 255),  # 红色关键点
        point_radius=5,
        background_color=(255, 255, 255, 0),  # 透明背景
        use_original_image=False,
        enable_antialiasing=True
    )
    
    return draw_skeleton(
        image_bytes=None,
        keypoints=keypoints,
        config=config,
        skeleton_format=SkeletonFormat.COCO_17,
        image_size=(size, size)
    )


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


def create_skeleton_overlay(
    image_bytes: bytes,
    keypoints: Dict[str, List[float]],
    config: Optional[DrawConfig] = None,
    skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17,
    alpha: float = 0.7
) -> BytesIO:
    """
    在原图上叠加骨架，返回合成图
    
    Args:
        image_bytes: 原始图片数据
        keypoints: 关键点字典
        config: 绘制配置
        skeleton_format: 骨架格式
        alpha: 骨架图层的透明度 (0-1)
        
    Returns:
        BytesIO: 包含叠加骨架的图片
    """
    if config is None:
        config = DrawConfig()
    
    # 先在原图上绘制骨架
    config.use_original_image = True
    skeleton_buffer = draw_skeleton(
        image_bytes=image_bytes,
        keypoints=keypoints,
        config=config,
        skeleton_format=skeleton_format
    )
    
    return skeleton_buffer


# 向后兼容的别名
MOVENET_SKELETON = SKELETON_CONNECTIONS[SkeletonFormat.MOVENET_17]
