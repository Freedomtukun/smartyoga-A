"""
姿势检测API服务
提供图片上传、姿势检测、骨架图生成和上传的完整服务
"""

import os
import time
import uuid
import logging
import traceback
from typing import Dict, Tuple, Optional, Any, Set # Added Set
from dataclasses import dataclass, field # Added field
from functools import wraps
from io import BytesIO

from flask import Flask, request, jsonify, g, has_request_context # Added has_request_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.datastructures import FileStorage
from PIL import Image, UnidentifiedImageError # Added for image validation

# import redis # Redis remains commented out

from pose_detector import (
    detect_pose,
    PoseDetectionError,
    NoKeypointError,
    InvalidPoseError,
    get_supported_poses,
    get_supported_pose_ids,
    DetectionConfig
)
# Assuming cos_uploader.py is in the same directory or accessible via PYTHONPATH
from cos_uploader import upload_buffer, CosUploadError, check_cos_connection
# Assuming draw.py (or utils/draw.py) contains SkeletonDrawError
# Based on user's last provided app.py, using utils.draw
try:
    from utils.draw import SkeletonDrawError
except ImportError:
    # Fallback if utils.draw is not found, try direct import
    try:
        from draw import SkeletonDrawError
    except ImportError:
        # If neither is found, create a placeholder to avoid crashing,
        # but log a critical error. This part might need adjustment based on actual project structure.
        # Ensure logger is available for this critical log, might need to define a temporary one if setup_logging hasn't run
        _temp_logger = logging.getLogger(__name__)
        _temp_logger.critical("Critical: SkeletonDrawError could not be imported from utils.draw or draw. Please check project structure.")
        class SkeletonDrawError(Exception): pass


# ======================== API Version ========================
API_VERSION = "1.1.0" # Define API version for responses

# ======================== 配置管理 ========================

@dataclass
class AppConfig:
    """应用配置"""
    # 服务器配置
    PORT: int = int(os.environ.get('PORT', 5000))
    DEBUG: bool = os.environ.get('DEBUG', 'False').lower() == 'true'

    # 文件上传配置
    MAX_CONTENT_LENGTH: int = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {'jpg', 'jpeg', 'png'})

    # COS配置
    COS_BUCKET: str = os.environ.get('COS_BUCKET', '')
    COS_REGION: str = os.environ.get('COS_REGION', '')
    COS_SECRET_ID: str = os.environ.get('COS_SECRET_ID', '')
    COS_SECRET_KEY: str = os.environ.get('COS_SECRET_KEY', '')
    COS_CDN_DOMAIN: str = os.environ.get('COS_CDN_DOMAIN', '') # Handled by cos_uploader

    # Redis配置（可选）
    # REDIS_URL: str = os.environ.get('REDIS_URL', '')
    # CACHE_TTL: int = int(os.environ.get('CACHE_TTL', 3600))  # 1小时

    # 性能配置
    REQUEST_TIMEOUT: int = int(os.environ.get('REQUEST_TIMEOUT', 30))  # 30秒
    ENABLE_ASYNC: bool = os.environ.get('ENABLE_ASYNC', 'False').lower() == 'true'

    # 监控配置
    ENABLE_METRICS: bool = os.environ.get('ENABLE_METRICS', 'True').lower() == 'true'

    # 日志配置
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO').upper() # Ensure uppercase for getattr
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    LOG_FILE: str = os.environ.get('LOG_FILE', 'pose_detection.log') # Default log file path

    # 检测配置
    DEFAULT_ANGLE_TOLERANCE: float = float(os.environ.get('DEFAULT_ANGLE_TOLERANCE', 15.0))


# ======================== 日志配置 ========================

class RequestIdFilter(logging.Filter):
    """Add request_id to log records (compatible with non-request context)"""
    def filter(self, record):
        try:
            # flask.has_request_context and flask.g are imported at the top of the file
            if has_request_context() and hasattr(g, 'request_id'):
                record.request_id = g.request_id
            else:
                record.request_id = 'NO-REQUEST-ID'
        except RuntimeError: # Specifically catch RuntimeError if context is pushed but g is not setup
            record.request_id = 'NO-REQUEST-ID-RUNTIME-ERROR'
        except Exception: # Catch any other unexpected errors during logging
            record.request_id = 'NO-REQUEST-ID-EXCEPTION' # Ensure request_id is always set
        return True


def setup_logging(config_obj: AppConfig):
    """配置日志系统"""
    handlers = [logging.StreamHandler()] # Always log to console

    # Add file handler if LOG_FILE is specified and not in serverless env
    if not os.environ.get('SERVERLESS') and config_obj.LOG_FILE:
        try:
            # Ensure the directory for the log file exists if it's in a subdirectory
            log_dir = os.path.dirname(config_obj.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(config_obj.LOG_FILE))
        except Exception as e:
            # Use a temporary basicConfig to log this specific error if main logging fails
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"Failed to create file handler for {config_obj.LOG_FILE}: {e}")


    # Configure basicConfig for the root logger
    logging.basicConfig(
        level=getattr(logging, config_obj.LOG_LEVEL, logging.INFO), # Fallback to INFO if invalid level
        format=config_obj.LOG_FORMAT,
        handlers=handlers,
        force=True # Force re-configuration if basicConfig was called before (e.g. by a library)
    )

    # Add RequestIdFilter to all handlers of the root logger
    # This ensures that even if other libraries add handlers, they also get the request_id
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        # Check if a RequestIdFilter instance is already present
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())

    # Adjust log levels for noisy third-party libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.INFO) # Pillow can be verbose at DEBUG


# ======================== 应用初始化 ========================

config = AppConfig() # Global config instance
setup_logging(config) # Setup logging with the global config
logger = logging.getLogger(__name__) # Initialize logger for the application module

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Redis remains commented out
# redis_client = None


# ======================== 工具函数 ========================

def generate_request_id() -> str:
    """生成唯一的请求ID"""
    return str(uuid.uuid4())


def generate_cos_key(pose_id: str) -> str:
    """
    为给定的姿势ID生成COS对象键。
    格式: skeletons/{pose_id}_{timestamp_ms}_{6_char_uuid}.png
    """
    timestamp_ms = int(time.time() * 1000)
    short_uuid = uuid.uuid4().hex[:6] # Use hex for a cleaner short UUID
    return f"skeletons/{pose_id}_{timestamp_ms}_{short_uuid}.png"


def get_client_ip() -> str:
    """
    安全地获取客户端IP地址，考虑X-Forwarded-For头部。
    """
    # X-Forwarded-For can be a comma-separated list of IPs. The first one is the original client.
    if request.headers.getlist("X-Forwarded-For"):
        # Take the first IP in the list (client's IP)
        return request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    # Fallback to remote_addr if X-Forwarded-For is not present or empty
    return request.remote_addr or 'unknown_ip'


# ======================== 响应构建 ========================

class ResponseBuilder:
    """统一构建API响应的辅助类"""

    @staticmethod
    def success(score: float, skeleton_url: str, pose_id: str, **kwargs) -> Dict[str, Any]:
        """构建成功的API响应"""
        request_id_val = 'N/A'
        if has_request_context(): # Check context before accessing g
            request_id_val = getattr(g, 'request_id', 'N/A-IN-CONTEXT')

        response = {
            "code": "OK",
            "version": API_VERSION,
            "score": round(score, 2), # Ensure score is rounded
            "skeletonUrl": skeleton_url,
            "poseId": pose_id,
            "msg": "检测成功",
            "ts": int(time.time() * 1000), # Millisecond timestamp
            "requestId": request_id_val
        }
        response.update(kwargs) # Allow additional fields
        return response

    @staticmethod
    def error(code: str, message: str, pose_id: str = "", score: float = 0.0, **kwargs) -> Dict[str, Any]:
        """构建错误的API响应"""
        request_id_val = 'N/A'
        if has_request_context(): # Check context before accessing g
            request_id_val = getattr(g, 'request_id', 'N/A-IN-CONTEXT')
        
        response = {
            "code": code,
            "version": API_VERSION,
            "score": round(score, 2), # Ensure score is rounded, even for errors
            "skeletonUrl": "", # No skeleton URL on error
            "poseId": pose_id,
            "msg": message,
            "ts": int(time.time() * 1000),
            "requestId": request_id_val
        }
        response.update(kwargs)
        return response


# ======================== 中间件 ========================

@app.before_request
def before_request_hook():
    """在每个请求处理之前执行"""
    g.request_id = request.headers.get('X-Request-ID', generate_request_id())
    g.start_time = time.monotonic() # Use monotonic clock for duration calculation
    logger.info(
        f"Incoming request: {request.method} {request.full_path} from {get_client_ip()} " # Use full_path for query params
        f"Content-Type: {request.content_type}, Content-Length: {request.content_length or 0}"
    )


@app.after_request
def after_request_hook(response: Flask.response_class) -> Flask.response_class: # Type hint for response
    """在每个请求处理之后执行，用于添加头部和记录"""
    if hasattr(g, 'request_id'): # Ensure request_id was set
        response.headers['X-Request-ID'] = g.request_id
    
    if hasattr(g, 'start_time'): # Ensure start_time was set
        duration_ms = (time.monotonic() - g.start_time) * 1000 # Duration in milliseconds
        response.headers['X-Response-Time-Ms'] = f"{duration_ms:.2f}" # More precise timing
        logger.info(
            f"Request completed: {response.status_code} {request.method} {request.full_path} "
            f"Duration: {duration_ms:.2f}ms"
        )
    else: # Should not happen if before_request_hook runs
        logger.warning("g.start_time not set in after_request_hook.")

    return response


# ======================== 验证器 ========================

class FileValidator:
    """文件上传验证逻辑"""
    @staticmethod
    def validate_file_extension_and_type(file: FileStorage) -> Tuple[bool, Optional[str]]:
        """验证上传文件的扩展名和初步声明的类型"""
        if not file or not file.filename:
            return False, "未选择文件或文件名为空。"

        # Check for directory traversal attempts in filename
        if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
            logger.warning(f"Potential directory traversal attempt in filename: '{file.filename}'")
            return False, "文件名包含无效字符。"

        if '.' not in file.filename:
            return False, "文件名无效（缺少扩展名）。"

        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return False, f"不支持的文件格式 '{ext}'。请上传以下格式之一: {', '.join(sorted(list(config.ALLOWED_EXTENSIONS)))}。"

        # Basic content type check from browser (can be spoofed, Pillow validation is more robust)
        if file.content_type and not file.content_type.lower().startswith('image/'):
            logger.warning(f"文件 '{file.filename}' 的 Content-Type ({file.content_type}) 不是图片类型。将依赖Pillow进行内容验证。")
        return True, None

    @staticmethod
    def validate_image_content(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """使用Pillow库验证图片内容的有效性和安全性"""
        if not image_bytes: # Should be caught earlier, but double-check
            return False, "图片内容为空。"
        try:
            img_buffer = BytesIO(image_bytes)
            with Image.open(img_buffer) as img:
                # 1. Verify: Basic check for format and corruption
                img.verify()
                
                # 2. Re-open and Load: More thorough check, loads pixel data
                # This helps catch truncated images or some forms of "decompression bombs"
                # (though Pillow has some internal limits too).
                img_buffer.seek(0) # Important: Reset buffer pointer after verify()
                with Image.open(img_buffer) as img_load:
                    # Check for excessively large images if not caught by MAX_CONTENT_LENGTH
                    # This is a rudimentary defense against decompression bombs.
                    # A more robust solution might involve setting limits in Pillow itself if possible,
                    # or more sophisticated image analysis.
                    # Example: if img_load.size[0] * img_load.size[1] > SOME_PIXEL_LIMIT:
                    #    return False, "图片尺寸过大（像素超限）。"
                    img_load.load()
            return True, None
        except UnidentifiedImageError:
            logger.warning("Pillow: 验证时无法识别图片格式或文件已损坏。")
            return False, "无法识别的图片格式或文件已损坏。"
        except FileNotFoundError: # Should not happen with BytesIO
            logger.error("Pillow: 使用BytesIO验证图片时发生FileNotFoundError（意外）。")
            return False, "图片文件未找到（内部错误）。"
        except IOError as e: # Catch other I/O related errors like "image file is truncated"
            logger.warning(f"Pillow: 图片验证时发生IOError: {e}")
            return False, f"图片文件损坏或不完整: {e}"
        except Exception as e: # Catch any other Pillow-related errors
            logger.error(f"Pillow图片验证失败: {type(e).__name__} - {str(e)}")
            return False, f"图片内容无效或处理时出错: ({type(e).__name__})"


# ======================== 装饰器 ========================

def track_metrics(metric_name: str):
    """用于追踪特定函数执行时间的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic() # Use monotonic for duration
            try:
                result = func(*args, **kwargs)
                return result
            finally: # Ensure duration is logged even if an exception occurs in 'func'
                duration_ms = (time.monotonic() - start_time) * 1000
                if config.ENABLE_METRICS:
                    logger.info(f"Metric: '{metric_name}' execution time: {duration_ms:.2f}ms")
        return wrapper
    return decorator


# ======================== 错误处理 ========================

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e: RequestEntityTooLarge):
    """处理上传文件过大的专用错误处理器"""
    logger.warning(f"File too large: Name='{e.name}', Description='{e.description}'")
    return jsonify(ResponseBuilder.error(
        "FILE_TOO_LARGE",
        f"上传的文件大小超过系统限制 ({config.MAX_CONTENT_LENGTH // 1024 // 1024}MB)。",
        maxSizeMB=config.MAX_CONTENT_LENGTH // 1024 // 1024
    )), 413


@app.errorhandler(Exception)
def handle_generic_error(e: Exception):
    """全局未捕获异常的处理器"""
    # 记录完整的异常堆栈信息，便于调试
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled Exception: {type(e).__name__}: {str(e)}\nTraceback:\n{error_traceback}")
    
    # 生产环境中不应暴露详细的错误信息给客户端
    error_message = "服务器内部发生未知错误，请稍后重试或联系管理员。"
    if config.DEBUG: # 在调试模式下，可以返回更详细的错误信息
        error_message = f"服务器内部错误: {type(e).__name__} - {str(e)}"
        
    return jsonify(ResponseBuilder.error("INTERNAL_SERVER_ERROR", error_message)), 500


# ======================== API路由 ========================

@app.route('/api/detect-pose-file', methods=['POST'])
@track_metrics('detect_pose_file_route') # Metric for the entire route handling
def detect_pose_file_route():
    """主要的姿势检测API端点，处理文件上传和姿势分析"""
    pose_id = request.form.get('poseId', '').strip()

    if not pose_id:
        logger.warning("请求缺少 'poseId' 参数。")
        return jsonify(ResponseBuilder.error("MISSING_PARAMETER", "必需的 'poseId' 参数缺失。")), 400

    if pose_id not in get_supported_pose_ids():
        logger.warning(f"Invalid poseId received: '{pose_id}'")
        return jsonify(
            ResponseBuilder.error(
                "INVALID_POSE_ID_SPECIFIED",
                "指定的体式ID不存在。",
                pose_id=pose_id,
                supportedPoses=get_supported_pose_ids()
            )
        ), 400

    if 'file' not in request.files:
        logger.warning("请求中未包含 'file' 部分。")
        return jsonify(ResponseBuilder.error("MISSING_FILE", "上传的文件参数 'file' 缺失。", pose_id=pose_id)), 400

    uploaded_file = request.files['file']

    is_ext_valid, ext_error_msg = FileValidator.validate_file_extension_and_type(uploaded_file)
    if not is_ext_valid:
        logger.warning(f"文件扩展名或类型验证失败: {ext_error_msg} (文件名: '{uploaded_file.filename}')")
        return jsonify(ResponseBuilder.error("INVALID_FILE_TYPE", ext_error_msg, pose_id=pose_id)), 400

    try:
        image_bytes = uploaded_file.read()
        if not image_bytes:
            logger.warning(f"上传的文件 '{uploaded_file.filename}' 内容为空。")
            return jsonify(ResponseBuilder.error("EMPTY_FILE", "上传的文件内容为空。", pose_id=pose_id)), 400
    except Exception as e:
        logger.error(f"读取上传文件 '{uploaded_file.filename}' 失败: {e}")
        return jsonify(ResponseBuilder.error("FILE_READ_ERROR", "读取上传文件内容失败。", pose_id=pose_id)), 500 # 500 for server-side read issue
    
    is_content_valid, content_error_msg = FileValidator.validate_image_content(image_bytes)
    if not is_content_valid:
        logger.warning(f"图片内容验证失败: {content_error_msg} (文件名: '{uploaded_file.filename}')")
        return jsonify(ResponseBuilder.error("INVALID_IMAGE_CONTENT", content_error_msg, pose_id=pose_id)), 400

    logger.info(f"成功接收并初步验证图片: 文件名='{uploaded_file.filename}', 大小={len(image_bytes)} bytes, poseId='{pose_id}'")

    detection_options_obj: Optional[DetectionConfig] = None
    options_str = request.form.get('options')
    if options_str:
        try:
            import json
            options_data = json.loads(options_str)
            angle_tolerance_val = options_data.get('angleTolerance')
            if angle_tolerance_val is not None:
                detection_options_obj = DetectionConfig(angle_tolerance_excellent=float(angle_tolerance_val))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"解析 'options' JSON参数失败 (poseId '{pose_id}'): {e}. 将使用默认检测配置。")
            # Consider if malformed options should be a hard error or a soft warning.
            # For now, it's a soft warning, proceeding with defaults.

    processing_times = {} # To store timing for different stages
    try:
        # Perform Pose Detection
        detect_time_start = time.monotonic()
        score, skeleton_buffer = detect_pose(image_bytes, pose_id, detection_options_obj)
        processing_times["detection_ms"] = (time.monotonic() - detect_time_start) * 1000
        logger.info(f"姿势检测完成 (poseId='{pose_id}'): 用时={processing_times['detection_ms']:.2f}ms, 分数={score}")

        # Upload Skeleton to COS
        cos_key = generate_cos_key(pose_id)
        upload_time_start = time.monotonic()
        skeleton_url = upload_buffer(skeleton_buffer, cos_key) # From cos_uploader
        processing_times["upload_ms"] = (time.monotonic() - upload_time_start) * 1000
        logger.info(f"骨架图上传完成 (poseId='{pose_id}'): 用时={processing_times['upload_ms']:.2f}ms, URL={skeleton_url}")

        # Build and return success response
        # Total processing time for this specific block, not entire request yet
        block_total_time_ms = processing_times["detection_ms"] + processing_times["upload_ms"]
        logger.info(f"Pose detection and upload for '{pose_id}' block total time: {block_total_time_ms:.2f}ms")

        return jsonify(ResponseBuilder.success(
            score=score,
            skeleton_url=skeleton_url,
            pose_id=pose_id,
            processingTimeMs=processing_times # Use more descriptive key
        )), 200

    except NoKeypointError as e:
        logger.info(f"未检测到关键点 (poseId='{pose_id}'): {e}")
        return jsonify(ResponseBuilder.error("NO_KEYPOINT_DETECTED", str(e), pose_id=pose_id, details=getattr(e, 'details', None))), 200
    except InvalidPoseError as e:
        logger.warning(f"无效的姿势ID (poseId='{pose_id}'): {e}")
        return jsonify(ResponseBuilder.error("INVALID_POSE_ID_SPECIFIED", str(e), pose_id=pose_id, supportedPoses=getattr(e, 'details', {}).get('supported_poses', []))), 400
    except SkeletonDrawError as e:
        logger.error(f"绘制骨架图时出错 (poseId='{pose_id}'): {e}\n{traceback.format_exc()}")
        return jsonify(ResponseBuilder.error("SKELETON_DRAWING_FAILED", f"骨架图生成失败: {str(e)}", pose_id=pose_id)), 500
    except PoseDetectionError as e:
        logger.error(f"姿势检测过程中出错 (poseId='{pose_id}'): {e}\n{traceback.format_exc()}")
        return jsonify(ResponseBuilder.error("POSE_DETECTION_PROCESSING_ERROR", f"姿势检测处理失败: {str(e)}", pose_id=pose_id)), 500
    except CosUploadError as e:
        logger.error(f"COS上传骨架图失败 (poseId='{pose_id}'): {e}\n{traceback.format_exc()}")
        return jsonify(ResponseBuilder.error("COS_SKELETON_UPLOAD_FAILED", f"骨架图上传至云存储失败: {str(e)}", pose_id=pose_id)), 500
    except Exception as e: # Catch-all for other unexpected errors during this specific processing block
        logger.error(f"处理姿势检测/上传时发生意外错误 (poseId='{pose_id}'): {traceback.format_exc()}")
        return jsonify(ResponseBuilder.error("UNEXPECTED_PROCESSING_ERROR", "处理您的请求时发生意外错误。", pose_id=pose_id)), 500


@app.route('/api/detect-pose-batch', methods=['POST'])
@track_metrics('detect_pose_batch_route')
def detect_pose_batch_route():
    """批量姿势检测接口（占位实现）"""
    logger.info("Access to /api/detect-pose-batch (Not Implemented)")
    return jsonify(ResponseBuilder.error("NOT_YET_IMPLEMENTED", "批量检测功能当前不可用，敬请期待。")), 501


@app.route('/api/detect-pose-async', methods=['POST'])
@track_metrics('detect_pose_async_route')
def detect_pose_async_route():
    """异步姿势检测接口（占位实现）"""
    logger.info(f"Access to /api/detect-pose-async (Async enabled: {config.ENABLE_ASYNC})")
    if not config.ENABLE_ASYNC:
        return jsonify(ResponseBuilder.error("ASYNC_FEATURE_DISABLED", "异步处理功能当前未启用。")), 501
    return jsonify(ResponseBuilder.error("NOT_YET_IMPLEMENTED", "异步检测功能当前不可用，敬请期待。")), 501


@app.route('/health', methods=['GET'])
@track_metrics('health_check_route') # Also track metrics for health check
def health_check_route():
    """健康检查端点，报告API及其依赖项的状态"""
    health_status: Dict[str, Any] = { # Define type for better clarity
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": int(time.time() * 1000), # Use ms timestamp
        "service_name": "pose-detection-api", # More descriptive key
        "checks": [] # List of checks
    }
    overall_http_status = 200 # HTTP status code

    # Check 1: COS Storage Connection
    if config.COS_BUCKET:
        cos_check = {"name": "COS_Storage_Connectivity", "status": "healthy", "details": ""}
        try:
            cos_accessible = check_cos_connection() # From cos_uploader
            if not cos_accessible:
                cos_check["status"] = "unhealthy"
                cos_check["details"] = f"无法访问配置的COS存储桶 '{config.COS_BUCKET}' (区域: {config.COS_REGION})。"
                health_status["status"] = "degraded" # Service is degraded if a dependency fails
                overall_http_status = 503 # Service Unavailable
            else:
                cos_check["details"] = f"成功连接到COS存储桶 '{config.COS_BUCKET}'。"
        except Exception as e:
            logger.error(f"健康检查: COS连接测试异常: {e}")
            cos_check["status"] = "unhealthy"
            cos_check["details"] = f"测试COS连接时发生错误: {str(e)}"
            health_status["status"] = "degraded"
            overall_http_status = 503
        health_status["checks"].append(cos_check)

    # Check 2: Pose Detector Module Functionality
    detector_check = {"name": "Pose_Detector_Module", "status": "healthy", "details": ""}
    try:
        pose_ids = get_supported_pose_ids()
        detector_check["details"] = f"模块可操作，支持 {len(pose_ids)} 种姿势。"
        if not pose_ids:
             logger.warning("健康检查: 姿势定义列表为空。")
    except Exception as e:
        logger.error(f"健康检查: 姿势检测模块异常: {e}")
        detector_check["status"] = "unhealthy" # Core functionality might be broken
        detector_check["details"] = f"测试姿势检测模块时发生错误: {str(e)}"
        health_status["status"] = "unhealthy" # If core module fails, service is unhealthy
        overall_http_status = 503
    health_status["checks"].append(detector_check)
        
    # Add other dependency checks here (e.g., database, other microservices)

    return jsonify(health_status), overall_http_status


@app.route('/api/info', methods=['GET'])
@track_metrics('api_info_route')
def api_info_route():
    """提供关于API的元数据信息"""
    # cache_enabled_status = 'redis_client' in globals() and redis_client is not None # Redis commented out
    return jsonify({
        "api_name": "Pose Detection API",
        "api_version": API_VERSION,
        "description": "提供人体姿势检测、评分及骨架图可视化功能。",
        "documentation_url": "/swagger-ui" if config.DEBUG else "请查阅API文档或联系技术支持获取详细信息。", # Example
        "contact_info": {"support_email": "tech-support@example.com"}, # Example
        "endpoints_overview": [
            {"path": "/api/detect-pose-file", "method": "POST", "summary": "从图片文件检测姿势。"},
            {"path": "/api/supported-poses", "method": "GET", "summary": "列出所有支持的姿势类型。"},
            {"path": "/health", "method": "GET", "summary": "执行API及其依赖的健康检查。"},
            {"path": "/api/info", "method": "GET", "summary": "获取API元数据信息。"}
        ],
        "system_capabilities": {
            "max_upload_size_mb": config.MAX_CONTENT_LENGTH // (1024 * 1024),
            "supported_image_formats": sorted(list(config.ALLOWED_EXTENSIONS)),
            # "caching_feature_enabled": cache_enabled_status,
            "asynchronous_processing_support": config.ENABLE_ASYNC,
            "cdn_for_results_enabled": bool(config.COS_CDN_DOMAIN)
        }
    })


@app.route('/api/supported-poses', methods=['GET'])
@track_metrics('list_supported_poses_route')
def list_supported_poses_route():
    """获取API支持的所有姿势类型列表"""
    try:
        poses_dict = get_supported_poses()
        poses_list = list(poses_dict.values())
        return jsonify({
            "code": "OK",
            "version": API_VERSION,
            "poses": poses_list,
            "count": len(poses_list),
            "ts": int(time.time() * 1000),
            "requestId": getattr(g, 'request_id', 'N/A')
        })
    except Exception as e:
        logger.error(f"获取支持的姿势列表失败: {e}\n{traceback.format_exc()}")
        return jsonify(ResponseBuilder.error("SUPPORTED_POSES_FETCH_ERROR", "获取支持的姿势列表时发生内部错误。")), 500


# ======================== 启动函数 ========================

def create_flask_app() -> Flask:
    """
    应用工厂函数，用于创建和配置Flask应用实例。
    (目前应用实例是全局的，此函数主要用于遵循工厂模式或未来扩展)
    """
    # If app were configured here (e.g., blueprints, extensions), this would be more complex.
    # For now, it simply returns the global 'app' instance.
    logger.info("Flask app instance created/retrieved via factory.")
    return app


if __name__ == '__main__':
    # 确保在直接运行脚本时，日志已配置
    # (如果通过WSGI服务器如Gunicorn启动，日志通常由服务器或启动脚本配置)
    if not logging.getLogger().hasHandlers():
        print("运行 __main__: 为独立执行模式配置基础日志...")
        setup_logging(config)

    logger.info(f"启动姿势检测API v{API_VERSION}，配置: {config}")
    
    # 生产环境部署提示:
    # logger.info("生产环境建议使用 Gunicorn 或 uWSGI 等WSGI服务器启动应用。")
    # logger.info(f"例如: gunicorn --workers 4 --bind 0.0.0.0:{config.PORT} 'app:create_flask_app()'")

    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=config.DEBUG, # debug=True 会启用Flask的调试器和重载器
        threaded=True # Flask开发服务器使用线程处理并发请求 (不适用于生产)
    )
