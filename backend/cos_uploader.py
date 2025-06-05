# -*- coding: utf-8 -*-
"""
cos_uploader.py (Optimized)
---------------------------
一个经过优化的腾讯云COS上传工具模块，专为高性能、高并发的Web服务设计。

核心特性:
- **面向对象设计**: 将功能封装在`CosUploader`类中，便于实例化和管理多个COS配置。
- **内存流操作**: 全程使用`BytesIO`在内存中处理文件流，避免磁盘I/O，适用于云原生环境。
- **线程安全**: 客户端初始化采用线程锁，确保在WSGI等多线程环境下安全可靠。
- **高性能并发**: 批量上传使用`ThreadPoolExecutor`实现并发，显著提升处理速度。
- **灵活的权限控制**: 支持上传为公有读对象，或私有对象，并能为私有对象生成预签名访问URL。
- **健壮的异常处理**: 定义了清晰的异常层次，便于上层应用统一捕获和处理。
- **可配置与可监控**: 支持从环境变量或构造函数参数初始化，并提供健康检查和状态统计方法。
"""

import os
import logging
import threading
from typing import Union, Optional, List, Dict, Any
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# 建议在您的应用入口处配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# 注意: 为避免与 app.py 中的日志配置冲突，这里的通用日志配置可以注释掉或调整
# 如果此模块作为独立脚本运行或测试，可以取消注释上面的日志配置
logger = logging.getLogger(__name__)


# 尝试导入腾讯云COS SDK
try:
    from qcloud_cos import CosConfig, CosS3Client
    from qcloud_cos.cos_exception import CosException
except ImportError:
    # 这个异常应该在模块加载时立即抛出，以便尽早发现依赖问题
    # logger.error("腾讯云COS Python SDK未安装。请运行: pip install -U cos-python-sdk-v5")
    raise ImportError("请先安装腾讯云COS Python SDK: pip install -U cos-python-sdk-v5")


class CosUploadError(Exception):
    """COS上传相关操作的通用异常"""
    pass


class CosConfigError(CosUploadError):
    """COS配置错误异常"""
    pass


class CosUploader:
    """
    腾讯云对象存储（COS）上传与管理工具类。

    该类封装了COS的常用操作，如上传、删除、检查存在性、生成预签名URL等。
    它被设计为线程安全的，并为批量操作提供了并发支持。
    """
    _client_lock = threading.Lock()

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        bucket: Optional[str] = None,
        cdn_domain: Optional[str] = None,
        timeout: int = 60,
        scheme: str = 'https'
    ):
        """
        初始化COS上传器。

        参数优先从构造函数传入，如果为None，则尝试从环境变量中读取。

        Args:
            secret_id (str, optional): 腾讯云SecretId. 默认: os.environ.get('COS_SECRET_ID').
            secret_key (str, optional): 腾讯云SecretKey. 默认: os.environ.get('COS_SECRET_KEY').
            region (str, optional): COS所在地域. 默认: os.environ.get('COS_REGION', 'ap-guangzhou').
            bucket (str, optional): COS存储桶名称. 默认: os.environ.get('COS_BUCKET').
            cdn_domain (str, optional): 自定义CDN域名，用于构建URL. 默认: os.environ.get('COS_CDN_DOMAIN', '').
            timeout (int, optional): 请求超时时间（秒）. 默认: 60.
            scheme (str, optional): 使用的协议 ('http' or 'https'). 默认: 'https'.

        Raises:
            CosConfigError: 当必需的配置（SecretId, SecretKey, Bucket）缺失时抛出。
        """
        self.secret_id = secret_id or os.environ.get('COS_SECRET_ID')
        self.secret_key = secret_key or os.environ.get('COS_SECRET_KEY')
        self.region = region or os.environ.get('COS_REGION', 'ap-guangzhou') # 默认地域
        self.bucket = bucket or os.environ.get('COS_BUCKET')
        self.cdn_domain = cdn_domain or os.environ.get('COS_CDN_DOMAIN', '')
        self.timeout = timeout
        self.scheme = scheme

        self._client: Optional[CosS3Client] = None
        self._validate_config() # 初始化时即校验配置

    def _validate_config(self):
        """验证必需的配置是否存在"""
        missing_vars = []
        if not self.secret_id:
            missing_vars.append('COS_SECRET_ID or secret_id parameter')
        if not self.secret_key:
            missing_vars.append('COS_SECRET_KEY or secret_key parameter')
        if not self.bucket:
            missing_vars.append('COS_BUCKET or bucket parameter')
        if not self.region: # region 也是必需的
            missing_vars.append('COS_REGION or region parameter')

        if missing_vars:
            raise CosConfigError(f"缺少COS配置: {', '.join(missing_vars)}")

    def _get_client(self) -> CosS3Client:
        """
        获取或初始化COS客户端实例（线程安全的单例模式，针对当前Uploader实例）。

        Returns:
            CosS3Client: COS客户端实例。

        Raises:
            CosConfigError: 当客户端初始化失败时抛出。
        """
        if self._client is None:
            with self._client_lock: # 确保多线程环境下客户端只初始化一次
                if self._client is None: # 双重检查锁定
                    try:
                        # 再次确认配置在此处仍然有效，因为环境变量可能在运行时被修改
                        # 但通常情况下，初始化时已经校验过一次
                        current_secret_id = self.secret_id or os.environ.get('COS_SECRET_ID')
                        current_secret_key = self.secret_key or os.environ.get('COS_SECRET_KEY')
                        current_region = self.region or os.environ.get('COS_REGION')

                        if not all([current_secret_id, current_secret_key, current_region]):
                            raise CosConfigError("COS客户端初始化时发现关键配置缺失。")

                        config = CosConfig(
                            Region=current_region,
                            SecretId=current_secret_id,
                            SecretKey=current_secret_key,
                            Timeout=self.timeout,
                            Scheme=self.scheme
                        )
                        self._client = CosS3Client(config)
                        logger.info(f"COS client initialized for region: {current_region}, bucket: {self.bucket}")
                    except CosException as e: #捕获更具体的COS SDK异常
                        raise CosConfigError(f"COS客户端初始化失败 (CosException): {str(e)}")
                    except Exception as e:
                        # 可以记录更详细的错误信息，如 e.__class__.__name__
                        raise CosConfigError(f"COS客户端初始化时发生未知错误: {str(e)}")
        return self._client

    def upload_buffer(
        self,
        buffer: BytesIO,
        key: str,
        content_type: str = 'application/octet-stream',
        acl: str = 'public-read', # 默认公有读
        cache_control: str = 'max-age=31536000' # 默认缓存一年
    ) -> str:
        """
        上传BytesIO缓冲区中的数据到COS。

        Args:
            buffer (BytesIO): 包含数据的BytesIO对象。
            key (str): COS中的对象键 (e.g., 'images/my-photo.jpg').
            content_type (str, optional): 文件的MIME类型. 默认 'application/octet-stream'.
            acl (str, optional): 访问控制列表 ('private', 'public-read'). 默认 'public-read'.
            cache_control (str, optional): 缓存控制头. 默认 'max-age=31536000' (1年).

        Returns:
            str: 对象的公网可访问URL（如果使用CDN则为CDN URL，否则为COS源站URL）。

        Raises:
            TypeError: 如果buffer不是BytesIO类型。
            ValueError: 如果key为空或buffer为空。
            CosUploadError: 上传失败或COS服务异常时抛出。
        """
        if not isinstance(buffer, BytesIO):
            raise TypeError(f"无效的缓冲区类型: {type(buffer).__name__}, 需要 BytesIO")

        if not key or not key.strip(): # 确保key不为空或仅包含空白字符
            raise ValueError("COS对象键 (key) 不能为空")

        try:
            client = self._get_client()

            buffer.seek(0) # 确保从头读取
            buffer_size = buffer.getbuffer().nbytes
            if buffer_size == 0:
                raise ValueError("缓冲区数据为空，无法上传")

            logger.info(f"Uploading to COS: bucket={self.bucket}, key={key}, size={buffer_size} bytes, ACL={acl}, ContentType={content_type}")

            # COS SDK的put_object方法可以处理BytesIO对象
            response = client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buffer,         # 直接传递BytesIO对象
                ContentType=content_type,
                ACL=acl,
                CacheControl=cache_control,
                # 可以添加其他参数，如 Metadata, StorageClass 等
            )

            # 检查上传是否成功，ETag通常表示成功
            if not response or 'ETag' not in response:
                # 可以尝试获取更详细的错误信息，如果SDK支持的话
                raise CosUploadError("上传失败：COS未返回ETag或有效响应")

            # 构建URL
            if self.cdn_domain:
                # 确保CDN域名不以斜杠结尾，key不以斜杠开头（除非是根目录对象）
                url = f"{self.scheme}://{self.cdn_domain.rstrip('/')}/{key.lstrip('/')}"
            else:
                # 使用SDK提供的方法获取标准对象URL，这通常是推荐的做法
                # 需要确认 get_object_url 是否考虑了 scheme (http/https)
                # CosS3Client.get_object_url 默认使用config中的Scheme
                url = client.get_object_url(Bucket=self.bucket, Key=key)

            logger.info(f"Upload successful: {url}, ETag: {response.get('ETag')}")
            return url

        except CosException as e:
            # 这是COS SDK抛出的特定异常
            error_msg = f"COS服务异常: Code={e.get_error_code()}, Message={e.get_error_msg()}, RequestId={e.get_request_id()}"
            logger.error(error_msg)
            raise CosUploadError(error_msg) from e # 保留原始异常链
        except (ValueError, TypeError) as e: # 参数验证错误，直接抛出
            logger.error(f"参数错误: {str(e)}")
            raise
        except Exception as e:
            # 捕获其他潜在错误，例如网络问题等
            error_msg = f"上传过程中发生未知错误: {e.__class__.__name__} - {str(e)}"
            logger.error(error_msg)
            raise CosUploadError(error_msg) from e

    def get_presigned_url(self, key: str, method: str = 'GET', expires_in_seconds: int = 3600) -> str:
        """
        为COS中的对象生成一个预签名的临时访问URL。

        Args:
            key (str): 要生成URL的对象键。
            method (str, optional): HTTP方法 ('GET', 'PUT', 'DELETE', etc.). 默认 'GET'.
            expires_in_seconds (int, optional): URL的有效时间（秒）。默认 3600 (1小时).

        Returns:
            str: 预签名的URL。

        Raises:
            CosUploadError: 生成URL失败时抛出。
        """
        if not key or not key.strip():
            raise ValueError("COS对象键 (key) 不能为空")

        try:
            client = self._get_client()
            url = client.get_presigned_url(
                Bucket=self.bucket,
                Key=key,
                Method=method, # 指定HTTP方法
                Expired=expires_in_seconds # SDK的Expired参数单位是秒
            )
            logger.info(f"Generated presigned URL for key '{key}' (method: {method}), expires in {expires_in_seconds}s.")
            return url
        except CosException as e:
            error_msg = f"生成预签名URL时COS服务异常: Code={e.get_error_code()}, Message={e.get_error_msg()}"
            logger.error(error_msg)
            raise CosUploadError(error_msg) from e
        except Exception as e:
            error_msg = f"生成预签名URL失败: {str(e)}"
            logger.error(error_msg)
            raise CosUploadError(error_msg) from e

    def delete_object(self, key: str) -> bool:
        """
        删除COS中的单个对象。

        Args:
            key (str): 要删除的对象键。

        Returns:
            bool: 删除成功返回True。

        Raises:
            CosUploadError: 删除操作失败时抛出。
        """
        if not key or not key.strip():
            raise ValueError("COS对象键 (key) 不能为空")
        try:
            client = self._get_client()
            # delete_object SDK调用通常在成功时不返回内容，或返回特定响应
            # 需要检查SDK文档确认成功删除的判断依据
            client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Object deleted successfully from bucket '{self.bucket}': {key}")
            return True
        except CosException as e:
            # 如果对象不存在，某些SDK配置下可能不会抛异常，或抛特定异常
            # 需要根据实际情况调整错误处理逻辑
            error_msg = f"删除对象时COS服务异常: Code={e.get_error_code()}, Message={e.get_error_msg()}"
            logger.error(error_msg)
            raise CosUploadError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to delete object {key}: {str(e)}")
            raise CosUploadError(f"删除对象失败: {str(e)}") from e


    def check_object_exists(self, key: str) -> bool:
        """
        检查COS中的对象是否存在。

        Args:
            key (str): 对象键。

        Returns:
            bool: 存在返回True，不存在或出错返回False。
                  注意：某些权限配置下，即使对象存在也可能因无权限访问而返回False。
        """
        if not key or not key.strip():
            logger.warning("检查对象存在性时，key为空。")
            return False
        try:
            client = self._get_client()
            # object_exists 是推荐的方法，它通常比 head_object 更直接
            exists = client.object_exists(Bucket=self.bucket, Key=key)
            logger.debug(f"Object '{key}' in bucket '{self.bucket}' exists: {exists}")
            return exists
        except CosException as e: # 更具体的异常捕获
            # 例如，如果因为权限问题无法检查，也应视为检查失败
            logger.warning(f"检查对象 '{key}' 存在性时COS服务异常: {e.get_error_code()} - {e.get_error_msg()}")
            return False # 或者根据错误代码决定是否重试或抛出
        except Exception as e:
            logger.error(f"检查对象 '{key}' 存在性时发生未知错误: {str(e)}")
            return False # 通常未知错误意味着无法确认存在性


    def batch_upload_buffers(self, files: List[Dict[str, Any]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        使用线程池并发批量上传多个BytesIO缓冲区。

        Args:
            files (List[Dict]): 文件信息列表。
                每个字典应包含:
                - "buffer": BytesIO对象 (必需)
                - "key": str, 对象键 (必需)
                - "content_type": (optional) str, MIME类型
                - "acl": (optional) str, 访问权限
                - "cache_control": (optional) str, 缓存控制头
            max_workers (int, optional): 线程池的最大工作线程数。默认 None (Python会自动选择合适数量，通常是CPU核心数*5)。

        Returns:
            list: 上传结果列表，每个元素为 {"key": str, "url": str | None, "success": bool, "error": str | None}。
        """
        if not files:
            return []

        results = []
        # 如果未指定max_workers, ThreadPoolExecutor会使用默认值
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file_info = {}
            for file_info in files:
                if not isinstance(file_info.get("buffer"), BytesIO) or not file_info.get("key"):
                    logger.warning(f"Skipping invalid file_info in batch: {file_info}")
                    results.append({
                        "key": file_info.get("key", "UNKNOWN_KEY"),
                        "url": None,
                        "success": False,
                        "error": "Invalid file_info (missing buffer or key)"
                    })
                    continue

                future = executor.submit(
                    self.upload_buffer,
                    buffer=file_info["buffer"],
                    key=file_info["key"],
                    content_type=file_info.get("content_type", 'application/octet-stream'),
                    acl=file_info.get("acl", 'public-read'),
                    cache_control=file_info.get("cache_control", 'max-age=31536000')
                )
                future_to_file_info[future] = file_info["key"]

            for future in as_completed(future_to_file_info):
                key = future_to_file_info[future]
                result_item = {"key": key, "success": False, "url": None, "error": None}
                try:
                    url = future.result() # 获取上传结果
                    result_item["success"] = True
                    result_item["url"] = url
                except Exception as e:
                    error_msg = f"批量上传文件 '{key}' 失败: {str(e)}"
                    logger.error(error_msg)
                    result_item["error"] = str(e) # 记录错误信息
                results.append(result_item)
        logger.info(f"Batch upload completed for {len(files)} files. Successes: {sum(1 for r in results if r['success'])}. Failures: {sum(1 for r in results if not r['success'])}.")
        return results


    def get_health_status(self) -> Dict[str, Any]:
        """
        获取关于COS连接和配置的健康检查状态。
        尝试轻量级操作 (如 head_bucket) 来验证连接。

        Returns:
            dict: 包含配置和可访问性信息的字典。
        """
        status_details: Dict[str, Any] = {
            "config_summary": { #提供更清晰的配置摘要
                "region": self.region,
                "bucket": self.bucket,
                "cdn_enabled": bool(self.cdn_domain),
                "cdn_domain": self.cdn_domain if self.cdn_domain else "N/A"
            },
            "client_initialized": False, # 标记客户端是否已初始化
            "bucket_accessible": False, # 标记存储桶是否可访问
            "error_message": None # 存储遇到的错误信息
        }

        try:
            client = self._get_client() # 这会尝试初始化客户端
            status_details["client_initialized"] = self._client is not None

            if status_details["client_initialized"]:
                # head_bucket 是一个轻量级的检查操作，验证桶是否存在且可访问
                client.head_bucket(Bucket=self.bucket)
                status_details["bucket_accessible"] = True
                logger.info(f"Health check: Successfully accessed bucket '{self.bucket}'.")
            else:
                # 如果客户端未初始化，通常意味着配置问题，在_get_client中已抛出异常
                status_details["error_message"] = "Client could not be initialized (check config errors)."
                logger.warning("Health check: Client not initialized.")

        except CosConfigError as e: #捕获配置相关的特定异常
            status_details["error_message"] = f"Configuration Error: {str(e)}"
            logger.error(f"Health check failed due to configuration error: {str(e)}")
        except CosException as e:
            status_details["error_message"] = f"COS Service Error: Code={e.get_error_code()}, Message={e.get_error_msg()}"
            logger.error(f"Health check: COS service error while accessing bucket '{self.bucket}': {str(e)}")
        except Exception as e:
            status_details["error_message"] = f"Unexpected Error: {str(e)}"
            logger.error(f"Health check: Unexpected error while accessing bucket '{self.bucket}': {str(e)}")

        return status_details

# ==============================================================================
# 兼容旧版函数式调用的全局单例和导出函数
# ==============================================================================
try:
    # 尝试初始化一个默认的 uploader 实例
    # 这将使用环境变量中的配置
    _default_uploader = CosUploader()
except CosConfigError as e:
    # 如果默认配置缺失或无效，记录警告但允许模块加载
    # 调用具体函数时，如果_default_uploader为None，则会再次尝试初始化或抛出配置错误
    logger.warning(f"无法初始化默认COS Uploader实例: {e}. "
                   f"请确保COS环境变量已正确设置，或在使用函数式调用前手动配置。")
    _default_uploader = None # 设置为None，以便后续函数可以检查
except ImportError:
    # ImportError 已经在模块顶部处理，但作为防御性措施也在这里捕获
    logger.critical("COS SDK未安装，cos_uploader模块无法正常工作。")
    _default_uploader = None


def get_default_uploader() -> CosUploader:
    """获取或初始化默认的COS Uploader实例"""
    global _default_uploader
    if _default_uploader is None:
        try:
            _default_uploader = CosUploader() # 尝试再次初始化
        except (CosConfigError, ImportError) as e: # 捕获可能在此处发生的错误
             logger.error(f"尝试获取默认Uploader时初始化失败: {e}")
             raise CosConfigError(f"默认COS Uploader配置错误或依赖缺失: {e}") from e
    return _default_uploader

def upload_buffer(
    buffer: BytesIO,
    key: str,
    content_type: str = 'application/octet-stream',
    acl: str = 'public-read',
    cache_control: str = 'max-age=31536000'
) -> str:
    """
    [兼容函数] 上传BytesIO流到COS，使用默认的Uploader实例。
    """
    uploader = get_default_uploader()
    return uploader.upload_buffer(buffer, key, content_type, acl, cache_control)

def check_cos_connection() -> bool:
    """
    [兼容函数] 检查与COS的连接（基于默认Uploader实例的健康状态）。
    返回True表示存储桶可访问，否则返回False。
    """
    try:
        uploader = get_default_uploader()
        health_status = uploader.get_health_status()
        return health_status.get("bucket_accessible", False)
    except (CosConfigError, CosUploadError): # 如果获取uploader或健康检查失败
        return False
    except Exception as e: # 捕获其他意外错误
        logger.error(f"检查COS连接时发生意外错误: {e}")
        return False


# 使用 __all__ 来明确指定模块希望导出的公共API
__all__ = [
    'CosUploader',      # 主要的类
    'CosUploadError',   # 自定义异常
    'CosConfigError',   # 自定义配置异常
    'upload_buffer',    # 兼容函数
    'check_cos_connection' # 兼容函数
    # 根据需要可以添加其他希望导出的类或函数
    # 'get_presigned_url' # 如果也需要函数式调用
    # 'delete_object'
    # 'check_object_exists'
    # 'batch_upload_buffers'
]


# ==============================================================================
# 示例用法 (Example Usage)
# ==============================================================================
if __name__ == '__main__':
    # 配置基本日志以便在直接运行时看到输出
    if not logger.handlers: # 避免重复添加处理器
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("COS Uploader Module - Example Usage")
    # 在运行此示例前，请确保已设置以下环境变量:
    # export COS_SECRET_ID="YOUR_SECRET_ID"
    # export COS_SECRET_KEY="YOUR_SECRET_KEY"
    # export COS_BUCKET="your-bucket-name-125xxxxxxx"
    # export COS_REGION="ap-guangzhou"  # (或者其他区域)
    # export COS_CDN_DOMAIN="your.cdn.domain.com" # (可选)

    # 1. 使用兼容函数 (依赖默认_default_uploader实例)
    print("\n--- Testing Compatibility Functions (using default uploader) ---")
    try:
        connection_ok = check_cos_connection()
        print(f"Default COS connection check: {'OK' if connection_ok else 'Failed'}")

        if connection_ok:
            # 创建一个内存中的PNG图片（仅为示例，实际应为真实数据）
            png_data_compat = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'
            png_buffer_compat = BytesIO(png_data_compat)
            compat_key = "example/compat_image.png"
            
            compat_url = upload_buffer(
                buffer=png_buffer_compat,
                key=compat_key,
                content_type='image/png'
            )
            print(f"Compatibility function upload_buffer successful: {compat_url}")
            # 清理兼容函数上传的文件
            if _default_uploader: # 确保实例存在
                _default_uploader.delete_object(compat_key)
                print(f"Deleted compatibility test object: {compat_key}")
        else:
            print("Skipping compatibility upload test due to connection failure.")

    except (CosUploadError, CosConfigError) as e:
        print(f"Error during compatibility function test: {e}")
    except Exception as e:
        print(f"Unexpected error during compatibility function test: {e}")


    # 2. 实例化和使用CosUploader类 (推荐方式)
    print("\n--- Testing CosUploader Class (recommended) ---")
    try:
        # 显式初始化Uploader，如果环境变量未设置，这里会因_validate_config失败
        # 或者你可以直接传入参数覆盖环境变量：
        # uploader_instance = CosUploader(
        #     secret_id="YOUR_ID_HERE_IF_DIFFERENT",
        #     secret_key="YOUR_KEY_HERE_IF_DIFFERENT",
        #     region="ap-shanghai", # 例如，指定不同区域
        #     bucket="your-other-bucket-125xxxxxxx"
        # )
        uploader_instance = CosUploader() # 使用环境变量
        
        # 健康检查
        print("--- Health Check (uploader_instance) ---")
        health = uploader_instance.get_health_status()
        print(f"Instance Health Status: {health}")
        if not health.get('bucket_accessible'):
            print("错误：通过 uploader_instance 无法访问存储桶，请检查配置和网络。")
            # 如果健康检查失败，后续操作可能无意义，可以选择退出
            # exit(1) # 在脚本中可以这样，但在模块中通常不直接退出

        # 上传单个公有读文件
        if health.get('bucket_accessible'):
            print("\n--- Uploading Single Public File (uploader_instance) ---")
            png_data_instance = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00\xf3\x7fL\xd5\x00\x00\x00\x0cIDATx\x9cc`d`\x00\x00\x00\x04\x00\x01\x0e\x12\xbf\x0e\x00\x00\x00\x00IEND\xaeB`\x82' # 稍作修改以示区别
            png_buffer_instance = BytesIO(png_data_instance)
            instance_key = "example/instance_image.png"
            
            public_url_instance = uploader_instance.upload_buffer(
                buffer=png_buffer_instance,
                key=instance_key,
                content_type='image/png',
                acl='public-read'
            )
            print(f"Instance file uploaded successfully: {public_url_instance}")

            # 检查文件是否存在
            exists_instance = uploader_instance.check_object_exists(instance_key)
            print(f"Does '{instance_key}' exist via instance? {exists_instance}")

            # 清理示例文件
            print("\n--- Cleaning up instance created object ---")
            if uploader_instance.delete_object(instance_key):
                print(f"Deleted instance test object: {instance_key}")
            else:
                print(f"Failed to delete instance test object: {instance_key}")
        else:
            print("Skipping instance upload test due to health check failure.")

    except CosConfigError as e:
        print(f"Configuration Error for uploader_instance: {e}")
    except CosUploadError as e:
        print(f"COS Upload Error for uploader_instance: {e}")
    except Exception as e: #捕获其他任何意外错误
        print(f"An unexpected error occurred with uploader_instance: {type(e).__name__} - {e}")

    logger.info("Example Usage Finished.")
