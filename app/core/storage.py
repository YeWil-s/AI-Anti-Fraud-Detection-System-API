"""
MinIO对象存储工具
"""
from minio import Minio
from minio.error import S3Error
from typing import Optional
import io
from app.core.config import settings
# [新增] 导入日志
from app.core.logger import get_logger

# [新增] 初始化模块级 logger
logger = get_logger(__name__)


class MinIOClient:
    """MinIO客户端封装"""
    
    def __init__(self):
        try:
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
            self.bucket_name = settings.MINIO_BUCKET_NAME
            self._ensure_bucket()
            logger.info(f"MinIO client initialized (Endpoint: {settings.MINIO_ENDPOINT})")
        except Exception as e:
            # [新增] 初始化失败通常是配置错误或服务未启动，属于严重错误
            logger.critical(f"Failed to initialize MinIO client: {e}", exc_info=True)
            # 这里可以选择是否抛出异常阻断启动，或者允许应用在无存储功能下运行
    
    def _ensure_bucket(self):
        """确保bucket存在"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                # [新增] 记录重要的变更操作
                logger.info(f"Created new MinIO bucket: {self.bucket_name}")
        except S3Error as e:
            # [修改] print -> logger.error
            logger.error(f"Error checking/creating bucket '{self.bucket_name}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in MinIO connection: {e}", exc_info=True)
    
    def upload_file(self, file_data: bytes, object_name: str, content_type: str = "application/octet-stream") -> Optional[str]:
        """上传文件"""
        try:
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(file_data),
                length=len(file_data),
                content_type=content_type
            )
            # [可选] 记录上传成功，如果文件量巨大可以改为 debug 级别
            logger.info(f"File uploaded successfully: {object_name} ({len(file_data)} bytes)")
            return f"{self.bucket_name}/{object_name}"
            
        except S3Error as e:
            # [修改] print -> logger.error
            logger.error(f"Error uploading file '{object_name}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}", exc_info=True)
            return None
    
    def get_file_url(self, object_name: str) -> Optional[str]:
        """获取文件访问URL"""
        try:
            url = self.client.presigned_get_object(self.bucket_name, object_name)
            return url
        except S3Error as e:
            # [修改] print -> logger.error
            logger.error(f"Error getting URL for '{object_name}': {e}", exc_info=True)
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """删除文件"""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"File deleted: {object_name}")
            return True
        except S3Error as e:
            # [修改] print -> logger.error
            logger.error(f"Error deleting file '{object_name}': {e}", exc_info=True)
            return False


# 全局MinIO客户端实例
minio_client = MinIOClient()


# 便捷函数
async def upload_to_minio(file_data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
    """
    上传文件到MinIO的便捷函数
    
    Args:
        file_data: 文件字节数据
        object_name: 对象名称(路径)
        content_type: 文件MIME类型
        
    Returns:
        文件URL
    """
    result = minio_client.upload_file(file_data, object_name, content_type)
    if result:
        return minio_client.get_file_url(object_name) or result
    
    # 这里的异常会被上层调用者捕获，日志中已经记录了底层的 S3Error，这里抛出通用错误即可
    raise Exception("Failed to upload file to MinIO")