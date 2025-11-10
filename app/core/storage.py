"""
MinIO对象存储工具
"""
from minio import Minio
from minio.error import S3Error
from typing import Optional
import io
from app.core.config import settings


class MinIOClient:
    """MinIO客户端封装"""
    
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """确保bucket存在"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            print(f"Error creating bucket: {e}")
    
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
            return f"{self.bucket_name}/{object_name}"
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return None
    
    def get_file_url(self, object_name: str) -> Optional[str]:
        """获取文件访问URL"""
        try:
            url = self.client.presigned_get_object(self.bucket_name, object_name)
            return url
        except S3Error as e:
            print(f"Error getting file URL: {e}")
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """删除文件"""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            return True
        except S3Error as e:
            print(f"Error deleting file: {e}")
            return False


# 全局MinIO客户端实例
minio_client = MinIOClient()
