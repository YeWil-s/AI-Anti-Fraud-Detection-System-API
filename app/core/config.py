"""
应用配置模块
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基础配置
    APP_NAME: str = "AI Anti-Fraud Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str
    
    # 数据库配置
    DATABASE_URL: str
    
    # Redis配置
    REDIS_URL: str
    
    # MinIO配置
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "fraud-detection"
    
    # JWT配置
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 短信服务配置
    SMS_ACCESS_KEY: Optional[str] = None
    SMS_SECRET_KEY: Optional[str] = None
    
    # Celery配置
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # AI模型路径
    VOICE_MODEL_PATH: str = "./models/voice_detection.onnx"
    VIDEO_MODEL_PATH: str = "./models/video_detection.onnx"
    TEXT_MODEL_PATH: str = "./models/text_detection"
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()
