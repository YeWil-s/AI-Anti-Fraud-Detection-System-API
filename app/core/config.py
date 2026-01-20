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
    SECRET_KEY: str = "dev-secret-key-change-in-production"  # 生产环境通过.env覆盖
    
    # 数据库配置
    DATABASE_URL: str = "mysql+aiomysql://root:123456@localhost:3307/ai_fraud_detection"  # 开发环境默认MySQL
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"  # 开发环境默认本地Redis
    
    # MinIO配置
    MINIO_ENDPOINT: str = "localhost:9000"  # 开发环境默认本地MinIO
    MINIO_ACCESS_KEY: str = "dev-minio-access-key"  # 生产环境通过.env覆盖
    MINIO_SECRET_KEY: str = "dev-minio-secret-key"  # 生产环境通过.env覆盖
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "fraud-detection"
    
    # JWT配置
    JWT_SECRET_KEY: str = "dev-jwt-secret-key-change-in-production"  # 生产环境通过.env覆盖
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 短信服务配置
    SMS_ACCESS_KEY: Optional[str] = None
    SMS_SECRET_KEY: Optional[str] = None
    
    # Celery配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"  # 开发环境默认本地Redis DB1
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"  # 开发环境默认本地Redis DB2
    
    # AI模型路径
    VOICE_MODEL_PATH: str = "./models/voice_detection.onnx"
    VIDEO_MODEL_PATH: str = "./models/video_detection.onnx"
    TEXT_MODEL_PATH: str = "./models/text_detection"
    # [新增] 数据采集开关 (默认开启，用于积累数据)
    COLLECT_TRAINING_DATA: bool = True

    # [新增] 视频预处理标准 (ImageNet 标准)
    # 以后如果模型换了输入尺寸(比如换成 256x256)，改这里就行
    VIDEO_INPUT_SIZE: tuple = (224, 224) 
    
    # 归一化参数 (Mean, Std) - 对应 PyTorch 的 transforms.Normalize
    VIDEO_NORM_MEAN: list = [0.485, 0.456, 0.406] 
    VIDEO_NORM_STD: list = [0.229, 0.224, 0.225]

    # [新增] AI 检测阈值配置
    # 超过此值则判定为 Fake/Scam
    VOICE_DETECTION_THRESHOLD: float = 0.5   # 声音通常 0.5
    VIDEO_DETECTION_THRESHOLD: float = 0.6   # 视频误报高，建议 0.6-0.7
    TEXT_DETECTION_THRESHOLD: float = 0.75   # 文本很容易误判，建议设高
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()
