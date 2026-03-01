"""
应用配置模块 (Pydantic V2 重构版)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基础配置
    APP_NAME: str = "AI Anti-Fraud Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key-change-in-production"  
    
    # 数据库配置
    DATABASE_URL: str = "mysql+aiomysql://root:123456@localhost:3307/ai_fraud_detection"  
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0" 
    
    # MinIO配置
    MINIO_ENDPOINT: str = "localhost:9000"  
    MINIO_ACCESS_KEY: str = "dev-minio-access-key"  
    MINIO_SECRET_KEY: str = "dev-minio-secret-key"  
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "fraud-detection"
    
    # JWT配置
    JWT_SECRET_KEY: str = "dev-jwt-secret-key-change-in-production"  
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 短信服务配置
    SMS_ACCESS_KEY: Optional[str] = None
    SMS_SECRET_KEY: Optional[str] = None
    
    # Celery配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"  
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"  
    
    # AI模型路径
    VOICE_MODEL_PATH: str = "./models/voice_detection.onnx"
    VIDEO_MODEL_PATH: str = "./models/video_detection.onnx"
    TEXT_MODEL_PATH: str = "./models/text_fraud_model.onnx"
    TEXT_VOCAB_PATH: str = "./models/vocab.txt"
    COLLECT_TRAINING_DATA: bool = True

    # 视频预处理标准
    VIDEO_INPUT_SIZE: tuple = (224, 224) 
    VIDEO_NORM_MEAN: list = [0.485, 0.456, 0.406] 
    VIDEO_NORM_STD: list = [0.229, 0.224, 0.225]

    # AI 检测阈值配置
    VOICE_DETECTION_THRESHOLD: float = 0.70   
    VIDEO_DETECTION_THRESHOLD: float = 0.75   
    TEXT_DETECTION_THRESHOLD: float = 0.75   
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30

    # LLM 大模型智能体配置
    LLM_MODEL_NAME: str = "deepseek-chat"
    LLM_API_KEY: str = "sk-d99efeb8065f4b6d9afbd3b1cee4bda5"
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    
    # Pydantic V2 规范的配置写法：忽略额外变量，读取 .env
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

# 创建全局配置实例
settings = Settings()