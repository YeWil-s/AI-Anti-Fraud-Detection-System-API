"""
应用配置模块
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
    
    # 邮件服务配置
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_TLS: bool = True
    EMAIL_FROM: str = "noreply@antifraud.com"
    EMAIL_FROM_NAME: str = "AI反诈系统"
    
    # Celery配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"  
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"  
    
    # AI模型路径 - 音频检测
    VOICE_MODEL_PATH: str = "./models/voice_detection.onnx"  # 旧版ONNX模型（备用）
    AASIST_MODEL_PATH: str = "./models/aasist/weights/best.pth"  # SpeechFake微调模型
    AASIST_CODE_PATH: str = "./models/aasist"  # AASIST源码路径
    
    # AI模型路径 - 视频检测
    VIDEO_MODEL_PATH: str = "./models/xception/deepfake_xception_lstm.onnx"
    
    # AI模型路径 - 文本检测
    TEXT_MODEL_PATH: str = "./models/text_fraud_model.onnx"  # 旧版ONNX模型（备用）
    TEXT_VOCAB_PATH: str = "./models"
    BERT_FINETUNED_PATH: str = "./models/bert_finetuned"  # 微调后的BERT模型
    BERT_BASE_PATH: str = "./models/damo/nlp_bert_backbone_base_std"  # 基础BERT backbone
    
    # 数据收集开关
    COLLECT_TRAINING_DATA: bool = True
    
    # AASIST 模型配置（与训练时保持一致）
    # 注意：pool_ratios 和 temperatures 需要与模型训练配置匹配
    # SpeechFake微调模型使用4个元素: [0.5, 0.7, 0.5, 0.5] 和 [2.0, 2.0, 100.0, 100.0]
    AASIST_CONFIG: dict = {
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],  # 第一个元素是初始通道数
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],  # 4个元素，与SpeechFake训练配置一致
        "temperatures": [2.0, 2.0, 100.0, 100.0],  # 4个元素，与SpeechFake训练配置一致
        "first_conv": 128,  # 与AASIST.conf保持一致
        "nb_samp": 64600,   # 音频采样点数（约4秒@16kHz）
    }

    # 视频 ONNX：安装 onnxruntime-gpu 且 CUDA 可用时自动用 GPU（见 ModelService._load_video_models 日志）
    ONNX_GPU_DEVICE_ID: int = 0

    # 视频预处理标准（与 Xception + BiLSTM 训练配置保持一致）
    VIDEO_INPUT_SIZE: tuple = (299, 299)
    VIDEO_SEQUENCE_LENGTH: int = 10
    VIDEO_TARGET_FPS: float = 15.0
    VIDEO_NORM_MEAN: list = [0.485, 0.456, 0.406] 
    VIDEO_NORM_STD: list = [0.229, 0.224, 0.225]

    # AI 检测阈值配置
    VOICE_DETECTION_THRESHOLD: float = 0.90
    VIDEO_DETECTION_THRESHOLD: float = 0.74
    TEXT_DETECTION_THRESHOLD: float = 0.80

    # 离线评测 / eval_benchmark：一段视频多窗推理结果的聚合方式
    # trimmed_mean：排序后两端各去掉 VIDEO_WINDOWS_TRIM 比例的窗再取平均，抑制偶发极端帧
    # iqr_mean：落在 [Q1-1.5*IQR, Q3+1.5*IQR] 内的窗取平均；latest 与线上一致（仅最后一窗）
    VIDEO_WINDOWS_AGGREGATE: str = "trimmed_mean"
    VIDEO_WINDOWS_TRIM: float = 0.1
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30

    # LLM 大模型智能体配置
    LLM_MODEL_NAME: str = "deepseek-chat"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    
    # 智谱AI GLM-4V-Flash 
    ZHIPU_API_KEY: str = ""
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"

    # 图片 OCR 优化配置
    OCR_IMAGE_MAX_SIDE: int = 1280
    OCR_IMAGE_JPEG_QUALITY: int = 70
    OCR_UPLOAD_MIN_INTERVAL_SECONDS: float = 2.0
    
    # Pydantic V2 规范的配置写法：忽略额外变量，读取 .env
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

# 创建全局配置实例
settings = Settings()