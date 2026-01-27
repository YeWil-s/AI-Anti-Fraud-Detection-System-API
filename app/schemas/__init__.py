"""
Pydantic Schema 模型
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime


# ========== 用户相关 ==========
class UserBase(BaseModel):
    """用户基础模型"""
    phone: str = Field(..., min_length=11, max_length=11, description="手机号")
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    name: Optional[str] = Field(None, min_length=2, max_length=50, description="用户姓名")


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6, max_length=20, description="密码")
    sms_code: str = Field(..., min_length=4, max_length=6, description="短信验证码")


class UserLogin(BaseModel):
    """用户登录模型"""
    phone: str = Field(..., min_length=11, max_length=11, description="手机号")
    password: str = Field(..., min_length=6, max_length=20, description="密码")


class UserResponse(UserBase):
    """用户响应模型"""
    user_id: int
    family_id: Optional[int] = None
    is_active: bool
    created_at: datetime
    
    # 修改为 V2 写法
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """Token响应模型"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ========== 通话记录相关 ==========
class CallRecordBase(BaseModel):
    """通话记录基础模型"""
    caller_number: Optional[str] = None
    start_time: datetime


class CallRecordCreate(CallRecordBase):
    """通话记录创建模型"""
    user_id: int


class CallRecordResponse(CallRecordBase):
    """通话记录响应模型"""
    call_id: int
    user_id: int
    end_time: Optional[datetime] = None
    duration: int
    detected_result: str
    audio_url: Optional[str] = None
    created_at: datetime
    
    # 修改为 V2 写法
    model_config = ConfigDict(from_attributes=True)


# ========== AI检测日志相关 ==========
class AIDetectionLogBase(BaseModel):
    """AI检测日志基础模型"""
    voice_confidence: Optional[float] = None
    video_confidence: Optional[float] = None
    text_confidence: Optional[float] = None
    overall_score: float


class AIDetectionLogCreate(AIDetectionLogBase):
    """AI检测日志创建模型"""
    call_id: int
    detected_keywords: Optional[str] = None
    model_version: Optional[str] = None
    model_config = ConfigDict(protected_namespaces=())


class AIDetectionLogResponse(AIDetectionLogBase):
    """AI检测日志响应模型"""
    log_id: int
    call_id: int
    detected_keywords: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime
    
    # 核心修复：合并配置，删除 class Config
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )


# ========== 通用响应 ==========
class ResponseModel(BaseModel):
    """通用响应模型"""
    code: int = 200
    message: str = "Success"
    data: Optional[dict] = None