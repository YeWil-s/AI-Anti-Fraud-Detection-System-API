"""
Pydantic Schema 模型
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import List, Optional, Any
from datetime import datetime


# ========== 用户相关 ==========
class UserBase(BaseModel):
    """用户基础模型"""
    phone: str = Field(..., min_length=11, max_length=11, description="手机号")
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    name: Optional[str] = Field(None, min_length=2, max_length=50, description="用户姓名")
    role_type: Optional[str] = Field("青壮年", description="角色类型(如老人、儿童、学生、青壮年)")
    gender: Optional[str] = Field(None, description="性别")
    profession: Optional[str] = Field(None, description="职业")
    marital_status: Optional[str] = Field(None, description="婚姻状况")

class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6, max_length=20, description="密码")
    sms_code: str = Field(..., min_length=4, max_length=6, description="验证码")
    email: Optional[str] = Field(None, description="邮箱（用于接收验证码）")


class PhoneRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11, description="手机号")
    email: Optional[str] = Field(None, description="邮箱（如果提供则发送邮箱验证码）")


class UserLogin(BaseModel):
    """用户登录模型"""
    phone: str = Field(..., min_length=11, max_length=11, description="手机号")
    password: str = Field(..., min_length=6, max_length=20, description="密码")


class UserUpdateProfile(BaseModel):
    """用户画像更新模型"""
    role_type: Optional[str] = Field(None, description="角色类型(如老人、儿童、学生、青壮年)")
    gender: Optional[str] = Field(None, description="性别")
    profession: Optional[str] = Field(None, description="职业")
    marital_status: Optional[str] = Field(None, description="婚姻状况")


class UserResponse(UserBase):
    """用户响应模型"""
    user_id: int
    family_id: Optional[int] = None
    is_active: bool
    created_at: datetime
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
    target_name: Optional[str] = None
    platform: Optional[str] = "phone"

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
    cover_image: Optional[str] = None
    video_url: Optional[str] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
    analysis: Optional[str] = Field(None, description="大模型对通话内容的完整分析")
    advice: Optional[str] = Field(None, description="大模型给出的专属防骗建议")


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
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )


# ========== 通用响应 ==========
class ResponseModel(BaseModel):
    """通用响应模型"""
    code: int = 200
    message: str = "Success"
    data: Optional[Any] = None

# ========== 案例数据库 ==========
# 1. 案例匹配的请求体   
class CaseMatchRequest(BaseModel):
    transcript: str
    top_k: int = 1

# 2. 学习记录提交的请求体
class LearningRecordRequest(BaseModel):
    item_id: int
    is_completed: bool = False

# 3. 推荐资料的响应体 (根据你数据库实际字段调整)
class KnowledgeItemResponse(BaseModel):
    id: int
    title: str
    content_type: str  
    url: Optional[str] = None
    fraud_type: Optional[str] = None
    
    class Config:
        from_attributes = True


# ========== 推荐系统相关 ==========
class RealtimeRecommendRequest(BaseModel):
    """实时推荐请求"""
    user_id: int
    conversation_text: str = Field(..., description="当前对话内容/转录文本")
    top_k: int = Field(3, ge=1, le=10, description="返回结果数量")


class CaseRecommendation(BaseModel):
    """案例推荐项"""
    id: str
    title: str
    content: str
    fraud_type: str
    risk_level: Optional[str] = None
    similarity: Optional[float] = None


class SloganRecommendation(BaseModel):
    """标语推荐项"""
    id: str
    content: str
    fraud_type: str


class VideoRecommendation(BaseModel):
    """视频推荐项"""
    id: str
    title: str
    url: str
    fraud_type: str
    description: Optional[str] = None


class ProfileRecommendationResponse(BaseModel):
    """基于画像的推荐响应"""
    cases: List[CaseRecommendation]
    slogans: List[SloganRecommendation]
    videos: List[VideoRecommendation]
    vulnerability_analysis: str
    recommended_types: List[str]


class RealtimeRecommendationResponse(BaseModel):
    """实时推荐响应"""
    cases: List[CaseRecommendation]
    slogans: List[SloganRecommendation]
    similarity_analysis: str
    alert_message: str
    matched_fraud_types: List[str]


# ========== 消息日志相关 ==========
class MessageLogResponse(BaseModel):
    """消息日志响应模型"""
    id: int
    user_id: int
    call_id: Optional[int] = None
    msg_type: str
    risk_level: str
    title: str
    content: str
    is_read: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ========== 对话消息相关 ==========
class ChatMessageResponse(BaseModel):
    """对话消息响应模型"""
    message_id: int
    call_id: int
    sequence: int
    speaker: str
    content: str
    timestamp: Optional[datetime] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ========== 家庭组相关 ==========
class FamilyGroupResponse(BaseModel):
    """家庭组响应模型"""
    id: int
    group_name: str
    admin_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)  