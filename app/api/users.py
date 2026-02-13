"""
用户管理API路由
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from pydantic import BaseModel
from app.db.database import get_db
from app.models.user import User
from app.schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    PhoneRequest,
    ResponseModel
)
from app.core.security import verify_password, get_password_hash, create_access_token, get_current_user_id
from app.core.sms import verify_sms_code, send_sms_code
# [新增] 导入日志和上下文绑定
from app.core.logger import get_logger, bind_context

# [新增] 初始化模块级 logger
logger = get_logger(__name__)

router = APIRouter(prefix="/api/users", tags=["用户管理"])

@router.post("/send-code", response_model=ResponseModel)
async def send_verification_code(request: PhoneRequest):
    """发送短信验证码"""
    # 验证手机号格式
    phone = request.phone
    if len(phone) != 11 or not phone.isdigit():
        logger.debug(f"Invalid phone format: {phone}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="手机号格式不正确"
        )
    
    # 发送验证码
    success = send_sms_code(phone)
    if not success:
        # 日志已在 send_sms_code 内部记录，这里只需抛出异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="验证码发送失败"
        )
    
    logger.info(f"Verification code sent to {phone}")
    return ResponseModel(
        code=200,
        message="验证码已发送",
        data={"phone": phone}
    )

@router.post("/register",
            response_model=ResponseModel, 
            status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """用户注册"""
    # 验证短信验证码
    if not verify_sms_code(user_data.phone, user_data.sms_code):
        logger.warning(f"Registration failed: Invalid SMS code for {user_data.phone}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="验证码错误或已过期"
        )
    
    # 检查手机号是否已存在
    result = await db.execute(select(User).where(User.phone == user_data.phone))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        logger.warning(f"Registration failed: Phone {user_data.phone} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该手机号已注册"
        )
    
    # 检查用户名是否已存在
    result = await db.execute(select(User).where(User.username == user_data.username))
    existing_username = result.scalar_one_or_none()
    
    if existing_username:
        logger.warning(f"Registration failed: Username {user_data.username} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该用户名已被使用"
        )
    
    # 创建新用户
    new_user = User(
        phone=user_data.phone,
        username=user_data.username,
        name=user_data.name,
        password_hash=get_password_hash(user_data.password)
    )
    
    try:
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # [新增] 关键审计日志
        logger.info(f"New user registered: {new_user.username} (ID: {new_user.user_id})")
        
        return ResponseModel(
            code=201,
            message="注册成功",
            data={"user_id": new_user.user_id}
        )
    except Exception as e:
        # [新增] 记录数据库异常
        logger.error(f"Registration failed for {user_data.username}: {e}", exc_info=True)
        # 不要向前端暴露具体数据库错误，返回通用提示
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )


@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """用户登录"""
    
    # 查询用户
    result = await db.execute(select(User).where(User.phone == login_data.phone))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(login_data.password, user.password_hash):  # type: ignore[arg-type]
        # [新增] 安全日志：记录登录失败尝试
        logger.warning(f"Login failed for phone: {login_data.phone} (Invalid credentials)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="手机号或密码错误"
        )
    
    if not user.is_active:  # type: ignore[truthy-bool]
        logger.warning(f"Login blocked for disabled user: {user.user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用"
        )
    
    # [关键] 登录成功，绑定上下文 user_id
    # 这样后续的处理（如果有）日志都会带上这个 ID
    bind_context(user_id=user.user_id)
    logger.info(f"User logged in successfully: {user.username} (ID: {user.user_id})")
    
    # 生成JWT令牌
    access_token = create_access_token(data={"sub": str(user.user_id), "phone": user.phone})
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户信息"""
    # 绑定上下文 (通常由中间件或 Depends 完成，这里显式绑定是个好习惯)
    bind_context(user_id=current_user_id)
    
    # 查询用户
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        logger.error(f"User ID {current_user_id} found in token but not in DB")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return UserResponse.model_validate(user)


@router.put("/family/{family_id}", response_model=ResponseModel)
async def bind_family(
    family_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """绑定家庭组"""
    bind_context(user_id=current_user_id)
    
    # 查询用户
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 更新家庭组ID
    old_family_id = user.family_id
    user.family_id = family_id  # type: ignore[assignment]
    
    try:
        await db.commit()
        await db.refresh(user)
        logger.info(f"User {current_user_id} family changed: {old_family_id} -> {family_id}")
    except Exception as e:
        logger.error(f"Failed to bind family for user {current_user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="绑定失败")
    
    return ResponseModel(
        code=200,
        message="绑定成功",
        data={
            "user_id": user.user_id,
            "family_id": user.family_id
        }
    )


@router.delete("/family", response_model=ResponseModel)
async def unbind_family(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """解绑家庭组"""
    bind_context(user_id=current_user_id)
    
    # 查询用户
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 解除家庭组绑定
    user.family_id = None  # type: ignore[assignment]
    
    try:
        await db.commit()
        await db.refresh(user)
        logger.info(f"User {current_user_id} unbound from family group")
    except Exception as e:
        logger.error(f"Failed to unbind family for user {current_user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="解绑失败")
    
    return ResponseModel(
        code=200,
        message="解绑成功",
        data={"user_id": user.user_id}
    )