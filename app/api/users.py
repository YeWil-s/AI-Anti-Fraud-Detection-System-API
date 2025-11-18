"""
用户管理API路由
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.db.database import get_db
from app.models.user import User
from app.schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    ResponseModel
)
from app.core.security import verify_password, get_password_hash, create_access_token

router = APIRouter(prefix="/api/users", tags=["用户管理"])

@router.post("/register",
            response_model=ResponseModel, 
            status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """用户注册"""
    # TODO: 验证短信验证码
    # if not verify_sms_code(user_data.phone, user_data.sms_code):
    #     raise HTTPException(status_code=400, detail="验证码错误")
    
    # 检查手机号是否已存在
    result = await db.execute(select(User).where(User.phone == user_data.phone))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该手机号已注册"
        )
    
    # 创建新用户
    new_user = User(
        phone=user_data.phone,
        name=user_data.name,
        password_hash=get_password_hash(user_data.password)
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return ResponseModel(
        code=201,
        message="注册成功",
        data={"user_id": new_user.user_id}
    )


@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """用户登录"""
    
    # 查询用户
    result = await db.execute(select(User).where(User.phone == login_data.phone))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(login_data.password, user.password_hash):  # type: ignore[arg-type]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="手机号或密码错误"
        )
    
    if not user.is_active:  # type: ignore[truthy-bool]
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用"
        )
    
    # 生成JWT令牌
    access_token = create_access_token(data={"sub": str(user.user_id), "phone": user.phone})
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(db: AsyncSession = Depends(get_db)):
    """获取当前用户信息"""
    # TODO: 实现JWT认证依赖
    # current_user_id = get_current_user_id_from_token()
    
    # 临时返回示例
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="待实现JWT认证中间件"
    )


@router.put("/family/{family_id}", response_model=ResponseModel)
async def bind_family(family_id: int, db: AsyncSession = Depends(get_db)):
    """绑定家庭组"""
    # TODO: 实现家庭组绑定逻辑
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="待实现"
    )
