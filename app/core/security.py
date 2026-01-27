"""
JWT认证工具
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
# [新增] 导入日志
from app.core.logger import get_logger

# [新增] 初始化模块级 logger
logger = get_logger(__name__)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建JWT访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """解析JWT令牌"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError as e:
        # [修改] 记录具体的认证失败原因 (过期/篡改/格式错误)
        # 使用 warning 级别，因为这通常是客户端错误，但对于安全审计很重要
        logger.warning(f"JWT verification failed: {e}")
        return None


# HTTP Bearer 认证方案
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    从JWT令牌中获取当前用户ID
    用于需要认证的路由
    """
    token = credentials.credentials
    
    # 解析令牌
    payload = decode_access_token(token)
    if payload is None:
        # 日志已在 decode_access_token 中记录，这里只需抛出 HTTP 异常
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 获取用户ID
    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        logger.warning("JWT token missing 'sub' claim")  # [新增] 补充日志
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌格式错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        return int(user_id)
    except ValueError as e:
        logger.warning(f"Invalid user_id format in token: {user_id}") # [新增] 补充日志
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌中的用户ID无效",
            headers={"WWW-Authenticate": "Bearer"},
        )