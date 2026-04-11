"""
用户管理API路由
"""
from datetime import datetime
from app.models.call_record import CallRecord, DetectionResult
from app.services.llm_service import llm_service
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, case, func
from typing import List
from pydantic import BaseModel
from app.db.database import get_db
from app.models.user import User
from app.models.ai_detection_log import AIDetectionLog
from app.schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    SendCodeRequest,
    ResponseModel,
    UserUpdateProfile
)
from app.core.security import verify_password, get_password_hash, create_access_token, get_current_user_id
from app.core.sms import verify_sms_code, send_sms_code
from app.core.email_code import send_email_code, verify_email_code
from app.core.logger import get_logger, bind_context
from app.core.time_utils import now_bj

logger = get_logger(__name__)

router = APIRouter(prefix="/api/users", tags=["用户管理"])

@router.post("/send-code", response_model=ResponseModel)
async def send_verification_code(request: SendCodeRequest):
    """发送邮箱验证码（注册时使用）"""
    email = request.email
    
    # 验证邮箱格式
    if not email or '@' not in email or '.' not in email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱格式不正确"
        )
    
    # 发送邮箱验证码
    success = await send_email_code(email)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="邮箱验证码发送失败，请检查邮箱配置"
        )
    
    logger.info(f"Email verification code sent to {email}")
    return ResponseModel(
        code=200,
        message="验证码已发送至邮箱",
        data={"email": email}
    )


@router.post("/send-login-code", response_model=ResponseModel)
async def send_login_verification_code(request: SendCodeRequest):
    """发送登录验证码（邮箱验证码登录时使用）"""
    email = request.email
    
    # 验证邮箱格式
    if not email or '@' not in email or '.' not in email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱格式不正确"
        )
    
    # 发送邮箱验证码
    success = await send_email_code(email)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="邮箱验证码发送失败，请检查邮箱配置"
        )
    
    logger.info(f"Login email verification code sent to {email}")
    return ResponseModel(
        code=200,
        message="登录验证码已发送至邮箱",
        data={"email": email}
    )

@router.post("/register",
            response_model=ResponseModel, 
            status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """用户注册（强制使用邮箱验证码）"""
    # 验证邮箱必填
    if not user_data.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱为必填项"
        )
    
    # 使用邮箱验证码验证
    if not verify_email_code(user_data.email, user_data.email_code):
        logger.warning(f"Registration failed: Invalid email code for {user_data.email}")
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
    
    # 检查邮箱是否已存在
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_email = result.scalar_one_or_none()
    
    if existing_email:
        logger.warning(f"Registration failed: Email {user_data.email} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该邮箱已注册"
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
        email=getattr(user_data, 'email', None),  # 保存用户邮箱
        username=user_data.username,
        name=user_data.name,
        password_hash=get_password_hash(user_data.password),
        role_type=user_data.role_type,
        gender=user_data.gender,
        profession=user_data.profession,
        marital_status=user_data.marital_status,
    )
    
    try:
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        logger.info(f"New user registered: {new_user.username} (ID: {new_user.user_id})")
        
        return ResponseModel(
            code=201,
            message="注册成功",
            data={"user_id": new_user.user_id}
        )
    except Exception as e:
        # 记录数据库异常
        logger.error(f"Registration failed for {user_data.username}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )


@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    用户登录 - 支持多种登录方式：
    1. 邮箱 + 密码
    2. 邮箱 + 验证码
    3. 手机号 + 密码
    """
    user = None
    login_method = ""
    
    # 方式1: 邮箱 + 验证码
    if login_data.email and login_data.email_code and not login_data.password:
        # 先验证邮箱验证码
        if not verify_email_code(login_data.email, login_data.email_code):
            logger.warning(f"Login failed: Invalid email code for {login_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="验证码错误或已过期"
            )
        # 验证码通过，查询用户
        result = await db.execute(select(User).where(User.email == login_data.email))
        user = result.scalar_one_or_none()
        login_method = "email_code"
    
    # 方式2: 邮箱 + 密码
    elif login_data.email and login_data.password and not login_data.email_code:
        result = await db.execute(select(User).where(User.email == login_data.email))
        user = result.scalar_one_or_none()
        if user and not verify_password(login_data.password, user.password_hash):
            user = None
        login_method = "email_password"
    
    # 方式3: 手机号 + 密码
    elif login_data.phone and login_data.password:
        result = await db.execute(select(User).where(User.phone == login_data.phone))
        user = result.scalar_one_or_none()
        if user and not verify_password(login_data.password, user.password_hash):
            user = None
        login_method = "phone_password"
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="登录参数不正确，请提供：邮箱+密码、邮箱+验证码 或 手机号+密码"
        )
    
    # 验证用户是否存在
    if not user:
        logger.warning(f"Login failed: User not found or invalid credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号或密码错误"
        )
    
    if not user.is_active:
        logger.warning(f"Login blocked for disabled user: {user.user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用"
        )
    
    bind_context(user_id=user.user_id)
    logger.info(f"User logged in successfully: {user.username} (ID: {user.user_id}, method={login_method})")
    
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

@router.put("/profile", response_model=ResponseModel)
async def update_user_profile(
    profile_data: UserUpdateProfile,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """更新用户画像与监护人信息"""
    bind_context(user_id=current_user_id)
    
    # 查询当前用户
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 动态更新字段
    if profile_data.role_type is not None:
        user.role_type = profile_data.role_type
    if profile_data.gender is not None:
        user.gender = profile_data.gender
    if profile_data.profession is not None:
        user.profession = profile_data.profession
    if profile_data.marital_status is not None:
        user.marital_status = profile_data.marital_status

    try:
        await db.commit()
        await db.refresh(user)
        logger.info(f"User {current_user_id} profile updated: role={user.role_type}")
    except Exception as e:
        logger.error(f"Failed to update profile for user {current_user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="更新画像失败")
    
    return ResponseModel(
        code=200,
        message="用户画像更新成功",
        data={
            "user_id": user.user_id,
            "role_type": user.role_type,
        }
    )

# 个人安全报告生成（支持流式输出）
@router.get("/{user_id}/security-report", summary="生成用户专属安全监测报告")
async def get_user_security_report(
    user_id: int, 
    stream: bool = Query(default=False, description="是否使用流式输出"),
    db: AsyncSession = Depends(get_db)
):
    """
    分析该用户近期的通话记录，调用大模型生成 Markdown 格式的专属防诈报告
    
    - stream=false: 返回完整报告 JSON
    - stream=true: 使用 SSE 流式输出，前端可实时显示生成内容
    """
    # 1. 校验用户是否存在
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 2. 查询该用户近期通话记录 (最多取最近 10 条进行分析，避免超出大模型上下文限制)
    calls_result = await db.execute(
        select(CallRecord, AIDetectionLog)
        .outerjoin(AIDetectionLog, CallRecord.call_id == AIDetectionLog.call_id)
        .where(CallRecord.user_id == user_id)
        .order_by(CallRecord.start_time.desc())
        .limit(10)
    )
    recent_calls_with_logs = calls_result.all()

    # 3. 获取统计数据（用于结构化展示）
    stats = await _get_user_call_stats(db, user_id)

    # 4. 流式输出模式
    if stream:
        from fastapi.responses import StreamingResponse
        import json
        
        async def report_stream_generator():
            """生成报告流"""
            # 首先发送元数据和统计信息
            metadata = {
                "type": "metadata",
                "data": {
                    "user_id": user_id,
                    "username": user.username,
                    "report_generated_at": now_bj().isoformat(),
                    "stats": stats
                }
            }
            yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            
            # 然后流式生成报告内容
            content_buffer = ""
            async for chunk in llm_service.generate_security_report_stream(user, recent_calls_with_logs):
                content_buffer += chunk
                chunk_data = {
                    "type": "content",
                    "data": {
                        "chunk": chunk,
                        "content": content_buffer
                    }
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            # 发送完成标记
            complete_data = {
                "type": "complete",
                "data": {
                    "user_id": user_id,
                    "username": user.username,
                    "report_generated_at": now_bj().isoformat(),
                    "report_content": content_buffer,
                    "stats": stats
                }
            }
            yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            report_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # 5. 非流式模式：直接返回完整报告
    report_markdown = await llm_service.generate_security_report(user, recent_calls_with_logs)

    return {
        "user_id": user_id,
        "username": user.username,
        "report_generated_at": now_bj().isoformat(),
        "report_content": report_markdown,
        "stats": stats
    }


async def _get_user_call_stats(db: AsyncSession, user_id: int) -> dict:
    """获取用户通话统计数据"""
    from datetime import timedelta

    risk_values = (DetectionResult.FAKE, DetectionResult.SUSPICIOUS)

    # 总通话数
    total_calls_result = await db.execute(
        select(func.count(CallRecord.call_id)).where(CallRecord.user_id == user_id)
    )
    total_calls = total_calls_result.scalar() or 0
    
    # 风险通话统计
    risk_calls_result = await db.execute(
        select(func.count(CallRecord.call_id)).where(
            CallRecord.user_id == user_id,
            CallRecord.detected_result.in_(risk_values)
        )
    )
    risk_calls = risk_calls_result.scalar() or 0
    
    # 各类风险统计
    fake_calls_result = await db.execute(
        select(func.count(CallRecord.call_id)).where(
            CallRecord.user_id == user_id,
            CallRecord.detected_result == DetectionResult.FAKE
        )
    )
    fake_calls = fake_calls_result.scalar() or 0
    
    suspicious_calls_result = await db.execute(
        select(func.count(CallRecord.call_id)).where(
            CallRecord.user_id == user_id,
            CallRecord.detected_result == DetectionResult.SUSPICIOUS
        )
    )
    suspicious_calls = suspicious_calls_result.scalar() or 0
    
    # 最近7天通话趋势
    seven_days_ago = now_bj() - timedelta(days=7)
    
    daily_stats_result = await db.execute(
        select(
            func.date(CallRecord.start_time).label("date"),
            func.count(CallRecord.call_id).label("count"),
            func.sum(
                case(
                    (CallRecord.detected_result.in_(risk_values), 1),
                    else_=0,
                )
            ).label("risk_count")
        )
        .where(
            CallRecord.user_id == user_id,
            CallRecord.start_time >= seven_days_ago
        )
        .group_by(func.date(CallRecord.start_time))
        .order_by(func.date(CallRecord.start_time))
    )
    daily_stats = [
        {
            "date": str(row.date),
            "total": row.count,
            "risk": row.risk_count or 0
        }
        for row in daily_stats_result.all()
    ]
    
    # 诈骗类型分布
    fraud_types_result = await db.execute(
        select(CallRecord.fraud_type, func.count(CallRecord.call_id))
        .where(
            CallRecord.user_id == user_id,
            CallRecord.fraud_type.isnot(None),
            CallRecord.fraud_type != ""
        )
        .group_by(CallRecord.fraud_type)
    )
    fraud_types = [
        {"type": row[0], "count": row[1]}
        for row in fraud_types_result.all()
    ]
    
    return {
        "total_calls": total_calls,
        "risk_calls": risk_calls,
        "fake_calls": fake_calls,
        "suspicious_calls": suspicious_calls,
        "safe_calls": total_calls - risk_calls,
        "risk_rate": round(risk_calls / total_calls * 100, 2) if total_calls > 0 else 0,
        "daily_trend": daily_stats,
        "fraud_type_distribution": fraud_types
    }

@router.get("/guardian", summary="获取当前用户的监护人信息")
async def get_user_guardian(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    通过查询家庭组管理员表，动态获取监护人的手机号和信息
    """
    from app.models.family_group import FamilyAdmin
    
    # 1. 查询当前用户的 family_id
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.family_id:
        return ResponseModel(code=200, message="未绑定家庭组", data={"guardian": None})
    
    # 2. 查询 FamilyAdmin 表获取该家庭组的所有管理员（排除自己）
    guardians_result = await db.execute(
        select(FamilyAdmin, User)
        .join(User, FamilyAdmin.user_id == User.user_id)
        .where(
            FamilyAdmin.family_id == user.family_id,
            FamilyAdmin.user_id != current_user_id  # 排除自己
        )
    )
    guardians = guardians_result.all()
    
    if not guardians:
        return ResponseModel(code=200, message="家庭组内未设置监护人", data={"guardian": None})
    
    # 3. 返回监护人信息 (如果有多个监护人，这里返回列表)
    guardian_data = [
        {
            "user_id": admin_user.user_id,
            "name": admin_user.name or admin_user.username,
            "phone": admin_user.phone,
            "admin_role": admin_record.admin_role
        }
        for admin_record, admin_user in guardians
    ]
    
    return ResponseModel(
        code=200,
        message="获取监护人成功",
        data={"guardians": guardian_data}
    )