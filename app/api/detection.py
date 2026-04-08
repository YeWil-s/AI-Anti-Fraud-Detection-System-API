"""
实时检测API路由
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import asyncio
import json
from datetime import datetime
from io import BytesIO

# 导入日志
from app.core.logger import get_logger
from sqlalchemy import select
from app.db.database import get_db, AsyncSessionLocal
from app.core.security import get_current_user_id, decode_access_token
from app.core.storage import upload_to_minio
from app.services.websocket_manager import connection_manager
from app.services.audio_processor import AudioProcessor
from app.services.video_processor import VideoProcessor
from app.models.call_record import CallRecord
from app.schemas import ResponseModel
from app.core.config import settings

from app.core.redis import get_all_user_preferences, get_redis
from PIL import Image

# 导入检测任务
from app.tasks.detection_tasks import (
    detect_video_task,
    detect_audio_task,
    detect_text_task,
    detect_image_task,
    detect_image_dialogue_task,
    generate_post_call_summary_task,
)

router = APIRouter(prefix="/api/detection", tags=["实时检测"])
logger = get_logger(__name__)

async def _can_dispatch_audio_task(user_id: int, call_id: int) -> bool:
    """
    同一用户音频任务节流：
    - 最小间隔窗口内仅允许一次 detect_audio_task 入队
    - 若文本任务刚刚入队，则短暂抑制音频任务，优先文本链路
    """
    redis = await get_redis()
    audio_interval = max(1, int(getattr(settings, "AUDIO_TASK_MIN_INTERVAL_SECONDS", 2)))
    text_preempt_window = max(1, int(getattr(settings, "TEXT_PREEMPT_AUDIO_SECONDS", 1)))

    # 文本任务优先窗口：若命中则暂不派发音频任务
    text_preempt_key = f"user:{user_id}:call:{call_id}:text_preempt_audio"
    if await redis.get(text_preempt_key):
        return False

    # 音频最小间隔节流
    audio_guard_key = f"user:{user_id}:call:{call_id}:audio_task_guard"
    ok = await redis.set(audio_guard_key, "1", ex=audio_interval, nx=True)
    return bool(ok)


async def _mark_text_preempt_audio(user_id: int, call_id: int) -> None:
    """文本任务入队时，短时标记音频抑制窗口。"""
    redis = await get_redis()
    text_preempt_window = max(1, int(getattr(settings, "TEXT_PREEMPT_AUDIO_SECONDS", 1)))
    text_preempt_key = f"user:{user_id}:call:{call_id}:text_preempt_audio"
    await redis.set(text_preempt_key, "1", ex=text_preempt_window)


def _compress_image_for_ocr(
    content: bytes, max_side: int, jpeg_quality: int
) -> tuple[bytes, str]:
    """
    将图片压缩为 OCR 友好的 JPEG，减少上传与多模态推理耗时。
    返回: (压缩后的字节, content_type)
    """
    try:
        with Image.open(BytesIO(content)) as img:
            # 统一到 RGB，避免 PNG/WebP 的透明通道导致编码问题
            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            longest = max(w, h)
            if longest > max_side > 0:
                ratio = max_side / float(longest)
                new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            out = BytesIO()
            img.save(
                out,
                format="JPEG",
                quality=max(40, min(int(jpeg_quality), 95)),
                optimize=True,
            )
            return out.getvalue(), "image/jpeg"
    except Exception as e:
        logger.warning(f"图片压缩失败，回退原图: {e}")
        return content, "image/jpeg"


async def _should_route_dialogue_only(
    call_id: int, requested_dialogue_only: bool
) -> bool:
    """
    优先使用客户端参数；若未指定 dialogue_only，则根据 Redis 中的环境识别状态自动分流。
    """
    if requested_dialogue_only:
        return True

    try:
        redis = await get_redis()
        env_type = await redis.get(f"call:{call_id}:env_type")
        return bool(env_type)
    except Exception as e:
        logger.warning(f"读取环境识别状态失败，按默认流程处理: {e}")
        return False


async def _enforce_image_upload_rate_limit(call_id: int, interval_seconds: float) -> None:
    """按 call_id 进行图片上传限流，避免 OCR 任务排队导致尾延迟放大。"""
    if interval_seconds <= 0:
        return

    redis = await get_redis()
    key = f"call:{call_id}:image_upload_guard"
    # set nx ex: 窗口内只能成功一次
    ok = await redis.set(key, "1", ex=max(1, int(interval_seconds)), nx=True)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"截图上传过于频繁，请至少间隔 {interval_seconds:.1f}s",
        )


async def _fallback_end_call_record(call_id: int) -> None:
    """WebSocket 断开或异常结束时，若尚未归档则补写 end_time（避免仅依赖客户端 POST /end）。"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
        record = result.scalar_one_or_none()
        if record and record.end_time is None:
            record.end_time = datetime.now()
            if record.start_time:
                record.duration = int((record.end_time - record.start_time).total_seconds())
            await db.commit()

@router.websocket("/ws/{user_id}/{call_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    user_id: int,
    call_id: int,
    token: str = Query(..., description="JWT认证Token")
):
    """
    WebSocket连接端点 - 实时音视频流处理 + 控制指令支持
    """
    # --- 1. 鉴权逻辑 ---
    payload = decode_access_token(token)
    
    if payload is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid Token")
        return

    token_user_id = payload.get("sub")
    if token_user_id is None or int(token_user_id) != user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="User Identity Mismatch")
        return

    # --- 2. 建立连接 ---
    await connection_manager.connect(websocket, user_id)

    # 连接建立时，从 Redis 恢复该用户的旧配置
    try:
        user_prefs = await get_all_user_preferences(user_id)
        if user_prefs:
            logger.info(f"🔄 Restored preferences for user {user_id}: {user_prefs}")
            # 可选: 将恢复的配置发送给前端
            # await websocket.send_json({"type": "config_sync", "data": user_prefs})
    except Exception as e:
        logger.warning(f"Failed to restore user preferences: {e}")

    # [关键] 为每个连接创建独立的处理器实例
    # 视频: 积攒训练配置对应的时序长度后再检测
    local_video_processor = VideoProcessor(sequence_length=settings.VIDEO_SEQUENCE_LENGTH)
    local_audio_processor = AudioProcessor()

    try:
        while True:
            # 接收数据
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                payload = message.get("data")
                
                # 控制指令处理 
                if msg_type == "control":
                    # 前端发送: {"type": "control", "data": {"action": "set_config", "fps": 5}}
                    logger.info(f"Received control command from {user_id}: {payload}")
                    # 交给 Manager 统一处理 (写入 Redis, 回执 ACK)
                    await connection_manager.handle_command(user_id, payload)
                    continue

                # --- A. 音频处理 ---
                if msg_type == "audio":
                    if payload:
                        # 异步投递任务到 Celery（带用户级时间锁 + 文本优先抑制）
                        if await _can_dispatch_audio_task(user_id, call_id):
                            detect_audio_task.delay(payload, user_id, call_id)
                        
                        # 回复 ACK
                        await websocket.send_json({
                            "type": "ack",
                            "msg_type": "audio",
                            "timestamp": datetime.now().isoformat()
                        })

                # --- B. 视频处理 ---
                elif msg_type == "video":
                    # 1. 放入处理器积攒帧
                    result = await local_video_processor.process_frame(payload, user_id)
                    
                    # 2. 检查缓冲区状态
                    if result["status"] == "ready":
                        # 缓冲区已满 (10帧)，发送给 Celery
                        logger.info(f"Video batch ready, sending to Celery. User: {user_id}")
                        
                        face_batch = result["celery_payload"]
                        detect_video_task.delay(face_batch, user_id, call_id)
                        
                        local_video_processor.clear_buffer(user_id) 
                        
                    elif result["status"] == "error":
                        logger.error(f"Video process error: {result.get('message')}")

                    # 回复确认
                    await websocket.send_json({
                        "type": "ack",
                        "msg_type": "video",
                        "status": result["status"], 
                        "timestamp": datetime.now().isoformat()
                    })

                # --- C. 文本处理 ---
                elif msg_type == "text":
                    # [修正 3] 兼容 payload 是字符串或字典的情况
                    text_content = ""
                    if isinstance(payload, dict):
                        text_content = payload.get("text", "")
                    elif isinstance(payload, str):
                        text_content = payload
                    
                    if text_content and len(text_content.strip()) > 1:
                        logger.info(f"Received text (User: {user_id}): {text_content[:20]}...")
                        # 文本任务抢跑：短时间抑制音频任务派发
                        await _mark_text_preempt_audio(user_id, call_id)
                        detect_text_task.delay(text_content, user_id, call_id)
                        
                        # 回复 ACK
                        await websocket.send_json({
                            "type": "ack",
                            "msg_type": "text",
                            "timestamp": datetime.now().isoformat()
                        })

                # --- D. 心跳维持 ---
                elif msg_type == "heartbeat":
                    await websocket.send_json({
                        "type": "heartbeat_ack",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message format"
                })
                
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
        local_video_processor.clear_buffer(user_id)
        logger.info(f"User {user_id} disconnected")
        await _fallback_end_call_record(call_id)
        generate_post_call_summary_task.delay(call_id, user_id)

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        connection_manager.disconnect(user_id)
        local_video_processor.clear_buffer(user_id)
        await _fallback_end_call_record(call_id)
        generate_post_call_summary_task.delay(call_id, user_id)

# --- Upload 接口保持不变 ---
@router.post("/upload/audio", response_model=ResponseModel)
async def upload_audio(
    file: UploadFile = File(...),
    call_id: Optional[int] = None,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """上传音频文件到MinIO存储"""
    allowed_types = ["audio/mpeg", "audio/wav", "audio/x-m4a", "audio/ogg", "audio/mp3"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的音频格式: {file.content_type}"
        )
    
    content = await file.read()
    file_url = await upload_to_minio(
        content,
        f"audio/{current_user_id}/{file.filename}",
        content_type=file.content_type
    )
    
    return ResponseModel(
        code=200,
        message="音频上传成功",
        data={"url": file_url, "filename": file.filename, "size": len(content)}
    )

@router.post("/upload/video", response_model=ResponseModel)
async def upload_video(
    file: UploadFile = File(...),
    call_id: Optional[int] = None,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """上传视频文件到MinIO存储"""
    allowed_types = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的视频格式: {file.content_type}"
        )
    
    content = await file.read()
    file_url = await upload_to_minio(
        content,
        f"video/{current_user_id}/{file.filename}",
        content_type=file.content_type
    )
    
    return ResponseModel(
        code=200,
        message="视频上传成功",
        data={"url": file_url, "filename": file.filename, "size": len(content)}
    )

@router.post("/extract-frames", response_model=ResponseModel)
async def extract_video_frames(
    file: UploadFile = File(...),
    frame_rate: int = int(settings.VIDEO_TARGET_FPS),
    current_user_id: int = Depends(get_current_user_id)
):
    """从视频中提取关键帧"""
    temp_processor = VideoProcessor()
    content = await file.read()
    frames = await temp_processor.extract_frames(content, frame_rate)
    
    return ResponseModel(
        code=200,
        message=f"成功提取{len(frames)}帧",
        data={
            "frame_count": len(frames),
            "frame_rate": frame_rate,
            "frames": frames
        }
    )


@router.post("/upload/image", response_model=ResponseModel)
async def upload_image(
    file: UploadFile = File(...),
    call_id: int = Query(
        ...,
        description="当前监测会话的通话 ID，必须与 POST /api/call-records/start 返回的 call_id 一致；"
        "未传或传错会导致 Celery 中 call_id 为 None、环境感知与通话记录无法关联。",
    ),
    dialogue_only: bool = Query(
        False,
        description="为 true 时仅排队增量聊天 OCR（detect_image_dialogue），不再重复做环境分类（detect_image）",
    ),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """上传图片并触发 OCR：默认识别环境；dialogue_only 时仅做通话内增量对话提取与文本检测。"""
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的图片格式: {file.content_type}"
        )

    content = await file.read()

    # 限制图片大小（最大10MB）
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片大小不能超过10MB"
        )

    # 限流上传频率（同一通话）
    await _enforce_image_upload_rate_limit(
        call_id, float(getattr(settings, "OCR_UPLOAD_MIN_INTERVAL_SECONDS", 2.0))
    )

    # 服务端压缩，降低多模态接口延迟
    compressed_content, content_type = _compress_image_for_ocr(
        content=content,
        max_side=int(getattr(settings, "OCR_IMAGE_MAX_SIDE", 1280)),
        jpeg_quality=int(getattr(settings, "OCR_IMAGE_JPEG_QUALITY", 70)),
    )

    # 上传到MinIO存储
    file_url = await upload_to_minio(
        compressed_content,
        f"image/{current_user_id}/{file.filename}",
        content_type=content_type,
    )

    import base64

    image_base64 = base64.b64encode(compressed_content).decode("utf-8")
    route_dialogue_only = await _should_route_dialogue_only(call_id, dialogue_only)

    if route_dialogue_only:
        task = detect_image_dialogue_task.delay(image_base64, current_user_id, call_id)
        msg = "图片上传成功，正在进行增量聊天 OCR（已跳过环境分类）"
    else:
        task = detect_image_task.delay(image_base64, current_user_id, call_id)
        msg = "图片上传成功，正在进行截图环境识别（文字聊天将另行提取对话并检测）"

    return ResponseModel(
        code=200,
        message=msg,
        data={
            "url": file_url,
            "filename": file.filename,
            "size": len(compressed_content),
            "task_id": task.id,
            "dialogue_only": route_dialogue_only,
        },
    )