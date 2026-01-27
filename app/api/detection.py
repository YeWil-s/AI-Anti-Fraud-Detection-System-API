"""
实时检测API路由
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import asyncio
import json
from datetime import datetime

# 导入日志
from app.core.logger import get_logger

from app.db.database import get_db
from app.core.security import get_current_user_id, decode_access_token
from app.core.storage import upload_to_minio
from app.services.websocket_manager import connection_manager
from app.services.audio_processor import AudioProcessor
from app.services.video_processor import VideoProcessor
from app.models.call_record import CallRecord
from app.schemas import ResponseModel

# [修正 1] 补充导入 detect_text_task
from app.tasks.detection_tasks import detect_video_task, detect_audio_task, detect_text_task

router = APIRouter(prefix="/api/detection", tags=["实时检测"])
logger = get_logger(__name__)

@router.websocket("/ws/{user_id}/{call_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    user_id: int,
    call_id: int,
    token: str = Query(..., description="JWT认证Token")
):
    """
    WebSocket连接端点 - 实时音视频流处理
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
    
    # [关键] 为每个连接创建独立的处理器实例
    # 视频: 设置 sequence_length=10 (积攒10帧才检测)
    local_video_processor = VideoProcessor(sequence_length=10)
    # 音频: 用于简单预处理或校验
    local_audio_processor = AudioProcessor()

    try:
        while True:
            # 接收数据
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                payload = message.get("data")
                
                # --- A. 音频处理 (Scheme B) ---
                if msg_type == "audio":
                    if payload:
                        # 异步投递任务到 Celery
                        detect_audio_task.delay(payload, user_id, call_id)
                        
                        # 回复 ACK
                        await websocket.send_json({
                            "type": "ack",
                            "msg_type": "audio",
                            "timestamp": datetime.now().isoformat()
                        })

                # --- B. 视频处理 (Scheme A) ---
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
                        
                    # [修正 2] 删除了重复的 error 判断块
                    elif result["status"] == "error":
                        logger.error(f"Video process error: {result.get('message')}")

                    # 回复确认
                    await websocket.send_json({
                        "type": "ack",
                        "msg_type": "video",
                        "status": result["status"], 
                        "timestamp": datetime.now().isoformat()
                    })

                # --- C. 文本处理 (实时通话转录) ---
                elif msg_type == "text":
                    # [修正 3] 兼容 payload 是字符串或字典的情况
                    text_content = ""
                    if isinstance(payload, dict):
                        text_content = payload.get("text", "")
                    elif isinstance(payload, str):
                        text_content = payload
                    
                    if text_content and len(text_content.strip()) > 1:
                        logger.info(f"Received text (User: {user_id}): {text_content[:20]}...")
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
        # 清理资源
        local_video_processor.clear_buffer(user_id)
        logger.info(f"User {user_id} disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await connection_manager.disconnect(user_id)

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
    frame_rate: int = 1,
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