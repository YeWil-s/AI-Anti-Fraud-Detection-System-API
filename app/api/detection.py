"""
实时检测API路由
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import asyncio
import json
from datetime import datetime

# [新增] 导入日志
from app.core.logger import get_logger

from app.db.database import get_db
from app.core.security import get_current_user_id, decode_access_token
from app.core.storage import upload_to_minio
from app.services.websocket_manager import connection_manager
from app.services.audio_processor import AudioProcessor
from app.services.video_processor import VideoProcessor
from app.models.call_record import CallRecord
from app.schemas import ResponseModel

# [新增] 导入 Celery 任务
from app.tasks.detection_tasks import detect_video_task, detect_audio_task

router = APIRouter(prefix="/api/detection", tags=["实时检测"])
logger = get_logger(__name__)

# [修改] 移除全局处理器实例，改为在 WebSocket 连接内部实例化
# audio_processor = AudioProcessor()
# video_processor = VideoProcessor()


# [修改] 路径增加 call_id，用于任务追踪
@router.websocket("/ws/{user_id}/{call_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    user_id: int,
    call_id: int,  # [新增] 通话ID
    token: str = Query(..., description="JWT认证Token")
):
    """
    WebSocket连接端点 - 实时音视频流处理
    
    连接方式: ws://localhost:8000/api/detection/ws/{user_id}/{call_id}?token={access_token}
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
    # 这样可以确保每个通话的帧缓冲区(Buffer)是隔离的
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
                    # 策略：简单的格式校验在本地做，繁重的 MFCC 提取和推理交给 Celery
                    if payload:
                        # 异步投递任务到 Celery 队列
                        # 参数: (base64_str, user_id, call_id)
                        detect_audio_task.delay(payload, user_id, call_id)
                        
                        # 仅回复接收确认，不等待 Celery 结果(结果会通过 send_personal_message 推送)
                        await websocket.send_json({
                            "type": "ack",
                            "msg_type": "audio",
                            "timestamp": datetime.now().isoformat()
                        })

                # --- B. 视频处理 (Scheme A) ---
                elif msg_type == "video":
                    # 1. 放入处理器积攒帧
                    # result 包含: status, input_tensor(如果ready), etc.
                    result = await local_video_processor.process_frame(payload, user_id)
                    
                    # 2. 检查缓冲区状态
                    if result["status"] == "ready":
                        # 缓冲区已满 (10帧)，数据准备就绪 -> 呼叫 Celery
                        logger.info(f"Video batch ready (10 frames), sending to Celery. User: {user_id}")
                        
                        # numpy tensor 转 list 以便 JSON 序列化传给 Celery
                        input_tensor_list = result["input_tensor"].tolist()
                        
                        detect_video_task.delay(input_tensor_list, user_id, call_id)
                    
                    elif result["status"] == "error":
                        logger.error(f"Video process error: {result.get('message')}")

                    # 回复确认
                    await websocket.send_json({
                        "type": "ack",
                        "msg_type": "video",
                        "status": result["status"], # buffering / ready
                        "timestamp": datetime.now().isoformat()
                    })
                    
                # --- C. 心跳维持 ---
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
        # [清理] 显式清理缓冲区 (虽然实例销毁也会释放，但这是一个好习惯)
        local_video_processor.frame_buffers.pop(user_id, None)
        logger.info(f"User {user_id} disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await connection_manager.disconnect(user_id)


# --- 以下上传接口逻辑保持不变，只需确保 processor 引用正确 ---

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
    # 临时实例化一个处理器用于处理此请求
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