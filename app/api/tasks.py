"""
任务管理API路由
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.security import get_current_user_id
from app.tasks.detection_tasks import detect_audio_task, detect_video_task, detect_text_task, get_task_status
from app.schemas import ResponseModel
from app.core.detection_guards import env_type_is_text_chat, is_env_recognition_ready
from app.core.config import settings
from typing import Dict

from pydantic import BaseModel
from typing import List


class AudioDetectionRequest(BaseModel):
    """音频检测请求"""
    audio_base64: str
    call_id: int


class VideoDetectionRequest(BaseModel):
    """视频检测请求"""
    frame_data: List[str]
    call_id: int


class TextDetectionRequest(BaseModel):
    """文本检测请求"""
    text: str
    call_id: int


router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


@router.post("/audio/detect", response_model=ResponseModel)
async def submit_audio_detection_task(
    request: AudioDetectionRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    提交音频检测异步任务
    
    Args:
        request: 音频检测请求数据
    """
    if not await is_env_recognition_ready(request.call_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="请先完成截图环境识别",
        )
    if await env_type_is_text_chat(request.call_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="文字聊天场景不执行语音检测",
        )
    task = detect_audio_task.apply_async(
        args=[request.audio_base64, current_user_id, request.call_id],
        priority=settings.CELERY_PRIORITY_AUDIO,
    )
    
    return ResponseModel(
        code=200,
        message="任务已提交",
        data={
            "task_id": task.id,
            "status": "submitted"
        }
    )


@router.post("/video/detect", response_model=ResponseModel)
async def submit_video_detection_task(
    request: VideoDetectionRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    提交视频检测异步任务
    
    Args:
        request: 视频检测请求数据
    """
    if not await is_env_recognition_ready(request.call_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="请先完成截图环境识别",
        )
    task = detect_video_task.apply_async(
        args=[request.frame_data, current_user_id, request.call_id],
        priority=settings.CELERY_PRIORITY_VIDEO,
    )
    
    return ResponseModel(
        code=200,
        message="任务已提交",
        data={
            "task_id": task.id,
            "status": "submitted"
        }
    )


@router.post("/text/detect", response_model=ResponseModel)
async def submit_text_detection_task(
    request: TextDetectionRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    提交文本检测异步任务
    
    Args:
        request: 文本检测请求数据
    """
    if not await is_env_recognition_ready(request.call_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="请先完成截图环境识别",
        )
    task = detect_text_task.apply_async(
        args=[request.text, current_user_id, request.call_id],
        priority=settings.CELERY_PRIORITY_TEXT,
    )
    
    return ResponseModel(
        code=200,
        message="任务已提交",
        data={
            "task_id": task.id,
            "status": "submitted"
        }
    )


@router.get("/status/{task_id}", response_model=ResponseModel)
async def get_task_status_api(task_id: str):
    """
    查询任务状态
    
    Args:
        task_id: 任务ID
    """
    status = get_task_status(task_id)
    
    return ResponseModel(
        code=200,
        message="查询成功",
        data=status
    )
