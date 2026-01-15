"""
AI检测异步任务
"""
from app.tasks.celery_app import celery_app
from app.services.model_service import model_service
from app.services.websocket_manager import connection_manager
import numpy as np
from typing import Dict
import asyncio
# [新增] 导入日志和上下文绑定工具
from app.core.logger import get_logger, bind_context

# [新增] 初始化模块级 logger
logger = get_logger(__name__)


@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_features: list, user_id: int, call_id: int) -> Dict:
    """
    音频检测异步任务
    """
    # [关键] 绑定上下文，让后续日志自动带上 user_id 和 call_id
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect audio (Features count: {len(audio_features)})")

    try:
        # 更新任务状态
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        # 转换为numpy数组
        features = np.array(audio_features)
        
        # 执行AI检测
        # 注意: 在Celery中运行async代码需要小心管理loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model_service.predict_voice(features))
            
            # 更新任务状态
            self.update_state(state='PROCESSING', meta={'progress': 80})
            
            # 如果检测到伪造,通过WebSocket推送警告
            if result.get('is_fake'):
                # [新增] 关键告警日志
                logger.warning(f"⚠️ DETECTED FAKE AUDIO! Confidence: {result.get('confidence')}")
                
                loop.run_until_complete(
                    connection_manager.send_personal_message({
                        "type": "alert",
                        "message": "检测到伪造语音!",
                        "confidence": result['confidence'],
                        "call_id": call_id
                    }, user_id)
                )
            else:
                logger.info("Audio detection passed (Real)")

        finally:
            loop.close()
        
        return {
            "status": "success",
            "result": result,
            "call_id": call_id
        }
        
    except Exception as e:
        # [修改] 记录详细堆栈
        logger.error(f"Audio detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }


@celery_app.task(name="detect_video", bind=True)
def detect_video_task(self, frame_data: list, user_id: int, call_id: int) -> Dict:
    """
    视频检测异步任务
    """
    bind_context(user_id=user_id, call_id=call_id)
    logger.info("Task started: Detect video frame")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        frame = np.array(frame_data)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model_service.predict_video(frame))
            
            self.update_state(state='PROCESSING', meta={'progress': 80})
            
            if result.get('is_deepfake'):
                logger.warning(f"⚠️ DETECTED DEEPFAKE VIDEO! Confidence: {result.get('confidence')}")
                
                loop.run_until_complete(
                    connection_manager.send_personal_message({
                        "type": "alert",
                        "message": "检测到Deepfake视频!",
                        "confidence": result['confidence'],
                        "call_id": call_id
                    }, user_id)
                )
            else:
                logger.info("Video detection passed (Real)")

        finally:
            loop.close()
        
        return {
            "status": "success",
            "result": result,
            "call_id": call_id
        }
        
    except Exception as e:
        logger.error(f"Video detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }


@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """
    文本检测异步任务
    """
    bind_context(user_id=user_id, call_id=call_id)
    # 截取前20个字符避免日志过长
    logger.info(f"Task started: Detect text (Preview: {text[:20]}...)")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model_service.predict_text(text))
            
            self.update_state(state='PROCESSING', meta={'progress': 80})
            
            if result.get('is_scam'):
                logger.warning(f"⚠️ DETECTED SCAM TEXT! Keywords: {result.get('keywords')}")
                
                loop.run_until_complete(
                    connection_manager.send_personal_message({
                        "type": "alert",
                        "message": "检测到诈骗话术!",
                        "confidence": result['confidence'],
                        "keywords": result.get('keywords', []),
                        "call_id": call_id
                    }, user_id)
                )
            else:
                logger.info("Text detection passed (Normal)")

        finally:
            loop.close()
        
        return {
            "status": "success",
            "result": result,
            "call_id": call_id
        }
        
    except Exception as e:
        logger.error(f"Text detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }


@celery_app.task(name="get_task_status")
def get_task_status(task_id: str) -> Dict:
    """
    获取任务状态
    """
    # 这里的查询通常很频繁，如果不是 debug 模式，建议不要打 info 日志，或者使用 debug 级别
    logger.debug(f"Querying task status: {task_id}")
    
    task_result = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.successful() else None,
        "info": task_result.info
    }