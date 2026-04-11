"""
系统维护定时任务 (Celery)
包含：日志清理、基于文件流的知识库自主学习
"""
import os
import json
import uuid
import shutil
from datetime import datetime, timedelta
from sqlalchemy import delete
from asgiref.sync import async_to_sync

from app.tasks.celery_app import celery_app
from app.db.database import AsyncSessionLocal
from app.models.ai_detection_log import AIDetectionLog
from app.models.message_log import MessageLog
from app.core.logger import get_logger
from app.core.time_utils import now_bj

# 引入我们的向量数据库服务
from app.services.vector_db_service import vector_service

logger = get_logger(__name__)

# 获取项目的根目录 (假定当前文件在 app/tasks/ 下，向上推3级)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义待学习和已归档的文件夹路径
PENDING_DIR = os.path.join(BASE_DIR, "data", "pending_cases")
LEARNED_DIR = os.path.join(BASE_DIR, "data", "learned_cases")

@celery_app.task(name="clean_old_logs")
def clean_old_logs_task(days_to_keep: int = 30):
    """
    清理超过指定天数的旧日志
    默认保留 30 天
    """
    logger.info(f"Starting database cleanup task (Keep days: {days_to_keep})")
    
    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                cutoff_date = now_bj() - timedelta(days=days_to_keep)
                
                # 1. 清理 AI 检测流水日志 
                result_ai = await db.execute(
                    delete(AIDetectionLog).where(AIDetectionLog.created_at < cutoff_date)
                )
                deleted_ai_count = result_ai.rowcount
                
                # 2. 清理消息通知日志
                result_msg = await db.execute(
                    delete(MessageLog).where(MessageLog.created_at < cutoff_date)
                )
                deleted_msg_count = result_msg.rowcount
                
                await db.commit()
                
                logger.info(f"清理完成. 删除: {deleted_ai_count} AI 日志, {deleted_msg_count} 消息.")
                return {"status": "success", "deleted_ai": deleted_ai_count, "deleted_msg": deleted_msg_count}
                
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}", exc_info=True)
                await db.rollback()
                return {"status": "error", "message": str(e)}

    # 使用 async_to_sync 代替手动的 loop 管理
    try:
        return async_to_sync(_process)()
    except Exception as e:
        logger.error(f"Maintenance task wrapper failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="auto_learn_new_cases")
def auto_learn_new_cases_task():
    """
    [基于文件的自适应进化模块]
    扫描 data/pending_cases 目录下的 JSON 文件，将其作为新知识灌入向量库，
    完成后将文件移动到 data/learned_cases 进行归档。
    """
    # 1. 确保目录存在
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(LEARNED_DIR, exist_ok=True)

    logger.info("启动知识库自主学习任务，正在扫描待处理文件...")

    # 2. 查找所有待学习的 JSON 文件
    pending_files = [f for f in os.listdir(PENDING_DIR) if f.endswith('.json')]

    if not pending_files:
        logger.info("当前没有新的待学习案例文件。")
        return {"status": "success", "message": "No new cases"}

    total_learned = 0

    # 3. 逐个文件处理
    for filename in pending_files:
        file_path = os.path.join(PENDING_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_cases = json.load(f)

            if not isinstance(new_cases, list):
                logger.warning(f"文件 {filename} 格式错误，跳过（应为 JSON 列表）。")
                continue

            documents = []
            metadatas = []
            ids = []

            # 解析文件中的案例
            for case in new_cases:
                doc_str = f"[{case.get('modality', 'text').upper()}] 案例类型：{case.get('fraud_type', '未知')}。详情：{case.get('content', '')}"
                documents.append(doc_str)
                
                metadatas.append({
                    "fraud_type": case.get("fraud_type", "未知"),
                    "modality": case.get("modality", "text"),
                    "risk_level": case.get("risk_level", "高危"),
                    "source": case.get("source", f"文件核实录入_{filename}"),
                    "learned_at": now_bj().isoformat()
                })
                ids.append(f"auto_learn_{uuid.uuid4().hex[:8]}")

            # 4. 写入 ChromaDB 向量知识库
            if documents:
                vector_service.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                total_learned += len(documents)

            # 5. 处理成功后，将文件归档到 learned_cases 目录，防止重复学习
            timestamp_str = now_bj().strftime('%Y%m%d_%H%M%S')
            target_path = os.path.join(LEARNED_DIR, f"learned_{timestamp_str}_{filename}")
            shutil.move(file_path, target_path)
            
            logger.info(f"文件 {filename} 学习完成并成功归档，提取案例 {len(documents)} 条。")

        except Exception as e:
            logger.error(f"学习文件 {filename} 时发生异常: {e}", exc_info=True)

    logger.info(f"本次进化任务结束，共吸收 {total_learned} 条新型诈骗策略进入反诈大脑。")
    return {"status": "success", "learned_count": total_learned}