from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from pathlib import Path

from app.db.database import get_db
from app.services.education_service import EducationService
from app.schemas import (
    CaseMatchRequest, LearningRecordRequest, KnowledgeItemResponse,
    RealtimeRecommendRequest,
    ProfileRecommendationEnvelope, RealtimeRecommendationEnvelope, ResponseModel
)

router = APIRouter(
    prefix="/api/education",
    tags=["Education & Anti-Fraud Cases"]
)

BASE_DIR = Path(__file__).resolve().parents[2]
EDU_VIDEO_DIR = BASE_DIR / "data" / "edu_video"

@router.get("/recommendations/{user_id}", response_model=List[KnowledgeItemResponse])
async def get_recommendations(user_id: int, limit: int = 5, db: AsyncSession = Depends(get_db)):
    """
    获取用户的个性化反诈学习推荐
    """
    service = EducationService(db)
    try:
        items = await service.get_personalized_recommendations(user_id, limit)
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{filename}")
async def get_education_video(filename: str):
    """学习中心本地视频文件访问接口。"""
    file_path = (EDU_VIDEO_DIR / filename).resolve()
    if not str(file_path).startswith(str(EDU_VIDEO_DIR.resolve())):
        raise HTTPException(status_code=400, detail="非法文件路径")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="视频不存在")
    return FileResponse(path=str(file_path), media_type="video/mp4", filename=file_path.name)

@router.post("/match_cases")
async def match_similar_cases(request: CaseMatchRequest, db: AsyncSession = Depends(get_db)):
    """
    根据当前的通话内容，匹配相似诈骗案例和相关法律法规
    """
    service = EducationService(db)
    try:
        result = await service.match_similar_cases_for_call(request.transcript, request.top_k)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/record/{user_id}")
async def record_learning(user_id: int, request: LearningRecordRequest, db: AsyncSession = Depends(get_db)):
    """
    记录或更新用户的学习进度
    """
    service = EducationService(db)
    try:
        record = await service.record_user_learning(user_id, request.item_id, request.is_completed)
        return {"status": "success", "message": "学习记录已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= 新增推荐 API =================

@router.get("/recommendations/profile/{user_id}", response_model=ProfileRecommendationEnvelope)
async def get_profile_recommendations(
    user_id: int,
    limit: int = 5,
    page: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """
    基于用户画像的个性化推荐
    
    根据用户的角色类型（老人/学生/宝妈/青壮年等），
    推荐该角色容易遭遇的诈骗类型相关案例、标语和视频
    """
    service = EducationService(db)
    try:
        result = await service.recommend_by_user_profile(user_id, limit, page)
        
        if "error" in result:
            return ResponseModel(code=404, message=result["error"], data=None)
        
        return ResponseModel(
            code=200, 
            message="获取个性化推荐成功", 
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/records/{user_id}", response_model=ResponseModel)
async def get_learning_records(
    user_id: int,
    limit: int = 20,
    completed_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    查询用户最近学习记录（支持仅看已完成）。
    """
    service = EducationService(db)
    try:
        records = await service.get_user_learning_history(
            user_id=user_id,
            limit=limit,
            completed_only=completed_only
        )
        return ResponseModel(
            code=200,
            message="获取学习记录成功",
            data={
                "items": records,
                "total": len(records),
                "completed_only": completed_only
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/realtime", response_model=RealtimeRecommendationEnvelope)
async def get_realtime_recommendations(request: RealtimeRecommendRequest, db: AsyncSession = Depends(get_db)):
    """
    基于实时对话内容的推荐
    
    当用户正在遭遇诈骗通话时，根据对话内容实时推荐相似案例和警示标语
    适用于实时检测场景
    """
    service = EducationService(db)
    try:
        result = await service.recommend_by_conversation(
            user_id=request.user_id,
            conversation_text=request.conversation_text,
            top_k=request.top_k
        )
        
        return ResponseModel(
            code=200,
            message="实时推荐成功",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/library/{user_id}", response_model=ResponseModel)
async def get_library_recommendations(
    user_id: int, 
    fraud_type: str = None,
    limit: int = 5, 
    db: AsyncSession = Depends(get_db)
):
    """
    从案例库和法律库中推荐内容
    
    融合大赛提供的案例库（300数据集标注）和法律库（反诈法条文）
    为用户提供案例警示和法律知识学习
    
    Args:
        user_id: 用户ID
        fraud_type: 可选，指定诈骗类型（如：冒充公检法诈骗）
        limit: 返回数量，默认5条
    """
    service = EducationService(db)
    try:
        result = await service.recommend_from_case_and_law_library(
            user_id=user_id,
            fraud_type=fraud_type,
            limit=limit
        )
        
        return ResponseModel(
            code=200,
            message=f"获取案例库和法律库推荐成功，共{result['source_stats']['cases']}个案例，{result['source_stats']['laws']}条法律",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))