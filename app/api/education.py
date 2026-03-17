from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.services.education_service import EducationService
from app.schemas import (
    CaseMatchRequest, LearningRecordRequest, KnowledgeItemResponse,
    RealtimeRecommendRequest, ProfileRecommendationResponse, RealtimeRecommendationResponse,
    ResponseModel
)

router = APIRouter(
    prefix="/api/education",
    tags=["Education & Anti-Fraud Cases"]
)

@router.get("/recommendations/{user_id}", response_model=List[KnowledgeItemResponse])
async def get_recommendations(user_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """
    获取用户的个性化反诈学习推荐
    """
    service = EducationService(db)
    try:
        items = await service.get_personalized_recommendations(user_id, limit)
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/match_cases")
async def match_similar_cases(request: CaseMatchRequest, db: Session = Depends(get_db)):
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
async def record_learning(user_id: int, request: LearningRecordRequest, db: Session = Depends(get_db)):
    """
    记录或更新用户的学习进度
    """
    service = EducationService(db)
    try:
        record = service.record_user_learning(user_id, request.item_id, request.is_completed)
        return {"status": "success", "message": "学习记录已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= 新增推荐 API =================

@router.get("/recommendations/profile/{user_id}", response_model=ResponseModel)
async def get_profile_recommendations(user_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """
    基于用户画像的个性化推荐
    
    根据用户的角色类型（老人/学生/宝妈/青壮年等），
    推荐该角色容易遭遇的诈骗类型相关案例、标语和视频
    """
    service = EducationService(db)
    try:
        result = await service.recommend_by_user_profile(user_id, limit)
        
        if "error" in result:
            return ResponseModel(code=404, message=result["error"], data=None)
        
        return ResponseModel(
            code=200, 
            message="获取个性化推荐成功", 
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/realtime", response_model=ResponseModel)
async def get_realtime_recommendations(request: RealtimeRecommendRequest, db: Session = Depends(get_db)):
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