from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

# 假设你的数据库依赖获取函数在这里
from app.db.database import get_db
from app.services.education_service import EducationService
from app.schemas import CaseMatchRequest, LearningRecordRequest, KnowledgeItemResponse

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