from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db.database import get_db
from app.core.security import get_current_user_id
from app.models.family_group import FamilyGroup, FamilyApplication, ApplicationStatus
from app.models.user import User
from app.schemas import ResponseModel

router = APIRouter(prefix="/api/family", tags=["家庭组管理"])

@router.post("/create", response_model=ResponseModel)
async def create_family_group(
    name: str = Query(..., description="家庭组名称"), 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """创建家庭组：自动将创建者设为管理员并绑定"""
    # 1. 检查用户是否已在家庭组中
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    if user and user.family_id:
        return ResponseModel(code=400, message="您已在一个家庭组中")

    # 2. 创建群组 (请根据你 FamilyGroup 实际的字段名修改，比如 group_name 或 name)
    new_family = FamilyGroup(group_name=name, admin_id=current_user_id) 
    db.add(new_family)
    await db.commit()
    await db.refresh(new_family)

    # 3. 将创建者自己绑定到这个新组
    if user:
        user.family_id = new_family.id
        await db.commit()

    return ResponseModel(
        code=200, 
        message="家庭组创建成功", 
        data={"family_id": new_family.id, "group_name": name}
    )

@router.post("/{family_id}/apply", response_model=ResponseModel)
async def apply_join_family(
    family_id: int, 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """申请加入家庭组 (前端输入ID后调用)"""
    # 1. 检查家庭组是否存在
    result = await db.execute(select(FamilyGroup).where(FamilyGroup.id == family_id))
    if not result.scalar_one_or_none():
        return ResponseModel(code=404, message="未找到该ID的家庭组")

    # 2. 防重复申请
    app_result = await db.execute(
        select(FamilyApplication).where(
            and_(
                FamilyApplication.family_id == family_id, 
                FamilyApplication.user_id == current_user_id, 
                FamilyApplication.status == ApplicationStatus.PENDING
            )
        )
    )
    if app_result.scalar_one_or_none():
        return ResponseModel(code=400, message="您已提交过申请，请等待管理员审批")

    # 3. 写入申请表
    application = FamilyApplication(family_id=family_id, user_id=current_user_id)
    db.add(application)
    await db.commit()
    
    return ResponseModel(code=200, message="申请已发送，等待管理员审批")

@router.get("/applications", response_model=ResponseModel)
async def get_applications(
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【管理员端】获取本群组的待审批列表"""
    # 1. 判断是否是管理员
    result = await db.execute(select(FamilyGroup).where(FamilyGroup.admin_id == current_user_id))
    family = result.scalar_one_or_none()
    if not family:
        return ResponseModel(code=403, message="无权查看，您不是家庭组管理员")

    # 2. 联表查询：找出申请人的基本信息
    apps_result = await db.execute(
        select(FamilyApplication, User).join(User, FamilyApplication.user_id == User.user_id)
        .where(
            and_(
                FamilyApplication.family_id == family.id, 
                FamilyApplication.status == ApplicationStatus.PENDING
            )
        )
    )
    
    data = []
    for app, user in apps_result.all():
        data.append({
            "application_id": app.id,
            "user_id": user.user_id,
            "phone": user.phone,
            "apply_time": app.created_at.strftime("%Y-%m-%d %H:%M:%S") if app.created_at else None
        })
        
    return ResponseModel(code=200, message="获取成功", data=data)

@router.put("/applications/{app_id}", response_model=ResponseModel)
async def review_application(
    app_id: int, 
    is_approve: bool = Query(..., description="True=同意，False=拒绝"), 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【管理员端】同意/拒绝成员加入"""
    # 1. 找申请记录
    app_result = await db.execute(select(FamilyApplication).where(FamilyApplication.id == app_id))
    application = app_result.scalar_one_or_none()
    if not application or application.status != ApplicationStatus.PENDING:
        return ResponseModel(code=404, message="无效的申请或已处理")

    # 2. 鉴权：确认操作者是这个家庭组的管理员
    fam_result = await db.execute(select(FamilyGroup).where(FamilyGroup.id == application.family_id))
    family = fam_result.scalar_one_or_none()
    if not family or family.admin_id != current_user_id:
        return ResponseModel(code=403, message="无权审批此申请")

    # 3. 执行审批
    if is_approve:
        application.status = ApplicationStatus.APPROVED
        # 核心：将该成员的 family_id 更新
        user_result = await db.execute(select(User).where(User.user_id == application.user_id))
        user = user_result.scalar_one_or_none()
        if user:
            user.family_id = family.id
        msg = "已同意"
    else:
        application.status = ApplicationStatus.REJECTED
        msg = "已拒绝"

    await db.commit()
    return ResponseModel(code=200, message=f"操作成功，{msg}")