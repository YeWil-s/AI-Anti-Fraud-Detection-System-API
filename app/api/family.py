from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete, update

from app.db.database import get_db
from app.core.security import get_current_user_id
from app.models.family_group import FamilyGroup, FamilyApplication, ApplicationStatus
from app.models.user import User, AdminRole
from app.schemas import ResponseModel
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/family", tags=["家庭组管理"])


# =======================
# 1. 创建家庭组
# =======================
@router.post("/create", response_model=ResponseModel)
async def create_family_group(
    name: str = Query(..., description="家庭组名称"), 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """创建家庭组：自动将创建者设为主管理员(primary)"""
    # 1. 检查用户是否已在家庭组中
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    if user and user.family_id:
        return ResponseModel(code=400, message="您已在一个家庭组中，请先退出当前家庭组")
    
    # 2. 创建群组 
    new_family = FamilyGroup(group_name=name, admin_id=current_user_id) 
    db.add(new_family)
    await db.commit()
    await db.refresh(new_family)

    # 3. 将创建者绑定到新组，设为主管理员
    if user:
        user.family_id = new_family.id
        user.is_admin = True  # 兼容老字段
        user.admin_role = AdminRole.PRIMARY.value
        await db.commit()

    logger.info(f"用户 {current_user_id} 创建家庭组 {new_family.id}，设为主管理员")

    return ResponseModel(
        code=200, 
        message="家庭组创建成功", 
        data={"family_id": new_family.id, "group_name": name}
    )


# =======================
# 2. 申请加入家庭组
# =======================
@router.post("/{family_id}/apply", response_model=ResponseModel)
async def apply_join_family(
    family_id: int, 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """申请加入家庭组"""
    # 1. 检查家庭组是否存在
    result = await db.execute(select(FamilyGroup).where(FamilyGroup.id == family_id))
    if not result.scalar_one_or_none():
        return ResponseModel(code=404, message="未找到该ID的家庭组")

    # 2. 检查用户是否已在其他家庭组
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = user_result.scalar_one_or_none()
    if user and user.family_id:
        return ResponseModel(code=400, message="您已在其他家庭组中，请先退出")

    # 3. 防重复申请
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

    # 4. 写入申请表
    application = FamilyApplication(family_id=family_id, user_id=current_user_id)
    db.add(application)
    await db.commit()
    
    return ResponseModel(code=200, message="申请已发送，等待管理员审批")


# =======================
# 3. 获取待审批列表
# =======================
@router.get("/applications", response_model=ResponseModel)
async def get_applications(
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【管理员端】获取本群组的待审批列表（主/副管理员均可）"""
    # 1. 判断是否是管理员
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = user_result.scalar_one_or_none()
    
    if not user or not user.is_any_admin:
        return ResponseModel(code=403, message="无权查看，您不是家庭组管理员")
    
    if not user.family_id:
        return ResponseModel(code=400, message="您还未加入家庭组")

    # 2. 联表查询：找出申请人的基本信息
    apps_result = await db.execute(
        select(FamilyApplication, User).join(User, FamilyApplication.user_id == User.user_id)
        .where(
            and_(
                FamilyApplication.family_id == user.family_id, 
                FamilyApplication.status == ApplicationStatus.PENDING
            )
        )
    )
    
    data = []
    for app, applicant in apps_result.all():
        data.append({
            "application_id": app.id,
            "user_id": applicant.user_id,
            "username": applicant.username,
            "phone": applicant.phone,
            "role_type": applicant.role_type,
            "apply_time": app.created_at.strftime("%Y-%m-%d %H:%M:%S") if app.created_at else None
        })
        
    return ResponseModel(code=200, message="获取成功", data={"items": data})


# =======================
# 4. 审批申请
# =======================
@router.put("/applications/{app_id}", response_model=ResponseModel)
async def review_application(
    app_id: int, 
    is_approve: bool = Query(..., description="True=同意，False=拒绝"), 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【管理员端】同意/拒绝成员加入（主/副管理员均可）"""
    # 1. 找申请记录
    app_result = await db.execute(select(FamilyApplication).where(FamilyApplication.id == app_id))
    application = app_result.scalar_one_or_none()
    if not application or application.status != ApplicationStatus.PENDING:
        return ResponseModel(code=404, message="无效的申请或已处理")

    # 2. 鉴权：确认操作者是管理员且属于同一家庭组
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()
    
    if not current_user or not current_user.is_any_admin:
        return ResponseModel(code=403, message="无权审批此申请")
    
    if current_user.family_id != application.family_id:
        return ResponseModel(code=403, message="无权审批此申请")

    # 3. 执行审批
    if is_approve:
        application.status = ApplicationStatus.APPROVED
        # 将该成员的 family_id 更新
        applicant_result = await db.execute(select(User).where(User.user_id == application.user_id))
        applicant = applicant_result.scalar_one_or_none()
        if applicant:
            applicant.family_id = application.family_id
        msg = "已同意"
    else:
        application.status = ApplicationStatus.REJECTED
        msg = "已拒绝"

    await db.commit()
    logger.info(f"管理员 {current_user_id} {msg} 申请 {app_id}")
    return ResponseModel(code=200, message=f"操作成功，{msg}")


# =======================
# 5. 获取家庭成员列表
# =======================
@router.get("/members", response_model=ResponseModel)
async def get_family_members(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取家庭组成员列表（含管理员角色信息）"""
    # 1. 查找当前用户
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()
    
    if not current_user or not current_user.family_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="您还未加入任何家庭组"
        )
    
    # 2. 获取家庭组信息
    family_result = await db.execute(
        select(FamilyGroup).where(FamilyGroup.id == current_user.family_id)
    )
    family = family_result.scalar_one_or_none()
        
    # 3. 查询同 family_id 的所有成员
    members_result = await db.execute(
        select(User).where(User.family_id == current_user.family_id)
    )
    members = members_result.scalars().all()
    
    # 4. 组装返回数据
    return ResponseModel(
        code=200,
        message="获取成员列表成功",
        data={
            "family_id": current_user.family_id,
            "group_name": family.group_name if family else "",
            "members": [
                {
                    "user_id": member.user_id,
                    "username": member.username,
                    "name": member.name,
                    "phone": member.phone,
                    "role_type": member.role_type,
                    "admin_role": member.admin_role or "none",
                    "is_primary_admin": member.is_primary_admin,
                    "is_secondary_admin": member.is_secondary_admin,
                }
                for member in members
            ]
        }
    )


# =======================
# 6. 设置/取消管理员
# =======================
@router.put("/members/{user_id}/admin-role", response_model=ResponseModel)
async def set_member_admin_role(
    user_id: int,
    role: str = Query(..., description="管理员角色: none/secondary/primary"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    【主管理员】设置成员的管理员角色
    
    规则：
    - 只有主管理员可以设置其他成员的角色
    - 主管理员可以将自己降级为普通成员（需先指定新的主管理员）
    - 一个家庭组只能有一个主管理员
    """
    # 1. 验证当前用户是否为主管理员
    current_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = current_result.scalar_one_or_none()
    
    if not current_user or not current_user.is_primary_admin:
        return ResponseModel(code=403, message="只有主管理员可以设置成员角色")
    
    # 2. 验证目标用户是否在同一家庭组
    target_result = await db.execute(select(User).where(User.user_id == user_id))
    target_user = target_result.scalar_one_or_none()
    
    if not target_user or target_user.family_id != current_user.family_id:
        return ResponseModel(code=404, message="该用户不在您的家庭组中")
    
    # 3. 验证角色值
    if role not in [AdminRole.NONE.value, AdminRole.SECONDARY.value, AdminRole.PRIMARY.value]:
        return ResponseModel(code=400, message="无效的角色值，可选: none/secondary/primary")
    
    # 4. 特殊处理：设置主管理员
    if role == AdminRole.PRIMARY.value:
        # 先将当前主管理员降级
        current_user.admin_role = AdminRole.SECONDARY.value
        current_user.is_admin = True  # 保持兼容
        
        # 更新家庭组的admin_id
        family_result = await db.execute(
            select(FamilyGroup).where(FamilyGroup.id == current_user.family_id)
        )
        family = family_result.scalar_one_or_none()
        if family:
            family.admin_id = user_id
        
        logger.info(f"主管理员转让: {current_user_id} -> {user_id}")
    
    # 5. 更新目标用户角色
    target_user.admin_role = role
    target_user.is_admin = (role != AdminRole.NONE.value)  # 兼容老字段
    
    await db.commit()
    
    role_names = {
        AdminRole.NONE.value: "普通成员",
        AdminRole.SECONDARY.value: "副管理员",
        AdminRole.PRIMARY.value: "主管理员"
    }
    
    return ResponseModel(
        code=200, 
        message=f"已将 {target_user.username} 设置为{role_names.get(role, role)}"
    )


# =======================
# 7. 移除成员
# =======================
@router.delete("/members/{user_id}", response_model=ResponseModel)
async def remove_family_member(
    user_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """【管理员】将成员移出家庭组（主/副管理员均可）"""
    # 1. 验证当前用户是否为管理员
    current_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = current_result.scalar_one_or_none()
    
    if not current_user or not current_user.is_any_admin:
        return ResponseModel(code=403, message="只有管理员可以移除成员")
    
    # 2. 验证目标用户
    target_result = await db.execute(select(User).where(User.user_id == user_id))
    target_user = target_result.scalar_one_or_none()
    
    if not target_user or target_user.family_id != current_user.family_id:
        return ResponseModel(code=404, message="该用户不在您的家庭组中")
    
    # 3. 不能移除主管理员
    if target_user.is_primary_admin:
        return ResponseModel(code=400, message="不能移除主管理员")
    
    # 4. 副管理员不能移除其他副管理员
    if current_user.is_secondary_admin and target_user.is_secondary_admin:
        return ResponseModel(code=403, message="副管理员不能移除其他副管理员")
    
    # 5. 执行移除
    target_user.family_id = None
    target_user.admin_role = None
    target_user.is_admin = False
    
    await db.commit()
    logger.info(f"管理员 {current_user_id} 移除成员 {user_id}")
    
    return ResponseModel(code=200, message="已将该成员移出家庭组")


# =======================
# 8. 退出家庭组
# =======================
@router.post("/leave", response_model=ResponseModel)
async def leave_family_group(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """退出家庭组"""
    # 1. 查询当前用户
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()
    
    if not current_user or not current_user.family_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="您还未加入任何家庭组"
        )
        
    target_family_id = current_user.family_id

    # 2. 判断身份并执行对应退出逻辑
    if current_user.is_primary_admin:
        # ==========================================
        # 主管理员退出逻辑：检查是否有其他成员
        # ==========================================
        members_result = await db.execute(
            select(User).where(User.family_id == target_family_id)
        )
        all_members = members_result.scalars().all()
        
        if len(all_members) > 1:
            return ResponseModel(
                code=400, 
                message="您是主管理员，家庭组内还有其他成员。请先转让主管理员身份或移除所有成员后再退出"
            )
        
        # 只剩主管理员一人，解散家庭组
        # 删除申请记录
        await db.execute(
            delete(FamilyApplication).where(FamilyApplication.family_id == target_family_id)
        )
        
        # 删除家庭组
        await db.execute(
            delete(FamilyGroup).where(FamilyGroup.id == target_family_id)
        )
        
        current_user.family_id = None
        current_user.admin_role = None
        current_user.is_admin = False
        
        msg = "家庭组已解散"
        
    elif current_user.is_secondary_admin:
        # ==========================================
        # 副管理员退出：直接退出
        # ==========================================
        current_user.family_id = None
        current_user.admin_role = None
        current_user.is_admin = False
        msg = "已退出家庭组"
        
    else:
        # ==========================================
        # 普通成员退出
        # ==========================================
        current_user.family_id = None
        msg = "已退出家庭组"
    
    await db.commit()
    logger.info(f"用户 {current_user_id} 退出家庭组: {msg}")
    
    return ResponseModel(code=200, message=msg)


# =======================
# 9. 获取家庭组详情
# =======================
@router.get("/info", response_model=ResponseModel)
async def get_family_info(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户所在家庭组的详细信息"""
    # 1. 查找当前用户
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()
    
    if not current_user or not current_user.family_id:
        return ResponseModel(code=400, message="您还未加入任何家庭组")
    
    # 2. 获取家庭组信息
    family_result = await db.execute(
        select(FamilyGroup).where(FamilyGroup.id == current_user.family_id)
    )
    family = family_result.scalar_one_or_none()
    
    if not family:
        return ResponseModel(code=404, message="家庭组不存在")
    
    # 3. 获取主管理员信息
    admin_result = await db.execute(
        select(User).where(User.user_id == family.admin_id)
    )
    primary_admin = admin_result.scalar_one_or_none()
    
    # 4. 统计成员数量
    count_result = await db.execute(
        select(User).where(User.family_id == current_user.family_id)
    )
    members = count_result.scalars().all()
    
    # 5. 统计管理员数量
    primary_count = sum(1 for m in members if m.is_primary_admin)
    secondary_count = sum(1 for m in members if m.is_secondary_admin)
    
    return ResponseModel(
        code=200,
        message="获取成功",
        data={
            "family_id": family.id,
            "group_name": family.group_name,
            "created_at": family.created_at.isoformat() if family.created_at else None,
            "my_role": current_user.admin_role or "none",
            "primary_admin": {
                "user_id": primary_admin.user_id,
                "username": primary_admin.username,
                "phone": primary_admin.phone
            } if primary_admin else None,
            "statistics": {
                "total_members": len(members),
                "primary_admins": primary_count,
                "secondary_admins": secondary_count,
                "normal_members": len(members) - primary_count - secondary_count
            }
        }
    )