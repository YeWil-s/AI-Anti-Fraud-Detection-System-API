from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete, update

from app.db.database import get_db
from app.core.security import get_current_user_id
from app.models.family_group import FamilyGroup, FamilyApplication, FamilyAdmin, ApplicationStatus
from app.models.user import User
from app.schemas import ResponseModel
from app.core.logger import get_logger
from app.core.time_utils import now_bj, isoformat_bj

logger = get_logger(__name__)
router = APIRouter(prefix="/api/family", tags=["家庭组管理"])


# =======================
# 辅助函数
# =======================
async def get_user_family_role(db: AsyncSession, user_id: int, family_id: int):
    """获取用户在某家庭组的管理员角色"""
    result = await db.execute(
        select(FamilyAdmin).where(
            and_(FamilyAdmin.user_id == user_id, FamilyAdmin.family_id == family_id)
        )
    )
    admin_record = result.scalar_one_or_none()
    return admin_record.admin_role if admin_record else None


async def is_family_admin(db: AsyncSession, user_id: int, family_id: int) -> bool:
    """检查用户是否是某家庭组的管理员"""
    result = await db.execute(
        select(FamilyAdmin).where(
            and_(FamilyAdmin.user_id == user_id, FamilyAdmin.family_id == family_id)
        )
    )
    return result.scalar_one_or_none() is not None


async def is_primary_admin(db: AsyncSession, user_id: int, family_id: int) -> bool:
    """检查用户是否是某家庭组的主管理员"""
    result = await db.execute(
        select(FamilyAdmin).where(
            and_(
                FamilyAdmin.user_id == user_id,
                FamilyAdmin.family_id == family_id,
                FamilyAdmin.admin_role == "primary"
            )
        )
    )
    return result.scalar_one_or_none() is not None


async def is_user_admin_anywhere(db: AsyncSession, user_id: int) -> bool:
    """检查用户是否仍然是任意家庭组的管理员"""
    result = await db.execute(
        select(FamilyAdmin).where(FamilyAdmin.user_id == user_id).limit(1)
    )
    return result.scalar_one_or_none() is not None


# =======================
# 1. 创建家庭组
# =======================
@router.post("/create", response_model=ResponseModel)
async def create_family_group(
    name: str = Query(..., description="家庭组名称"), 
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """创建家庭组：自动将创建者设为主管理员
    
    说明：
    - 管理员是家庭组的管理者，负责接收成员的被诈骗通知
    - 管理员本身不是家庭组的普通成员，不享受该家庭组的保护
    - 用户要么作为普通成员加入家庭组（有family_id），要么作为管理员创建家庭组（无family_id）
    - 一个用户可以是多个家庭组的管理员
    """
    # 1. 检查用户信息
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    # 2. 检查用户是否已有家庭组（作为普通成员）
    # 管理员不是普通成员，所以如果已有family_id，不能再创建家庭组
    if user and user.family_id:
        return ResponseModel(
            code=400, 
            message="您已作为普通成员加入一个家庭组，请先退出后再创建家庭组"
        )
    
    # 3. 检查用户是否已有待审批的申请
    pending_apps = await db.execute(
        select(FamilyApplication).where(
            and_(
                FamilyApplication.user_id == current_user_id,
                FamilyApplication.status == ApplicationStatus.PENDING
            )
        )
    )
    if pending_apps.scalar_one_or_none():
        return ResponseModel(
            code=400, 
            message="您有正在审批中的家庭组申请，请先取消申请或等待审批完成后再创建家庭组"
        )
    
    # 4. 创建群组 
    new_family = FamilyGroup(group_name=name, admin_id=current_user_id) 
    db.add(new_family)
    await db.commit()
    await db.refresh(new_family)

    # 5. 将创建者设为家庭组主管理员
    if user:
        # 在FamilyAdmin表中添加管理员记录
        # 管理员不是普通成员，所以不修改family_id
        user.is_admin = True
        
        admin_record = FamilyAdmin(
            user_id=current_user_id,
            family_id=new_family.id,
            admin_role="primary"
        )
        db.add(admin_record)
        await db.commit()

    logger.info(f"用户 {current_user_id} 创建家庭组 {new_family.id}")

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

    # 2. 检查用户是否已有家庭组（普通成员只能有一个）
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = user_result.scalar_one_or_none()
    if user and user.family_id:
        return ResponseModel(code=400, message="您已加入一个家庭组，请先退出")

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
# 3. 获取待审批列表（管理员端）
# =======================
@router.get("/applications", response_model=ResponseModel)
async def get_applications(
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【管理员端】获取待审批列表（用户是所有管理员的组）"""
    # 1. 查询用户是管理员的所有家庭组
    admin_families = await db.execute(
        select(FamilyAdmin.family_id).where(FamilyAdmin.user_id == current_user_id)
    )
    family_ids = [row[0] for row in admin_families.all()]
    
    if not family_ids:
        return ResponseModel(code=403, message="您不是任何家庭组的管理员")

    # 2. 查询这些家庭组的待审批申请
    apps_result = await db.execute(
        select(FamilyApplication, User, FamilyGroup)
        .join(User, FamilyApplication.user_id == User.user_id)
        .join(FamilyGroup, FamilyApplication.family_id == FamilyGroup.id)
        .where(
            and_(
                FamilyApplication.family_id.in_(family_ids),
                FamilyApplication.status == ApplicationStatus.PENDING
            )
        )
    )
    
    data = []
    for app, applicant, family in apps_result.all():
        data.append({
            "application_id": app.id,
            "family_id": family.id,
            "family_name": family.group_name,
            "user_id": applicant.user_id,
            "username": applicant.username,
            "phone": applicant.phone,
            "role_type": applicant.role_type,
            "apply_time": app.created_at.strftime("%Y-%m-%d %H:%M:%S") if app.created_at else None
        })
        
    return ResponseModel(code=200, message="获取成功", data={"items": data})


# =======================
# 3.1 获取我的申请列表（普通用户端）
# =======================
@router.get("/my-applications", response_model=ResponseModel)
async def get_my_applications(
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【用户端】获取我提交的家庭组申请列表"""
    # 查询当前用户提交的所有申请
    apps_result = await db.execute(
        select(FamilyApplication, FamilyGroup)
        .join(FamilyGroup, FamilyApplication.family_id == FamilyGroup.id)
        .where(FamilyApplication.user_id == current_user_id)
        .order_by(FamilyApplication.created_at.desc())
    )
    
    data = []
    for app, family in apps_result.all():
        data.append({
            "application_id": app.id,
            "family_id": family.id,
            "family_name": family.group_name,
            "status": app.status.value,
            "apply_time": app.created_at.strftime("%Y-%m-%d %H:%M:%S") if app.created_at else None
        })
        
    return ResponseModel(code=200, message="获取成功", data={"items": data})


# =======================
# 3.2 取消/删除我的申请
# =======================
@router.delete("/applications/{app_id}", response_model=ResponseModel)
async def cancel_my_application(
    app_id: int,
    current_user_id: int = Depends(get_current_user_id), 
    db: AsyncSession = Depends(get_db)
):
    """【用户端】取消/删除我提交的申请（仅限pending状态的申请）"""
    # 1. 查询申请记录
    app_result = await db.execute(
        select(FamilyApplication).where(FamilyApplication.id == app_id)
    )
    application = app_result.scalar_one_or_none()
    
    if not application:
        return ResponseModel(code=404, message="申请不存在")
    
    # 2. 验证是否是当前用户提交的申请
    if application.user_id != current_user_id:
        return ResponseModel(code=403, message="无权操作此申请")
    
    # 3. 只能取消pending状态的申请
    if application.status != ApplicationStatus.PENDING:
        return ResponseModel(code=400, message="只能取消待审批的申请")
    
    # 4. 删除申请记录
    await db.delete(application)
    await db.commit()
    
    logger.info(f"用户 {current_user_id} 取消了申请 {app_id}")
    return ResponseModel(code=200, message="申请已取消")


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
    """【管理员端】同意/拒绝成员加入"""
    # 1. 找申请记录
    app_result = await db.execute(
        select(FamilyApplication, FamilyGroup)
        .join(FamilyGroup, FamilyApplication.family_id == FamilyGroup.id)
        .where(FamilyApplication.id == app_id)
    )
    app_data = app_result.first()
    if not app_data:
        return ResponseModel(code=404, message="申请不存在")
    
    application, family = app_data
    if application.status != ApplicationStatus.PENDING:
        return ResponseModel(code=400, message="该申请已处理")

    # 2. 鉴权：确认操作者是该家庭组的管理员
    if not await is_family_admin(db, current_user_id, application.family_id):
        return ResponseModel(code=403, message="无权审批此申请")

    # 3. 执行审批
    if is_approve:
        # 3.1 检查申请人是否已经有家庭组
        user_result = await db.execute(select(User).where(User.user_id == application.user_id))
        user = user_result.scalar_one_or_none()
        
        if not user:
            return ResponseModel(code=404, message="申请人不存在")
        
        if user.family_id is not None:
            # 申请人已经有家庭组了，不能批准
            application.status = ApplicationStatus.REJECTED
            await db.commit()
            logger.info(f"管理员 {current_user_id} 尝试批准申请 {app_id}，但申请人已有家庭组，自动拒绝")
            return ResponseModel(
                code=400, 
                message="该用户已加入其他家庭组，无法批准此申请",
                data={
                    "auto_rejected": True,
                    "reason": "user_already_in_family",
                    "current_family_id": user.family_id
                }
            )
        
        # 3.2 批准申请
        application.status = ApplicationStatus.APPROVED
        user.family_id = application.family_id
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
    family_id: int = Query(None, description="家庭组ID，不传则查询当前用户所在家庭组"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取家庭组成员列表
    
    - 如果传入 family_id：查询指定家庭组的成员（需要是当前用户管理的家庭组）
    - 如果不传 family_id：查询当前用户所在家庭组的成员
    """
    # 1. 确定要查询的家庭组ID
    target_family_id = family_id
    
    if target_family_id is None:
        # 普通成员用 users.family_id；监护人可能只在 family_admins 登记，无 users.family_id
        user_result = await db.execute(select(User).where(User.user_id == current_user_id))
        current_user = user_result.scalar_one_or_none()
        if not current_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

        if current_user.family_id:
            target_family_id = current_user.family_id
        else:
            managed_result = await db.execute(
                select(FamilyAdmin.family_id).where(FamilyAdmin.user_id == current_user_id)
            )
            managed_ids = [row[0] for row in managed_result.all()]
            if len(managed_ids) == 1:
                target_family_id = managed_ids[0]
            elif len(managed_ids) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="您管理多个家庭组，请指定 family_id 查询参数",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="您还未加入任何家庭组",
                )
    else:
        # 传入了family_id，需要验证当前用户是否有权限查看该家庭组
        # 检查当前用户是否是该家庭组的管理员
        is_admin_result = await db.execute(
            select(FamilyAdmin).where(
                and_(
                    FamilyAdmin.user_id == current_user_id,
                    FamilyAdmin.family_id == target_family_id
                )
            )
        )
        if not is_admin_result.scalar_one_or_none():
            # 不是管理员，检查是否是该家庭组的普通成员
            member_result = await db.execute(
                select(User).where(
                    and_(
                        User.user_id == current_user_id,
                        User.family_id == target_family_id
                    )
                )
            )
            if not member_result.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="您无权查看该家庭组的成员信息"
                )
    
    # 2. 获取家庭组信息
    family_result = await db.execute(
        select(FamilyGroup).where(FamilyGroup.id == target_family_id)
    )
    family = family_result.scalar_one_or_none()
    
    if not family:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="家庭组不存在"
        )
        
    # 3. 查询同 family_id 的所有成员
    members_result = await db.execute(
        select(User).where(User.family_id == target_family_id)
    )
    members = members_result.scalars().all()
    
    # 4. 查询该家庭组的所有管理员
    admins_result = await db.execute(
        select(FamilyAdmin).where(FamilyAdmin.family_id == target_family_id)
    )
    admin_map = {a.user_id: a.admin_role for a in admins_result.scalars().all()}
    
    # 5. 组装返回数据
    return ResponseModel(
        code=200,
        message="获取成员列表成功",
        data={
            "family_id": target_family_id,
            "group_name": family.group_name if family else "",
            "members": [
                {
                    "user_id": member.user_id,
                    "username": member.username,
                    "name": member.name,
                    "phone": member.phone,
                    "role_type": member.role_type,
                    "admin_role": admin_map.get(member.user_id, "none"),
                    "is_me": member.user_id == current_user_id
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
    - 一个家庭组只能有一个主管理员
    """
    # 1. 获取当前用户所在家庭组
    current_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = current_result.scalar_one_or_none()
    
    if not current_user or not current_user.family_id:
        return ResponseModel(code=400, message="您还未加入家庭组")
    
    family_id = current_user.family_id
    
    # 2. 验证当前用户是否为主管理员
    if not await is_primary_admin(db, current_user_id, family_id):
        return ResponseModel(code=403, message="只有主管理员可以设置成员角色")
    
    # 3. 验证目标用户是否在同一家庭组
    target_result = await db.execute(select(User).where(User.user_id == user_id))
    target_user = target_result.scalar_one_or_none()
    
    if not target_user or target_user.family_id != family_id:
        return ResponseModel(code=404, message="该用户不在您的家庭组中")
    
    # 4. 验证角色值
    if role not in ["none", "secondary", "primary"]:
        return ResponseModel(code=400, message="无效的角色值，可选: none/secondary/primary")
    
    # 5. 查询现有的管理员记录
    existing_result = await db.execute(
        select(FamilyAdmin).where(
            and_(FamilyAdmin.user_id == user_id, FamilyAdmin.family_id == family_id)
        )
    )
    existing_record = existing_result.scalar_one_or_none()
    
    # 6. 处理角色变更
    if role == "none":
        # 取消管理员
        if existing_record:
            await db.delete(existing_record)
            # 检查用户是否还是其他家庭组的管理员
            target_user.is_admin = await is_user_admin_anywhere(db, user_id)
            await db.commit()
        return ResponseModel(code=200, message=f"已取消 {target_user.username} 的管理员权限")
    
    elif role == "primary":
        # 设置为主管理员：先将当前主管理员降级
        current_primary = await db.execute(
            select(FamilyAdmin).where(
                and_(FamilyAdmin.family_id == family_id, FamilyAdmin.admin_role == "primary")
            )
        )
        for record in current_primary.scalars().all():
            record.admin_role = "secondary"
        
        # 更新家庭组的admin_id
        family_result = await db.execute(
            select(FamilyGroup).where(FamilyGroup.id == family_id)
        )
        family = family_result.scalar_one_or_none()
        if family:
            family.admin_id = user_id
        
        logger.info(f"主管理员转让: {current_user_id} -> {user_id}")
    
    # 7. 更新或创建管理员记录
    if existing_record:
        existing_record.admin_role = role
    else:
        new_admin = FamilyAdmin(
            user_id=user_id,
            family_id=family_id,
            admin_role=role
        )
        db.add(new_admin)
        target_user.is_admin = True
    
    await db.commit()
    
    role_names = {"none": "普通成员", "secondary": "副管理员", "primary": "主管理员"}
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
    """【管理员】将成员移出家庭组"""
    # 1. 获取当前用户所在家庭组
    current_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = current_result.scalar_one_or_none()
    
    if not current_user or not current_user.family_id:
        return ResponseModel(code=400, message="您还未加入家庭组")
    
    family_id = current_user.family_id
    
    # 2. 验证当前用户是否为管理员
    current_role = await get_user_family_role(db, current_user_id, family_id)
    if not current_role:
        return ResponseModel(code=403, message="只有管理员可以移除成员")
    
    # 3. 验证目标用户
    target_result = await db.execute(select(User).where(User.user_id == user_id))
    target_user = target_result.scalar_one_or_none()
    
    if not target_user or target_user.family_id != family_id:
        return ResponseModel(code=404, message="该用户不在您的家庭组中")
    
    # 4. 不能移除主管理员
    target_role = await get_user_family_role(db, user_id, family_id)
    if target_role == "primary":
        return ResponseModel(code=400, message="不能移除主管理员")
    
    # 5. 副管理员不能移除其他副管理员
    if current_role == "secondary" and target_role == "secondary":
        return ResponseModel(code=403, message="副管理员不能移除其他副管理员")
    
    # 6. 执行移除
    target_user.family_id = None
    
    # 删除管理员记录（如果有）
    if target_role:
        await db.execute(
            delete(FamilyAdmin).where(
                and_(FamilyAdmin.user_id == user_id, FamilyAdmin.family_id == family_id)
            )
        )
    
    # 检查用户是否还是其他家庭组的管理员，再决定 is_admin
    target_user.is_admin = await is_user_admin_anywhere(db, user_id)
    
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
    user_role = await get_user_family_role(db, current_user_id, target_family_id)

    # 2. 判断身份并执行对应退出逻辑
    if user_role == "primary":
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
        await db.execute(
            delete(FamilyApplication).where(FamilyApplication.family_id == target_family_id)
        )
        await db.execute(
            delete(FamilyAdmin).where(FamilyAdmin.family_id == target_family_id)
        )
        await db.execute(
            delete(FamilyGroup).where(FamilyGroup.id == target_family_id)
        )
        
        current_user.family_id = None
        # 检查用户是否还是其他家庭组的管理员
        current_user.is_admin = await is_user_admin_anywhere(db, current_user_id)
        
        msg = "家庭组已解散"
        
    else:
        # ==========================================
        # 普通成员或副管理员退出
        # ==========================================
        current_user.family_id = None
        
        # 删除管理员记录（如果有）
        if user_role:
            await db.execute(
                delete(FamilyAdmin).where(
                    and_(
                        FamilyAdmin.user_id == current_user_id,
                        FamilyAdmin.family_id == target_family_id
                    )
                )
            )
        
        # 检查用户是否还是其他家庭组的管理员
        current_user.is_admin = await is_user_admin_anywhere(db, current_user_id)
        
        msg = "已退出家庭组"
    
    await db.commit()
    logger.info(f"用户 {current_user_id} 退出家庭组: {msg}")
    
    return ResponseModel(code=200, message=msg)


# =======================
# 9. 获取家庭组详情
# =======================
@router.get("/info", response_model=ResponseModel)
async def get_family_info(
    family_id: int = Query(None, description="家庭组ID；监护人未在 users 登记 family_id 时须传入或仅管理一个家庭时可省略"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户所在家庭组的详细信息（成员以 users.family_id 为准；监护人以 FamilyAdmin 为准）"""
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()
    if not current_user:
        return ResponseModel(code=404, message="用户不存在")

    target_family_id: int
    if family_id is not None:
        if current_user.family_id == family_id:
            target_family_id = family_id
        else:
            fa = await db.execute(
                select(FamilyAdmin).where(
                    and_(
                        FamilyAdmin.user_id == current_user_id,
                        FamilyAdmin.family_id == family_id,
                    )
                )
            )
            if not fa.scalar_one_or_none():
                return ResponseModel(code=403, message="您无权查看该家庭组信息")
            target_family_id = family_id
    elif current_user.family_id:
        target_family_id = current_user.family_id
    else:
        managed_result = await db.execute(
            select(FamilyAdmin.family_id).where(FamilyAdmin.user_id == current_user_id)
        )
        managed_ids = [row[0] for row in managed_result.all()]
        if len(managed_ids) == 1:
            target_family_id = managed_ids[0]
        elif len(managed_ids) > 1:
            return ResponseModel(
                code=400,
                message="您管理多个家庭组，请通过 family_id 参数指定要查看的家庭组",
            )
        else:
            return ResponseModel(code=400, message="您还未加入任何家庭组")

    family_result = await db.execute(
        select(FamilyGroup).where(FamilyGroup.id == target_family_id)
    )
    family = family_result.scalar_one_or_none()

    if not family:
        return ResponseModel(code=404, message="家庭组不存在")

    admin_result = await db.execute(
        select(User).where(User.user_id == family.admin_id)
    )
    primary_admin = admin_result.scalar_one_or_none()

    count_result = await db.execute(
        select(User).where(User.family_id == target_family_id)
    )
    members = count_result.scalars().all()

    admins_result = await db.execute(
        select(FamilyAdmin).where(FamilyAdmin.family_id == target_family_id)
    )
    admins = admins_result.scalars().all()
    primary_count = sum(1 for a in admins if a.admin_role == "primary")
    secondary_count = sum(1 for a in admins if a.admin_role == "secondary")

    my_role = await get_user_family_role(db, current_user_id, target_family_id)
    
    return ResponseModel(
        code=200,
        message="获取成功",
        data={
            "family_id": family.id,
            "group_name": family.group_name,
            "created_at": isoformat_bj(family.created_at),
            "my_role": my_role or "member",
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


# =======================
# 10. 获取我管理的家庭组列表
# =======================
@router.get("/my-admin-families", response_model=ResponseModel)
async def get_my_admin_families(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户作为管理员的所有家庭组"""
    result = await db.execute(
        select(FamilyAdmin, FamilyGroup)
        .join(FamilyGroup, FamilyAdmin.family_id == FamilyGroup.id)
        .where(FamilyAdmin.user_id == current_user_id)
    )
    
    data = []
    for admin_record, family in result.all():
        # 统计成员数
        member_count = await db.execute(
            select(User).where(User.family_id == family.id)
        )
        count = len(member_count.scalars().all())
        
        data.append({
            "family_id": family.id,
            "group_name": family.group_name,
            "my_role": admin_record.admin_role,
            "member_count": count,
            "created_at": isoformat_bj(family.created_at)
        })
    
    return ResponseModel(code=200, message="获取成功", data={"items": data})


# =======================
# 11. 一键通报：用户主动向监护人发送求助
# =======================
@router.post("/sos", response_model=ResponseModel)
async def send_sos_alert(
    call_id: int = Query(..., description="当前通话ID"),
    message: str = Query(default="我正在遭遇可疑通话，请立即联系我！", description="求助信息"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    一键通报功能：用户主动向家庭组监护人发送求助信号
    
    场景：用户察觉到异常但系统尚未告警，或需要监护人立即介入时使用
    """
    import redis
    import json
    from datetime import datetime
    from app.core.config import settings
    
    # 1. 获取用户信息
    result = await db.execute(select(User).where(User.user_id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    if not user.family_id:
        return ResponseModel(code=400, message="您未加入任何家庭组，无法使用此功能")
    
    # 2. 查询家庭组所有管理员
    admins_result = await db.execute(
        select(FamilyAdmin, User)
        .join(User, FamilyAdmin.user_id == User.user_id)
        .where(FamilyAdmin.family_id == user.family_id)
    )
    admins = admins_result.all()
    
    if not admins:
        return ResponseModel(code=400, message="您的家庭组没有监护人，无法发送求助")
    
    # 3. 构造求助消息
    sos_payload = {
        "type": "sos_alert",
        "data": {
            "title": "紧急求助",
            "message": f"您的家人【{user.name or user.username}】正在请求帮助！{message}",
            "victim_id": current_user_id,
            "victim_name": user.name or user.username,
            "victim_phone": user.phone,
            "call_id": call_id,
            "family_id": user.family_id,
            "timestamp": now_bj().isoformat(),
            "display_mode": "popup",
            "action": "vibrate",  # 前端执行震动
            "urgency": "high"
        }
    }
    
    # 4. 通过WebSocket推送给所有监护人
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    notified_count = 0
    
    for admin_record, admin_user in admins:
        try:
            message_data = {
                "user_id": admin_user.user_id,
                "payload": sos_payload
            }
            redis_client.publish("fraud_alerts", json.dumps(message_data))
            notified_count += 1
            logger.info(f"SOS sent to Admin {admin_user.user_id} (role={admin_record.admin_role})")
        except Exception as e:
            logger.error(f"Failed to send SOS to Admin {admin_user.user_id}: {e}")
    
    return ResponseModel(
        code=200, 
        message=f"求助信号已发送给 {notified_count} 位监护人",
        data={
            "notified_count": notified_count,
            "call_id": call_id,
            "timestamp": now_bj().isoformat()
        }
    )


# =======================
# 12. 远程干预：监护人远程控制被监护人通话
# =======================
@router.post("/remote-intervene", response_model=ResponseModel)
async def remote_intervene(
    target_user_id: int = Query(..., description="被干预的用户ID"),
    action: str = Query(..., description="干预动作: block_call(强制挂断) / warn(发送警告) / check_status(检查状态)"),
    message: str = Query(default="", description="附加消息"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    远程干预功能：监护人远程控制被监护人的通话
    
    支持的动作：
    - block_call: 强制挂断当前通话
    - warn: 向被监护人发送警告消息
    - check_status: 检查被监护人当前状态
    """
    import redis
    import json
    from datetime import datetime
    from app.core.config import settings
    
    # 1. 加载操作者（监护人姓名等仅来自 User；family 归属以 FamilyAdmin 为准）
    admin_result = await db.execute(select(User).where(User.user_id == current_user_id))
    admin_user = admin_result.scalar_one_or_none()
    
    if not admin_user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 2. 加载被干预用户（普通成员在 users.family_id 登记所属家庭）
    target_result = await db.execute(select(User).where(User.user_id == target_user_id))
    target_user = target_result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(status_code=404, detail="目标用户不存在")
    
    # 3. 权限：被监护人须已加入家庭；操作者须在 FamilyAdmin 中登记为该家庭的管理员（不要求管理员 users.family_id）
    if not target_user.family_id:
        raise HTTPException(status_code=403, detail="目标用户未加入家庭组，无法干预")
    
    admin_role_result = await db.execute(
        select(FamilyAdmin).where(
            and_(
                FamilyAdmin.user_id == current_user_id,
                FamilyAdmin.family_id == target_user.family_id,
            )
        )
    )
    admin_role = admin_role_result.scalar_one_or_none()
    
    if not admin_role:
        raise HTTPException(status_code=403, detail="您不是该家庭组的管理员")
    
    # 4. 构造干预指令
    control_payload = {
        "type": "remote_control",
        "data": {
            "action": action,
            "from_admin_id": current_user_id,
            "from_admin_name": admin_user.name or admin_user.username,
            "target_user_id": target_user_id,
            "message": message,
            "timestamp": now_bj().isoformat()
        }
    }
    
    # 5. 根据动作类型设置具体指令
    if action == "block_call":
        control_payload["data"]["control"] = {
            "block_call": True,
            "warning_mode": "fullscreen",
            "ui_message": f"监护人 {admin_user.name or admin_user.username} 已为您强制挂断此通话"
        }
    elif action == "warn":
        control_payload["data"]["control"] = {
            "block_call": False,
            "warning_mode": "popup",
            "ui_message": message or f"监护人 {admin_user.name or admin_user.username} 提醒您注意通话安全"
        }
    elif action == "check_status":
        control_payload["data"]["control"] = {
            "request_status": True
        }
    else:
        return ResponseModel(code=400, message=f"不支持的操作类型: {action}")
    
    # 6. 发送指令到目标用户
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    try:
        message_data = {
            "user_id": target_user_id,
            "payload": control_payload
        }
        redis_client.publish("fraud_alerts", json.dumps(message_data))
        logger.info(f"Remote control sent: Admin {current_user_id} -> User {target_user_id}, action={action}")
    except Exception as e:
        logger.error(f"Failed to send remote control: {e}")
        return ResponseModel(code=500, message=f"指令发送失败: {str(e)}")
    
    return ResponseModel(
        code=200, 
        message=f"干预指令已发送给 {target_user.name or target_user.username}",
        data={
            "action": action,
            "target_user_id": target_user_id,
            "target_user_name": target_user.name or target_user.username,
            "timestamp": now_bj().isoformat()
        }
    )
