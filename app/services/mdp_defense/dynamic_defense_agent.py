from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.services.email_service import email_service
from .mdp_env import UserVulnerability, RiskLevel, InteractionDepth, DefenseAction
import json
import os

def calculate_vulnerability(user: User) -> UserVulnerability:
    """
    将现有的用户画像维度（角色、职业、婚姻等）映射为易感度等级。
    这里契合了大赛背景中提到的不同人群被骗倾向：
    老年人/儿童/学生 -> 容易受骗；稳定职业且已婚 -> 防范较强
    """
    # 1. 安全获取字段（防止数据库中存在 None 导致 .lower() 报错）
    profession = (getattr(user, 'profession', '') or '').lower()
    role_type = getattr(user, 'role_type', '') or ''
    marital_status = getattr(user, 'marital_status', '') or ''
    
    # 2. 判断高危群体：结合 role_type (老人/儿童/学生) 和 profession (退休/无业)
    high_risk_roles = ['老人', '儿童', '学生']
    high_risk_occupations = ['学生', '退休', '无业']
    
    # 如果角色属于高危，或者职业属于高危，直接判定为高易感度
    if any(role in role_type for role in high_risk_roles) or \
       any(job in profession for job in high_risk_occupations):
        return UserVulnerability.HIGH
        
    # 3. 判断低危群体：防范意识较强的职业 + 相对稳定的生活状态(已婚)
    low_risk_occupations = ['程序员', '教师', '公务员', '医生', 'it', 'teacher']
    if any(job in profession for job in low_risk_occupations) and marital_status == '已婚':
        return UserVulnerability.LOW
        
    # 4. 其他普通情况默认为中易感度
    return UserVulnerability.MEDIUM

class DynamicDefenseAgent:
    def __init__(self, q_table_path="app/services/mdp_defense/q_table_policy.json"):
        """初始化 Agent，加载训练好的策略表"""
        self.q_table = {}
        if os.path.exists(q_table_path):
            with open(q_table_path, 'r', encoding='utf-8') as f:
                self.q_table = json.load(f)
        else:
            print(f"Warning: Q-Table 文件未找到: {q_table_path}")

    def get_best_action(self, vuln: UserVulnerability, risk: RiskLevel, depth: InteractionDepth) -> DefenseAction:
        """根据当前状态查询最佳防御动作"""
        # 使用枚举名作为 key，与 q_table_policy.json 格式保持一致
        state_key = f"{vuln.name}_{risk.name}_{depth.name}"
        
        if state_key in self.q_table:
            action_value = self.q_table[state_key]
            return DefenseAction(action_value)
            
        # 如果没查到，根据风险等级返回默认防御级别
        if risk == RiskLevel.HIGH:
            return DefenseAction.LEVEL_3
        elif risk == RiskLevel.MEDIUM:
            return DefenseAction.LEVEL_2
        else:
            return DefenseAction.LEVEL_1

    async def notify_guardian_by_email(
        self,
        db: AsyncSession,
        user_id: int,
        risk_level: str,
        details: str = ""
    ) -> bool:
        """
        向家庭组管理员发送邮件预警
        
        Args:
            db: 数据库会话
            user_id: 当前用户ID（受害者）
            risk_level: 风险等级
            details: 风险详情
            
        Returns:
            bool: 是否成功发送邮件
        """
        # 1. 查询受害者信息
        result = await db.execute(select(User).where(User.user_id == user_id))
        victim = result.scalar_one_or_none()
        
        if not victim or not victim.family_id:
            return False
        
        # 2. 查询家庭组管理员（is_admin=True 且同一家庭组）
        guardian_result = await db.execute(
            select(User).where(
                User.family_id == victim.family_id,
                User.is_admin == True,
                User.user_id != user_id,
                User.email.isnot(None)  # 必须有邮箱
            )
        )
        guardians = guardian_result.scalars().all()
        
        if not guardians:
            return False
        
        # 3. 向每个管理员发送邮件
        victim_name = victim.name or victim.username
        success_count = 0
        
        for guardian in guardians:
            if guardian.email:
                sent = await email_service.send_guardian_alert(
                    to_email=guardian.email,
                    victim_name=victim_name,
                    risk_level=risk_level,
                    details=details
                )
                if sent:
                    success_count += 1
        
        return success_count > 0

    def get_defense_action(
        self,
        user: User,
        current_risk_score: float,
        message_count: int
    ) -> int:
        """
        根据用户画像、风险分数和交互轮次获取防御动作等级
        
        Args:
            user: 用户对象
            current_risk_score: 当前风险分数 (0-100)
            message_count: 交互轮次
            
        Returns:
            int: 防御动作等级 (0=LEVEL_1, 1=LEVEL_2, 2=LEVEL_3)
        """
        # 1. 计算用户易感度
        vuln = calculate_vulnerability(user)
        
        # 2. 根据风险分数确定风险等级
        if current_risk_score >= 80:
            risk = RiskLevel.HIGH
        elif current_risk_score >= 40:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW
        
        # 3. 根据交互轮次确定交互深度
        if message_count > 10:
            depth = InteractionDepth.DEEP
        elif message_count > 3:
            depth = InteractionDepth.MEDIUM
        else:
            depth = InteractionDepth.SHALLOW
        
        # 4. 查询最佳防御动作
        action = self.get_best_action(vuln, risk, depth)
        
        # 返回动作等级 (0, 1, 2)
        return action.value

    async def get_defense_action_with_notification(
        self,
        db: AsyncSession,
        user_id: int,
        user: User,
        current_risk_score: float,
        message_count: int,
        details: str = ""
    ) -> int:
        """
        获取防御动作，并在需要时发送邮件通知
        
        Args:
            db: 数据库会话
            user_id: 当前用户ID
            user: 用户对象
            current_risk_score: 当前风险分数 (0-100)
            message_count: 交互轮次
            details: 风险详情
            
        Returns:
            int: 防御动作等级 (0=LEVEL_1, 1=LEVEL_2, 2=LEVEL_3)
        """
        # 1. 计算用户易感度
        vuln = calculate_vulnerability(user)
        
        # 2. 根据风险分数确定风险等级
        if current_risk_score >= 80:
            risk = RiskLevel.HIGH
        elif current_risk_score >= 40:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW
        
        # 3. 根据交互轮次确定交互深度
        if message_count > 10:
            depth = InteractionDepth.DEEP
        elif message_count > 3:
            depth = InteractionDepth.MEDIUM
        else:
            depth = InteractionDepth.SHALLOW
        
        # 4. 查询最佳防御动作
        action = self.get_best_action(vuln, risk, depth)
        
        # 5. 当防御等级为 LEVEL_3（强制阻断）时，发送邮件通知监护人
        if action == DefenseAction.LEVEL_3:
            await self.notify_guardian_by_email(
                db=db,
                user_id=user_id,
                risk_level=risk.name.lower(),
                details=details
            )
        
        # 返回动作等级 (0, 1, 2)
        return action.value