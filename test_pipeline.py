import sys
import os

# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.user import User
from app.services.risk_fusion_engine import RiskFusionEngine
from app.services.mdp_defense.mdp_env import UserVulnerability, RiskLevel, InteractionDepth, DefenseAction

# ==========================================
# 1. 修复版的易感度计算逻辑 (直接贴在这里方便测试)
# ==========================================
def calculate_vulnerability(user: User) -> UserVulnerability:
    profession = (getattr(user, 'profession', '') or '').lower()
    role_type = getattr(user, 'role_type', '') or ''
    marital_status = getattr(user, 'marital_status', '') or ''
    
    high_risk_roles = ['老人', '儿童', '学生']
    high_risk_occupations = ['学生', '退休', '无业']
    
    if any(role in role_type for role in high_risk_roles) or \
       any(job in profession for job in high_risk_occupations):
        return UserVulnerability.HIGH
        
    low_risk_occupations = ['程序员', '教师', '公务员', '医生']
    if any(job in profession for job in low_risk_occupations) and marital_status == '已婚':
        return UserVulnerability.LOW
        
    return UserVulnerability.MEDIUM

# ==========================================
# 2. 模拟训练好的 MDP Agent 决策逻辑
# (完全按照你 mdp_env.py 里的 Reward 设计)
# ==========================================
def get_mock_mdp_action(vuln: UserVulnerability, risk: RiskLevel, depth: InteractionDepth) -> DefenseAction:
    if risk == RiskLevel.LOW:
        return DefenseAction.LEVEL_1 # 低危不打扰
        
    elif risk == RiskLevel.HIGH:
        return DefenseAction.LEVEL_3 # 高危直接阻断
        
    else: # 中危 (最体现动态防御的地方)
        if vuln == UserVulnerability.HIGH:
            # 高易感人群 (老人/学生)
            if depth == InteractionDepth.DEEP:
                return DefenseAction.LEVEL_3 # 聊久了极其危险，升级为强制干预
            else:
                return DefenseAction.LEVEL_2
                
        elif vuln == UserVulnerability.LOW:
            # 防骗意识强的人 (已婚程序员)
            if depth == InteractionDepth.SHALLOW:
                return DefenseAction.LEVEL_1 # 刚开始聊，给个弱提示就行，免得烦人
            else:
                return DefenseAction.LEVEL_2
                
        else:
            return DefenseAction.LEVEL_2

def map_score_to_risk_level(score: float) -> RiskLevel:
    if score <= 40: return RiskLevel.LOW
    elif score <= 80: return RiskLevel.MEDIUM
    else: return RiskLevel.HIGH

def map_round_to_depth(rounds: int) -> InteractionDepth:
    if rounds <= 3: return InteractionDepth.SHALLOW
    elif rounds <= 10: return InteractionDepth.MEDIUM
    else: return InteractionDepth.DEEP

# ==========================================
# 3. 核心测试流程
# ==========================================
def run_simulation():
    print("🚀 开始多模态融合打分 & MDP 动态防御流程测试...\n")
    engine = RiskFusionEngine()

    # 创建两个对比用户
    user_elderly = User(role_type="老人", profession="退休", marital_status="丧偶")
    user_coder = User(role_type="青壮年", profession="程序员", marital_status="已婚")
    
    vuln_elderly = calculate_vulnerability(user_elderly)
    vuln_coder = calculate_vulnerability(user_coder)

    # 模拟一场诈骗通话的 3 个阶段：
    # 阶段 1: 刚加好友，寒暄 (浅度交互，低风险)
    # 阶段 2: 聊了几天，开始推荐理财 (中度交互，中风险)
    # 阶段 3: 诱导转账 (深度交互，高风险 + 熔断)
    
    stages = [
        {
            "round": 2, "desc": "阶段 1: 骗子假装熟人寒暄",
            "llm": {"intent": "日常聊天", "risk_level": "safe", "match_script": "无"},
            "text_conf": 0.1, "audio_conf": 0.05, "video_conf": 0.0
        },
        {
            "round": 3, "desc": "阶段 2: 骗子开始推荐所谓的高回报理财产品",
            "llm": {"intent": "推荐理财", "risk_level": "suspicious", "match_script": "杀猪盘起手式"},
            "text_conf": 0.65, "audio_conf": 0.3, "video_conf": 0.0
        },
        {
            "round": 12, "desc": "阶段 3: 骗子发起屏幕共享，并诱导操作银行APP",
            "llm": {"intent": "屏幕共享、转账", "risk_level": "fake", "match_script": "屏幕共享诈骗"},
            "text_conf": 0.9, "audio_conf": 0.85, "video_conf": 0.8
        }
    ]

    for stage in stages:
        print(f"{"="*50}")
        print(f"🎬 {stage['desc']} (对话第 {stage['round']} 轮)")
        
        # 1. 计算融合分
        fused_score = engine.calculate_score(
            llm_classification=stage['llm'],
            local_text_conf=stage['text_conf'],
            audio_conf=stage['audio_conf'],
            video_conf=stage['video_conf']
        )
        
        # 2. 状态映射
        current_risk = map_score_to_risk_level(fused_score)
        current_depth = map_round_to_depth(stage['round'])
        
        print(f"📊 【算分结果】 综合风险分: {fused_score} -> {current_risk.name}")
        print(f"🔄 【交互深度】 {current_depth.name}")
        
        # 3. 对比不同人群的防御等级
        action_elderly = get_mock_mdp_action(vuln_elderly, current_risk, current_depth)
        action_coder = get_mock_mdp_action(vuln_coder, current_risk, current_depth)
        
        print("\n🛡️ 【MDP 动态防御决策】:")
        print(f"  🧑‍🦳 针对高危易感人群 (如老人): {action_elderly.name}")
        print(f"  🧑‍💻 针对低危防骗人群 (如程序员): {action_coder.name}")
        print(f"{"="*50}\n")

if __name__ == "__main__":
    run_simulation()