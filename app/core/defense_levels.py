"""
统一防御等级定义

本模块定义了系统中所有防御等级相关的常量与映射，确保各模块使用一致的等级表示。

防御等级体系：
- MDP 层使用数字等级 (DefenseAction: 0/1/2)
- 业务层使用字符串标签 (risk_level: "low"/"medium"/"high"/"critical")
- 用户展示层使用中文标签

对应关系：
  MDP等级 (DefenseAction.value) | 系统等级 | risk_level | 中文标签
  -------------------------------------------------------------------------
  0 (LEVEL_1)                   | 1        | low        | 隐式监测
  1 (LEVEL_2)                   | 2        | high       | 强力干预  
  2 (LEVEL_3)                   | 3        | critical   | 强制阻断
"""

from typing import Dict, Any


# =============================================================================
# 核心映射表：MDP动作等级 -> 业务属性
# =============================================================================
DEFENSE_LEVEL_MAP: Dict[int, Dict[str, Any]] = {
    0: {
        "risk_level": "low",
        "label": "隐式监测",
        "system_level": 1,
        "description": "一级防御：隐式监测、侧边温和提示",
        "notify_guardian": False,
        "display_mode": "toast",
        "action": "none"
    },
    1: {
        "risk_level": "high",
        "label": "强力干预",
        "system_level": 2,
        "description": "二级防御：强力干预、多模态核验、推送案例",
        "notify_guardian": True,
        "display_mode": "popup",
        "action": "vibrate"
    },
    2: {
        "risk_level": "critical",
        "label": "强制阻断",
        "system_level": 3,
        "description": "三级防御：强制阻断、监护人联动告警",
        "notify_guardian": True,
        "display_mode": "fullscreen",
        "action": "alarm"
    }
}


# =============================================================================
# 反向映射：risk_level字符串 -> MDP动作等级
# =============================================================================
RISK_LEVEL_TO_ACTION: Dict[str, int] = {
    "safe": 0,
    "low": 0,
    "medium": 1,
    "high": 1,
    "critical": 2
}


# =============================================================================
# 工具函数
# =============================================================================
def get_defense_config(mdp_action: int) -> Dict[str, Any]:
    """
    根据MDP动作等级获取完整的防御配置
    
    Args:
        mdp_action: MDP动作等级 (0/1/2)
        
    Returns:
        包含 risk_level, label, system_level 等的配置字典
    """
    return DEFENSE_LEVEL_MAP.get(mdp_action, DEFENSE_LEVEL_MAP[0])


def get_risk_level(mdp_action: int) -> str:
    """
    根据MDP动作等级获取risk_level字符串
    
    Args:
        mdp_action: MDP动作等级 (0/1/2)
        
    Returns:
        risk_level字符串 ("low"/"high"/"critical")
    """
    config = get_defense_config(mdp_action)
    return config["risk_level"]


def get_display_mode(risk_level: str) -> str:
    """
    根据risk_level获取前端显示模式
    
    Args:
        risk_level: 风险等级字符串
        
    Returns:
        display_mode ("toast"/"popup"/"fullscreen")
    """
    mode_map = {
        "safe": "toast",
        "low": "toast",
        "medium": "popup",
        "high": "popup",
        "critical": "fullscreen"
    }
    return mode_map.get(risk_level.lower(), "toast")


def should_notify_guardian(risk_level: str) -> bool:
    """
    判断是否需要通知监护人
    
    Args:
        risk_level: 风险等级字符串
        
    Returns:
        是否需要发送监护人通知
    """
    return risk_level.lower() in ["high", "critical"]


def should_send_email(risk_level: str) -> bool:
    """
    判断是否需要发送邮件
    
    仅在 critical 等级时发送邮件
    
    Args:
        risk_level: 风险等级字符串
        
    Returns:
        是否需要发送邮件
    """
    return risk_level.lower() == "critical"
