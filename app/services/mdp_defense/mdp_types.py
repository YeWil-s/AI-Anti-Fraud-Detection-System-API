from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class UserVulnerability(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class InteractionDepth(str, Enum):
    SHALLOW = "SHALLOW"
    MEDIUM = "MEDIUM"
    DEEP = "DEEP"


class EnvironmentType(str, Enum):
    TEXT_CHAT = "TEXT_CHAT"
    VOICE_CHAT = "VOICE_CHAT"
    PHONE_CALL = "PHONE_CALL"
    VIDEO_CALL = "VIDEO_CALL"
    UNKNOWN = "UNKNOWN"


class RuleHitLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DefenseAction(Enum):
    LEVEL_1 = 0
    LEVEL_2 = 1
    LEVEL_3 = 2


@dataclass
class MDPObservation:
    user_id: int
    call_id: int
    risk_score: float
    message_count: int
    environment_type: str = "unknown"
    audio_risk_flag: bool = False
    video_risk_flag: bool = False
    rule_hit_level: int = 0
    text_conf: float = 0.0
    audio_conf: float = 0.0
    video_conf: float = 0.0
    llm_result: Dict[str, Any] = field(default_factory=dict)
    rule_hit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MDPState:
    user_vulnerability: UserVulnerability
    risk_level: RiskLevel
    interaction_depth: InteractionDepth
    environment_type: EnvironmentType
    audio_risk_flag: bool
    video_risk_flag: bool
    rule_hit_level: RuleHitLevel

    def policy_key(self) -> str:
        return (
            f"{self.user_vulnerability.value}_"
            f"{self.risk_level.value}_"
            f"{self.interaction_depth.value}_"
            f"{self.environment_type.value}_"
            f"A{int(self.audio_risk_flag)}_"
            f"V{int(self.video_risk_flag)}_"
            f"{self.rule_hit_level.value}"
        )

    def legacy_policy_key(self) -> str:
        return (
            f"{self.user_vulnerability.value}_"
            f"{self.risk_level.value}_"
            f"{self.interaction_depth.value}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_vulnerability": self.user_vulnerability.value,
            "risk_level": self.risk_level.value,
            "interaction_depth": self.interaction_depth.value,
            "environment_type": self.environment_type.value,
            "audio_risk_flag": self.audio_risk_flag,
            "video_risk_flag": self.video_risk_flag,
            "rule_hit_level": self.rule_hit_level.value,
            "state_key": self.policy_key(),
            "legacy_state_key": self.legacy_policy_key(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MDPState":
        return cls(
            user_vulnerability=UserVulnerability(data["user_vulnerability"]),
            risk_level=RiskLevel(data["risk_level"]),
            interaction_depth=InteractionDepth(data["interaction_depth"]),
            environment_type=EnvironmentType(data.get("environment_type", EnvironmentType.UNKNOWN.value)),
            audio_risk_flag=bool(data.get("audio_risk_flag", False)),
            video_risk_flag=bool(data.get("video_risk_flag", False)),
            rule_hit_level=RuleHitLevel(data.get("rule_hit_level", RuleHitLevel.NONE.value)),
        )


@dataclass
class PolicyDecision:
    state: MDPState
    action: DefenseAction
    policy_version: str
    reason_codes: List[str]
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "action_level": self.action.value,
            "action_name": self.action.name,
            "policy_version": self.policy_version,
            "reason_codes": self.reason_codes,
            "fallback_used": self.fallback_used,
        }
