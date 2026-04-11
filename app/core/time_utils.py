"""
北京时间工具函数（Asia/Shanghai）。
统一提供带时区的当前时间与序列化输出。
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def now_bj() -> datetime:
    """返回带时区的北京时间。"""
    return datetime.now(BEIJING_TZ)


def ensure_bj(dt: datetime | None) -> datetime | None:
    """将任意 datetime 规范化为北京时间。

    约定：若 dt 无 tzinfo，按 UTC 解释后转换为北京时间。
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(BEIJING_TZ)
    return dt.astimezone(BEIJING_TZ)


def isoformat_bj(dt: datetime | None) -> str | None:
    """以 ISO8601 输出北京时间（含 +08:00）。"""
    normalized = ensure_bj(dt)
    return normalized.isoformat() if normalized else None
