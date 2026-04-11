"""
检测任务门禁：环境识别完成态、OCR 聊天提取互斥、与 Redis 键约定。
"""
from __future__ import annotations

import redis as redis_sync
from app.core.config import settings
from app.core.logger import get_logger
from app.core.redis import get_redis

logger = get_logger(__name__)

ENV_READY_TTL_SECONDS = 6 * 3600
OCR_DIALOGUE_INFLIGHT_TTL_SECONDS = 600
ENV_CLASSIFY_INFLIGHT_TTL_SECONDS = 600

_sync_redis = None


def _get_sync_redis():
    global _sync_redis
    if _sync_redis is None:
        _sync_redis = redis_sync.from_url(settings.REDIS_URL, decode_responses=True)
    return _sync_redis


def env_ready_key(call_id: int) -> str:
    return f"call:{call_id}:env_ready"


def ocr_dialogue_inflight_key(call_id: int) -> str:
    return f"call:{call_id}:ocr_dialogue_inflight"


def env_classify_inflight_key(call_id: int) -> str:
    return f"call:{call_id}:env_classify_inflight"


async def is_env_recognition_ready(call_id: int) -> bool:
    redis = await get_redis()
    v = await redis.get(env_ready_key(call_id))
    return v == "1"


async def env_type_is_text_chat(call_id: int) -> bool:
    redis = await get_redis()
    t = await redis.get(f"call:{call_id}:env_type")
    return t == "text_chat"


async def get_call_env_type(call_id: int) -> str:
    redis = await get_redis()
    return (await redis.get(f"call:{call_id}:env_type")) or "unknown"


async def try_acquire_ocr_dialogue_inflight(call_id: int) -> bool:
    redis = await get_redis()
    ok = await redis.set(
        ocr_dialogue_inflight_key(call_id),
        "1",
        ex=OCR_DIALOGUE_INFLIGHT_TTL_SECONDS,
        nx=True,
    )
    return bool(ok)


async def release_ocr_dialogue_inflight(call_id: int) -> None:
    redis = await get_redis()
    await redis.delete(ocr_dialogue_inflight_key(call_id))


async def try_acquire_env_classify_inflight(call_id: int) -> bool:
    redis = await get_redis()
    ok = await redis.set(
        env_classify_inflight_key(call_id),
        "1",
        ex=ENV_CLASSIFY_INFLIGHT_TTL_SECONDS,
        nx=True,
    )
    return bool(ok)


async def release_env_classify_inflight(call_id: int) -> None:
    redis = await get_redis()
    await redis.delete(env_classify_inflight_key(call_id))


def sync_set_env_recognition_ready(call_id: int) -> None:
    _get_sync_redis().setex(env_ready_key(call_id), ENV_READY_TTL_SECONDS, "1")


def sync_is_env_recognition_ready(call_id: int) -> bool:
    return _get_sync_redis().get(env_ready_key(call_id)) == "1"


def sync_env_type_is_text_chat(call_id: int) -> bool:
    return _get_sync_redis().get(f"call:{call_id}:env_type") == "text_chat"


def sync_get_call_env_type(call_id: int) -> str:
    return _get_sync_redis().get(f"call:{call_id}:env_type") or "unknown"


def sync_try_acquire_ocr_dialogue_inflight(call_id: int) -> bool:
    return bool(
        _get_sync_redis().set(
            ocr_dialogue_inflight_key(call_id),
            "1",
            ex=OCR_DIALOGUE_INFLIGHT_TTL_SECONDS,
            nx=True,
        )
    )


def sync_release_ocr_dialogue_inflight(call_id: int) -> None:
    try:
        _get_sync_redis().delete(ocr_dialogue_inflight_key(call_id))
    except Exception as e:
        logger.warning(f"释放 OCR 互斥失败 call_id={call_id}: {e}")


def sync_release_env_classify_inflight(call_id: int) -> None:
    try:
        _get_sync_redis().delete(env_classify_inflight_key(call_id))
    except Exception as e:
        logger.warning(f"释放环境识别互斥失败 call_id={call_id}: {e}")
