from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from app.core.time_utils import now_bj


class PolicyRepository:
    def __init__(self, artifact_path: str | None = None):
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parents[2]
        if artifact_path:
            candidate = Path(artifact_path)
            self.artifact_path = candidate if candidate.is_absolute() else project_root / candidate
        else:
            self.artifact_path = base_dir / "q_table_policy.json"

    def load_policy(self) -> Tuple[Dict[str, int], str]:
        if not self.artifact_path.exists():
            return {}, "fallback-only"

        with open(self.artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "policy" in data:
            return data.get("policy", {}), data.get("version", "policy-v2")

        # 兼容旧版纯 state_key -> action 的策略表
        return data, "legacy-q-table"

    def save_policy(self, policy: Dict[str, int], metadata: Dict[str, Any] | None = None) -> str:
        version = f"policy-{now_bj().strftime('%Y%m%d%H%M%S')}"
        payload = {
            "version": version,
            "created_at": now_bj().isoformat(),
            "state_schema_version": "v2",
            "policy": policy,
            "metadata": metadata or {},
        }
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return version
