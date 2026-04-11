from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mdp_decision_event import MDPDecisionEvent
from .mdp_types import DefenseAction, MDPState


@dataclass
class TrajectorySample:
    state: MDPState
    action: DefenseAction
    reward: float
    next_state: MDPState


class OfflineTrajectoryDataset:
    async def load_completed_samples(self, db: AsyncSession) -> List[TrajectorySample]:
        result = await db.execute(
            select(MDPDecisionEvent).where(
                MDPDecisionEvent.next_state_json.is_not(None),
                MDPDecisionEvent.reward.is_not(None),
            )
        )
        events = result.scalars().all()

        samples: List[TrajectorySample] = []
        for event in events:
            try:
                state = MDPState.from_dict(json.loads(event.state_json))
                next_state = MDPState.from_dict(json.loads(event.next_state_json))
                action = DefenseAction(event.action_level)
                samples.append(
                    TrajectorySample(
                        state=state,
                        action=action,
                        reward=float(event.reward),
                        next_state=next_state,
                    )
                )
            except Exception:
                continue
        return samples


offline_dataset = OfflineTrajectoryDataset()
