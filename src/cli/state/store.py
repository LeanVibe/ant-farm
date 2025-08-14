"""State store for YAML-driven plan runner.

Persists per-plan checkpoints under `.agent_state/` so runs are resumable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

STATE_DIR = Path(".agent_state")
# Defer directory creation to save_checkpoint to respect CWD in tests


@dataclass
class PlanCheckpoint:
    plan_id: str
    last_completed_batch: Optional[str] = None
    created_by: str = "plan_runner"
    version: int = 1


def _checkpoint_path(plan_id: str) -> Path:
    safe = plan_id.replace("/", "_")
    return STATE_DIR / f"{safe}.json"


def load_checkpoint(plan_id: str) -> PlanCheckpoint:
    path = _checkpoint_path(plan_id)
    if not path.exists():
        return PlanCheckpoint(plan_id=plan_id)
    data = json.loads(path.read_text())
    return PlanCheckpoint(**data)


def save_checkpoint(cp: PlanCheckpoint) -> None:
    # Ensure state dir exists relative to current working directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(cp.plan_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(cp), indent=2))
    os.replace(tmp, path)
