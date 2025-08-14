"""State store for YAML-driven plan runner.

Persists per-plan checkpoints under `.agent_state/` so runs are resumable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

STATE_DIR = Path(".agent_state")
# Defer directory creation to save_checkpoint to respect CWD in tests


@dataclass
class PlanCheckpoint:
    plan_id: str
    last_completed_batch: str | None = None
    created_by: str = "plan_runner"
    version: int = 1
    completed_batches: list[str] = field(default_factory=list)


def _checkpoint_path(plan_id: str) -> Path:
    safe = plan_id.replace("/", "_")
    return STATE_DIR / f"{safe}.json"


def load_checkpoint(plan_id: str) -> PlanCheckpoint:
    path = _checkpoint_path(plan_id)
    if not path.exists():
        return PlanCheckpoint(plan_id=plan_id)
    data = json.loads(path.read_text())
    # Back-compat: synthesize completed_batches from last_completed_batch if missing
    if "completed_batches" not in data:
        last = data.get("last_completed_batch")
        data["completed_batches"] = [last] if last else []
    return PlanCheckpoint(**data)


def save_checkpoint(cp: PlanCheckpoint) -> None:
    # Ensure state dir exists relative to current working directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(cp.plan_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(cp), indent=2))
    os.replace(tmp, path)
