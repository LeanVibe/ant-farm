from src.cli.state.store import PlanCheckpoint, load_checkpoint, save_checkpoint
from pathlib import Path
import json


def test_checkpoint_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.cli.state.store import STATE_DIR  # re-evaluate under tmp cwd
    # Ensure state dir in tmp
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cp = PlanCheckpoint(plan_id="abc", last_completed_batch=None)
    save_checkpoint(cp)

    # File exists
    path = tmp_path / ".agent_state" / "abc.json"
    assert path.exists()

    # Load back
    loaded = load_checkpoint("abc")
    assert loaded.plan_id == "abc"
    assert loaded.last_completed_batch is None

    # Update and save atomically
    loaded.last_completed_batch = "b1"
    save_checkpoint(loaded)
    loaded2 = load_checkpoint("abc")
    assert loaded2.last_completed_batch == "b1"
