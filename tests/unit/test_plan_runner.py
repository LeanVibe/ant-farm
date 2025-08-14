import json
from pathlib import Path

import pytest

from src.cli.plan_runner import main


MIN_PLAN = {
    "version": 1,
    "plan_id": "p-min",
    "goal": "test",
    "batches": [
        {
            "name": "b1",
            "tasks": [{"type": "command", "run": "echo hi"}],
            "verify": [{"run": "true"}],
            "commit": {"message": "chore: test"},
        }
    ],
}


def write_yaml(path: Path, data: dict):
    path.write_text("\n".join([
        "version: 1",
        "plan_id: p-min",
        "goal: test",
        "batches:",
        "  - name: b1",
        "    tasks:",
        "      - type: command",
        "        run: echo hi",
        "    verify:",
        "      - run: \"true\"",
        "    commit:",
        "      message: 'chore: test'",
    ]))


class TestMultiBatch:
    def test_dependency_blocking(self, tmp_path: Path, monkeypatch):
        plan = tmp_path / "p.yaml"
        plan.write_text("\n".join([
            "version: 1",
            "plan_id: p",
            "goal: g",
            "batches:",
            "  - name: a",
            "    tasks: []",
            "  - name: b",
            "    depends_on: ['a']",
            "    tasks:",
            "      - type: command",
            "        run: echo hi",
            "    verify:",
            "      - run: \"true\"",
        ]))
        monkeypatch.chdir(tmp_path)
        # Without resume and last_completed, selecting b directly should error on deps
        with pytest.raises(Exception):
            main(["run", str(plan), "--batch", "b"])  # no --resume


@pytest.mark.parametrize("execute", [False])
def test_run_dry_run(tmp_path: Path, monkeypatch, execute):
    plan = tmp_path / "plan.yaml"
    write_yaml(plan, MIN_PLAN)
    monkeypatch.chdir(tmp_path)

    args = ["run", str(plan)]
    args += ["--execute"] if execute else ["--resume"]
    rc = main(args)
    assert rc == 0
    # checkpoint written
    cp = json.loads((tmp_path / ".agent_state" / "p-min.json").read_text())
    assert cp["last_completed_batch"] == "b1"
