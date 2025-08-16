import json
from pathlib import Path

import pytest

from src.cli.plan_runner import main, RunnerError


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
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "plan_id: p-min",
                "goal: test",
                "batches:",
                "  - name: b1",
                "    tasks:",
                "      - type: command",
                "        run: echo hi",
                "    verify:",
                '      - run: "true"',
                "    commit:",
                "      message: 'chore: test'",
            ]
        )
    )


class TestMultiBatch:
    def test_dependency_blocking(self, tmp_path: Path, monkeypatch):
        plan = tmp_path / "p.yaml"
        plan.write_text(
            "\n".join(
                [
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
                    '      - run: "true"',
                ]
            )
        )
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


def test_dag_validation(tmp_path: Path, monkeypatch):
    """Test DAG validation detects cycles."""
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "\n".join(
            [
                "version: 1",
                "plan_id: p-cycle",
                "goal: test cycle detection",
                "batches:",
                "  - name: a",
                "    depends_on: ['c']",
                "    tasks: []",
                "  - name: b",
                "    depends_on: ['a']",
                "    tasks: []",
                "  - name: c",
                "    depends_on: ['b']",
                "    tasks: []",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    # Should raise RunnerError due to cycle
    with pytest.raises(RunnerError, match="Plan contains dependency cycles"):
        main(["run", str(plan)])


def test_dag_validation_missing_dependency(tmp_path: Path, monkeypatch):
    """Test DAG validation detects missing dependencies."""
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "\n".join(
            [
                "version: 1",
                "plan_id: p-missing",
                "goal: test missing dependency",
                "batches:",
                "  - name: a",
                "    depends_on: ['nonexistent']",
                "    tasks: []",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    # Should raise RunnerError due to missing dependency
    with pytest.raises(RunnerError, match="Dependency 'nonexistent' not found"):
        main(["run", str(plan)])


def test_failed_batches_tracking(tmp_path: Path, monkeypatch):
    """Test failed batches are recorded and can be rerun."""
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "\n".join(
            [
                "version: 1",
                "plan_id: p-failed",
                "goal: test failed batches",
                "batches:",
                "  - name: failing",
                "    tasks: []",
                "    verify:",
                '      - run: "false"',  # This will fail
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    # First run should fail
    rc = main(["run", str(plan), "--execute"])
    assert rc == 1

    # Check checkpoint has failed batch recorded
    cp = json.loads((tmp_path / ".agent_state" / "p-failed.json").read_text())
    assert "failing" in cp["failed_batches"]
    assert "failing" in cp["failed_batch_errors"]

    # Should be able to rerun failed batch
    rc = main(["run", str(plan), "--rerun-failed", "--execute"])
    assert rc == 1  # Still fails, but that's expected
