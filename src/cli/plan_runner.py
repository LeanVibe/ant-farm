"""YAML-driven plan runner (dry-run/execute/resume).

Usage:
  uv run python -m src.cli.plan_runner run plans/epic.yaml [--execute] [--resume] [--batch name]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from .plan_schema import Plan, validate_plan_dict
from .state.store import PlanCheckpoint, load_checkpoint, save_checkpoint


class RunnerError(Exception):
    pass


def _run_cmd(cmd: str) -> int:
    """Run a short-lived shell command; return exit code."""
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode


def _find_next_batch(plan: Plan, last_completed: Optional[str], only_batch: Optional[str]) -> Optional[str]:
    if only_batch:
        return only_batch
    if last_completed is None:
        return plan.batches[0].name if plan.batches else None
    names = [b.name for b in plan.batches]
    if last_completed not in names:
        return names[0] if names else None
    idx = names.index(last_completed)
    return names[idx + 1] if idx + 1 < len(names) else None


def _exec_verify(verify_steps: list[str]) -> None:
    for step in verify_steps:
        code = _run_cmd(step)
        if code != 0:
            raise RunnerError(f"verify failed: {step}")


def _apply_tasks_dry_run(plan_path: Path, batch_name: str) -> None:
    print(f"[dry-run] Would apply batch '{batch_name}' from {plan_path}")


def _apply_tasks_execute(plan_path: Path, batch_name: str) -> None:
    # Intentionally minimal scaffold: real edits are executed by dev flow/tools.
    print(f"[execute] Applying batch '{batch_name}' from {plan_path}")


def run_plan(plan_path: Path, execute: bool, resume: bool, only_batch: Optional[str]) -> int:
    plan_dict = yaml.safe_load(plan_path.read_text())
    plan = validate_plan_dict(plan_dict)

    cp = load_checkpoint(plan.plan_id)
    last_completed = cp.last_completed_batch if resume else None

    next_batch_name = _find_next_batch(plan, last_completed, only_batch)
    if not next_batch_name:
        print("No batch to run (plan complete or empty).")
        return 0

    # Locate batch
    batch = next(b for b in plan.batches if b.name == next_batch_name)

    # 1) Apply
    if execute:
        _apply_tasks_execute(plan_path, batch.name)
    else:
        _apply_tasks_dry_run(plan_path, batch.name)

    # 2) Verify
    verify_cmds = [v.run for v in batch.verify]
    if verify_cmds:
        _exec_verify(verify_cmds)

    # 3) Commit
    if execute and batch.commit:
        message = batch.commit.message
        # Commit any staged/untracked changes in workflows, tests, or src
        commit_cmd = (
            "git add -A && git commit -m \"$(cat <<'EOF'\n"
            + message
            + "\nEOF\n)\""
        )
        code = _run_cmd(commit_cmd)
        if code != 0:
            raise RunnerError("commit failed")

    # 4) Update checkpoint
    cp.last_completed_batch = batch.name
    save_checkpoint(cp)
    print(f"Batch '{batch.name}' completed.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="YAML-driven plan runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a plan batch")
    run_p.add_argument("plan", type=str)
    run_p.add_argument("--execute", action="store_true")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--batch", type=str, default=None, help="Run only this batch name")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return run_plan(Path(args.plan), args.execute, args.resume, args.batch)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
