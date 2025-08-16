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
import os

import yaml

from .plan_schema import Plan, validate_plan_dict
from .state.store import PlanCheckpoint, load_checkpoint, save_checkpoint


class RunnerError(Exception):
    pass


def _run_cmd(cmd: str) -> int:
    """Run a short-lived shell command; return exit code."""
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode


def _validate_dag(plan: Plan) -> None:
    """Validate DAG has no cycles."""
    from collections import defaultdict, deque

    # Build graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_batches = {batch.name for batch in plan.batches}

    for batch in plan.batches:
        batch_name = batch.name
        deps = getattr(batch, "depends_on", [])
        for dep in deps:
            if dep not in all_batches:
                raise RunnerError(f"Dependency '{dep}' not found in plan batches")
            graph[dep].append(batch_name)
            in_degree[batch_name] += 1
        if batch_name not in in_degree:
            in_degree[batch_name] = 0

    # Topological sort to detect cycles
    queue = deque([name for name in in_degree if in_degree[name] == 0])
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(plan.batches):
        raise RunnerError("Plan contains dependency cycles")


def _find_next_batch(
    plan: Plan,
    last_completed: Optional[str],
    only_batch: Optional[str],
    continue_mode: bool = False,
    rerun_failed: bool = False,
    cp: Optional[PlanCheckpoint] = None,
) -> Optional[str]:
    if only_batch:
        return only_batch

    if rerun_failed and cp and cp.failed_batches:
        return cp.failed_batches[0]

    if continue_mode and last_completed:
        # Find all remaining batches after last_completed
        names = [b.name for b in plan.batches]
        if last_completed in names:
            idx = names.index(last_completed)
            remaining = names[idx + 1 :]
            # Filter out already completed batches
            if cp:
                remaining = [
                    name for name in remaining if name not in cp.completed_batches
                ]
            return remaining[0] if remaining else None

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


def run_plan(
    plan_path: Path,
    execute: bool,
    resume: bool,
    only_batch: Optional[str],
    continue_mode: bool = False,
    rerun_failed: bool = False,
) -> int:
    plan_dict = yaml.safe_load(plan_path.read_text())
    plan = validate_plan_dict(plan_dict)

    # Validate DAG
    _validate_dag(plan)

    cp = load_checkpoint(plan.plan_id)
    last_completed = cp.last_completed_batch if resume else None

    next_batch_name = _find_next_batch(
        plan, last_completed, only_batch, continue_mode, rerun_failed, cp
    )
    if not next_batch_name:
        print("No batch to run (plan complete or empty).")
        return 0

    # Locate batch
    batch = next(b for b in plan.batches if b.name == next_batch_name)

    # Check dependencies
    completed = set(cp.completed_batches)
    if resume and cp.last_completed_batch and not cp.completed_batches:
        # fallback for old checkpoints without completed_batches
        names = [b.name for b in plan.batches]
        try:
            idx = names.index(cp.last_completed_batch)
            completed = set(names[: idx + 1])
        except ValueError:
            completed = set()
    unmet = [d for d in getattr(batch, "depends_on", []) if d not in completed]
    if unmet:
        raise RunnerError(f"dependencies not met for {batch.name}: {unmet}")

    # Run batch with error handling
    try:
        # 1) Apply
        # Inject env for this batch
        for k, v in getattr(batch, "env", {}).items():
            os.environ[str(k)] = str(v)
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
                "git add -A && git commit -m \"$(cat <<'EOF'\n" + message + '\nEOF\n)"'
            )
            code = _run_cmd(commit_cmd)
            if code != 0:
                raise RunnerError("commit failed")

        # 4) Update checkpoint
        cp.last_completed_batch = batch.name
        if batch.name not in cp.completed_batches:
            cp.completed_batches.append(batch.name)
        # Remove from failed batches if it was there
        if batch.name in cp.failed_batches:
            cp.failed_batches.remove(batch.name)
            cp.failed_batch_errors.pop(batch.name, None)
        save_checkpoint(cp)
        print(f"Batch '{batch.name}' completed.")
        return 0

    except RunnerError as e:
        # Record failed batch
        if batch.name not in cp.failed_batches:
            cp.failed_batches.append(batch.name)
        cp.failed_batch_errors[batch.name] = str(e)
        save_checkpoint(cp)

        # Print error summary
        print(f"Batch '{batch.name}' failed: {e}")
        if rerun_failed:
            print("Failed batches recorded. Use --rerun-failed to retry.")
        return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="YAML-driven plan runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a plan batch")
    run_p.add_argument("plan", type=str)
    run_p.add_argument("--execute", action="store_true")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument(
        "--batch", type=str, default=None, help="Run only this batch name"
    )
    run_p.add_argument(
        "--continue", action="store_true", help="Run all remaining batches"
    )
    run_p.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Run only batches saved under failed_batches",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return run_plan(
            Path(args.plan),
            args.execute,
            args.resume,
            args.batch,
            getattr(args, "continue", False),
            args.rerun_failed,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
