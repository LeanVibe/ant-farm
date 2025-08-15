"""Hive CLI wrappers for YAML plan runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..plan_runner import main as runner_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hive plan", description="Plan runner commands")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Execute a plan batch")
    run_p.add_argument("plan", type=str)
    run_p.add_argument("--execute", action="store_true")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--batch", type=str, default=None)

    dry_p = sub.add_parser("dry-run", help="Dry-run a plan batch")
    dry_p.add_argument("plan", type=str)
    dry_p.add_argument("--resume", action="store_true")
    dry_p.add_argument("--batch", type=str, default=None)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return runner_main(["run", args.plan, *(["--execute"]), *(["--resume"] if args.resume else []), *(["--batch", args.batch] if args.batch else [])])
    if args.cmd == "dry-run":
        return runner_main(["run", args.plan, *(["--resume"] if args.resume else []), *(["--batch", args.batch] if args.batch else [])])

    return 0
