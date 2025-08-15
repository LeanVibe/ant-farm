# Cursor Agent Handoff Prompt

You are a senior engineer agent continuing work on LeanVibe Agent Hive 2.0. Adopt the plan→batch→verify approach with short-lived `uv run` commands and persist state under `.agent_state/`. Keep PR-sized changes small and test-first.

## Context
- Package manager: `uv`
- Test: `uv run pytest`
- Plan runner: `src/cli/plan_runner.py` (supports depends_on/env, resume)
- Metrics bulk helper: `AsyncDatabaseManager.record_metrics_bulk`
- Brokers: `MessageBroker` and `EnhancedMessageBroker`; structured API `send_message_with_result` exists (legacy bool remains)
- Orchestrator: clean shutdown implemented; tracks background tasks

## Immediate Epics to Execute (in order)

### Epic 1: Deterministic broker pub/sub test doubles
- Implement `src/testing/fakes/fake_pubsub.py` that provides:
  - `listen()` async generator yielding pre-seeded messages
  - `publish(channel, data)` to append messages
- Add DI seam in `MessageBroker` to inject pubsub/client under test (e.g., optional ctor args or setters).
- Update `tests/unit/test_message_broker_comprehensive.py` to use the fake where appropriate and remove AsyncMock warnings.
- Verify: `uv run pytest -q tests/unit/test_message_broker_comprehensive.py --no-cov`
- Commit: `feat(testing): add deterministic pubsub fake and inject into broker`

### Epic 2: Metrics wiring + minimal exporter
- Wire selected counters to DB using `record_metrics_bulk`:
  - In `BaseAgent` (CLI tool counters): create a flush method that converts counters to metric dicts and calls bulk write (feature flag `HIVE_METRICS_FLUSH=1`).
  - In `MessageBroker` DLQ path: accumulate reason counters and flush periodically via bulk write (feature flag).
- Extend `/api/v1/metrics` (or add `/api/v1/metrics/custom`) to include aggregated counters (read from DB or in-memory if flag disabled).
- Tests:
  - Counter flush triggers `record_metrics_bulk` (monkeypatch DB manager).
  - DLQ increments persisted.
  - Endpoint returns aggregated counters.
- Verify: run targeted tests.
- Commit: `feat(metrics): persist CLI counters and broker DLQ via bulk write; expose in metrics API`

### Epic 3: Plan runner UX v3
- `src/cli/plan_runner.py`:
  - Add `--continue` to execute all remaining batches.
  - Add `--rerun-failed` to run only batches recorded in `failed_batches`.
  - Validate plan DAG for cycles; raise friendly error.
  - On failure, record batch under `failed_batches` and print a concise summary.
- `src/cli/state/store.py`: add `failed_batches: list[str]` and load/save with back-compat.
- Tests: extend `tests/unit/test_plan_runner.py` for continue, rerun-failed, and DAG validation.
- Verify: `uv run pytest -q tests/unit/test_plan_runner.py --no-cov`
- Commit: `feat(plan): add continue/rerun-failed and DAG validation`

### Epic 4: Enhanced broker structured API adoption
- Add contract tests that, when `send_message_with_result` exists, assert `BrokerSendResult` fields and reasons.
- Migrate one internal site (e.g., orchestrator broadcast/ping) to use structured API and log reason on failure.
- Verify: orchestrator and contract tests.
- Commit: `feat(broker): adopt structured result in orchestrator broadcast path`

## Operational Guidelines
- Use small, resumable YAML plans in `plans/` and run via the plan runner.
- After each change, run focused tests and keep suites green.
- Prefer DI and fakes over heavy mocks for reliability.
- Enforce conventional commits.
- Push after each epic; open a PR when the series is complete.

## Commands Cheat Sheet
- Dry-run a plan: `uv run python -m src.cli.plan_runner run plans/<file>.yaml --resume`
- Execute a plan: `uv run python -m src.cli.plan_runner run plans/<file>.yaml --execute --resume`
- Run focused tests: `uv run pytest -q tests/unit/test_<name>.py --no-cov`

## Definition of Done
- Tests added/updated, all green.
- Code is readable, minimal, and cohesive.
- Changes are committed with descriptive messages and pushed.
- If adding flags, defaults must be safe (features off by default).
