# Cursor Agent Handoff Prompt (Execution-Ready)

You are a senior engineer agent continuing work on LeanVibe Agent Hive 2.0. Operate with a plan → batch → verify approach using short-lived `uv run` commands. Persist state under `.agent_state/`. Keep PR-sized changes small and test-first.

## Ground Rules
- Package manager: `uv` (never `pip`).
- Tests: `uv run pytest` (prefer focused `-q` with `--no-cov` for speed locally).
- Plan runner: `src/cli/plan_runner.py` supports `depends_on`, `env`, `--resume`.
- Coding style: async-first, clear naming, DI-friendly seams, docstrings; match repo conventions.
- Feature flags default OFF; make additive changes.
- Prefer fakes/adapters over heavy mocks.

## Immediate Epics (execute in order)

### Epic 1: Deterministic broker pub/sub test doubles
- Implement `src/testing/fakes/fake_pubsub.py` with:
  - `class FakePubSub`:
    - `__init__(self, preseed: dict[str, list[dict]] | None = None, delay_ms: int = 0)`
    - `async def listen(self)` yields `{"channel": str, "data": bytes|str}` in FIFO order per channel
    - `def publish(self, channel: str, data: bytes|str)` adds to in-memory queue
  - Optional `advance_clock(ms:int)` or `delay_ms` to simulate timing.
- Add DI seam in broker: `MessageBroker(pubsub: Optional[PubSubProto] = None)`; default to real client; accept our fake in tests.
- Update tests to use fake where appropriate to eliminate `AsyncMock` warnings.
- Verify:
  - `uv run pytest -q tests/unit/test_message_broker_comprehensive.py --no-cov`
  - `uv run pytest -q tests/unit/test_message_broker_isolation.py --no-cov`
- Commit: `feat(testing): deterministic FakePubSub and broker DI seam`

### Epic 2: Metrics wiring + minimal exporter
- In `src/agents/base_agent.py`: add `flush_metrics_if_enabled()` to batch CLI counters and call `AsyncDatabaseManager.record_metrics_bulk` behind `HIVE_METRICS_FLUSH=1`.
- In `src/core/message_broker.py`: accumulate DLQ reason counters; periodic flush behind same flag.
- In `src/api/main.py`: extend `/api/v1/metrics` (or `/api/v1/metrics/custom`) to include these aggregates.
- Tests:
  - `tests/unit/test_base_agent.py` (monkeypatch bulk writer; assert called)
  - `tests/unit/test_message_broker_comprehensive.py` (DLQ increments persisted)
  - `tests/integration/test_api.py::test_metrics_api` (returns aggregated counters)
- Verify: run above tests focused.
- Commit: `feat(metrics): persist CLI counters and broker DLQ; expose in metrics API`

### Epic 3: Plan runner UX v3
- In `src/cli/plan_runner.py`:
  - Add `--continue` to run all remaining batches.
  - Add `--rerun-failed` to run only batches saved under `failed_batches`.
  - Validate DAG (no cycles) for `depends_on`; raise friendly error.
  - On failure, record batch under `failed_batches` and print concise summary (batch and first error line).
- In `src/cli/state/store.py`: persist `failed_batches` with backward-compatible load/save.
- Tests: extend `tests/unit/test_plan_runner.py` for continue, rerun-failed, and DAG validation.
- Verify: `uv run pytest -q tests/unit/test_plan_runner.py --no-cov`.
- Commit: `feat(plan): continue/rerun-failed flags and DAG validation`

### Epic 4: Enhanced broker structured API adoption
- Add contract tests for `BrokerSendResult` in `tests/contracts/test_component_contracts.py`.
- Migrate one internal usage (orchestrator broadcast/heartbeat) to `send_message_with_result`, logging reason on failure.
- Tests: contract + `tests/unit/test_orchestrator.py` path for structured result.
- Verify: run focused tests.
- Commit: `feat(broker): adopt structured send result in orchestrator`

## How to Work (loop)
1. Create a small plan under `plans/epic<N>.yaml` with 2–5 batches.
2. Dry-run: `uv run python -m src.cli.plan_runner run plans/epic<N>.yaml --resume`.
3. Implement the first batch; run focused tests; commit.
4. Execute the plan with `--execute --resume`; fix failures; keep commits small.
5. Open a PR after each epic with a brief summary and validation details.

## Commands
- Run a single test file: `uv run pytest -q tests/unit/test_<name>.py --no-cov`
- Run API tests: `uv run pytest -q tests/integration/test_api.py -k metrics --no-cov`
- Plan runner: `uv run python -m src.cli.plan_runner run plans/<file>.yaml --execute --resume`

## Definition of Done
- Tests added/updated and all green.
- Flags default off; no breaking changes.
- Code readable, cohesive, and DI-friendly.
- Commits conventional; PR includes rationale and validation.

## Guardrails
- Avoid long-running processes; prefer short `uv run` steps.
- Keep deltas small; do not refactor broadly.
- No `pip`; no global installs; stay within repo.
- Respect existing interfaces; prefer adapters and fakes.

## Starting Checklist (first 60–90 minutes)
- [ ] Read `docs/PLAN.md` and scan `src/core/message_broker.py`, `src/agents/base_agent.py`, `src/cli/plan_runner.py`.
- [ ] Draft `plans/epic1.yaml` with batches: fake, DI seam, tests, verify.
- [ ] Implement `FakePubSub`; run focused broker tests; commit.
- [ ] Add DI seam; switch tests; verify green; commit.
- [ ] Prepare PR with summary and validation steps.
