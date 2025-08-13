# LeanVibe Agent Hive 2.0 - Handoff Prompt

## 🎯 Current status: Epic 1 complete, test shims in place

You are taking over development of LeanVibe Agent Hive 2.0, a self-improving autonomous multi-agent system.

- Epic 1 (Process Resilience) is complete and pushed to `main`.
- Orchestrator/tmux resilience is implemented; unit tests pass locally when run without the global coverage gate.
- Test-aligned shims were added to unblock suites for collaboration and knowledge base.

### System snapshot
- ✅ Orchestrator stabilized for tests (registry, spawner, health monitor)
- ✅ Tmux resilience via `RetryableTmuxManager`; orchestrator uses subprocess in tests for compatibility
- ✅ Realtime collaboration suite unblocked via `RealTimeCollaborationManager`
- ✅ Knowledge base suite unblocked via extended `SharedKnowledgeBase` test API
- ⚠️ Coverage gate (<50%) blocks full-suite; use focused runs with `--no-cov` during hardening (see PLAN)

### Branch & environment
- Branch: `main`
- Service ports: API(9001), PostgreSQL(5433), Redis(6381), pgAdmin(9050)

---

## Your next priorities (choose a path)

OPTION A: Coverage & CI Hardening (Phase 5)
- Establish coverage roadmap for high-signal modules (`orchestrator`, `tmux_manager`, `message_broker`, `caching`, `async_db`)
- Add smoke tests to lift coverage floor on legacy modules
- Introduce `.coveragerc` and split CI jobs (fast vs nightly coverage)
- Document local guidance for focused runs

OPTION B: Stabilize Tmux Bridge for Prod (Phase 4.1.1)
- Add `TmuxBackendProtocol`; make `AgentSpawner` backend-injectable
- Default to resilient manager; tests supply subprocess backend
- Expand `RetryableTmuxManager` tests: timeouts, optimistic tracking, termination idempotency
- Feature-flag orphaned-session cleanup (`HIVE_CLEANUP_ORPHANS=1`)

OPTION C: Test‑Shim Consolidation (Phase 6)
- Introduce `CollaborationSyncService` and `KnowledgeBaseService` interfaces
- Refactor tests to depend on interfaces; production binds full impls
- Remove direct references to test helpers from prod paths

See `docs/PLAN.md` (Phase 4/5/6) for details and acceptance criteria.

Key suites to keep green as you iterate:
- `tests/unit/test_orchestrator.py`
- `tests/unit/test_tmux_manager.py`
- `tests/unit/test_realtime_collaboration.py`
- `tests/unit/test_shared_knowledge_base.py`

---

## Coverage strategy (Phase 5)
- Target: restore CI by lifting coverage ≥50% first; then aim for 80%+
- Prioritize orchestrator/tmux_manager; add smoke tests to large legacy files

Recent highlights
- Fixed Redis port conflicts (6381 vs 6379)
- Orchestrator registry/spawner/health monitor stabilized for tests
- Retryable tmux manager implemented and refined
- Test shims added for collaboration/knowledge base suites

### Key files & areas
- Coverage/Hardening: `src/core/orchestrator.py`, `src/core/tmux_manager.py`, `src/core/message_broker.py`, `src/core/caching.py`, `src/core/async_db.py`, `tests/`
- Tmux Bridge: `src/core/orchestrator.py`, `src/core/tmux_manager.py`
- Shim Consolidation: `src/core/realtime_collaboration.py`, `src/core/shared_knowledge_base.py`

---

## Success metrics for your session

Path A: Coverage & CI
- ✅ Coverage ≥50% in full CI by adding targeted tests and `.coveragerc`
- ✅ No regressions in orchestrator/tmux suites
- ✅ Documented developer guidance for focused runs

Path B: Tmux Bridge Injection
- ✅ Spawner uses injectable backend; prod uses resilient manager
- ✅ Expanded tmux manager tests for timeouts/validation/termination
- ✅ Orphan cleanup behind feature flag

Path C: Shim Consolidation
- ✅ Services behind interfaces; tests bind test adapters
- ✅ No direct imports from test helpers in prod paths
- ✅ Contracts documented in `tests/contracts/`

---

## Getting started commands

Quick local check
```bash
# from repo root
git status && git log --oneline -3
pytest -q tests/unit/test_orchestrator.py tests/unit/test_tmux_manager.py --no-cov
pytest -q tests/unit/test_realtime_collaboration.py tests/unit/test_shared_knowledge_base.py --no-cov
```

Choose your path

Path A - Coverage & CI
```bash
# Inspect near-term plan excerpt
sed -n '660,760p' docs/PLAN.md
# Run focused coverage locally as you add tests
pytest --cov=src/core --cov-report=term-missing -k "orchestrator or tmux_manager"
```

Path B - Tmux Bridge Injection
```bash
# Green suites while iterating
pytest -q tests/unit/test_tmux_manager.py tests/unit/test_orchestrator.py --no-cov
# After adding backend protocol
pytest -q tests/unit/test_tmux_manager.py -q
```

Path C - Shim Consolidation
```bash
# Find direct references to test helpers
rg "RealTimeCollaborationManager|SharedKnowledgeBase" -n src | sed -n '1,200p'
```

Current CI constraints & solutions
- Coverage gate <50% is blocking; many legacy modules at 0%
- Workaround: run focused suites with `--no-cov` locally while adding coverage
- Next: introduce `.coveragerc` and split CI jobs (fast vs nightly)

---

## Principles & guardrails
- TDD; async I/O; comprehensive typing; meaningful error handling
- Keep code simple, well-tested, and thoroughly documented
- Every change should make the system easier for future agents to understand and enhance

Good luck. Keep suites green, raise coverage pragmatically, and harden the tmux bridge for production usage. 