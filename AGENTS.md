# AGENTS.md

# Repository Guidelines

## Project Structure & Modules
- `src/`: application code
  - `src/cli/`: `hive` CLI entrypoint and commands
  - `src/api/`: FastAPI app, middleware, handlers
  - `src/agents/`: meta/architect/qa/devops agents
  - `src/core/`: broker, orchestration, coordination, monitoring
  - `src/web/`: dashboard static assets
- `tests/`: unit and integration tests (`pytest` configured in `pyproject.toml`)
- `docker/`, `scripts/`, `plans/`, `logs/`: infra, helpers, plans, runtime logs

## Build, Test, and Development
- Prefer `hive` CLI:
  - Init: `hive init`
  - Start/Status/Stop: `hive system start | status | stop`
  - Agents: `hive agent spawn meta`, `hive agent list`
  - Tasks: `hive task submit "Analyze module X"`
  - Monitoring: `hive monitor agents --live`, `hive monitor tasks --live`
  - Batch Ops: `hive agent batch-spawn --file agents.json`
  - Metrics: `hive metrics agents`, `hive metrics tasks`
- Python tools:
  - Tests: `pytest -q` (or `pytest -q tests/unit`)
  - Coverage: `pytest --cov=src --cov-report=term-missing`
  - Lint/Format: `ruff check .`; `ruff format .`
  - Types: `mypy .`
- Docker infra (dev): `docker compose up -d postgres redis`; logs: `docker compose logs -f`

## Coding Style & Naming
- Python 3.11, line length 88. Use absolute imports, no wildcards.
- Naming: snake_case (funcs/vars), PascalCase (classes), UPPER_SNAKE (consts).
- Keep modules small, focused; prefer typed `async` boundaries; avoid blocking.
- Secrets/config only via `.env`; never commit or log secrets.

## Testing Guidelines
- Framework: `pytest` with `tests/` discovery (`test_*.py`, `Test*`, `test_*`).
- Coverage target: configured at 50% minimum (raise with stability).
- Write unit tests first; use HTTPX for API tests; provide fakes over mocks where possible.
- Run: `pytest -q` locally; for APIs, verify `/api/v1/health` when system runs.

## Commit & Pull Requests
- Use Conventional Commits: `feat(scope): summary`, `fix`, `docs`, `test`, `refactor`, `chore`.
- PRs: clear description, linked issues, test coverage for changes, and notes on migrations/infra.
- Keep changes scoped; update docs where relevant (e.g., `README.md`, `docs/`).

## Security & Configuration
- Default ports: API `9001`, Postgres `5433`, Redis `6381` (via Docker). Optional tools: pgAdmin `9050`, Redis Commander `9081`.
- Validate environment via `hive init`; review `.env.example` to configure credentials.
