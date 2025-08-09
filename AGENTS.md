# AGENTS.md

## LeanVibe Hive CLI Commands (Primary Interface)

**🎯 Always use the `hive` CLI - it's the proper way to manage the system!**

**Initialize & Start System:**
- Setup: `hive init` (initializes DB, migrations, Docker services)
- Start system: `hive system start` (API server, core services)
- System status: `hive system status` (health check all services)
- Stop system: `hive system stop` (graceful shutdown)

**Agent Management:**
- Spawn agent: `hive agent spawn meta` (or architect, qa, devops)
- List agents: `hive agent list`
- Agent details: `hive agent describe <name>`
- Stop agent: `hive agent stop <name>`
- Bootstrap system: `hive agent bootstrap`

**Task Management:**
- Submit task: `hive task submit "description"`
- List tasks: `hive task list`
- Task status: `hive task status <id>`

**Context & Memory:**
- Populate context: `hive context populate`
- Search context: `hive context search "query"`

## 🔧 Service Ports Configuration

**Current service port configuration:**
- **API**: 9001 (non-standard for security)
- **PostgreSQL**: 5432 (standard port)
- **Redis**: 6379 (standard port)
- **pgAdmin**: 9050 (development only)

## CLI Agentic Coding Tools

- Auto-detection: system checks for opencode, claude, gemini CLI tools
- Priority order: opencode (preferred) → claude → gemini → API fallback
- Check tools: `hive agent tools` (or `make tools`)
- Force specific: export PREFERRED_CLI_TOOL=opencode

## Development Commands (Alternative)

- Setup: make setup (auto-detects CLI tools); env: .env (API keys optional)
- Dev API: `hive system start-api --port 9001 --reload`
- Tests (all): pytest -q
- Single test: pytest -q path/to/test_file.py::TestClass::test_case -k "expr"
- Coverage: pytest --cov --cov-report=term-missing
- Lint/format: ruff check .; ruff format .
- Typecheck: mypy . (if configured) or pyright
- DB/infra: docker compose up -d postgres redis; alembic upgrade head

## Code style

- Imports: stdlib, third-party, local; absolute imports preferred; no wildcard
- Formatting: ruff format (PEP 8/black-like); max line length per ruff config
- Types: Python 3.11+ typing, strict where feasible; no Any unless necessary
- Naming: snake_case for funcs/vars, PascalCase for classes, UPPER_SNAKE for consts
- Errors: raise domain-specific exceptions; never swallow; log with context; no bare except
- FastAPI: pydantic models for IO; validate at boundaries; return typed responses
- Concurrency: use async/await patterns; prefer async def; avoid blocking calls
- Config/secrets: only from .env; never hardcode or log secrets
- Tests: TDD; >90% coverage; use HTTPX for API; unit before integration; deterministic
- Docs: update docs/INDEX.md references; scratch work only in /scratchpad
- Git: feature branches; commit small, tested changes; PRs include rationale

## Cursor/Copilot rules

- If .cursor/rules or .cursorrules or .github/copilot-instructions.md exist, follow them; include import/order, formatting, and review gates accordingly.