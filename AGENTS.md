# AGENTS.md

CLI Agentic Coding Tools
- Auto-detection: system checks for opencode, claude, gemini CLI tools
- Priority order: opencode (preferred) → claude → gemini → API fallback
- Check tools: make tools
- Force specific: export PREFERRED_CLI_TOOL=opencode

Build/lint/test
- Setup: make setup (auto-detects CLI tools); env: .env.local (API keys optional)
- Start: make start; Dev API: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
- Tests (all): pytest -q
- Single test: pytest -q path/to/test_file.py::TestClass::test_case -k "expr"
- Coverage: pytest --cov --cov-report=term-missing
- Lint/format: ruff check .; ruff format .
- Typecheck: mypy . (if configured) or pyright
- DB/infra: docker compose up -d postgres redis; alembic upgrade head

Code style
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

Cursor/Copilot rules
- If .cursor/rules or .cursorrules or .github/copilot-instructions.md exist, follow them; include import/order, formatting, and review gates accordingly.