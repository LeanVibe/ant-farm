# LeanVibe Agent Hive - Hybrid Architecture Makefile
# Docker runs infrastructure, agents run in tmux on host
# 
# NOTE: The 'hive' CLI is the preferred way to manage the system.
# Use 'hive --help' for modern command interface.

.PHONY: help
help: ## Show this help message
	@echo "LeanVibe Agent Hive 2.0 - Hybrid Architecture"
	@echo ""
	@echo "PREFERRED: Use 'hive --help' for modern CLI interface"
	@echo ""
	@echo "Legacy Make commands:"
	@echo ""
	@echo "Infrastructure (Docker):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(docker|db|redis)' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Agents (tmux on host):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(agent|bootstrap|tmux)' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Development:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE '(docker|db|redis|agent|bootstrap|tmux)' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============== SETUP ==============

.PHONY: check
check: ## Check prerequisites (Claude Code, tmux, Docker)
	@echo "Checking prerequisites..."
	@which claude >/dev/null 2>&1 && echo "✓ Claude Code installed" || echo "✗ Claude Code not found - install from https://claude.ai/cli"
	@which tmux >/dev/null 2>&1 && echo "✓ tmux installed" || echo "✗ tmux not found - run: brew install tmux"
	@which docker >/dev/null 2>&1 && echo "✓ Docker installed" || echo "✗ Docker not found"
	@echo ""
	@echo "Checking API key..."
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then echo "✗ ANTHROPIC_API_KEY not set"; else echo "✓ ANTHROPIC_API_KEY set"; fi

.PHONY: setup
setup: ## Initial setup - create directories and check prerequisites
	@mkdir -p src/core src/agents src/api src/web/dashboard
	@mkdir -p tests/unit tests/integration logs workspace
	@mkdir -p scripts docker
	@touch src/__init__.py src/core/__init__.py src/agents/__init__.py src/api/__init__.py
	@make check
	@echo ""
	@echo "Setup complete! Next: make docker-up"

# ============== DOCKER INFRASTRUCTURE ==============

.PHONY: docker-up
docker-up: ## Start Docker infrastructure (PostgreSQL, Redis)
	docker-compose up -d postgres redis
	@echo "Waiting for services to be healthy..."
	@sleep 5
	docker-compose ps

.PHONY: docker-down
docker-down: ## Stop Docker infrastructure
	docker-compose down

.PHONY: docker-clean
docker-clean: ## Stop Docker and remove volumes (full reset)
	docker-compose down -v

.PHONY: docker-logs
docker-logs: ## View Docker service logs
	docker-compose logs -f

.PHONY: docker-status
docker-status: ## Check Docker service status
	@docker-compose ps

.PHONY: db-shell
db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U hive_user -d leanvibe_hive

.PHONY: redis-cli
redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

.PHONY: db-backup
db-backup: ## Backup PostgreSQL database
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U hive_user leanvibe_hive > backups/db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backed up to backups/"

# ============== AGENT MANAGEMENT (tmux) ==============

.PHONY: bootstrap
bootstrap: ## Run bootstrap agent to build the system
	@python bootstrap/init_agent.py bootstrap

.PHONY: agent-spawn
agent-spawn: ## Spawn a new agent (use TYPE=meta NAME=agent-001)
	@python bootstrap/init_agent.py spawn $(or $(TYPE),worker) --name $(NAME)

.PHONY: agent-list
agent-list: ## List all active agent tmux sessions
	@echo "Active agent sessions:"
	@tmux ls 2>/dev/null | grep agent || echo "No agent sessions running"

.PHONY: agent-attach
agent-attach: ## Attach to agent session (use NAME=agent-name)
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make agent-attach NAME=agent-name"; \
		echo "Available sessions:"; \
		tmux ls | grep agent; \
	else \
		tmux attach -t $(NAME); \
	fi

.PHONY: agent-kill
agent-kill: ## Kill an agent session (use NAME=agent-name)
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make agent-kill NAME=agent-name"; \
	else \
		tmux kill-session -t $(NAME) && echo "Killed session: $(NAME)"; \
	fi

.PHONY: agent-killall
agent-killall: ## Kill all agent sessions
	@tmux ls | grep agent | cut -d: -f1 | xargs -I {} tmux kill-session -t {} 2>/dev/null || true
	@echo "All agent sessions terminated"

.PHONY: tmux-ls
tmux-ls: ## List all tmux sessions
	@tmux ls 2>/dev/null || echo "No tmux sessions"

# ============== HIVE CLI ==============

.PHONY: hive
hive: ## Run hive CLI command (use: make hive ARGS="system status")
	@python -m src.cli.main $(ARGS)

.PHONY: status
status: ## Check complete system status
	@python -m src.cli.main system status

.PHONY: start-hive
start-hive: ## Start the hive system (API + services)
	@python -m src.cli.main system start

.PHONY: hive-init
hive-init: ## Initialize the hive system
	@python -m src.cli.main init

# ============== DEVELOPMENT ==============

.PHONY: generate
generate: ## Generate system with Claude Code
	claude "Create the complete LeanVibe Agent Hive 2.0 system. Follow IMPLEMENTATION.md"

.PHONY: test
test: ## Run tests (quick version without coverage)
	python -m pytest tests/test_infrastructure.py -v --no-cov

.PHONY: test-unit
test-unit: ## Run unit tests only
	python -m pytest tests/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests
	python -m pytest tests/integration/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage (when implementations exist)
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=50

.PHONY: logs
logs: ## Tail all log files
	tail -f logs/*.log 2>/dev/null || echo "No log files yet"

.PHONY: status
status: ## Check complete system status (DEPRECATED: use 'hive system status')
	@echo "⚠️  DEPRECATED: Use 'hive system status' instead"
	@python -m src.cli.main system status

.PHONY: api-test
api-test: ## Test API endpoints
	@curl -s http://localhost:9001/api/v1/health | jq . || echo "API not responding on port 9001"

.PHONY: submit-task
submit-task: ## Submit a test task to the queue
	@python -c "import redis; r = redis.Redis(); import json, uuid; \
	r.lpush('task_queue:normal', json.dumps({ \
		'id': str(uuid.uuid4()), \
		'title': 'Test task', \
		'type': 'analysis', \
		'description': 'Analyze the system and suggest improvements' \
	})); print('Task submitted to queue')"

# ============== QUICK COMMANDS ==============

.PHONY: quick
quick: setup docker-up bootstrap ## Complete setup from scratch
	@echo "System bootstrapped! View agents with: make agent-list"

.PHONY: start
start: docker-up ## Start infrastructure
	@echo "Infrastructure started. Run 'make bootstrap' to spawn agents"

.PHONY: stop
stop: agent-killall docker-down ## Stop everything

.PHONY: restart
restart: stop start ## Restart everything

.PHONY: clean
clean: agent-killall docker-clean ## Full cleanup and reset
	@rm -rf logs/*.log workspace/*
	@echo "System reset complete"

# ============== MONITORING ==============

.PHONY: monitor
monitor: ## Open monitoring tools
	@echo "Opening monitoring tools..."
	@echo "pgAdmin: http://localhost:9050 (admin@leanvibe.com / admin)"
	@echo "Redis Commander: http://localhost:9081" 
	@echo "API Docs: http://localhost:9001/api/docs"

.PHONY: watch
watch: ## Watch system activity (split screen)
	@echo "Starting split screen monitor..."
	@tmux new-session -d -s monitor
	@tmux send-keys -t monitor "make docker-logs" Enter
	@tmux split-window -h -t monitor
	@tmux send-keys -t monitor "make agent-list && watch -n 5 'tmux ls | grep agent'" Enter
	@tmux split-window -v -t monitor
	@tmux send-keys -t monitor "make logs" Enter
	@tmux attach -t monitor

# ============== TOOLS ==============

.PHONY: tools
tools: ## Check available CLI agentic coding tools
	@echo "CLI Agentic Coding Tools:"
	@echo "========================"
	@which opencode >/dev/null 2>&1 && echo "✓ opencode detected (PREFERRED)" || echo "✗ opencode not found - install: curl -fsSL https://opencode.ai/install | bash"
	@which claude >/dev/null 2>&1 && echo "✓ Claude Code CLI detected" || echo "✗ Claude Code CLI not found - install: https://claude.ai/cli"
	@which gemini >/dev/null 2>&1 && echo "✓ Gemini CLI detected" || echo "✗ Gemini CLI not found - install: https://ai.google.dev/gemini-api/docs/cli"
	@which gcloud >/dev/null 2>&1 && gcloud ai --version >/dev/null 2>&1 && echo "✓ Google Cloud AI detected" || true
	@echo ""
	@if ! which opencode >/dev/null 2>&1 && ! which claude >/dev/null 2>&1 && ! which gemini >/dev/null 2>&1 && ! (which gcloud >/dev/null 2>&1 && gcloud ai --version >/dev/null 2>&1); then \
		echo "⚠️  WARNING: No CLI agentic coding tools detected!"; \
		echo "   Install at least one tool for optimal experience:"; \
		echo "   • opencode (recommended): curl -fsSL https://opencode.ai/install | bash"; \
		echo "   • Claude Code CLI: https://claude.ai/cli"; \
		echo "   • Gemini CLI: https://ai.google.dev/gemini-api/docs/cli"; \
		echo ""; \
		echo "   System will still work with API keys as fallback."; \
	else \
		echo "✓ At least one CLI tool detected - system ready!"; \
	fi
	@echo ""
	@echo "Priority order: opencode (preferred) → claude → gemini → API fallback"
	@echo "Force specific tool: export PREFERRED_CLI_TOOL=opencode"

.PHONY: tools-up
tools-up: ## Start optional tools (pgAdmin, Redis Commander)
	docker-compose --profile tools up -d
	@echo "Tools started:"
	@echo "pgAdmin: http://localhost:9050"
	@echo "Redis Commander: http://localhost:9081"

.PHONY: tools-down
tools-down: ## Stop optional tools
	docker-compose --profile tools down