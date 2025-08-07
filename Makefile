# LeanVibe Agent Hive - Hybrid Architecture Makefile
# Docker runs infrastructure, agents run in tmux on host

.PHONY: help
help: ## Show this help message
	@echo "LeanVibe Agent Hive 2.0 - Hybrid Architecture"
	@echo ""
	@echo "Usage: make [command]"
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

# ============== DEVELOPMENT ==============

.PHONY: generate
generate: ## Generate system with Claude Code
	claude "Create the complete LeanVibe Agent Hive 2.0 system. Follow IMPLEMENTATION.md"

.PHONY: test
test: ## Run tests locally
	python -m pytest tests/ -v

.PHONY: logs
logs: ## Tail all log files
	tail -f logs/*.log 2>/dev/null || echo "No log files yet"

.PHONY: status
status: ## Check complete system status
	@make check
	@echo ""
	@echo "Docker services:"
	@make docker-status
	@echo ""
	@echo "Agent sessions:"
	@make agent-list

.PHONY: api-test
api-test: ## Test API endpoints
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"

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
	@echo "pgAdmin: http://localhost:5050 (admin@leanvibe.com / admin)"
	@echo "Redis Commander: http://localhost:8081"
	@echo "API Docs: http://localhost:8000/docs"

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
	@echo "Checking CLI agentic coding tools..."
	@which opencode >/dev/null 2>&1 && echo "✓ opencode installed" || echo "✗ opencode not found"
	@which claude >/dev/null 2>&1 && echo "✓ claude installed" || echo "✗ claude not found"
	@which gemini >/dev/null 2>&1 && echo "✓ gemini installed" || echo "✗ gemini not found"
	@echo ""
	@echo "Priority order: opencode (preferred) → claude → gemini → API fallback"
	@echo "Force specific tool: export PREFERRED_CLI_TOOL=opencode"

.PHONY: tools-up
tools-up: ## Start optional tools (pgAdmin, Redis Commander)
	docker-compose --profile tools up -d
	@echo "Tools started:"
	@echo "pgAdmin: http://localhost:5050"
	@echo "Redis Commander: http://localhost:8081"

.PHONY: tools-down
tools-down: ## Stop optional tools
	docker-compose --profile tools down