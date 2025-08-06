.PHONY: setup build up start stop logs clean test bootstrap dev prod

# Default target
help:
	@echo "LeanVibe Agent Hive 2.0 - Make Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup     - Complete system setup (install dependencies, start services)"
	@echo "  build     - Build Docker images"
	@echo ""
	@echo "Service Management:"
	@echo "  up        - Start core services (PostgreSQL, Redis)"
	@echo "  start     - Start full system"
	@echo "  stop      - Stop all services"
	@echo "  restart   - Restart services"
	@echo ""
	@echo "Development:"
	@echo "  dev       - Start development environment with tools"
	@echo "  bootstrap - Run bootstrap agent to build system"
	@echo "  test      - Run test suite"
	@echo "  lint      - Run code linting"
	@echo "  format    - Format code"
	@echo ""
	@echo "Monitoring:"
	@echo "  logs      - Show service logs"
	@echo "  status    - Show system status"
	@echo "  tools     - Check available CLI agentic coding tools"
	@echo "  health    - Check system health"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  reset     - Full system reset"
	@echo "  backup    - Create system backup"

# Setup and Installation
setup:
	@echo "Setting up LeanVibe Agent Hive 2.0..."
	@# Check for required tools
	@which docker > /dev/null || (echo "Docker not found. Please install Docker Desktop." && exit 1)
	@which tmux > /dev/null || (echo "tmux not found. Installing..." && brew install tmux)
	@# Check for CLI agentic coding tools
	@echo "Checking for CLI agentic coding tools..."
	@make check-cli-tools
	@# Ollama is optional
	@curl -s http://localhost:11434/api/version > /dev/null 2>&1 && echo "Ollama detected" || echo "Ollama not running (optional)"
	@# Create environment file if it doesn't exist
	@test -f .env || make .env
	@echo "Setup complete! System will auto-detect and use available CLI tools."

# Check for available CLI agentic coding tools
check-cli-tools:
	@echo "CLI Agentic Coding Tools:"
	@echo "========================"
	@which opencode > /dev/null && echo "✓ opencode detected (PREFERRED)" || echo "✗ opencode not found - install: curl -fsSL https://opencode.ai/install | bash"
	@which claude > /dev/null && echo "✓ Claude Code CLI detected" || echo "✗ Claude Code CLI not found - install: https://claude.ai/cli"
	@which gemini > /dev/null && echo "✓ Gemini CLI detected" || echo "✗ Gemini CLI not found - install: https://ai.google.dev/gemini-api/docs/cli"
	@# Check for gcloud ai as alternative to gemini
	@which gcloud > /dev/null && gcloud ai --version > /dev/null 2>&1 && echo "✓ Google Cloud AI detected" || true
	@echo ""
	@# Check if any tools are available
	@if ! which opencode > /dev/null && ! which claude > /dev/null && ! which gemini > /dev/null && ! (which gcloud > /dev/null && gcloud ai --version > /dev/null 2>&1); then \
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

# Environment setup
.env:
	@echo "Creating .env file..."
	@echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
	@echo "OPENAI_API_KEY=your-openai-key-here" >> .env
	@echo "DATABASE_URL=postgresql://hive_user:hive_pass@localhost:5432/leanvibe_hive" >> .env
	@echo "REDIS_URL=redis://localhost:6379" >> .env
	@echo "LOG_LEVEL=INFO" >> .env
	@echo "Edit .env file with your actual API keys!"

# Docker operations
build:
	@echo "Building Docker images..."
	docker-compose build

up:
	@echo "Starting core services (PostgreSQL, Redis)..."
	docker-compose up -d postgres redis
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services started. Check status with 'make status'"

start: up
	@echo "Starting full system..."
	docker-compose --profile production up -d
	@echo "System started. API available at http://localhost:8000"

dev: up
	@echo "Starting development environment..."
	docker-compose --profile dev up -d
	@echo "Development tools started:"
	@echo "  - pgAdmin: http://localhost:5050"
	@echo "  - Redis Commander: http://localhost:8081"

stop:
	@echo "Stopping all services..."
	docker-compose down

restart: stop start

# Bootstrap and initialization
bootstrap: up
	@echo "Running bootstrap agent..."
	@# Ensure Python dependencies are available
	@pip install -e .
	@# Run bootstrap agent
	python bootstrap/init_agent.py
	@echo "Bootstrap complete! System is building itself..."

quick: setup up bootstrap start logs

# Development tools
test:
	@echo "Running test suite..."
	@# Run tests in container if available, otherwise locally
	@if docker-compose ps api | grep -q "Up"; then \
		docker-compose exec api pytest tests/ -v --cov=src --cov-report=term-missing; \
	else \
		pip install -e .[dev] && pytest tests/ -v --cov=src --cov-report=term-missing; \
	fi

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

lint:
	@echo "Running code linting..."
	ruff check .
	mypy src/

format:
	@echo "Formatting code..."
	black .
	ruff check . --fix

# Monitoring and status
logs:
	@echo "Showing service logs (Ctrl+C to stop)..."
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-bootstrap:
	docker-compose logs -f bootstrap

status:
	@echo "System Status:"
	@echo "=============="
	@docker-compose ps
	@echo ""
	@echo "CLI Tools:"
	@make check-cli-tools
	@echo ""
	@echo "Tmux Sessions:"
	@tmux list-sessions 2>/dev/null || echo "No tmux sessions"
	@echo ""
	@echo "Agent Status:"
	@curl -s http://localhost:8000/api/v1/agents 2>/dev/null | jq '.' || echo "API not available"

health:
	@echo "Health Check:"
	@echo "============="
	@echo -n "PostgreSQL: "
	@docker-compose exec postgres pg_isready -U hive_user -d leanvibe_hive 2>/dev/null && echo "✓ OK" || echo "✗ FAIL"
	@echo -n "Redis: "
	@docker-compose exec redis redis-cli ping 2>/dev/null && echo "✓ OK" || echo "✗ FAIL"
	@echo -n "API: "
	@curl -s http://localhost:8000/health >/dev/null 2>&1 && echo "✓ OK" || echo "✗ FAIL"

# Tmux session management
tmux-list:
	@echo "Active tmux sessions:"
	@tmux list-sessions 2>/dev/null || echo "No sessions"

tmux-kill:
	@echo "Killing all agent tmux sessions..."
	@tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -E "(hive-|meta-|dev-)" | xargs -I {} tmux kill-session -t {} 2>/dev/null || true

# Database operations
db-shell:
	docker-compose exec postgres psql -U hive_user -d leanvibe_hive

db-backup:
	@echo "Creating database backup..."
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U hive_user leanvibe_hive > backups/db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/"

db-restore:
	@echo "Restoring database from latest backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make db-restore BACKUP_FILE=backups/db_20240101_120000.sql"; \
		exit 1; \
	fi
	docker-compose exec -T postgres psql -U hive_user leanvibe_hive < $(BACKUP_FILE)

# Redis operations
redis-shell:
	docker-compose exec redis redis-cli

redis-monitor:
	docker-compose exec redis redis-cli MONITOR

redis-flush:
	@echo "Flushing Redis data..."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose exec redis redis-cli FLUSHALL

# Maintenance
clean:
	@echo "Cleaning up containers and images..."
	docker-compose down -v --remove-orphans
	docker system prune -f

reset: clean tmux-kill
	@echo "Full system reset..."
	@read -p "This will delete ALL data. Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose down -v --remove-orphans
	docker volume rm leanvibe-hive_postgres_data leanvibe-hive_redis_data 2>/dev/null || true
	@echo "System reset complete. Run 'make setup' to start fresh."

backup: db-backup
	@echo "Creating full system backup..."
	@mkdir -p backups
	@# Backup Redis data
	docker-compose exec -T redis redis-cli --rdb - > backups/redis_$(shell date +%Y%m%d_%H%M%S).rdb
	@# Backup configuration
	tar -czf backups/config_$(shell date +%Y%m%d_%H%M%S).tar.gz .env docker-compose.yml
	@echo "Full backup created in backups/"

# API operations
api-test:
	@echo "Testing API endpoints..."
	@curl -s http://localhost:8000/health | jq '.' || echo "Health endpoint failed"
	@curl -s http://localhost:8000/api/v1/agents | jq '.' || echo "Agents endpoint failed"

api-docs:
	@echo "Opening API documentation..."
	@open http://localhost:8000/docs || echo "Could not open browser. Visit http://localhost:8000/docs"

# Agent management
agent-create:
	@echo "Creating new agent..."
	@if [ -z "$(TYPE)" ]; then \
		echo "Usage: make agent-create TYPE=meta|developer|qa NAME=agent-name"; \
		exit 1; \
	fi
	@curl -X POST http://localhost:8000/api/v1/agents \
		-H "Content-Type: application/json" \
		-d '{"name": "$(NAME)", "type": "$(TYPE)", "role": "$(TYPE)"}' | jq '.'

agent-list:
	@echo "Active agents:"
	@curl -s http://localhost:8000/api/v1/agents | jq '.'

tools: check-cli-tools

# Task management
task-submit:
	@echo "Submitting task..."
	@if [ -z "$(TASK)" ]; then \
		echo "Usage: make task-submit TASK='Analyze system performance'"; \
		exit 1; \
	fi
	@curl -X POST http://localhost:8000/api/v1/tasks \
		-H "Content-Type: application/json" \
		-d '{"title": "$(TASK)", "type": "analysis"}' | jq '.'

task-list:
	@echo "Current tasks:"
	@curl -s http://localhost:8000/api/v1/tasks | jq '.'

# Install development environment
install-dev:
	@echo "Installing development dependencies..."
	pip install -e .[dev]
	pre-commit install