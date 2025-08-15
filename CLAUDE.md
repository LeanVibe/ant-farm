# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# LeanVibe Agent Hive 2.0 - Development Context

## Project Overview

You are working on LeanVibe Agent Hive 2.0, a self-improving autonomous multi-agent development system. The system uses multiple CLI agentic coding tools (opencode, Claude CLI, Gemini CLI) to build and enhance itself continuously through a hybrid architecture: Docker for infrastructure services, tmux for agent processes.

## Core Principles

1. **Simplicity First**: Every component must be simple enough for an agent to understand and modify
2. **Test-Driven Development**: Write tests before implementation (pytest, >90% coverage)
3. **Self-Documenting**: Code should be clear with comprehensive docstrings
4. **Fail-Safe Design**: All operations must be reversible (git-based rollback)
5. **Incremental Progress**: Small, working commits that build on each other

## Essential Commands

### Development Workflow
```bash
# Setup and initialization
hive init                              # Initialize system (DB, migrations, Docker)
hive system start                      # Start API server and services
hive system status                     # Check complete system health

# Agent management
hive agent spawn meta                  # Spawn meta-agent
hive agent spawn architect             # Spawn architect agent
hive agent list                        # List all active agents
hive agent tools                       # Check available CLI tools

# Task management
hive task submit "description"          # Submit task to queue
hive task list                         # List tasks
```

### Development Commands
```bash
# Package management (IMPORTANT: Use uv, not pip)
uv sync --extra dev                    # Install dependencies
uv run python -m src.cli.main         # Run CLI commands
uv run pytest                         # Run tests

# Testing (aligned with CI)
uv run pytest -q tests/unit/test_orchestrator.py tests/unit/test_tmux_manager.py --no-cov
make test-fast                         # Fast unit tests
make test-coverage-core                # Coverage for core modules

# Infrastructure
make docker-up                        # Start PostgreSQL + Redis
make docker-down                       # Stop Docker services
alembic upgrade head                   # Run database migrations
```

## Technology Stack

- **Language**: Python 3.11+ with type hints
- **Backend**: FastAPI (async/await everywhere)
- **Database**: PostgreSQL 15 with pgvector extension
- **Cache/Queue**: Redis 7.0+ (persistence enabled)
- **Frontend**: LitPWA (Web Components + PWA)
- **Package Manager**: uv (NEVER use pip)
- **Testing**: pytest with pytest-asyncio
- **Logging**: structlog for structured logging
- **Agent Runtime**: tmux sessions with CLI tool integration

## Core Architecture

### Hybrid System Design
- **Infrastructure**: Docker containers for PostgreSQL, Redis, monitoring tools
- **Agents**: tmux sessions on host for direct system access and CLI tool execution
- **Communication**: Redis pub/sub for agent messaging, PostgreSQL for persistence
- **CLI Integration**: Multi-tool support (opencode → Claude CLI → Gemini → API fallback)

### Key Components (`src/core/`)
- **orchestrator.py**: Agent lifecycle, spawning, health monitoring, task assignment
- **task_queue.py**: Redis-based priority task distribution with dependencies
- **message_broker.py**: Inter-agent communication via pub/sub
- **tmux_manager.py**: Resilient tmux session management with multiple backends
- **async_db.py**: PostgreSQL connection pooling and async operations

### Agent System (`src/agents/`)
- **base_agent.py**: Multi-CLI tool integration with fallback strategy
- **runner.py**: Agent process entry point for tmux sessions
- **Specialized agents**: meta_agent.py, architect_agent.py, etc.

### Architecture Patterns

#### Async-First Design
```python
async def process_task(self, task: Task) -> Result:
    """All I/O operations must be async."""
    async with self.db.begin() as conn:
        result = await conn.execute(query)
    return await self.process_result(result)
```

#### Multi-Backend tmux Management
```python
# Environment-based backend selection
backend = select_default_tmux_backend(project_root)
session_created = await backend.create_session(session_name, command)
```

#### CLI Tool Integration
```python
# Priority order: opencode → Claude CLI → Gemini → API
tool = detect_available_cli_tools()
result = await tool.execute_command(prompt)
```

## Development Workflow

### Starting Development
1. **Check prerequisites**: `make check` (Claude Code, tmux, Docker)
2. **Initialize system**: `hive init` (database, migrations, Docker services)
3. **Start services**: `hive system start` (API server, background tasks)
4. **Spawn agents**: `hive agent spawn meta` (first agent)
5. **Verify status**: `hive system status` (health check)

### Testing Strategy
- **Fast tests**: Core component tests without coverage gates
- **Coverage tests**: Focused on orchestrator, tmux_manager, message_broker
- **Integration tests**: Component interaction validation
- **CI alignment**: Use same commands as `.github/workflows/ci-fast.yml`

### CLI Tool Priority
1. **opencode** (preferred): `curl -fsSL https://opencode.ai/install | bash`
2. **Claude CLI**: https://claude.ai/cli
3. **Gemini CLI**: https://ai.google.dev/gemini-api/docs/cli
4. **API fallback**: Direct API calls when CLI tools unavailable

### Component Integration Points

#### Agent Lifecycle (`src/core/orchestrator.py`)
- **AgentRegistry**: In-memory + PostgreSQL persistence with async locks
- **AgentSpawner**: tmux-based spawning with backend abstraction
- **HealthMonitor**: Heartbeat monitoring, cleanup loops, database maintenance
- **Task Assignment**: Capability-based routing to available agents

#### Database Patterns
- **Async sessions**: All database operations use async SQLAlchemy
- **Connection pooling**: Managed via `async_db.py`
- **Migrations**: Alembic for schema evolution
- **Vector search**: pgvector for semantic context retrieval

## Database Conventions

### Naming
- Tables: plural, snake_case (e.g., `agents`, `tasks`)
- Columns: snake_case (e.g., `created_at`, `agent_id`)
- Indexes: `idx_table_column` (e.g., `idx_agents_status`)
- Foreign keys: `table_id` (e.g., `agent_id`)

### Required Columns
```sql
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
created_at TIMESTAMP DEFAULT NOW(),
updated_at TIMESTAMP DEFAULT NOW()
```

## API Conventions

### Endpoints
- RESTful naming: `/api/v1/resource`
- Use proper HTTP methods (GET, POST, PUT, DELETE)
- Return consistent JSON responses
- Include request ID in headers

### Response Format
```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid"
}
```

## Testing Requirements

### Test Organization
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
└── e2e/           # Full system tests
```

### Test Patterns
```python
@pytest.mark.asyncio
async def test_operation_success():
    """Test description - what and why."""
    # Arrange
    component = await create_test_component()
    
    # Act
    result = await component.operation()
    
    # Assert
    assert result.success is True
    assert result.data is not None
```

## Logging Standards

```python
import structlog
logger = structlog.get_logger()

# Use structured logging
logger.info("Operation completed",
           operation="task_process",
           task_id=task.id,
           duration=elapsed_time)
```

## Git Workflow

### Commit Messages
```
type(scope): description

- feat: New feature
- fix: Bug fix
- refactor: Code refactoring
- test: Test additions/changes
- docs: Documentation updates
```

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `agent/*`: Agent-generated changes

## Performance Targets

- API response time: <100ms (p95)
- Task processing: >1000/minute
- Memory per agent: <500MB
- Database queries: <50ms
- Message latency: <10ms

## Security Considerations

- No hardcoded secrets (use environment variables)
- Input validation on all endpoints
- SQL injection prevention (use parameterized queries)
- Rate limiting on all APIs
- Audit logging for sensitive operations

## Common Patterns

### Singleton Services
```python
class ServiceClass:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Repository Pattern
```python
class AgentRepository:
    async def create(self, agent: Agent) -> Agent:
        """Create pattern for database operations."""
        pass
    
    async def get(self, agent_id: UUID) -> Optional[Agent]:
        """Get by ID pattern."""
        pass
    
    async def update(self, agent: Agent) -> Agent:
        """Update pattern."""
        pass
```

### Factory Pattern
```python
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str) -> BaseAgent:
        """Factory for creating different agent types."""
        if agent_type == "meta":
            return MetaAgent()
        elif agent_type == "developer":
            return DeveloperAgent()
        raise ValueError(f"Unknown agent type: {agent_type}")
```

## Development Principles

### Core Guidelines
- **Package manager**: Use `uv`, never `pip`
- **Testing**: All I/O operations must be mockable
- **Async patterns**: All database and Redis operations are async
- **CLI integration**: Support multiple tools with graceful fallback
- **Monitoring**: Structured logging with contextual information

### Architecture Insights
- **Hybrid design**: Docker for infrastructure, tmux for agents
- **Multi-backend**: Abstraction layer for different execution environments
- **Agent communication**: Redis pub/sub with PostgreSQL persistence
- **Health monitoring**: Automated cleanup and failure detection
- **Tool agnostic**: Works with any available CLI agentic coding tool

### Remember
- This system builds itself through autonomous agents
- Every component must be resilient and self-monitoring
- CLI tool availability varies - always have fallbacks
- tmux sessions provide direct system access for agents
- Test everything that can fail in production