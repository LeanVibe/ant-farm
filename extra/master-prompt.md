# Master Prompt for Claude Code - Complete System Generation

Use this single prompt to generate the entire LeanVibe Agent Hive 2.0 system:

```
Create the complete LeanVibe Agent Hive 2.0 system - a self-improving multi-agent development platform using Docker.

## Project Structure to Create:

```
leanvibe-hive/
├── docker-compose.yml          # All services defined
├── docker-compose.override.yml # Development overrides
├── Makefile                    # Easy command shortcuts
├── .env.example               # Example environment variables
├── pyproject.toml             # Python dependencies with uv
├── docker/
│   ├── Dockerfile.bootstrap   # Bootstrap agent container
│   ├── Dockerfile.api         # FastAPI server container
│   ├── Dockerfile.agent       # Agent worker container
│   └── Dockerfile.dev         # Development tools container
├── scripts/
│   ├── init.sql              # Database initialization
│   ├── backup.sh             # Backup script
│   └── migrate.py            # Database migrations
├── bootstrap/
│   └── init_agent.py         # Initial agent that builds everything
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── task_queue.py     # Redis-based priority queue
│   │   ├── orchestrator.py   # Agent lifecycle management
│   │   ├── message_broker.py # Inter-agent communication
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── context_engine.py # Vector memory with pgvector
│   │   ├── self_modifier.py  # Safe code modification
│   │   ├── sleep_wake.py     # Consolidation cycles
│   │   └── config.py         # Pydantic settings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Abstract base class
│   │   ├── meta_agent.py     # Self-improvement coordinator
│   │   ├── worker.py         # Generic worker process
│   │   └── specialized/
│   │       ├── architect.py
│   │       ├── developer.py
│   │       └── qa.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── routes/
│   │   ├── models/
│   │   └── dependencies.py
│   └── web/
│       └── dashboard/       # LitPWA components
└── tests/
    ├── unit/
    ├── integration/
    └── conftest.py         # Pytest fixtures
```

## Key Requirements:

### 1. Docker Compose Services:
- PostgreSQL 15 with pgvector extension
- Redis 7 with persistence
- Bootstrap agent container
- API server (FastAPI)
- Agent workers (scalable)
- Development tools (pgAdmin, Redis Commander)
- Monitoring (Prometheus, Grafana)

### 2. Core Components:

**Task Queue (src/core/task_queue.py)**:
- Redis-based with LPUSH/BRPOP
- 5 priority levels
- Task dependencies
- Retry with exponential backoff
- Timeout monitoring

**Agent Orchestrator (src/core/orchestrator.py)**:
- Agent lifecycle (spawn, monitor, terminate)
- Health checks every 30 seconds
- Capability-based task assignment
- Graceful shutdown

**Message Broker (src/core/message_broker.py)**:
- Redis pub/sub
- Message persistence
- Topic-based routing
- Dead letter queue

**Context Engine (src/core/context_engine.py)**:
- Vector storage with pgvector
- OpenAI embeddings
- Semantic search
- Context compression

**Base Agent (src/agents/base_agent.py)**:
- Anthropic Claude integration
- Task processing loop
- Message handling
- Context management

**Meta Agent (src/agents/meta_agent.py)**:
- System performance analysis
- Improvement proposals
- Safe testing
- Git operations

### 3. Database Schema:
```sql
-- Use PostgreSQL with pgvector
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Tables: agents, tasks, sessions, contexts, conversations
-- All with proper indexes and relationships
```

### 4. API Endpoints:
```
POST   /api/v1/agents
GET    /api/v1/agents
POST   /api/v1/tasks
GET    /api/v1/tasks/{id}
POST   /api/v1/sessions
WS     /api/v1/ws/events
GET    /api/v1/health
```

### 5. Configuration:

**pyproject.toml**:
- Use uv package manager
- Dependencies: fastapi, sqlalchemy, redis, anthropic, pgvector, etc.
- Dev dependencies: pytest, black, ruff

**Makefile**:
- Commands: setup, build, up, start, test, logs, clean
- Make development easy with simple commands

### 6. Testing:
- pytest with pytest-asyncio
- >90% code coverage target
- Unit tests for each component
- Integration tests for system flows

## Technical Standards:

1. **Python 3.11+** with type hints everywhere
2. **Async/await** for all I/O operations
3. **Pydantic** for data validation
4. **structlog** for structured logging
5. **Error handling** with proper exceptions
6. **Docstrings** on all functions/classes
7. **Git-based** version control for self-modification

## Docker-Specific Requirements:

1. All services communicate via container names
2. Use health checks for service dependencies
3. Mount code volumes for development hot-reload
4. Use profiles for optional services
5. Include restart policies
6. Set proper environment variables

## The system should be able to:

1. Bootstrap itself from a single command
2. Generate its own improvements
3. Run tests automatically
4. Operate 24/7 without intervention
5. Scale horizontally
6. Recover from failures
7. Track all operations in logs

Start by creating docker-compose.yml, then the Dockerfiles, then pyproject.toml, then bootstrap/init_agent.py, then implement all core components in priority order.

Make everything production-ready with proper error handling, logging, and testing.
```

## Alternative Shorter Prompt:

```
Create a complete Docker-based multi-agent system called LeanVibe Agent Hive 2.0.

Stack: FastAPI, PostgreSQL+pgvector, Redis, Docker Compose, Python 3.11

Create:
1. docker-compose.yml with postgres, redis, api, workers
2. Dockerfiles in docker/ directory  
3. pyproject.toml for uv package manager
4. bootstrap/init_agent.py to build the system
5. src/core/task_queue.py - Redis priority queue
6. src/core/orchestrator.py - Agent management
7. src/agents/base_agent.py - Agent base class
8. src/agents/meta_agent.py - Self-improvement
9. src/api/main.py - FastAPI endpoints
10. Makefile for easy commands

The system should self-build and self-improve using Claude API.
Use async Python, type hints, proper error handling, and >90% test coverage.
```