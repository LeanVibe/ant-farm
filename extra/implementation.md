# Implementation Guide - LeanVibe Agent Hive 2.0

## Overview

This guide provides step-by-step implementation instructions for building the complete system using Claude Code.

## Phase 0: Bootstrap Foundation (Priority 1)

### Step 1: Project Setup

Create the following structure:
```
leanvibe-hive/
├── pyproject.toml
├── bootstrap/
│   └── init_agent.py
├── src/
│   ├── __init__.py
│   └── core/
│       └── __init__.py
└── tests/
```

### Step 2: Core Dependencies (pyproject.toml)

```toml
[project]
name = "leanvibe-agent-hive"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.29.0",
    "pgvector>=0.2.0",
    "redis>=5.0.0",
    "anthropic>=0.18.0",
    "pydantic>=2.5.0",
    "gitpython>=3.1.0",
    "structlog>=24.0.0",
    "typer>=0.9.0",
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
]
```

### Step 3: Bootstrap Agent

The bootstrap agent is the first component that builds everything else.

**Key Requirements:**
- Execute Claude Code CLI commands
- Create project structure
- Generate other components
- Track progress in Redis

## Phase 1: Core Infrastructure (Priority 2)

### Component 1: Task Queue System
**Location**: `src/core/task_queue.py`

**Requirements:**
- Redis-based with LPUSH/BRPOP
- Priority levels (1-9)
- Task dependencies
- Retry logic with exponential backoff
- Timeout monitoring
- At-least-once delivery guarantee

**Key Classes:**
- `Task`: Pydantic model with id, type, payload, status
- `TaskQueue`: Main queue manager
- `TaskStatus`: Enum (pending, assigned, in_progress, completed, failed)
- `TaskPriority`: Enum (critical=1, high=3, normal=5, low=7, background=9)

### Component 2: Agent Orchestrator
**Location**: `src/core/orchestrator.py`

**Requirements:**
- Agent lifecycle management (spawn, monitor, terminate)
- Health checks every 30 seconds
- Graceful shutdown
- Agent registry in PostgreSQL
- Capability matching for task assignment

**Key Classes:**
- `AgentOrchestrator`: Central coordinator
- `AgentRegistry`: Track active agents
- `AgentSpawner`: Create new agent instances
- `HealthMonitor`: Check agent status

### Component 3: Message Broker
**Location**: `src/core/message_broker.py`

**Requirements:**
- Redis pub/sub for real-time messaging
- Message persistence for offline agents
- Topic-based routing
- Message ordering guarantees
- Dead letter queue for failed messages

**Key Classes:**
- `MessageBroker`: Main messaging hub
- `Message`: Standard message format
- `MessageRouter`: Route messages to agents
- `MessageStore`: Persist messages

### Component 4: Database Models
**Location**: `src/core/models.py`

**SQLAlchemy Models:**
```python
- Agent: id, name, type, capabilities, status, created_at
- Task: id, type, payload, status, agent_id, created_at
- Session: id, name, agents, state, created_at
- Context: id, content, embedding, importance_score
- Conversation: id, from_agent, to_agent, content, timestamp
```

## Phase 2: Agent System (Priority 3)

### Component 5: Base Agent Class
**Location**: `src/agents/base_agent.py`

**Requirements:**
- Abstract base class for all agents
- Claude API integration
- Task processing loop
- Message handling
- Context management
- Graceful shutdown

**Key Methods:**
- `process_task()`: Main task processing
- `send_message()`: Communicate with other agents
- `store_context()`: Save to context engine
- `execute_tool()`: Run Claude Code tools

### Component 6: Meta-Agent
**Location**: `src/agents/meta_agent.py`

**Requirements:**
- Analyze system performance
- Generate improvement proposals
- Test changes in sandbox
- Apply approved modifications
- Update agent prompts

**Special Capabilities:**
- Access to system metrics
- Git operations for versioning
- Prompt A/B testing
- Performance benchmarking

### Component 7: Context Engine
**Location**: `src/core/context_engine.py`

**Requirements:**
- Vector storage using pgvector
- OpenAI embeddings (text-embedding-ada-002)
- Semantic search with similarity threshold
- Context compression for long conversations
- Hierarchical memory (short/medium/long-term)

**Key Classes:**
- `ContextEngine`: Main context manager
- `EmbeddingService`: Generate embeddings
- `ContextCompressor`: Summarize conversations
- `SemanticSearch`: Retrieve relevant context

## Phase 3: Self-Improvement (Priority 4)

### Component 8: Self-Modifier
**Location**: `src/core/self_modifier.py`

**Requirements:**
- Safe code generation
- Sandboxed testing environment
- Git-based version control
- Rollback on failure
- Performance validation

**Safety Measures:**
- All changes in feature branches
- Automated test execution
- Performance benchmarks
- Human approval for critical changes
- Automatic rollback triggers

### Component 9: Sleep-Wake Manager
**Location**: `src/core/sleep_wake.py`

**Requirements:**
- Scheduled sleep cycles (2-4 AM UTC default)
- Context consolidation during sleep
- State checkpointing
- Graceful handoff between cycles
- Wake restoration

**Key Processes:**
- Consolidate conversations to summaries
- Update vector indices
- Cleanup temporary data
- Git checkpoint creation
- Performance metrics analysis

## Phase 4: API & UI (Priority 5)

### Component 10: FastAPI Application
**Location**: `src/api/main.py`

**Endpoints:**
```
POST   /agents                 - Create agent
GET    /agents                 - List agents
POST   /tasks                  - Submit task
GET    /tasks/{id}            - Get task status
POST   /sessions              - Create session
WS     /ws/events             - Real-time events
GET    /health                - Health check
```

### Component 11: Web Dashboard
**Location**: `src/web/dashboard/`

**Requirements:**
- LitPWA components
- Real-time WebSocket updates
- Task kanban board
- Agent status panel
- System metrics
- Mobile responsive

## Testing Strategy

### Test Structure
```
tests/
├── unit/
│   ├── test_task_queue.py
│   ├── test_orchestrator.py
│   └── test_agents.py
├── integration/
│   ├── test_agent_communication.py
│   └── test_self_modification.py
└── e2e/
    └── test_full_system.py
```

### Coverage Requirements
- Minimum 90% code coverage
- All critical paths tested
- Integration tests for agent communication
- E2E tests for complete workflows

## Database Setup

### PostgreSQL with pgvector
```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Use provided schema from models.py
```

### Redis Configuration
```
# Persistence
save 900 1
save 300 10
save 60 10000

# Max memory policy
maxmemory-policy allkeys-lru
```

## Docker Compose Setup

```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: leanvibe_hive
      POSTGRES_USER: hive_user
      POSTGRES_PASSWORD: hive_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Success Criteria

Each component must:
1. Have comprehensive tests (>90% coverage)
2. Include error handling and logging
3. Support graceful shutdown
4. Be simple enough for agents to understand
5. Include inline documentation
6. Follow Python type hints
7. Use async/await patterns
8. Implement retry logic where appropriate

## Deployment Checklist

- [ ] All tests passing
- [ ] Database migrations complete
- [ ] Redis configured with persistence
- [ ] Environment variables set
- [ ] Docker containers running
- [ ] Health checks passing
- [ ] Monitoring enabled
- [ ] Logs aggregated
- [ ] Backup strategy in place
- [ ] Rollback procedure documented