# Claude Code Generation Prompts

Use these prompts with Claude Code to generate each component of the system.

## Initial Bootstrap Prompt

```
Create the complete foundation for LeanVibe Agent Hive 2.0 following these specifications:

1. Create pyproject.toml with all dependencies using uv package manager
2. Create bootstrap/init_agent.py that can spawn other agents and build the system
3. Create the basic project structure with all necessary directories
4. Set up git repository with proper .gitignore

Technology stack:
- Python 3.11+ with type hints and async/await
- FastAPI for backend
- PostgreSQL with pgvector for database
- Redis for task queue and messaging
- pytest for testing (>90% coverage required)

Follow the specifications in IMPLEMENTATION.md and use the patterns from .claude/CLAUDE.md
```

## Phase 1: Core Infrastructure

### Task Queue System

```
Create src/core/task_queue.py with a complete Redis-based task queue system:

Requirements:
- Use Redis with LPUSH/BRPOP for queue operations
- Support 5 priority levels (critical=1, high=3, normal=5, low=7, background=9)
- Implement task dependencies (task can wait for other tasks)
- Add retry logic with exponential backoff (max 3 retries)
- Monitor for task timeouts (default 300 seconds)
- Provide at-least-once delivery guarantee
- Track task status: pending, assigned, in_progress, completed, failed, cancelled

Include these classes:
- Task (Pydantic model with all fields)
- TaskQueue (main queue manager)
- TaskStatus (enum)
- TaskPriority (enum)
- QueueStats (dataclass for statistics)

Add comprehensive error handling and structured logging with structlog.
Write with 100% type hints and async/await patterns.
Include detailed docstrings explaining the architecture.
```

### Agent Orchestrator

```
Create src/core/orchestrator.py for central agent coordination:

Requirements:
- Manage agent lifecycle (spawn, monitor, terminate)
- Health checks every 30 seconds
- Agent registry with PostgreSQL persistence
- Capability-based task assignment
- Graceful shutdown handling
- Support for 50+ concurrent agents

Key classes:
- AgentOrchestrator (main coordinator)
- AgentRegistry (track active agents)
- AgentSpawner (create new agent instances)
- HealthMonitor (check agent status)

Integrate with the task queue for task delegation.
Use SQLAlchemy for database operations.
Include WebSocket support for real-time updates.
```

### Message Broker

```
Create src/core/message_broker.py for inter-agent communication:

Requirements:
- Redis pub/sub for real-time messaging
- Message persistence for offline agents
- Topic-based routing (agent-to-agent, broadcast, multicast)
- Message ordering guarantees per channel
- Dead letter queue for failed messages
- Support for request-reply pattern

Message format:
{
  "id": "uuid",
  "from_agent": "agent_id",
  "to_agent": "agent_id or broadcast",
  "topic": "topic_name",
  "payload": {},
  "timestamp": "iso8601",
  "reply_to": "message_id (optional)"
}

Include message history and replay capabilities.
```

### Database Models

```
Create src/core/models.py with SQLAlchemy models:

Tables needed:
1. agents - id, name, type, role, capabilities (JSON), system_prompt, status, created_at, updated_at
2. tasks - id, title, description, type, payload (JSON), priority, status, agent_id, dependencies (JSON), created_at, started_at, completed_at
3. sessions - id, name, type, agents (JSON), state (JSON), created_at, last_active
4. contexts - id, agent_id, session_id, content, embedding (vector), importance_score, parent_id, created_at, accessed_at
5. conversations - id, session_id, from_agent_id, to_agent_id, message_type, content, embedding (vector), created_at
6. system_checkpoints - id, type, state (JSON), git_commit_hash, created_at

Use pgvector for embedding columns.
Add proper indexes for performance.
Include relationships and constraints.
```

## Phase 2: Agent System

### Base Agent

```
Create src/agents/base_agent.py as the abstract base class for all agents:

Requirements:
- Integrate with Anthropic Claude API
- Task processing loop with claim/process/complete cycle
- Message sending and receiving
- Context storage and retrieval
- Tool execution (Claude Code integration)
- Graceful shutdown
- Health reporting

Core methods:
- async def process_task(task: Task) -> Result
- async def send_message(to_agent: str, content: dict)
- async def receive_messages() -> List[Message]
- async def store_context(content: str, importance: float)
- async def retrieve_context(query: str, limit: int) -> List[Context]
- async def execute_tool(tool_name: str, params: dict)
- async def health_check() -> HealthStatus

Include rate limiting for Claude API calls.
Add comprehensive error handling and retry logic.
```

### Meta-Agent

```
Create src/agents/meta_agent.py for system self-improvement:

This agent inherits from BaseAgent and adds:
- System performance analysis
- Improvement proposal generation
- Safe testing in sandboxed environment
- Git-based change management
- Prompt optimization with A/B testing
- Performance benchmarking

Special capabilities:
- Access to system metrics (Prometheus)
- Git operations (create branch, commit, merge)
- Database query access for analytics
- Ability to modify other agents' prompts
- Can trigger system-wide operations

Include safety checks:
- Changes must pass all tests
- Performance must improve or maintain
- Rollback on regression
- Human approval for critical changes
```

### Context Engine

```
Create src/core/context_engine.py for semantic memory:

Requirements:
- Use pgvector for vector storage
- OpenAI text-embedding-ada-002 for embeddings
- Semantic search with cosine similarity
- Context compression using Claude
- Hierarchical memory (short/medium/long-term)
- Automatic importance scoring

Key classes:
- ContextEngine (main interface)
- EmbeddingService (generate embeddings)
- ContextCompressor (summarize long contexts)
- SemanticSearch (retrieve relevant contexts)
- MemoryConsolidator (sleep-time consolidation)

Support operations:
- Store context with automatic embedding
- Search by semantic similarity
- Compress conversations to summaries
- Prune old/irrelevant contexts
- Share contexts between agents
```

## Phase 3: Self-Improvement

### Self-Modifier

```
Create src/core/self_modifier.py for safe code modification:

Requirements:
- Generate code improvements using Claude
- Test in Docker sandbox environment
- Git branch for each modification
- Automated test execution
- Performance validation
- Rollback mechanism

Safety measures:
- All changes in feature branches
- Must pass existing tests
- Performance benchmarks required
- Critical changes need approval
- Automatic rollback triggers

Workflow:
1. Analyze current code
2. Generate improvement proposal
3. Create feature branch
4. Apply changes
5. Run tests in sandbox
6. Measure performance
7. Merge or rollback
```

### Sleep-Wake Manager

```
Create src/core/sleep_wake.py for consolidation cycles:

Requirements:
- Configurable sleep schedule (default 2-4 AM)
- Context consolidation during sleep
- Memory compression and organization
- State checkpointing to disk
- Graceful handoff between cycles
- Quick wake restoration

Sleep operations:
- Summarize conversations
- Update vector indices
- Cleanup temporary data
- Create git checkpoint
- Analyze performance metrics
- Plan next day priorities

Wake operations:
- Restore from checkpoint
- Reload context
- Resume pending tasks
- Reconnect to services
```

## Phase 4: API and UI

### FastAPI Application

```
Create src/api/main.py with complete REST API:

Endpoints needed:
# Agents
POST   /api/v1/agents              - Create new agent
GET    /api/v1/agents              - List all agents
GET    /api/v1/agents/{id}         - Get agent details
PUT    /api/v1/agents/{id}         - Update agent
DELETE /api/v1/agents/{id}         - Terminate agent

# Tasks
POST   /api/v1/tasks               - Submit task
GET    /api/v1/tasks               - List tasks
GET    /api/v1/tasks/{id}          - Get task status
PUT    /api/v1/tasks/{id}/cancel   - Cancel task

# Sessions
POST   /api/v1/sessions            - Create session
GET    /api/v1/sessions            - List sessions
POST   /api/v1/sessions/{id}/start - Start session

# System
GET    /api/v1/health              - Health check
GET    /api/v1/metrics             - Prometheus metrics
WS     /api/v1/ws/events           - WebSocket events

Include:
- Pydantic models for validation
- Dependency injection
- Error handling middleware
- CORS configuration
- Authentication (JWT)
- Rate limiting
- OpenAPI documentation
```

### Web Dashboard

```
Create src/web/dashboard/ with LitPWA components:

Components needed:
1. AgentStatus - Show all agents with health indicators
2. TaskBoard - Kanban-style task visualization
3. MessageFlow - Real-time message visualization
4. SystemMetrics - Performance graphs
5. ContextExplorer - Browse semantic memory
6. LogViewer - Real-time log streaming

Features:
- WebSocket connection for live updates
- Mobile responsive design
- PWA with offline support
- Push notifications
- Dark mode support

Use Lit Element with TypeScript.
Include service worker for offline functionality.
Add installable PWA manifest.
```

## Testing Suite

### Unit Tests

```
Create comprehensive unit tests in tests/unit/:

For each component, test:
- Happy path scenarios
- Error conditions
- Edge cases
- Concurrent operations
- Resource cleanup

Example for task_queue:
- Test task submission
- Test priority ordering
- Test dependency resolution
- Test retry logic
- Test timeout handling
- Test concurrent workers

Use pytest with pytest-asyncio.
Mock external dependencies.
Aim for >95% coverage.
```

### Integration Tests

```
Create integration tests in tests/integration/:

Test interactions between:
- Agents and task queue
- Agents and message broker
- Context engine and database
- Self-modifier and git
- API and database

Include:
- Database fixtures
- Redis fixtures
- Multi-agent scenarios
- End-to-end workflows
```

## Docker Setup

```
Create docker-compose.yml for local development:

Services:
1. PostgreSQL 15 with pgvector
2. Redis 7 with persistence
3. API service (FastAPI)
4. Agent workers (multiple instances)
5. Prometheus for metrics
6. Grafana for visualization

Include:
- Volume mounts for development
- Environment variables
- Health checks
- Restart policies
- Network configuration
```

## Final System Test

```
After all components are created, run this comprehensive test:

1. Start all services with docker-compose
2. Bootstrap the first agent
3. Have it create a meta-agent
4. Meta-agent creates specialized agents
5. Submit a complex task requiring collaboration
6. Verify task completion
7. Trigger self-improvement cycle
8. Validate improvements
9. Test sleep-wake cycle
10. Verify system recovery after restart

The system should be able to continue developing itself after this point.
```