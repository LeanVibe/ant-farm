# LeanVibe Agent Hive 2.0 - System Context

## Project Overview

You are building LeanVibe Agent Hive 2.0, a self-improving autonomous multi-agent system that uses Claude instances to develop and enhance itself continuously. The system should be able to build itself, improve its own code, and operate 24/7 with minimal human intervention.

## Core Principles

1. **Simplicity First**: Every component must be simple enough for an agent to understand and modify
2. **Test-Driven Development**: Write tests before implementation (pytest, >90% coverage)
3. **Self-Documenting**: Code should be clear with comprehensive docstrings
4. **Fail-Safe Design**: All operations must be reversible (git-based rollback)
5. **Incremental Progress**: Small, working commits that build on each other

## Technology Stack

- **Language**: Python 3.11+ with type hints
- **Backend**: FastAPI (async/await patterns everywhere)
- **Database**: PostgreSQL 15 with pgvector extension
- **Cache/Queue**: Redis 7.0+ (persistence enabled)
- **Frontend**: LitPWA (Web Components + PWA)
- **Package Manager**: uv (not pip)
- **Testing**: pytest with pytest-asyncio
- **Logging**: structlog for structured logging

## Architecture Patterns

### Async Everywhere
```python
async def process_task(self, task: Task) -> Result:
    """All I/O operations must be async."""
    async with self.db.begin() as conn:
        result = await conn.execute(query)
    return await self.process_result(result)
```

### Dependency Injection
```python
def __init__(self, db: AsyncSession, redis: Redis, config: Settings):
    """Use constructor injection for dependencies."""
    self.db = db
    self.redis = redis
    self.config = config
```

### Error Handling
```python
try:
    result = await self.execute_operation()
except SpecificError as e:
    logger.warning("Operation failed", error=str(e))
    await self.rollback()
    raise
finally:
    await self.cleanup()
```

## Component Specifications

### Task Queue
- Redis-based with priority levels (1-9)
- At-least-once delivery guarantee
- Exponential backoff for retries
- Task dependencies support
- Timeout monitoring

### Agent System
- Base agent class with standard interface
- Specialized agents inherit from base
- Each agent has unique capabilities
- Graceful shutdown support
- Health monitoring

### Context Engine
- Vector embeddings with pgvector
- Semantic search capabilities
- Context compression for efficiency
- Hierarchical memory structure

### Self-Modification
- Safe sandboxed execution
- Git-based version control
- Automated testing before apply
- Performance validation
- Rollback on regression

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

## Development Workflow

1. Read requirements in IMPLEMENTATION.md
2. Write tests first (TDD)
3. Implement minimal code to pass tests
4. Refactor for clarity
5. Add comprehensive documentation
6. Commit with descriptive message
7. Run full test suite
8. Create pull request if applicable

## Remember

- This system builds itself - make code that agents can understand
- Every component should be testable in isolation
- Prefer composition over inheritance
- Use type hints everywhere
- Handle errors gracefully
- Log important events
- Keep functions small and focused
- Document "why" not just "what"