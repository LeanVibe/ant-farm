# LeanVibe Agent Hive Rewrite - System Architecture Specifications

## Executive Summary

The LeanVibe Agent Hive rewrite is a comprehensive multi-agent orchestration system designed for autonomous software development using Claude instances. This document outlines the technical architecture, database design, and implementation specifications for a complete ground-up rewrite incorporating modern patterns and lessons learned from existing systems.

## Technology Stack

- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: LitPWA (Progressive Web App)
- **Database**: PostgreSQL 15+ with pgvector extension
- **Cache/Message Broker**: Redis 7.0+
- **Orchestration**: Docker Compose
- **LLM Integration**: Claude 3.5 Sonnet/Haiku via Anthropic API

## System Architecture Overview

### Core Components

1. **Agent Orchestrator** (`src/core/orchestrator.py`)
   - Central coordination engine
   - Agent lifecycle management
   - Task delegation and monitoring
   - Sleep-wake cycle coordination

2. **Message Broker** (`src/core/message_broker.py`)
   - Redis-based pub/sub system
   - Inter-agent communication
   - Real-time event streaming
   - Message persistence and delivery guarantees

3. **Context Engine** (`src/core/context_engine.py`)
   - Vector-based semantic storage
   - Context compression and retrieval
   - Memory consolidation during sleep cycles
   - Hierarchical context management

4. **Self-Modification System** (`src/core/self_modifier.py`)
   - Code generation and modification
   - Safe execution sandbox
   - Version control integration
   - Rollback mechanisms

5. **Observability Layer** (`src/observability/`)
   - Real-time monitoring dashboard
   - Performance metrics collection
   - Event logging and analysis
   - Health checks and alerting

### Agent Types

1. **Strategic Partner Agent**
   - Human-AI interaction interface
   - Requirements gathering
   - Decision approval workflows
   - Strategic guidance

2. **Product Manager Agent**
   - Project planning and coordination
   - Backlog management
   - Sprint planning
   - Resource allocation

3. **Specialized Development Agents**
   - Architect Agent (system design)
   - Backend Agent (API development)
   - Frontend Agent (UI/UX)
   - DevOps Agent (deployment/monitoring)
   - QA Agent (testing/validation)

4. **Meta-Agent**
   - Self-improvement coordination
   - Prompt optimization
   - System evolution

## Database Schema Design

### PostgreSQL Tables

```sql
-- Core agent registry
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    role TEXT,
    capabilities JSON,
    system_prompt TEXT,
    status VARCHAR(50) DEFAULT 'inactive',
    tmux_session VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Session management
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    type VARCHAR(100) NOT NULL,
    agents JSON,
    state JSON,
    tmux_session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW()
);

-- Task queue and execution
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    assigned_agent_id UUID REFERENCES agents(id),
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    dependencies JSON,
    context JSON,
    result JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Inter-agent conversations with semantic search
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    from_agent_id UUID REFERENCES agents(id),
    to_agent_id UUID REFERENCES agents(id),
    message_type VARCHAR(100),
    content TEXT,
    embedding vector(1536), -- OpenAI embedding dimension
    context_refs UUID[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context storage with hierarchical structure
CREATE TABLE contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    session_id UUID REFERENCES sessions(id),
    type VARCHAR(100),
    title VARCHAR(255),
    content TEXT,
    embedding vector(1536),
    importance_score FLOAT DEFAULT 0.5,
    parent_context_id UUID REFERENCES contexts(id),
    created_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP DEFAULT NOW()
);

-- System state checkpoints
CREATE TABLE system_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_type VARCHAR(100),
    state JSON,
    git_commit_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sleep-wake cycle management
CREATE TABLE sleep_wake_cycles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    cycle_type VARCHAR(100),
    sleep_time TIMESTAMP,
    wake_time TIMESTAMP,
    consolidation_summary TEXT,
    context_changes JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance and observability
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(255),
    metric_value FLOAT,
    tags JSON,
    agent_id UUID REFERENCES agents(id),
    session_id UUID REFERENCES sessions(id),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent ON tasks(assigned_agent_id);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_contexts_agent ON contexts(agent_id);
CREATE INDEX idx_contexts_embedding ON contexts USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_performance_timestamp ON performance_metrics(timestamp);
```

### Redis Data Structures

```python
# Message queues
agent_messages:{agent_id} = List[Message]

# Real-time communication channels
pub/sub channels:
- agent_events:{agent_id}
- system_events
- task_updates
- health_checks

# Session state caching
session_state:{session_id} = JSON

# Distributed locks
locks:agent:{agent_id} = timestamp
locks:task:{task_id} = agent_id

# Rate limiting
rate_limit:{agent_id}:{minute} = counter

# Performance metrics streaming
metrics:stream = time-series data
```

## API Design

### Core Endpoints

```python
# Agent Management
POST /api/v1/agents - Create new agent
GET /api/v1/agents - List all agents
GET /api/v1/agents/{agent_id} - Get agent details
PUT /api/v1/agents/{agent_id} - Update agent
DELETE /api/v1/agents/{agent_id} - Deactivate agent

# Session Management
POST /api/v1/sessions - Create new session
GET /api/v1/sessions - List active sessions
GET /api/v1/sessions/{session_id} - Get session details
POST /api/v1/sessions/{session_id}/start - Start session
POST /api/v1/sessions/{session_id}/stop - Stop session

# Task Management
POST /api/v1/tasks - Create new task
GET /api/v1/tasks - List tasks with filters
GET /api/v1/tasks/{task_id} - Get task details
PUT /api/v1/tasks/{task_id}/assign - Assign task to agent
POST /api/v1/tasks/{task_id}/complete - Mark task complete

# Communication
POST /api/v1/messages/send - Send message between agents
GET /api/v1/messages/{agent_id} - Get agent messages
POST /api/v1/context/search - Semantic context search
POST /api/v1/context/store - Store context entry

# System Operations
POST /api/v1/system/checkpoint - Create system checkpoint
POST /api/v1/system/sleep - Initiate sleep cycle
POST /api/v1/system/wake - Initiate wake cycle
GET /api/v1/system/health - System health check

# Self-Modification
POST /api/v1/self-modify/propose - Propose system change
POST /api/v1/self-modify/apply - Apply approved change
GET /api/v1/self-modify/history - Get modification history
```

## Security Considerations

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- Agent-specific permissions
- API key management for Claude integration

### Data Security
- Encrypted communication channels
- Secure secret management
- Database encryption at rest
- Input validation and sanitization

### System Security
- Sandboxed code execution
- Resource limitations
- Network isolation
- Audit logging

## Performance Requirements

### Scalability Targets
- Support 50+ concurrent agents
- Handle 1000+ tasks per minute
- Sub-100ms API response times
- 99.9% uptime

### Resource Optimization
- Connection pooling for PostgreSQL
- Redis caching strategy
- Async I/O for all operations
- Memory-efficient context storage

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- Database setup and migrations
- FastAPI application structure
- Redis integration
- Basic agent lifecycle management

### Phase 2: Agent Communication (Weeks 3-4)
- Message broker implementation
- Inter-agent messaging
- Context storage and retrieval
- Basic observability

### Phase 3: Advanced Features (Weeks 5-6)
- Sleep-wake cycle implementation
- Self-modification system
- Performance optimization
- Security hardening

### Phase 4: Frontend & Polish (Weeks 7-8)
- LitPWA dashboard development
- Real-time monitoring UI
- Mobile optimization
- Documentation completion

## Success Metrics

### Technical Metrics
- Agent uptime: >95%
- Message delivery rate: >99%
- API response time: <100ms
- Context retrieval accuracy: >90%

### Functional Metrics
- Task completion rate: >85%
- Agent collaboration effectiveness: >80%
- Self-improvement iterations: >5 per week
- System adaptation speed: <24 hours

## Risk Assessment

### High Risk
- **Context window overflow**: Implement sliding window and compression
- **Agent communication failures**: Add circuit breakers and fallbacks
- **Self-modification bugs**: Comprehensive testing and rollback mechanisms

### Medium Risk
- **Performance bottlenecks**: Monitoring and optimization strategies
- **Claude API rate limits**: Request queuing and fallback models
- **Data consistency**: ACID transactions and eventual consistency patterns

### Low Risk
- **Infrastructure failures**: High availability and backup strategies
- **Security vulnerabilities**: Regular security audits and updates

## Conclusion

This architecture provides a solid foundation for building a production-ready, self-improving multi-agent system that can autonomously develop software while maintaining reliability, security, and performance standards.