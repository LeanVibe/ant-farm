# Project Backlog: LeanVibe Agent Hive

This document provides a comprehensive, prioritized backlog for the LeanVibe Agent Hive project using the MoSCoW method. Each task is detailed enough to serve as a prompt for coding agents.

## Current Project Status

**Reality Check**: Despite extensive documentation claiming advanced features, the codebase consists primarily of placeholder files. The project is in Phase 0 - foundational scaffolding with significant implementation gaps.

---

## Phase 0: Foundation & Technical Debt (Immediate Priority)

### Task T1: Consolidate Startup Scripts into Unified CLI
**Priority**: Must Have  
**Status**: Not Started  
**Files**: Create `src/cli/main.py`, modify `pyproject.toml`, remove old scripts

**Problem**: Multiple confusing startup scripts (`bootstrap.py`, `start_agent.py`, `agent:runner.py`, etc.) clutter the root directory.

**Implementation**:
1. Create directory `src/cli/`
2. Create `src/cli/main.py` using Typer library
3. Add script entry in `pyproject.toml`: `hive = "src.cli.main:app"`
4. Implement commands:
   - `hive run-agent`: Replace `start_agent.py` and `src/agents/runner.py`
   - `hive bootstrap`: Replace `bootstrap.py` and `autonomous_bootstrap.py`
   - `hive start-api`: Replace `start_api.py`
   - `hive init-db`: Replace `init_db.py`
5. Delete redundant root scripts

**Acceptance Criteria**: Single `hive` command manages all operations, clean root directory.

### Task T2: Reconcile Documentation with Reality
**Priority**: Must Have  
**Status**: Not Started  
**Files**: `README.md`, various markdown files

**Problem**: Documentation falsely claims "Phase 3 Complete" and advanced features that don't exist.

**Implementation**:
1. Review and edit `README.md` to remove false completion claims
2. Update status to "Phase 0: Foundational Scaffolding"
3. Correct Quick Start instructions to use new CLI
4. Remove or qualify claims about non-existent features

**Acceptance Criteria**: Documentation accurately reflects current development stage.

### Task T3: Define Monorepo Structure
**Priority**: Must Have  
**Status**: Not Started  
**Files**: Consolidate `leanvibe-hive/` contents

**Problem**: Unclear relationship between root project and `leanvibe-hive/` subdirectory.

**Implementation**:
1. Review `leanvibe-hive/` directory contents
2. Merge relevant code into primary `src/` directory
3. Remove or clearly document separate concerns
4. Ensure single source of truth for configuration

**Acceptance Criteria**: Clear, unambiguous project structure.

### Task T4: Establish Testing Foundation
**Priority**: Must Have  
**Status**: Not Started  
**Files**: Create `tests/unit/`, `tests/integration/`, configure pytest

**Problem**: Minimal testing despite >90% coverage requirement.

**Implementation**:
1. Create structured test directories
2. Configure pytest in `pyproject.toml`
3. Set up coverage reporting
4. Create test templates for core components

**Acceptance Criteria**: Testing infrastructure ready for >90% coverage target.

---

## Phase 1: Core Infrastructure (Must Have)

### Task M1: Implement Task Queue System
**Priority**: Must Have  
**Status**: Placeholder exists  
**Files**: `src/core/task_queue.py`, `src/core/models.py`, `tests/unit/test_task_queue.py`

**Problem**: `task_queue.py` is placeholder lacking priority, dependencies, retries, timeouts.

**Implementation**:
1. Define `Task`, `TaskStatus`, `TaskPriority` models in `models.py`
2. Implement `TaskQueue` class with Redis backend
3. Use separate Redis lists for priorities (`queue:p1`, `queue:p3`, etc.)
4. Implement dependency logic using Redis hashes
5. Add exponential backoff retry mechanism
6. Implement timeout monitoring for in-progress tasks
7. Write comprehensive unit tests

**Acceptance Criteria**: 
- Tasks handled by priority
- Dependencies respected
- Failed tasks retry with backoff
- All tests pass with >90% coverage

### Task M2: Implement Agent Orchestrator
**Priority**: Must Have  
**Status**: Placeholder exists  
**Files**: `src/core/orchestrator.py`, `tests/integration/test_orchestrator.py`

**Problem**: No agent lifecycle management or task assignment logic.

**Implementation**:
1. Implement `AgentOrchestrator` class
2. Add `spawn`, `monitor`, `terminate` methods
3. Create health check mechanism
4. Use SQLAlchemy for persistent agent registry
5. Implement capability-based task assignment
6. Write integration tests for full lifecycle

**Acceptance Criteria**:
- Agents can be spawned, monitored, terminated
- Tasks assigned based on capabilities
- Health monitoring functional

### Task M3: Implement Message Broker
**Priority**: Must Have  
**Status**: Placeholder exists  
**Files**: `src/core/message_broker.py`, `tests/unit/test_message_broker.py`

**Problem**: No messaging implementation for inter-agent communication.

**Implementation**:
1. Implement `MessageBroker` using Redis Pub/Sub
2. Add persistence layer with Redis Streams
3. Create dead-letter queue for failed messages
4. Implement topic-based routing
5. Add message history and replay
6. Write unit tests for all functionality

**Acceptance Criteria**:
- Real-time messaging between agents
- Offline agent support
- Message persistence and replay

### Task M4: Complete Database Models
**Priority**: Must Have  
**Status**: Incomplete implementation  
**Files**: `src/core/models.py`, `alembic/versions/`

**Problem**: SQLAlchemy models incomplete, missing relationships and pgvector.

**Implementation**:
1. Define all tables as complete SQLAlchemy models:
   - `agents` (id, name, type, capabilities, status, etc.)
   - `tasks` (id, title, description, priority, dependencies, etc.)
   - `sessions` (id, name, agents, state, etc.)
   - `contexts` (id, content, embedding vector, importance, etc.)
   - `conversations` (id, from/to agents, content, embedding, etc.)
   - `system_checkpoints` (id, state, git_commit_hash, etc.)
2. Add pgvector support for embedding columns
3. Define all relationships and foreign keys
4. Add performance indexes
5. Create Alembic migration

**Acceptance Criteria**:
- Complete schema matches documentation
- pgvector integration working
- Migration applies successfully

### Task M5: Implement Base Agent Logic
**Priority**: Must Have  
**Status**: Placeholder exists  
**Files**: `src/agents/base_agent.py`

**Problem**: Abstract methods not implemented, no task processing.

**Implementation**:
1. Implement task processing loop (claim/process/complete)
2. Integrate with MessageBroker for communication
3. Integrate with ContextEngine for memory
4. Add tool execution framework
5. Implement health reporting
6. Add graceful shutdown handling

**Acceptance Criteria**:
- Agents can process tasks end-to-end
- Inter-agent communication working
- Context storage/retrieval functional

### Task M6: Implement CLI Tool Execution
**Priority**: Must Have  
**Status**: Configuration exists, no implementation  
**Files**: `src/agents/base_agent.py`, `src/core/config.py`

**Problem**: No actual CLI tool execution despite configuration for opencode/claude/gemini.

**Implementation**:
1. Implement tool selection logic in BaseAgent
2. Add subprocess execution for CLI tools
3. Implement smart fallback mechanism
4. Handle stdin/stdout/stderr properly
5. Add rate limiting and error handling
6. Write unit tests for tool execution

**Acceptance Criteria**:
- Agents can execute configured CLI tools
- Fallback works when preferred tool fails
- Proper error handling and logging

### Task M7: Implement Core API Endpoints
**Priority**: Must Have  
**Status**: Placeholder exists  
**Files**: `src/api/main.py`

**Problem**: Missing most required endpoints for system control.

**Implementation**:
1. Implement agent CRUD endpoints:
   - `POST /api/v1/agents` - Create agent
   - `GET /api/v1/agents` - List agents
   - `GET /api/v1/agents/{id}` - Get agent details
   - `PUT /api/v1/agents/{id}` - Update agent
   - `DELETE /api/v1/agents/{id}` - Terminate agent
2. Implement task endpoints:
   - `POST /api/v1/tasks` - Submit task
   - `GET /api/v1/tasks` - List tasks
   - `GET /api/v1/tasks/{id}` - Get status
   - `PUT /api/v1/tasks/{id}/cancel` - Cancel task
3. Add Pydantic models for validation
4. Implement JWT authentication
5. Add error handling middleware

**Acceptance Criteria**:
- All CRUD operations functional
- Proper validation and authentication
- Error handling implemented

### Task M8: Implement WebSocket Service
**Priority**: Must Have  
**Status**: Empty files  
**Files**: `src/api/main.py`, `src/web/dashboard/services/websocket-service.js`

**Problem**: Dashboard claims real-time updates but WebSocket not implemented.

**Implementation**:
1. Add `/ws/events` endpoint in FastAPI
2. Integrate with MessageBroker for event streaming
3. Implement client-side WebSocket connection
4. Add reconnection logic and error handling
5. Create event filtering and routing

**Acceptance Criteria**:
- Real-time events stream to dashboard
- Robust connection handling
- Event filtering working

---

## Phase 2: Autonomous Capabilities (Should Have)

### Task S1: Implement Context Engine
**Priority**: Should Have  
**Status**: Placeholder exists  
**Files**: `src/core/context_engine.py`, `src/core/advanced_context_engine.py`

**Problem**: No semantic search or context compression implementation.

**Implementation**:
1. Implement semantic search using pgvector cosine similarity
2. Add OpenAI embedding generation
3. Implement context compression using LLM
4. Create hierarchical memory system
5. Add importance scoring
6. Implement context sharing between agents

**Acceptance Criteria**:
- Semantic search functional
- Context compression working
- Memory hierarchy implemented

### Task S2: Implement Meta-Agent
**Priority**: Should Have  
**Status**: Placeholder exists  
**Files**: `src/agents/meta_agent.py`

**Problem**: No self-improvement capabilities implemented.

**Implementation**:
1. Inherit from BaseAgent
2. Implement system performance analysis
3. Add improvement proposal generation
4. Integrate with SelfModifier
5. Add safety checks for changes
6. Implement A/B testing for prompts

**Acceptance Criteria**:
- System can analyze its own performance
- Improvement proposals generated
- Safe testing implemented

### Task S3: Implement Self-Modifier
**Priority**: Should Have  
**Status**: Empty file  
**Files**: `src/core/self_modifier.py`

**Problem**: Core self-improvement feature completely missing.

**Implementation**:
1. Integrate GitPython for branch management
2. Create Docker-based sandbox environment
3. Implement automated testing in sandbox
4. Add performance validation
5. Create rollback mechanism
6. Add human approval gates

**Acceptance Criteria**:
- Code changes tested safely
- Git integration working
- Rollback mechanism functional

### Task S4: Implement Sleep-Wake Manager
**Priority**: Should Have  
**Status**: Empty file  
**Files**: `src/core/sleep_wake_manager.py`

**Problem**: Sleep-wake cycles claimed but not implemented.

**Implementation**:
1. Add configurable sleep scheduling
2. Implement context consolidation
3. Create state checkpointing
4. Add graceful handoff between cycles
5. Implement wake restoration
6. Add performance optimization during sleep

**Acceptance Criteria**:
- Sleep cycles functional
- Context consolidation working
- State preservation across cycles

---

## Phase 3: User Interface (Could Have)

### Task C1: Implement Dashboard Components
**Priority**: Could Have  
**Status**: Static files exist  
**Files**: `src/web/dashboard/components/*.js`

**Problem**: Dashboard components are static, not connected to backend.

**Implementation**:
1. Connect AgentStatus to live agent data
2. Implement TaskBoard with real task data
3. Add MessageFlow visualization
4. Create SystemMetrics dashboard
5. Implement ContextExplorer
6. Add LogViewer with streaming

**Acceptance Criteria**:
- All components show live data
- Real-time updates working
- Interactive functionality implemented

### Task C2: Implement PWA Features
**Priority**: Could Have  
**Status**: Empty service worker  
**Files**: `src/web/dashboard/sw.js`, `src/web/dashboard/manifest.json`

**Problem**: Claims PWA support but service worker empty.

**Implementation**:
1. Implement service worker for offline support
2. Add caching strategies
3. Implement push notifications
4. Add install prompts
5. Create offline functionality

**Acceptance Criteria**:
- Offline support working
- Push notifications functional
- Installable as PWA

---

## Phase 4: Advanced Features (Won't Have - Current Phase)

### Task W1: Advanced Observability
**Priority**: Won't Have (for now)  
**Status**: Not started

**Implementation**: Prometheus/Grafana integration for advanced metrics.

### Task W2: Additional Specialized Agents
**Priority**: Won't Have (for now)  
**Status**: Not started

**Implementation**: ArchitectAgent, QAAgent, DeveloperAgent classes.

### Task W3: Prompt Optimization System
**Priority**: Won't Have (for now)  
**Status**: Not started

**Implementation**: A/B testing framework for prompt optimization.

---

## Testing Requirements

All tasks must include:
- Unit tests with >90% coverage
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance benchmarks where applicable

## Definition of Done

Each task is complete when:
1. Implementation matches specifications
2. All tests pass with required coverage
3. Documentation updated
4. Code review completed
5. Integration with existing components verified

---

*Last Updated*: Current analysis based on codebase evaluation  
*Next Review*: After Phase 0 completion