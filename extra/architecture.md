# LeanVibe Agent Hive 2.0 - Hybrid Architecture

## Core Concept

The system uses a **hybrid architecture**:
- **Docker**: Runs infrastructure (PostgreSQL, Redis, API)
- **Host Machine**: Runs Claude Code agents in tmux sessions
- **Orchestrator**: Bridges Docker services with host tmux sessions

## Architecture Diagram

```
┌─── Host Machine (MacBook) ──────────────────────────────┐
│                                                          │
│  ┌─── tmux Sessions ────────────────────┐               │
│  │                                      │               │
│  │  Session: bootstrap-agent            │               │
│  │  > claude "Generate task_queue.py"   │               │
│  │                                      │               │
│  │  Session: meta-agent-001             │               │
│  │  > claude "Analyze and improve"      │               │
│  │                                      │               │
│  │  Session: dev-agent-001              │               │
│  │  > claude "Implement feature X"      │               │
│  │                                      │               │
│  └──────────────────────────────────────┘               │
│                     ▲                                    │
│                     │ Spawns & Controls                  │
│                     │                                    │
│  ┌─── Docker Containers ────────────────┐               │
│  │                                      │               │
│  │  ┌─────────────┐  ┌──────────────┐  │               │
│  │  │ PostgreSQL  │  │    Redis     │  │               │
│  │  │  + pgvector │  │ Task Queue   │  │               │
│  │  └─────────────┘  └──────────────┘  │               │
│  │          ▲               ▲           │               │
│  │          └───────┬───────┘           │               │
│  │                  │                   │               │
│  │         ┌──────────────┐             │               │
│  │         │ Orchestrator │             │               │
│  │         │   (Python)   │             │               │
│  │         └──────────────┘             │               │
│  │                  ▲                   │               │
│  │                  │                   │               │
│  │         ┌──────────────┐             │               │
│  │         │   FastAPI    │             │               │
│  │         │   (Port 8000)│             │               │
│  │         └──────────────┘             │               │
│  │                                      │               │
│  └──────────────────────────────────────┘               │
│                                                          │
│  Claude Code CLI (installed locally)                    │
│  ANTHROPIC_API_KEY (in shell environment)               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Bootstrap Process
```bash
# User runs locally (not in Docker):
claude "Create bootstrap/init_agent.py"

# Bootstrap agent then:
1. Connects to PostgreSQL/Redis in Docker
2. Spawns tmux session for meta-agent
3. Sends command: tmux send-keys "claude 'Create task_queue.py'" Enter
4. Monitors output and stores in database
```

### 2. Agent Orchestration

The orchestrator (running in Docker) manages tmux sessions on the host:

```python
class AgentOrchestrator:
    def spawn_agent(self, agent_type: str) -> str:
        """Spawn a new agent in a tmux session."""
        session_name = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        
        # Create tmux session on host
        subprocess.run([
            "tmux", "new-session", "-d", "-s", session_name
        ])
        
        # Agent will poll tasks from Redis and execute via Claude Code
        subprocess.run([
            "tmux", "send-keys", "-t", session_name,
            f"python src/agents/runner.py --type {agent_type}",
            "Enter"
        ])
        
        return session_name
```

### 3. Agent Runner (Executes on Host)

```python
# src/agents/runner.py - Runs in tmux session on host
class AgentRunner:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.redis = redis.from_url("redis://localhost:6379")
        self.db = psycopg2.connect("postgresql://localhost:5432/leanvibe_hive")
    
    async def run(self):
        while True:
            # Get task from Redis queue
            task = await self.get_next_task()
            
            # Execute via Claude Code CLI
            result = subprocess.run(
                ["claude", "--no-interactive", task.prompt],
                capture_output=True,
                text=True
            )
            
            # Store result in database
            await self.store_result(task.id, result.stdout)
```

## Key Differences from Pure Docker

### What's in Docker:
- PostgreSQL + pgvector (persistent storage)
- Redis (task queue & messaging)
- FastAPI (REST API & WebSocket)
- Orchestrator (manages tmux sessions)
- Web Dashboard (UI)

### What's on Host:
- Claude Code CLI (installed via Homebrew/curl)
- tmux sessions (one per agent)
- Agent runner scripts
- Workspace files (code being modified)
- Git repository

### What's NOT in Docker:
- ANTHROPIC_API_KEY (only needed on host)
- Claude Code execution
- Agent processes themselves
- Code generation/modification

## Benefits of Hybrid Approach

1. **Direct Claude Access**: Agents use your local Claude Code with your API key
2. **Better Performance**: No Docker overhead for LLM calls
3. **Easier Debugging**: Can attach to tmux sessions directly
4. **Local File Access**: Agents work directly with your filesystem
5. **Simpler Networking**: No complex Docker networking for Claude API

## Setup Requirements

### On Host Machine:
```bash
# Install Claude Code
brew install claude  # or official installer

# Install tmux
brew install tmux

# Set API key in shell
export ANTHROPIC_API_KEY="sk-ant-..."

# Install Python for agent runners
brew install python@3.11
pip install redis psycopg2-binary
```

### Docker Only Needs:
```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg15
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  api:
    build: .
    ports:
      - "8000:8000"
    # No ANTHROPIC_API_KEY needed!
```

## Agent Lifecycle

### 1. Spawn
```bash
# Orchestrator creates tmux session
tmux new-session -d -s meta-agent-001

# Starts agent runner in session
tmux send-keys -t meta-agent-001 \
  "python src/agents/runner.py --type meta" Enter
```

### 2. Execute
```bash
# Agent runner polls Redis for tasks
# When task received, executes:
claude "Improve the task_queue.py module"
```

### 3. Monitor
```bash
# View agent output
tmux attach -t meta-agent-001

# List all agents
tmux ls

# Kill agent
tmux kill-session -t meta-agent-001
```

## File Access Pattern

```
Host Filesystem
├── leanvibe-hive/        # Project root
│   ├── src/              # Source code (agents modify this)
│   ├── tests/            # Tests (agents write these)
│   ├── workspace/        # Temp files for agents
│   └── logs/             # Agent logs
│
├── ~/.tmux/              # tmux configs
└── ~/.anthropic/         # Claude Code config
```

## Communication Flow

1. **User** → API (Docker) → Redis (Docker)
2. **Orchestrator** (Docker) → tmux (Host) → Agent Runner (Host)
3. **Agent Runner** (Host) → Claude Code (Host) → Files (Host)
4. **Agent Runner** (Host) → PostgreSQL (Docker) → Store results

## Example Commands

### Start Infrastructure
```bash
# Only starts PostgreSQL, Redis, API
docker-compose up -d

# No agents running yet
tmux ls  # Empty
```

### Bootstrap System
```bash
# Run locally, not in Docker
claude "Create the bootstrap agent"
python bootstrap/init_agent.py

# Bootstrap spawns other agents in tmux
tmux ls
# bootstrap-agent
# meta-agent-001
# dev-agent-001
```

### Monitor Agents
```bash
# Attach to see what agent is doing
tmux attach -t meta-agent-001

# View all agent sessions
tmux ls | grep agent

# Check agent status via API
curl http://localhost:8000/api/v1/agents
```

## Important Notes

1. **No Docker networking complexity** - Agents run on host with direct internet access
2. **No API key in Docker** - Only needed in host shell environment
3. **Direct file access** - Agents modify files directly, no volume mounting needed
4. **Easy debugging** - Just attach to tmux session to see Claude Code output
5. **Resource efficient** - LLM calls don't go through Docker overhead

This hybrid approach gives us the best of both worlds:
- **Docker** for reliable infrastructure
- **Host execution** for Claude Code agents