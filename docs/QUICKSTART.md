# LeanVibe Agent Hive 2.0 - Quick Start Guide

## Overview

Get your autonomous multi-agent development system up and running in 5 minutes using the unified `hive` CLI.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose  
- At least one CLI agentic coding tool (opencode, claude, gemini)

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive

# Install dependencies
uv sync
# or: pip install -r requirements.txt

# Configure environment (copy and customize)
cp .env.example .env
```

### 2. Check CLI Tools

```bash
# Check available CLI agentic coding tools
hive agent tools

# If no tools found, install one:
# opencode (recommended):
curl -fsSL https://opencode.ai/install | bash
# Claude CLI: https://claude.ai/cli  
# Gemini CLI: https://ai.google.dev/gemini-api/docs/cli
```

### 3. Initialize System

```bash
# Initialize everything (Docker services, DB, migrations)
hive init

# Start the system (API server, core services)
hive system start

# Check system health
hive system status
```

## First Steps

### 1. Bootstrap the Agent System

```bash
# Bootstrap with automatic CLI tool detection
hive agent bootstrap

# Or spawn individual agents
hive agent spawn meta
hive agent spawn architect  
hive agent spawn qa
```

### 2. Submit Your First Task

```bash
# Submit a development task
hive task submit "Create a simple hello world application"

# List all tasks
hive task list

# Check task status
hive task status <task-id>
```

### 3. Monitor System

```bash
# Check agent status
hive agent list
hive agent describe meta

# System health
hive system status

# Access API documentation
open http://localhost:9001/api/docs
```

## API Access

The system runs on **non-standard ports for security**:

- **API Server**: http://localhost:9001 (instead of 80/443)
- **PostgreSQL**: localhost:5433 (instead of 5432)
- **Redis**: localhost:6381 (instead of 6379)

### API Examples

```bash
# Health check
curl http://localhost:9001/api/v1/health

# List agents (when auth implemented)
curl http://localhost:9001/api/v1/agents

# Submit task via API
curl -X POST http://localhost:9001/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Implement feature X", 
    "priority": "high",
    "agent_type": "meta"
  }'
```

## CLI Command Reference

### System Management
```bash
hive init                    # Initialize system (DB, Docker, migrations)
hive system start            # Start API server and services
hive system stop             # Stop system gracefully  
hive system status           # Check system health
hive system restart          # Restart system
```

### Agent Management  
```bash
hive agent spawn meta        # Spawn meta-agent
hive agent spawn architect   # Spawn architect agent
hive agent list              # List all agents
hive agent describe <name>   # Agent details
hive agent stop <name>       # Stop specific agent
hive agent bootstrap         # Auto-bootstrap with available tools
hive agent tools             # Check available CLI tools
```

### Task Management
```bash
hive task submit "description"  # Submit new task
hive task list                  # List all tasks  
hive task status <id>           # Check task status
```

### Context & Memory
```bash
hive context populate          # Populate context engine with codebase
hive context search "query"    # Search system knowledge
```

## Advanced Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Database (non-standard port)
DATABASE_URL=postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive

# Redis (non-standard port)
REDIS_URL=redis://localhost:6381

# API (non-standard port)
API_HOST=0.0.0.0
API_PORT=9001

# CLI Tool Preferences
PREFERRED_CLI_TOOL=opencode

# Optional API Keys (for fallback)
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
GOOGLE_AI_API_KEY=your_gemini_key

# Agent Configuration
MAX_CONCURRENT_AGENTS=50
AGENT_HEARTBEAT_INTERVAL=30
SELF_IMPROVEMENT_ENABLED=true
```

### CLI Tool Preferences

```bash
# Set preferred tool
export PREFERRED_CLI_TOOL=opencode

# Check detection
hive agent tools

# Force specific tool for session
PREFERRED_CLI_TOOL=claude hive agent spawn meta
```

## Development Workflow

### 1. Make Changes
```bash
# Create feature branch
git checkout -b feature/new-capability

# Use the system to help develop
hive task submit "Implement new agent capability X"

# Monitor progress
hive agent list
hive task list
```

### 2. Testing
```bash
# Run tests
pytest -q

# Run specific test
pytest tests/unit/test_task_queue.py -v

# Coverage
pytest --cov --cov-report=term-missing

# Lint and format
ruff check .
ruff format .
```

### 3. Self-Improvement
```bash
# Submit self-improvement task
hive task submit "Optimize task queue performance"

# The meta-agent will:
# 1. Analyze the codebase
# 2. Propose improvements  
# 3. Test changes
# 4. Commit if successful
```

## Troubleshooting

### Common Issues

#### 1. No CLI Tools Found
```bash
# Check what's available
hive agent tools

# Install opencode (recommended)
curl -fsSL https://opencode.ai/install | bash

# Verify installation
which opencode
```

#### 2. Database Connection Error
```bash
# Check Docker services
docker compose ps

# Restart services
docker compose down
docker compose up -d postgres redis

# Check connection
hive system status
```

#### 3. Port Already in Use
```bash
# Check what's using port 9001
lsof -i :9001

# Stop conflicting process or change port
export API_PORT=9002
hive system start
```

#### 4. Agent Spawn Failures
```bash
# Check agent logs
hive agent describe <agent-name>

# Check available tools
hive agent tools

# Try different agent type
hive agent spawn architect
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start with verbose output
hive system start --reload

# Check system logs
hive system logs
```

### Health Checks

```bash
# Full system status
hive system status

# Individual service checks
curl http://localhost:8000/api/v1/health
docker compose ps
```

## Next Steps

1. **Submit Complex Tasks**: Try multi-step development tasks
2. **Monitor Self-Improvement**: Watch the meta-agent enhance itself
3. **Customize Agents**: Modify agent behavior in `src/agents/`
4. **Explore API**: Visit http://localhost:8000/api/docs
5. **Read Architecture**: See `docs/system-architecture.md`

## CLI Tool Details

### opencode (Recommended)
- **Install**: `curl -fsSL https://opencode.ai/install | bash`
- **Features**: Responsive TUI, multi-session, 75+ LLM providers
- **Best for**: Interactive development, complex refactoring

### Claude CLI  
- **Install**: https://claude.ai/cli
- **Features**: Direct Claude Pro/Max access
- **Best for**: Advanced reasoning, code review

### Gemini CLI
- **Install**: https://ai.google.dev/gemini-api/docs/cli  
- **Features**: Google AI integration
- **Best for**: Research, documentation

## Support

- **Issues**: https://github.com/leanvibe/agent-hive/issues
- **Documentation**: `docs/` directory
- **API Reference**: http://localhost:9001/api/docs
- **CLI Help**: `hive --help`

---

**🤖 Happy autonomous development! The agents are ready to build themselves! 🚀**