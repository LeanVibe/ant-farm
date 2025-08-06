# LeanVibe Agent Hive 2.0

A self-improving multi-agent development platform that works with multiple CLI agentic coding tools.

## 🛠️ Supported CLI Tools

The system auto-detects and uses available CLI agentic coding tools in priority order:

1. **🥇 opencode** (Preferred) - AI coding agent built for the terminal
   - Install: `curl -fsSL https://opencode.ai/install | bash`
   - Usage: Responsive terminal UI with multi-session support

2. **🥈 Claude Code CLI** - Anthropic's command-line coding assistant  
   - Install: https://claude.ai/cli
   - Usage: Direct Claude access via terminal

3. **🥉 Gemini CLI** - Google's AI coding assistant
   - Install: https://ai.google.dev/gemini-api/docs/cli
   - Usage: Google AI integration

**Fallback**: Anthropic/OpenAI APIs (if CLI tools unavailable)

## Quick Start

### Prerequisites
- Docker Desktop  
- tmux (auto-installed on macOS)
- **At least one CLI tool** (opencode, Claude CLI, or Gemini CLI)
- **Optional**: Ollama (for local embeddings)
- **Optional**: API keys for fallback

### Setup (5 minutes)

```bash
# 1. Check/install CLI tools
make tools

# 2. Set up environment  
make setup

# 3. Start everything
make quick
```

This will:
- Start PostgreSQL and Redis in Docker
- Run the bootstrap agent
- Generate all core components
- Start the API server
- Spawn initial agents

### Verify Installation

```bash
# Check available CLI tools
make tools

# Check system status
make status

# View API documentation
open http://localhost:8000/docs

# Submit a test task
make task-submit TASK="Analyze system performance"

# Watch agents work
make logs
```

## Architecture

This system uses a **hybrid architecture** with **multi-CLI support**:

- **Docker**: Runs infrastructure (PostgreSQL, Redis, API)
- **Host Machine**: Runs AI coding agents in tmux sessions using any available CLI tool
- **Orchestrator**: Bridges Docker services with host tmux sessions
- **Smart Fallback**: Automatically switches between opencode → Claude → Gemini → API

### Why Hybrid with Multi-CLI Support?

1. **Tool Flexibility**: Use any preferred CLI agentic coding tool
2. **Smart Fallback**: Automatic failover between tools and APIs
3. **Better Performance**: No Docker overhead for LLM calls
4. **Easier Debugging**: Can attach to tmux sessions directly
5. **Local File Access**: Agents work directly with your filesystem

## Components

### Core Infrastructure
- **Task Queue**: Redis-based priority queue with dependencies
- **Message Broker**: Agent-to-agent communication via Redis pub/sub
- **Context Engine**: Semantic memory with pgvector embeddings
- **Orchestrator**: Agent lifecycle management

### Agent Types
- **Bootstrap Agent**: Builds the initial system
- **Meta Agent**: Analyzes and improves the system
- **Developer Agents**: Generate and modify code
- **QA Agents**: Write tests and verify quality

### API & UI
- **FastAPI Backend**: REST API with WebSocket support
- **Real-time Dashboard**: Monitor agents and tasks
- **Development Tools**: pgAdmin, Redis Commander

## Usage

### CLI Tool Management

```bash
# Check available tools
make tools

# View detailed status
make status

# Force specific tool (optional)
export PREFERRED_CLI_TOOL=opencode
```

```bash
# Start core services only
make up

# Start with development tools
make dev

# Full production setup
make start

# Stop everything
make stop
```

### Agent Operations

```bash
# List active agents
make agent-list

# Create new agent
make agent-create TYPE=developer NAME=dev-agent-002

# View tmux sessions
make tmux-list

# Attach to agent session
tmux attach -t meta-agent
```

### Task Management

```bash
# Submit task
make task-submit TASK="Create user authentication system"

# List tasks
make task-list

# Monitor task queue
make redis-monitor
```

### Development

```bash
# Run tests
make test

# Format code
make format

# Check health
make health

# View logs
make logs
```

## Self-Improvement Process

Once bootstrapped, the system will:

1. **Meta-Agent Analysis**: Continuously analyze system performance
2. **Improvement Proposals**: Generate and test optimizations
3. **Safe Deployment**: Apply changes with automatic rollback
4. **Learning**: Update agent prompts based on results

## Monitoring

### Service URLs
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:5050 (admin@leanvibe.com / admin)
- **Redis Commander**: http://localhost:8081

### System Status
```bash
# Overall health
make health

# Service logs
make logs

# Agent status
curl http://localhost:8000/api/v1/agents
```

## Configuration

### Environment Variables (.env)
```bash
# CLI Tool Preference (optional - auto-detected)
PREFERRED_CLI_TOOL=opencode  # opencode, claude, gemini

# API Keys (fallback if CLI tools unavailable)
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here  
GOOGLE_AI_API_KEY=your-google-key-here

# Infrastructure  
DATABASE_URL=postgresql://hive_user:hive_pass@localhost:5432/leanvibe_hive
REDIS_URL=redis://localhost:6379
```

### Agent Configuration
- Agents run in individual tmux sessions
- System prompts stored in database
- Configuration via API endpoints

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
make status
make logs
```

**Database connection fails:**
```bash
make db-shell
docker-compose logs postgres
```

**Redis issues:**
```bash
make redis-shell
docker-compose logs redis
```

**API not responding:**
```bash
curl http://localhost:8000/health
make logs-api
```

### Full Reset
```bash
make reset  # Deletes all data
make setup  # Start fresh
```

## Development

### Project Structure
```
leanvibe-hive/
├── src/core/           # Core system components
├── src/agents/         # Agent implementations  
├── src/api/            # FastAPI application
├── bootstrap/          # Bootstrap agent
├── docker/            # Docker configurations
├── scripts/           # Database and utility scripts
└── tests/             # Test suite
```

### Adding New Agents

1. Create agent class in `src/agents/`
2. Register with orchestrator
3. Define capabilities and prompts
4. Submit via API or let meta-agent create

### Testing
```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration
```

## Production Deployment

### Security Checklist
- [ ] Change default passwords
- [ ] Set secure SECRET_KEY
- [ ] Use environment-specific .env files
- [ ] Enable HTTPS
- [ ] Set up monitoring and alerting

### Scaling
- Scale agent workers: `docker-compose --scale agent-worker=10`
- Add Redis clustering for high availability
- Use read replicas for PostgreSQL

## Contributing

The system improves itself, but you can:

1. Submit improvement tasks via API
2. Review and approve meta-agent proposals
3. Add new agent types and capabilities
4. Enhance monitoring and observability

## License

MIT License - See LICENSE file for details.

---

**The system builds and improves itself. Your job is to guide it toward your goals! 🚀**