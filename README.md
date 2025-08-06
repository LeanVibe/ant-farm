# LeanVibe Agent Hive 2.0

A self-improving, autonomous multi-agent development system that supports multiple CLI agentic coding tools.

## 🛠️ Supported CLI Tools

The system auto-detects and uses any of these CLI agentic coding tools:

1. **🥇 opencode** (Preferred) - AI coding agent built for the terminal
   - Install: `curl -fsSL https://opencode.ai/install | bash`
   - Features: Responsive TUI, multi-session support, 75+ LLM providers

2. **🥈 Claude Code CLI** - Anthropic's command-line assistant  
   - Install: https://claude.ai/cli
   - Features: Direct Claude Pro/Max access

3. **🥉 Gemini CLI** - Google's AI coding assistant
   - Install: https://ai.google.dev/gemini-api/docs/cli
   - Features: Google AI integration

**Smart Fallback**: Automatically switches between tools, with API fallback

## 🚀 Quick Start

```bash
# Prerequisites: Docker + at least one CLI tool
docker --version  # Docker 20.10+

# Check available CLI tools  
make tools

# Setup (auto-detects available tools)
cd leanvibe-hive
make setup && make quick

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:3000  
# pgAdmin: http://localhost:5050
```

## 🏗️ Architecture

### Multi-CLI Support
- **Tool Detection**: Auto-discovers opencode, Claude CLI, Gemini CLI
- **Priority Fallback**: opencode → Claude → Gemini → API  
- **Unified Interface**: Same agent code works with any tool
- **Smart Switching**: Automatic failover if one tool fails

### Core Components

1. **Bootstrap Agent** - Initial agent that creates the system
2. **Agent Orchestrator** - Central coordination and lifecycle management
3. **Task Queue** - Redis-based priority task distribution
4. **Message Broker** - Inter-agent communication via pub/sub
5. **Context Engine** - Vector-based semantic memory with pgvector
6. **Self-Modifier** - Safe code generation and improvement system

### Agent Types

- **Meta-Agent**: System self-improvement coordinator
- **Architect Agent**: System design and technical decisions
- **Developer Agents**: Specialized for backend/frontend/devops
- **QA Agent**: Testing and quality assurance
- **Product Manager Agent**: Task planning and coordination

## 📁 Project Structure

```
leanvibe-hive/
├── bootstrap/           # System bootstrap scripts
│   └── init_agent.py   # First agent that builds everything
├── src/
│   ├── core/           # Core system components
│   │   ├── orchestrator.py
│   │   ├── task_queue.py
│   │   ├── message_broker.py
│   │   ├── context_engine.py
│   │   └── models.py
│   ├── agents/         # Agent implementations
│   │   ├── base_agent.py
│   │   ├── meta_agent.py
│   │   └── specialized/
│   ├── api/            # FastAPI endpoints
│   └── web/            # LitPWA dashboard
├── tests/              # Test suite (TDD approach)
├── .claude/            # Claude Code configuration
│   └── CLAUDE.md       # System context for Claude
└── pyproject.toml      # UV dependencies
```

## 🔄 Self-Improvement Loop

1. **Analyze**: Meta-agent analyzes system performance
2. **Propose**: Generate improvement suggestions
3. **Test**: Validate changes in sandboxed environment
4. **Apply**: Deploy improvements with rollback capability
5. **Learn**: Update prompts and patterns based on outcomes

## 🛠️ Technology Stack

- **Backend**: FastAPI (async Python)
- **Database**: PostgreSQL 15+ with pgvector
- **Cache/Queue**: Redis 7.0+
- **Frontend**: LitPWA (Progressive Web App)
- **LLM**: Claude 3.5 Sonnet/Haiku
- **Testing**: pytest with >90% coverage
- **Deployment**: Docker Compose

## 📊 Key Features

- **24/7 Autonomous Operation**: Agents work continuously with sleep-wake cycles
- **Self-Building**: System uses Claude Code to develop itself
- **Context Persistence**: Semantic memory across sessions
- **Real-time Monitoring**: WebSocket-based observability
- **Safe Modifications**: Sandboxed testing with Git-based rollback
- **Prompt Optimization**: A/B testing for continuous improvement

## 🎯 Success Metrics

- System uptime: >99.9%
- Task completion rate: >85%
- Self-improvement frequency: >5 iterations/week
- Token efficiency: 60-80% reduction via context management
- Agent collaboration: <100ms message latency

## 📝 Development Principles

1. **Simplicity First** - Every component should be understandable by an agent
2. **Test-Driven** - Write tests before implementation
3. **Self-Documenting** - Code generates its own documentation
4. **Fail-Safe** - All operations must be reversible
5. **Incremental** - Small, working releases daily

## 🚦 Current Status

- [ ] Phase 0: Bootstrap Core (Days 1-3)
- [ ] Phase 1: Core Infrastructure (Days 4-7)
- [ ] Phase 2: Self-Improvement Loop (Week 2)
- [ ] Phase 3: Intelligence Layer (Week 3)
- [ ] Phase 4: Observability & UI (Week 4)

## 📚 Documentation

- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Detailed implementation guide
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture details
- [API.md](docs/API.md) - API endpoint documentation
- [AGENTS.md](docs/AGENTS.md) - Agent specifications

## 🤝 Contributing

This system builds itself! Contributions happen through the Meta-Agent's self-improvement process.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details