# LeanVibe Agent Hive 2.0

A multi-agent development system designed to build and improve itself continuously using multiple CLI agentic coding tools.

## ✨ Planned Capabilities

🤖 **Autonomous Development**: (Planned) Continues development 24/7 without human intervention  
🔄 **Self-Improvement**: (In Development) Analyzes and enhances its own code automatically  
🎯 **Intelligent Coordination**: (Planned) Smart task assignment based on agent capabilities  
🛡️ **Safe Modifications**: (Planned) Automatic testing and rollback for all changes  
📊 **Performance Learning**: (Planned) Improves efficiency through experience  

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
hive tools

# Start Docker services
docker compose up -d

# Initialize database
hive init-db

# Bootstrap the system (creates initial structure)
hive bootstrap

# Start individual agents
hive run-agent meta
hive run-agent developer

# Start API server
hive start-api

# Monitor system status
hive status
# API: http://localhost:8000/api/v1/status (when implemented)
# Dashboard: http://localhost:8000/api/docs (when implemented)
```

## 🏗️ Architecture

### Agent System (In Development)
- **Meta-Agent**: (Placeholder) Self-improvement coordinator with system analysis and code modification
- **Developer Agent**: (Basic Implementation) Code implementation, proactive development, and refactoring
- **QA Agent**: (Placeholder) Testing, validation, and quality assurance  
- **Architect Agent**: (Placeholder) System design, architecture decisions, and planning
- **Research Agent**: (Placeholder) Knowledge acquisition, trend analysis, and improvement research

### Multi-CLI Support (Working)
- **Tool Detection**: Auto-discovers opencode, Claude CLI, Gemini CLI
- **Priority Fallback**: opencode → Claude → Gemini → API  
- **Unified Interface**: (Planned) Same agent code works with any tool
- **Smart Switching**: (Planned) Automatic failover if one tool fails

### Core Infrastructure (Mixed Status)

1. **Task Coordinator** - (Placeholder) Intelligent task assignment with capability-based matching
2. **Self-Bootstrapper** - (Placeholder) Autonomous development continuation system
3. **Agent Orchestrator** - (Placeholder) Agent lifecycle management and spawning
4. **Task Queue** - (Placeholder) Redis-based priority task distribution with dependencies
5. **Message Broker** - (Placeholder) Inter-agent communication via pub/sub
6. **Context Engine** - (Placeholder) Vector-based semantic memory with pgvector
7. **API Server** - (Basic Implementation) RESTful interface for system control and monitoring

## 🔄 Self-Improvement Capabilities (Planned)

### Development Phases
1. **Analysis** - System health assessment and bottleneck identification
2. **Planning** - Goal creation and development roadmap updates  
3. **Implementation** - Autonomous code changes with testing
4. **Testing** - Automated validation and regression testing
5. **Deployment** - Safe rollout with rollback capabilities
6. **Monitoring** - Continuous performance tracking

### Target System Capabilities
- **Self-Modification**: Safe code editing with backup/rollback (Target: 90%)
- **Autonomous Learning**: Experience-based performance improvement (Target: 80%)
- **System Monitoring**: Comprehensive health and performance tracking (Target: 90%)
- **Task Optimization**: Intelligent scheduling and agent coordination (Target: 80%)
- **Code Quality**: Automated standards enforcement and refactoring (Target: 90%)
- **Security Hardening**: Continuous vulnerability assessment and patching (Target: 80%)

## 📁 Project Structure

```
ant-farm/
├── bootstrap.py            # Enhanced bootstrap system with multi-CLI support
├── autonomous_bootstrap.py # Autonomous operation starter (placeholder)
├── src/
│   ├── cli/               # Unified CLI interface (NEW)
│   │   └── main.py            # Single hive command entry point
│   ├── core/              # Core system components (placeholders)
│   │   ├── orchestrator.py      # Agent lifecycle management  
│   │   ├── task_queue.py        # Priority task distribution
│   │   ├── task_coordinator.py  # Intelligent task assignment
│   │   ├── message_broker.py    # Inter-agent communication
│   │   ├── context_engine.py    # Semantic memory system
│   │   ├── self_bootstrap.py    # Autonomous development system
│   │   ├── config.py           # Multi-CLI configuration
│   │   └── models.py           # Database models
│   ├── agents/            # Agent implementations (basic)
│   │   ├── base_agent.py       # Multi-CLI base agent
│   │   ├── meta_agent.py       # Self-improvement coordinator
│   │   └── runner.py           # Agent process management
│   └── api/               # FastAPI server (basic)
│       └── main.py            # REST API endpoints
├── tests/                 # Test suite (to be created)
├── docs/                  # Architecture and planning documentation
├── docker-compose.yaml    # Infrastructure services
├── AGENTS.md             # CLI tool configuration context
└── pyproject.toml        # Python dependencies and configuration
```

## 📊 Key Features

### Planned Autonomous Operation
- **24/7 Development**: (Planned) Continuous improvement without human intervention
- **Self-Bootstrapping**: (Planned) Extends its own capabilities autonomously  
- **Intelligent Coordination**: (Planned) Multi-agent task optimization
- **Performance Learning**: (Planned) Adapts based on task success patterns

### Planned Safety & Reliability  
- **Safe Modifications**: (Planned) Automatic backup before any code changes
- **Rollback System**: (Planned) Instant recovery from failed improvements
- **Testing Integration**: (Planned) All changes validated automatically
- **Health Monitoring**: (Planned) Continuous system status assessment

### Multi-CLI Integration (Working)
- **Tool Flexibility**: Works with any available CLI agentic coding tool
- **Smart Fallback**: (Planned) Graceful degradation across tool failures
- **Performance Optimization**: (Planned) Learns which tools work best for specific tasks
- **Unified Experience**: (Planned) Same capabilities regardless of underlying tool

## 🎯 Current Status: Phase 0 - Foundational Scaffolding 🚧

**Reality Check**: The project is in early development with foundational infrastructure being built.

### Completed ✅
- **CLI Unification**: Single `hive` command replaces multiple startup scripts
- **Multi-CLI Detection**: Auto-discovery of opencode, Claude CLI, Gemini CLI
- **Project Structure**: Basic directory layout and configuration
- **Docker Setup**: PostgreSQL and Redis service configuration

### In Progress 🔄
- **Core Infrastructure**: Task queue, message broker, context engine (placeholders exist)
- **Agent System**: Base agent implementations (basic structure)
- **Database Models**: SQLAlchemy models (incomplete)

### Planned 📋
- **Phase 1: Core Infrastructure** - Task queue, message broker, context engine
- **Phase 2: Agent System** - Meta-agent, specialized agents, autonomous development  
- **Phase 3: Advanced Intelligence** - Self-modification, memory consolidation, predictive optimization
- **Phase 4: Scale & Production** - Multi-node deployment, enterprise features

## 🛠️ Technology Stack

- **Backend**: FastAPI (async Python) with multi-CLI integration
- **Database**: PostgreSQL 15+ with pgvector for semantic search
- **Cache/Queue**: Redis 7.0+ with task coordination and message broker
- **CLI Tools**: opencode, Claude CLI, Gemini CLI with smart fallback
- **Testing**: pytest with automated validation for all changes
- **Deployment**: Docker Compose with tmux session management
- **Monitoring**: Real-time metrics and health assessment

## 🎯 Target Success Metrics (Future Goals)

- **Autonomous Operation**: >95% uptime without human intervention
- **Task Success Rate**: >85% successful task completion
- **Self-Improvement**: >3 autonomous improvements per week
- **Response Time**: <100ms for task assignment decisions
- **Tool Reliability**: <5% fallback rate to secondary CLI tools
- **Safety**: 0 system crashes from self-modifications

## 📚 Usage Examples

### Basic Operation (Current)
```bash
# Check available CLI tools
hive tools

# Start system components
hive init-db
hive start-api

# Run individual agents
hive run-agent meta
hive run-agent developer

# Check system status
hive status
```

### Planned Advanced Operation (Future)
```bash
# Start advanced autonomous system with Phase 3 capabilities
hive autonomous

# Check comprehensive system status
hive status --detailed

# Monitor memory consolidation and patterns
hive memory

# Check performance optimization status
hive performance

# View sleep-wake cycle status
hive sleep
```

### API Monitoring (Planned)
```bash
# System health with basic metrics (when implemented)
curl http://localhost:8000/api/v1/status

# Agent information (when implemented)
curl http://localhost:8000/api/v1/agents

# Task coordination (when implemented)
curl http://localhost:8000/api/v1/tasks

# Performance metrics (when implemented)
curl http://localhost:8000/api/v1/metrics

# Trigger system analysis (when implemented)
curl -X POST http://localhost:8000/api/v1/system/analyze
```

### Advanced Operations (Planned)
```bash
# Create complex task with dependencies (when implemented)
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Implement machine learning optimization", 
    "description": "Add ML-based task scheduling",
    "type": "development",
    "priority": "HIGH",
    "dependencies": ["system_analysis_001"],
    "metadata": {"complexity": "high", "estimated_hours": 8}
  }'

# Force memory consolidation (when implemented)
curl -X POST http://localhost:8000/api/v1/system/consolidate

# Trigger performance optimization (when implemented)
curl -X POST http://localhost:8000/api/v1/system/optimize
```

## 📚 Documentation

- [AGENTS.md](AGENTS.md) - CLI tool integration and configuration
- [CLAUDE.md](CLAUDE.md) - Claude-specific setup and context
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step setup guide
- [API Documentation](http://localhost:8000/api/docs) - Interactive API explorer (when running)

## 🔮 Roadmap

### Phase 1: Core Infrastructure (Current)
- Implement task queue with Redis backend
- Create message broker for inter-agent communication
- Build context engine with pgvector semantic search
- Complete database models and migrations

### Phase 2: Agent System (Next)
- Complete agent orchestrator implementation
- Build specialized agent implementations  
- Implement autonomous development capabilities
- Add agent communication and coordination

### Phase 3: Advanced Intelligence (Future)
- Machine learning integration for pattern recognition
- Advanced performance optimization algorithms  
- Predictive task scheduling based on resource patterns
- Enhanced agent specialization and skill development

### Phase 4: Scale & Production (Future)
- Multi-node deployment with load balancing
- Advanced monitoring and alerting systems
- Production-grade security and access controls
- Enterprise integration capabilities

## 📝 Development Principles

1. **Autonomous First** - Every component designed for self-operation
2. **Safety by Design** - All operations must be reversible and validated
3. **Multi-Tool Support** - CLI tool agnostic architecture  
4. **Self-Documenting** - Code generates its own documentation
5. **Continuous Learning** - System improves from every interaction

## 🤝 Contributing

This system is designed to develop itself autonomously (when completed)! Currently in early development, you can:

1. **Submit Issues**: Report bugs or suggest features via GitHub issues
2. **Core Implementation**: Help implement task queue, message broker, and context engine
3. **CLI Tool Support**: Help add support for new agentic coding tools
4. **Documentation**: Improve setup guides and architectural documentation
5. **Testing**: Contribute test cases and validation scenarios

Once the Meta-Agent is fully implemented, it will analyze contributions and integrate improvements automatically.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details