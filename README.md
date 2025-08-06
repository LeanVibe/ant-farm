# LeanVibe Agent Hive 2.0

A **fully autonomous** multi-agent development system that builds and improves itself continuously using multiple CLI agentic coding tools.

## ✨ Core Capabilities

🤖 **Autonomous Development**: Continues development 24/7 without human intervention  
🔄 **Self-Improvement**: Analyzes and enhances its own code automatically  
🎯 **Intelligent Coordination**: Smart task assignment based on agent capabilities  
🛡️ **Safe Modifications**: Automatic testing and rollback for all changes  
📊 **Performance Learning**: Improves efficiency through experience  

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
python bootstrap.py tools

# Start Docker services
docker-compose up -d

# Bootstrap the autonomous system
python bootstrap.py bootstrap

# Or start autonomous mode directly
python autonomous_bootstrap.py autonomous

# Monitor system status
python autonomous_bootstrap.py status
# API: http://localhost:8000/api/v1/status
# Dashboard: http://localhost:8000/api/docs
```

## 🏗️ Architecture

### Autonomous Agent System
- **Meta-Agent**: Self-improvement coordinator with system analysis and code modification
- **Developer Agent**: Code implementation, proactive development, and refactoring
- **QA Agent**: Testing, validation, and quality assurance  
- **Architect Agent**: System design, architecture decisions, and planning
- **Research Agent**: Knowledge acquisition, trend analysis, and improvement research

### Multi-CLI Support
- **Tool Detection**: Auto-discovers opencode, Claude CLI, Gemini CLI
- **Priority Fallback**: opencode → Claude → Gemini → API  
- **Unified Interface**: Same agent code works with any tool
- **Smart Switching**: Automatic failover if one tool fails

### Core Infrastructure

1. **Task Coordinator** - Intelligent task assignment with capability-based matching
2. **Self-Bootstrapper** - Autonomous development continuation system
3. **Agent Orchestrator** - Agent lifecycle management and spawning
4. **Task Queue** - Redis-based priority task distribution with dependencies
5. **Message Broker** - Inter-agent communication via pub/sub
6. **Context Engine** - Vector-based semantic memory with pgvector
7. **API Server** - RESTful interface for system control and monitoring

## 🔄 Self-Improvement Capabilities

### Development Phases
1. **Analysis** - System health assessment and bottleneck identification
2. **Planning** - Goal creation and development roadmap updates  
3. **Implementation** - Autonomous code changes with testing
4. **Testing** - Automated validation and regression testing
5. **Deployment** - Safe rollout with rollback capabilities
6. **Monitoring** - Continuous performance tracking

### System Capabilities
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
├── autonomous_bootstrap.py # Autonomous operation starter
├── src/
│   ├── core/              # Core system components
│   │   ├── orchestrator.py      # Agent lifecycle management  
│   │   ├── task_queue.py        # Priority task distribution
│   │   ├── task_coordinator.py  # Intelligent task assignment
│   │   ├── message_broker.py    # Inter-agent communication
│   │   ├── context_engine.py    # Semantic memory system
│   │   ├── self_bootstrap.py    # Autonomous development system
│   │   ├── config.py           # Multi-CLI configuration
│   │   └── models.py           # Database models
│   ├── agents/            # Agent implementations
│   │   ├── base_agent.py       # Multi-CLI base agent
│   │   ├── meta_agent.py       # Self-improvement coordinator
│   │   └── runner.py           # Agent process management
│   └── api/               # FastAPI server
│       └── main.py            # REST API endpoints
├── tests/                 # Comprehensive test suite
├── docs/                  # Architecture and API documentation
├── docker-compose.yaml    # Infrastructure services
├── AGENTS.md             # CLI tool configuration context
└── requirements.txt      # Python dependencies
```

## 📊 Key Features

### Autonomous Operation
- **24/7 Development**: Continuous improvement without human intervention
- **Self-Bootstrapping**: Extends its own capabilities autonomously  
- **Intelligent Coordination**: Multi-agent task optimization
- **Performance Learning**: Adapts based on task success patterns

### Safety & Reliability  
- **Safe Modifications**: Automatic backup before any code changes
- **Rollback System**: Instant recovery from failed improvements
- **Testing Integration**: All changes validated automatically
- **Health Monitoring**: Continuous system status assessment

### Multi-CLI Integration
- **Tool Flexibility**: Works with any available CLI agentic coding tool
- **Smart Fallback**: Graceful degradation across tool failures
- **Performance Optimization**: Learns which tools work best for specific tasks
- **Unified Experience**: Same capabilities regardless of underlying tool

## 🎯 Current Status: Phase 2 Complete ✅

- ✅ **Phase 1: Core Infrastructure** - Task queue, message broker, context engine
- ✅ **Phase 2: Agent System** - Meta-agent, specialized agents, autonomous development
- 🚧 **Phase 3: Advanced Features** - Machine learning integration, performance optimization
- 📋 **Phase 4: Scale & Polish** - Multi-node deployment, advanced monitoring

### Immediate Capabilities
- Autonomous system analysis and improvement proposal generation
- Safe self-modification with backup/rollback mechanisms  
- Intelligent task assignment based on agent capabilities and performance
- Multi-agent coordination for complex development tasks
- Continuous monitoring and optimization of system performance
- Real-time API for system control and status monitoring

## 🛠️ Technology Stack

- **Backend**: FastAPI (async Python) with multi-CLI integration
- **Database**: PostgreSQL 15+ with pgvector for semantic search
- **Cache/Queue**: Redis 7.0+ with task coordination and message broker
- **CLI Tools**: opencode, Claude CLI, Gemini CLI with smart fallback
- **Testing**: pytest with automated validation for all changes
- **Deployment**: Docker Compose with tmux session management
- **Monitoring**: Real-time metrics and health assessment

## 🎯 Success Metrics

- **Autonomous Operation**: >95% uptime without human intervention
- **Task Success Rate**: >85% successful task completion
- **Self-Improvement**: >3 autonomous improvements per week
- **Response Time**: <100ms for task assignment decisions
- **Tool Reliability**: <5% fallback rate to secondary CLI tools
- **Safety**: 0 system crashes from self-modifications

## 📚 Usage Examples

### Basic Operation
```bash
# Start autonomous system
python autonomous_bootstrap.py autonomous

# Check system status
curl http://localhost:8000/api/v1/status

# View agents
curl http://localhost:8000/api/v1/agents

# Create a task
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "Optimize performance", "description": "Improve system performance", "type": "optimization"}'
```

### Monitoring & Control
```bash
# System health check
python autonomous_bootstrap.py status

# View tmux sessions
tmux ls

# Attach to meta-agent
tmux attach -t meta-agent-001

# View agent logs
docker-compose logs -f
```

## 📚 Documentation

- [AGENTS.md](AGENTS.md) - CLI tool integration and configuration
- [CLAUDE.md](CLAUDE.md) - Claude-specific setup and context
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step setup guide
- [API Documentation](http://localhost:8000/api/docs) - Interactive API explorer (when running)

## 🔮 Roadmap

### Phase 3: Advanced Intelligence (Next)
- Machine learning integration for pattern recognition
- Advanced performance optimization algorithms  
- Predictive task scheduling based on resource patterns
- Enhanced agent specialization and skill development

### Phase 4: Scale & Production
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

This system develops itself autonomously! However, you can:

1. **Submit Issues**: Report bugs or suggest features via GitHub issues
2. **CLI Tool Support**: Help add support for new agentic coding tools
3. **Documentation**: Improve setup guides and architectural documentation
4. **Testing**: Contribute test cases and validation scenarios

The Meta-Agent will analyze contributions and integrate improvements automatically.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details