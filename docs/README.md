# LeanVibe Agent Hive 2.0 - Documentation Hub

## 🚀 Quick Start Guides

- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment
- **[../AGENTS.md](../AGENTS.md)** - CLI commands and development workflow

## 📊 System Status & Health

- **[../README.md](../README.md)** - Main project overview and current status
- **[../TECHNICAL_DEBT_CONSOLIDATED.md](../TECHNICAL_DEBT_CONSOLIDATED.md)** - Comprehensive technical debt assessment
- **[../SYSTEM_READINESS_ASSESSMENT.md](../SYSTEM_READINESS_ASSESSMENT.md)** - System validation report
- **[API.md](API.md)** - Complete API reference

## 📋 Development Resources

- **[PLAN.md](PLAN.md)** - Development roadmap and milestones
- **[system-architecture.md](system-architecture.md)** - Technical architecture
- **[ADW_IMPLEMENTATION_SUMMARY.md](ADW_IMPLEMENTATION_SUMMARY.md)** - Autonomous Development Workflow
- **[../ENHANCED_COMMUNICATION_SUMMARY.md](../ENHANCED_COMMUNICATION_SUMMARY.md)** - Inter-agent communication system
- **[PROMPT.md](PROMPT.md)** - AI agent system context

## 🧪 Quality Assurance & Testing

- **[testing-strategy-user-profile-crud.md](testing-strategy-user-profile-crud.md)** - Comprehensive testing strategy for user profile operations
- **[security-test-scenarios-user-profile-crud.md](security-test-scenarios-user-profile-crud.md)** - Security testing framework and scenarios
- **[performance-edge-case-testing-user-profile-crud.md](performance-edge-case-testing-user-profile-crud.md)** - Performance testing and edge case validation
- **[qa-analysis-user-profile-crud-summary.md](qa-analysis-user-profile-crud-summary.md)** - QA analysis summary with recommendations

## 📄 Historical Design Documents

*Maintained for reference - these PRDs guided the system design:*

### Core System Design
- **[product-requirements.md](product-requirements.md)** - Master requirements
- **[agent-orchestrator-prd.md](agent-orchestrator-prd.md)** - Agent lifecycle
- **[PRD-context-engine.md](PRD-context-engine.md)** - Memory system

### Feature Specifications  
- **[PRD-mobile-pwa-dashboard.md](PRD-mobile-pwa-dashboard.md)** - Dashboard PWA
- **[PRD-sleep-wake-manager.md](PRD-sleep-wake-manager.md)** - Session persistence
- **[self-modification-engine-prd.md](self-modification-engine-prd.md)** - Self-improvement
- **[prompt-optimization-system-prd.md](prompt-optimization-system-prd.md)** - Prompt optimization

### Integration Specifications
- **[github-integration-prd.md](github-integration-prd.md)** - Git workflow
- **[communication-prd.md](communication-prd.md)** - Inter-agent messaging  
- **[observability-prd.md](observability-prd.md)** - System monitoring

## 🎯 Current System Metrics

| Metric | Status | Value |
|--------|--------|-------|
| **System Readiness** | ✅ | 100% (22/22 checks) |
| **API Security** | ✅ | 87% endpoints protected |
| **Technical Debt** | ✅ | Very Low (manageable) |
| **Test Coverage** | ✅ | 90%+ core components |
| **QA Testing Strategy** | ✅ | Complete CRUD validation framework |
| **Security Testing** | ✅ | Comprehensive security test scenarios |
| **Performance Testing** | ✅ | Load testing with defined SLAs |
| **Communication System** | ✅ | Enhanced multi-agent coordination |
| **Autonomous Operation** | ✅ | 16-24 hour sessions |

## 📱 Essential Commands

```bash
# System lifecycle
hive init && hive system start
hive system status

# Agent management
hive agent spawn meta
hive agent list

# Task management  
hive task submit "description"
hive task list

# Development
pytest tests/
ruff check . && ruff format .
```

## 🆕 Latest System Capabilities

### Enhanced Inter-Agent Communication (v2.0)
- **Real-time Collaboration**: Synchronized agent workflows with shared context
- **Intelligent Load Balancing**: 5 routing strategies for optimal message distribution  
- **Priority Message Routing**: Critical, high, normal, low priority channels
- **Shared Knowledge Base**: Vector-based context sharing across all agents
- **Performance Monitoring**: Real-time communication metrics and optimization

### Production-Ready Features
- **Emergency Intervention**: 5-level safety system with automatic rollback
- **Session Persistence**: Long-running autonomous operations (16-24 hours)
- **Security Framework**: JWT-based API protection with rate limiting
- **Observability Stack**: Comprehensive logging, metrics, and health monitoring
- **Self-Modification Engine**: Safe, tested autonomous code improvements

## 📚 Navigation Tips

- **New Users**: Start with [QUICKSTART.md](QUICKSTART.md)
- **Developers**: See [../AGENTS.md](../AGENTS.md) for workflows
- **Operations**: Check [DEPLOYMENT.md](DEPLOYMENT.md) 
- **Architecture**: Review [system-architecture.md](system-architecture.md)
- **Troubleshooting**: See [../TECHNICAL_DEBT_CONSOLIDATED.md](../TECHNICAL_DEBT_CONSOLIDATED.md)

---

**Last Updated**: August 10, 2025  
**Version**: 2.0.0  
**Status**: Production Ready - Autonomous Multi-Agent System Operational