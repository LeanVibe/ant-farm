# LeanVibe Agent Hive 2.0 - Quick Start Guide

## Overview

Get your autonomous multi-agent development system up and running in 5 minutes.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker (optional, recommended)

## Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive

# Start services
docker compose up -d postgres redis

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Manual Setup

```bash
# Install PostgreSQL and Redis
# Ubuntu/Debian:
sudo apt install postgresql redis-server
# macOS:
brew install postgresql redis

# Start services
sudo systemctl start postgresql redis
# macOS:
brew services start postgresql redis

# Create database
sudo -u postgres createdb leanvibe_hive
sudo -u postgres createuser -P hive_user

# Clone and setup
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive
pip install -r requirements.txt

# Configure environment
cp .env.example .env.local
# Edit .env.local with your database credentials

# Run migrations
alembic upgrade head

# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## First Steps

### 1. Access the Dashboard

Open your browser and navigate to:
- **API Documentation**: http://localhost:8000/api/docs
- **Web Dashboard**: http://localhost:8000 (if serving static files)

### 2. Login

Default admin credentials:
- **Username**: `admin`
- **Password**: `change_me_now_123!`

**⚠️ Important**: Change the default password immediately!

### 3. Get Your API Token

Using curl:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "change_me_now_123!"}'
```

Save the `access_token` from the response.

### 4. Create Your First Agent

```bash
# Set your token
TOKEN="your_access_token_here"

# Spawn a developer agent
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "developer", "agent_name": "my-first-agent"}'
```

### 5. Create a Task

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Hello World Task",
    "description": "Create a simple hello world application",
    "type": "development",
    "priority": 5,
    "assigned_to": "my-first-agent"
  }'
```

### 6. Monitor Progress

```bash
# Check system status
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/status

# List all agents
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/agents

# List all tasks
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks
```

## CLI Tool Integration

The system supports multiple CLI tools for agent operations:

### Configure CLI Tools

Edit your `.env.local` file:
```bash
# Preferred CLI tool (opencode, claude, gemini)
PREFERRED_CLI_TOOL=opencode

# Tool availability
OPENCODE_AVAILABLE=true
CLAUDE_AVAILABLE=true
GEMINI_AVAILABLE=false

# API keys (if needed)
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

### Available Commands

```bash
# Check which tools are available
make tools

# Use specific CLI tool
export PREFERRED_CLI_TOOL=opencode
python src/agents/runner.py

# Force tool detection
make setup
```

## WebSocket Real-time Updates

Connect to the WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/events');

ws.onopen = () => {
  console.log('Connected to Agent Hive');
  
  // Subscribe to specific events
  ws.send(JSON.stringify({
    type: 'subscribe',
    payload: { event_types: ['agent-update', 'task-update'] }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event received:', data.type, data.payload);
  
  switch(data.type) {
    case 'agent-update':
      console.log('Agent status changed:', data.payload);
      break;
    case 'task-update':
      console.log('Task status changed:', data.payload);
      break;
    case 'system-status':
      console.log('System metrics:', data.payload);
      break;
  }
};
```

## Progressive Web App (PWA)

The dashboard can be installed as a Progressive Web App:

1. Visit the dashboard in Chrome/Edge
2. Click the install icon in the address bar
3. Or go to Chrome menu → "Install app"

### PWA Features

- **Offline functionality**: Dashboard works without internet
- **Push notifications**: Real-time alerts
- **Home screen icon**: Native app experience
- **Background sync**: Automatic updates when online

## Development Workflow

### 1. Create a Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/new-agent-type

# Make your changes
# ...

# Commit changes
git add .
git commit -m "Add new agent type support"
```

### 2. Use the Self-Modification System

```bash
# Create a modification proposal
curl -X POST http://localhost:8000/api/v1/modifications/proposal \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Add new agent capability",
    "description": "Implement specialized QA agent",
    "modification_type": "feature",
    "file_paths": ["src/agents/qa_agent.py"],
    "agent_id": "meta-agent"
  }'

# Validate the proposal
curl -X POST http://localhost:8000/api/v1/modifications/{proposal_id}/validate \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Monitor System Health

```bash
# Get comprehensive diagnostics
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/diagnostics

# Check performance metrics
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/performance

# Trigger system optimization
curl -X POST http://localhost:8000/api/v1/performance/optimize \
  -H "Authorization: Bearer $TOKEN"
```

## Configuration

### Environment Variables

Key configuration options in `.env.local`:

```bash
# Database
DATABASE_URL="postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive"

# Redis
REDIS_URL="redis://localhost:6381"

# Security
JWT_SECRET_KEY="your-secret-key-here"
JWT_EXPIRE_MINUTES=30

# System
MAX_CONCURRENT_AGENTS=10
LOG_LEVEL="INFO"

# CLI Tools
PREFERRED_CLI_TOOL="opencode"
OPENCODE_AVAILABLE=true
CLAUDE_AVAILABLE=true

# Optional API Keys
ANTHROPIC_API_KEY="your-key"
OPENAI_API_KEY="your-key"
```

### Agent Configuration

Customize agent behavior in `src/core/config.py`:

```python
class AgentConfig:
    max_task_retry_attempts = 3
    heartbeat_interval = 30  # seconds
    memory_consolidation_interval = 3600  # 1 hour
    context_window_size = 4000  # tokens
    
    # CLI tool preferences per agent type
    agent_tool_preferences = {
        "meta": "opencode",
        "developer": "claude", 
        "qa": "gemini"
    }
```

## Testing

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov --cov-report=term-missing

# Specific test
pytest tests/unit/test_task_queue.py::test_add_task
```

### Test with Real CLI Tools

```bash
# Test agent runner
python tests/test_agent_runner.py

# Test API endpoints
python tests/test_api_integration.py

# Test WebSocket connection
python tests/test_websocket.py
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```
sqlalchemy.exc.OperationalError: could not connect to server
```
**Solution**: Ensure PostgreSQL is running and credentials are correct.

#### 2. Redis Connection Error
```
redis.exceptions.ConnectionError: Error connecting to Redis
```
**Solution**: Start Redis service or check the Redis URL.

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

#### 4. Permission Denied
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Check file permissions or run with appropriate user.

#### 5. Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change port in configuration or kill existing process.

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn src.api.main:app --reload --log-level debug
```

### Check System Health

```bash
# System diagnostics
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/diagnostics

# Health check
curl http://localhost:8000/health

# Check specific agent
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/agents/{agent_name}/health
```

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/api/docs
2. **Read the full documentation**: See `docs/API.md`
3. **Customize agents**: Modify `src/agents/` 
4. **Add new capabilities**: Extend the system architecture
5. **Deploy to production**: Follow the deployment guide

## Support

- **Issues**: https://github.com/leanvibe/agent-hive/issues
- **Discussions**: https://github.com/leanvibe/agent-hive/discussions  
- **Email**: support@leanvibe.ai
- **Documentation**: http://localhost:8000/api/docs

---

**Happy coding with your autonomous agent hive! 🤖🚀**