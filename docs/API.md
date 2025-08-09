# LeanVibe Agent Hive 2.0 API Documentation

## Overview

The LeanVibe Agent Hive 2.0 API provides comprehensive control over an autonomous multi-agent development system. This RESTful API enables you to manage agents, tasks, system metrics, and real-time communications in a self-improving AI development environment.

## Base URL

- **Development**: `http://localhost:9001`
- **Production**: `https://hive.leanvibe.ai`

## Authentication

The API uses JWT (JSON Web Token) authentication with Bearer tokens. Most endpoints require authentication.

### Getting Started

1. **Login** to obtain tokens:
   ```bash
   curl -X POST /api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "your_password"}'
   ```

2. **Use the access token** in subsequent requests:
   ```bash
   curl -H "Authorization: Bearer <your_access_token>" /api/v1/agents
   ```

3. **Refresh tokens** when they expire:
   ```bash
   curl -X POST /api/v1/auth/refresh \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "<your_refresh_token>"}'
   ```

### Token Lifecycle

- **Access Token**: Expires in 30 minutes
- **Refresh Token**: Expires in 7 days
- **Auto-refresh**: Use refresh tokens to get new access tokens

## Rate Limiting

API requests are rate-limited per IP address:

| Endpoint Category | Limit |
|------------------|-------|
| Authentication | 10 requests/minute |
| General API | 100 requests/minute |
| WebSocket Connections | 50 connections/minute |

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets

## Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": 1640995200.0,
  "request_id": "req_123456"
}
```

### Error Responses

```json
{
  "success": false,
  "data": null,
  "error": "Error description",
  "timestamp": 1640995200.0,
  "request_id": "req_123456"
}
```

## HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## Permissions System

The API uses role-based access control with granular permissions:

### System Permissions
- `system:read` - View system status
- `system:write` - Modify system settings
- `system:admin` - Full system administration

### Agent Permissions
- `agent:read` - View agent information
- `agent:write` - Modify agent settings
- `agent:spawn` - Create new agents
- `agent:terminate` - Stop agents

### Task Permissions
- `task:read` - View tasks
- `task:write` - Modify tasks
- `task:create` - Create new tasks
- `task:cancel` - Cancel tasks

### Additional Permissions
- `message:read`, `message:send`, `message:broadcast`
- `metrics:read`, `metrics:write`
- `context:read`, `context:write`
- `modification:read`, `modification:propose`, `modification:approve`

## WebSocket API

Connect to `/api/v1/ws/events` for real-time updates.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:9001/api/v1/ws/events');

ws.onopen = () => {
  console.log('Connected to agent hive');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data.type, data.payload);
};
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `system-status` | System health and metrics updates |
| `agent-update` | Agent lifecycle changes |
| `task-update` | Task status changes |
| `message` | Inter-agent communications |
| `metrics-update` | Performance metrics |
| `notification` | System notifications |

### Subscribing to Specific Events

```javascript
// Subscribe to specific event types
ws.send(JSON.stringify({
  type: 'subscribe',
  payload: {
    event_types: ['agent-update', 'task-update']
  }
}));

// Request current status
ws.send(JSON.stringify({
  type: 'request-status'
}));
```

## API Endpoints

### Authentication

#### POST /api/v1/auth/login
Authenticate user and receive JWT tokens.

**Request Body:**
```json
{
  "username": "admin",
  "password": "your_password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "jwt_token_here",
    "refresh_token": "refresh_token_here",
    "token_type": "bearer",
    "expires_in": 1800,
    "user": {
      "id": "user_id",
      "username": "admin",
      "email": "admin@example.com",
      "is_admin": true,
      "permissions": ["system:admin"]
    }
  }
}
```

#### POST /api/v1/auth/refresh
Refresh access token using refresh token.

#### GET /api/v1/auth/me
Get current user information (requires authentication).

#### POST /api/v1/auth/logout
Logout current user.

### System Management

#### GET /health
Basic health check (no authentication required).

#### GET /api/v1/status
Comprehensive system status (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "health_score": 0.95,
    "active_agents": 3,
    "total_tasks": 150,
    "completed_tasks": 142,
    "failed_tasks": 3,
    "queue_depth": 5,
    "uptime": 86400.0
  }
}
```

#### POST /api/v1/system/analyze
Trigger system analysis by meta-agent (requires `system:write`).

#### POST /api/v1/system/shutdown
Gracefully shutdown system (requires admin privileges).

#### GET /api/v1/emergency/status
Get emergency intervention system status (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "level": "NORMAL",
    "is_paused": false,
    "session_id": "session_123",
    "last_intervention": null,
    "active_monitoring": true
  }
}
```

#### POST /api/v1/emergency/pause
Pause autonomous development session (requires `system:write`).

#### POST /api/v1/emergency/resume
Resume paused autonomous development session (requires `system:write`).

### Agent Management

#### GET /api/v1/agents
List all agents (requires `agent:read`).

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "meta-agent-001",
      "type": "meta",
      "role": "system_coordinator",
      "status": "active",
      "capabilities": {
        "analysis": true,
        "optimization": true
      },
      "last_heartbeat": 1640995200.0,
      "tasks_completed": 25,
      "tasks_failed": 1,
      "uptime": 3600.0
    }
  ]
}
```

#### GET /api/v1/agents/{agent_name}
Get specific agent information.

#### POST /api/v1/agents
Spawn a new agent (requires `agent:spawn`).

**Request Body:**
```json
{
  "agent_type": "developer",
  "agent_name": "dev-agent-002"
}
```

#### POST /api/v1/agents/{agent_name}/stop
Stop a specific agent (requires `agent:terminate`).

#### POST /api/v1/agents/{agent_name}/health
Check agent health.

### Task Management

#### GET /api/v1/tasks
List tasks with optional filtering (requires `task:read`).

**Query Parameters:**
- `status`: Filter by task status
- `assigned_to`: Filter by assigned agent

#### POST /api/v1/tasks
Create a new task (requires `task:create`).

**Request Body:**
```json
{
  "title": "Implement user authentication",
  "description": "Add JWT-based authentication to the API",
  "type": "development",
  "priority": 5,
  "assigned_to": "dev-agent-001",
  "dependencies": [],
  "metadata": {
    "estimated_hours": 4,
    "complexity": "medium"
  }
}
```

#### GET /api/v1/tasks/{task_id}
Get specific task information.

#### POST /api/v1/tasks/{task_id}/cancel
Cancel a task (requires `task:cancel`).

### Messaging

#### POST /api/v1/messages
Send a message to an agent (requires `message:send`).

#### POST /api/v1/broadcast
Broadcast a message to all agents (requires `message:broadcast`).

### Metrics and Monitoring

#### GET /api/v1/metrics
Get system metrics (requires `metrics:read`).

#### GET /api/v1/performance
Get performance optimization statistics.

#### POST /api/v1/performance/optimize
Trigger performance optimization.

### Context and Memory

#### GET /api/v1/context/{agent_id}
Get context/memory for a specific agent.

#### POST /api/v1/context/{agent_id}/consolidate
Trigger memory consolidation for an agent.

### Self-Modification

#### GET /api/v1/modifications
Get self-modification statistics.

#### POST /api/v1/modifications/proposal
Create a new modification proposal.

#### POST /api/v1/modifications/{proposal_id}/validate
Validate a modification proposal.

### Git Workflows

#### GET /api/v1/workflows
List active development workflows.

#### POST /api/v1/workflows
Create a new development workflow.

#### POST /api/v1/workflows/{workflow_id}/complete
Complete a development workflow.

### Diagnostics

#### GET /api/v1/diagnostics
Get comprehensive system diagnostics.

## Code Examples

### Python (using requests)

```python
import requests

# Login
response = requests.post('http://localhost:9001/api/v1/auth/login', json={
    'username': 'admin',
    'password': 'your_password'
})
tokens = response.json()['data']
access_token = tokens['access_token']

# Create headers for authenticated requests
headers = {'Authorization': f'Bearer {access_token}'}

# List agents
agents = requests.get('http://localhost:9001/api/v1/agents', headers=headers)
print(agents.json())

# Create a task
task_data = {
    'title': 'Process data analysis',
    'description': 'Analyze user behavior data',
    'type': 'analysis',
    'priority': 5
}
task = requests.post('http://localhost:9001/api/v1/tasks', 
                    json=task_data, headers=headers)
print(task.json())
```

### JavaScript (using fetch)

```javascript
// Login
const loginResponse = await fetch('/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'admin',
    password: 'your_password'
  })
});
const { data: tokens } = await loginResponse.json();

// Create headers for authenticated requests
const headers = {
  'Authorization': `Bearer ${tokens.access_token}`,
  'Content-Type': 'application/json'
};

// Get system status
const statusResponse = await fetch('/api/v1/status', { headers });
const status = await statusResponse.json();
console.log('System status:', status.data);

// Spawn an agent
const agentResponse = await fetch('/api/v1/agents', {
  method: 'POST',
  headers,
  body: JSON.stringify({
    agent_type: 'developer',
    agent_name: 'my-dev-agent'
  })
});
const agent = await agentResponse.json();
console.log('Agent created:', agent.data);
```

### curl Examples

```bash
# Login and save token
TOKEN=$(curl -s -X POST /api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your_password"}' | \
  jq -r '.data.access_token')

# Get system status
curl -H "Authorization: Bearer $TOKEN" /api/v1/status

# Create a task
curl -X POST /api/v1/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Code review",
    "description": "Review pull request #123",
    "type": "review",
    "priority": 3
  }'

# List active agents
curl -H "Authorization: Bearer $TOKEN" /api/v1/agents
```

## Security Best Practices

1. **Store tokens securely** - Never log or expose JWT tokens
2. **Use HTTPS** in production environments
3. **Implement token refresh** logic in your applications
4. **Handle rate limits** gracefully with exponential backoff
5. **Validate permissions** for each operation
6. **Monitor API usage** for unusual patterns

## Error Handling

### Common Error Scenarios

1. **Token Expiration (401)**
   ```json
   {
     "success": false,
     "error": "Token expired",
     "timestamp": 1640995200.0
   }
   ```
   Solution: Use refresh token to get new access token

2. **Rate Limit Exceeded (429)**
   ```json
   {
     "success": false,
     "error": "Rate limit exceeded: 100 requests per minute",
     "timestamp": 1640995200.0
   }
   ```
   Solution: Implement exponential backoff

3. **Insufficient Permissions (403)**
   ```json
   {
     "success": false,
     "error": "Missing required permissions: agent:write",
     "timestamp": 1640995200.0
   }
   ```
   Solution: Contact admin to grant required permissions

## Changelog

### v2.0.0 (Current)
- Added JWT authentication system
- Implemented role-based permissions
- Added WebSocket real-time events
- Enhanced security with rate limiting
- Added comprehensive API documentation
- Implemented PWA support

### v1.0.0
- Initial API release
- Basic agent and task management
- Simple authentication

## Support

- **Documentation**: [API Docs](http://localhost:9001/api/docs)
- **Interactive API**: [Swagger UI](http://localhost:9001/api/docs)
- **Alternative Docs**: [ReDoc](http://localhost:9001/api/redoc)
- **GitHub Issues**: [Report Issues](https://github.com/leanvibe/agent-hive/issues)
- **Email Support**: support@leanvibe.ai