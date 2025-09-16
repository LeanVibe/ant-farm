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

#### POST /api/v1/system/batch/config
Apply batch configuration changes to system components (requires `system:write`).

**Request Body:**
```json
{
  "configurations": {
    "agents": {
      "default_timeout": 300,
      "max_concurrent_tasks": 5
    },
    "tasks": {
      "priority_levels": ["low", "medium", "high", "critical"],
      "auto_reassignment": true
    },
    "monitoring": {
      "metrics_interval": 30,
      "alert_thresholds": {
        "cpu_usage": 80,
        "memory_usage": 85
      }
    }
  }
}
```

#### POST /api/v1/system/batch/export
Export system data for backup or analysis (requires `system:read`).

**Request Body:**
```json
{
  "export_types": ["agents", "tasks", "metrics"],
  "date_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "format": "json"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "export_id": "export_12345",
    "files": [
      {
        "type": "agents",
        "filename": "agents_export_20240101_20240131.json",
        "size_bytes": 2048000
      },
      {
        "type": "tasks",
        "filename": "tasks_export_20240101_20240131.json",
        "size_bytes": 5120000
      }
    ],
    "download_urls": {
      "agents": "/api/v1/downloads/export_12345_agents.json",
      "tasks": "/api/v1/downloads/export_12345_tasks.json"
    }
  }
}
```

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

#### POST /api/v1/agents/batch/spawn
Spawn multiple agents from configuration file (requires `agent:spawn`).

**Request Body:**
```json
{
  "agents": [
    {
      "name": "dev-agent-001",
      "type": "developer",
      "config": {
        "max_concurrent_tasks": 3,
        "specialization": "backend"
      }
    },
    {
      "name": "qa-agent-001",
      "type": "qa",
      "config": {
        "test_types": ["unit", "integration"],
        "coverage_target": 90
      }
    }
  ]
}
```

#### POST /api/v1/agents/batch/stop
Stop multiple agents based on filter criteria (requires `agent:terminate`).

**Request Body:**
```json
{
  "filter": {
    "type": "meta",
    "status": "idle"
  }
}
```

#### POST /api/v1/agents/batch/update
Update configuration for multiple agents (requires `agent:write`).

**Request Body:**
```json
{
  "filter": {
    "type": "developer"
  },
  "config": {
    "max_concurrent_tasks": 5,
    "timeout": 300
  }
}
```

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

#### POST /api/v1/tasks/batch/reassign
Reassign multiple tasks based on filter criteria (requires `task:write`).

**Request Body:**
```json
{
  "filter": {
    "status": "pending",
    "type": "code_review"
  },
  "new_assignee": "qa-agent-001",
  "reassignment_reason": "Load balancing optimization"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "reassigned_count": 5,
    "tasks": [
      {
        "task_id": "task_123",
        "old_assignee": "dev-agent-001",
        "new_assignee": "qa-agent-001"
      }
    ]
  }
}
```

### Messaging

#### POST /api/v1/messages
Send a message to an agent (requires `message:send`).

#### POST /api/v1/broadcast
Broadcast a message to all agents (requires `message:broadcast`).

### Metrics and Monitoring

#### GET /api/v1/metrics
Get system metrics (requires `metrics:read`).

#### GET /api/v1/metrics/agents
Get agent performance metrics and analytics (requires `metrics:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "total_agents": 5,
    "active_agents": 4,
    "avg_cpu_usage": 45.2,
    "avg_memory_usage": 67.8,
    "top_performers": [
      {
        "agent_name": "meta-agent-001",
        "tasks_completed": 25,
        "success_rate": 0.96,
        "avg_response_time": 2.3
      }
    ],
    "performance_trends": {
      "cpu_trend": "stable",
      "memory_trend": "increasing",
      "task_completion_trend": "improving"
    }
  }
}
```

#### GET /api/v1/metrics/tasks
Get task performance metrics and completion statistics (requires `metrics:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "total_tasks": 150,
    "completed_tasks": 142,
    "failed_tasks": 3,
    "pending_tasks": 5,
    "avg_completion_time": 45.6,
    "success_rate": 0.947,
    "task_distribution": {
      "development": 45,
      "analysis": 30,
      "review": 25,
      "testing": 20
    },
    "performance_trends": {
      "completion_rate_trend": "improving",
      "failure_rate_trend": "stable"
    }
  }
}
```

#### GET /api/v1/monitoring/agents
Real-time agent monitoring data (requires `metrics:read`).

**Query Parameters:**
- `live`: Enable WebSocket streaming for real-time updates
- `filter`: Filter by agent type, status, or name

#### GET /api/v1/monitoring/tasks
Real-time task monitoring data (requires `metrics:read`).

**Query Parameters:**
- `live`: Enable WebSocket streaming for real-time updates
- `status`: Filter by task status
- `assigned_to`: Filter by assigned agent

#### GET /api/v1/monitoring/system
Real-time system health and resource monitoring (requires `metrics:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "health_score": 0.95,
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.4,
    "network_io": {
      "bytes_sent": 1024000,
      "bytes_received": 2048000
    },
    "active_connections": 12,
    "uptime_seconds": 86400,
    "system_load": {
      "1min": 1.2,
      "5min": 1.1,
      "15min": 1.0
    }
  }
}
```

#### GET /api/v1/performance
Get performance optimization statistics.

#### POST /api/v1/performance/optimize
Trigger performance optimization.

### Advanced Monitoring (Batch 6 - Performance Monitoring)

#### GET /api/v1/monitoring/metrics
Get comprehensive system metrics including CPU, memory, disk, network, database, broker, and agent metrics (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics": {
      "cpu_percent": {"latest": 45.2, "average": 42.1, "min": 12.3, "max": 89.5},
      "memory_percent": {"latest": 67.8, "average": 65.4, "min": 45.2, "max": 78.9},
      "db_health_status": {"latest": 1.0, "average": 0.99},
      "agents_total": {"latest": 5, "average": 5},
      "agents_active": {"latest": 4, "average": 3.8}
    },
    "health_score": 0.87,
    "timestamp": 1640995200.0
  }
}
```

#### GET /api/v1/monitoring/alerts
Get active performance alerts and recent alert history (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "alerts": [
      {
        "id": "alert_1640995200_cpu_high",
        "title": "High CPU Usage Detected",
        "description": "CPU usage is 85.2%, exceeding threshold of 80%",
        "severity": "high",
        "metric_name": "cpu_percent",
        "current_value": 85.2,
        "threshold_value": 80.0,
        "created_at": 1640995200.0,
        "status": "active"
      }
    ]
  }
}
```

#### GET /api/v1/monitoring/bottlenecks
Get system bottleneck analysis and recommendations (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "bottlenecks": {
      "cpu_bottleneck": true,
      "memory_bottleneck": false,
      "disk_bottleneck": false,
      "database_bottleneck": false,
      "broker_bottleneck": true,
      "agent_bottleneck": false,
      "recommendations": [
        "High CPU usage detected - consider scaling compute resources",
        "Message broker queue depth is high - check message processing rate"
      ]
    }
  }
}
```

#### GET /api/v1/monitoring/health
Get monitoring system health status (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics_collector": true,
    "bottleneck_detector": true,
    "alert_manager": true,
    "health_score": 0.92,
    "timestamp": 1640995200.0
  }
}
```

#### POST /api/v1/monitoring/alerts/{alert_id}/acknowledge
Acknowledge an active alert to prevent further notifications (requires `system:write`).

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Alert acknowledged successfully"
  }
}
```

#### GET /api/v1/monitoring/performance
Get performance optimization statistics and historical performance data (requires `system:read`).

**Query Parameters:**
- `time_range`: Time range for performance data (default: "1h", options: "1h", "24h", "7d", "30d")
- `metrics`: Comma-separated list of metrics to include (default: all)
- `include_trends`: Include performance trend analysis (default: true)

**Response:**
```json
{
  "success": true,
  "data": {
    "performance_stats": {
      "optimization_events": 12,
      "avg_improvement_percent": 45.2,
      "total_optimizations_applied": 8,
      "performance_trends": {
        "response_time_trend": "improving",
        "throughput_trend": "stable",
        "error_rate_trend": "improving"
      }
    },
    "recent_optimizations": [
      {
        "id": "opt_1640995200",
        "type": "database_query_optimization",
        "description": "Optimized slow queries with new indexes",
        "improvement_percent": 60.5,
        "applied_at": 1640995200.0,
        "status": "successful"
      }
    ],
    "baseline_comparison": {
      "current_response_time": 45.2,
      "baseline_response_time": 100.0,
      "improvement_percent": 54.8
    }
  }
}
```

#### POST /api/v1/monitoring/performance/optimize
Trigger performance optimization analysis and apply recommended optimizations (requires `system:write`).

**Request Body:**
```json
{
  "optimization_types": ["database", "cache", "agent_load"],
  "aggressive_mode": false,
  "dry_run": true
}
```

**Parameters:**
- `optimization_types`: Array of optimization types to run (optional)
- `aggressive_mode`: Apply more aggressive optimizations (default: false)
- `dry_run`: Preview optimizations without applying (default: false)

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_1640995300",
    "recommendations": [
      {
        "type": "database_index",
        "description": "Add composite index on tasks table",
        "estimated_improvement": 35.2,
        "risk_level": "low",
        "applied": false
      },
      {
        "type": "cache_optimization",
        "description": "Increase Redis cache size",
        "estimated_improvement": 25.8,
        "risk_level": "low",
        "applied": false
      }
    ],
    "estimated_total_improvement": 61.0,
    "execution_time_estimate": 45,
    "dry_run": true
  }
}
```

#### GET /api/v1/monitoring/performance/history
Get historical performance optimization results and trends (requires `system:read`).

**Query Parameters:**
- `start_date`: Start date for history (ISO 8601 format)
- `end_date`: End date for history (ISO 8601 format)
- `optimization_type`: Filter by optimization type

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_history": [
      {
        "id": "opt_1640995200",
        "type": "database_optimization",
        "description": "Query optimization and indexing",
        "improvement_percent": 45.2,
        "applied_at": 1640995200.0,
        "status": "successful",
        "duration_seconds": 120
      }
    ],
    "performance_trends": {
      "monthly_improvements": [
        {"month": "2024-01", "avg_improvement": 42.3},
        {"month": "2024-02", "avg_improvement": 38.7}
      ],
      "optimization_success_rate": 0.92
    }
  }
}
```

#### GET /api/v1/monitoring/anomalies
Get detected performance anomalies and unusual patterns (requires `system:read`).

**Query Parameters:**
- `severity`: Filter by anomaly severity ("low", "medium", "high", "critical")
- `time_range`: Time range to analyze (default: "1h")
- `include_resolved`: Include resolved anomalies (default: false)

**Response:**
```json
{
  "success": true,
  "data": {
    "anomalies": [
      {
        "id": "anom_1640995200",
        "type": "response_time_spike",
        "description": "API response time increased by 300% in last 5 minutes",
        "severity": "high",
        "detected_at": 1640995200.0,
        "affected_components": ["api", "database"],
        "potential_cause": "Database connection pool exhausted",
        "recommendations": [
          "Scale database connections",
          "Review slow queries"
        ],
        "status": "active"
      }
    ],
    "anomaly_summary": {
      "total_anomalies": 3,
      "by_severity": {
        "low": 1,
        "medium": 1,
        "high": 1,
        "critical": 0
      },
      "detection_accuracy": 0.94
    }
  }
}
```

#### POST /api/v1/monitoring/thresholds
Update monitoring thresholds for alerts and performance tracking (requires `system:write`).

**Request Body:**
```json
{
  "thresholds": {
    "cpu_usage_percent": {
      "warning": 70,
      "critical": 85
    },
    "memory_usage_percent": {
      "warning": 75,
      "critical": 90
    },
    "api_response_time_ms": {
      "warning": 100,
      "critical": 200
    },
    "database_connection_pool_utilization": {
      "warning": 80,
      "critical": 95
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Monitoring thresholds updated successfully",
    "updated_thresholds": 8,
    "effective_from": 1640995200.0
  }
}
```

#### GET /api/v1/monitoring/thresholds
Get current monitoring thresholds and alert configurations (requires `system:read`).

**Response:**
```json
{
  "success": true,
  "data": {
    "thresholds": {
      "cpu_usage_percent": {
        "warning": 70,
        "critical": 85,
        "description": "System CPU usage thresholds"
      },
      "memory_usage_percent": {
        "warning": 75,
        "critical": 90,
        "description": "System memory usage thresholds"
      },
      "api_response_time_ms": {
        "warning": 100,
        "critical": 200,
        "description": "API response time thresholds"
      },
      "database_connection_pool_utilization": {
        "warning": 80,
        "critical": 95,
        "description": "Database connection pool utilization thresholds"
      }
    },
    "last_updated": 1640995200.0,
    "version": "2.0"
  }
}
```

#### GET /api/v1/monitoring/reports
Generate comprehensive monitoring reports (requires `system:read`).

**Query Parameters:**
- `report_type`: Type of report ("daily", "weekly", "monthly", "custom")
- `start_date`: Report start date (ISO 8601 format)
- `end_date`: Report end date (ISO 8601 format)
- `format`: Report format ("json", "pdf", "html")

**Response:**
```json
{
  "success": true,
  "data": {
    "report_id": "report_1640995200",
    "report_type": "weekly",
    "time_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-07T23:59:59Z"
    },
    "summary": {
      "total_uptime_percent": 99.7,
      "avg_response_time_ms": 45.2,
      "total_requests": 150000,
      "error_rate_percent": 0.05
    },
    "performance_metrics": {
      "cpu_usage_avg": 42.1,
      "memory_usage_avg": 65.4,
      "database_performance_score": 0.92
    },
    "incidents": [
      {
        "id": "incident_1640995200",
        "title": "Database Connection Pool Exhausted",
        "severity": "high",
        "duration_minutes": 15,
        "resolution": "Scaled database connections"
      }
    ],
    "download_url": "/api/v1/downloads/report_1640995200.pdf"
  }
}
```

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

### v2.2.0 (Current)
- ✅ **Batch 6: Performance Monitoring System**
  - Comprehensive metrics collection (system, database, broker, agent metrics)
  - Intelligent bottleneck detection with configurable thresholds
  - Multi-channel alerting (email, Slack, webhooks) with escalation
  - Alert deduplication and correlation
  - Real-time monitoring dashboard endpoints
  - Performance optimization validation (40-60% improvement target)
- ✅ Enhanced monitoring endpoints (`/api/v1/monitoring/*`)
- ✅ Added real-time monitoring endpoints (`/api/v1/monitoring/*`)
- ✅ Added performance metrics endpoints (`/api/v1/metrics/*`)
- ✅ Added batch operations for agents (`/api/v1/agents/batch/*`)
- ✅ Added batch operations for tasks (`/api/v1/tasks/batch/*`)
- ✅ Added batch operations for system (`/api/v1/system/batch/*`)
- ✅ Enhanced WebSocket streaming for live monitoring
- ✅ Added advanced filtering and analytics capabilities

### v2.0.0
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