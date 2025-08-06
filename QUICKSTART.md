# Docker Quick Start - LeanVibe Agent Hive 2.0

## Prerequisites

You only need:
- Docker Desktop (includes Docker Compose)
- Claude Code CLI
- Your Anthropic API key

## Step 1: Initial Setup (2 minutes)

```bash
# Clone or create project
mkdir leanvibe-hive && cd leanvibe-hive

# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-key-here  # For embeddings
EOF

# Create project structure
mkdir -p docker src bootstrap scripts config
```

## Step 2: Generate System with Claude Code

```bash
# Option A: Generate everything at once
claude "Create the complete LeanVibe Agent Hive 2.0 system with Docker support. 
Follow IMPLEMENTATION.md specifications. Create:
1. docker-compose.yml with all services
2. Dockerfiles in docker/ directory  
3. pyproject.toml with dependencies
4. bootstrap/init_agent.py
5. All core components in src/
Make everything Docker-ready with proper networking and volumes."

# Option B: Generate step by step
claude "Create docker-compose.yml for LeanVibe Agent Hive with PostgreSQL, Redis, and application containers"
claude "Create Dockerfiles for bootstrap, API, and agent containers in docker/ directory"
claude "Create the bootstrap agent in bootstrap/init_agent.py"
```

## Step 3: Start Core Services

```bash
# Start PostgreSQL and Redis only
docker-compose up -d postgres redis

# Verify they're running
docker-compose ps

# Check logs
docker-compose logs postgres redis
```

## Step 4: Run Bootstrap

```bash
# Run bootstrap in container
docker-compose run --rm bootstrap

# Or run specific bootstrap command
docker-compose run --rm bootstrap python bootstrap/init_agent.py test

# Watch bootstrap logs
docker-compose logs -f bootstrap
```

## Step 5: Start Full System

```bash
# Start API and agents
docker-compose --profile production up -d

# Start with monitoring
docker-compose --profile production --profile monitoring up -d

# Start development tools
docker-compose --profile dev up -d

# View all running services
docker-compose ps
```

## Service URLs

Once running, access:

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000
- **pgAdmin**: http://localhost:5050
  - Email: admin@leanvibe.com
  - Password: admin
- **Redis Commander**: http://localhost:8081
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001
  - Username: admin
  - Password: admin

## Common Docker Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

### Execute Commands in Containers
```bash
# Run Python shell in API container
docker-compose exec api python

# Access PostgreSQL
docker-compose exec postgres psql -U hive_user -d leanvibe_hive

# Access Redis CLI
docker-compose exec redis redis-cli

# Run tests
docker-compose run --rm api pytest tests/
```

### Manage Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (full reset)
docker-compose down -v

# Restart a service
docker-compose restart api

# Scale agent workers
docker-compose --profile agents up -d --scale agent-worker=5

# Rebuild after code changes
docker-compose build api
docker-compose up -d api
```

## Development Workflow

### Hot Reload Development
```bash
# Start services with code mounting
docker-compose --profile dev up -d

# Your code changes will auto-reload
# Edit src/ files and see changes immediately
```

### Run Tests in Container
```bash
# Unit tests
docker-compose run --rm api pytest tests/unit/ -v

# Integration tests  
docker-compose run --rm api pytest tests/integration/ -v

# With coverage
docker-compose run --rm api pytest --cov=src --cov-report=html
```

### Database Operations
```bash
# Create database backup
docker-compose exec postgres pg_dump -U hive_user leanvibe_hive > backup.sql

# Restore database
docker-compose exec -T postgres psql -U hive_user leanvibe_hive < backup.sql

# Run migrations
docker-compose run --rm api python scripts/migrate.py
```

## Debugging

### Container Shell Access
```bash
# Open shell in running container
docker-compose exec api /bin/bash

# Or start new container with shell
docker-compose run --rm api /bin/bash
```

### View Real-time Metrics
```bash
# CPU and memory usage
docker stats

# Detailed container inspection
docker-compose exec api ps aux
docker-compose exec api free -m
```

### Network Debugging
```bash
# Test connectivity between containers
docker-compose exec api ping postgres
docker-compose exec api nc -zv redis 6379

# View network configuration
docker network ls
docker network inspect leanvibe-hive_hive-network
```

## Production Deployment

### Build for Production
```bash
# Build production images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Push to registry
docker-compose push
```

### Environment Variables
```bash
# Production .env
cat > .env.production << EOF
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379
LOG_LEVEL=INFO
MAX_WORKERS=10
EOF

# Use production config
docker-compose --env-file .env.production up -d
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs <service-name>

# Check health
docker-compose ps

# Restart service
docker-compose restart <service-name>
```

### Database Connection Issues
```bash
# Verify PostgreSQL is running
docker-compose exec postgres pg_isready

# Check connection from API container
docker-compose exec api python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://hive_user:hive_pass@postgres:5432/leanvibe_hive')
engine.connect()
print('Connected!')
"
```

### Redis Connection Issues
```bash
# Test Redis connection
docker-compose exec redis redis-cli ping

# From API container
docker-compose exec api python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(r.ping())
"
```

### Port Conflicts
```bash
# If ports are already in use, modify docker-compose.yml:
# Change "8000:8000" to "8001:8000" for different host port

# Or stop conflicting services
lsof -i :8000  # Find what's using port
kill -9 <PID>  # Stop it
```

### Container Disk Space
```bash
# Clean up unused containers and images
docker system prune -a

# Remove specific volumes
docker volume rm leanvibe-hive_postgres_data

# Check disk usage
docker system df
```

## Advanced Configuration

### Custom Networks
```yaml
# Add to docker-compose.yml for network isolation
networks:
  frontend:
  backend:
  monitoring:
```

### Resource Limits
```yaml
# Add to service definition
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

### Health Checks
```yaml
# Add custom health check
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Monitoring

### View Prometheus Metrics
1. Open http://localhost:9090
2. Query examples:
   - `up` - Service availability
   - `rate(http_requests_total[5m])` - Request rate
   - `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` - 95th percentile latency

### Create Grafana Dashboard
1. Open http://localhost:3001
2. Add Prometheus data source: http://prometheus:9090
3. Import dashboard from `config/grafana/dashboards/`

## Backup and Recovery

### Automated Backups
```bash
# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T postgres pg_dump -U hive_user leanvibe_hive > backups/db_$DATE.sql
docker-compose exec -T redis redis-cli --rdb backups/redis_$DATE.rdb
EOF

chmod +x scripts/backup.sh
```

### Restore from Backup
```bash
# Database
docker-compose exec -T postgres psql -U hive_user leanvibe_hive < backups/db_20240101_120000.sql

# Redis
docker-compose exec redis redis-cli --rdb /data/dump.rdb < backups/redis_20240101_120000.rdb
```

## Tips for Success

1. **Always use Docker networks** - Services communicate using container names (e.g., `postgres`, `redis`)

2. **Mount code for development** - Your local changes reflect immediately in containers

3. **Use profiles** - Start only what you need:
   ```bash
   docker-compose up -d  # Core only
   docker-compose --profile dev up -d  # With dev tools
   ```

4. **Check logs frequently** - Most issues are visible in logs:
   ```bash
   docker-compose logs -f --tail=50
   ```

5. **Resource cleanup** - Periodically clean up:
   ```bash
   docker system prune -a --volumes
   ```

## Next Steps

Once everything is running:

1. **Verify bootstrap completed**:
   ```bash
   docker-compose logs bootstrap | grep "Bootstrap Complete"
   ```

2. **Check API health**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Submit first task**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/tasks \
     -H "Content-Type: application/json" \
     -d '{"title": "Analyze system", "type": "meta_analysis"}'
   ```

4. **Watch agents work**:
   ```bash
   docker-compose logs -f agent-worker
   ```

The system should now be self-improving and building itself! 🚀