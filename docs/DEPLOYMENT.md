# LeanVibe Agent Hive 2.0 - Production Deployment Guide

This guide covers deploying LeanVibe Agent Hive 2.0 to production using Docker containers.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: Minimum 50GB available space
- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Network**: Internet connection for downloading dependencies

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Git
- OpenSSL (for generating secrets)
- curl (for health checks)

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ant-farm
```

### 2. Configure Environment
```bash
# Copy and edit production environment
cp .env.example .env.production

# Edit with your production settings
nano .env.production
```

### 3. Deploy
```bash
# Run the deployment script
./scripts/deployment/deploy-production.sh
```

The deployment script will:
- Check prerequisites
- Generate secure secrets
- Build Docker images
- Deploy all services
- Run health checks
- Display access information

### 4. Access the Application
- **Dashboard**: http://your-server/
- **API**: http://your-server/api/v1
- **API Documentation**: http://your-server/api/v1/docs

## Manual Deployment Steps

### 1. Environment Configuration

Create production environment file:
```bash
cp .env.example .env.production
```

Edit `.env.production` with your production settings:
```bash
# Required settings
ENVIRONMENT=production
SECRET_KEY=your-secure-secret-key
POSTGRES_PASSWORD=your-secure-db-password
REDIS_PASSWORD=your-secure-redis-password

# API Keys (at least one required)
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key

# Domain configuration
DOMAIN=yourdomain.com
CORS_ORIGINS=["https://yourdomain.com","https://api.yourdomain.com"]
```

### 2. Generate Secrets

Create secrets directory and generate secure passwords:
```bash
mkdir -p secrets

# Generate secret key for JWT tokens
openssl rand -hex 32 > secrets/secret_key.txt

# Generate database password
openssl rand -base64 32 > secrets/postgres_password.txt

# Generate Redis password
openssl rand -base64 32 > secrets/redis_password.txt

# Set secure permissions
chmod 600 secrets/*
```

### 3. Build and Deploy

Build Docker images:
```bash
export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
export VCS_REF=$(git rev-parse --short HEAD)

docker-compose -f docker-compose.production.yaml build
```

Deploy services:
```bash
docker-compose -f docker-compose.production.yaml up -d
```

### 4. Initialize Database

Run database migrations:
```bash
docker-compose -f docker-compose.production.yaml exec api alembic upgrade head
```

### 5. Verify Deployment

Check service status:
```bash
docker-compose -f docker-compose.production.yaml ps
```

Run health checks:
```bash
curl -f http://localhost:8000/api/v1/health
```

## Production Architecture

### Services

1. **nginx** (Port 80/443)
   - Reverse proxy and static file server
   - SSL termination
   - Rate limiting and security headers

2. **api** (Port 8000, internal)
   - FastAPI application server
   - 4 worker processes for production
   - Health checks and metrics

3. **agent-runner** (Internal)
   - Agent orchestration and execution
   - Tmux session management
   - CLI tool integration

4. **postgres** (Port 5432, internal)
   - PostgreSQL database with pgvector
   - Persistent data storage
   - Connection pooling

5. **redis** (Port 6379, internal)
   - Task queue and message broker
   - Session storage
   - Caching layer

### Optional Services

6. **prometheus** (Port 9090)
   - Metrics collection and alerting
   - Enable with `--profile monitoring`

7. **grafana** (Port 3000)
   - Metrics visualization
   - Enable with `--profile monitoring`

## Security Configuration

### Network Security
- All services run in isolated Docker network
- Only nginx exposes public ports
- Internal service communication only

### Authentication
- JWT-based authentication with RS256
- Configurable token expiration
- Rate limiting on authentication endpoints

### Data Protection
- Secrets stored in separate files with restricted permissions
- Database and Redis password protection
- SSL/TLS termination at nginx

### Security Headers
- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- X-Frame-Options, X-Content-Type-Options
- Referrer Policy

## Monitoring and Observability

### Comprehensive Monitoring Setup

The LeanVibe Agent Hive includes a comprehensive monitoring stack with metrics collection, alerting, and visualization capabilities.

#### Core Monitoring Components

1. **Metrics Collection**: Built-in metrics collector with real-time performance monitoring
2. **Alert Manager**: Intelligent alerting with multi-channel notifications
3. **Bottleneck Detector**: Automated performance analysis and optimization recommendations
4. **Health Monitoring**: System health scoring and component status tracking

#### Deployment Options

**Option 1: Built-in Monitoring (Recommended)**
```bash
# Enable monitoring in environment
echo "ENABLE_MONITORING=true" >> .env.production
echo "MONITORING_RETENTION_DAYS=30" >> .env.production

# Deploy with monitoring
docker-compose -f docker-compose.production.yaml up -d
```

**Option 2: External Monitoring Stack**
```bash
# Deploy with external monitoring profile
docker-compose -f docker-compose.production.yaml --profile monitoring up -d
```

### Monitoring Configuration

#### Environment Variables
```bash
# Core monitoring settings
ENABLE_MONITORING=true
MONITORING_RETENTION_DAYS=30
METRICS_COLLECTION_INTERVAL=30
HEALTH_CHECK_INTERVAL=60

# Alerting configuration
ALERT_EMAIL_RECIPIENTS=admin@yourdomain.com,ops@yourdomain.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
PAGERDUTY_ROUTING_KEY=your_pagerduty_key

# Performance thresholds
CPU_WARNING_THRESHOLD=70
CPU_CRITICAL_THRESHOLD=85
MEMORY_WARNING_THRESHOLD=75
MEMORY_CRITICAL_THRESHOLD=90
API_RESPONSE_TIME_WARNING=100
API_RESPONSE_TIME_CRITICAL=200

# External integrations
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
DATADOG_ENABLED=false
DATADOG_API_KEY=your_datadog_key
```

#### Alerting Rules
```yaml
# alerting_rules.yml
groups:
  - name: leanvibe_alerts
    rules:
      - alert: HighCPUUsage
        expr: leanvibe_system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}%"

      - alert: HighMemoryUsage
        expr: leanvibe_system_memory_usage_percent > 85
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}%"

      - alert: APISlowResponse
        expr: histogram_quantile(0.95, rate(leanvibe_api_request_duration_seconds_bucket[5m])) > 0.2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Slow API response time"
          description: "95th percentile response time is {{ $value }}s"
```

### Application Logs

#### Log Aggregation
```bash
# View all logs
docker-compose -f docker-compose.production.yaml logs -f

# View specific service logs
docker-compose -f docker-compose.production.yaml logs -f api
docker-compose -f docker-compose.production.yaml logs -f agent-runner

# Follow logs with timestamps
docker-compose -f docker-compose.production.yaml logs -f -t --tail=100
```

#### Log Rotation
```yaml
# docker-compose.production.yaml logging configuration
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  agent-runner:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Health Monitoring

#### Health Checks
```bash
# Check all service health
docker-compose -f docker-compose.production.yaml ps

# API health endpoint
curl -f http://localhost/api/v1/health

# Database health
docker-compose -f docker-compose.production.yaml exec postgres pg_isready -U hive_user -d leanvibe_hive

# Redis health
docker-compose -f docker-compose.production.yaml exec redis redis-cli ping

# Comprehensive health check
curl http://localhost/api/v1/monitoring/health
```

#### Health Endpoints
- **API Health**: `/api/v1/health` - Basic service availability
- **System Health**: `/api/v1/monitoring/health` - Comprehensive system health
- **Metrics Health**: `/metrics` - Prometheus metrics endpoint
- **Database Health**: Check via `pg_isready`
- **Redis Health**: Check via `redis-cli ping`

### Metrics and Dashboards

#### Built-in Metrics
The system exposes comprehensive metrics for monitoring:

**System Metrics:**
- CPU usage and load averages
- Memory usage and swap
- Disk I/O and space
- Network I/O and connections

**Application Metrics:**
- API request rates and response times
- Database connection pool utilization
- Redis memory usage and hit rates
- Agent performance and task completion rates

**Business Metrics:**
- Active agents and sessions
- Task completion rates
- Error rates and types
- User activity patterns

#### Grafana Dashboard Setup
```bash
# Access Grafana
# URL: http://localhost:3000
# Default credentials: admin/admin

# Add Prometheus data source
# Configuration > Data Sources > Add data source > Prometheus
# URL: http://prometheus:9090
```

#### Sample Dashboard Panels

**API Performance Panel:**
```json
{
  "title": "API Response Time",
  "type": "graph",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(leanvibe_api_request_duration_seconds_bucket[5m]))",
      "legendFormat": "95th percentile"
    },
    {
      "expr": "histogram_quantile(0.50, rate(leanvibe_api_request_duration_seconds_bucket[5m]))",
      "legendFormat": "50th percentile"
    }
  ]
}
```

**System Resources Panel:**
```json
{
  "title": "System Resources",
  "type": "bargauge",
  "targets": [
    {
      "expr": "leanvibe_system_cpu_usage_percent",
      "title": "CPU Usage %"
    },
    {
      "expr": "leanvibe_system_memory_usage_percent",
      "title": "Memory Usage %"
    }
  ]
}
```

### Alerting Configuration

#### Email Alerts
```bash
# Configure SMTP settings
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@yourdomain.com
SMTP_PASSWORD=your_smtp_password
ALERT_EMAIL_FROM=alerts@yourdomain.com
ALERT_EMAIL_RECIPIENTS=admin@yourdomain.com,ops@yourdomain.com
```

#### Slack Integration
```bash
# Slack webhook configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_CHANNEL=#alerts
SLACK_USERNAME=LeanVibe Monitor
```

#### PagerDuty Integration
```bash
# PagerDuty configuration
PAGERDUTY_ROUTING_KEY=your_pagerduty_integration_key
PAGERDUTY_SEVERITY_MAPPING='{"warning": "warning", "critical": "critical"}'
```

### Performance Monitoring

#### Real-time Performance Tracking
```bash
# Monitor real-time performance
curl http://localhost/api/v1/monitoring/metrics

# Get performance optimization recommendations
curl http://localhost/api/v1/monitoring/bottlenecks

# View active alerts
curl http://localhost/api/v1/monitoring/alerts
```

#### Performance Baselines
- **API Response Time**: <100ms average, <200ms 95th percentile
- **CPU Usage**: <70% sustained, <85% peak
- **Memory Usage**: <75% sustained, <90% peak
- **Database Connections**: <80% pool utilization
- **Agent Uptime**: >95% availability

#### Automated Optimization
```bash
# Trigger performance optimization
curl -X POST http://localhost/api/v1/monitoring/performance/optimize \
  -H "Content-Type: application/json" \
  -d '{"optimization_types": ["database", "cache"], "dry_run": false}'
```

### Security Monitoring

#### Security Event Logging
```bash
# Security-relevant logs
docker-compose -f docker-compose.production.yaml logs | grep -E "(auth|security|unauthorized|forbidden)"

# Failed authentication attempts
docker-compose -f docker-compose.production.yaml logs api | grep "authentication failed"
```

#### Security Alerts
- Failed authentication attempts (>5 per minute)
- Unauthorized access attempts
- Suspicious API usage patterns
- Configuration changes
- Secret access events

### Log Aggregation and Analysis

#### Centralized Logging Setup
```yaml
# docker-compose.production.yaml logging configuration
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service,environment"

  agent-runner:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service,environment"
```

#### Log Analysis Commands
```bash
# Search for errors in last hour
docker-compose -f docker-compose.production.yaml logs --since 1h | grep ERROR

# Count error occurrences by service
docker-compose -f docker-compose.production.yaml logs | grep ERROR | awk '{print $1}' | sort | uniq -c

# Monitor authentication failures
docker-compose -f docker-compose.production.yaml logs -f | grep "authentication failed"
```

### Monitoring Best Practices

#### Production Monitoring Checklist
- [ ] Enable comprehensive metrics collection
- [ ] Configure alerting for critical thresholds
- [ ] Set up log aggregation and rotation
- [ ] Configure external monitoring integrations
- [ ] Establish performance baselines
- [ ] Set up automated alerting channels
- [ ] Configure log retention policies
- [ ] Set up monitoring dashboards
- [ ] Test alerting and notification systems
- [ ] Document monitoring procedures

#### Monitoring Maintenance
```bash
# Regular monitoring tasks
# 1. Review alert configurations quarterly
# 2. Update performance baselines monthly
# 3. Archive old logs according to retention policy
# 4. Test alerting systems monthly
# 5. Review monitoring dashboards for relevance
# 6. Update monitoring documentation as needed
```

#### Scaling Monitoring
```bash
# Scale monitoring services if needed
docker-compose -f docker-compose.production.yaml up -d --scale prometheus=2
docker-compose -f docker-compose.production.yaml up -d --scale grafana=2
```

### Troubleshooting Monitoring Issues

#### Common Monitoring Problems

1. **Metrics not appearing in Prometheus**
   ```bash
   # Check metrics endpoint
   curl http://localhost/metrics

   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets

   # Verify scrape configuration
   curl http://localhost:9090/api/v1/status/config
   ```

2. **Alerts not firing**
   ```bash
   # Check alerting rules
   curl http://localhost:9090/api/v1/rules

   # Check alertmanager status
   curl http://localhost:9093/api/v2/alerts
   ```

3. **Grafana dashboards not loading**
   ```bash
   # Check Grafana logs
   docker-compose -f docker-compose.production.yaml logs grafana

   # Verify data source configuration
   # Grafana UI > Configuration > Data Sources > Prometheus
   ```

4. **High monitoring resource usage**
   ```bash
   # Check monitoring service resource usage
   docker stats

   # Adjust monitoring intervals
   METRICS_COLLECTION_INTERVAL=60  # Increase from 30s to 60s
   ```

### Monitoring Architecture

#### Monitoring Data Flow
```
Application Metrics → Metrics Collector → Redis Cache → Prometheus → Grafana
       ↓
Alert Manager → Notification Channels (Email/Slack/PagerDuty)
       ↓
Incident Response → Automated Actions → Performance Optimization
```

#### Monitoring Components
- **Metrics Collector**: Real-time metrics gathering (30s intervals)
- **Alert Manager**: Intelligent alerting with deduplication
- **Bottleneck Detector**: Performance analysis and recommendations
- **Health Monitor**: System health scoring and component tracking
- **Performance Optimizer**: Automated optimization suggestions

This comprehensive monitoring setup ensures the LeanVibe Agent Hive runs reliably in production with full observability and automated performance optimization.

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker-compose -f docker-compose.production.yaml exec postgres pg_dump -U hive_user leanvibe_hive > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
docker-compose -f docker-compose.production.yaml exec -T postgres psql -U hive_user leanvibe_hive < backup_file.sql
```

### Data Volumes
Production data is stored in Docker volumes:
- `postgres_data`: Database files
- `redis_data`: Redis persistence
- `api_logs`, `agent_logs`: Application logs
- `api_workspace`, `agent_workspace`: Working directories

### Full Backup
```bash
# Stop services
docker-compose -f docker-compose.production.yaml down

# Backup volumes
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup_$(date +%Y%m%d).tar.gz -C /data .
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup_$(date +%Y%m%d).tar.gz -C /data .

# Restart services
docker-compose -f docker-compose.production.yaml up -d
```

## Maintenance

### Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
./scripts/deployment/deploy-production.sh
```

### Scaling
```bash
# Scale API service
docker-compose -f docker-compose.production.yaml up -d --scale api=3

# Scale agent runners
docker-compose -f docker-compose.production.yaml up -d --scale agent-runner=2
```

### Resource Limits
Resource limits are configured in docker-compose.production.yaml:
- **API**: 2GB RAM, 1 CPU
- **Agent Runner**: 4GB RAM, 2 CPU
- **PostgreSQL**: 2GB RAM, 1 CPU
- **Redis**: 1GB RAM, 0.5 CPU

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   docker-compose -f docker-compose.production.yaml logs service-name
   
   # Check resource usage
   docker stats
   ```

2. **Database connection issues**
   ```bash
   # Check PostgreSQL logs
   docker-compose -f docker-compose.production.yaml logs postgres
   
   # Test connection
   docker-compose -f docker-compose.production.yaml exec postgres psql -U hive_user -d leanvibe_hive -c "SELECT 1;"
   ```

3. **Redis connection issues**
   ```bash
   # Check Redis logs
   docker-compose -f docker-compose.production.yaml logs redis
   
   # Test connection
   docker-compose -f docker-compose.production.yaml exec redis redis-cli ping
   ```

4. **Agent execution issues**
   ```bash
   # Check agent runner logs
   docker-compose -f docker-compose.production.yaml logs agent-runner
   
   # Check tmux sessions
   docker-compose -f docker-compose.production.yaml exec agent-runner tmux list-sessions
   ```

### Performance Issues

1. **High memory usage**
   - Check Docker stats: `docker stats`
   - Adjust resource limits in docker-compose.production.yaml
   - Consider scaling horizontally

2. **Slow API responses**
   - Check nginx logs: `docker-compose logs nginx`
   - Monitor database performance
   - Check Redis memory usage

3. **Database performance**
   - Monitor connection pool usage
   - Check slow query logs
   - Consider database tuning

## Production Checklist

Before deploying to production:

- [ ] Configure secure secrets (SECRET_KEY, passwords)
- [ ] Set up SSL certificates for HTTPS
- [ ] Configure proper domain and CORS settings
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Set up automated backups
- [ ] Test disaster recovery procedures
- [ ] Configure firewall rules
- [ ] Set up CI/CD pipeline
- [ ] Performance testing completed
- [ ] Security audit completed

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review service logs
3. Check the GitHub issues page
4. Contact the development team

---

**Note**: This deployment is suitable for production use but may require additional configuration for high-availability setups or specific compliance requirements.