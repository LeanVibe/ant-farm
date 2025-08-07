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

## Monitoring and Logging

### Application Logs
```bash
# View all logs
docker-compose -f docker-compose.production.yaml logs -f

# View specific service logs
docker-compose -f docker-compose.production.yaml logs -f api
docker-compose -f docker-compose.production.yaml logs -f agent-runner
```

### Health Monitoring
```bash
# Check all service health
docker-compose -f docker-compose.production.yaml ps

# API health endpoint
curl http://localhost/api/v1/health

# Database health
docker-compose -f docker-compose.production.yaml exec postgres pg_isready -U hive_user -d leanvibe_hive
```

### Metrics (Optional)
Enable monitoring profile:
```bash
ENABLE_MONITORING=true docker-compose -f docker-compose.production.yaml --profile monitoring up -d
```

Access monitoring:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

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