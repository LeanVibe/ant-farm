#!/bin/bash
set -euo pipefail

# LeanVibe Agent Hive 2.0 - Production Deployment Script
# This script deploys the application to production environment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="docker-compose.production.yaml"
ENV_FILE=".env.production"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root is not recommended for production deployment"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create production env file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warn "Production environment file not found. Creating from template..."
        cp .env.example "$ENV_FILE"
        log_warn "Please edit $ENV_FILE with your production settings"
        exit 1
    fi
    
    # Create secrets directory
    mkdir -p secrets
    
    # Generate secrets if they don't exist
    if [[ ! -f secrets/secret_key.txt ]]; then
        log_info "Generating secret key..."
        openssl rand -hex 32 > secrets/secret_key.txt
    fi
    
    if [[ ! -f secrets/postgres_password.txt ]]; then
        log_info "Generating PostgreSQL password..."
        openssl rand -base64 32 > secrets/postgres_password.txt
    fi
    
    if [[ ! -f secrets/redis_password.txt ]]; then
        log_info "Generating Redis password..."
        openssl rand -base64 32 > secrets/redis_password.txt
    fi
    
    # Set secure permissions
    chmod 600 secrets/*
    
    log_info "Environment setup completed"
}

build_images() {
    log_info "Building Docker images..."
    
    # Set build args
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse --short HEAD)
    
    # Build images
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    log_info "Docker images built successfully"
}

deploy_application() {
    log_info "Deploying application..."
    
    # Pull latest images for services we don't build
    docker-compose -f "$COMPOSE_FILE" pull postgres redis nginx
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    timeout 300 bash -c 'until docker-compose -f '"$COMPOSE_FILE"' ps | grep -q "healthy"; do sleep 5; done'
    
    # Run database migrations
    log_info "Running database migrations..."
    docker-compose -f "$COMPOSE_FILE" exec -T api alembic upgrade head
    
    log_info "Application deployed successfully"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Check API health
    if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
        log_info "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check database connection
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U hive_user -d leanvibe_hive &> /dev/null; then
        log_info "Database health check passed"
    else
        log_error "Database health check failed"
        return 1
    fi
    
    # Check Redis connection
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &> /dev/null; then
        log_info "Redis health check passed"
    else
        log_error "Redis health check failed"
        return 1
    fi
    
    log_info "All health checks passed"
}

setup_monitoring() {
    log_info "Setting up monitoring (optional)..."
    
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        # Create Prometheus config if it doesn't exist
        mkdir -p docker/prometheus
        if [[ ! -f docker/prometheus/prometheus.yml ]]; then
            cat > docker/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hive-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
EOF
        fi
        
        # Start monitoring services
        docker-compose -f "$COMPOSE_FILE" --profile monitoring up -d
        
        log_info "Monitoring setup completed"
    else
        log_info "Monitoring disabled. Set ENABLE_MONITORING=true to enable."
    fi
}

show_deployment_info() {
    log_info "Deployment completed successfully!"
    echo
    echo "Services are running at:"
    echo "  Dashboard: http://localhost"
    echo "  API: http://localhost/api/v1"
    echo "  API Docs: http://localhost/api/v1/docs"
    echo
    echo "Useful commands:"
    echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  Check status: docker-compose -f $COMPOSE_FILE ps"
    echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  Update: ./scripts/deployment/deploy-production.sh"
    echo
    
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        echo "Monitoring:"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana: http://localhost:3000 (admin/admin)"
        echo
    fi
}

# Main execution
main() {
    log_info "Starting LeanVibe Agent Hive 2.0 production deployment..."
    
    check_prerequisites
    setup_environment
    build_images
    deploy_application
    run_health_checks
    setup_monitoring
    show_deployment_info
    
    log_info "Deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping services..."
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    "restart")
        log_info "Restarting services..."
        docker-compose -f "$COMPOSE_FILE" restart
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    "status")
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "health")
        run_health_checks
        ;;
    *)
        echo "Usage: $0 [deploy|stop|restart|logs|status|health]"
        exit 1
        ;;
esac