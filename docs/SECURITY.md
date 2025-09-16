# LeanVibe Agent Hive - Security Guide

## Overview

The LeanVibe Agent Hive implements a comprehensive security framework designed to protect sensitive data, ensure system integrity, and maintain compliance with industry standards. This guide covers authentication, data protection, network security, compliance requirements, and operational security practices.

## Authentication and Authorization

### JWT-Based Authentication

The system uses JSON Web Tokens (JWT) for secure API authentication:

```python
# JWT token structure
{
  "sub": "user_id",
  "exp": 1640995200,
  "iat": 1640991600,
  "role": "admin",
  "permissions": ["read", "write", "admin"]
}
```

**Configuration:**
- Token expiration: 24 hours
- Refresh token validity: 7 days
- Algorithm: RS256 (RSA signature)

### Role-Based Access Control (RBAC)

| Role | Permissions | Description |
|------|-------------|-------------|
| `admin` | Full system access | System administration and configuration |
| `operator` | Agent and task management | Day-to-day operations |
| `viewer` | Read-only access | Monitoring and reporting |
| `agent` | Limited API access | Agent-specific operations |

### API Key Management

For external integrations (Claude API, monitoring systems):

```bash
# Environment configuration
ANTHROPIC_API_KEY=your_secure_api_key
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DATADOG_API_KEY=your_datadog_key
```

**Security measures:**
- Keys encrypted at rest
- Access logging for all API calls
- Rate limiting per API key
- Automatic key rotation capabilities

## Data Protection

### Encryption at Rest

- **Database encryption**: PostgreSQL with pgcrypto extension
- **File storage**: AES-256 encryption for sensitive files
- **Configuration files**: Encrypted environment variables

```sql
-- Example: Encrypt sensitive data
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt API keys in database
UPDATE agents SET api_key = pgp_sym_encrypt('your_api_key', 'encryption_key');
```

### Encryption in Transit

- **HTTPS enforcement**: All API endpoints require TLS 1.3
- **Database connections**: SSL/TLS encryption
- **Redis connections**: TLS encryption enabled
- **WebSocket connections**: WSS (WebSocket Secure)

### Secure Secret Management

**Environment Variables:**
```bash
# .env configuration
JWT_SECRET_KEY=your_256_bit_secret
DB_ENCRYPTION_KEY=your_db_encryption_key
REDIS_PASSWORD=secure_redis_password
ANTHROPIC_API_KEY=encrypted_api_key
```

**Best practices:**
- Never commit secrets to version control
- Use environment-specific secret stores
- Implement secret rotation policies
- Log access to secrets (without revealing values)

## Network Security

### Firewall Configuration

**Recommended iptables rules:**
```bash
# Allow only necessary ports
iptables -A INPUT -p tcp --dport 9001 -j ACCEPT  # API
iptables -A INPUT -p tcp --dport 5433 -j ACCEPT  # PostgreSQL
iptables -A INPUT -p tcp --dport 6381 -j ACCEPT  # Redis
iptables -A INPUT -p tcp --dport 80 -j ACCEPT   # HTTP (redirect to HTTPS)
iptables -A INPUT -p tcp --dport 443 -j ACCEPT  # HTTPS
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -P INPUT DROP
```

### Rate Limiting

**API Rate Limits:**
- General endpoints: 100 requests/minute per IP
- Authentication endpoints: 10 requests/minute per IP
- Agent operations: 500 requests/minute per agent
- Monitoring endpoints: 1000 requests/minute

**Implementation:**
```python
# FastAPI rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

### CORS Configuration

```python
# CORS settings
CORS_ORIGINS = [
    "https://yourdomain.com",
    "https://dashboard.yourdomain.com"
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
CORS_ALLOW_HEADERS = ["*"]
```

## Compliance Considerations

### GDPR Compliance

**Data Subject Rights:**
- Right to access personal data
- Right to rectification
- Right to erasure ("right to be forgotten")
- Right to data portability

**Implementation:**
- Data minimization principles
- Consent management
- Automated data deletion workflows
- Audit trails for data access

### HIPAA Compliance (Healthcare Use Cases)

**Protected Health Information (PHI):**
- Agent conversation logs containing health data
- User profile information
- System performance data with user identifiers

**Security measures:**
- PHI encryption at rest and in transit
- Access controls with audit logging
- Business Associate Agreements (BAA) for cloud providers
- Incident response procedures for PHI breaches

### SOC 2 Compliance

**Trust Service Criteria:**
- Security: Protect against unauthorized access
- Availability: System uptime and performance
- Processing Integrity: Accurate and timely processing
- Confidentiality: Protect sensitive information
- Privacy: Personal information handling

**Controls:**
- Regular security assessments
- Change management procedures
- Incident response planning
- Continuous monitoring

## Incident Response

### Security Incident Classification

| Severity | Description | Response Time | Escalation |
|----------|-------------|---------------|------------|
| Critical | System compromise, data breach | Immediate (<1 hour) | Executive team |
| High | Unauthorized access, service disruption | <4 hours | Security team |
| Medium | Suspicious activity, policy violation | <24 hours | IT operations |
| Low | Minor security events | <72 hours | Standard process |

### Incident Response Process

1. **Detection**: Automated alerts and monitoring
2. **Assessment**: Initial triage and impact analysis
3. **Containment**: Isolate affected systems
4. **Eradication**: Remove threats and vulnerabilities
5. **Recovery**: Restore systems and validate integrity
6. **Lessons Learned**: Post-incident review and improvements

### Incident Response Team

- **Incident Commander**: Overall coordination
- **Technical Lead**: Technical investigation
- **Communications Lead**: Stakeholder updates
- **Legal/Compliance**: Regulatory requirements
- **Recovery Lead**: System restoration

## Security Monitoring

### Security Event Logging

**Log Categories:**
- Authentication events
- Authorization failures
- Data access patterns
- Configuration changes
- System integrity checks

**Centralized Logging:**
```python
# Structured logging configuration
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### Intrusion Detection

**Indicators of Compromise (IoCs):**
- Unusual login patterns
- High-volume API calls
- Unauthorized configuration changes
- Abnormal agent behavior
- Database query anomalies

**Detection Rules:**
```python
# Example detection rule
def detect_brute_force(attempts, timeframe_minutes=5):
    recent_attempts = get_login_attempts(timeframe_minutes)
    if len(recent_attempts) > 10:
        alert_security_team("Brute force attempt detected")
        block_ip(recent_attempts[0]['ip'])
```

### Audit Trails

**Audit Requirements:**
- All data access events
- Administrative actions
- Security policy changes
- System configuration modifications

**Retention:**
- Security logs: 7 years
- Audit trails: 7 years
- System logs: 1 year

## Best Practices

### Code Security

**Secure Coding Guidelines:**
- Input validation and sanitization
- SQL injection prevention (parameterized queries)
- XSS protection (output encoding)
- CSRF protection
- Secure session management

**Code Review Checklist:**
- [ ] Authentication and authorization checks
- [ ] Input validation
- [ ] Error handling (no sensitive data leakage)
- [ ] Secure configuration
- [ ] Dependency security

### Dependency Management

**Security Scanning:**
```bash
# Dependency vulnerability scanning
pip-audit
safety check
bandit -r src/
```

**Update Process:**
- Automated dependency updates
- Security patch prioritization
- Compatibility testing
- Rollback procedures

### Access Control

**Principle of Least Privilege:**
- Users get minimum required permissions
- Regular permission reviews
- Automated deprovisioning
- Just-in-time access for elevated privileges

### Security Testing

**Testing Types:**
- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST)
- Penetration testing
- Dependency scanning

**Schedule:**
- Automated scans: Daily
- Manual penetration testing: Quarterly
- Security audits: Annually

## Configuration

### Security Settings in .env

```bash
# Authentication
JWT_SECRET_KEY=your_secure_256_bit_secret
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Encryption
DB_ENCRYPTION_KEY=your_32_byte_encryption_key
REDIS_SSL_CERT=/path/to/redis.crt
REDIS_SSL_KEY=/path/to/redis.key

# Network Security
CORS_ORIGINS=https://yourdomain.com,https://dashboard.yourdomain.com
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST=20

# Monitoring
SECURITY_LOG_LEVEL=INFO
AUDIT_LOG_RETENTION_DAYS=2555
ALERT_EMAIL_RECIPIENTS=security@yourdomain.com,admin@yourdomain.com

# Compliance
GDPR_DATA_RETENTION_DAYS=2555
HIPAA_ENCRYPTION_ENABLED=true
SOC2_AUDIT_ENABLED=true
```

### Security Checklist

**Pre-Deployment:**
- [ ] All secrets configured securely
- [ ] HTTPS certificates installed
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Security monitoring active
- [ ] Backup systems tested
- [ ] Incident response plan documented

**Post-Deployment:**
- [ ] Security scans completed
- [ ] Penetration testing performed
- [ ] Access controls verified
- [ ] Monitoring alerts configured
- [ ] Team security training completed

## Emergency Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| Security Lead | security@yourdomain.com | 24/7 |
| System Administrator | admin@yourdomain.com | Business hours |
| Incident Response | incident@yourdomain.com | 24/7 |
| Legal/Compliance | legal@yourdomain.com | Business hours |

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [GDPR Guidelines](https://gdpr-info.eu/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/guidance/index.html)

---

**Last Updated**: September 16, 2025
**Version**: 1.0
**Review Cycle**: Quarterly