# LeanVibe Agent Hive - Data Retention and Backup Guide

## Overview

This document outlines the data retention policies, backup procedures, and data management practices for the LeanVibe Agent Hive system. It ensures compliance with data protection regulations while maintaining system performance and operational continuity.

## Data Classification

### Data Categories

| Category | Description | Sensitivity Level | Retention Period |
|----------|-------------|-------------------|------------------|
| **System Logs** | Application and system logs | Medium | 90 days |
| **Audit Logs** | Security and access logs | High | 7 years |
| **Metrics Data** | Performance and monitoring metrics | Medium | 1 year |
| **Agent Conversations** | Inter-agent communication logs | High | 2 years |
| **Task Data** | Task definitions and execution history | Medium | 3 years |
| **User Sessions** | Session data and context | High | 1 year |
| **Database Backups** | Full database snapshots | Critical | 7 years |
| **Configuration Files** | System and agent configurations | Medium | Indefinite |

### Data Sensitivity Levels

- **Critical**: System-critical data requiring maximum protection
- **High**: Sensitive data with compliance implications
- **Medium**: Operational data with business value
- **Low**: Temporary or ephemeral data

## Retention Policies

### System Logs

**Retention**: 90 days
**Storage**: Local filesystem with compression
**Archival**: Monthly archives to long-term storage
**Deletion**: Automatic cleanup via logrotate

```bash
# Log rotation configuration (/etc/logrotate.d/leanvibe)
/var/log/leanvibe/*.log {
    daily
    rotate 90
    compress
    missingok
    notifempty
    create 644 leanvibe leanvibe
    postrotate
        docker-compose -f /path/to/docker-compose.yml exec api kill -HUP 1
    endscript
}
```

### Audit Logs

**Retention**: 7 years (GDPR/SOC2 compliance)
**Storage**: Encrypted, tamper-proof storage
**Access**: Restricted to authorized personnel only
**Monitoring**: Automated integrity checks

**Audit Log Categories:**
- Authentication events
- Authorization decisions
- Data access events
- Configuration changes
- Security incidents
- Administrative actions

### Metrics and Monitoring Data

**Retention**: 1 year for detailed metrics, 3 years for aggregated data
**Storage**: Time-series database with compression
**Downsampling**: Automatic aggregation for long-term storage

```sql
-- Metrics retention policy
CREATE RETENTION POLICY "one_year" ON "leanvibe"."autogen"
DURATION 1y REPLICATION 1 SHARD DURATION 7d;

CREATE RETENTION POLICY "three_years" ON "leanvibe"."autogen"
DURATION 3y REPLICATION 1 SHARD DURATION 30d;
```

### Agent Conversations and Context

**Retention**: 2 years for active context, 5 years for archived
**Storage**: Vector database with encryption
**Privacy**: Automatic PII detection and masking
**Access**: Role-based access controls

### Task and Session Data

**Retention**: 3 years
**Storage**: PostgreSQL with partitioning
**Archival**: Automatic migration to archive tables
**Deletion**: GDPR-compliant deletion procedures

## Backup Procedures

### Automated Backup Strategy

#### Daily Backups
```bash
# Database backup script (backup_daily.sh)
#!/bin/bash
BACKUP_DIR="/backups/daily"
DATE=$(date +%Y%m%d_%H%M%S)
CONTAINER="leanvibe_postgres_1"

# Create backup
docker exec $CONTAINER pg_dump -U hive_user leanvibe_hive > $BACKUP_DIR/leanvibe_$DATE.sql

# Compress
gzip $BACKUP_DIR/leanvibe_$DATE.sql

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup successful: leanvibe_$DATE.sql.gz"
    # Send notification
    curl -X POST $SLACK_WEBHOOK -d "{\"text\":\"Daily backup completed successfully\"}"
else
    echo "Backup failed"
    # Send alert
    curl -X POST $SLACK_WEBHOOK -d "{\"text\":\"Daily backup FAILED\"}"
fi

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

#### Weekly Full Backups
```bash
# Weekly backup script (backup_weekly.sh)
#!/bin/bash
BACKUP_DIR="/backups/weekly"
DATE=$(date +%Y%m%d)

# Full system backup
docker run --rm \
  -v leanvibe_postgres_data:/data/postgres \
  -v leanvibe_redis_data:/data/redis \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/leanvibe_full_$DATE.tar.gz -C /data .

# Offsite upload (example with AWS S3)
aws s3 cp $BACKUP_DIR/leanvibe_full_$DATE.tar.gz s3://leanvibe-backups/weekly/

# Verify upload
if [ $? -eq 0 ]; then
    # Send success notification
    echo "Weekly backup uploaded successfully"
fi
```

#### Configuration Backups
```bash
# Configuration backup
BACKUP_CONFIG() {
    CONFIG_DIR="/opt/leanvibe/config"
    BACKUP_DIR="/backups/config"

    # Backup .env and configuration files
    tar czf $BACKUP_DIR/config_$(date +%Y%m%d).tar.gz \
        -C $CONFIG_DIR . \
        --exclude='*.log' \
        --exclude='*.tmp'

    # Encrypt sensitive files
    openssl enc -aes-256-cbc -salt \
        -in $BACKUP_DIR/config_$(date +%Y%m%d).tar.gz \
        -out $BACKUP_DIR/config_$(date +%Y%m%d).enc \
        -k $ENCRYPTION_KEY
}
```

### Backup Verification

#### Automated Verification
```bash
# Backup verification script
verify_backup() {
    BACKUP_FILE=$1

    # Check file integrity
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "Backup file not found: $BACKUP_FILE"
        return 1
    fi

    # Check file size (minimum 1MB for database)
    SIZE=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE")
    if [ $SIZE -lt 1048576 ]; then
        echo "Backup file too small: $SIZE bytes"
        return 1
    fi

    # Test database restore (sample)
    if [[ $BACKUP_FILE == *.sql.gz ]]; then
        gunzip -c $BACKUP_FILE | head -20 | grep -q "PostgreSQL database dump"
        if [ $? -ne 0 ]; then
            echo "Invalid database backup format"
            return 1
        fi
    fi

    echo "Backup verification successful"
    return 0
}
```

#### Manual Verification Steps
1. **File Integrity**: Check backup file exists and has expected size
2. **Format Validation**: Verify backup format (SQL, tar.gz, etc.)
3. **Test Restore**: Perform test restore to temporary environment
4. **Data Consistency**: Validate restored data integrity
5. **Access Verification**: Ensure restored system functions correctly

## Data Deletion Procedures

### GDPR Right to Erasure

#### Automated Deletion
```python
# GDPR deletion request handler
def process_deletion_request(user_id, data_categories=None):
    """
    Process GDPR deletion request for specified user data
    """
    if data_categories is None:
        data_categories = ['conversations', 'sessions', 'tasks', 'metrics']

    deletion_log = []

    for category in data_categories:
        if category == 'conversations':
            # Delete user conversations
            deleted_count = delete_user_conversations(user_id)
            deletion_log.append(f"Deleted {deleted_count} conversations")

        elif category == 'sessions':
            # Delete user sessions
            deleted_count = delete_user_sessions(user_id)
            deletion_log.append(f"Deleted {deleted_count} sessions")

        elif category == 'tasks':
            # Delete user tasks
            deleted_count = delete_user_tasks(user_id)
            deletion_log.append(f"Deleted {deleted_count} tasks")

        elif category == 'metrics':
            # Anonymize user metrics
            anonymized_count = anonymize_user_metrics(user_id)
            deletion_log.append(f"Anonymized {anonymized_count} metrics")

    # Log deletion event
    audit_log('DATA_DELETION', {
        'user_id': user_id,
        'categories': data_categories,
        'timestamp': datetime.utcnow(),
        'deleted_by': 'SYSTEM'
    })

    return deletion_log
```

#### Manual Deletion Procedures

**Step 1: Verify Deletion Request**
- Confirm user identity and authorization
- Document deletion scope and justification
- Create audit trail entry

**Step 2: Execute Deletion**
```sql
-- Safe deletion with logging
BEGIN;

-- Log deletion intent
INSERT INTO deletion_log (user_id, deletion_type, requested_at, executed_by)
VALUES ($1, $2, NOW(), $3);

-- Delete user data
DELETE FROM conversations WHERE user_id = $1;
DELETE FROM sessions WHERE user_id = $1;
DELETE FROM tasks WHERE user_id = $1;

-- Anonymize metrics (GDPR compliance)
UPDATE metrics SET user_id = NULL, user_hash = $4 WHERE user_id = $1;

COMMIT;
```

**Step 3: Verification**
- Confirm data deletion from all systems
- Update audit logs
- Notify user of completion
- Schedule follow-up verification

### Data Retention Enforcement

#### Automated Cleanup Jobs
```python
# Scheduled cleanup job
@scheduler.scheduled_job('cron', day='*', hour=2)  # Daily at 2 AM
def daily_data_cleanup():
    """
    Perform daily data cleanup according to retention policies
    """

    # Clean old system logs
    cleanup_old_logs(days=90)

    # Archive old metrics
    archive_old_metrics(days=365)

    # Delete expired sessions
    delete_expired_sessions(days=30)

    # Compress old audit logs
    compress_old_audit_logs(days=180)

    # Update retention counters
    update_retention_stats()

def cleanup_old_logs(retention_days):
    """Clean up old log files"""
    log_dir = Path('/var/log/leanvibe')
    cutoff_date = datetime.now() - timedelta(days=retention_days)

    for log_file in log_dir.glob('*.log.*'):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            log_file.unlink()

def archive_old_metrics(retention_days):
    """Archive old metrics data"""
    # Move old metrics to archive table
    archive_query = """
    INSERT INTO metrics_archive
    SELECT * FROM metrics
    WHERE created_at < NOW() - INTERVAL '%s days'
    """
    db.execute(archive_query, (retention_days,))

    # Delete from main table
    delete_query = """
    DELETE FROM metrics
    WHERE created_at < NOW() - INTERVAL '%s days'
    """
    db.execute(delete_query, (retention_days,))
```

## Recovery Procedures

### Database Recovery

#### Point-in-Time Recovery
```bash
# Restore to specific point in time
RECOVERY_TARGET_TIME="2024-01-15 14:30:00"

# Stop application
docker-compose down

# Restore base backup
docker exec -i leanvibe_postgres_1 psql -U hive_user leanvibe_hive < /backups/base_backup.sql

# Apply WAL files until target time
docker exec leanvibe_postgres_1 pg_wal_replay --until="$RECOVERY_TARGET_TIME"

# Start application
docker-compose up -d
```

#### Full System Recovery
```bash
# Complete disaster recovery
#!/bin/bash

# 1. Stop all services
docker-compose down

# 2. Restore database
docker run --rm \
  -v leanvibe_postgres_data:/var/lib/postgresql/data \
  -v /backups:/backups \
  postgres:15 \
  bash -c "dropdb -U hive_user leanvibe_hive && createdb -U hive_user leanvibe_hive && psql -U hive_user leanvibe_hive < /backups/latest.sql"

# 3. Restore Redis data
docker run --rm \
  -v leanvibe_redis_data:/data \
  -v /backups:/backups \
  alpine tar xzf /backups/redis_backup.tar.gz -C /data

# 4. Restore configuration
tar xzf /backups/config_backup.tar.gz -C /opt/leanvibe

# 5. Start services
docker-compose up -d

# 6. Verify recovery
curl http://localhost:9001/api/v1/health
```

### Data Integrity Verification

#### Post-Recovery Checks
```bash
# Verify database integrity
docker exec leanvibe_postgres_1 psql -U hive_user leanvibe_hive -c "SELECT COUNT(*) FROM agents;"
docker exec leanvibe_postgres_1 psql -U hive_user leanvibe_hive -c "SELECT COUNT(*) FROM tasks;"

# Check Redis data
docker exec leanvibe_redis_1 redis-cli KEYS "*"

# Verify API functionality
curl http://localhost:9001/api/v1/agents
curl http://localhost:9001/api/v1/monitoring/health

# Test agent spawning
curl -X POST http://localhost:9001/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "test", "agent_name": "recovery_test"}'
```

## Compliance Considerations

### GDPR Compliance

**Data Protection Measures:**
- Data minimization principles
- Purpose limitation
- Storage limitation
- Accuracy requirements
- Integrity and confidentiality
- Accountability

**Retention Schedule:**
- Personal data: Minimum necessary period
- Consent records: 7 years
- Audit logs: 7 years
- System logs: 90 days

### HIPAA Compliance (Healthcare Data)

**Protected Health Information (PHI):**
- Patient conversation data
- Medical context information
- Health-related task data
- PHI audit trails

**Security Measures:**
- PHI encryption at rest and in transit
- Access controls with audit logging
- Business Associate Agreements (BAA)
- Incident response procedures

### SOC 2 Compliance

**Trust Service Criteria:**
- Security: Data protection and access controls
- Availability: Backup and recovery procedures
- Processing Integrity: Data accuracy and processing
- Confidentiality: Sensitive data protection
- Privacy: Personal information handling

## Monitoring and Alerts

### Backup Monitoring
```yaml
# Prometheus alerting rules for backups
groups:
  - name: backup_alerts
    rules:
      - alert: BackupFailed
        expr: leanvibe_backup_status{job="daily_backup"} == 0
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Daily backup failed"
          description: "The daily backup job has failed"

      - alert: BackupSizeAnomaly
        expr: rate(leanvibe_backup_size_bytes[1h]) < 0.5 * rate(leanvibe_backup_size_bytes[24h])
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Backup size anomaly detected"
          description: "Backup size has decreased significantly"
```

### Retention Compliance Monitoring
```python
# Compliance monitoring job
def check_retention_compliance():
    """
    Verify data retention policies are being followed
    """
    violations = []

    # Check audit log retention
    old_audit_logs = db.query("""
        SELECT COUNT(*) FROM audit_logs
        WHERE created_at < NOW() - INTERVAL '7 years'
    """)

    if old_audit_logs > 0:
        violations.append(f"Found {old_audit_logs} audit logs older than 7 years")

    # Check metrics retention
    old_metrics = db.query("""
        SELECT COUNT(*) FROM metrics
        WHERE created_at < NOW() - INTERVAL '1 year'
    """)

    if old_metrics > 0:
        violations.append(f"Found {old_metrics} metrics older than 1 year")

    # Report violations
    if violations:
        send_compliance_alert("Retention Policy Violations", violations)

    return violations
```

## Archive Storage

### Long-term Archive Strategy

#### Archive Storage Options
1. **Cloud Storage**: AWS S3, Google Cloud Storage, Azure Blob Storage
2. **On-premises NAS**: Network-attached storage with redundancy
3. **Tape Backup**: For regulatory archives requiring long-term retention

#### Archive Procedures
```bash
# Monthly archive job
archive_old_data() {
    YEAR_MONTH=$(date +%Y%m)

    # Archive old conversations
    docker exec leanvibe_postgres_1 pg_dump -U hive_user \
        -t conversations \
        --where "created_at < NOW() - INTERVAL '2 years'" \
        leanvibe_hive > /archives/conversations_$YEAR_MONTH.sql

    # Archive old metrics
    docker exec leanvibe_influxdb_1 influx -execute \
        "SELECT * FROM metrics WHERE time < now() - 1y" \
        -database leanvibe > /archives/metrics_$YEAR_MONTH.txt

    # Compress and encrypt archives
    tar czf /archives/$YEAR_MONTH.tar.gz /archives/*_$YEAR_MONTH.*
    openssl enc -aes-256-cbc -salt \
        -in /archives/$YEAR_MONTH.tar.gz \
        -out /archives/$YEAR_MONTH.enc \
        -k $ARCHIVE_ENCRYPTION_KEY

    # Upload to cloud storage
    aws s3 cp /archives/$YEAR_MONTH.enc s3://leanvibe-archives/$YEAR_MONTH/

    # Cleanup local archives
    rm /archives/*_$YEAR_MONTH.*
}
```

### Archive Access Procedures

**Archive Retrieval:**
1. Request archive access through ticketing system
2. Verify authorization and business justification
3. Retrieve from cloud storage
4. Decrypt and extract archive
5. Provide temporary access to authorized personnel
6. Log all access events
7. Re-encrypt and re-archive after use

## Summary

This data retention and backup guide ensures:

- **Compliance**: GDPR, HIPAA, and SOC 2 compliance
- **Data Protection**: Encryption and access controls
- **Operational Continuity**: Reliable backup and recovery procedures
- **Cost Efficiency**: Optimized storage and retention policies
- **Auditability**: Comprehensive logging and monitoring

Regular reviews of retention policies and backup procedures are recommended to ensure continued compliance and effectiveness.

---

**Last Updated**: September 16, 2025
**Version**: 1.0
**Review Cycle**: Quarterly
**Retention Policy Review**: Annual