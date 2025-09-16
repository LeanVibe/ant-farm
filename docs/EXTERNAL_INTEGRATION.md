# LeanVibe Agent Hive - External Integration Guide

## Overview

The LeanVibe Agent Hive supports integration with various external monitoring, logging, and analytics platforms. This guide covers setup and configuration for popular tools including Prometheus, Grafana, DataDog, and others.

## Prometheus Integration

### Metrics Export Setup

The system exposes Prometheus-compatible metrics on the `/metrics` endpoint:

```python
# FastAPI metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.core.metrics_collector import MetricsCollector

@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'leanvibe-api'
    static_configs:
      - targets: ['localhost:9001']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'leanvibe-agents'
    static_configs:
      - targets: ['localhost:9001']
    metrics_path: '/metrics/agents'
    scrape_interval: 30s

  - job_name: 'leanvibe-database'
    static_configs:
      - targets: ['localhost:5433']
    scrape_interval: 30s
```

### Available Metrics

**API Metrics:**
```
leanvibe_api_requests_total{method="GET",endpoint="/api/v1/agents",status="200"} 12543
leanvibe_api_request_duration_seconds{method="POST",endpoint="/api/v1/tasks"} 0.045
leanvibe_api_requests_in_progress 3
```

**Database Metrics:**
```
leanvibe_db_connections_active 12
leanvibe_db_connections_idle 8
leanvibe_db_query_duration_seconds{query_type="SELECT"} 0.012
leanvibe_db_connection_pool_utilization 0.65
```

**Agent Metrics:**
```
leanvibe_agents_active_total 8
leanvibe_agents_memory_usage_bytes{agent_id="architect"} 156789123
leanvibe_agents_cpu_usage_percent{agent_id="qa"} 23.5
leanvibe_agents_tasks_completed_total{agent_id="dev"} 456
```

**System Metrics:**
```
leanvibe_system_cpu_usage_percent 45.2
leanvibe_system_memory_usage_bytes 2147483648
leanvibe_system_disk_usage_percent 32.1
leanvibe_system_network_bytes_total{direction="rx"} 1073741824
```

## Grafana Integration

### Dashboard Setup

1. **Add Prometheus Data Source:**
   ```
   URL: http://localhost:9090
   Access: Browser
   ```

2. **Import Dashboard:**
   ```json
   {
     "dashboard": {
       "title": "LeanVibe Agent Hive",
       "tags": ["leanvibe", "monitoring"],
       "timezone": "browser",
       "panels": [
         {
           "title": "API Response Time",
           "type": "graph",
           "targets": [
             {
               "expr": "histogram_quantile(0.95, rate(leanvibe_api_request_duration_seconds_bucket[5m]))",
               "legendFormat": "95th percentile"
             }
           ]
         }
       ]
     }
   }
   ```

### Sample Dashboard Panels

**API Performance Panel:**
```json
{
  "title": "API Performance",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(leanvibe_api_requests_total[5m])",
      "legendFormat": "Requests/sec"
    },
    {
      "expr": "leanvibe_api_request_duration_seconds{quantile=\"0.5\"}",
      "legendFormat": "Median response time"
    }
  ]
}
```

**Agent Health Panel:**
```json
{
  "title": "Agent Health",
  "type": "table",
  "targets": [
    {
      "expr": "up{job=\"leanvibe-agents\"}",
      "legendFormat": "Status"
    },
    {
      "expr": "leanvibe_agents_memory_usage_bytes",
      "legendFormat": "Memory Usage"
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
      "title": "CPU Usage"
    },
    {
      "expr": "leanvibe_system_memory_usage_percent",
      "title": "Memory Usage"
    }
  ]
}
```

## DataDog Integration

### Agent Installation

```bash
# Install DataDog agent
DD_API_KEY=your_datadog_api_key bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"
```

### Configuration

```yaml
# /etc/datadog-agent/datadog.yaml
api_key: your_datadog_api_key
tags:
  - env:production
  - service:leanvibe

# /etc/datadog-agent/conf.d/prometheus.d/conf.yaml
instances:
  - prometheus_url: http://localhost:9001/metrics
    namespace: leanvibe
    metrics:
      - leanvibe_api_*
      - leanvibe_db_*
      - leanvibe_agents_*
      - leanvibe_system_*
```

### Custom Metrics

```python
# Send custom metrics to DataDog
from datadog import initialize, statsd

initialize(api_key='your_api_key', app_key='your_app_key')

# Send metrics
statsd.increment('leanvibe.custom_metric', tags=['env:prod'])
statsd.gauge('leanvibe.agent_performance', 95.2, tags=['agent:architect'])
```

### Dashboard Creation

**DataDog Dashboard JSON:**
```json
{
  "title": "LeanVibe Overview",
  "widgets": [
    {
      "definition": {
        "type": "timeseries",
        "requests": [
          {
            "q": "avg:leanvibe.api.response_time{*}",
            "display_type": "line"
          }
        ],
        "title": "API Response Time"
      }
    }
  ]
}
```

## ELK Stack Integration (Elasticsearch, Logstash, Kibana)

### Logstash Configuration

```conf
# logstash.conf
input {
  file {
    path => "/var/log/leanvibe/*.log"
    start_position => "beginning"
  }
  http {
    port => 8080
    codec => json
  }
}

filter {
  json {
    source => "message"
  }
  date {
    match => ["timestamp", "ISO8601"]
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "leanvibe-%{+YYYY.MM.dd}"
  }
}
```

### Elasticsearch Index Template

```json
PUT _template/leanvibe_logs
{
  "index_patterns": ["leanvibe-*"],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "timestamp": {
        "type": "date"
      },
      "level": {
        "type": "keyword"
      },
      "message": {
        "type": "text"
      },
      "agent_id": {
        "type": "keyword"
      },
      "request_id": {
        "type": "keyword"
      },
      "duration": {
        "type": "float"
      }
    }
  }
}
```

### Kibana Dashboard

**Sample Kibana Query:**
```
level:ERROR OR level:CRITICAL
```

**Visualization Configuration:**
```json
{
  "title": "Error Rate Over Time",
  "type": "line",
  "params": {
    "query": {
      "bool": {
        "filter": [
          {
            "term": {
              "level": "ERROR"
            }
          }
        ]
      }
    },
    "dateField": "@timestamp",
    "valueField": "count"
  }
}
```

## CloudWatch Integration (AWS)

### IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    }
  ]
}
```

### CloudWatch Agent Configuration

```json
# /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
{
  "metrics": {
    "namespace": "LeanVibe",
    "metrics_collected": {
      "prometheus": {
        "prometheus_config_path": "/etc/prometheus/prometheus.yml",
        "ecs_service_discovery": {
          "sd_frequency": "1m",
          "sd_target_cluster": "leanvibe-cluster",
          "sd_service_name_pattern": "^leanvibe"
        }
      }
    }
  }
}
```

### Custom Metrics Publishing

```python
# Publish metrics to CloudWatch
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='LeanVibe',
    MetricData=[
        {
            'MetricName': 'APIResponseTime',
            'Value': 45.2,
            'Unit': 'Milliseconds',
            'Timestamp': datetime.utcnow(),
            'Dimensions': [
                {
                    'Name': 'Service',
                    'Value': 'API'
                }
            ]
        }
    ]
)
```

## New Relic Integration

### APM Setup

```yaml
# newrelic.ini
[newrelic]
license_key = your_new_relic_license_key
app_name = LeanVibe Agent Hive
monitor_mode = true
```

### Custom Instrumentation

```python
# Instrument FastAPI with New Relic
import newrelic.agent

newrelic.agent.initialize('/path/to/newrelic.ini')

@app.middleware("http")
async def add_new_relic_header(request, call_next):
    response = await call_next(request)
    newrelic.agent.record_custom_event('APIRequest', {
        'method': request.method,
        'endpoint': request.url.path,
        'status_code': response.status_code,
        'duration': time.time() - request.state.start_time
    })
    return response
```

### Dashboard Configuration

**New Relic NRQL Queries:**
```sql
SELECT average(duration) FROM APIRequest WHERE method = 'GET' TIMESERIES
SELECT count(*) FROM APIRequest WHERE status_code >= 400 FACET endpoint
SELECT average(memory_usage) FROM SystemSample TIMESERIES
```

## Alerting Integrations

### Slack Integration

```python
# Slack webhook integration
import requests

def send_slack_alert(message, channel="#alerts"):
    webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    payload = {
        "channel": channel,
        "text": message,
        "username": "LeanVibe Monitor"
    }
    requests.post(webhook_url, json=payload)
```

### PagerDuty Integration

```python
# PagerDuty alert integration
import requests

def trigger_pagerduty_incident(description, severity="error"):
    routing_key = "your_pagerduty_integration_key"
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": description,
            "severity": severity,
            "source": "LeanVibe Agent Hive"
        }
    }
    requests.post("https://events.pagerduty.com/v2/enqueue", json=payload)
```

### Email Alerts

```python
# SMTP email alerts
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body, recipients):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@leanvibe.com'
    msg['To'] = ', '.join(recipients)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('your_email@gmail.com', 'your_password')
    server.sendmail('alerts@leanvibe.com', recipients, msg.as_string())
    server.quit()
```

## Webhook Integration

### Generic Webhook Setup

```python
# Generic webhook for custom integrations
import requests

def send_webhook_alert(webhook_url, alert_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(webhook_url, json=alert_data, headers=headers)
    return response.status_code == 200
```

### Webhook Payload Format

```json
{
  "alert_id": "alert_12345",
  "title": "High API Response Time",
  "description": "API response time exceeded 100ms threshold",
  "severity": "warning",
  "timestamp": "2025-09-16T10:30:00Z",
  "metrics": {
    "response_time_avg": 125.5,
    "threshold": 100.0,
    "service": "api"
  },
  "links": {
    "dashboard": "https://grafana.yourdomain.com/d/leanvibe-overview",
    "logs": "https://kibana.yourdomain.com/app/logs"
  }
}
```

## Configuration Examples

### Environment Variables

```bash
# External integrations configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

GRAFANA_ENABLED=true
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=your_grafana_api_key

DATADOG_ENABLED=true
DATADOG_API_KEY=your_datadog_api_key
DATADOG_APP_KEY=your_datadog_app_key

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
PAGERDUTY_ROUTING_KEY=your_pagerduty_key

ELASTICSEARCH_ENABLED=true
ELASTICSEARCH_URL=http://localhost:9200
KIBANA_URL=http://localhost:5601

NEW_RELIC_LICENSE_KEY=your_new_relic_key
```

### Docker Compose Integration

```yaml
# docker-compose.external.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  datadog-agent:
    image: datadog/agent:latest
    environment:
      - DD_API_KEY=${DATADOG_API_KEY}
      - DD_SITE=datadoghq.com
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock

volumes:
  grafana_data:
```

## Best Practices

### Monitoring Strategy
1. **Layered Monitoring**: Combine multiple tools for comprehensive coverage
2. **Alert Fatigue Prevention**: Set appropriate thresholds and use alert aggregation
3. **Data Retention**: Configure appropriate retention policies for cost optimization
4. **Security**: Use secure connections and API keys for all integrations

### Performance Considerations
1. **Metrics Volume**: Monitor the volume of metrics to avoid overwhelming external systems
2. **Network Latency**: Consider network latency when setting up remote monitoring
3. **Resource Usage**: Monitor the resource usage of monitoring agents themselves
4. **Backup Monitoring**: Have fallback monitoring in case primary systems fail

### Troubleshooting
1. **Connectivity Issues**: Verify network connectivity and firewall rules
2. **Authentication Problems**: Check API keys and credentials
3. **Data Format Issues**: Ensure metrics are in the correct format for each platform
4. **Rate Limiting**: Monitor for rate limiting from external services

---

**Last Updated**: September 16, 2025
**Version**: 1.0
**Supported Platforms**: Prometheus, Grafana, DataDog, ELK Stack, CloudWatch, New Relic