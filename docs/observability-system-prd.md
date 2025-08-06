# PRD: Observability & Monitoring System

> **--- ARCHIVED DOCUMENT ---**
> **This document is historical and no longer maintained.**
> **The authoritative source is now [docs/prd/observability-system.md](/docs/prd/observability-system.md).**
**Priority**: Must-Have (Phase 1) | **Estimated Effort**: 3-4 weeks | **Technical Complexity**: Medium–High

## Executive Summary
A real-time observability platform providing full visibility into agent lifecycle events, performance metrics, and system health. Inspired by Claude Code Hooks, SuperClaude, and industry best practices, it captures and streams event data via Redis Streams and stores structured logs in PostgreSQL for analysis. Grafana dashboards offer live visibility, while alerting rules detect anomalies[23][27][31].

## Problem Statement
As the Agent Hive scales, debugging, optimizing, and ensuring reliability require end-to-end visibility across hundreds of concurrent agents. Current gaps include:
- No unified event stream across agents and tools
- Limited insight into performance bottlenecks and failures
- Manual log inspection with no dashboards or alerts
- Lack of structured audit trails for lifecycle hooks

## Success Metrics
- **Event capture coverage**: 100% lifecycle hooks instrumented
- **Dashboard latency**: <2 seconds lag from event to visualization
- **Mean Time To Detect (MTTD)**: <1 minute for critical failures
- **Alert false-positive rate**: <5%
- **Retention compliance**: 30-day searchable logs with <1 % data loss

## Technical Requirements

### Core Components
1. **Hook Interceptor** – Captures PreToolUse, PostToolUse, Notification, Stop, SubagentStop events and publishes to Redis Streams.
2. **Event Stream Broker** – Redis Streams with consumer groups; supports at-least-once delivery.
3. **Event Processor** – Async FastAPI workers persisting events to PostgreSQL (JSONB) and emitting Prometheus metrics.
4. **Metrics & Dashboards** – Prometheus + Grafana with colour-coded timelines, session filters, and drill-downs.
5. **Alerting Engine** – Grafana Alerting or Alertmanager rules for latency, error-rate, and throughput thresholds.
6. **Transcript Storage** – Optional S3/MinIO bucket for chat logs with indices referencing DB rows.

### API Specifications
```
POST /observability/event
{
  "event_type": "PreToolUse|PostToolUse|Notification|Stop|SubagentStop",
  "agent_id": "uuid",
  "session_id": "uuid",
  "payload": { ... },
  "timestamp": "iso8601"
}
Response: {"status": "queued"}

GET /observability/events?session_id={uuid}&from={ts}&to={ts}&type=PostToolUse
Response: {"events": [...]}  # Paginated

GET /observability/metrics
Response: Prometheus text format
```

### Database Schema
```sql
CREATE TABLE agent_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_events_session ON agent_events(session_id);
CREATE INDEX idx_events_type_time ON agent_events(event_type, created_at);

CREATE TABLE chat_transcripts (
    id UUID PRIMARY KEY,
    session_id UUID,
    agent_id UUID,
    s3_key VARCHAR(500),
    size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## User Stories & Acceptance Tests

### Story 1: Real-Time Event Streaming
**As** an operator  **I want** to see live agent events  **So that** I can detect anomalies quickly.
```python
# Test: Event published appears on stream within 1s
start = time.time()
publish_event(event)
msg = redis.xread(streams={"agent_events": 0}, block=2000)
assert msg is not None and time.time() - start < 1.0
```

### Story 2: Grafana Timeline Visualisation
**As** a developer **I want** colour-coded timelines per session **So that** I can trace tool calls and durations.
```python
def test_timeline_dashboard():
    dash = grafana.get_dashboard("session-timeline")
    assert dash is not None
    assert dash.variables["session_id"].query == "agent_events"
```

### Story 3: Alert on Error Spike
```python
# Simulate 5 failed PostToolUse events in 30s
for _ in range(5):
    publish_event({"event_type": "PostToolUse", "payload": {"success": False}})
# Grafana alert should trigger
assert alertmanager.get_active_alerts(name="ToolFailureSpike")
```

## Implementation Phases
1. **Week 1** – Hook instrumentation & Redis Streams broker.
2. **Week 2** – Event processor, PostgreSQL persistence, Prometheus metrics.
3. **Week 3** – Grafana dashboards, alert rules, transcript storage.
4. **Week 4** – Load testing, data retention policies, documentation.

## Security Considerations
- TLS for Redis & API endpoints.
- Signed event payloads to prevent tampering.
- RBAC on Grafana & API.

## Risks & Mitigations
- **High ingest rate** – Use Redis sharding, back-pressure and batch inserts.
- **Dashboard overload** – Pre-aggregate metrics; limit panel queries.

This PRD delivers a production-grade observability pipeline giving developers and operators deep visibility into Agent Hive behaviour and performance.