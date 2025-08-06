# PRD: Real-Time Observability & Hooks

> **--- ARCHIVED DOCUMENT ---**
> **This document is historical and no longer maintained.**
> **The authoritative source is now [docs/prd/observability-system.md](/docs/prd/observability-system.md).**

## Executive Summary

The Observability & Hooks subsystem delivers full-stack visibility into LeanVibe Agent Hive with hook-based event interception, structured logging, distributed tracing, and WebSocket streaming dashboards. Inspired by Claude Code hooks, it allows deterministic lifecycle event tracking (PreToolUse, PostToolUse, Notification, Stop, SubAgentStop) and granular performance metrics for debugging and optimization.

## Problem Statement

Agents operate autonomously, making it difficult to understand failures, performance bottlenecks, or security issues. Existing logging is ad-hoc and lacks real-time visibility. Developers need a robust observability stack to:
- **Trace agent actions** across tools and sub-agents
- **Diagnose errors** quickly with context-rich logs
- **Measure performance** (latency, token usage, retries)
- **Audit security** via deterministic event hooks

## Success Metrics

| KPI | Target |
|---|---|
| Hook coverage | 100% lifecycle events captured |
| Event latency (P95) | <150 ms from emit to storage |
| Dashboard refresh rate | <1 s |
| Error detection MTTR | <5 min |
| Performance overhead | <3% CPU per agent |

## User Stories

1. **As a developer**, I can view real-time agent tool usage and outputs on a dashboard.
2. **As a security auditor**, I can receive alerts when a `PreToolUse` hook blocks a dangerous command.
3. **As an SRE**, I can query historical chat transcripts to diagnose anomalies.
4. **As a PM agent**, I can filter events by session and color-coded agent role.

## Technical Requirements

### Architecture
- **Hook Scripts**: Bash/Python scripts triggered by Claude Code events
- **Event Collector**: Bun/TypeScript HTTP + WebSocket server (`/events`, `/stream`)
- **Database**: PostgreSQL (events) + pgvector (embeddings for log search)
- **Message Broker**: Redis Streams for decoupled ingestion
- **Dashboard**: Vue 3 + Vite SPA
- **Metrics Exporter**: Prometheus + Grafana panels

### Event Schema
```json
{
  "id": "uuid",
  "session_id": "uuid",
  "agent_id": "agent-123",
  "event_type": "PreToolUse|PostToolUse|...",
  "tool": "Write",
  "status": "success|blocked|error",
  "payload": {},
  "timestamp": 1699999999
}
```

### Hook Implementation Example (PreToolUse)
```bash
#!/usr/bin/env bash
set -euo pipefail
TOOL="$CLAUDE_TOOL_NAME"
if [[ "$TOOL" == "Bash" && "$CLAUDE_TOOL_INPUT" =~ "rm -rf /" ]]; then
  echo "Dangerous command blocked" >&2
  exit 2  # Block tool
fi

# Send event
jq -n --arg tool "$TOOL" --arg event "PreToolUse" '{tool:$tool,event_type:$event,timestamp:(now|floor)}' | \
  curl -X POST -H 'Content-Type: application/json' -d @- http://observability-server:7070/events
```

### API Endpoints
- `POST /events` — ingest event (bulk supported)
- `GET /events/recent?session=uuid` — recent events
- `WS /stream?filter=session:uuid` — real-time stream

### Dashboard Features
- Live system timeline with session color coding
- Token usage heatmap per agent
- Error overlay with stack traces and payload excerpt
- Histogram of tool latencies
- Filter chips: event type, agent role, timeframe

## Implementation Strategy

| Phase | Duration | Deliverables |
|---|---|---|
| 1. Hook MVP | 1 wk | PreToolUse & PostToolUse hooks, event collector |
| 2. DB + Ingestion | 1 wk | PostgreSQL schema, Redis buffer, bulk API |
| 3. Dashboard v1 | 1 wk | Live timeline, filtering, color coding |
| 4. Metrics & Alerts | 1 wk | Prometheus exporter, Grafana panels, Alertmanager rules |
| 5. Embeddings Search | 1 wk | pgvector similarity search, chat transcript viewer |

## Testing Strategy
- **Unit**: Hook exit codes, schema validation, DB inserts
- **Integration**: Simulated agent run with 1k events/sec
- **Chaos**: Kill collector pod; ensure Redis buffer drains on restart
- **Load**: WebSocket fan-out to 100 dashboard clients

## Risks & Mitigations
| Risk | Mitigation |
|---|---|
| High event volume overload | Redis buffer + batch inserts |
| Sensitive data exposure | PII redaction hook stage, RBAC on dashboard |
| Hook misconfiguration blocking agents | Canary hooks in non-blocking mode first |

## Dependencies
- Agent Orchestrator Core
- PostgreSQL with pgvector
- Redis ≥7.2
- Prometheus stack

## Acceptance Criteria
- All lifecycle events visible in dashboard within 1 s
- Prometheus metrics expose event_rate, error_rate
- Hooks block dangerous commands in e2e tests
- Performance overhead <3% CPU on benchmark

## Definition of Done
1. Event flow from hook → DB verified end-to-end
2. Grafana dashboard with token usage and error panels
3. CI tests (>90% coverage) passing
4. Security review completed with no critical findings
