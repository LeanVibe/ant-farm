# PRD: Agent Communication System

## Executive Summary

The Agent Communication System provides reliable, low-latency message passing between LeanVibe agents using Redis Streams and Pub/Sub. It delivers at-least-once delivery guarantees, persistence, and back-pressure handling, enabling scalable real-time collaboration across agents and services.

## Problem Statement

Existing tmux-based message passing is ad-hoc, lacks durability, and fails under load. Agents need a fault-tolerant queue that:
- Ensures reliable delivery even if consumers crash
- Supports ordered, traceable messages
- Handles burst traffic without message loss
- Allows filtering and consumer groups for horizontal scaling

## Success Metrics

| KPI | Target |
|---|---|
| Message delivery success | >99.9% |
| End-to-end latency (P95) | <200 ms |
| Max sustained throughput | ≥10 k msgs/sec |
| Queue durability | 24 h retention with zero loss |
| Mean time to recovery | <30 s |

## User Stories

1. **As a sub-agent**, I publish task results and status updates so other agents can consume them.
2. **As the orchestrator**, I broadcast workflow commands to all listening agents and expect acknowledgements.
3. **As a DevOps engineer**, I can replay messages for debugging within 24 h.

## Technical Requirements

### Architecture
- **Redis Streams** (`XADD`, `XREADGROUP`) for durable queues
- **Consumer Groups** per agent type (e.g., `architects`, `security`)
- **Pub/Sub Channels** for fire-and-forget notifications
- **Message Schema** (JSON):
  ```json
  {
    "id": "uuid",
    "from": "agent_id",
    "to": "agent_id|broadcast",
    "type": "task_request|task_result|event",
    "payload": {},
    "timestamp": 1699999999
  }
  ```
- **Ack Flow**: `XACK` after successful processing; failed consumers move messages to a Dead-Letter Stream (`DLQ`)
- **Back-pressure**: Max length trimming (`MAXLEN ~ 1M`) and consumer lag monitoring

### Interfaces
- **Python SDK**: `send_message()`, `stream_consume()`
- **FastAPI Endpoints** for diagnostics: `GET /queue/lag`, `POST /queue/replay`

### Security
- Redis AUTH & TLS
- Message signing (HMAC) optional
- RBAC: Agents restricted to specific streams

## Implementation Strategy

| Phase | Duration | Deliverables |
|---|---|---|
| 1. Prototype | 1 week | Redis instance, Python SDK, basic XADD/XREAD |
| 2. Consumer Groups | 1 week | Auto-claim stalled messages, lag metrics |
| 3. Monitoring & Back-pressure | 1 week | Prometheus exporter, alert rules |
| 4. DLQ & Replay Tools | 1 week | CLI + API for replay |
| 5. Hardening & Benchmarks | 1 week | Load tests ≥10 k msgs/sec |

## Testing Strategy
- **Unit**: Schema validation, serialization
- **Integration**: Produce/consume across multiple agents, failover tests
- **Load**: Locust script pushing 100 k msgs/min
- **Chaos**: Kill consumer pods; verify `XCLAIM` recovery

## Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Redis single-point failure | Use Redis Cluster or Sentinel, RDB+AOF persistence |
| Consumer lag under spikes | Auto-scale worker pods, back-pressure alerts |
| Message schema drift | Version field + contract tests |

## Dependencies
- Redis ≥7.2 with Streams enabled
- Kubernetes or Docker-Compose for deployment
- Prometheus + Grafana for metrics

## Acceptance Criteria
- All success KPIs met under load test
- Automatic recovery demonstrated via chaos test
- End-to-end tests green in CI

## Definition of Done
1. Code merged to `main` with >90% test coverage
2. Helm/Docker deploy manifests published
3. Runbook & API docs completed
4. Grafana dashboard imported to monitoring stack
