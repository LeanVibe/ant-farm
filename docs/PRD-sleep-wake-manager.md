# PRD: Sleep-Wake Manager – Asynchronous Consolidation & Recovery

## Executive Summary

The Sleep-Wake Manager orchestrates background **consolidation cycles** ("sleep") and foreground **active work cycles** ("wake") for every agent in LeanVibe Agent Hive 2.0. During sleep, agents off-load high-cost reasoning—summarization, refactoring, large-scale searches—to spare compute, producing condensed artifacts that turbo-charge next-day performance. On wake, agents seamlessly resume tasks with refreshed context snapshots, ensuring 24/7 autonomy while minimizing resource waste.

## Problem Statement

Agents currently run continuously, consuming tokens and compute even when idle or blocked by long-running tasks. This leads to:
- **Compute Waste**: >40 % of billable LLM tokens happen during low-value idle chatter
- **Context Drift**: Memory grows without structure, causing retrieval noise
- **Crash Recovery Gaps**: Manual intervention required after tmux or host restarts
- **Nocturnal Bottlenecks**: High-latency tasks block real-time interactions

## Success Metrics

| Metric | Target |
|--------|--------|
|LLM Token Reduction|≥55 % per 24 h cycle|
|Average Latency Improvement|≥40 % faster first-token time post-wake|
|Crash Recovery Time|<60 s full state restore|
|Consolidation Accuracy|≥95 % important-fact retention (manual review)|
|Background Utilization|≥70 % of off-peak CPU allocated to batch jobs|

## Core Features

### 1. Scheduled Sleep Cycles
Configurable cron-like scheduler triggers sleep windows (default 02:00–04:00 UTC). Supports manual "nap" command for ad-hoc consolidation.

### 2. Consolidation Pipeline
Multi-stage reducer that:
1. **Snapshot** live task queues → Git checkpoint
2. **Context Compression** via Claude summarizer
3. **Vector Index Update** (Context Engine)
4. **Performance Audit** logs → Grafana panel

### 3. Safe-State Checkpointing
Atomic save of agent registries, Redis stream offsets, and Postgres transactions—written to `/var/lib/hive/checkpoints/<ts>.tar.zst` with SHA-256 hashes.

### 4. Wake Restoration
Idempotent bootstrap script loads latest valid checkpoint, repopulates Redis Streams, restarts tmux panes, and re-hydrates in-memory caches.

### 5. Recovery & Fallback Logic
If checkpoint fails validation, fallback to previous checkpoint (max 3 generations) and alert human via mobile push.

## Technical Architecture

```
+--------------+              +-------------------+
|   Orchestrator|  sleep cmd→ | Sleep-Wake Manager|
+------+-------+              +---------+---------+
       |                                 |
       | Cron Trigger (2 AM UTC)         |
       v                                 v
+--------------+              +-------------------+
|   Agents     | — batch →   | Consolidation Jobs |
+--------------+              +-------------------+
       ^                                 |
       | wake cmd                        |
       |                                 v
+--------------+              +-------------------+
| Redis Streams| ← restore — | State Rehydrator  |
+--------------+              +-------------------+
```

### Core Modules
- **SchedulerService** – APScheduler wrapper; reads `sleep_windows` table
- **SnapshotService** – Git + Postgres + Redis exporter
- **Consolidator** – Uses Context Engine API to compress histories
- **Rehydrator** – Validates checksum, restores services, publishes `WAKE` event

### Database Tables
```sql
CREATE TABLE sleep_windows (
  id SERIAL PRIMARY KEY,
  agent_id UUID REFERENCES agents(id),
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  timezone VARCHAR(64) DEFAULT 'UTC',
  active BOOLEAN DEFAULT TRUE
);

CREATE TABLE checkpoints (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id UUID,
  created_at TIMESTAMP DEFAULT NOW(),
  path TEXT NOT NULL,
  sha256 CHAR(64),
  size_bytes BIGINT,
  is_valid BOOLEAN DEFAULT FALSE,
  validation_errors JSONB
);
```

### API Surface (FastAPI)
```python
@router.post("/agents/{agent_id}/sleep")
async def force_sleep(agent_id: UUID):
    """Put agent into immediate sleep state."""

@router.post("/agents/{agent_id}/wake")
async def force_wake(agent_id: UUID):
    """Wake agent and restore from latest checkpoint."""

@router.get("/checkpoints/{checkpoint_id}")
async def get_checkpoint_meta(checkpoint_id: UUID) -> CheckpointMeta:
    pass
```

## Implementation Plan

| Sprint | Goals | Key Deliverables |
|--------|-------|------------------|
|Week 1|MVP Scheduler & Snapshot|`SnapshotService`, Git+DB dumps, unit tests|
|Week 2|Consolidation + Checkpoint DB|`Consolidator`, `checkpoints` schema, integration tests|
|Week 3|Rehydrator & Crash Recovery|tmux revive script, Redis offset restore|
|Week 4|Metrics & Alerting|Prometheus exporters, Grafana panels, push alerts|

### Sample Tests (PyTest)
```python
def test_snapshot_creation(tmp_path):
    cp = snapshot_service.create(agent_id)
    assert cp.is_valid
    assert Path(cp.path).exists()

@pytest.mark.asyncio
async def test_wake_restores_queues():
    await sleep_manager.force_sleep(agent_id)
    # Simulate crash by clearing Redis
    redis.flushall()
    await sleep_manager.force_wake(agent_id)
    assert redis.xlen(f"agent:{agent_id}:queue") > 0
```

## Risk Assessment & Mitigation
| Risk | Impact | Mitigation |
|------|--------|-----------|
|Checkpoint Corruption|Agents start with stale state|Redundant off-site backups + checksum validation|
|Race Conditions|Tasks enqueue during snapshot|Use Redis `MULTI` block + agent quiescence flag|
|LLM Quota Spikes|Bulk summarization may exceed monthly quota|Rate-limit consolidator, stagger jobs|

## Dependencies
- Context Engine API for vector updates
- Git-based Checkpoint System
- Observability stack (Prometheus/Grafana)

## Acceptance Criteria
- Automated sleep at configured window
- Complete checkpoint created in under 120 s for 1 GB state
- Successful restore passes integration tests
- Alert on any checkpoint failure within 30 s

## Future Enhancements
- **Adaptive Scheduling**: ML model predicts optimal sleep periods based on workload
- **Parallel Consolidation**: Shard long histories across workers for faster summarization
- **Cold Storage Tiering**: Archive >90-day checkpoints to S3 Glacier
- **Self-Healing**: Auto-rerun failed consolidations with back-off
