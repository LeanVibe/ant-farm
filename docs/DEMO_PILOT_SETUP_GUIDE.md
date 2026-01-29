# Ant Farm - Demo/Pilot Setup Guide

**Generated:** 2026-01-20T02:51:46.224541
**Project:** Ant Farm
**Location:** `leanvibe-dev/ant-farm/`

---

## Overview

This guide helps you set up a demo environment and pilot program for Ant Farm's ADW (Autonomous Development Workflow) system.

---

## Prerequisites

### Required
- Python 3.12+
- Docker & Docker Compose
- PostgreSQL (via Docker)
- Redis (via Docker)
- At least one CLI tool: opencode, Claude CLI, or Gemini CLI

### Recommended
- All three CLI tools for maximum flexibility
- Monitoring setup (Prometheus/Grafana optional)

---

## Demo Environment Setup

### Step 1: Install CLI Tools

**opencode (Recommended):**
```bash
curl -fsSL https://opencode.ai/install | bash
```

**Claude CLI:**
```bash
# Install from https://claude.ai/cli
```

**Gemini CLI:**
```bash
# Install from https://ai.google.dev/gemini-api/docs/cli
```

### Step 2: Start Infrastructure
```bash
cd leanvibe-dev/ant-farm
docker-compose up -d postgres redis
```

### Step 3: Initialize System
```bash
# Check available CLI tools
hive agent tools

# Initialize system
hive init

# Start the system
hive system start
```

### Step 4: Enable ADW
```bash
# Enable autonomous development
hive agent bootstrap

# Spawn test agents
hive agent spawn meta
hive agent spawn architect
hive agent spawn qa
```

### Step 5: Run Test Session
```bash
# Submit a test task
hive task submit "Implement a simple feature X"

# Monitor ADW system
# Dashboard: http://localhost:9001/dashboard
# API: http://localhost:9001/api/v1/health
```

---

## Pilot Program Structure

### Phase 1: Setup (Week 1)
- [ ] Recruit 1-2 engineering teams
- [ ] Set up pilot environment
- [ ] Configure CLI tools
- [ ] Train teams on ADW system

### Phase 2: Initial Sessions (Week 2-3)
- [ ] Run 3+ test ADW sessions (1-2 hours each)
- [ ] Monitor safety controls
- [ ] Collect feedback
- [ ] Document issues

### Phase 3: Extended Sessions (Week 4+)
- [ ] Run 16+ hour ADW sessions (if validated)
- [ ] Monitor cognitive load
- [ ] Verify autonomous operation
- [ ] Collect success metrics

---

## Monitoring & Metrics

### Key Metrics to Track
- **Session Duration:** Target 16-24 hours
- **Safety Interventions:** Should be minimal
- **Cognitive Load:** Should stay within thresholds
- **Task Completion:** Success rate
- **Agent Utilization:** Resource efficiency

### Monitoring Endpoints
- **Dashboard:** http://localhost:9001/dashboard
- **API Health:** http://localhost:9001/api/v1/health
- **System Status:** `hive system status`

---

## Safety Controls

### 5-Level Intervention System
1. **Level 1:** Warning (log only)
2. **Level 2:** Alert (notify)
3. **Level 3:** Pause (temporary halt)
4. **Level 4:** Rollback (revert changes)
5. **Level 5:** Terminate (full stop)

### Emergency Controls
- **API Endpoint:** Emergency stop via API
- **CLI Command:** `hive system stop`
- **Dashboard:** Emergency controls in UI

---

## Troubleshooting

### Common Issues

1. **ADW Session Fails to Start**
   - Check CLI tools are installed
   - Verify infrastructure (PostgreSQL, Redis)
   - Check system logs

2. **Safety Controls Not Working**
   - Verify safety directory exists
   - Check emergency intervention code
   - Test rollback mechanism

3. **Multi-CLI Not Detected**
   - Install at least one CLI tool
   - Verify PATH includes CLI tools
   - Check CLI detection code

---

## Success Criteria

### For Demo
- ✅ ADW session initializes successfully
- ✅ Safety controls respond correctly
- ✅ Multi-CLI support works
- ✅ System runs for 1-2 hours without issues

### For Pilot
- ✅ 16+ hour sessions complete successfully
- ✅ Safety interventions are rare (<5% of sessions)
- ✅ Cognitive load stays within thresholds
- ✅ Teams provide positive feedback

---

**Last Updated:** 2026-01-20T02:51:46.224547
