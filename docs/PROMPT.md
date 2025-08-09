# LeanVibe Agent Hive 2.0 - OpenCode Handoff Prompt

## 🎯 IMMEDIATE PRIORITY: Fix Critical TaskQueue Serialization Bug

You are taking over development of LeanVibe Agent Hive 2.0, a self-improving autonomous multi-agent system. **The system is 95% operational but has ONE CRITICAL BUG blocking all integration tests.**

### Current System Status: OPERATIONAL BUT BLOCKED
- ✅ **Infrastructure**: All services running (API:9002, Redis:6381, PostgreSQL:5433)
- ✅ **Core Components**: TaskQueue, MessageBroker, Cache systems implemented
- ✅ **Test Framework**: Comprehensive unit and integration tests created
- 🔥 **BLOCKER**: TaskQueue Redis serialization failing with "keywords must be strings" error

### The Critical Bug You Must Fix FIRST

**Location**: `src/core/task_queue.py:255` in `_dict_to_task()` method
**Error**: `TypeError: keywords must be strings` when creating Task objects from Redis hash data
**Impact**: ALL integration tests failing (4/5 tests), system cannot retrieve tasks after submission

**Test Command to Reproduce**:
```bash
pytest tests/integration/test_core_system_integration.py::TestCoreSystemIntegration::test_complete_task_workflow -v -s
```

**Error Pattern**:
1. Task submitted successfully to Redis ✅
2. Task data stored in Redis hash ✅ 
3. `get_task()` calls `_dict_to_task()` ❌
4. Task creation fails with "keywords must be strings" ❌
5. Returns `None`, all tests fail ❌

**Root Cause**: The `_dict_to_task()` method at line 284 calls `Task(**data)` but Redis returns field names as bytes or invalid types that can't be used as Python keyword arguments.

### Your First Task (Critical Priority)

1. **Debug the serialization issue**:
   - Check what Redis actually returns in `task_data` dict
   - Identify non-string keys causing the keyword argument error
   - Fix field name conversion from Redis hash format

2. **Fix the `_dict_to_task()` method**:
   - Ensure all dict keys are proper Python strings
   - Validate field types match Task model expectations
   - Handle edge cases in Redis data format

3. **Validate the fix**:
   - Run integration test to confirm it passes
   - Verify task retrieval works end-to-end
   - Check all other tests still pass

### System Architecture Overview

**Technology Stack**:
- Python 3.11+ with FastAPI (async/await everywhere)
- PostgreSQL 15 + Redis 7.0+ (non-standard ports to avoid conflicts)
- pytest with >90% coverage target (currently 2.6% due to blocked tests)
- LitPWA frontend (future), uv package manager

**Core Components**:
```
src/core/
├── task_queue.py      ← 🔥 YOUR MAIN FOCUS (serialization bug)
├── message_broker.py  ← Working (36% coverage)
├── caching.py         ← Working (32% coverage)
├── orchestrator.py    ← Needs testing (0% coverage)
└── models.py          ← Task model definitions
```

**Test Structure**:
```
tests/
├── integration/
│   └── test_core_system_integration.py ← 🔥 ALL FAILING due to TaskQueue bug
├── unit/
│   ├── test_task_queue_comprehensive.py ← Some tests passing
│   └── test_message_broker_comprehensive.py ← Working
```

### Key System Context

**Service Ports (Non-Standard to Avoid Conflicts)**:
- API Server: 9002 (FastAPI)
- Redis: 6381 (Docker mapped from 6379)
- PostgreSQL: 5433 (Docker mapped from 5432)
- pgAdmin: 9050, Redis Commander: 9081

**CLI Commands**:
- Start system: `hive system start`
- System status: `hive system status`
- Run tests: `pytest -q` or `pytest tests/integration/ -v`
- Coverage: `pytest --cov --cov-report=term-missing`

**Development Workflow**:
1. Fix the critical bug FIRST
2. Validate integration tests pass
3. Improve test coverage on core components
4. Add missing functionality (orchestrator, etc.)

### Current Test Coverage Status

```
CRITICAL COMPONENTS:
- TaskQueue: 29% coverage (BLOCKED by serialization bug)
- MessageBroker: 36% coverage (working)
- Cache: 32% coverage (working)
- Orchestrator: 0% coverage (needs implementation)

TARGET: 90% coverage minimum
ACTUAL: 2.6% (due to failing integration tests)
```

### Recent Progress Summary

**Infrastructure Stabilization (COMPLETED)**:
- Fixed Redis port conflicts (6381 vs 6379)
- Resolved API startup asyncio event loop issues
- Updated all port configurations across services
- Docker infrastructure fully operational

**Test Development (COMPLETED)**:
- Comprehensive integration tests covering 5 major workflows
- TaskQueue unit tests with priority queue validation
- MessageBroker tests with async dataclass serialization fixes
- Real Redis integration for authentic testing

**Critical Components (MOSTLY WORKING)**:
- TaskQueue: Core functionality works, only Redis serialization failing
- MessageBroker: Fully functional, API contracts fixed
- Cache: Operational with performance optimizations

### Files You Need to Focus On

**PRIMARY (Fix First)**:
- `src/core/task_queue.py:255` - `_dict_to_task()` method with serialization bug
- `tests/integration/test_core_system_integration.py` - Integration tests to validate fix

**SECONDARY (After Fix)**:
- `src/core/orchestrator.py` - Needs comprehensive testing (0% coverage)
- `tests/unit/test_orchestrator.py` - Create if missing
- API integration tests - HTTP layer validation

**REFERENCE**:
- `src/core/models.py` - Task model definition for field validation
- `AGENTS.md` - CLI commands and port configurations
- `docs/system-architecture.md` - Overall system design

### Success Metrics for Your Session

**CRITICAL (Must Complete)**:
1. ✅ Fix TaskQueue serialization bug in `_dict_to_task()`
2. ✅ All 5 integration tests passing
3. ✅ TaskQueue coverage above 50%

**HIGH PRIORITY (Secondary Goals)**:
4. ✅ Orchestrator component testing (push coverage from 0% to 30%+)
5. ✅ System-wide coverage above 70%
6. ✅ Performance validation (basic benchmarks)

**STRATEGIC (If Time Permits)**:
7. ✅ API endpoint integration tests
8. ✅ Error handling edge cases
9. ✅ Production readiness assessment

### Debugging Hints for the Critical Bug

**Likely Issues in `_dict_to_task()`**:
1. Redis returns bytes instead of strings for keys
2. Field names have prefixes/suffixes from Redis hash format
3. Data type conversion issues (UUID, datetime, JSON fields)
4. Missing or None fields causing model validation errors

**Debugging Approach**:
```python
# Add to _dict_to_task() method for debugging:
print(f"Raw task_data: {task_data}")
print(f"Keys: {list(task_data.keys())}")
print(f"Key types: {[type(k) for k in task_data.keys()]}")
data = dict(task_data)
print(f"Processed data: {data}")
print(f"Data keys: {list(data.keys())}")
```

**Quick Fix Pattern**:
```python
# Ensure all keys are strings
data = {str(k): v for k, v in task_data.items()}
```

### Development Philosophy

- **Test-Driven Development**: Fix tests first, then features
- **Incremental Progress**: Small, working commits
- **Fail-Safe Design**: Always have rollback capability
- **Async Everywhere**: Use async/await for all I/O operations
- **Type Safety**: Full type hints, validate at boundaries

### Next Steps After Bug Fix

1. **Complete Integration Testing**: All 5 workflows passing
2. **Orchestrator Testing**: Agent coordination and lifecycle
3. **Performance Validation**: Meet basic performance targets
4. **Production Readiness**: Security, error handling, monitoring
5. **Self-Improvement Features**: Meta-learning and autonomous development

## 🚀 START HERE

Run this command to see the current failure:
```bash
cd /Users/bogdan/work/leanvibe-dev/ant-farm
pytest tests/integration/test_core_system_integration.py::TestCoreSystemIntegration::test_complete_task_workflow -v -s
```

Then examine `src/core/task_queue.py:255` in the `_dict_to_task()` method. The system is 95% complete - this one bug is the only thing blocking full system validation!

**Remember**: This is a self-improving system. Make code that future agents can understand and modify. Keep it simple, well-tested, and thoroughly documented.