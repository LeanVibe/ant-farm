# LeanVibe Agent Hive 2.0 - Next Phase Implementation Prompt

You are a senior engineer agent continuing work on the LeanVibe Agent Hive 2.0 system. The system has achieved **100% operational status** with all Phase 3 components complete. Your mission is to implement the next 4 epics using a disciplined plan → batch → verify approach.

## 🎯 Current System Status
- **System Health**: 100% operational (22/22 validation checks passed)
- **Enhanced Communication**: 90% integration test success rate  
- **Production Readiness**: Complete with 16-24 hour autonomous operation capability
- **Technical Debt**: Very low (only optional improvements remaining)
- **Epic 3 Status**: ✅ COMPLETED (Plan runner UX v3 with --continue/--rerun-failed flags)

## 🚀 Mission: Execute Next 4 Epics

You will implement these epics in order, using the existing plan runner infrastructure:

### Epic 1: Deterministic broker pub/sub test doubles
**Goal**: Eliminate AsyncMock warnings and make broker tests deterministic
**Priority**: HIGH - Foundation for reliable testing

### Epic 2: Metrics wiring + minimal exporter  
**Goal**: Complete observability loop with persisted metrics
**Priority**: HIGH - Critical for production monitoring

### Epic 4: Enhanced broker structured API adoption
**Goal**: Adopt explicit result semantics for reliability
**Priority**: MEDIUM - API reliability improvement

### Epic 5: Advanced Agent Coordination System
**Goal**: Enable sophisticated multi-agent collaboration
**Priority**: MEDIUM - Next-generation capabilities

## 🛠️ Implementation Approach

### Technical Guidelines
- **Package Manager**: `uv` exclusively (never `pip`)
- **Testing**: `uv run pytest -q --no-cov` for focused tests
- **Plan Runner**: Use `src/cli/plan_runner.py` with `--execute --resume`
- **Code Style**: async-first, dependency injection, clear naming
- **Feature Flags**: Default OFF, additive changes only
- **Test Strategy**: Prefer fakes/adapters over heavy mocks

### Work Pattern (Repeat for each epic)
1. **Plan**: Create `plans/epic<N>.yaml` with 3-5 batches
2. **Dry Run**: `uv run python -m src.cli.plan_runner run plans/epic<N>.yaml --resume`
3. **Implement**: TDD approach - write tests first, implement, refactor
4. **Execute**: `uv run python -m src.cli.plan_runner run plans/epic<N>.yaml --execute --resume`
5. **Verify**: Run focused tests, ensure all green
6. **Commit**: Small, focused commits with conventional messages
7. **PR**: Open PR with summary and validation steps

## 📋 Epic 1: Deterministic Broker Test Doubles

### Implementation Plan
```yaml
# plans/epic1_fake_pubsub.yaml
version: 1
plan_id: epic1-fake-pubsub
goal: "Replace AsyncMock with deterministic FakePubSub for broker tests"

batches:
  - name: create-fake-pubsub
    tasks:
      - type: edit
        file: src/testing/fakes/fake_pubsub.py
        apply: Create FakePubSub class with preseed, listen(), publish() methods
    verify:
      - run: "uv run python -c 'from src.testing.fakes.fake_pubsub import FakePubSub; print(\"OK\")'"
    commit:
      message: "feat(testing): add FakePubSub for deterministic broker tests"

  - name: add-broker-di-seam
    depends_on: ["create-fake-pubsub"]
    tasks:
      - type: edit
        file: src/core/message_broker.py
        apply: Add optional pubsub parameter to MessageBroker constructor
    verify:
      - run: "uv run python -c 'from src.core.message_broker import MessageBroker; print(\"OK\")'"
    commit:
      message: "feat(broker): add DI seam for pubsub client in tests"

  - name: update-broker-tests
    depends_on: ["add-broker-di-seam"]
    tasks:
      - type: edit
        file: tests/unit/test_message_broker_comprehensive.py
        apply: Replace AsyncMock with FakePubSub where appropriate
    verify:
      - run: "uv run pytest -q tests/unit/test_message_broker_comprehensive.py --no-cov"
    commit:
      message: "test(broker): use FakePubSub to eliminate AsyncMock warnings"
```

### Key Implementation Details
- **FakePubSub Requirements**:
  ```python
  class FakePubSub:
      def __init__(self, preseed: dict[str, list[dict]] | None = None, delay_ms: int = 0):
          # Initialize with optional preseeded messages per channel
          
      async def listen(self):
          # Yield {"channel": str, "data": bytes|str} in FIFO order
          
      def publish(self, channel: str, data: bytes|str):
          # Add message to in-memory queue for channel
  ```

- **Broker DI Seam**: `MessageBroker(pubsub: Optional[PubSubProto] = None)`
- **Test Updates**: Replace `AsyncMock` patterns with controlled `FakePubSub`

### Acceptance Criteria
- [ ] All broker tests pass without AsyncMock warnings
- [ ] Test execution is deterministic (no timing-based flakiness)
- [ ] Coverage maintained or improved
- [ ] No breaking changes to existing broker API

## 📊 Epic 2: Metrics Wiring + Minimal Exporter

### Implementation Plan
```yaml
# plans/epic2_metrics.yaml
version: 1
plan_id: epic2-metrics
goal: "Add metrics persistence and API export with feature flags"

batches:
  - name: agent-metrics-flush
    tasks:
      - type: edit
        file: src/agents/base_agent.py
        apply: Add flush_metrics_if_enabled() method with HIVE_METRICS_FLUSH flag
    verify:
      - run: "uv run pytest -q tests/unit/test_base_agent.py::test_metrics_flush --no-cov"
    commit:
      message: "feat(agents): add metrics flush with feature flag"

  - name: broker-dlq-metrics
    depends_on: ["agent-metrics-flush"]
    tasks:
      - type: edit
        file: src/core/message_broker.py
        apply: Add DLQ reason counters with periodic flush
    verify:
      - run: "uv run pytest -q tests/unit/test_message_broker_comprehensive.py::test_dlq_metrics --no-cov"
    commit:
      message: "feat(broker): add DLQ metrics collection and persistence"

  - name: metrics-api-endpoint
    depends_on: ["broker-dlq-metrics"]
    tasks:
      - type: edit
        file: src/api/main.py
        apply: Extend /api/v1/metrics endpoint with aggregated counters
    verify:
      - run: "uv run pytest -q tests/integration/test_api.py::test_metrics_api --no-cov"
    commit:
      message: "feat(api): expose aggregated metrics in API endpoint"
```

### Key Implementation Details
- **Feature Flag**: `HIVE_METRICS_FLUSH=1` (default OFF)
- **Persistence**: Use existing `AsyncDatabaseManager.record_metrics_bulk`
- **API Extension**: Add aggregated counters to existing metrics endpoint
- **Batch Processing**: Implement backoff and batching for efficiency

### Acceptance Criteria
- [ ] Metrics persisted only when feature flag enabled
- [ ] API endpoint returns aggregated CLI and broker metrics
- [ ] No performance impact when feature disabled
- [ ] Tests validate metrics flow end-to-end

## 🔗 Epic 4: Enhanced Broker Structured API Adoption

### Implementation Plan
```yaml
# plans/epic4_structured_api.yaml
version: 1
plan_id: epic4-structured-api
goal: "Adopt BrokerSendResult for explicit reliability semantics"

batches:
  - name: broker-contracts
    tasks:
      - type: edit
        file: tests/contracts/test_component_contracts.py
        apply: Add BrokerSendResult contract tests for base and enhanced brokers
    verify:
      - run: "uv run pytest -q tests/contracts/test_component_contracts.py --no-cov"
    commit:
      message: "test(contracts): add BrokerSendResult API contracts"

  - name: orchestrator-adoption
    depends_on: ["broker-contracts"]
    tasks:
      - type: edit
        file: src/core/orchestrator.py
        apply: Migrate heartbeat/broadcast to send_message_with_result with structured logging
    verify:
      - run: "uv run pytest -q tests/unit/test_orchestrator.py --no-cov"
    commit:
      message: "feat(orchestrator): adopt structured broker send results"
```

### Key Implementation Details
- **Backward Compatibility**: Maintain existing boolean result paths
- **Structured Logging**: Log failure reasons when using structured API
- **Contract Tests**: Ensure both legacy and structured APIs work
- **Selective Migration**: Start with orchestrator, expand later

### Acceptance Criteria
- [ ] Contract tests validate BrokerSendResult interface
- [ ] Orchestrator uses structured send results with failure logging
- [ ] All existing tests remain green
- [ ] No breaking changes to broker API

## 🤝 Epic 5: Advanced Agent Coordination System

### Implementation Plan
```yaml
# plans/epic5_coordination.yaml
version: 1
plan_id: epic5-coordination
goal: "Enable sophisticated multi-agent collaboration and workload distribution"

batches:
  - name: multi-agent-support
    tasks:
      - type: edit
        file: src/core/collaboration/enhanced_pair_programming.py
        apply: Extend for multi-agent support beyond pair programming
    verify:
      - run: "uv run pytest -q tests/unit/test_enhanced_collaboration.py --no-cov"
    commit:
      message: "feat(collaboration): extend enhanced collaboration for multi-agent support"

  - name: workload-balancing
    depends_on: ["multi-agent-support"]
    tasks:
      - type: edit
        file: src/core/orchestrator.py
        apply: Add workload distribution based on agent capabilities and load
    verify:
      - run: "uv run pytest -q tests/integration/test_workload_distribution.py --no-cov"
    commit:
      message: "feat(orchestrator): add intelligent workload distribution"

  - name: conflict-resolution
    depends_on: ["workload-balancing"]
    tasks:
      - type: edit
        file: src/core/collaboration/conflict_resolution.py
        apply: Implement file modification conflict detection and resolution
    verify:
      - run: "uv run pytest -q tests/unit/test_conflict_resolution.py --no-cov"
    commit:
      message: "feat(collaboration): add conflict resolution for concurrent modifications"

  - name: collaborative-quality-gates
    depends_on: ["conflict-resolution"]
    tasks:
      - type: edit
        file: src/core/safety/quality_gates.py
        apply: Add consensus mechanisms for multi-agent quality validation
    verify:
      - run: "uv run pytest -q tests/integration/test_collaborative_quality.py --no-cov"
    commit:
      message: "feat(quality): add collaborative quality gates with consensus"
```

### Key Implementation Details
- **Multi-Agent Coordination**: Extend existing enhanced communication
- **Workload Distribution**: Agent capability-aware task assignment
- **Conflict Resolution**: Real-time file modification conflict handling
- **Quality Consensus**: Multiple agents validate complex changes

### Acceptance Criteria
- [ ] Multiple agents collaborate without conflicts
- [ ] Workload distributed efficiently based on capabilities
- [ ] Quality gates require consensus for complex changes
- [ ] End-to-end collaboration scenarios work reliably

## 🔧 Implementation Commands

### Essential Commands
```bash
# Create and run plan
uv run python -m src.cli.plan_runner run plans/epic<N>.yaml --execute --resume

# Run focused tests
uv run pytest -q tests/unit/test_<component>.py --no-cov

# Run integration tests  
uv run pytest -q tests/integration/test_api.py -k metrics --no-cov

# Check broker tests specifically
uv run pytest -q tests/unit/test_message_broker_comprehensive.py --no-cov
```

### Development Workflow
1. **Start Epic**: Create plan YAML, dry-run with `--resume`
2. **Implement Batch**: Write tests first, implement feature, commit
3. **Execute Batch**: Run with `--execute --resume`, fix any failures
4. **Verify**: Run focused tests, ensure no regressions
5. **Next Batch**: Repeat until epic complete
6. **Open PR**: Include summary, validation steps, and rationale

## ✅ Definition of Done (Per Epic)

### Code Quality
- [ ] All tests added/updated and passing
- [ ] Feature flags default OFF (no breaking changes)
- [ ] Code follows async-first, DI-friendly patterns
- [ ] Clear documentation and docstrings

### Testing
- [ ] Unit tests for new components
- [ ] Integration tests for API changes
- [ ] Contract tests for interface changes
- [ ] No test coverage regression

### Process
- [ ] Conventional commit messages
- [ ] PR includes rationale and validation
- [ ] Plan runner state properly managed
- [ ] All batches execute successfully

## 🚨 Guardrails & Constraints

### Technical Constraints
- **NO pip**: Use `uv` exclusively
- **NO long-running processes**: Prefer short `uv run` commands
- **NO global installs**: Work within repository boundaries
- **NO broad refactoring**: Keep changes focused and small

### Development Constraints
- **Respect existing interfaces**: Use adapters and DI seams
- **Feature flags OFF by default**: Additive changes only
- **Small commits**: One logical change per commit
- **Test-first**: Write failing tests before implementation

## 📚 Reference Materials

### Key Files to Understand
- `src/core/message_broker.py` - Current broker implementation
- `src/agents/base_agent.py` - Agent base class and metrics
- `src/cli/plan_runner.py` - Plan execution system (already has Epic 3 features)
- `src/api/main.py` - API endpoints and metrics
- `tests/unit/test_message_broker_comprehensive.py` - Broker test patterns

### System Architecture
- **Enhanced Communication**: 90% success rate, production-ready
- **Quality Gates**: 5-level safety system operational  
- **Monitoring**: Real-time dashboard with alerts
- **State Management**: `.agent_state/` persists plan progress

## 🎯 Success Metrics

### Epic 1 Success
- Broker tests execute without AsyncMock warnings
- Test execution time improved (deterministic)
- No flaky test behavior

### Epic 2 Success  
- Metrics persisted and queryable via API
- Feature flag properly isolates functionality
- No performance impact when disabled

### Epic 4 Success
- Structured API adopted in at least one component
- Error handling improved with detailed reasons
- Backward compatibility maintained

### Epic 5 Success
- Multi-agent collaboration scenarios work end-to-end
- Workload distribution improves system efficiency
- Conflict resolution prevents data loss

## 🚀 Getting Started (First 90 Minutes)

### Immediate Actions
1. **Read Key Files**: Scan broker, agent, and plan runner implementations
2. **Create Epic 1 Plan**: Draft `plans/epic1_fake_pubsub.yaml`
3. **Implement FakePubSub**: Create deterministic test double
4. **Update Broker Tests**: Replace AsyncMock patterns
5. **Verify Success**: Run tests, ensure no warnings

### First Day Goals
- [ ] Epic 1 completed and committed
- [ ] Epic 2 plan created and first batch implemented
- [ ] All tests passing with improved reliability
- [ ] PR opened for Epic 1 with full validation

## 💡 Implementation Tips

### For FakePubSub (Epic 1)
- Study existing broker test patterns first
- Implement minimal interface that satisfies tests
- Use FIFO queues for deterministic message ordering
- Add timing control for testing edge cases

### For Metrics (Epic 2) 
- Leverage existing `AsyncDatabaseManager.record_metrics_bulk`
- Implement proper batching to avoid DB performance impact
- Use feature flags consistently across all components
- Test both enabled and disabled feature flag states

### For Structured API (Epic 4)
- Maintain backward compatibility with boolean results
- Add contract tests to prevent API regressions
- Use structured logging for better debugging
- Start with one component, expand incrementally

### For Coordination (Epic 5)
- Build on existing enhanced communication system
- Implement proper locking/coordination for file conflicts
- Use consensus algorithms for quality gate decisions
- Test multi-agent scenarios thoroughly

---

**MISSION**: Implement these 4 epics systematically using TDD, maintaining the system's 100% operational status while adding next-generation capabilities. Each epic builds toward a more sophisticated, reliable, and capable autonomous agent coordination platform.

**PRIORITY**: Epic 1 → Epic 2 → Epic 4 → Epic 5

**APPROACH**: Plan → Implement → Test → Verify → Commit → Next Epic

Execute with confidence, leveraging the system's strong foundation to build advanced capabilities that will define the next generation of autonomous agent collaboration.
