# PRD: Context Engine - Long-Term Memory System

## Executive Summary

The Context Engine provides persistent, semantic memory capabilities for LeanVibe Agent Hive 2.0, enabling agents to recall, search, and build upon previous interactions and knowledge. Using PostgreSQL with pgvector extension, it transforms short-term conversations into long-term institutional memory, reducing token costs by 60-80% and enabling true continuity across agent sessions.

## Problem Statement

Current agent systems suffer from:
- **Context Window Limitations**: Conversations truncate after ~8K-32K tokens
- **Knowledge Fragmentation**: Each agent interaction starts from zero context  
- **Repetitive Token Usage**: Same information processed repeatedly across sessions
- **No Learning Continuity**: Insights lost between agent restarts
- **Poor Cross-Agent Knowledge Sharing**: Isolated agent knowledge silos

## Success Metrics

### Performance Targets
- **Context Retrieval Speed**: <50ms for semantic searches
- **Token Reduction**: 60-80% decrease in context tokens per interaction
- **Memory Accuracy**: >90% relevant context retrieval precision
- **Storage Efficiency**: <1GB per 10,000 agent conversations
- **Concurrent Access**: Support 50+ agents with <100ms latency

### Business Impact
- **Development Velocity**: 40% faster agent task completion
- **Cost Reduction**: 70% lower LLM API costs through context optimization
- **Agent Intelligence**: Measurable improvement in multi-turn reasoning
- **Knowledge Retention**: Zero critical context loss across sessions

## Core Features

### 1. Semantic Memory Storage
Store and index agent conversations, decisions, and learned patterns using vector embeddings for semantic search and retrieval.

**User Story**: As an agent, I want to recall similar past problems and their solutions so I can avoid repeating analysis and build on previous work.

### 2. Context Compression Engine
Automatically summarize and compress lengthy conversations into dense, retrievable knowledge while preserving critical details.

**User Story**: As the system, I want to condense 20,000-token conversations into 500-token summaries without losing actionable insights.

### 3. Cross-Agent Knowledge Sharing
Enable agents to access and build upon insights from other agents within appropriate security boundaries.

**User Story**: As an agent, I want to discover what other agents learned about similar projects so I can leverage their expertise.

### 4. Temporal Context Windows
Maintain sliding windows of recent, medium-term, and long-term memory with intelligent prioritization.

**User Story**: As an agent, I want to prioritize recent context while still accessing relevant historical patterns when needed.

## Technical Architecture

### Database Schema
```sql
-- Core context storage with vector embeddings
CREATE TABLE contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    session_id UUID REFERENCES sessions(id),
    content_type context_type_enum NOT NULL,
    title VARCHAR(255) NOT NULL,
    summary TEXT,
    full_content JSONB,
    embedding vector(1536), -- OpenAI ada-002 dimensions
    importance_score FLOAT DEFAULT 0.5,
    access_level access_level_enum DEFAULT 'private',
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Semantic relationships between contexts
CREATE TABLE context_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_context_id UUID REFERENCES contexts(id),
    target_context_id UUID REFERENCES contexts(id),
    relationship_type relationship_enum,
    similarity_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context usage analytics
CREATE TABLE context_retrievals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_id UUID REFERENCES contexts(id),
    requesting_agent_id UUID REFERENCES agents(id),
    query_embedding vector(1536),
    similarity_score FLOAT,
    was_helpful BOOLEAN,
    feedback_score INTEGER CHECK (feedback_score BETWEEN 1 AND 5),
    retrieved_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_contexts_embedding ON contexts USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_contexts_agent_importance ON contexts(agent_id, importance_score DESC);
CREATE INDEX idx_contexts_temporal ON contexts(created_at DESC, last_accessed DESC);
```

### API Endpoints

```python
# Context Storage API
@router.post("/contexts", response_model=ContextResponse)
async def create_context(
    context: ContextCreate,
    current_agent: Agent = Depends(get_current_agent)
) -> ContextResponse:
    """Store new context with automatic embedding generation"""
    pass

@router.get("/contexts/search", response_model=List[ContextMatch])
async def search_contexts(
    query: str,
    agent_id: UUID,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    access_levels: List[AccessLevel] = None
) -> List[ContextMatch]:
    """Semantic search across agent contexts"""
    pass

@router.post("/contexts/{context_id}/compress")
async def compress_context(
    context_id: UUID,
    compression_level: CompressionLevel = CompressionLevel.STANDARD
) -> ContextResponse:
    """Intelligently compress context while preserving key information"""
    pass
```

### Core Components

#### 1. Embedding Service
```python
class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.client = OpenAI()
        self.model_name = model_name
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text content"""
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch generate embeddings for efficiency"""
        pass
```

#### 2. Context Compressor
```python
class ContextCompressor:
    def __init__(self, llm_client: AsyncAnthropic):
        self.llm = llm_client
    
    async def compress_conversation(
        self, 
        conversation: List[Message],
        compression_ratio: float = 0.3
    ) -> CompressedContext:
        """Compress conversation preserving key insights"""
        # Use Claude to intelligently summarize while preserving:
        # - Key decisions made
        # - Important discoveries
        # - Failure modes encountered
        # - Successful patterns
        pass
```

#### 3. Retrieval Engine
```python
class ContextRetrieval:
    def __init__(self, db: AsyncSession, embedding_service: EmbeddingService):
        self.db = db
        self.embeddings = embedding_service
    
    async def semantic_search(
        self,
        query: str,
        agent_id: UUID,
        limit: int = 10,
        filters: ContextFilters = None
    ) -> List[ContextMatch]:
        """Perform semantic search with relevance scoring"""
        query_embedding = await self.embeddings.generate_embedding(query)
        
        results = await self.db.execute(
            select(Context)
            .where(Context.agent_id == agent_id)
            .order_by(Context.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        return results.scalars().all()
```

## Implementation Plan

### Phase 1: Core Storage (Week 1)
**Objectives**: Basic context storage and retrieval
- [ ] Database schema implementation
- [ ] Basic CRUD operations for contexts
- [ ] OpenAI embedding integration
- [ ] Simple vector similarity search

**Tests**:
```python
def test_context_storage():
    """Test basic context storage and retrieval"""
    context = create_test_context("Test conversation about Redis setup")
    stored = context_service.store_context(context)
    assert stored.id is not None
    assert stored.embedding is not None

def test_semantic_search():
    """Test semantic search functionality"""
    # Store contexts about different topics
    contexts = [
        "Redis configuration for high availability",
        "PostgreSQL performance tuning",
        "Agent communication patterns"
    ]
    for ctx in contexts:
        context_service.store_context(create_test_context(ctx))
    
    # Search should return relevant results
    results = context_service.search("database optimization")
    assert len(results) == 2  # Redis and PostgreSQL contexts
    assert results[0].similarity_score > 0.7
```

### Phase 2: Compression & Intelligence (Week 2)
**Objectives**: Intelligent context compression and summarization
- [ ] LLM-based context compression
- [ ] Importance scoring algorithm
- [ ] Hierarchical summarization
- [ ] Context aging and pruning

**Tests**:
```python
def test_context_compression():
    """Test intelligent context compression"""
    long_context = create_large_context(50000)  # 50k character context
    compressed = context_service.compress_context(long_context, ratio=0.3)
    
    assert len(compressed.summary) < len(long_context.content) * 0.4
    assert compressed.importance_score > 0.0
    assert "key insight" in compressed.summary.lower()
```

### Phase 3: Cross-Agent Sharing (Week 3)
**Objectives**: Enable knowledge sharing between agents
- [ ] Access control implementation
- [ ] Cross-agent discovery mechanisms
- [ ] Knowledge graph relationships
- [ ] Privacy-preserving sharing

**Tests**:
```python
def test_cross_agent_knowledge_sharing():
    """Test agents can access shared knowledge"""
    # Agent A stores public knowledge
    context_a = context_service.store_context(
        create_test_context("Redis cluster setup guide", access_level="public")
    )
    
    # Agent B should be able to find it
    results = context_service.search("Redis setup", agent_id=agent_b.id)
    assert len(results) > 0
    assert context_a.id in [r.id for r in results]
```

### Phase 4: Performance Optimization (Week 4)
**Objectives**: Production-ready performance and monitoring
- [ ] Query optimization and indexing
- [ ] Caching layer implementation
- [ ] Performance monitoring
- [ ] Context relevance feedback loops

## Risk Assessment

### High-Risk Areas

**Vector Index Performance**: pgvector performance may degrade with large datasets
- *Mitigation*: Implement IVFFlat indexing, query optimization, and regular VACUUM ANALYZE

**Context Relevance Accuracy**: Retrieved contexts may not be relevant to current task
- *Mitigation*: Implement feedback loops, relevance scoring, and continuous model tuning

**Storage Growth**: Context database may grow unbounded
- *Mitigation*: Implement aging policies, compression ratios, and automated cleanup

### Data Privacy & Security
- Implement proper access controls for sensitive contexts
- Ensure embeddings don't leak private information
- Regular security audits of context access patterns

## Dependencies

### Technical Dependencies
- PostgreSQL 15+ with pgvector extension
- OpenAI API access for embeddings
- Redis for caching and session management
- Anthropic Claude API for compression

### Service Dependencies
- Agent Orchestrator Core (for agent identity)
- Communication System (for cross-agent sharing)
- Observability System (for performance monitoring)

## Acceptance Criteria

### Functional Requirements
- [ ] Agents can store conversation contexts with automatic embedding
- [ ] Semantic search returns relevant contexts within 50ms
- [ ] Context compression reduces token count by 60% while preserving key insights
- [ ] Cross-agent knowledge sharing respects access controls
- [ ] System handles 10,000+ contexts without performance degradation

### Non-Functional Requirements
- [ ] 99.9% uptime for context retrieval operations
- [ ] Support for 50+ concurrent agents
- [ ] Graceful degradation when embedding service is unavailable
- [ ] Complete test coverage >90% for core functionality
- [ ] Comprehensive monitoring and alerting

## Future Enhancements

### Advanced Features (Post-MVP)
- **Multi-modal Context**: Support for image, audio, and code embeddings
- **Temporal Reasoning**: Understanding context evolution over time
- **Knowledge Graphs**: Rich relationship modeling between contexts
- **Federated Learning**: Learn from context patterns without sharing data
- **Real-time Context Streaming**: Live context updates during conversations

### Integration Opportunities
- **External Knowledge Bases**: Wikipedia, documentation sites, code repositories
- **Version Control Integration**: Context tied to specific code commits
- **Analytics Dashboard**: Context usage patterns and effectiveness metrics