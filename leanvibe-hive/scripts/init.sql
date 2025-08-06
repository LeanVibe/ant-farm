-- LeanVibe Agent Hive 2.0 Database Schema
-- PostgreSQL with pgvector for semantic memory

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Agents table: Core agent information
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    type VARCHAR(100) NOT NULL, -- meta, developer, qa, architect, etc.
    role VARCHAR(100) NOT NULL, -- system_improver, code_generator, etc.
    capabilities JSONB DEFAULT '{}', -- Agent-specific capabilities
    system_prompt TEXT, -- Current system prompt
    status VARCHAR(50) DEFAULT 'inactive', -- inactive, active, sleeping, error
    tmux_session VARCHAR(255), -- Associated tmux session name
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tasks table: Work items for agents
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL, -- analysis, code_generation, testing, etc.
    payload JSONB DEFAULT '{}', -- Task-specific data
    priority INTEGER DEFAULT 5, -- 1=critical, 5=normal, 9=background
    status VARCHAR(50) DEFAULT 'pending', -- pending, assigned, in_progress, completed, failed, cancelled
    agent_id UUID REFERENCES agents(id),
    dependencies JSONB DEFAULT '[]', -- Array of task IDs this depends on
    result JSONB, -- Task execution result
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 300,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Sessions table: Agent collaboration sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL, -- development, analysis, planning, etc.
    agents JSONB DEFAULT '[]', -- Array of participating agent IDs
    state JSONB DEFAULT '{}', -- Session state and metadata
    status VARCHAR(50) DEFAULT 'active', -- active, paused, completed
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW()
);

-- Contexts table: Semantic memory storage
CREATE TABLE contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    session_id UUID REFERENCES sessions(id),
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- OpenAI ada-002 embedding dimension
    importance_score FLOAT DEFAULT 0.5, -- 0.0 to 1.0
    parent_id UUID REFERENCES contexts(id), -- For hierarchical contexts
    tags JSONB DEFAULT '[]', -- Searchable tags
    metadata JSONB DEFAULT '{}', -- Additional context metadata
    created_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- Conversations table: Inter-agent communication
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    from_agent_id UUID REFERENCES agents(id),
    to_agent_id UUID REFERENCES agents(id), -- NULL for broadcast
    message_type VARCHAR(100) NOT NULL, -- request, response, broadcast, notification
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- For semantic search of conversations
    reply_to UUID REFERENCES conversations(id), -- For threaded conversations
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- System checkpoints table: Self-modification tracking
CREATE TABLE system_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(100) NOT NULL, -- code_change, config_update, schema_migration
    description TEXT,
    state JSONB NOT NULL, -- Complete system state snapshot
    git_commit_hash VARCHAR(64), -- Associated git commit
    agent_id UUID REFERENCES agents(id), -- Agent that created checkpoint
    performance_metrics JSONB, -- Performance before/after
    rollback_data JSONB, -- Data needed for rollback
    status VARCHAR(50) DEFAULT 'created', -- created, applied, rolled_back
    created_at TIMESTAMP DEFAULT NOW(),
    applied_at TIMESTAMP,
    rolled_back_at TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_type ON agents(type);
CREATE INDEX idx_agents_updated_at ON agents(updated_at);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_priority ON tasks(priority);
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_created_at ON tasks(created_at);
CREATE INDEX idx_tasks_type ON tasks(type);

CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_last_active ON sessions(last_active);

CREATE INDEX idx_contexts_agent_id ON contexts(agent_id);
CREATE INDEX idx_contexts_session_id ON contexts(session_id);
CREATE INDEX idx_contexts_importance_score ON contexts(importance_score);
CREATE INDEX idx_contexts_created_at ON contexts(created_at);
CREATE INDEX idx_contexts_accessed_at ON contexts(accessed_at);

CREATE INDEX idx_conversations_session_id ON conversations(session_id);
CREATE INDEX idx_conversations_from_agent_id ON conversations(from_agent_id);
CREATE INDEX idx_conversations_to_agent_id ON conversations(to_agent_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);
CREATE INDEX idx_conversations_reply_to ON conversations(reply_to);

CREATE INDEX idx_checkpoints_type ON system_checkpoints(type);
CREATE INDEX idx_checkpoints_status ON system_checkpoints(status);
CREATE INDEX idx_checkpoints_created_at ON system_checkpoints(created_at);

-- Vector similarity indexes for semantic search
CREATE INDEX idx_contexts_embedding ON contexts USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_conversations_embedding ON conversations USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initial system data
INSERT INTO agents (name, type, role, system_prompt, status) VALUES 
(
    'bootstrap-agent',
    'bootstrap',
    'system_initializer',
    'You are the bootstrap agent responsible for initializing the LeanVibe Agent Hive system.',
    'active'
),
(
    'meta-agent-001',
    'meta',
    'system_improver',
    'You are a meta-agent responsible for analyzing system performance and proposing improvements.',
    'inactive'
);