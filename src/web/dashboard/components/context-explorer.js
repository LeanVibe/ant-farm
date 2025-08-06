import { LitElement, html, css } from 'lit';

export class ContextExplorer extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .context-explorer {
            height: 600px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .explorer-header {
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .search-section {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .search-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            padding: 0.75rem;
            color: #ffffff;
            font-size: 0.9rem;
        }

        .search-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        .search-input:focus {
            outline: none;
            border-color: #00d4aa;
            box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
        }

        .search-button {
            background: #00d4aa;
            color: #1a1a2e;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .search-button:hover {
            background: #00e5bb;
        }

        .filters {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .filter-select {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 0.5rem;
            color: #ffffff;
            font-size: 0.8rem;
        }

        .content-area {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        .memory-tree {
            width: 250px;
            background: rgba(0, 0, 0, 0.1);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
            padding: 1rem;
        }

        .tree-node {
            margin-bottom: 0.5rem;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 4px;
            transition: background 0.3s ease;
        }

        .tree-node:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .tree-node.active {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
        }

        .tree-icon {
            margin-right: 0.5rem;
        }

        .tree-label {
            font-size: 0.9rem;
        }

        .tree-count {
            float: right;
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.6);
        }

        .context-list {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }

        .context-item {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .context-item:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .context-item.selected {
            border-color: #00d4aa;
            background: rgba(0, 212, 170, 0.1);
        }

        .context-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .context-agent {
            color: #00d4aa;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .context-timestamp {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.5);
        }

        .context-preview {
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.4;
            margin-bottom: 0.5rem;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .context-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
        }

        .importance-score {
            padding: 0.2rem 0.4rem;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: bold;
        }

        .importance-high { background: #ff4444; color: white; }
        .importance-medium { background: #ffd700; color: #1a1a2e; }
        .importance-low { background: #666; color: white; }

        .similarity-score {
            color: rgba(255, 255, 255, 0.6);
        }

        .context-detail {
            width: 300px;
            background: rgba(0, 0, 0, 0.2);
            border-left: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
            overflow-y: auto;
        }

        .detail-header {
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .detail-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #00d4aa;
        }

        .detail-content {
            color: #ffffff;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        .detail-metadata {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 1rem;
        }

        .metadata-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
        }

        .metadata-label {
            color: rgba(255, 255, 255, 0.7);
        }

        .metadata-value {
            color: #ffffff;
        }

        .related-contexts {
            margin-top: 1rem;
        }

        .related-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .related-item:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .empty-state {
            text-align: center;
            padding: 3rem;
            color: rgba(255, 255, 255, 0.5);
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: rgba(255, 255, 255, 0.7);
        }

        .compact .context-explorer {
            height: 400px;
        }

        .compact .memory-tree {
            width: 200px;
        }

        .compact .context-detail {
            width: 250px;
        }

        @media (max-width: 768px) {
            .content-area {
                flex-direction: column;
            }
            
            .memory-tree {
                width: 100%;
                height: 150px;
            }
            
            .context-detail {
                width: 100%;
                height: 200px;
            }
        }
    `;

    static properties = {
        contexts: { type: Array },
        selectedContext: { type: Object },
        searchQuery: { type: String },
        activeAgent: { type: String },
        importanceFilter: { type: String },
        isLoading: { type: Boolean },
        compact: { type: Boolean },
        memoryStats: { type: Object }
    };

    constructor() {
        super();
        this.contexts = [];
        this.selectedContext = null;
        this.searchQuery = '';
        this.activeAgent = 'all';
        this.importanceFilter = 'all';
        this.isLoading = false;
        this.compact = false;
        this.memoryStats = {
            total: 0,
            byAgent: {},
            byImportance: { high: 0, medium: 0, low: 0 }
        };
    }

    connectedCallback() {
        super.connectedCallback();
        this.loadContexts();
        this.loadMemoryStats();
    }

    async loadContexts() {
        this.isLoading = true;
        try {
            const params = new URLSearchParams();
            if (this.searchQuery) params.append('query', this.searchQuery);
            if (this.activeAgent !== 'all') params.append('agent', this.activeAgent);
            if (this.importanceFilter !== 'all') params.append('importance', this.importanceFilter);
            
            const response = await fetch(`/api/v1/contexts?${params}`);
            if (response.ok) {
                const data = await response.json();
                this.contexts = data.data || [];
            }
        } catch (error) {
            console.error('Failed to load contexts:', error);
        } finally {
            this.isLoading = false;
        }
    }

    async loadMemoryStats() {
        try {
            const response = await fetch('/api/v1/contexts/stats');
            if (response.ok) {
                const data = await response.json();
                this.memoryStats = data.data || this.memoryStats;
            }
        } catch (error) {
            console.error('Failed to load memory stats:', error);
        }
    }

    async performSearch() {
        await this.loadContexts();
    }

    handleSearchInput(e) {
        this.searchQuery = e.target.value;
        // Debounce search
        clearTimeout(this.searchTimeout);
        this.searchTimeout = setTimeout(() => {
            this.performSearch();
        }, 500);
    }

    selectContext(context) {
        this.selectedContext = context;
        this.loadRelatedContexts(context.id);
    }

    async loadRelatedContexts(contextId) {
        try {
            const response = await fetch(`/api/v1/contexts/${contextId}/related`);
            if (response.ok) {
                const data = await response.json();
                this.selectedContext.related = data.data || [];
                this.requestUpdate();
            }
        } catch (error) {
            console.error('Failed to load related contexts:', error);
        }
    }

    getImportanceClass(score) {
        if (score >= 0.8) return 'importance-high';
        if (score >= 0.5) return 'importance-medium';
        return 'importance-low';
    }

    getImportanceLabel(score) {
        if (score >= 0.8) return 'High';
        if (score >= 0.5) return 'Medium';
        return 'Low';
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    render() {
        return html`
            <div class="${this.compact ? 'compact' : ''}">
                <div class="context-explorer">
                    <div class="explorer-header">
                        <div class="search-section">
                            <input 
                                type="text" 
                                class="search-input"
                                placeholder="Search semantic memory..."
                                .value=${this.searchQuery}
                                @input=${this.handleSearchInput}
                                @keydown=${(e) => e.key === 'Enter' && this.performSearch()}
                            >
                            <button class="search-button" @click=${this.performSearch}>
                                🔍 Search
                            </button>
                        </div>
                        
                        <div class="filters">
                            <select 
                                class="filter-select"
                                .value=${this.activeAgent}
                                @change=${(e) => { this.activeAgent = e.target.value; this.loadContexts(); }}
                            >
                                <option value="all">All Agents</option>
                                ${Object.keys(this.memoryStats.byAgent || {}).map(agent => html`
                                    <option value="${agent}">${agent} (${this.memoryStats.byAgent[agent]})</option>
                                `)}
                            </select>
                            
                            <select 
                                class="filter-select"
                                .value=${this.importanceFilter}
                                @change=${(e) => { this.importanceFilter = e.target.value; this.loadContexts(); }}
                            >
                                <option value="all">All Importance</option>
                                <option value="high">High (${this.memoryStats.byImportance.high})</option>
                                <option value="medium">Medium (${this.memoryStats.byImportance.medium})</option>
                                <option value="low">Low (${this.memoryStats.byImportance.low})</option>
                            </select>
                        </div>
                    </div>

                    <div class="content-area">
                        <div class="memory-tree">
                            <div class="tree-node ${this.activeAgent === 'all' ? 'active' : ''}"
                                 @click=${() => { this.activeAgent = 'all'; this.loadContexts(); }}>
                                <span class="tree-icon">🧠</span>
                                <span class="tree-label">All Memory</span>
                                <span class="tree-count">${this.memoryStats.total}</span>
                            </div>
                            
                            ${Object.entries(this.memoryStats.byAgent || {}).map(([agent, count]) => html`
                                <div class="tree-node ${this.activeAgent === agent ? 'active' : ''}"
                                     @click=${() => { this.activeAgent = agent; this.loadContexts(); }}>
                                    <span class="tree-icon">🤖</span>
                                    <span class="tree-label">${agent}</span>
                                    <span class="tree-count">${count}</span>
                                </div>
                            `)}
                        </div>

                        <div class="context-list">
                            ${this.isLoading ? html`
                                <div class="loading">🔄 Loading contexts...</div>
                            ` : this.contexts.length === 0 ? html`
                                <div class="empty-state">
                                    No contexts found matching your criteria
                                </div>
                            ` : this.contexts.map(context => html`
                                <div class="context-item ${this.selectedContext?.id === context.id ? 'selected' : ''}"
                                     @click=${() => this.selectContext(context)}>
                                    <div class="context-header">
                                        <div class="context-agent">${context.agent_name}</div>
                                        <div class="context-timestamp">${this.formatTimestamp(context.created_at)}</div>
                                    </div>
                                    
                                    <div class="context-preview">
                                        ${context.content}
                                    </div>
                                    
                                    <div class="context-meta">
                                        <div class="importance-score ${this.getImportanceClass(context.importance_score)}">
                                            ${this.getImportanceLabel(context.importance_score)}
                                        </div>
                                        ${context.similarity_score ? html`
                                            <div class="similarity-score">
                                                ${(context.similarity_score * 100).toFixed(1)}% match
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            `)}
                        </div>

                        ${this.selectedContext ? html`
                            <div class="context-detail">
                                <div class="detail-header">
                                    <div class="detail-title">Context Details</div>
                                </div>
                                
                                <div class="detail-content">
                                    ${this.selectedContext.content}
                                </div>
                                
                                <div class="detail-metadata">
                                    <div class="metadata-row">
                                        <span class="metadata-label">Agent:</span>
                                        <span class="metadata-value">${this.selectedContext.agent_name}</span>
                                    </div>
                                    <div class="metadata-row">
                                        <span class="metadata-label">Importance:</span>
                                        <span class="metadata-value">${this.selectedContext.importance_score?.toFixed(3)}</span>
                                    </div>
                                    <div class="metadata-row">
                                        <span class="metadata-label">Created:</span>
                                        <span class="metadata-value">${this.formatTimestamp(this.selectedContext.created_at)}</span>
                                    </div>
                                    <div class="metadata-row">
                                        <span class="metadata-label">Accessed:</span>
                                        <span class="metadata-value">${this.formatTimestamp(this.selectedContext.accessed_at)}</span>
                                    </div>
                                </div>
                                
                                ${this.selectedContext.related?.length > 0 ? html`
                                    <div class="related-contexts">
                                        <h4>Related Contexts</h4>
                                        ${this.selectedContext.related.map(related => html`
                                            <div class="related-item" @click=${() => this.selectContext(related)}>
                                                ${related.content.substring(0, 100)}...
                                            </div>
                                        `)}
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
}

customElements.define('context-explorer', ContextExplorer);