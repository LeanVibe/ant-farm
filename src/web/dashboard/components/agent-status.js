import { LitElement, html, css } from 'lit';

export class AgentStatus extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }

        .agent-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .agent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .agent-name {
            font-weight: 600;
            color: #00d4aa;
        }

        .agent-type {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .status-active { background: #00d4aa; box-shadow: 0 0 8px #00d4aa; }
        .status-idle { background: #ffd700; }
        .status-error { background: #ff4444; }
        .status-offline { background: #666; }

        .agent-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .stat {
            text-align: center;
            padding: 0.5rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }

        .stat-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #00d4aa;
        }

        .stat-label {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
        }

        .compact .agent-grid {
            grid-template-columns: 1fr;
        }

        .compact .agent-card {
            padding: 0.75rem;
        }

        .heartbeat {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.6);
        }
    `;

    static properties = {
        agents: { type: Array },
        compact: { type: Boolean }
    };

    constructor() {
        super();
        this.agents = [];
        this.compact = false;
    }

    connectedCallback() {
        super.connectedCallback();
        this.loadAgents();
        this.initializeWebSocketListeners();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeWebSocketListeners();
    }

    initializeWebSocketListeners() {
        // Listen for real-time agent updates via WebSocket
        this.handleAgentUpdate = (event) => {
            const agentData = event.detail;
            if (agentData.action === 'spawned') {
                // Add new agent to list
                this.agents = [...this.agents, agentData.agent];
            } else if (agentData.action === 'stopped') {
                // Remove or update agent status
                this.agents = this.agents.map(agent => 
                    agent.name === agentData.agent.name 
                        ? { ...agent, status: 'stopping' }
                        : agent
                );
            } else if (agentData.action === 'status_update') {
                // Update existing agent
                this.agents = this.agents.map(agent =>
                    agent.name === agentData.agent.name
                        ? { ...agent, ...agentData.agent }
                        : agent
                );
            }
            this.requestUpdate();
        };

        this.handleSystemStatus = (event) => {
            // System status updates may contain agent count changes
            const systemData = event.detail;
            if (systemData.active_agents !== undefined) {
                // Trigger agents reload if count mismatch
                const activeCount = this.agents.filter(a => a.status === 'active').length;
                if (activeCount !== systemData.active_agents) {
                    this.loadAgents();
                }
            }
        };

        window.addEventListener('agent-status-update', this.handleAgentUpdate);
        window.addEventListener('hive-status-update', this.handleSystemStatus);
    }

    removeWebSocketListeners() {
        if (this.handleAgentUpdate) {
            window.removeEventListener('agent-status-update', this.handleAgentUpdate);
        }
        if (this.handleSystemStatus) {
            window.removeEventListener('hive-status-update', this.handleSystemStatus);
        }
    }

    async loadAgents() {
        try {
            const response = await fetch('/api/v1/agents');
            if (response.ok) {
                const data = await response.json();
                this.agents = data.data || [];
            }
        } catch (error) {
            console.error('Failed to load agents:', error);
        }
    }

    getStatusClass(status) {
        switch (status) {
            case 'active': return 'status-active';
            case 'idle': return 'status-idle';
            case 'error': return 'status-error';
            default: return 'status-offline';
        }
    }

    formatUptime(seconds) {
        if (!seconds) return 'N/A';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }

    render() {
        return html`
            <div class="${this.compact ? 'compact' : ''}">
                <div class="agent-grid">
                    ${this.agents.map(agent => html`
                        <div class="agent-card">
                            <div class="agent-header">
                                <div class="agent-name">${agent.name}</div>
                                <div class="agent-type">${agent.type}</div>
                            </div>
                            
                            <div class="status-indicator">
                                <div class="status-dot ${this.getStatusClass(agent.status)}"></div>
                                <span>${agent.status}</span>
                                ${agent.last_heartbeat ? html`
                                    <span class="heartbeat">
                                        • ${new Date(agent.last_heartbeat).toLocaleTimeString()}
                                    </span>
                                ` : ''}
                            </div>

                            ${!this.compact ? html`
                                <div class="agent-stats">
                                    <div class="stat">
                                        <div class="stat-value">${agent.tasks_completed || 0}</div>
                                        <div class="stat-label">Completed</div>
                                    </div>
                                    <div class="stat">
                                        <div class="stat-value">${agent.tasks_failed || 0}</div>
                                        <div class="stat-label">Failed</div>
                                    </div>
                                    <div class="stat">
                                        <div class="stat-value">${(agent.load_factor * 100 || 0).toFixed(0)}%</div>
                                        <div class="stat-label">Load</div>
                                    </div>
                                    <div class="stat">
                                        <div class="stat-value">${this.formatUptime(agent.uptime)}</div>
                                        <div class="stat-label">Uptime</div>
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    `)}
                </div>

                ${this.agents.length === 0 ? html`
                    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.6);">
                        No agents detected. System may be starting up...
                    </div>
                ` : ''}
            </div>
        `;
    }
}

customElements.define('agent-status', AgentStatus);