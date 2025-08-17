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

        // Listen for system status updates
        this.handleSystemStatus = (event) => {
            const systemData = event.detail;
            if (systemData.active_agents !== undefined) {
                // Trigger agents reload if count mismatch
                const activeCount = this.agents.filter(a => a.status === 'active').length;
                if (activeCount !== systemData.active_agents) {
                    this.loadAgents();
                }
            }
            
            // Handle system events that may affect agents
            if (systemData.event && systemData.event.source && systemData.event.data) {
                const event = systemData.event;
                if (event.source.includes('agent')) {
                    this.updateAgentFromEvent(event);
                }
            }
        };

        // Listen for metrics updates to get agent performance data
        this.handleMetricsUpdate = (event) => {
            const metricsData = event.detail;
            
            if (metricsData.recent_metrics && Array.isArray(metricsData.recent_metrics)) {
                // Process agent-specific metrics
                const agentMetrics = {};
                
                metricsData.recent_metrics.forEach(metric => {
                    if (metric.tags && metric.tags.agent_id) {
                        const agentId = metric.tags.agent_id;
                        if (!agentMetrics[agentId]) {
                            agentMetrics[agentId] = {};
                        }
                        
                        switch (metric.name) {
                            case 'agent_cpu_usage':
                                agentMetrics[agentId].cpu_usage = metric.value;
                                break;
                            case 'agent_memory_usage':
                                agentMetrics[agentId].memory_usage = metric.value;
                                break;
                            case 'agent_response_time':
                                agentMetrics[agentId].response_time = metric.value;
                                break;
                            case 'agent_task_count':
                                agentMetrics[agentId].active_tasks = metric.value;
                                break;
                            case 'agent_health_score':
                                agentMetrics[agentId].health_score = metric.value;
                                break;
                        }
                    }
                });
                
                // Update agents with new metrics
                this.agents = this.agents.map(agent => {
                    const metrics = agentMetrics[agent.id] || agentMetrics[agent.name];
                    return metrics ? { ...agent, ...metrics } : agent;
                });
                
                this.requestUpdate();
            }
        };

        // Listen for dashboard data updates
        this.handleDashboardData = (event) => {
            const dashboardData = event.detail;
            
            if (dashboardData.recent_events) {
                // Process recent events for agent-related updates
                dashboardData.recent_events.forEach(event => {
                    if (event.source && event.source.includes('agent')) {
                        this.updateAgentFromEvent(event);
                    }
                });
            }
        };

        window.addEventListener('agent-status-update', this.handleAgentUpdate);
        window.addEventListener('system-status-update', this.handleSystemStatus);
        window.addEventListener('hive-status-update', this.handleSystemStatus);
        window.addEventListener('metrics-update', this.handleMetricsUpdate);
        window.addEventListener('real-time-metrics', this.handleMetricsUpdate);
        window.addEventListener('dashboard-data', this.handleDashboardData);
    }

    removeWebSocketListeners() {
        if (this.handleAgentUpdate) {
            window.removeEventListener('agent-status-update', this.handleAgentUpdate);
        }
        if (this.handleSystemStatus) {
            window.removeEventListener('system-status-update', this.handleSystemStatus);
            window.removeEventListener('hive-status-update', this.handleSystemStatus);
        }
        if (this.handleMetricsUpdate) {
            window.removeEventListener('metrics-update', this.handleMetricsUpdate);
            window.removeEventListener('real-time-metrics', this.handleMetricsUpdate);
        }
        if (this.handleDashboardData) {
            window.removeEventListener('dashboard-data', this.handleDashboardData);
        }
    }

    updateAgentFromEvent(event) {
        // Update agent based on system event
        if (event.source && event.data) {
            const agentId = this.extractAgentIdFromSource(event.source);
            if (agentId) {
                this.agents = this.agents.map(agent => {
                    if (agent.id === agentId || agent.name === agentId) {
                        return {
                            ...agent,
                            last_event: event.timestamp,
                            last_event_type: event.event_type
                        };
                    }
                    return agent;
                });
                this.requestUpdate();
            }
        }
    }

    extractAgentIdFromSource(source) {
        // Extract agent ID from source string like "agent:meta-agent-123"
        const match = source.match(/agent:(.+)/);
        return match ? match[1] : null;
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
                                        <div class="stat-value">${agent.active_tasks || 0}</div>
                                        <div class="stat-label">Active</div>
                                    </div>
                                    <div class="stat">
                                        <div class="stat-value">${this.formatUptime(agent.uptime)}</div>
                                        <div class="stat-label">Uptime</div>
                                    </div>
                                    ${agent.cpu_usage !== undefined ? html`
                                        <div class="stat">
                                            <div class="stat-value">${Math.round(agent.cpu_usage)}%</div>
                                            <div class="stat-label">CPU</div>
                                        </div>
                                    ` : ''}
                                    ${agent.memory_usage !== undefined ? html`
                                        <div class="stat">
                                            <div class="stat-value">${Math.round(agent.memory_usage)}%</div>
                                            <div class="stat-label">Memory</div>
                                        </div>
                                    ` : ''}
                                    ${agent.response_time !== undefined ? html`
                                        <div class="stat">
                                            <div class="stat-value">${Math.round(agent.response_time)}ms</div>
                                            <div class="stat-label">Response</div>
                                        </div>
                                    ` : ''}
                                    ${agent.health_score !== undefined ? html`
                                        <div class="stat">
                                            <div class="stat-value ${this.getHealthScoreClass(agent.health_score)}">${Math.round(agent.health_score * 100)}%</div>
                                            <div class="stat-label">Health</div>
                                        </div>
                                    ` : ''}
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