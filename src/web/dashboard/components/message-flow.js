import { LitElement, html, css } from 'lit';

export class MessageFlow extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .message-flow-container {
            height: 600px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .flow-header {
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .flow-title {
            font-weight: 600;
            color: #ffffff;
        }

        .flow-controls {
            display: flex;
            gap: 0.5rem;
        }

        .control-button {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
            border: 1px solid #00d4aa;
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }

        .control-button:hover {
            background: rgba(0, 212, 170, 0.3);
        }

        .control-button.active {
            background: #00d4aa;
            color: #1a1a2e;
        }

        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }

        .message {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid;
            background: rgba(255, 255, 255, 0.1);
            position: relative;
            animation: messageSlideIn 0.3s ease-out;
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .message.system { border-left-color: #00d4aa; }
        .message.agent { border-left-color: #6495ed; }
        .message.error { border-left-color: #ff4444; }
        .message.broadcast { border-left-color: #ffd700; }

        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .message-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .message-type {
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: bold;
        }

        .type-system { background: rgba(0, 212, 170, 0.2); color: #00d4aa; }
        .type-agent { background: rgba(100, 149, 237, 0.2); color: #6495ed; }
        .type-error { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
        .type-broadcast { background: rgba(255, 215, 0, 0.2); color: #ffd700; }

        .message-agents {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.8);
        }

        .message-timestamp {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.5);
        }

        .message-content {
            color: #ffffff;
            line-height: 1.4;
        }

        .message-topic {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.6);
            margin-top: 0.5rem;
            font-style: italic;
        }

        .filter-section {
            padding: 0.5rem 1rem;
            background: rgba(0, 0, 0, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .filter-chip {
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: rgba(255, 255, 255, 0.7);
        }

        .filter-chip:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .filter-chip.active {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
            border-color: #00d4aa;
        }

        .stats-bar {
            padding: 0.5rem 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
        }

        .empty-state {
            text-align: center;
            padding: 3rem;
            color: rgba(255, 255, 255, 0.5);
        }

        .compact .message-flow-container {
            height: 400px;
        }

        .compact .message {
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .compact .message-header {
            margin-bottom: 0.25rem;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .status-connected { background: #00d4aa; box-shadow: 0 0 6px #00d4aa; }
        .status-disconnected { background: #ff4444; }
        .status-connecting { background: #ffd700; animation: pulse 1s infinite; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    `;

    static properties = {
        messages: { type: Array },
        compact: { type: Boolean },
        autoScroll: { type: Boolean },
        filters: { type: Array },
        activeFilters: { type: Array },
        connectionStatus: { type: String }
    };

    constructor() {
        super();
        this.messages = [];
        this.compact = false;
        this.autoScroll = true;
        this.filters = ['system', 'agent', 'error', 'broadcast'];
        this.activeFilters = [...this.filters];
        this.connectionStatus = 'connecting';
    }

    connectedCallback() {
        super.connectedCallback();
        this.initializeWebSocket();
        this.loadRecentMessages();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.ws) {
            this.ws.close();
        }
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
    }

    initializeWebSocket() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/messages`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.connectionStatus = 'connected';
                this.requestUpdate();
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.addMessage(message);
            };
            
            this.ws.onclose = () => {
                this.connectionStatus = 'disconnected';
                this.requestUpdate();
                // Attempt reconnection after 5 seconds
                setTimeout(() => this.initializeWebSocket(), 5000);
            };
            
            this.ws.onerror = () => {
                this.connectionStatus = 'disconnected';
                this.requestUpdate();
            };
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.connectionStatus = 'disconnected';
            // Fallback to polling
            this.startPolling();
        }
    }

    startPolling() {
        this.pollInterval = setInterval(() => {
            this.loadRecentMessages();
        }, 2000); // Poll every 2 seconds
    }

    async loadRecentMessages() {
        try {
            const response = await fetch('/api/v1/messages?limit=50');
            if (response.ok) {
                const data = await response.json();
                this.messages = data.data || [];
                this.requestUpdate();
            }
        } catch (error) {
            console.error('Failed to load messages:', error);
        }
    }

    addMessage(message) {
        this.messages = [message, ...this.messages.slice(0, 99)]; // Keep last 100 messages
        this.requestUpdate();
        
        if (this.autoScroll) {
            this.updateComplete.then(() => {
                const container = this.shadowRoot.querySelector('.messages-container');
                if (container) {
                    container.scrollTop = 0; // Scroll to top for newest messages
                }
            });
        }
    }

    toggleFilter(filter) {
        if (this.activeFilters.includes(filter)) {
            this.activeFilters = this.activeFilters.filter(f => f !== filter);
        } else {
            this.activeFilters = [...this.activeFilters, filter];
        }
        this.requestUpdate();
    }

    getFilteredMessages() {
        return this.messages.filter(message => 
            this.activeFilters.includes(message.type || 'system')
        );
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }

    getMessageTypeClass(type) {
        switch (type) {
            case 'system': return 'type-system';
            case 'agent': return 'type-agent';
            case 'error': return 'type-error';
            case 'broadcast': return 'type-broadcast';
            default: return 'type-system';
        }
    }

    render() {
        const filteredMessages = this.getFilteredMessages();
        const stats = this.calculateStats();

        return html`
            <div class="${this.compact ? 'compact' : ''}">
                <div class="message-flow-container">
                    <div class="flow-header">
                        <div class="flow-title">Message Flow</div>
                        <div class="connection-status">
                            <div class="status-dot status-${this.connectionStatus}"></div>
                            <span>${this.connectionStatus}</span>
                        </div>
                        <div class="flow-controls">
                            <button 
                                class="control-button ${this.autoScroll ? 'active' : ''}"
                                @click=${() => { this.autoScroll = !this.autoScroll; }}
                            >
                                Auto-scroll
                            </button>
                            <button 
                                class="control-button"
                                @click=${() => { this.messages = []; }}
                            >
                                Clear
                            </button>
                        </div>
                    </div>

                    <div class="filter-section">
                        ${this.filters.map(filter => html`
                            <div 
                                class="filter-chip ${this.activeFilters.includes(filter) ? 'active' : ''}"
                                @click=${() => this.toggleFilter(filter)}
                            >
                                ${filter} (${stats[filter] || 0})
                            </div>
                        `)}
                    </div>

                    <div class="messages-container">
                        ${filteredMessages.length === 0 ? html`
                            <div class="empty-state">
                                No messages matching current filters
                            </div>
                        ` : filteredMessages.map(message => html`
                            <div class="message ${message.type || 'system'}">
                                <div class="message-header">
                                    <div class="message-info">
                                        <div class="message-type ${this.getMessageTypeClass(message.type)}">
                                            ${message.type || 'system'}
                                        </div>
                                        <div class="message-agents">
                                            ${message.from_agent} → ${message.to_agent || 'broadcast'}
                                        </div>
                                    </div>
                                    <div class="message-timestamp">
                                        ${this.formatTimestamp(message.timestamp)}
                                    </div>
                                </div>
                                <div class="message-content">
                                    ${typeof message.payload === 'string' ? message.payload : JSON.stringify(message.payload, null, 2)}
                                </div>
                                ${message.topic ? html`
                                    <div class="message-topic">Topic: ${message.topic}</div>
                                ` : ''}
                            </div>
                        `)}
                    </div>

                    <div class="stats-bar">
                        <span>Total: ${this.messages.length} messages</span>
                        <span>Filtered: ${filteredMessages.length} shown</span>
                        <span>Rate: ${stats.rate || 0} msg/min</span>
                    </div>
                </div>
            </div>
        `;
    }

    calculateStats() {
        const stats = {
            system: 0,
            agent: 0,
            error: 0,
            broadcast: 0,
            rate: 0
        };

        this.messages.forEach(message => {
            const type = message.type || 'system';
            if (stats.hasOwnProperty(type)) {
                stats[type]++;
            }
        });

        // Calculate message rate (messages per minute in last 5 minutes)
        const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
        const recentMessages = this.messages.filter(message => 
            new Date(message.timestamp) > fiveMinutesAgo
        );
        stats.rate = Math.round(recentMessages.length / 5);

        return stats;
    }
}

customElements.define('message-flow', MessageFlow);