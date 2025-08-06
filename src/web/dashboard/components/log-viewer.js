import { LitElement, html, css } from 'lit';

export class LogViewer extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .log-viewer {
            height: 600px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        .viewer-header {
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .viewer-title {
            font-weight: 600;
            color: #ffffff;
        }

        .viewer-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
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

        .log-filters {
            padding: 0.5rem 1rem;
            background: rgba(0, 0, 0, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .filter-group {
            display: flex;
            gap: 0.25rem;
            align-items: center;
        }

        .filter-label {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
            margin-right: 0.5rem;
        }

        .level-filter {
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .level-filter.active {
            border-color: currentColor;
        }

        .level-debug { background: rgba(128, 128, 128, 0.2); color: #808080; }
        .level-info { background: rgba(0, 212, 170, 0.2); color: #00d4aa; }
        .level-warning { background: rgba(255, 215, 0, 0.2); color: #ffd700; }
        .level-error { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
        .level-critical { background: rgba(139, 0, 0, 0.2); color: #8b0000; }

        .search-input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 0.5rem;
            color: #ffffff;
            font-size: 0.8rem;
            width: 200px;
        }

        .search-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        .log-content {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
            background: #000000;
        }

        .log-line {
            padding: 0.25rem 0.5rem;
            margin-bottom: 1px;
            border-left: 3px solid transparent;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.8rem;
            line-height: 1.4;
            transition: background 0.3s ease;
        }

        .log-line:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .log-line.highlighted {
            background: rgba(255, 215, 0, 0.2);
        }

        .log-debug {
            color: #808080;
            border-left-color: #808080;
        }

        .log-info {
            color: #00d4aa;
            border-left-color: #00d4aa;
        }

        .log-warning {
            color: #ffd700;
            border-left-color: #ffd700;
        }

        .log-error {
            color: #ff4444;
            border-left-color: #ff4444;
        }

        .log-critical {
            color: #8b0000;
            border-left-color: #8b0000;
            background: rgba(139, 0, 0, 0.1);
        }

        .log-timestamp {
            color: rgba(255, 255, 255, 0.5);
            margin-right: 1rem;
        }

        .log-level {
            display: inline-block;
            width: 60px;
            text-align: center;
            margin-right: 1rem;
            font-weight: bold;
        }

        .log-source {
            color: rgba(255, 255, 255, 0.7);
            margin-right: 1rem;
        }

        .log-message {
            color: inherit;
        }

        .status-bar {
            padding: 0.5rem 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
        }

        .connection-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .status-connected { background: #00d4aa; box-shadow: 0 0 6px #00d4aa; }
        .status-disconnected { background: #ff4444; }
        .status-streaming { background: #ffd700; animation: pulse 1s infinite; }

        .log-stats {
            display: flex;
            gap: 1rem;
        }

        .compact .log-viewer {
            height: 400px;
        }

        .compact .log-line {
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
        }

        .compact .viewer-header {
            padding: 0.75rem;
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: rgba(255, 255, 255, 0.5);
            text-align: center;
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: rgba(255, 255, 255, 0.7);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    `;

    static properties = {
        logs: { type: Array },
        filteredLogs: { type: Array },
        activeFilters: { type: Array },
        searchQuery: { type: String },
        autoScroll: { type: Boolean },
        isStreaming: { type: Boolean },
        connectionStatus: { type: String },
        compact: { type: Boolean },
        limit: { type: Number },
        stats: { type: Object }
    };

    constructor() {
        super();
        this.logs = [];
        this.filteredLogs = [];
        this.activeFilters = ['debug', 'info', 'warning', 'error', 'critical'];
        this.searchQuery = '';
        this.autoScroll = true;
        this.isStreaming = false;
        this.connectionStatus = 'disconnected';
        this.compact = false;
        this.limit = this.compact ? 10 : 100;
        this.stats = {
            total: 0,
            debug: 0,
            info: 0,
            warning: 0,
            error: 0,
            critical: 0
        };
    }

    connectedCallback() {
        super.connectedCallback();
        this.loadRecentLogs();
        this.initializeLogStream();
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

    async loadRecentLogs() {
        try {
            const response = await fetch(`/api/v1/logs?limit=${this.limit || 100}`);
            if (response.ok) {
                const data = await response.json();
                this.logs = data.data || [];
                this.updateFilteredLogs();
                this.updateStats();
            }
        } catch (error) {
            console.error('Failed to load logs:', error);
        }
    }

    initializeLogStream() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/logs`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.connectionStatus = 'connected';
                this.isStreaming = true;
                this.requestUpdate();
            };
            
            this.ws.onmessage = (event) => {
                const logEntry = JSON.parse(event.data);
                this.addLogEntry(logEntry);
            };
            
            this.ws.onclose = () => {
                this.connectionStatus = 'disconnected';
                this.isStreaming = false;
                this.requestUpdate();
                // Attempt reconnection after 5 seconds
                setTimeout(() => this.initializeLogStream(), 5000);
            };
            
            this.ws.onerror = () => {
                this.connectionStatus = 'disconnected';
                this.isStreaming = false;
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
            this.loadRecentLogs();
        }, 2000); // Poll every 2 seconds
    }

    addLogEntry(logEntry) {
        this.logs = [logEntry, ...this.logs.slice(0, (this.limit || 100) - 1)];
        this.updateFilteredLogs();
        this.updateStats();
        
        if (this.autoScroll) {
            this.updateComplete.then(() => {
                const container = this.shadowRoot.querySelector('.log-content');
                if (container) {
                    container.scrollTop = 0; // Scroll to top for newest logs
                }
            });
        }
    }

    updateFilteredLogs() {
        let filtered = this.logs;
        
        // Apply level filters
        if (this.activeFilters.length < 5) {
            filtered = filtered.filter(log => 
                this.activeFilters.includes(log.level?.toLowerCase() || 'info')
            );
        }
        
        // Apply search filter
        if (this.searchQuery) {
            const query = this.searchQuery.toLowerCase();
            filtered = filtered.filter(log => 
                log.message?.toLowerCase().includes(query) ||
                log.source?.toLowerCase().includes(query) ||
                log.level?.toLowerCase().includes(query)
            );
        }
        
        this.filteredLogs = filtered;
        this.requestUpdate();
    }

    updateStats() {
        this.stats = {
            total: this.logs.length,
            debug: this.logs.filter(log => log.level === 'debug').length,
            info: this.logs.filter(log => log.level === 'info').length,
            warning: this.logs.filter(log => log.level === 'warning').length,
            error: this.logs.filter(log => log.level === 'error').length,
            critical: this.logs.filter(log => log.level === 'critical').length
        };
    }

    toggleFilter(level) {
        if (this.activeFilters.includes(level)) {
            this.activeFilters = this.activeFilters.filter(f => f !== level);
        } else {
            this.activeFilters = [...this.activeFilters, level];
        }
        this.updateFilteredLogs();
    }

    handleSearch(e) {
        this.searchQuery = e.target.value;
        clearTimeout(this.searchTimeout);
        this.searchTimeout = setTimeout(() => {
            this.updateFilteredLogs();
        }, 300);
    }

    clearLogs() {
        this.logs = [];
        this.filteredLogs = [];
        this.updateStats();
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }

    getLogLevelClass(level) {
        return `log-${level?.toLowerCase() || 'info'}`;
    }

    highlightSearchTerm(text, searchTerm) {
        if (!searchTerm) return text;
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    render() {
        return html`
            <div class="${this.compact ? 'compact' : ''}">
                <div class="log-viewer">
                    <div class="viewer-header">
                        <div class="viewer-title">System Logs</div>
                        <div class="viewer-controls">
                            <button 
                                class="control-button ${this.autoScroll ? 'active' : ''}"
                                @click=${() => { this.autoScroll = !this.autoScroll; }}
                            >
                                Auto-scroll
                            </button>
                            <button 
                                class="control-button ${this.isStreaming ? 'active' : ''}"
                                @click=${() => this.isStreaming ? this.ws?.close() : this.initializeLogStream()}
                            >
                                ${this.isStreaming ? 'Stop' : 'Start'} Stream
                            </button>
                            <button class="control-button" @click=${this.clearLogs}>
                                Clear
                            </button>
                        </div>
                    </div>

                    <div class="log-filters">
                        <div class="filter-group">
                            <span class="filter-label">Levels:</span>
                            ${['debug', 'info', 'warning', 'error', 'critical'].map(level => html`
                                <div 
                                    class="level-filter level-${level} ${this.activeFilters.includes(level) ? 'active' : ''}"
                                    @click=${() => this.toggleFilter(level)}
                                >
                                    ${level.toUpperCase()} (${this.stats[level]})
                                </div>
                            `)}
                        </div>
                        
                        <input 
                            type="text" 
                            class="search-input"
                            placeholder="Search logs..."
                            .value=${this.searchQuery}
                            @input=${this.handleSearch}
                        >
                    </div>

                    <div class="log-content">
                        ${this.filteredLogs.length === 0 ? html`
                            <div class="empty-state">
                                <div>📝</div>
                                <div>No logs matching current filters</div>
                            </div>
                        ` : this.filteredLogs.map(log => html`
                            <div class="log-line ${this.getLogLevelClass(log.level)} ${this.searchQuery && log.message?.toLowerCase().includes(this.searchQuery.toLowerCase()) ? 'highlighted' : ''}">
                                <span class="log-timestamp">${this.formatTimestamp(log.timestamp)}</span>
                                <span class="log-level">${(log.level || 'INFO').toUpperCase()}</span>
                                <span class="log-source">${log.source || 'system'}</span>
                                <span class="log-message">${log.message || ''}</span>
                            </div>
                        `)}
                    </div>

                    <div class="status-bar">
                        <div class="connection-indicator">
                            <div class="status-dot status-${this.connectionStatus}"></div>
                            <span>${this.connectionStatus === 'connected' && this.isStreaming ? 'Streaming' : this.connectionStatus}</span>
                        </div>
                        
                        <div class="log-stats">
                            <span>Total: ${this.stats.total}</span>
                            <span>Shown: ${this.filteredLogs.length}</span>
                            <span>Errors: ${this.stats.error + this.stats.critical}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

customElements.define('log-viewer', LogViewer);