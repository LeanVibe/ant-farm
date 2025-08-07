import { LitElement, html, css } from 'lit';
import { pwaManager } from '../services/pwa-manager.js';

export class HiveDashboard extends LitElement {
    static styles = css`
        :host {
            display: block;
            min-height: 100vh;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .dashboard {
            display: grid;
            grid-template-columns: 250px 1fr;
            grid-template-rows: 60px 1fr;
            grid-template-areas: 
                "sidebar header"
                "sidebar main";
            min-height: 100vh;
        }

        .header {
            grid-area: header;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00d4aa;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00d4aa;
            box-shadow: 0 0 10px #00d4aa;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .sidebar {
            grid-area: sidebar;
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            color: rgba(255, 255, 255, 0.7);
        }

        .nav-item:hover, .nav-item.active {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
        }

        .main {
            grid-area: main;
            padding: 2rem;
            overflow-y: auto;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .widget {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }

        .widget-header {
            padding: 1rem 1.5rem;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 600;
        }

        .widget-content {
            padding: 1.5rem;
        }

        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
                grid-template-rows: 60px auto 1fr;
                grid-template-areas: 
                    "header"
                    "sidebar"
                    "main";
            }

            .sidebar {
                display: flex;
                overflow-x: auto;
                padding: 1rem;
            }

            .nav-item {
                white-space: nowrap;
                margin-right: 1rem;
            }

            .grid-layout {
                grid-template-columns: 1fr;
            }
        }
    `;

    static properties = {
        activeView: { type: String },
        systemStatus: { type: Object },
        notifications: { type: Array }
    };

    constructor() {
        super();
        this.activeView = 'overview';
        this.systemStatus = {
            health: 'healthy',
            activeAgents: 0,
            completedTasks: 0,
            uptime: 0
        };
        this.notifications = [];
    }

    connectedCallback() {
        super.connectedCallback();
        this.initializeWebSocket();
        this.loadSystemStatus();
        this.initializePWA();
    }

    initializePWA() {
        // Initialize PWA features
        if (pwaManager) {
            // Set up offline status indicator
            window.addEventListener('online', () => {
                this.updateConnectivityStatus(true);
            });
            
            window.addEventListener('offline', () => {
                this.updateConnectivityStatus(false);
            });
            
            // Check initial connectivity
            this.updateConnectivityStatus(navigator.onLine);
        }
    }

    updateConnectivityStatus(isOnline) {
        const statusIndicator = this.shadowRoot?.querySelector('.status-indicator span');
        if (statusIndicator) {
            statusIndicator.textContent = isOnline ? 'System Operational' : 'Offline Mode';
        }
    }

    initializeWebSocket() {
        // WebSocket connection will be handled by websocket-service.js
        window.addEventListener('hive-status-update', (e) => {
            this.systemStatus = e.detail;
            this.requestUpdate();
        });

        window.addEventListener('hive-notification', (e) => {
            this.notifications = [...this.notifications, e.detail];
            this.requestUpdate();
        });
    }

    async loadSystemStatus() {
        try {
            const response = await fetch('/api/v1/status');
            if (response.ok) {
                const data = await response.json();
                this.systemStatus = data.data;
            }
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }

    render() {
        return html`
            <div class="dashboard">
                <header class="header">
                    <div class="logo">🤖 LeanVibe Hive 2.0</div>
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span>System Operational</span>
                    </div>
                </header>

                <nav class="sidebar">
                    ${this.renderNavigation()}
                </nav>

                <main class="main">
                    ${this.renderActiveView()}
                </main>
            </div>
        `;
    }

    renderNavigation() {
        const navItems = [
            { id: 'overview', icon: '📊', label: 'Overview' },
            { id: 'agents', icon: '🤖', label: 'Agents' },
            { id: 'tasks', icon: '📋', label: 'Tasks' },
            { id: 'messages', icon: '💬', label: 'Messages' },
            { id: 'metrics', icon: '📈', label: 'Metrics' },
            { id: 'context', icon: '🧠', label: 'Memory' },
            { id: 'logs', icon: '📝', label: 'Logs' }
        ];

        return navItems.map(item => html`
            <div 
                class="nav-item ${this.activeView === item.id ? 'active' : ''}"
                @click=${() => this.setActiveView(item.id)}
            >
                <span>${item.icon}</span>
                <span>${item.label}</span>
            </div>
        `);
    }

    renderActiveView() {
        switch (this.activeView) {
            case 'overview':
                return this.renderOverview();
            case 'agents':
                return html`<agent-status></agent-status>`;
            case 'tasks':
                return html`<task-board></task-board>`;
            case 'messages':
                return html`<message-flow></message-flow>`;
            case 'metrics':
                return html`<system-metrics></system-metrics>`;
            case 'context':
                return html`<context-explorer></context-explorer>`;
            case 'logs':
                return html`<log-viewer></log-viewer>`;
            default:
                return this.renderOverview();
        }
    }

    renderOverview() {
        return html`
            <div class="grid-layout">
                <div class="widget">
                    <div class="widget-header">System Status</div>
                    <div class="widget-content">
                        <agent-status compact></agent-status>
                    </div>
                </div>

                <div class="widget">
                    <div class="widget-header">Active Tasks</div>
                    <div class="widget-content">
                        <task-board compact></task-board>
                    </div>
                </div>

                <div class="widget">
                    <div class="widget-header">Performance Metrics</div>
                    <div class="widget-content">
                        <system-metrics compact></system-metrics>
                    </div>
                </div>

                <div class="widget">
                    <div class="widget-header">Recent Activity</div>
                    <div class="widget-content">
                        <log-viewer compact limit="10"></log-viewer>
                    </div>
                </div>
            </div>
        `;
    }

    setActiveView(view) {
        this.activeView = view;
        this.requestUpdate();
    }
}

customElements.define('hive-dashboard', HiveDashboard);