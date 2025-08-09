/**
 * Autonomous Development Workflow (ADW) Monitoring Component
 * 
 * Real-time monitoring dashboard for extended autonomous development sessions.
 * Provides visualization of cognitive load, failure prediction, and system health.
 */

import { LitElement, html, css } from 'lit';
import { WebSocketService } from '../services/websocket-service.js';
import { ApiService } from '../services/api-service.js';

export class AdwMonitoringComponent extends LitElement {
    static styles = css`
        :host {
            display: block;
            padding: 1rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .monitoring-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .monitor-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1rem;
            color: #e0e0e0;
        }

        .monitor-card h3 {
            margin: 0 0 1rem 0;
            color: #fff;
            font-size: 1.1rem;
            border-bottom: 1px solid #333;
            padding-bottom: 0.5rem;
        }

        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.5rem 0;
            padding: 0.25rem 0;
        }

        .metric-value {
            font-weight: bold;
            font-family: 'SF Mono', 'Monaco', monospace;
        }

        .status-good { color: #4ade80; }
        .status-warning { color: #fbbf24; }
        .status-critical { color: #ef4444; }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ade80, #22c55e);
            transition: width 0.3s ease;
        }

        .progress-fill.warning {
            background: linear-gradient(90deg, #fbbf24, #f59e0b);
        }

        .progress-fill.critical {
            background: linear-gradient(90deg, #ef4444, #dc2626);
        }

        .session-timeline {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .timeline-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .timeline-phases {
            display: flex;
            height: 40px;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #333;
        }

        .phase-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8rem;
            font-weight: bold;
            position: relative;
            transition: all 0.3s ease;
        }

        .phase-reconnaissance { background: #3b82f6; }
        .phase-development { background: #10b981; }
        .phase-integration { background: #f59e0b; }
        .phase-learning { background: #8b5cf6; }
        .phase-rest { background: #6b7280; }

        .active-phase {
            box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.3);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .metrics-charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 2rem;
        }

        .chart-container {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1rem;
            min-height: 200px;
        }

        .mini-chart {
            width: 100%;
            height: 60px;
            background: #0a0a0a;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }

        .chart-line {
            stroke: #4ade80;
            stroke-width: 2;
            fill: none;
        }

        .alerts-panel {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .alert-item {
            display: flex;
            align-items: center;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            font-size: 0.9rem;
        }

        .alert-warning {
            background: rgba(251, 191, 36, 0.1);
            border-left: 3px solid #fbbf24;
        }

        .alert-critical {
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
        }

        .alert-info {
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
        }

        .session-controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .control-button {
            padding: 0.5rem 1rem;
            border: 1px solid #333;
            background: #2a2a2a;
            color: #e0e0e0;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .control-button:hover {
            background: #3a3a3a;
            border-color: #555;
        }

        .control-button.active {
            background: #4ade80;
            color: #000;
            border-color: #4ade80;
        }
    `;

    static properties = {
        sessionData: { type: Object },
        cognitiveState: { type: Object },
        failurePredictions: { type: Object },
        systemMetrics: { type: Object },
        currentPhase: { type: String },
        alerts: { type: Array },
        isMonitoring: { type: Boolean },
        sessionHistory: { type: Array }
    };

    constructor() {
        super();
        this.sessionData = {};
        this.cognitiveState = {};
        this.failurePredictions = {};
        this.systemMetrics = {};
        this.currentPhase = 'reconnaissance';
        this.alerts = [];
        this.isMonitoring = false;
        this.sessionHistory = [];
        
        this.wsService = new WebSocketService();
        this.apiService = new ApiService();
        
        this.initializeMonitoring();
    }

    async initializeMonitoring() {
        // Connect to real-time updates
        await this.wsService.connect();
        
        // Subscribe to ADW monitoring events
        this.wsService.subscribe('adw_metrics', (data) => {
            this.updateMetrics(data);
        });
        
        this.wsService.subscribe('adw_alerts', (alert) => {
            this.addAlert(alert);
        });
        
        this.wsService.subscribe('adw_phase_change', (phase) => {
            this.currentPhase = phase.name;
            this.requestUpdate();
        });
        
        // Start periodic data fetching
        this.startDataRefresh();
    }

    startDataRefresh() {
        setInterval(async () => {
            if (this.isMonitoring) {
                await this.fetchLatestData();
            }
        }, 5000); // Refresh every 5 seconds
    }

    async fetchLatestData() {
        try {
            const [metrics, cognitive, predictions] = await Promise.all([
                this.apiService.get('/api/v1/adw/metrics/current'),
                this.apiService.get('/api/v1/adw/cognitive/state'),
                this.apiService.get('/api/v1/adw/predictions/current')
            ]);
            
            this.systemMetrics = metrics.data || {};
            this.cognitiveState = cognitive.data || {};
            this.failurePredictions = predictions.data || {};
            
            this.requestUpdate();
        } catch (error) {
            console.error('Failed to fetch ADW data:', error);
            this.addAlert({
                type: 'warning',
                message: 'Failed to fetch monitoring data',
                timestamp: Date.now()
            });
        }
    }

    updateMetrics(data) {
        this.sessionData = { ...this.sessionData, ...data };
        this.requestUpdate();
    }

    addAlert(alert) {
        this.alerts = [
            { ...alert, id: Date.now() },
            ...this.alerts.slice(0, 9) // Keep last 10 alerts
        ];
        this.requestUpdate();
    }

    async toggleMonitoring() {
        if (this.isMonitoring) {
            await this.stopMonitoring();
        } else {
            await this.startMonitoring();
        }
    }

    async startMonitoring() {
        try {
            await this.apiService.post('/api/v1/adw/monitoring/start');
            this.isMonitoring = true;
            this.addAlert({
                type: 'info',
                message: 'ADW monitoring started',
                timestamp: Date.now()
            });
        } catch (error) {
            this.addAlert({
                type: 'critical',
                message: 'Failed to start monitoring',
                timestamp: Date.now()
            });
        }
    }

    async stopMonitoring() {
        try {
            await this.apiService.post('/api/v1/adw/monitoring/stop');
            this.isMonitoring = false;
            this.addAlert({
                type: 'info',
                message: 'ADW monitoring stopped',
                timestamp: Date.now()
            });
        } catch (error) {
            this.addAlert({
                type: 'warning',
                message: 'Failed to stop monitoring cleanly',
                timestamp: Date.now()
            });
        }
    }

    getProgressColor(value, thresholds = { warning: 0.7, critical: 0.9 }) {
        if (value >= thresholds.critical) return 'critical';
        if (value >= thresholds.warning) return 'warning';
        return 'good';
    }

    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }

    formatPercentage(value) {
        return `${Math.round(value * 100)}%`;
    }

    renderSessionControls() {
        return html`
            <div class="session-controls">
                <button 
                    class="control-button ${this.isMonitoring ? 'active' : ''}"
                    @click="${this.toggleMonitoring}"
                >
                    ${this.isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
                </button>
                <button class="control-button" @click="${this.exportMetrics}">
                    Export Data
                </button>
                <button class="control-button" @click="${this.resetAlerts}">
                    Clear Alerts
                </button>
            </div>
        `;
    }

    renderOverviewCards() {
        const sessionDuration = this.sessionData.duration || 0;
        const cognitiveLoad = this.cognitiveState.fatigue_level || 0;
        const failureRisk = this.failurePredictions.risk_score || 0;
        const autonomyScore = this.systemMetrics.autonomy_score || 0;

        return html`
            <div class="monitoring-grid">
                <div class="monitor-card">
                    <h3>Session Overview</h3>
                    <div class="metric-item">
                        <span>Duration</span>
                        <span class="metric-value">${this.formatDuration(sessionDuration)}</span>
                    </div>
                    <div class="metric-item">
                        <span>Current Phase</span>
                        <span class="metric-value">${this.currentPhase}</span>
                    </div>
                    <div class="metric-item">
                        <span>Commits Made</span>
                        <span class="metric-value">${this.sessionData.commits || 0}</span>
                    </div>
                    <div class="metric-item">
                        <span>Tests Passed</span>
                        <span class="metric-value">${this.sessionData.tests_passed || 0}/${this.sessionData.tests_total || 0}</span>
                    </div>
                </div>

                <div class="monitor-card">
                    <h3>Cognitive State</h3>
                    <div class="metric-item">
                        <span>Fatigue Level</span>
                        <span class="metric-value status-${this.getProgressColor(cognitiveLoad)}">
                            ${this.formatPercentage(cognitiveLoad)}
                        </span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${this.getProgressColor(cognitiveLoad)}" 
                             style="width: ${cognitiveLoad * 100}%"></div>
                    </div>
                    <div class="metric-item">
                        <span>Focus Efficiency</span>
                        <span class="metric-value">${this.formatPercentage(this.cognitiveState.focus_efficiency || 0)}</span>
                    </div>
                    <div class="metric-item">
                        <span>Current Mode</span>
                        <span class="metric-value">${this.cognitiveState.mode || 'focus'}</span>
                    </div>
                </div>

                <div class="monitor-card">
                    <h3>Failure Prediction</h3>
                    <div class="metric-item">
                        <span>Risk Score</span>
                        <span class="metric-value status-${this.getProgressColor(failureRisk)}">
                            ${this.formatPercentage(failureRisk)}
                        </span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${this.getProgressColor(failureRisk)}" 
                             style="width: ${failureRisk * 100}%"></div>
                    </div>
                    <div class="metric-item">
                        <span>Next Check</span>
                        <span class="metric-value">${this.failurePredictions.next_check || 'N/A'}</span>
                    </div>
                    <div class="metric-item">
                        <span>Confidence</span>
                        <span class="metric-value">${this.formatPercentage(this.failurePredictions.confidence || 0)}</span>
                    </div>
                </div>

                <div class="monitor-card">
                    <h3>System Health</h3>
                    <div class="metric-item">
                        <span>Autonomy Score</span>
                        <span class="metric-value status-good">${Math.round(autonomyScore)}</span>
                    </div>
                    <div class="metric-item">
                        <span>Memory Usage</span>
                        <span class="metric-value">${Math.round(this.systemMetrics.memory_percent || 0)}%</span>
                    </div>
                    <div class="metric-item">
                        <span>CPU Usage</span>
                        <span class="metric-value">${Math.round(this.systemMetrics.cpu_percent || 0)}%</span>
                    </div>
                    <div class="metric-item">
                        <span>Uptime</span>
                        <span class="metric-value">${this.formatDuration(this.systemMetrics.uptime || 0)}</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderSessionTimeline() {
        const phases = [
            { name: 'reconnaissance', label: 'Recon', duration: 15 },
            { name: 'development', label: 'Dev', duration: 180 },
            { name: 'integration', label: 'Test', duration: 30 },
            { name: 'learning', label: 'Learn', duration: 15 }
        ];

        const totalDuration = phases.reduce((sum, phase) => sum + phase.duration, 0);

        return html`
            <div class="session-timeline">
                <div class="timeline-header">
                    <h3>Session Timeline</h3>
                    <span>${this.formatDuration(this.sessionData.duration || 0)} / ${this.formatDuration(totalDuration * 60)}</span>
                </div>
                <div class="timeline-phases">
                    ${phases.map(phase => html`
                        <div class="phase-segment phase-${phase.name} ${this.currentPhase === phase.name ? 'active-phase' : ''}"
                             style="flex: ${phase.duration}">
                            ${phase.label}
                        </div>
                    `)}
                </div>
            </div>
        `;
    }

    renderMetricsCharts() {
        return html`
            <div class="metrics-charts">
                <div class="chart-container">
                    <h3>Performance Trend</h3>
                    <div class="mini-chart">
                        <svg width="100%" height="100%">
                            <polyline class="chart-line" 
                                      points="0,50 50,30 100,20 150,25 200,15 250,10"></polyline>
                        </svg>
                    </div>
                    <div class="metric-item">
                        <span>Commits/Hour</span>
                        <span class="metric-value">${(this.sessionData.commits_per_hour || 0).toFixed(1)}</span>
                    </div>
                </div>

                <div class="chart-container">
                    <h3>Resource Usage</h3>
                    <div class="mini-chart">
                        <svg width="100%" height="100%">
                            <polyline class="chart-line" 
                                      points="0,40 50,45 100,50 150,48 200,55 250,52"></polyline>
                        </svg>
                    </div>
                    <div class="metric-item">
                        <span>Avg Memory</span>
                        <span class="metric-value">${Math.round(this.systemMetrics.avg_memory || 0)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderAlertsPanel() {
        return html`
            <div class="alerts-panel">
                <h3>Recent Alerts</h3>
                ${this.alerts.length === 0 ? 
                    html`<div class="metric-item">No alerts</div>` :
                    this.alerts.map(alert => html`
                        <div class="alert-item alert-${alert.type}">
                            <span>${alert.message}</span>
                            <span>${new Date(alert.timestamp).toLocaleTimeString()}</span>
                        </div>
                    `)
                }
            </div>
        `;
    }

    async exportMetrics() {
        try {
            const data = {
                sessionData: this.sessionData,
                cognitiveState: this.cognitiveState,
                failurePredictions: this.failurePredictions,
                systemMetrics: this.systemMetrics,
                alerts: this.alerts,
                timestamp: Date.now()
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `adw-metrics-${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            this.addAlert({
                type: 'info',
                message: 'Metrics exported successfully',
                timestamp: Date.now()
            });
        } catch (error) {
            this.addAlert({
                type: 'warning',
                message: 'Failed to export metrics',
                timestamp: Date.now()
            });
        }
    }

    resetAlerts() {
        this.alerts = [];
        this.requestUpdate();
    }

    render() {
        return html`
            ${this.renderSessionControls()}
            ${this.renderOverviewCards()}
            ${this.renderSessionTimeline()}
            ${this.renderMetricsCharts()}
            ${this.renderAlertsPanel()}
        `;
    }
}

customElements.define('adw-monitoring', AdwMonitoringComponent);