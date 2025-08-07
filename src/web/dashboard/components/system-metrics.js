import { LitElement, html, css } from 'lit';

export class SystemMetrics extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00d4aa;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }

        .metric-change {
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }

        .metric-up { color: #00d4aa; }
        .metric-down { color: #ff4444; }
        .metric-stable { color: rgba(255, 255, 255, 0.6); }

        .charts-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 2rem;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .chart-title {
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }

        .simple-chart {
            height: 200px;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 4px;
            padding: 1rem 0;
        }

        .chart-bar {
            background: linear-gradient(to top, #00d4aa, rgba(0, 212, 170, 0.3));
            min-height: 10px;
            border-radius: 2px 2px 0 0;
            transition: all 0.3s ease;
        }

        .chart-bar:hover {
            background: linear-gradient(to top, #00e5bb, rgba(0, 229, 187, 0.5));
        }

        .performance-indicators {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }

        .indicator {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
        }

        .indicator-value {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .indicator-health { color: #00d4aa; }
        .indicator-warning { color: #ffd700; }
        .indicator-error { color: #ff4444; }

        .compact .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .compact .charts-section {
            grid-template-columns: 1fr;
        }

        .compact .performance-indicators {
            grid-template-columns: 1fr;
        }

        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr 1fr;
            }
            
            .charts-section {
                grid-template-columns: 1fr;
            }
        }
    `;

    static properties = {
        metrics: { type: Object },
        compact: { type: Boolean },
        chartData: { type: Array }
    };

    constructor() {
        super();
        this.metrics = {
            activeAgents: 0,
            completedTasks: 0,
            failedTasks: 0,
            systemHealth: 100,
            cpuUsage: 0,
            memoryUsage: 0,
            queueDepth: 0,
            averageResponseTime: 0
        };
        this.compact = false;
        this.chartData = [];
    }

    connectedCallback() {
        super.connectedCallback();
        this.loadMetrics();
        this.generateChartData();
        this.initializeWebSocketListeners();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeWebSocketListeners();
    }

    initializeWebSocketListeners() {
        // Listen for real-time system status updates
        this.handleSystemStatus = (event) => {
            const systemData = event.detail;
            
            // Update metrics from WebSocket data
            this.metrics = {
                ...this.metrics,
                activeAgents: systemData.active_agents || this.metrics.activeAgents,
                completedTasks: systemData.completed_tasks || this.metrics.completedTasks,
                queueDepth: systemData.queue_depth || this.metrics.queueDepth,
                systemHealth: Math.round((systemData.health_score || 0) * 100),
                uptime: systemData.uptime || this.metrics.uptime
            };
            
            // Update chart data with new point
            if (this.chartData.length >= 24) {
                this.chartData.shift(); // Remove oldest data point
            }
            
            this.chartData.push({
                time: new Date().getHours(),
                tasks: systemData.completed_tasks || 0,
                performance: Math.round((systemData.health_score || 0) * 100)
            });
            
            this.requestUpdate();
        };

        // Listen for metrics updates
        this.handleMetricsUpdate = (event) => {
            const metricsData = event.detail;
            
            // Process metrics data and update display
            if (Array.isArray(metricsData)) {
                // Calculate aggregated metrics from individual metrics
                const latestMetrics = {};
                metricsData.forEach(metric => {
                    if (metric.name === 'cpu_usage') latestMetrics.cpuUsage = metric.value;
                    if (metric.name === 'memory_usage') latestMetrics.memoryUsage = metric.value;
                    if (metric.name === 'response_time') latestMetrics.averageResponseTime = metric.value;
                });
                
                this.metrics = { ...this.metrics, ...latestMetrics };
                this.requestUpdate();
            }
        };

        window.addEventListener('hive-status-update', this.handleSystemStatus);
        window.addEventListener('metrics-update', this.handleMetricsUpdate);
    }

    removeWebSocketListeners() {
        if (this.handleSystemStatus) {
            window.removeEventListener('hive-status-update', this.handleSystemStatus);
        }
        if (this.handleMetricsUpdate) {
            window.removeEventListener('metrics-update', this.handleMetricsUpdate);
        }
    }

    async loadMetrics() {
        try {
            const response = await fetch('/api/v1/metrics');
            if (response.ok) {
                const data = await response.json();
                this.metrics = { ...this.metrics, ...data.data };
            }
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    generateChartData() {
        // Generate sample chart data for demonstration
        this.chartData = Array.from({ length: 24 }, (_, i) => ({
            time: i,
            tasks: Math.floor(Math.random() * 50) + 10,
            performance: Math.floor(Math.random() * 30) + 70
        }));
    }

    getHealthStatus(value) {
        if (value >= 90) return 'indicator-health';
        if (value >= 70) return 'indicator-warning';
        return 'indicator-error';
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    render() {
        return html`
            <div class="${this.compact ? 'compact' : ''}">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">${this.metrics.activeAgents}</div>
                        <div class="metric-label">Active Agents</div>
                        <div class="metric-change metric-stable">No change</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-value">${this.metrics.completedTasks}</div>
                        <div class="metric-label">Tasks Completed</div>
                        <div class="metric-change metric-up">↗ +12 this hour</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-value">${this.metrics.queueDepth}</div>
                        <div class="metric-label">Queue Depth</div>
                        <div class="metric-change metric-down">↘ -3 since last check</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-value">${this.metrics.averageResponseTime}ms</div>
                        <div class="metric-label">Avg Response Time</div>
                        <div class="metric-change metric-up">↗ +5ms</div>
                    </div>
                </div>

                ${!this.compact ? html`
                    <div class="charts-section">
                        <div class="chart-container">
                            <div class="chart-title">Task Completion (24h)</div>
                            <div class="simple-chart">
                                ${this.chartData.map(point => html`
                                    <div 
                                        class="chart-bar" 
                                        style="height: ${(point.tasks / 60) * 100}%; flex: 1;"
                                        title="${point.tasks} tasks at hour ${point.time}"
                                    ></div>
                                `)}
                            </div>
                        </div>

                        <div class="chart-container">
                            <div class="chart-title">System Performance (24h)</div>
                            <div class="simple-chart">
                                ${this.chartData.map(point => html`
                                    <div 
                                        class="chart-bar" 
                                        style="height: ${point.performance}%; flex: 1;"
                                        title="${point.performance}% at hour ${point.time}"
                                    ></div>
                                `)}
                            </div>
                        </div>
                    </div>

                    <div class="performance-indicators">
                        <div class="indicator">
                            <div class="indicator-value ${this.getHealthStatus(this.metrics.systemHealth)}">
                                ${this.metrics.systemHealth}%
                            </div>
                            <div>System Health</div>
                        </div>

                        <div class="indicator">
                            <div class="indicator-value ${this.getHealthStatus(100 - this.metrics.cpuUsage)}">
                                ${this.metrics.cpuUsage}%
                            </div>
                            <div>CPU Usage</div>
                        </div>

                        <div class="indicator">
                            <div class="indicator-value ${this.getHealthStatus(100 - this.metrics.memoryUsage)}">
                                ${this.metrics.memoryUsage}%
                            </div>
                            <div>Memory Usage</div>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }
}

customElements.define('system-metrics', SystemMetrics);