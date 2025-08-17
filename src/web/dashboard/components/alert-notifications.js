import { LitElement, html, css } from 'lit';

export class AlertNotifications extends LitElement {
    static styles = css`
        :host {
            display: block;
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
            max-width: 400px;
        }

        .alerts-container {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .alert-item {
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(10px);
            color: white;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            opacity: 0;
            transform: translateX(100%);
            animation: slideIn 0.3s ease-out forwards;
            position: relative;
            overflow: hidden;
        }

        .alert-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--alert-color), transparent);
            animation: shimmer 2s ease-in-out infinite;
        }

        @keyframes slideIn {
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes slideOut {
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }

        @keyframes shimmer {
            0%, 100% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
        }

        .alert-critical {
            border-left-color: #ff4444;
            --alert-color: #ff4444;
        }

        .alert-warning {
            border-left-color: #ffd700;
            --alert-color: #ffd700;
        }

        .alert-info {
            border-left-color: #00d4aa;
            --alert-color: #00d4aa;
        }

        .alert-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .alert-title {
            font-weight: bold;
            font-size: 0.9rem;
        }

        .alert-severity {
            font-size: 0.8rem;
            text-transform: uppercase;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
        }

        .alert-message {
            font-size: 0.85rem;
            line-height: 1.4;
            margin-bottom: 0.5rem;
            color: rgba(255, 255, 255, 0.9);
        }

        .alert-metadata {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.7);
        }

        .alert-timestamp {
            font-family: monospace;
        }

        .alert-source {
            font-style: italic;
        }

        .close-button {
            background: none;
            border: none;
            color: rgba(255, 255, 255, 0.7);
            cursor: pointer;
            font-size: 1.2rem;
            padding: 0;
            margin-left: 0.5rem;
            transition: color 0.2s ease;
        }

        .close-button:hover {
            color: white;
        }

        .no-alerts {
            color: rgba(255, 255, 255, 0.5);
            text-align: center;
            padding: 1rem;
            font-style: italic;
        }

        .alert-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .alert-action {
            padding: 0.3rem 0.6rem;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            border-radius: 4px;
            color: white;
            font-size: 0.75rem;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .alert-action:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .compact .alert-item {
            padding: 0.75rem;
        }

        .compact .alert-title {
            font-size: 0.85rem;
        }

        .compact .alert-message {
            font-size: 0.8rem;
        }
    `;

    static properties = {
        alerts: { type: Array },
        maxAlerts: { type: Number },
        autoHideDelay: { type: Number },
        compact: { type: Boolean }
    };

    constructor() {
        super();
        this.alerts = [];
        this.maxAlerts = 5;
        this.autoHideDelay = 10000; // 10 seconds
        this.compact = false;
        this.alertTimeouts = new Map();
    }

    connectedCallback() {
        super.connectedCallback();
        this.initializeWebSocketListeners();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeWebSocketListeners();
        this.clearAllTimeouts();
    }

    initializeWebSocketListeners() {
        // Listen for alert updates from WebSocket
        this.handleAlertUpdate = (event) => {
            const alertData = event.detail;
            
            if (alertData.alert && alertData.action) {
                if (alertData.action === 'triggered' || alertData.action === 'created') {
                    this.addAlert(alertData.alert);
                } else if (alertData.action === 'resolved' || alertData.action === 'dismissed') {
                    this.removeAlert(alertData.alert.id);
                }
            }
        };

        // Listen for alert notifications
        this.handleAlertNotification = (event) => {
            const notification = event.detail;
            this.addAlert({
                id: `notification-${Date.now()}`,
                severity: notification.level || 'info',
                title: notification.title || 'Notification',
                message: notification.message || '',
                source: notification.source || 'system',
                timestamp: new Date().toISOString(),
                metadata: notification.metadata || {}
            });
        };

        // Listen for WebSocket errors
        this.handleWebSocketError = (event) => {
            const error = event.detail;
            this.addAlert({
                id: `ws-error-${Date.now()}`,
                severity: 'warning',
                title: 'WebSocket Error',
                message: error.message || 'Connection issue detected',
                source: 'websocket',
                timestamp: new Date().toISOString(),
                metadata: error
            });
        };

        window.addEventListener('alert-update', this.handleAlertUpdate);
        window.addEventListener('alert-notification', this.handleAlertNotification);
        window.addEventListener('websocket-error', this.handleWebSocketError);
    }

    removeWebSocketListeners() {
        if (this.handleAlertUpdate) {
            window.removeEventListener('alert-update', this.handleAlertUpdate);
        }
        if (this.handleAlertNotification) {
            window.removeEventListener('alert-notification', this.handleAlertNotification);
        }
        if (this.handleWebSocketError) {
            window.removeEventListener('websocket-error', this.handleWebSocketError);
        }
    }

    addAlert(alert) {
        // Ensure we have required properties
        const processedAlert = {
            id: alert.id || `alert-${Date.now()}`,
            severity: alert.severity || 'info',
            title: alert.title || alert.rule_name || 'Alert',
            message: alert.message || alert.description || '',
            source: alert.source || alert.agent_id || 'system',
            timestamp: alert.timestamp || alert.triggered_at || new Date().toISOString(),
            metadata: alert.metadata || alert.context || {}
        };

        // Remove oldest alerts if we exceed max
        if (this.alerts.length >= this.maxAlerts) {
            const oldestAlert = this.alerts[0];
            this.removeAlert(oldestAlert.id);
        }

        // Add new alert
        this.alerts = [...this.alerts, processedAlert];
        this.requestUpdate();

        // Set auto-hide timeout
        if (this.autoHideDelay > 0) {
            const timeout = setTimeout(() => {
                this.removeAlert(processedAlert.id);
            }, this.autoHideDelay);
            
            this.alertTimeouts.set(processedAlert.id, timeout);
        }

        // Emit custom event for other components
        this.dispatchEvent(new CustomEvent('alert-added', {
            detail: processedAlert,
            bubbles: true,
            composed: true
        }));
    }

    removeAlert(alertId) {
        const alertElement = this.shadowRoot?.querySelector(`[data-alert-id="${alertId}"]`);
        
        if (alertElement) {
            alertElement.style.animation = 'slideOut 0.3s ease-in forwards';
            
            setTimeout(() => {
                this.alerts = this.alerts.filter(alert => alert.id !== alertId);
                this.requestUpdate();
                
                // Clear timeout
                if (this.alertTimeouts.has(alertId)) {
                    clearTimeout(this.alertTimeouts.get(alertId));
                    this.alertTimeouts.delete(alertId);
                }
            }, 300);
        } else {
            // Fallback if element not found
            this.alerts = this.alerts.filter(alert => alert.id !== alertId);
            this.requestUpdate();
        }
    }

    clearAllTimeouts() {
        this.alertTimeouts.forEach(timeout => clearTimeout(timeout));
        this.alertTimeouts.clear();
    }

    dismissAlert(alertId) {
        this.removeAlert(alertId);
        
        // Emit dismiss event
        this.dispatchEvent(new CustomEvent('alert-dismissed', {
            detail: { alertId },
            bubbles: true,
            composed: true
        }));
    }

    acknowledgeAlert(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (alert) {
            // Mark as acknowledged
            alert.acknowledged = true;
            this.requestUpdate();
            
            // Emit acknowledge event
            this.dispatchEvent(new CustomEvent('alert-acknowledged', {
                detail: { alertId, alert },
                bubbles: true,
                composed: true
            }));
        }
    }

    getSeverityClass(severity) {
        switch (severity.toLowerCase()) {
            case 'critical':
            case 'error':
                return 'alert-critical';
            case 'warning':
            case 'warn':
                return 'alert-warning';
            case 'info':
            case 'information':
            default:
                return 'alert-info';
        }
    }

    formatTimestamp(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString();
        } catch (error) {
            return 'Unknown';
        }
    }

    render() {
        if (this.alerts.length === 0) {
            return html``;
        }

        return html`
            <div class="alerts-container ${this.compact ? 'compact' : ''}">
                ${this.alerts.map(alert => html`
                    <div 
                        class="alert-item ${this.getSeverityClass(alert.severity)}"
                        data-alert-id="${alert.id}"
                    >
                        <div class="alert-header">
                            <div class="alert-title">${alert.title}</div>
                            <div class="alert-severity">${alert.severity}</div>
                            <button 
                                class="close-button"
                                @click=${() => this.dismissAlert(alert.id)}
                                title="Dismiss"
                            >×</button>
                        </div>
                        
                        <div class="alert-message">${alert.message}</div>
                        
                        <div class="alert-metadata">
                            <span class="alert-source">Source: ${alert.source}</span>
                            <span class="alert-timestamp">${this.formatTimestamp(alert.timestamp)}</span>
                        </div>

                        ${alert.severity === 'critical' && !alert.acknowledged ? html`
                            <div class="alert-actions">
                                <button 
                                    class="alert-action"
                                    @click=${() => this.acknowledgeAlert(alert.id)}
                                >
                                    Acknowledge
                                </button>
                            </div>
                        ` : ''}
                    </div>
                `)}
            </div>
        `;
    }
}

customElements.define('alert-notifications', AlertNotifications);