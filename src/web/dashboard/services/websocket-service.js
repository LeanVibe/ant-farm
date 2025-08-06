// WebSocket Service for real-time dashboard updates
class WebSocketService {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.eventListeners = new Map();
    }

    connect() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/events`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.dispatchEvent('connection-status', { status: 'connected' });
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.dispatchEvent('connection-status', { status: 'disconnected' });
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.dispatchEvent('connection-status', { status: 'error' });
            };
            
        } catch (error) {
            console.error('Failed to establish WebSocket connection:', error);
            this.dispatchEvent('connection-status', { status: 'failed' });
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('Max reconnection attempts reached');
            this.dispatchEvent('connection-status', { status: 'failed' });
        }
    }

    handleMessage(data) {
        const { type, payload } = data;
        
        switch (type) {
            case 'system-status':
                this.dispatchEvent('hive-status-update', payload);
                break;
            case 'agent-update':
                this.dispatchEvent('agent-status-update', payload);
                break;
            case 'task-update':
                this.dispatchEvent('task-status-update', payload);
                break;
            case 'message':
                this.dispatchEvent('new-message', payload);
                break;
            case 'log':
                this.dispatchEvent('new-log', payload);
                break;
            case 'metrics':
                this.dispatchEvent('metrics-update', payload);
                break;
            case 'notification':
                this.dispatchEvent('hive-notification', payload);
                break;
            default:
                console.warn('Unknown WebSocket message type:', type);
        }
    }

    subscribe(eventType, callback) {
        if (!this.eventListeners.has(eventType)) {
            this.eventListeners.set(eventType, new Set());
        }
        this.eventListeners.get(eventType).add(callback);
        
        // Add DOM event listener
        window.addEventListener(eventType, callback);
    }

    unsubscribe(eventType, callback) {
        if (this.eventListeners.has(eventType)) {
            this.eventListeners.get(eventType).delete(callback);
        }
        
        // Remove DOM event listener
        window.removeEventListener(eventType, callback);
    }

    dispatchEvent(eventType, data) {
        const event = new CustomEvent(eventType, {
            detail: data,
            bubbles: true,
            composed: true
        });
        window.dispatchEvent(event);
    }

    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, cannot send message');
        }
    }

    getConnectionStatus() {
        if (!this.ws) return 'disconnected';
        
        switch (this.ws.readyState) {
            case WebSocket.CONNECTING:
                return 'connecting';
            case WebSocket.OPEN:
                return 'connected';
            case WebSocket.CLOSING:
                return 'disconnecting';
            case WebSocket.CLOSED:
                return 'disconnected';
            default:
                return 'unknown';
        }
    }
}

// Create singleton instance
const webSocketService = new WebSocketService();

// Auto-connect when page loads
document.addEventListener('DOMContentLoaded', () => {
    webSocketService.connect();
});

// Reconnect when page becomes visible again
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && webSocketService.getConnectionStatus() === 'disconnected') {
        webSocketService.connect();
    }
});

export { webSocketService };