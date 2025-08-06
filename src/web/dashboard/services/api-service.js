// API Service for dashboard HTTP requests
class ApiService {
    constructor() {
        this.baseUrl = '/api/v1';
        this.defaultHeaders = {
            'Content-Type': 'application/json',
        };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }

    // System endpoints
    async getSystemStatus() {
        return this.request('/status');
    }

    async getSystemHealth() {
        return this.request('/health');
    }

    async getSystemMetrics() {
        return this.request('/metrics');
    }

    // Agent endpoints
    async getAgents() {
        return this.request('/agents');
    }

    async getAgent(agentId) {
        return this.request(`/agents/${agentId}`);
    }

    async createAgent(agentData) {
        return this.request('/agents', {
            method: 'POST',
            body: JSON.stringify(agentData)
        });
    }

    async updateAgent(agentId, agentData) {
        return this.request(`/agents/${agentId}`, {
            method: 'PUT',
            body: JSON.stringify(agentData)
        });
    }

    async deleteAgent(agentId) {
        return this.request(`/agents/${agentId}`, {
            method: 'DELETE'
        });
    }

    // Task endpoints
    async getTasks(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/tasks${queryString ? '?' + queryString : ''}`);
    }

    async getTask(taskId) {
        return this.request(`/tasks/${taskId}`);
    }

    async createTask(taskData) {
        return this.request('/tasks', {
            method: 'POST',
            body: JSON.stringify(taskData)
        });
    }

    async cancelTask(taskId) {
        return this.request(`/tasks/${taskId}/cancel`, {
            method: 'PUT'
        });
    }

    // Session endpoints
    async getSessions() {
        return this.request('/sessions');
    }

    async getSession(sessionId) {
        return this.request(`/sessions/${sessionId}`);
    }

    async createSession(sessionData) {
        return this.request('/sessions', {
            method: 'POST',
            body: JSON.stringify(sessionData)
        });
    }

    async startSession(sessionId) {
        return this.request(`/sessions/${sessionId}/start`, {
            method: 'POST'
        });
    }

    // Context endpoints
    async getContexts(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/contexts${queryString ? '?' + queryString : ''}`);
    }

    async getContext(contextId) {
        return this.request(`/contexts/${contextId}`);
    }

    async searchContexts(query, params = {}) {
        const searchParams = { query, ...params };
        const queryString = new URLSearchParams(searchParams).toString();
        return this.request(`/contexts/search?${queryString}`);
    }

    async getRelatedContexts(contextId) {
        return this.request(`/contexts/${contextId}/related`);
    }

    async getContextStats() {
        return this.request('/contexts/stats');
    }

    // Message endpoints
    async getMessages(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/messages${queryString ? '?' + queryString : ''}`);
    }

    async sendMessage(messageData) {
        return this.request('/messages', {
            method: 'POST',
            body: JSON.stringify(messageData)
        });
    }

    // Log endpoints
    async getLogs(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return this.request(`/logs${queryString ? '?' + queryString : ''}`);
    }

    // Performance and monitoring
    async getPerformanceMetrics(timeRange = '1h') {
        return this.request(`/metrics/performance?range=${timeRange}`);
    }

    async getAgentMetrics(agentId, timeRange = '1h') {
        return this.request(`/agents/${agentId}/metrics?range=${timeRange}`);
    }

    async getTaskMetrics(timeRange = '1h') {
        return this.request(`/tasks/metrics?range=${timeRange}`);
    }

    // Configuration
    async getConfig() {
        return this.request('/config');
    }

    async updateConfig(configData) {
        return this.request('/config', {
            method: 'PUT',
            body: JSON.stringify(configData)
        });
    }

    // Utility methods
    formatResponse(response) {
        if (response && response.success !== undefined) {
            return response.data;
        }
        return response;
    }

    handleError(error, context = '') {
        console.error(`API Error ${context}:`, error);
        
        // Dispatch error event for global handling
        window.dispatchEvent(new CustomEvent('api-error', {
            detail: { error, context },
            bubbles: true,
            composed: true
        }));
        
        throw error;
    }

    // Batch requests
    async batchRequest(requests) {
        try {
            const promises = requests.map(({ endpoint, options }) => 
                this.request(endpoint, options)
            );
            return await Promise.all(promises);
        } catch (error) {
            this.handleError(error, 'batch request');
        }
    }

    // Cache management
    createCachedRequest(endpoint, cacheDuration = 5000) {
        const cache = new Map();
        
        return async (params = {}) => {
            const key = JSON.stringify({ endpoint, params });
            const cached = cache.get(key);
            
            if (cached && Date.now() - cached.timestamp < cacheDuration) {
                return cached.data;
            }
            
            const data = await this.request(endpoint, params);
            cache.set(key, { data, timestamp: Date.now() });
            
            return data;
        };
    }

    // Polling utilities
    createPoller(endpoint, interval = 5000, callback) {
        let isPolling = false;
        let pollInterval;
        
        const poll = async () => {
            if (!isPolling) return;
            
            try {
                const data = await this.request(endpoint);
                callback(data);
            } catch (error) {
                console.error('Polling error:', error);
            }
        };
        
        return {
            start() {
                if (isPolling) return;
                isPolling = true;
                pollInterval = setInterval(poll, interval);
                poll(); // Initial poll
            },
            
            stop() {
                isPolling = false;
                if (pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                }
            },
            
            isRunning() {
                return isPolling;
            }
        };
    }
}

// Create singleton instance
const apiService = new ApiService();

export { apiService };