import { LitElement, html, css } from 'lit';

export class TaskBoard extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .kanban-board {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            min-height: 400px;
        }

        .column {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .column-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .column-title {
            font-weight: 600;
            color: #ffffff;
        }

        .task-count {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
        }

        .task-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            min-height: 300px;
        }

        .task-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .task-card:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .task-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #ffffff;
            font-size: 0.9rem;
        }

        .task-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .task-type {
            background: rgba(100, 149, 237, 0.2);
            color: #6495ed;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.7rem;
        }

        .task-priority {
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: bold;
        }

        .priority-critical { background: #ff4444; color: white; }
        .priority-high { background: #ff8800; color: white; }
        .priority-normal { background: #00d4aa; color: white; }
        .priority-low { background: #666; color: white; }

        .task-agent {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
        }

        .task-created {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.5);
        }

        .compact .kanban-board {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }

        .compact .column {
            padding: 0.75rem;
        }

        .compact .task-list {
            min-height: 200px;
        }

        .empty-state {
            text-align: center;
            padding: 2rem;
            color: rgba(255, 255, 255, 0.5);
            font-style: italic;
        }
    `;

    static properties = {
        tasks: { type: Array },
        compact: { type: Boolean }
    };

    constructor() {
        super();
        this.tasks = [];
        this.compact = false;
    }

    connectedCallback() {
        super.connectedCallback();
        this.loadTasks();
        this.initializeWebSocketListeners();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeWebSocketListeners();
    }

    initializeWebSocketListeners() {
        // Listen for real-time task updates via WebSocket
        this.handleTaskUpdate = (event) => {
            const taskData = event.detail;
            
            if (taskData.action === 'created') {
                // Add new task to list
                this.tasks = [...this.tasks, taskData.task];
            } else if (taskData.action === 'updated') {
                // Update existing task
                this.tasks = this.tasks.map(task =>
                    task.id === taskData.task.id
                        ? { ...task, ...taskData.task }
                        : task
                );
            } else if (taskData.action === 'completed') {
                // Update task status to completed
                this.tasks = this.tasks.map(task =>
                    task.id === taskData.task.id
                        ? { ...task, status: 'completed', completed_at: taskData.task.completed_at }
                        : task
                );
            } else if (taskData.action === 'failed') {
                // Update task status to failed
                this.tasks = this.tasks.map(task =>
                    task.id === taskData.task.id
                        ? { ...task, status: 'failed', error: taskData.task.error }
                        : task
                );
            } else if (taskData.action === 'assigned') {
                // Update task assignment
                this.tasks = this.tasks.map(task =>
                    task.id === taskData.task.id
                        ? { ...task, assigned_to: taskData.task.assigned_to, status: 'in_progress' }
                        : task
                );
            }
            
            this.requestUpdate();
        };

        window.addEventListener('task-status-update', this.handleTaskUpdate);
    }

    removeWebSocketListeners() {
        if (this.handleTaskUpdate) {
            window.removeEventListener('task-status-update', this.handleTaskUpdate);
        }
    }

    async loadTasks() {
        try {
            const response = await fetch('/api/v1/tasks');
            if (response.ok) {
                const data = await response.json();
                this.tasks = data.data || [];
            }
        } catch (error) {
            console.error('Failed to load tasks:', error);
        }
    }

    getTasksByStatus(status) {
        return this.tasks.filter(task => task.status === status);
    }

    getPriorityClass(priority) {
        switch (priority) {
            case 1: return 'priority-critical';
            case 2: 
            case 3: return 'priority-high';
            case 5: return 'priority-normal';
            default: return 'priority-low';
        }
    }

    getPriorityLabel(priority) {
        switch (priority) {
            case 1: return 'Critical';
            case 2: 
            case 3: return 'High';
            case 5: return 'Normal';
            default: return 'Low';
        }
    }

    formatTimeAgo(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        const now = new Date();
        const diffInMinutes = Math.floor((now - date) / (1000 * 60));
        
        if (diffInMinutes < 1) return 'Just now';
        if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
        if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
        return `${Math.floor(diffInMinutes / 1440)}d ago`;
    }

    renderColumn(title, status, tasks) {
        return html`
            <div class="column">
                <div class="column-header">
                    <div class="column-title">${title}</div>
                    <div class="task-count">${tasks.length}</div>
                </div>
                <div class="task-list">
                    ${tasks.length === 0 ? html`
                        <div class="empty-state">No ${status} tasks</div>
                    ` : tasks.map(task => html`
                        <div class="task-card" @click=${() => this.showTaskDetails(task)}>
                            <div class="task-title">${task.title}</div>
                            <div class="task-meta">
                                <div class="task-type">${task.type}</div>
                                <div class="task-priority ${this.getPriorityClass(task.priority)}">
                                    ${this.getPriorityLabel(task.priority)}
                                </div>
                            </div>
                            ${task.agent_name ? html`
                                <div class="task-agent">👤 ${task.agent_name}</div>
                            ` : ''}
                            <div class="task-created">${this.formatTimeAgo(task.created_at)}</div>
                        </div>
                    `)}
                </div>
            </div>
        `;
    }

    showTaskDetails(task) {
        // Dispatch event for parent components to handle
        this.dispatchEvent(new CustomEvent('task-selected', {
            detail: task,
            bubbles: true,
            composed: true
        }));
    }

    render() {
        const pendingTasks = this.getTasksByStatus('pending');
        const inProgressTasks = this.getTasksByStatus('in_progress');
        const completedTasks = this.getTasksByStatus('completed');
        const failedTasks = this.getTasksByStatus('failed');

        return html`
            <div class="${this.compact ? 'compact' : ''}">
                <div class="kanban-board">
                    ${this.renderColumn('📋 Pending', 'pending', pendingTasks)}
                    ${this.renderColumn('⚡ In Progress', 'in_progress', inProgressTasks)}
                    ${this.renderColumn('✅ Completed', 'completed', completedTasks)}
                    ${this.renderColumn('❌ Failed', 'failed', failedTasks)}
                </div>
            </div>
        `;
    }
}

customElements.define('task-board', TaskBoard);