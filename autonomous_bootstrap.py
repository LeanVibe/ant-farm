"""Enhanced bootstrap integration for autonomous operation."""

import asyncio
import time
from bootstrap import BootstrapAgent
from src.core.self_bootstrap import self_bootstrapper
from src.core.task_coordinator import task_coordinator
from src.core.orchestrator import orchestrator

class AutonomousBootstrap(BootstrapAgent):
    """Enhanced bootstrap agent for autonomous operation."""
    
    def __init__(self):
        super().__init__()
        self.autonomous_mode = False
    
    async def bootstrap_autonomous_system(self):
        """Bootstrap the system for fully autonomous operation."""
        self.console.print("\n[bold cyan]Starting Autonomous LeanVibe Agent Hive 2.0[/bold cyan]\n")
        
        # Run basic bootstrap first
        super().bootstrap_system()
        
        # Wait for basic system to be ready
        await asyncio.sleep(10)
        
        # Initialize autonomous components
        await self._initialize_autonomous_components()
        
        # Start autonomous development cycle
        await self._start_autonomous_cycle()
        
        self.console.print("\n[bold green]Autonomous System Operational![/bold green]")
        self.console.print("\nThe system will now continue development autonomously.")
        self.console.print("Monitor progress via:")
        self.console.print("• API: http://localhost:8000/api/v1/status")
        self.console.print("• Dashboard: http://localhost:8000/api/docs")
        self.console.print("• Logs: tail -f logs/*.log")
    
    async def _initialize_autonomous_components(self):
        """Initialize components for autonomous operation."""
        self.console.print("[cyan]Initializing autonomous components...[/cyan]")
        
        # Initialize task coordinator
        await task_coordinator.initialize()
        
        # Initialize self-bootstrapper
        await self_bootstrapper.initialize_autonomous_development()
        
        # Start API server in background
        await self._start_api_server()
        
        self.console.print("[green]✓[/green] Autonomous components initialized")
    
    async def _start_api_server(self):
        """Start the API server."""
        # Create API server startup task
        api_task = self.execute_claude_task(
            "Start the API server: cd src/api && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload",
            session_name="api-server"
        )
        
        self.console.print("[green]✓[/green] API server starting on port 8000")
    
    async def _start_autonomous_cycle(self):
        """Start the autonomous development cycle."""
        self.console.print("[cyan]Starting autonomous development cycle...[/cyan]")
        
        # Create initial autonomous tasks
        initial_tasks = [
            {
                "title": "System Health Check",
                "description": "Perform initial system health assessment",
                "type": "system_analysis",
                "assigned_to": "meta-agent",
                "priority": "high"
            },
            {
                "title": "Initialize Continuous Monitoring",
                "description": "Set up continuous system monitoring",
                "type": "monitoring",
                "assigned_to": "meta-agent",
                "priority": "medium"
            },
            {
                "title": "Begin Autonomous Development",
                "description": "Start the first autonomous development cycle",
                "type": "development",
                "assigned_to": "developer-agent",
                "priority": "medium"
            }
        ]
        
        # Submit initial tasks
        for task_data in initial_tasks:
            task_prompt = f"""
            Create and submit a task:
            Title: {task_data['title']}
            Description: {task_data['description']}
            Type: {task_data['type']}
            Priority: {task_data['priority']}
            Assigned to: {task_data['assigned_to']}
            
            Use the task queue system to submit this task.
            """
            
            self.execute_claude_task(task_prompt)
        
        # Start autonomous development cycle
        asyncio.create_task(self._autonomous_development_loop())
        
        self.console.print("[green]✓[/green] Autonomous development cycle started")
    
    async def _autonomous_development_loop(self):
        """Main autonomous development loop."""
        while True:
            try:
                # Execute one development cycle
                await self_bootstrapper.execute_autonomous_development_cycle()
                
                # Wait before next cycle (1 hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.console.print(f"[red]Autonomous development error:[/red] {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def autonomous():
        """Start the autonomous agent hive system."""
        agent = AutonomousBootstrap()
        try:
            asyncio.run(agent.bootstrap_autonomous_system())
        finally:
            agent.cleanup()
    
    @app.command() 
    def status():
        """Check autonomous system status."""
        import requests
        
        try:
            response = requests.get("http://localhost:8000/api/v1/status")
            if response.status_code == 200:
                data = response.json()
                print(f"System Health: {data['data']['health_score']:.2f}")
                print(f"Active Agents: {data['data']['active_agents']}")
                print(f"Tasks Completed: {data['data']['completed_tasks']}")
                print(f"Queue Depth: {data['data']['queue_depth']}")
            else:
                print("System not responding")
        except:
            print("System not running")
    
    app()