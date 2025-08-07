"""Agent Factory for creating and managing specialized agents."""

from typing import Any, Dict, Optional

import structlog

from .base_agent import BaseAgent
from .qa_agent import QAAgent
from .architect_agent import ArchitectAgent
from .devops_agent import DevOpsAgent
from .meta_agent import MetaAgent

logger = structlog.get_logger()


class AgentFactory:
    """Factory for creating different types of agents."""

    # Registry of available agent types
    AGENT_TYPES = {
        "qa": QAAgent,
        "architect": ArchitectAgent,
        "devops": DevOpsAgent,
        "meta": MetaAgent,
    }

    # Default capabilities for each agent type
    AGENT_CAPABILITIES = {
        "qa": [
            "test_code",
            "review_code",
            "check_quality",
            "run_tests",
            "security_scan",
            "performance_test",
            "validate_changes",
        ],
        "architect": [
            "design_system",
            "review_architecture",
            "refactor_code",
            "plan_migration",
            "optimize_performance",
            "design_api",
            "create_documentation",
            "analyze_dependencies",
            "design_database",
            "plan_scaling",
        ],
        "devops": [
            "deploy_application",
            "setup_infrastructure",
            "configure_monitoring",
            "setup_ci_cd",
            "backup_data",
            "scale_services",
            "update_dependencies",
            "security_hardening",
            "disaster_recovery",
            "performance_tuning",
            "log_analysis",
            "incident_response",
        ],
        "meta": [
            "analyze_system",
            "optimize_prompts",
            "improve_architecture",
            "coordinate_agents",
            "self_modify",
            "performance_analysis",
        ],
    }

    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create (qa, architect, devops, meta)
            name: Optional custom name for the agent
            config: Optional configuration parameters

        Returns:
            BaseAgent: Instance of the requested agent type

        Raises:
            ValueError: If agent_type is not supported
        """
        if agent_type not in cls.AGENT_TYPES:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {list(cls.AGENT_TYPES.keys())}"
            )

        agent_class = cls.AGENT_TYPES[agent_type]

        # Use custom name or generate default
        if name is None:
            name = cls._generate_agent_name(agent_type)

        try:
            # Create agent instance
            if config:
                # If agent class supports config, pass it
                agent = agent_class(name=name, **config)
            else:
                agent = agent_class(name=name)

            logger.info(
                "Agent created successfully",
                agent_type=agent_type,
                agent_name=name,
                capabilities=cls.AGENT_CAPABILITIES.get(agent_type, []),
            )

            return agent

        except Exception as e:
            logger.error(
                "Failed to create agent",
                agent_type=agent_type,
                agent_name=name,
                error=str(e),
            )
            raise

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available agent types."""
        return list(cls.AGENT_TYPES.keys())

    @classmethod
    def get_agent_capabilities(cls, agent_type: str) -> list[str]:
        """Get capabilities for a specific agent type."""
        return cls.AGENT_CAPABILITIES.get(agent_type, [])

    @classmethod
    def get_agent_info(cls, agent_type: str) -> Dict[str, Any]:
        """Get detailed information about an agent type."""
        if agent_type not in cls.AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = cls.AGENT_TYPES[agent_type]

        return {
            "type": agent_type,
            "class": agent_class.__name__,
            "capabilities": cls.AGENT_CAPABILITIES.get(agent_type, []),
            "description": agent_class.__doc__ or "No description available",
        }

    @classmethod
    def create_agent_team(
        cls, team_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, BaseAgent]:
        """
        Create a team of agents based on configuration.

        Args:
            team_config: Dictionary mapping agent names to their configurations
                        {
                            "qa-primary": {"type": "qa", "config": {...}},
                            "architect-lead": {"type": "architect", "config": {...}}
                        }

        Returns:
            Dict[str, BaseAgent]: Dictionary mapping agent names to instances
        """
        team = {}

        for agent_name, agent_spec in team_config.items():
            agent_type = agent_spec.get("type")
            agent_config = agent_spec.get("config", {})

            if not agent_type:
                logger.warning(
                    "No agent type specified for team member", agent_name=agent_name
                )
                continue

            try:
                agent = cls.create_agent(
                    agent_type=agent_type, name=agent_name, config=agent_config
                )
                team[agent_name] = agent

            except Exception as e:
                logger.error(
                    "Failed to create team member",
                    agent_name=agent_name,
                    agent_type=agent_type,
                    error=str(e),
                )

        logger.info(
            "Agent team created",
            team_size=len(team),
            agent_types=[agent.agent_type for agent in team.values()],
        )

        return team

    @classmethod
    def create_balanced_team(cls, team_size: int = 4) -> Dict[str, BaseAgent]:
        """
        Create a balanced team with different agent types.

        Args:
            team_size: Desired team size (will create up to available types)

        Returns:
            Dict[str, BaseAgent]: Balanced team of agents
        """
        available_types = cls.get_available_types()
        team_config = {}

        # Create one agent of each type up to team_size
        for i, agent_type in enumerate(available_types[:team_size]):
            agent_name = f"{agent_type}-{i + 1}"
            team_config[agent_name] = {"type": agent_type}

        return cls.create_agent_team(team_config)

    @classmethod
    def _generate_agent_name(cls, agent_type: str) -> str:
        """Generate a default name for an agent."""
        import uuid

        short_id = str(uuid.uuid4())[:8]
        return f"{agent_type}-{short_id}"

    @classmethod
    def validate_agent_config(
        cls, agent_type: str, config: Dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate agent configuration.

        Args:
            agent_type: Type of agent
            config: Configuration to validate

        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_errors)
        """
        errors = []

        if agent_type not in cls.AGENT_TYPES:
            errors.append(f"Unknown agent type: {agent_type}")
            return False, errors

        # Basic validation - can be extended per agent type
        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return False, errors

        # Agent-specific validation could be added here
        # For example, checking required fields, valid values, etc.

        return len(errors) == 0, errors


# Convenience functions for common agent creation patterns
def create_qa_agent(name: str = None) -> QAAgent:
    """Create a QA agent."""
    return AgentFactory.create_agent("qa", name)


def create_architect_agent(name: str = None) -> ArchitectAgent:
    """Create an Architect agent."""
    return AgentFactory.create_agent("architect", name)


def create_devops_agent(name: str = None) -> DevOpsAgent:
    """Create a DevOps agent."""
    return AgentFactory.create_agent("devops", name)


def create_meta_agent(name: str = None) -> MetaAgent:
    """Create a Meta agent."""
    return AgentFactory.create_agent("meta", name)


def create_development_team() -> Dict[str, BaseAgent]:
    """Create a typical development team with all agent types."""
    return AgentFactory.create_agent_team(
        {
            "qa-lead": {"type": "qa"},
            "architect-senior": {"type": "architect"},
            "devops-engineer": {"type": "devops"},
            "meta-coordinator": {"type": "meta"},
        }
    )


# Agent coordination utilities
class AgentCoordinator:
    """Coordinates communication and task distribution between agents."""

    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.task_assignments = {}

    async def assign_task_to_best_agent(self, task: Any) -> Optional[BaseAgent]:
        """
        Assign a task to the most suitable agent based on capabilities.

        Args:
            task: Task object with task_type attribute

        Returns:
            BaseAgent: Best agent for the task, or None if no suitable agent
        """
        task_type = getattr(task, "task_type", None)
        if not task_type:
            return None

        # Find agents capable of handling this task type
        capable_agents = []

        for agent_name, agent in self.agents.items():
            agent_capabilities = AgentFactory.get_agent_capabilities(agent.agent_type)
            if task_type in agent_capabilities:
                capable_agents.append((agent_name, agent))

        if not capable_agents:
            logger.warning(
                "No agent found capable of handling task",
                task_type=task_type,
                available_agents=list(self.agents.keys()),
            )
            return None

        # Simple load balancing - choose agent with fewest current tasks
        best_agent = min(
            capable_agents, key=lambda x: len(self.task_assignments.get(x[0], []))
        )[1]

        # Track assignment
        if best_agent.name not in self.task_assignments:
            self.task_assignments[best_agent.name] = []
        self.task_assignments[best_agent.name].append(task)

        logger.info(
            "Task assigned to agent",
            task_type=task_type,
            agent_name=best_agent.name,
            agent_type=best_agent.agent_type,
        )

        return best_agent

    async def broadcast_message(
        self, topic: str, content: Dict[str, Any], exclude_agents: list[str] = None
    ) -> None:
        """
        Broadcast a message to all agents in the team.

        Args:
            topic: Message topic
            content: Message content
            exclude_agents: List of agent names to exclude from broadcast
        """
        exclude_agents = exclude_agents or []

        for agent_name, agent in self.agents.items():
            if agent_name not in exclude_agents:
                try:
                    # Use agent's message sending capability
                    await agent.send_message(
                        to_agent="broadcast", topic=topic, content=content
                    )
                except Exception as e:
                    logger.error(
                        "Failed to send broadcast message",
                        agent_name=agent_name,
                        topic=topic,
                        error=str(e),
                    )

    def get_team_status(self) -> Dict[str, Any]:
        """Get status of all agents in the team."""
        status = {"team_size": len(self.agents), "agents": {}, "task_distribution": {}}

        for agent_name, agent in self.agents.items():
            status["agents"][agent_name] = {
                "type": agent.agent_type,
                "status": agent.status,
                "tasks_completed": getattr(agent, "tasks_completed", 0),
                "tasks_failed": getattr(agent, "tasks_failed", 0),
            }

            # Task assignment info
            assigned_tasks = len(self.task_assignments.get(agent_name, []))
            status["task_distribution"][agent_name] = assigned_tasks

        return status
