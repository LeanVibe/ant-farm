"""
Task Decomposition Agent - Breaks complex tasks into manageable subagent workflows.

This agent analyzes complex tasks and decomposes them into optimized sequences of
subagent operations, considering dependencies, resource requirements, and success
criteria for each subtask.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

from ..constants import Intervals
from ..message_broker import MessageType, message_broker
from ..task_queue import Task

logger = structlog.get_logger()


class TaskComplexity(Enum):
    """Task complexity levels for decomposition strategy."""
    SIMPLE = "simple"          # Single step, no dependencies
    MODERATE = "moderate"      # 2-5 steps, simple dependencies
    COMPLEX = "complex"        # 5-15 steps, complex dependencies
    ENTERPRISE = "enterprise"  # 15+ steps, requires coordination


class SubTaskType(Enum):
    """Types of subtasks that can be created."""
    ANALYSIS = "analysis"           # Code analysis, requirements gathering
    DESIGN = "design"              # Architecture and design decisions
    IMPLEMENTATION = "implementation"  # Code writing and modification
    TESTING = "testing"            # Unit and integration testing
    VALIDATION = "validation"      # Quality gates and validation
    DEPLOYMENT = "deployment"      # Build, deploy, and release
    DOCUMENTATION = "documentation"  # Documentation and knowledge capture
    OPTIMIZATION = "optimization"  # Performance and code optimization


@dataclass
class SubTask:
    """A decomposed subtask with all necessary information."""
    id: str
    type: SubTaskType
    title: str
    description: str
    agent_requirements: List[str]  # Required agent capabilities
    dependencies: List[str]  # IDs of prerequisite subtasks
    estimated_effort: int  # Estimated time in minutes
    success_criteria: List[str]
    context_requirements: Dict[str, Any]
    priority: int = 5
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class DecompositionResult:
    """Result of task decomposition analysis."""
    original_task_id: str
    complexity: TaskComplexity
    subtasks: List[SubTask]
    execution_strategy: str
    estimated_total_effort: int
    risk_factors: List[str]
    optimization_opportunities: List[str]
    success_probability: float


class TaskDecompositionAgent:
    """Agent responsible for breaking down complex tasks into subagent workflows."""

    def __init__(self):
        self.decomposition_history: Dict[str, DecompositionResult] = {}
        self.pattern_library = self._load_decomposition_patterns()
        self.complexity_thresholds = {
            TaskComplexity.SIMPLE: 30,      # < 30 minutes
            TaskComplexity.MODERATE: 120,   # < 2 hours
            TaskComplexity.COMPLEX: 480,    # < 8 hours
            TaskComplexity.ENTERPRISE: 1440 # > 24 hours
        }

    def _load_decomposition_patterns(self) -> Dict[str, Any]:
        """Load common task decomposition patterns."""
        return {
            "feature_development": {
                "subtasks": [
                    SubTaskType.ANALYSIS,
                    SubTaskType.DESIGN,
                    SubTaskType.IMPLEMENTATION,
                    SubTaskType.TESTING,
                    SubTaskType.VALIDATION,
                    SubTaskType.DOCUMENTATION
                ],
                "dependencies": {
                    SubTaskType.IMPLEMENTATION: [SubTaskType.ANALYSIS, SubTaskType.DESIGN],
                    SubTaskType.TESTING: [SubTaskType.IMPLEMENTATION],
                    SubTaskType.VALIDATION: [SubTaskType.TESTING],
                    SubTaskType.DOCUMENTATION: [SubTaskType.VALIDATION]
                }
            },
            "bug_fix": {
                "subtasks": [
                    SubTaskType.ANALYSIS,
                    SubTaskType.IMPLEMENTATION,
                    SubTaskType.TESTING,
                    SubTaskType.VALIDATION
                ],
                "dependencies": {
                    SubTaskType.IMPLEMENTATION: [SubTaskType.ANALYSIS],
                    SubTaskType.TESTING: [SubTaskType.IMPLEMENTATION],
                    SubTaskType.VALIDATION: [SubTaskType.TESTING]
                }
            },
            "refactoring": {
                "subtasks": [
                    SubTaskType.ANALYSIS,
                    SubTaskType.IMPLEMENTATION,
                    SubTaskType.TESTING,
                    SubTaskType.OPTIMIZATION,
                    SubTaskType.VALIDATION
                ],
                "dependencies": {
                    SubTaskType.IMPLEMENTATION: [SubTaskType.ANALYSIS],
                    SubTaskType.TESTING: [SubTaskType.IMPLEMENTATION],
                    SubTaskType.OPTIMIZATION: [SubTaskType.TESTING],
                    SubTaskType.VALIDATION: [SubTaskType.OPTIMIZATION]
                }
            }
        }

    async def decompose_task(self, task: Task) -> DecompositionResult:
        """Decompose a complex task into an optimized subagent workflow."""
        logger.info("Starting task decomposition", task_id=task.id, title=task.title)

        # Analyze task complexity
        complexity = await self._analyze_complexity(task)

        # Select appropriate decomposition pattern
        pattern = await self._select_decomposition_pattern(task, complexity)

        # Generate subtasks
        subtasks = await self._generate_subtasks(task, pattern, complexity)

        # Optimize execution strategy
        execution_strategy = await self._optimize_execution_strategy(subtasks, complexity)

        # Calculate success probability
        success_probability = await self._calculate_success_probability(subtasks, complexity)

        # Identify risk factors
        risk_factors = await self._identify_risk_factors(task, subtasks, complexity)

        # Find optimization opportunities
        optimization_opportunities = await self._find_optimization_opportunities(subtasks)

        result = DecompositionResult(
            original_task_id=task.id,
            complexity=complexity,
            subtasks=subtasks,
            execution_strategy=execution_strategy,
            estimated_total_effort=sum(sub.estimated_effort for sub in subtasks),
            risk_factors=risk_factors,
            optimization_opportunities=optimization_opportunities,
            success_probability=success_probability
        )

        # Store for learning and analysis
        self.decomposition_history[task.id] = result

        logger.info(
            "Task decomposition completed",
            task_id=task.id,
            complexity=complexity.value,
            subtask_count=len(subtasks),
            estimated_effort=result.estimated_total_effort,
            success_probability=success_probability
        )

        return result

    async def _analyze_complexity(self, task: Task) -> TaskComplexity:
        """Analyze task complexity based on multiple factors."""
        # Factor 1: Task size and scope
        size_score = await self._analyze_task_size(task)

        # Factor 2: Technical complexity
        technical_score = await self._analyze_technical_complexity(task)

        # Factor 3: Dependencies and integration points
        dependency_score = await self._analyze_dependencies(task)

        # Factor 4: Domain knowledge requirements
        domain_score = await self._analyze_domain_requirements(task)

        # Combine scores
        total_score = (size_score + technical_score + dependency_score + domain_score) / 4

        # Map to complexity level
        if total_score < 2.5:
            return TaskComplexity.SIMPLE
        elif total_score < 5.0:
            return TaskComplexity.MODERATE
        elif total_score < 7.5:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.ENTERPRISE

    async def _analyze_task_size(self, task: Task) -> float:
        """Analyze task size based on description length and keywords."""
        description_length = len(task.description or "")
        title_length = len(task.title or "")

        # Size indicators
        size_score = 0
        if description_length > 1000:
            size_score += 3
        elif description_length > 500:
            size_score += 2
        elif description_length > 100:
            size_score += 1

        # Keywords indicating large scope
        large_scope_keywords = ["multiple", "several", "many", "complex", "enterprise", "system"]
        for keyword in large_scope_keywords:
            if keyword in (task.title + " " + task.description).lower():
                size_score += 1

        return min(size_score, 5.0)

    async def _analyze_technical_complexity(self, task: Task) -> float:
        """Analyze technical complexity based on keywords and patterns."""
        text = (task.title + " " + task.description).lower()

        complexity_indicators = {
            "microservices": 3,
            "distributed": 3,
            "real-time": 2,
            "concurrent": 2,
            "security": 3,
            "performance": 2,
            "scalability": 2,
            "integration": 2,
            "migration": 3,
            "refactor": 2,
            "optimization": 2
        }

        complexity_score = 0
        for indicator, score in complexity_indicators.items():
            if indicator in text:
                complexity_score += score

        return min(complexity_score, 5.0)

    async def _analyze_dependencies(self, task: Task) -> float:
        """Analyze dependency complexity."""
        text = (task.title + " " + task.description).lower()

        dependency_indicators = {
            "depends on": 2,
            "requires": 1,
            "integrate with": 2,
            "coordinate": 2,
            "multiple teams": 3,
            "external systems": 2,
            "third-party": 2,
            "legacy": 2
        }

        dependency_score = 0
        for indicator, score in dependency_indicators.items():
            if indicator in text:
                dependency_score += score

        return min(dependency_score, 5.0)

    async def _analyze_domain_requirements(self, task: Task) -> float:
        """Analyze domain knowledge requirements."""
        text = (task.title + " " + task.description).lower()

        domain_indicators = {
            "business logic": 2,
            "domain rules": 2,
            "compliance": 3,
            "regulatory": 3,
            "industry-specific": 2,
            "specialized knowledge": 2
        }

        domain_score = 0
        for indicator, score in domain_indicators.items():
            if indicator in text:
                domain_score += score

        return min(domain_score, 5.0)

    async def _select_decomposition_pattern(self, task: Task, complexity: TaskComplexity) -> Dict[str, Any]:
        """Select the most appropriate decomposition pattern."""
        text = (task.title + " " + task.description).lower()

        # Pattern matching based on task content
        if "bug" in text or "fix" in text or "issue" in text:
            return self.pattern_library["bug_fix"]
        elif "refactor" in text or "optimize" in text or "improve" in text:
            return self.pattern_library["refactoring"]
        else:
            # Default to feature development pattern
            return self.pattern_library["feature_development"]

    async def _generate_subtasks(self, task: Task, pattern: Dict[str, Any], complexity: TaskComplexity) -> List[SubTask]:
        """Generate specific subtasks based on the selected pattern."""
        subtasks = []

        for subtask_type in pattern["subtasks"]:
            subtask = await self._create_subtask(task, subtask_type, complexity)
            subtasks.append(subtask)

        # Add dependencies based on pattern
        for subtask in subtasks:
            dependencies = pattern.get("dependencies", {}).get(subtask.type, [])
            subtask.dependencies = [s.id for s in subtasks if s.type in dependencies]

        return subtasks

    async def _create_subtask(self, task: Task, subtask_type: SubTaskType, complexity: TaskComplexity) -> SubTask:
        """Create a specific subtask with appropriate details."""
        base_effort = {
            SubTaskType.ANALYSIS: 30,
            SubTaskType.DESIGN: 45,
            SubTaskType.IMPLEMENTATION: 120,
            SubTaskType.TESTING: 60,
            SubTaskType.VALIDATION: 30,
            SubTaskType.DEPLOYMENT: 45,
            SubTaskType.DOCUMENTATION: 30,
            SubTaskType.OPTIMIZATION: 60
        }

        # Adjust effort based on complexity
        complexity_multiplier = {
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 1.5,
            TaskComplexity.ENTERPRISE: 2.0
        }

        estimated_effort = int(base_effort[subtask_type] * complexity_multiplier[complexity])

        # Generate subtask details
        title, description = await self._generate_subtask_details(task, subtask_type)
        agent_requirements = await self._determine_agent_requirements(subtask_type, complexity)
        success_criteria = await self._define_success_criteria(subtask_type, task)
        context_requirements = await self._define_context_requirements(subtask_type, task)

        return SubTask(
            id=f"{task.id}_{subtask_type.value}_{uuid.uuid4().hex[:8]}",
            type=subtask_type,
            title=title,
            description=description,
            agent_requirements=agent_requirements,
            dependencies=[],  # Will be set later
            estimated_effort=estimated_effort,
            success_criteria=success_criteria,
            context_requirements=context_requirements
        )

    async def _generate_subtask_details(self, task: Task, subtask_type: SubTaskType) -> tuple[str, str]:
        """Generate title and description for a subtask."""
        templates = {
            SubTaskType.ANALYSIS: (
                f"Analyze {task.title}",
                f"Analyze the requirements, constraints, and scope for {task.title}. "
                f"Identify technical challenges, dependencies, and success criteria."
            ),
            SubTaskType.DESIGN: (
                f"Design solution for {task.title}",
                f"Design the technical solution including architecture, components, "
                f"and integration points for {task.title}."
            ),
            SubTaskType.IMPLEMENTATION: (
                f"Implement {task.title}",
                f"Implement the designed solution for {task.title} following "
                f"best practices and coding standards."
            ),
            SubTaskType.TESTING: (
                f"Test implementation of {task.title}",
                f"Create and execute comprehensive tests for {task.title} "
                f"including unit tests, integration tests, and validation."
            ),
            SubTaskType.VALIDATION: (
                f"Validate {task.title}",
                f"Validate that {task.title} meets all requirements and "
                f"quality standards through comprehensive testing and review."
            ),
            SubTaskType.DOCUMENTATION: (
                f"Document {task.title}",
                f"Create comprehensive documentation for {task.title} "
                f"including usage instructions, API documentation, and examples."
            ),
            SubTaskType.DEPLOYMENT: (
                f"Deploy {task.title}",
                f"Deploy {task.title} to production environment with proper "
                f"monitoring and rollback capabilities."
            ),
            SubTaskType.OPTIMIZATION: (
                f"Optimize {task.title}",
                f"Optimize performance, code quality, and maintainability of {task.title}."
            )
        }

        return templates.get(subtask_type, (f"Handle {subtask_type.value} for {task.title}", f"Handle {subtask_type.value} aspects of {task.title}"))

    async def _determine_agent_requirements(self, subtask_type: SubTaskType, complexity: TaskComplexity) -> List[str]:
        """Determine what agent capabilities are required for this subtask."""
        base_requirements = {
            SubTaskType.ANALYSIS: ["analysis", "problem_solving"],
            SubTaskType.DESIGN: ["architecture", "design", "system_thinking"],
            SubTaskType.IMPLEMENTATION: ["coding", "development", "programming"],
            SubTaskType.TESTING: ["testing", "quality_assurance", "debugging"],
            SubTaskType.VALIDATION: ["validation", "verification", "quality_assurance"],
            SubTaskType.DOCUMENTATION: ["documentation", "technical_writing"],
            SubTaskType.DEPLOYMENT: ["deployment", "devops", "infrastructure"],
            SubTaskType.OPTIMIZATION: ["optimization", "performance", "code_quality"]
        }

        requirements = base_requirements.get(subtask_type, ["general"])

        # Add complexity-based requirements
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE]:
            requirements.extend(["advanced_problem_solving", "system_design"])

        return list(set(requirements))  # Remove duplicates

    async def _define_success_criteria(self, subtask_type: SubTaskType, task: Task) -> List[str]:
        """Define success criteria for a subtask."""
        criteria_map = {
            SubTaskType.ANALYSIS: [
                "Requirements clearly documented",
                "Technical challenges identified",
                "Scope and constraints defined",
                "Success metrics established"
            ],
            SubTaskType.DESIGN: [
                "Architecture design completed",
                "Component interfaces defined",
                "Integration points identified",
                "Design review passed"
            ],
            SubTaskType.IMPLEMENTATION: [
                "Code implemented according to design",
                "Coding standards followed",
                "No critical bugs introduced",
                "Code review passed"
            ],
            SubTaskType.TESTING: [
                "All tests passing",
                "Test coverage meets requirements",
                "Edge cases covered",
                "Performance requirements met"
            ],
            SubTaskType.VALIDATION: [
                "All acceptance criteria met",
                "Quality gates passed",
                "Stakeholder approval obtained",
                "Ready for deployment"
            ]
        }

        return criteria_map.get(subtask_type, ["Subtask completed successfully"])

    async def _define_context_requirements(self, subtask_type: SubTaskType, task: Task) -> Dict[str, Any]:
        """Define what context is needed for this subtask."""
        context_map = {
            SubTaskType.ANALYSIS: {
                "project_context": True,
                "technical_requirements": True,
                "stakeholder_input": True
            },
            SubTaskType.DESIGN: {
                "analysis_results": True,
                "technical_constraints": True,
                "architecture_patterns": True
            },
            SubTaskType.IMPLEMENTATION: {
                "design_documents": True,
                "coding_standards": True,
                "development_environment": True
            },
            SubTaskType.TESTING: {
                "implementation_code": True,
                "test_requirements": True,
                "testing_framework": True
            }
        }

        return context_map.get(subtask_type, {"general_context": True})

    async def _optimize_execution_strategy(self, subtasks: List[SubTask], complexity: TaskComplexity) -> str:
        """Optimize the execution strategy for the subtasks."""
        if complexity == TaskComplexity.SIMPLE:
            return "sequential"
        elif complexity == TaskComplexity.MODERATE:
            return "parallel_independent"
        elif complexity == TaskComplexity.COMPLEX:
            return "parallel_with_dependencies"
        else:  # Enterprise
            return "coordinated_parallel"

    async def _calculate_success_probability(self, subtasks: List[SubTask], complexity: TaskComplexity) -> float:
        """Calculate the probability of successful completion."""
        base_probability = 0.9  # Base success rate

        # Adjust for complexity
        complexity_adjustment = {
            TaskComplexity.SIMPLE: 0.05,
            TaskComplexity.MODERATE: 0.0,
            TaskComplexity.COMPLEX: -0.1,
            TaskComplexity.ENTERPRISE: -0.2
        }

        # Adjust for number of subtasks
        subtask_adjustment = max(-0.1, (len(subtasks) - 5) * -0.02)

        # Adjust for dependencies
        dependency_count = sum(len(sub.dependencies) for sub in subtasks)
        dependency_adjustment = max(-0.1, dependency_count * -0.01)

        probability = base_probability + complexity_adjustment[complexity] + subtask_adjustment + dependency_adjustment

        return max(0.1, min(0.95, probability))  # Clamp between 10% and 95%

    async def _identify_risk_factors(self, task: Task, subtasks: List[SubTask], complexity: TaskComplexity) -> List[str]:
        """Identify potential risk factors for the task."""
        risks = []

        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE]:
            risks.append("High complexity may lead to integration issues")

        if len(subtasks) > 10:
            risks.append("Large number of subtasks increases coordination overhead")

        dependency_count = sum(len(sub.dependencies) for sub in subtasks)
        if dependency_count > 15:
            risks.append("Complex dependency chain may cause cascading failures")

        # Task-specific risks
        text = (task.title + " " + task.description).lower()
        if "new technology" in text or "unfamiliar" in text:
            risks.append("Learning curve for new technologies")
        if "tight deadline" in text or "urgent" in text:
            risks.append("Time pressure may affect quality")
        if "legacy" in text or "old" in text:
            risks.append("Legacy system integration challenges")

        return risks

    async def _find_optimization_opportunities(self, subtasks: List[SubTask]) -> List[str]:
        """Find opportunities to optimize the task execution."""
        opportunities = []

        # Check for parallel execution opportunities
        independent_subtasks = [sub for sub in subtasks if not sub.dependencies]
        if len(independent_subtasks) > 1:
            opportunities.append(f"Execute {len(independent_subtasks)} independent subtasks in parallel")

        # Check for batching opportunities
        testing_subtasks = [sub for sub in subtasks if sub.type == SubTaskType.TESTING]
        if len(testing_subtasks) > 1:
            opportunities.append("Batch testing subtasks to reduce setup overhead")

        # Check for automation opportunities
        automation_candidates = [sub for sub in subtasks if sub.type in [SubTaskType.VALIDATION, SubTaskType.DEPLOYMENT]]
        if automation_candidates:
            opportunities.append("Automate validation and deployment subtasks")

        return opportunities

    async def get_decomposition_history(self, task_id: str) -> Optional[DecompositionResult]:
        """Get the decomposition history for a specific task."""
        return self.decomposition_history.get(task_id)

    async def get_decomposition_statistics(self) -> Dict[str, Any]:
        """Get statistics about task decomposition performance."""
        if not self.decomposition_history:
            return {"total_decompositions": 0}

        complexities = [result.complexity for result in self.decomposition_history.values()]
        success_probabilities = [result.success_probability for result in self.decomposition_history.values()]

        return {
            "total_decompositions": len(self.decomposition_history),
            "complexity_distribution": {
                complexity.value: complexities.count(complexity)
                for complexity in TaskComplexity
            },
            "average_success_probability": sum(success_probabilities) / len(success_probabilities),
            "average_subtasks_per_task": sum(len(result.subtasks) for result in self.decomposition_history.values()) / len(self.decomposition_history)
        }