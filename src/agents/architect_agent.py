"""Architecture Agent for LeanVibe Agent Hive 2.0 with coordination support."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

# Handle both module and direct execution imports
try:
    from ..core.task_queue import Task
    from .base_agent import BaseAgent, HealthStatus, TaskResult
except ImportError:
    # Direct execution - add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from agents.base_agent import BaseAgent, HealthStatus, TaskResult
    from core.task_queue import Task

logger = structlog.get_logger()


class ArchitectAgent(BaseAgent):
    """Architecture Agent specializing in system design and architectural decisions."""

    def __init__(self, name: str = "architect-agent"):
        super().__init__(
            name=name,
            agent_type="architect",
            role="System Architect - Design, architecture, and system planning",
        )

        # Architecture-specific capabilities
        self.design_patterns = [
            "microservices",
            "event-driven",
            "layered",
            "mvc",
            "repository",
            "factory",
            "observer",
            "singleton",
            "strategy",
            "dependency-injection",
        ]
        self.architecture_types = [
            "monolith",
            "microservices",
            "serverless",
            "event-driven",
            "pipeline",
        ]
        self.technology_stacks = [
            "python",
            "fastapi",
            "postgresql",
            "redis",
            "docker",
            "kubernetes",
        ]

        # Performance tracking
        self.designs_created = 0
        self.architectures_reviewed = 0
        self.decisions_made = 0
        self.refactoring_plans = 0
        self.collaborations_led = 0

        logger.info("Architect Agent initialized", agent=self.name)

    async def run(self) -> None:
        """Main Architect agent execution loop."""
        logger.info("Architect Agent starting main execution loop", agent=self.name)

        while self.status == "active":
            try:
                # Check for architecture tasks
                task = await self._get_next_architecture_task()

                if task:
                    logger.info(
                        "Processing architecture task",
                        agent=self.name,
                        task_id=task.id,
                        task_type=task.task_type,
                    )
                    await self.process_task(task)
                else:
                    # No individual tasks, check for collaboration opportunities
                    await self._check_collaboration_opportunities()

                    # Monitor system architecture
                    await self._monitor_system_architecture()

                    # Wait before next check
                    await asyncio.sleep(10)

            except Exception as e:
                logger.error(
                    "Architect Agent execution error", agent=self.name, error=str(e)
                )
                await asyncio.sleep(15)

    async def _check_collaboration_opportunities(self) -> None:
        """Check for opportunities to initiate architectural collaborations."""

        # Example: Initiate design review collaboration periodically
        current_time = time.time()
        if hasattr(self, "_last_design_review"):
            time_since_review = current_time - self._last_design_review
            if time_since_review > 3600:  # 1 hour
                await self._initiate_design_review_collaboration()
        else:
            self._last_design_review = current_time

    async def _initiate_design_review_collaboration(self) -> None:
        """Initiate a collaborative design review."""

        try:
            collaboration_id = await self.initiate_collaboration(
                title="System Architecture Review",
                description="Comprehensive review of current system architecture and identification of improvement opportunities",
                collaboration_type="sequential",
                required_capabilities=[
                    "system_design",
                    "code_review",
                    "testing",
                    "deployment",
                ],
                deadline=datetime.now() + timedelta(hours=2),
                priority=4,
                metadata={
                    "review_type": "architecture",
                    "scope": "full_system",
                    "focus_areas": ["performance", "scalability", "maintainability"],
                },
            )

            self.collaborations_led += 1
            self._last_design_review = time.time()

            logger.info(
                "Initiated design review collaboration",
                agent=self.name,
                collaboration_id=collaboration_id,
            )

        except Exception as e:
            logger.error(
                "Failed to initiate design review collaboration",
                agent=self.name,
                error=str(e),
            )

    async def _evaluate_collaboration_invitation(
        self, invitation: dict[str, Any]
    ) -> bool:
        """Evaluate whether to accept a collaboration invitation."""

        # Always accept architecture-related collaborations
        if invitation.get("collaboration_type") in ["consensus", "sequential"]:
            title = invitation.get("title", "").lower()
            description = invitation.get("description", "").lower()

            architecture_keywords = [
                "design",
                "architecture",
                "system",
                "refactor",
                "structure",
                "pattern",
                "scalability",
                "performance",
            ]

            if any(
                keyword in title or keyword in description
                for keyword in architecture_keywords
            ):
                logger.info(
                    "Accepting architecture collaboration",
                    agent=self.name,
                    title=invitation.get("title"),
                )
                return True

        # Use base evaluation for other collaborations
        return await super()._evaluate_collaboration_invitation(invitation)

    async def _process_collaborative_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a collaborative architecture task."""

        description = task.get("description", "")
        task_type = self._determine_architecture_task_type(description)

        logger.info(
            "Processing collaborative architecture task",
            agent=self.name,
            task_type=task_type,
            description=description[:100],
        )

        try:
            if task_type == "system_design":
                result = await self._create_system_design(task)
            elif task_type == "architecture_review":
                result = await self._review_architecture(task)
            elif task_type == "refactoring_plan":
                result = await self._create_refactoring_plan(task)
            elif task_type == "technology_decision":
                result = await self._make_technology_decision(task)
            else:
                result = await self._general_architecture_analysis(task)

            return {
                "success": True,
                "task_type": task_type,
                "result": result,
                "agent": self.name,
                "timestamp": time.time(),
                "artifacts": result.get("artifacts", []),
            }

        except Exception as e:
            logger.error(
                "Collaborative task failed",
                agent=self.name,
                task_type=task_type,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
                "agent": self.name,
                "timestamp": time.time(),
            }

    def _determine_architecture_task_type(self, description: str) -> str:
        """Determine the type of architecture task based on description."""

        description_lower = description.lower()

        if any(
            word in description_lower for word in ["design", "architect", "structure"]
        ):
            return "system_design"
        elif any(word in description_lower for word in ["review", "audit", "analyze"]):
            return "architecture_review"
        elif any(
            word in description_lower for word in ["refactor", "improve", "optimize"]
        ):
            return "refactoring_plan"
        elif any(
            word in description_lower
            for word in ["technology", "tech", "stack", "tool"]
        ):
            return "technology_decision"
        else:
            return "general_analysis"

    async def _create_system_design(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create a system design."""

        prompt = f"""
        As an expert system architect, create a comprehensive system design for:

        Task: {task.get("description", "")}

        Please provide:
        1. High-level architecture overview
        2. Component breakdown and responsibilities
        3. Data flow diagrams
        4. Technology stack recommendations
        5. Scalability considerations
        6. Security architecture
        7. Deployment strategy
        8. Monitoring and observability

        Focus on enterprise-grade, production-ready design patterns.
        """

        result = await self.execute_with_cli_tool(prompt)
        self.designs_created += 1

        return {
            "design_document": result.output if result.success else None,
            "technology_stack": self.technology_stacks,
            "design_patterns": self.design_patterns,
            "artifacts": ["system_design.md", "architecture_diagram.svg"],
            "recommendations": self._generate_architecture_recommendations(),
        }

    async def _review_architecture(self, task: dict[str, Any]) -> dict[str, Any]:
        """Review existing architecture."""

        prompt = f"""
        As an expert system architect, conduct a comprehensive architecture review:

        Task: {task.get("description", "")}

        Please analyze:
        1. Current architecture strengths and weaknesses
        2. Scalability bottlenecks
        3. Security vulnerabilities
        4. Performance issues
        5. Code maintainability
        6. Technical debt
        7. Compliance with best practices
        8. Recommendations for improvement

        Provide specific, actionable recommendations with priority levels.
        """

        result = await self.execute_with_cli_tool(prompt)
        self.architectures_reviewed += 1

        return {
            "review_report": result.output if result.success else None,
            "issues_identified": self._identify_common_issues(),
            "recommendations": self._generate_improvement_recommendations(),
            "artifacts": ["architecture_review.md", "improvement_roadmap.md"],
            "priority_actions": ["high", "medium", "low"],
        }

    async def _create_refactoring_plan(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create a refactoring plan."""

        prompt = f"""
        As an expert system architect, create a comprehensive refactoring plan:

        Task: {task.get("description", "")}

        Please provide:
        1. Current state analysis
        2. Target state definition
        3. Step-by-step refactoring plan
        4. Risk assessment and mitigation
        5. Testing strategy
        6. Rollback procedures
        7. Timeline and milestones
        8. Success metrics

        Ensure minimal disruption to existing functionality.
        """

        result = await self.execute_with_cli_tool(prompt)
        self.refactoring_plans += 1

        return {
            "refactoring_plan": result.output if result.success else None,
            "phases": ["analysis", "design", "implementation", "testing", "deployment"],
            "risks": self._assess_refactoring_risks(),
            "artifacts": ["refactoring_plan.md", "migration_guide.md"],
            "estimated_effort": "2-4 weeks",
        }

    async def _make_technology_decision(self, task: dict[str, Any]) -> dict[str, Any]:
        """Make technology stack decisions."""

        prompt = f"""
        As an expert system architect, make informed technology decisions:

        Task: {task.get("description", "")}

        Please evaluate:
        1. Current technology stack assessment
        2. Alternative technology options
        3. Pros and cons comparison
        4. Migration complexity
        5. Long-term maintainability
        6. Team expertise requirements
        7. Cost implications
        8. Final recommendations

        Provide evidence-based recommendations with reasoning.
        """

        result = await self.execute_with_cli_tool(prompt)
        self.decisions_made += 1

        return {
            "decision_document": result.output if result.success else None,
            "recommended_stack": self.technology_stacks,
            "alternatives_considered": ["option_a", "option_b", "option_c"],
            "decision_criteria": [
                "performance",
                "maintainability",
                "team_expertise",
                "cost",
            ],
            "artifacts": ["technology_decision.md", "migration_plan.md"],
        }

    async def _general_architecture_analysis(
        self, task: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform general architecture analysis."""

        prompt = f"""
        As an expert system architect, analyze the following:

        Task: {task.get("description", "")}

        Please provide:
        1. Comprehensive analysis
        2. Key findings and insights
        3. Recommendations
        4. Best practices
        5. Implementation guidance

        Focus on architectural excellence and long-term sustainability.
        """

        result = await self.execute_with_cli_tool(prompt)

        return {
            "analysis_report": result.output if result.success else None,
            "insights": self._generate_architectural_insights(),
            "recommendations": self._generate_architecture_recommendations(),
            "artifacts": ["analysis_report.md"],
        }

    def _generate_architecture_recommendations(self) -> list[str]:
        """Generate standard architecture recommendations."""
        return [
            "Implement microservices architecture for better scalability",
            "Use event-driven patterns for loose coupling",
            "Implement proper caching strategies",
            "Ensure comprehensive monitoring and observability",
            "Follow domain-driven design principles",
            "Implement circuit breaker patterns for resilience",
            "Use infrastructure as code for deployments",
            "Implement proper security patterns and practices",
        ]

    def _generate_improvement_recommendations(self) -> list[str]:
        """Generate improvement recommendations."""
        return [
            "Refactor monolithic components into microservices",
            "Implement better error handling and logging",
            "Optimize database queries and indexing",
            "Improve API design and documentation",
            "Enhance security measures and authentication",
            "Implement automated testing strategies",
            "Optimize performance bottlenecks",
            "Improve deployment and CI/CD processes",
        ]

    def _identify_common_issues(self) -> list[str]:
        """Identify common architectural issues."""
        return [
            "Tight coupling between components",
            "Lack of proper error handling",
            "Insufficient monitoring and logging",
            "Poor API design and documentation",
            "Inadequate security measures",
            "Performance bottlenecks",
            "Technical debt accumulation",
            "Lack of automated testing",
        ]

    def _assess_refactoring_risks(self) -> list[str]:
        """Assess common refactoring risks."""
        return [
            "Breaking existing functionality",
            "Data migration complexities",
            "Integration challenges",
            "Performance regression",
            "Team learning curve",
            "Extended development timeline",
            "Increased maintenance overhead",
            "Compatibility issues",
        ]

    def _generate_architectural_insights(self) -> list[str]:
        """Generate architectural insights."""
        return [
            "System shows good separation of concerns",
            "Database design follows normalization principles",
            "API endpoints are well-structured",
            "Security measures are appropriately implemented",
            "Monitoring coverage is comprehensive",
            "Code maintainability is above average",
            "Performance characteristics are acceptable",
            "Scalability patterns are properly implemented",
        ]

    async def _on_collaboration_completed(self, result: dict[str, Any]) -> None:
        """Handle collaboration completion."""

        logger.info(
            "Architecture collaboration completed",
            agent=self.name,
            collaboration_id=result.get("collaboration_id"),
            duration=result.get("duration", 0),
            success=result.get("success", False),
        )

        # Store insights from the collaboration
        insights = {
            "collaboration_type": "architecture",
            "outcome": "success" if result.get("success") else "failure",
            "lessons_learned": [
                "Effective communication is key to successful collaboration",
                "Clear task definitions improve execution efficiency",
                "Regular status updates help maintain momentum",
            ],
            "performance_metrics": {
                "total_collaborations": self.collaborations_led,
                "designs_created": self.designs_created,
                "reviews_completed": self.architectures_reviewed,
            },
        }

        # Store context for future use
        await self.store_context(
            content=f"Architecture collaboration completed: {result.get('title', 'Unknown')}",
            context_type="collaboration_outcome",
            metadata=insights,
        )

    async def _get_next_architecture_task(self) -> Task | None:
        """Get next architecture-related task from queue."""
        from ..core.task_queue import task_queue

        # Look for architecture-specific task types
        _architecture_task_types = [
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
        ]

        return await task_queue.get_task(agent_id=self.name)

    async def _process_task_implementation(self, task: Task) -> TaskResult:
        """Architecture-specific task processing."""
        start_time = time.time()

        try:
            # Route to specific architecture method based on task type
            if task.task_type == "design_system":
                result = await self._design_system(task)
            elif task.task_type == "review_architecture":
                result = await self._review_architecture(task)
            elif task.task_type == "refactor_code":
                result = await self._plan_refactoring(task)
            elif task.task_type == "plan_migration":
                result = await self._plan_migration(task)
            elif task.task_type == "optimize_performance":
                result = await self._optimize_performance(task)
            elif task.task_type == "design_api":
                result = await self._design_api(task)
            elif task.task_type == "create_documentation":
                result = await self._create_architecture_documentation(task)
            elif task.task_type == "analyze_dependencies":
                result = await self._analyze_dependencies(task)
            elif task.task_type == "design_database":
                result = await self._design_database(task)
            elif task.task_type == "plan_scaling":
                result = await self._plan_scaling(task)
            else:
                # Fallback to base implementation
                result = await super()._process_task_implementation(task)

            execution_time = time.time() - start_time

            # Store architecture context
            await self.store_context(
                content=f"Architecture Task: {task.task_type}\nResult: {'Success' if result.success else 'Failed'}\nSummary: {str(result.data)[:500]}",
                importance_score=0.9,  # Architecture decisions are high importance
                category="architecture",
                metadata={
                    "task_type": task.task_type,
                    "execution_time": execution_time,
                    "success": result.success,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Architecture task processing failed",
                agent=self.name,
                task_id=task.id,
                error=str(e),
            )
            return TaskResult(success=False, error=str(e))

    async def _design_system(self, task: Task) -> TaskResult:
        """Design a new system or component."""
        requirements = task.metadata.get("requirements", task.description)
        scope = task.metadata.get("scope", "component")
        constraints = task.metadata.get("constraints", [])

        # Get context about existing architecture
        context_results = await self.retrieve_context(
            f"system design architecture {scope}", limit=10
        )

        context_text = "\n".join([r.context.content for r in context_results])

        prompt = f"""
        As a Senior System Architect, design a {scope} system based on the following requirements.

        Requirements: {requirements}
        Scope: {scope}
        Constraints: {constraints}

        Existing Architecture Context:
        {context_text}

        Please provide:
        1. High-level system architecture diagram (text-based)
        2. Component breakdown and responsibilities
        3. Data flow and interaction patterns
        4. Technology stack recommendations
        5. Scalability considerations
        6. Security architecture
        7. Deployment strategy
        8. Monitoring and observability approach
        9. Risk assessment and mitigation strategies
        10. Implementation roadmap with phases

        Format the output as a comprehensive architecture document with clear sections.
        Consider maintainability, scalability, and operational requirements.
        """

        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            self.designs_created += 1

            # Extract key architectural decisions
            decisions = self._extract_architectural_decisions(result.output)

            return TaskResult(
                success=True,
                data={
                    "architecture_design": result.output,
                    "architectural_decisions": decisions,
                    "technology_stack": self._extract_technology_stack(result.output),
                    "tool_used": result.tool_used,
                    "design_patterns": self._identify_patterns(result.output),
                },
                metrics={
                    "decisions_count": len(decisions),
                    "design_complexity": self._assess_design_complexity(result.output),
                    "completeness_score": self._assess_design_completeness(
                        result.output
                    ),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _review_architecture(self, task: Task) -> TaskResult:
        """Review existing architecture for improvements."""
        target_path = task.metadata.get("target_path", "src/")
        focus_areas = task.metadata.get("focus_areas", ["all"])

        # Analyze current codebase structure
        structure_analysis = await self._analyze_codebase_structure(target_path)

        # Get historical context
        context_results = await self.retrieve_context(
            f"architecture review {target_path}", limit=15
        )

        context_text = "\n".join([r.context.content for r in context_results])

        prompt = f"""
        As a Senior System Architect, review the current architecture and identify improvements.

        Target: {target_path}
        Focus Areas: {focus_areas}

        Current Structure Analysis:
        {json.dumps(structure_analysis, indent=2)}

        Historical Context:
        {context_text}

        Please analyze:
        1. Current architecture strengths and weaknesses
        2. Design pattern usage and appropriateness
        3. Code organization and modularity
        4. Coupling and cohesion analysis
        5. Scalability bottlenecks
        6. Security vulnerabilities in design
        7. Performance implications
        8. Maintainability concerns
        9. Technical debt assessment
        10. Specific improvement recommendations with priorities

        Provide actionable recommendations with implementation effort estimates.
        """

        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            self.architectures_reviewed += 1

            # Extract recommendations
            recommendations = self._extract_recommendations(result.output)

            return TaskResult(
                success=True,
                data={
                    "architecture_review": result.output,
                    "structure_analysis": structure_analysis,
                    "recommendations": recommendations,
                    "priority_issues": self._identify_priority_issues(result.output),
                    "tool_used": result.tool_used,
                },
                metrics={
                    "issues_identified": len(recommendations),
                    "technical_debt_score": self._assess_technical_debt(result.output),
                    "architecture_quality_score": self._assess_architecture_quality(
                        structure_analysis
                    ),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _plan_refactoring(self, task: Task) -> TaskResult:
        """Create refactoring plan for code improvement."""
        target_path = task.metadata.get("target_path", "src/")
        refactoring_goals = task.metadata.get("goals", ["improve maintainability"])

        # Analyze current state
        structure_analysis = await self._analyze_codebase_structure(target_path)

        prompt = f"""
        As a Senior System Architect, create a comprehensive refactoring plan.

        Target: {target_path}
        Goals: {refactoring_goals}

        Current Structure:
        {json.dumps(structure_analysis, indent=2)}

        Please create:
        1. Refactoring objectives and success criteria
        2. Phase-by-phase refactoring plan
        3. Risk analysis and mitigation strategies
        4. Dependencies and order of operations
        5. Testing strategy during refactoring
        6. Rollback plans for each phase
        7. Resource and time estimates
        8. Code migration strategies
        9. Communication plan for team
        10. Metrics to track progress

        Ensure minimal disruption to ongoing development.
        """

        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            self.refactoring_plans += 1

            phases = self._extract_refactoring_phases(result.output)

            return TaskResult(
                success=True,
                data={
                    "refactoring_plan": result.output,
                    "phases": phases,
                    "risk_assessment": self._extract_risks(result.output),
                    "estimated_effort": self._extract_effort_estimates(result.output),
                    "tool_used": result.tool_used,
                },
                metrics={
                    "phases_count": len(phases),
                    "estimated_weeks": self._extract_timeline(result.output),
                    "complexity_score": self._assess_refactoring_complexity(
                        result.output
                    ),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _design_api(self, task: Task) -> TaskResult:
        """Design API architecture and specifications."""
        api_purpose = task.metadata.get("purpose", task.description)
        api_type = task.metadata.get("type", "REST")
        requirements = task.metadata.get("requirements", [])

        prompt = f"""
        As a Senior API Architect, design a comprehensive API solution.

        Purpose: {api_purpose}
        API Type: {api_type}
        Requirements: {requirements}

        Please design:
        1. API architecture and patterns
        2. Endpoint specifications with HTTP methods
        3. Request/response schemas
        4. Authentication and authorization strategy
        5. Error handling and status codes
        6. Rate limiting and throttling
        7. Versioning strategy
        8. Documentation structure
        9. Security considerations
        10. Performance optimization strategies
        11. Testing approach
        12. Monitoring and observability

        Provide OpenAPI/Swagger specifications where applicable.
        """

        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            # Extract API specifications
            endpoints = self._extract_api_endpoints(result.output)

            return TaskResult(
                success=True,
                data={
                    "api_design": result.output,
                    "endpoints": endpoints,
                    "authentication_strategy": self._extract_auth_strategy(
                        result.output
                    ),
                    "specifications": self._extract_api_specs(result.output),
                    "tool_used": result.tool_used,
                },
                metrics={
                    "endpoints_count": len(endpoints),
                    "design_completeness": self._assess_api_design_completeness(
                        result.output
                    ),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _design_database(self, task: Task) -> TaskResult:
        """Design database schema and architecture."""
        requirements = task.metadata.get("requirements", task.description)
        database_type = task.metadata.get("type", "relational")

        prompt = f"""
        As a Senior Database Architect, design a comprehensive database solution.

        Requirements: {requirements}
        Database Type: {database_type}

        Please design:
        1. Database schema with tables/collections
        2. Relationships and constraints
        3. Indexing strategy
        4. Data types and validation rules
        5. Performance optimization
        6. Scalability considerations
        7. Backup and recovery strategy
        8. Security and access control
        9. Migration strategy
        10. Monitoring and maintenance

        Include SQL DDL statements or NoSQL schema definitions.
        """

        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            tables = self._extract_database_tables(result.output)

            return TaskResult(
                success=True,
                data={
                    "database_design": result.output,
                    "tables": tables,
                    "indexes": self._extract_indexes(result.output),
                    "relationships": self._extract_relationships(result.output),
                    "tool_used": result.tool_used,
                },
                metrics={
                    "tables_count": len(tables),
                    "design_complexity": self._assess_database_complexity(
                        result.output
                    ),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _analyze_codebase_structure(self, path: str) -> dict[str, Any]:
        """Analyze codebase structure and metrics."""
        structure = {
            "total_files": 0,
            "python_files": 0,
            "directories": [],
            "file_sizes": {},
            "imports": {},
            "complexity_estimate": 0,
        }

        try:
            for root, dirs, files in os.walk(path):
                structure["directories"].extend(dirs)

                for file in files:
                    file_path = os.path.join(root, file)
                    structure["total_files"] += 1

                    if file.endswith(".py"):
                        structure["python_files"] += 1

                        # Get file size
                        try:
                            size = os.path.getsize(file_path)
                            structure["file_sizes"][file_path] = size
                            structure["complexity_estimate"] += (
                                size // 1000
                            )  # Simple complexity metric
                        except OSError:
                            pass

                        # Analyze imports (simplified)
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                content = f.read()
                                import_count = content.count("import ")
                                structure["imports"][file_path] = import_count
                        except (OSError, UnicodeDecodeError):
                            pass

            # Calculate metrics
            structure["avg_file_size"] = sum(structure["file_sizes"].values()) // max(
                1, len(structure["file_sizes"])
            )
            structure["total_imports"] = sum(structure["imports"].values())

        except Exception as e:
            logger.warning("Error analyzing codebase structure", error=str(e))
            structure["error"] = str(e)

        return structure

    def _extract_architectural_decisions(self, output: str) -> list[dict[str, Any]]:
        """Extract architectural decisions from design output."""
        decisions = []
        lines = output.split("\n")

        current_decision = None
        for line in lines:
            if "decision" in line.lower() or "choose" in line.lower():
                if current_decision:
                    decisions.append(current_decision)
                current_decision = {
                    "decision": line.strip(),
                    "rationale": "",
                    "alternatives": [],
                }
            elif current_decision and (
                "because" in line.lower() or "rationale" in line.lower()
            ):
                current_decision["rationale"] = line.strip()

        if current_decision:
            decisions.append(current_decision)

        return decisions

    def _extract_technology_stack(self, output: str) -> list[str]:
        """Extract technology stack from design output."""
        stack = []
        technologies = [
            "python",
            "fastapi",
            "postgresql",
            "redis",
            "docker",
            "kubernetes",
            "react",
            "vue",
            "angular",
            "nginx",
            "gunicorn",
            "uvicorn",
        ]

        output_lower = output.lower()
        for tech in technologies:
            if tech in output_lower:
                stack.append(tech)

        return stack

    def _identify_patterns(self, output: str) -> list[str]:
        """Identify design patterns mentioned in output."""
        patterns = []
        output_lower = output.lower()

        for pattern in self.design_patterns:
            if pattern in output_lower:
                patterns.append(pattern)

        return patterns

    def _assess_design_complexity(self, output: str) -> float:
        """Assess design complexity score."""
        # Simple heuristic based on output length and key terms
        complexity_indicators = [
            "microservice",
            "distributed",
            "async",
            "queue",
            "cache",
            "load balancer",
            "database",
            "api gateway",
            "event",
        ]

        base_score = len(output) / 1000  # Base complexity from length

        for indicator in complexity_indicators:
            if indicator in output.lower():
                base_score += 0.5

        return min(10.0, base_score)  # Cap at 10

    def _assess_design_completeness(self, output: str) -> float:
        """Assess completeness of design."""
        required_sections = [
            "architecture",
            "component",
            "data",
            "security",
            "deployment",
            "monitoring",
            "scalability",
            "risk",
            "implementation",
        ]

        found_sections = 0
        output_lower = output.lower()

        for section in required_sections:
            if section in output_lower:
                found_sections += 1

        return (found_sections / len(required_sections)) * 100

    async def _monitor_system_architecture(self) -> None:
        """Monitor system architecture health."""
        # Placeholder for architecture monitoring
        # This could check for architecture drift, performance issues, etc.
        pass

    async def health_check(self) -> HealthStatus:
        """Architecture-specific health check."""
        base_health = await super().health_check()

        if base_health != HealthStatus.HEALTHY:
            return base_health

        # Check architecture-specific health
        try:
            # Verify we can analyze code structure
            test_analysis = await self._analyze_codebase_structure("src/")

            if "error" in test_analysis:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.DEGRADED

    async def _on_collaboration_completed(self, result: dict[str, Any]) -> None:
        """Called when a collaboration is completed."""
        logger.info(
            "Architect collaboration completed",
            agent=self.name,
            collaboration_id=result.get("collaboration_id"),
            success=result.get("success"),
        )

    async def _on_collaboration_failed(self, failure_info: dict[str, Any]) -> None:
        """Called when a collaboration fails."""
        logger.warning(
            "Architect collaboration failed",
            agent=self.name,
            collaboration_id=failure_info.get("collaboration_id"),
            reason=failure_info.get("reason"),
        )

    # Additional helper methods for extraction and analysis
    def _extract_recommendations(self, output: str) -> list[str]:
        """Extract recommendations from output."""
        recommendations = []
        lines = output.split("\n")

        for line in lines:
            if any(
                word in line.lower()
                for word in ["recommend", "should", "consider", "improve"]
            ):
                recommendations.append(line.strip())

        return recommendations[:20]  # Limit to top 20

    def _identify_priority_issues(self, output: str) -> list[str]:
        """Identify high priority issues."""
        issues = []
        lines = output.split("\n")

        for line in lines:
            if any(
                word in line.lower()
                for word in ["critical", "urgent", "high priority", "blocker"]
            ):
                issues.append(line.strip())

        return issues

    def _assess_technical_debt(self, output: str) -> float:
        """Assess technical debt score from review."""
        debt_indicators = [
            "technical debt",
            "legacy",
            "deprecated",
            "workaround",
            "hack",
            "todo",
            "fixme",
            "refactor",
        ]

        score = 0.0
        output_lower = output.lower()

        for indicator in debt_indicators:
            score += output_lower.count(indicator) * 10

        return min(100.0, score)  # Cap at 100

    def _assess_architecture_quality(self, structure: dict[str, Any]) -> float:
        """Assess overall architecture quality."""
        if "error" in structure:
            return 0.0

        # Simple quality metrics
        quality_score = 100.0

        # Penalize for very large files
        avg_size = structure.get("avg_file_size", 0)
        if avg_size > 10000:  # Files larger than 10KB
            quality_score -= 20

        # Penalize for too many imports per file
        if structure.get("total_imports", 0) > structure.get("python_files", 1) * 10:
            quality_score -= 15

        return max(0.0, quality_score)

    def _extract_refactoring_phases(self, output: str) -> list[dict[str, str]]:
        """Extract refactoring phases from plan."""
        phases = []
        lines = output.split("\n")

        current_phase = None
        for line in lines:
            if "phase" in line.lower() and any(char.isdigit() for char in line):
                if current_phase:
                    phases.append(current_phase)
                current_phase = {
                    "name": line.strip(),
                    "description": "",
                    "duration": "",
                }
            elif current_phase and line.strip():
                current_phase["description"] += line.strip() + " "

        if current_phase:
            phases.append(current_phase)

        return phases

    def _extract_risks(self, output: str) -> list[str]:
        """Extract risk items from output."""
        risks = []
        lines = output.split("\n")

        for line in lines:
            if "risk" in line.lower():
                risks.append(line.strip())

        return risks

    def _extract_effort_estimates(self, output: str) -> dict[str, str]:
        """Extract effort estimates from output."""
        estimates = {}
        lines = output.split("\n")

        for line in lines:
            if any(
                word in line.lower() for word in ["days", "weeks", "months", "hours"]
            ):
                estimates[line.split(":")[0].strip()] = line.strip()

        return estimates

    def _extract_timeline(self, output: str) -> int:
        """Extract timeline in weeks from output."""
        import re

        # Look for week mentions
        week_matches = re.findall(r"(\d+)\s*weeks?", output.lower())
        if week_matches:
            return max(int(match) for match in week_matches)

        # Look for month mentions and convert
        month_matches = re.findall(r"(\d+)\s*months?", output.lower())
        if month_matches:
            return max(int(match) * 4 for match in month_matches)

        return 0

    def _assess_refactoring_complexity(self, output: str) -> float:
        """Assess refactoring complexity."""
        complexity_indicators = [
            "migration",
            "database",
            "breaking change",
            "api change",
            "dependency",
            "architecture",
            "framework",
        ]

        score = len(output) / 2000  # Base from length

        for indicator in complexity_indicators:
            if indicator in output.lower():
                score += 1.0

        return min(10.0, score)

    def _extract_api_endpoints(self, output: str) -> list[dict[str, str]]:
        """Extract API endpoints from design."""
        endpoints = []
        lines = output.split("\n")

        for line in lines:
            if any(
                method in line.upper()
                for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]
            ):
                endpoints.append(
                    {"endpoint": line.strip(), "method": "", "description": ""}
                )

        return endpoints

    def _extract_auth_strategy(self, output: str) -> str:
        """Extract authentication strategy."""
        auth_keywords = ["jwt", "oauth", "basic auth", "api key", "bearer token"]

        for keyword in auth_keywords:
            if keyword in output.lower():
                return keyword

        return "not specified"

    def _extract_api_specs(self, output: str) -> dict[str, Any]:
        """Extract API specifications."""
        specs = {
            "version": "1.0",
            "base_url": "",
            "authentication": self._extract_auth_strategy(output),
            "rate_limiting": "rate limiting" in output.lower(),
        }

        return specs

    def _assess_api_design_completeness(self, output: str) -> float:
        """Assess API design completeness."""
        required_elements = [
            "endpoint",
            "authentication",
            "error",
            "status code",
            "request",
            "response",
            "documentation",
            "versioning",
        ]

        found = 0
        output_lower = output.lower()

        for element in required_elements:
            if element in output_lower:
                found += 1

        return (found / len(required_elements)) * 100

    def _extract_database_tables(self, output: str) -> list[str]:
        """Extract database tables from design."""
        tables = []
        lines = output.split("\n")

        for line in lines:
            if "table" in line.lower() or "create table" in line.lower():
                tables.append(line.strip())

        return tables

    def _extract_indexes(self, output: str) -> list[str]:
        """Extract database indexes."""
        indexes = []
        lines = output.split("\n")

        for line in lines:
            if "index" in line.lower():
                indexes.append(line.strip())

        return indexes

    def _extract_relationships(self, output: str) -> list[str]:
        """Extract database relationships."""
        relationships = []
        lines = output.split("\n")

        for line in lines:
            if any(
                word in line.lower()
                for word in ["foreign key", "relationship", "references"]
            ):
                relationships.append(line.strip())

        return relationships

    def _assess_database_complexity(self, output: str) -> float:
        """Assess database design complexity."""
        complexity_indicators = [
            "foreign key",
            "index",
            "constraint",
            "trigger",
            "procedure",
            "view",
            "partition",
        ]

        score = len(output) / 1000

        for indicator in complexity_indicators:
            score += output.lower().count(indicator) * 0.5

        return min(10.0, score)
