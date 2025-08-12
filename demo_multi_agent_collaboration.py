"""
Multi-Agent Collaboration Demo

This demo showcases agents using enhanced communication features to collaborate
on a real code review workflow.
"""

import asyncio
import time
from typing import Any

from src.agents.base_agent import BaseAgent, TaskResult
from src.core.enhanced_message_broker import get_enhanced_message_broker
from src.core.realtime_collaboration import get_collaboration_sync


class DeveloperAgent(BaseAgent):
    """Developer agent that creates and shares code."""

    def __init__(self, name: str):
        super().__init__(name=name, agent_type="developer", role="code_development")

    async def run(self) -> None:
        """Developer agent main loop - simplified for demo."""
        await asyncio.sleep(0.1)  # Minimal run loop for demo

    async def create_code_module(
        self, module_name: str, content: str
    ) -> dict[str, Any]:
        """Create a new code module."""

        # Create shared work session for the code review
        session_id = await self.create_shared_work_session(
            title=f"Code Review: {module_name}",
            participants=[self.name],  # Will invite others later
            task="code_review",
            shared_resources={
                "module_name": module_name,
                "module_type": "authentication",
                "status": "draft",
            },
        )

        if not session_id:
            return {"success": False, "error": "Failed to create work session"}

        # Share the code context
        context_id = await self.share_work_context(
            context_type="work_session",
            data={
                "code": content,
                "module_name": module_name,
                "created_by": self.name,
                "created_at": time.time(),
                "review_status": "pending",
                "language": "python",
            },
        )

        # Update agent status
        await self.update_agent_status(
            status="busy",
            current_task=f"developing_{module_name}",
            capabilities=["python_development", "code_creation", "module_design"],
        )

        return {
            "success": True,
            "session_id": session_id,
            "context_id": context_id,
            "module_name": module_name,
        }


class QAAgent(BaseAgent):
    """QA agent that reviews and tests code."""

    def __init__(self, name: str):
        super().__init__(name=name, agent_type="qa", role="quality_assurance")

    async def run(self) -> None:
        """QA agent main loop - simplified for demo."""
        await asyncio.sleep(0.1)  # Minimal run loop for demo

    async def join_code_review(self, session_id: str) -> dict[str, Any]:
        """Join a code review session and provide feedback."""

        # Join the collaboration session
        join_success = await self.join_shared_work_session(session_id)
        if not join_success:
            return {"success": False, "error": "Failed to join session"}

        # Update status to show we're reviewing
        await self.update_agent_status(
            status="busy",
            current_task=f"reviewing_code_session_{session_id}",
            capabilities=[
                "code_review",
                "testing",
                "security_analysis",
                "performance_analysis",
            ],
        )

        # Simulate reviewing the code and finding issues
        review_feedback = {
            "overall_rating": "good_with_issues",
            "security_issues": [
                {
                    "severity": "medium",
                    "issue": "Missing input validation on user credentials",
                    "suggestion": "Add parameter validation and sanitization",
                    "line_numbers": [15, 23],
                }
            ],
            "performance_issues": [
                {
                    "severity": "low",
                    "issue": "Inefficient database query pattern",
                    "suggestion": "Consider using connection pooling",
                    "line_numbers": [45],
                }
            ],
            "best_practices": [
                "Consider adding comprehensive error handling",
                "Add unit tests for edge cases",
                "Document the authentication flow",
            ],
            "approval_status": "conditional",
            "required_changes": ["input_validation", "error_handling"],
        }

        return {
            "success": True,
            "feedback": review_feedback,
            "reviewer": self.name,
            "reviewed_at": time.time(),
        }


class ArchitectAgent(BaseAgent):
    """Architect agent that provides high-level design guidance."""

    def __init__(self, name: str):
        super().__init__(name=name, agent_type="architect", role="system_architecture")

    async def run(self) -> None:
        """Architect agent main loop - simplified for demo."""
        await asyncio.sleep(0.1)  # Minimal run loop for demo

    async def provide_architectural_guidance(self, session_id: str) -> Dict[str, Any]:
        """Join session and provide architectural guidance."""

        # Join the session
        join_success = await self.join_shared_work_session(session_id)
        if not join_success:
            return {"success": False, "error": "Failed to join session"}

        # Update status
        await self.update_agent_status(
            status="busy",
            current_task=f"architectural_review_{session_id}",
            capabilities=[
                "system_design",
                "architecture_review",
                "scalability_analysis",
            ],
        )

        # Provide architectural feedback
        architectural_guidance = {
            "design_patterns": [
                {
                    "pattern": "Factory Pattern",
                    "reasoning": "For creating different authentication providers",
                    "implementation": "Create AuthProviderFactory class",
                },
                {
                    "pattern": "Strategy Pattern",
                    "reasoning": "For different authentication strategies (OAuth, LDAP, etc)",
                    "implementation": "Define AuthStrategy interface",
                },
            ],
            "scalability_concerns": [
                "Consider implementing caching layer for user sessions",
                "Design for horizontal scaling with stateless authentication",
                "Plan for rate limiting and DDoS protection",
            ],
            "integration_points": [
                "User management system",
                "Logging and monitoring",
                "Configuration management",
                "Security audit trail",
            ],
            "recommended_tech_stack": {
                "auth_library": "PyJWT for token handling",
                "caching": "Redis for session storage",
                "database": "PostgreSQL with proper indexing",
                "monitoring": "Structured logging with correlation IDs",
            },
        }

        return {
            "success": True,
            "guidance": architectural_guidance,
            "architect": self.name,
            "provided_at": time.time(),
        }


async def collaborative_code_review_workflow():
    """
    Demonstrates a complete multi-agent collaboration workflow for code review.

    Flow:
    1. Developer creates authentication module and starts code review session
    2. QA agent joins to review for testing and security issues
    3. Architect joins to provide high-level design guidance
    4. All agents collaborate in real-time with shared context
    """

    print("🚀 Starting Multi-Agent Collaborative Code Review Demo")
    print("=" * 60)

    # Initialize agents
    developer = DeveloperAgent("alice_dev")
    qa_agent = QAAgent("bob_qa")
    architect = ArchitectAgent("charlie_architect")

    # Sample authentication module code
    auth_module_code = '''
class AuthenticationModule:
    """Handles user authentication and session management."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.session_timeout = 3600  # 1 hour
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials."""
        # TODO: Add input validation
        user = self.db.query(f"SELECT * FROM users WHERE username='{username}'")
        
        if user and self.verify_password(password, user.password_hash):
            session_token = self.create_session(user.id)
            return {"success": True, "token": session_token}
        else:
            return {"success": False, "error": "Invalid credentials"}
    
    def verify_password(self, password, hash):
        """Verify password against stored hash."""
        # Simplified password verification
        return hash == f"hash_{password}"
    
    def create_session(self, user_id):
        """Create user session."""
        # TODO: Add proper session management
        import uuid
        session_id = str(uuid.uuid4())
        
        # Store session in database
        self.db.execute(f"INSERT INTO sessions (id, user_id) VALUES ('{session_id}', {user_id})")
        
        return session_id
'''

    try:
        # Phase 1: Developer creates code and starts collaboration
        print("📝 Phase 1: Developer creates authentication module")

        creation_result = await developer.create_code_module(
            module_name="AuthenticationModule", content=auth_module_code
        )

        if not creation_result["success"]:
            print(f"❌ Failed to create module: {creation_result['error']}")
            return

        session_id = creation_result["session_id"]
        print(f"✅ Code module created and shared in session: {session_id}")
        print(f"🔗 Context ID: {creation_result['context_id']}")

        # Phase 2: QA agent joins and reviews
        print("\n🔍 Phase 2: QA agent joins for security and testing review")

        qa_review_result = await qa_agent.join_code_review(session_id)

        if qa_review_result["success"]:
            feedback = qa_review_result["feedback"]
            print(f"✅ QA Review completed by {qa_review_result['reviewer']}")
            print(f"📊 Overall Rating: {feedback['overall_rating']}")
            print(f"🔒 Security Issues: {len(feedback['security_issues'])}")
            print(f"⚡ Performance Issues: {len(feedback['performance_issues'])}")
            print(f"✨ Best Practices: {len(feedback['best_practices'])}")
            print(f"🎯 Approval Status: {feedback['approval_status']}")
        else:
            print(f"❌ QA Review failed: {qa_review_result['error']}")

        # Phase 3: Architect provides design guidance
        print("\n🏗️ Phase 3: Architect provides design guidance")

        arch_guidance_result = await architect.provide_architectural_guidance(
            session_id
        )

        if arch_guidance_result["success"]:
            guidance = arch_guidance_result["guidance"]
            print(
                f"✅ Architectural guidance provided by {arch_guidance_result['architect']}"
            )
            print(f"🎨 Design Patterns: {len(guidance['design_patterns'])}")
            print(f"📈 Scalability Concerns: {len(guidance['scalability_concerns'])}")
            print(f"🔌 Integration Points: {len(guidance['integration_points'])}")

            # Show specific recommendations
            print("\n📋 Key Recommendations:")
            for pattern in guidance["design_patterns"]:
                print(f"  • {pattern['pattern']}: {pattern['reasoning']}")
        else:
            print(f"❌ Architectural guidance failed: {arch_guidance_result['error']}")

        # Phase 4: Show collaboration opportunities
        print("\n🤝 Phase 4: Discovering collaboration opportunities")

        dev_opportunities = await developer.get_collaboration_opportunities()
        print(
            f"🔍 Developer found {len(dev_opportunities)} collaboration opportunities:"
        )

        for opportunity in dev_opportunities:
            print(
                f"  • {opportunity['agent_name']} ({opportunity['status']}) - {', '.join(opportunity['capabilities'][:3])}..."
            )

        # Phase 5: Enhanced messaging demonstration
        print("\n📨 Phase 5: Enhanced messaging with context")

        # Developer sends high-priority message to QA with context
        context_ids = (
            [creation_result["context_id"]] if creation_result["context_id"] else []
        )

        message_sent = await developer.send_enhanced_message(
            to_agent=qa_agent.name,
            topic="urgent_security_review",
            content={
                "message": "Please prioritize security review of authentication module",
                "priority_reason": "Production deployment scheduled for tomorrow",
                "specific_concerns": [
                    "SQL injection",
                    "password handling",
                    "session management",
                ],
            },
            context_ids=context_ids,
            priority="high",
            include_context=True,
        )

        if message_sent:
            print("✅ High-priority context-aware message sent successfully")
        else:
            print("❌ Failed to send enhanced message")

        # Show final collaboration summary
        print("\n📊 Collaboration Summary")
        print("=" * 40)
        print("👥 Participants: Developer, QA, Architect")
        print("🎯 Objective: Authentication module review")
        print("📈 Status: Collaborative review completed")
        print("🔄 Next Steps: Address feedback and iterate")

        # Show enhanced communication metrics if available
        if developer.enhanced_broker:
            try:
                metrics = await developer.enhanced_broker.get_communication_performance_metrics()
                if metrics and "enhanced_features" in metrics:
                    enhanced_stats = metrics["enhanced_features"]
                    print(
                        f"📡 Active Contexts: {enhanced_stats.get('shared_contexts_active', 0)}"
                    )
                    print(
                        f"🔧 Sync Tasks: {enhanced_stats.get('sync_tasks_running', 0)}"
                    )
            except Exception as e:
                print(f"⚠️ Could not retrieve metrics: {e}")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up demo resources...")

        # Update all agents to idle status
        for agent in [developer, qa_agent, architect]:
            try:
                await agent.update_agent_status(status="idle", current_task=None)
            except Exception:
                pass  # Ignore cleanup errors

        print("✅ Demo completed successfully!")


if __name__ == "__main__":
    """Run the collaborative code review demo."""
    asyncio.run(collaborative_code_review_workflow())
