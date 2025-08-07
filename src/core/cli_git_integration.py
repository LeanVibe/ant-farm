"""Enhanced CLI tool execution with integrated Git workflow management."""

import asyncio
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .config import settings

logger = structlog.get_logger()


class WorkflowType(Enum):
    """Types of development workflows."""

    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    OPTIMIZATION = "optimization"


class WorkflowStatus(Enum):
    """Status of a development workflow."""

    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    READY_FOR_REVIEW = "ready_for_review"
    APPROVED = "approved"
    MERGED = "merged"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowContext:
    """Context for a development workflow."""

    id: str
    workflow_type: WorkflowType
    description: str
    branch_name: str
    base_commit: str
    target_branch: str = "main"
    status: WorkflowStatus = WorkflowStatus.INITIALIZING
    created_at: float = 0.0
    updated_at: float = 0.0
    agent_id: str = ""
    files_modified: list[str] = None
    commits: list[str] = None
    test_results: dict[str, Any] = None
    review_comments: list[str] = None

    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []
        if self.commits is None:
            self.commits = []
        if self.test_results is None:
            self.test_results = {}
        if self.review_comments is None:
            self.review_comments = []
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = time.time()


@dataclass
class CLIExecutionContext:
    """Context for CLI tool execution with git integration."""

    task_description: str
    workflow_context: WorkflowContext | None = None
    auto_commit: bool = True
    run_tests: bool = True
    create_branch: bool = True
    commit_message_prefix: str = ""
    expected_file_changes: list[str] = None
    success_criteria: list[str] = None

    def __post_init__(self):
        if self.expected_file_changes is None:
            self.expected_file_changes = []
        if self.success_criteria is None:
            self.success_criteria = []


@dataclass
class ExecutionResult:
    """Result of CLI tool execution with git integration."""

    success: bool
    cli_output: str
    cli_error: str | None = None
    tool_used: str | None = None
    execution_time: float = 0.0
    workflow_context: WorkflowContext | None = None
    files_changed: list[str] = None
    commit_hash: str | None = None
    test_results: dict[str, Any] = None
    git_status: str = ""

    def __post_init__(self):
        if self.files_changed is None:
            self.files_changed = []
        if self.test_results is None:
            self.test_results = {}


class GitWorkflowManager:
    """Manages Git workflows for development tasks."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.active_workflows: dict[str, WorkflowContext] = {}

    async def create_workflow(
        self,
        workflow_type: WorkflowType,
        description: str,
        agent_id: str,
        target_branch: str = "main",
    ) -> WorkflowContext:
        """Create a new development workflow."""

        workflow_id = str(uuid.uuid4())[:8]

        # Ensure we're on the target branch and up to date
        await self._ensure_clean_state(target_branch)

        # Get current commit as base
        base_commit = await self._get_current_commit()

        # Generate branch name
        branch_name = f"{workflow_type.value.replace('_', '-')}/{workflow_id}"

        # Create workflow context
        workflow_context = WorkflowContext(
            id=workflow_id,
            workflow_type=workflow_type,
            description=description,
            branch_name=branch_name,
            base_commit=base_commit,
            target_branch=target_branch,
            status=WorkflowStatus.INITIALIZING,
            agent_id=agent_id,
        )

        # Create branch
        await self._create_branch(branch_name)

        workflow_context.status = WorkflowStatus.IN_PROGRESS
        workflow_context.updated_at = time.time()

        self.active_workflows[workflow_id] = workflow_context

        logger.info(
            "Workflow created",
            workflow_id=workflow_id,
            type=workflow_type.value,
            branch=branch_name,
        )

        return workflow_context

    async def update_workflow_status(
        self, workflow_id: str, status: WorkflowStatus, commit_hash: str | None = None
    ) -> bool:
        """Update workflow status."""

        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]
        workflow.status = status
        workflow.updated_at = time.time()

        if commit_hash:
            workflow.commits.append(commit_hash)

        logger.info(
            "Workflow status updated", workflow_id=workflow_id, status=status.value
        )

        return True

    async def complete_workflow(
        self, workflow_id: str, merge_to_main: bool = False
    ) -> bool:
        """Complete a workflow and optionally merge to main."""

        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]

        try:
            if merge_to_main and workflow.status == WorkflowStatus.APPROVED:
                # Switch to target branch
                await self._run_git_command(["checkout", workflow.target_branch])

                # Pull latest changes
                await self._run_git_command(["pull", "origin", workflow.target_branch])

                # Merge workflow branch
                await self._run_git_command(
                    [
                        "merge",
                        workflow.branch_name,
                        "--no-ff",
                        "-m",
                        f"Merge {workflow.description}",
                    ]
                )

                # Delete workflow branch
                await self._run_git_command(["branch", "-d", workflow.branch_name])

                workflow.status = WorkflowStatus.MERGED

                logger.info(
                    "Workflow merged to main",
                    workflow_id=workflow_id,
                    branch=workflow.branch_name,
                )
            else:
                workflow.status = WorkflowStatus.READY_FOR_REVIEW

            workflow.updated_at = time.time()

            # Remove from active workflows if merged
            if workflow.status == WorkflowStatus.MERGED:
                del self.active_workflows[workflow_id]

            return True

        except Exception as e:
            logger.error(
                "Workflow completion failed", workflow_id=workflow_id, error=str(e)
            )
            workflow.status = WorkflowStatus.FAILED
            return False

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow and clean up."""

        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]

        try:
            # Switch back to target branch
            await self._run_git_command(["checkout", workflow.target_branch])

            # Delete workflow branch
            await self._run_git_command(["branch", "-D", workflow.branch_name])

            workflow.status = WorkflowStatus.CANCELLED
            del self.active_workflows[workflow_id]

            logger.info(
                "Workflow cancelled",
                workflow_id=workflow_id,
                branch=workflow.branch_name,
            )

            return True

        except Exception as e:
            logger.error(
                "Workflow cancellation failed", workflow_id=workflow_id, error=str(e)
            )
            return False

    async def get_workflow_status(self, workflow_id: str) -> WorkflowContext | None:
        """Get workflow status."""
        return self.active_workflows.get(workflow_id)

    async def list_active_workflows(self) -> list[WorkflowContext]:
        """List all active workflows."""
        return list(self.active_workflows.values())

    async def _ensure_clean_state(self, target_branch: str) -> None:
        """Ensure repository is in clean state on target branch."""

        # Check if we have uncommitted changes
        status_result = await self._run_git_command(["status", "--porcelain"])
        if status_result.stdout.strip():
            # Stash changes
            await self._run_git_command(
                ["stash", "push", "-m", "Auto-stash before workflow"]
            )
            logger.info("Uncommitted changes stashed")

        # Switch to target branch
        await self._run_git_command(["checkout", target_branch])

        # Pull latest changes
        try:
            await self._run_git_command(["pull", "origin", target_branch])
        except subprocess.CalledProcessError:
            # Pull might fail if no remote - that's ok for local development
            pass

    async def _create_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch."""
        await self._run_git_command(["checkout", "-b", branch_name])

    async def _get_current_commit(self) -> str:
        """Get current commit hash."""
        result = await self._run_git_command(["rev-parse", "HEAD"])
        return result.stdout.strip()

    async def _run_git_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        return result


class EnhancedCLIExecutor:
    """Enhanced CLI tool executor with git integration."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.git_manager = GitWorkflowManager(workspace_path)

    async def execute_with_workflow(
        self, cli_tools_manager, execution_context: CLIExecutionContext, agent_id: str
    ) -> ExecutionResult:
        """Execute CLI tools with integrated git workflow."""

        start_time = time.time()
        workflow_context = execution_context.workflow_context

        try:
            # Create workflow if requested and not provided
            if execution_context.create_branch and not workflow_context:
                workflow_type = self._determine_workflow_type(
                    execution_context.task_description
                )
                workflow_context = await self.git_manager.create_workflow(
                    workflow_type=workflow_type,
                    description=execution_context.task_description,
                    agent_id=agent_id,
                )

            # Record initial state
            initial_status = await self._get_git_status()
            initial_files = await self._get_tracked_files()

            # Execute CLI tool
            cli_result = await cli_tools_manager.execute_prompt(
                execution_context.task_description
            )

            execution_time = time.time() - start_time

            if not cli_result.success:
                if workflow_context:
                    await self.git_manager.update_workflow_status(
                        workflow_context.id, WorkflowStatus.FAILED
                    )

                return ExecutionResult(
                    success=False,
                    cli_output=cli_result.output,
                    cli_error=cli_result.error,
                    tool_used=cli_result.tool_used,
                    execution_time=execution_time,
                    workflow_context=workflow_context,
                )

            # Analyze changes
            final_status = await self._get_git_status()
            final_files = await self._get_tracked_files()
            files_changed = await self._get_changed_files(initial_files, final_files)

            # Auto-commit if requested and changes detected
            commit_hash = None
            if execution_context.auto_commit and files_changed:
                commit_message = self._generate_commit_message(
                    execution_context, files_changed, cli_result.tool_used
                )
                commit_hash = await self._commit_changes(commit_message)

                if workflow_context:
                    workflow_context.files_modified.extend(files_changed)
                    await self.git_manager.update_workflow_status(
                        workflow_context.id, WorkflowStatus.IN_PROGRESS, commit_hash
                    )

            # Run tests if requested
            test_results = {}
            if execution_context.run_tests and files_changed:
                test_results = await self._run_tests()

                if workflow_context:
                    workflow_context.test_results.update(test_results)

                    # Update status based on test results
                    if test_results.get("passed", False):
                        await self.git_manager.update_workflow_status(
                            workflow_context.id, WorkflowStatus.TESTING
                        )
                    else:
                        await self.git_manager.update_workflow_status(
                            workflow_context.id, WorkflowStatus.FAILED
                        )

            # Check success criteria
            success = self._evaluate_success_criteria(
                execution_context.success_criteria,
                cli_result,
                test_results,
                files_changed,
            )

            return ExecutionResult(
                success=success,
                cli_output=cli_result.output,
                cli_error=cli_result.error,
                tool_used=cli_result.tool_used,
                execution_time=execution_time,
                workflow_context=workflow_context,
                files_changed=files_changed,
                commit_hash=commit_hash,
                test_results=test_results,
                git_status=final_status,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if workflow_context:
                await self.git_manager.update_workflow_status(
                    workflow_context.id, WorkflowStatus.FAILED
                )

            logger.error(
                "Enhanced CLI execution failed",
                task=execution_context.task_description,
                error=str(e),
            )

            return ExecutionResult(
                success=False,
                cli_output="",
                cli_error=f"Execution failed: {str(e)}",
                execution_time=execution_time,
                workflow_context=workflow_context,
            )

    def _determine_workflow_type(self, task_description: str) -> WorkflowType:
        """Determine workflow type from task description."""
        description_lower = task_description.lower()

        if any(word in description_lower for word in ["bug", "fix", "error", "issue"]):
            return WorkflowType.BUG_FIX
        elif any(word in description_lower for word in ["test", "testing", "spec"]):
            return WorkflowType.TESTING
        elif any(
            word in description_lower for word in ["doc", "documentation", "readme"]
        ):
            return WorkflowType.DOCUMENTATION
        elif any(
            word in description_lower for word in ["refactor", "cleanup", "restructure"]
        ):
            return WorkflowType.REFACTORING
        elif any(
            word in description_lower for word in ["optimize", "performance", "speed"]
        ):
            return WorkflowType.OPTIMIZATION
        else:
            return WorkflowType.FEATURE_DEVELOPMENT

    async def _get_git_status(self) -> str:
        """Get git status."""
        try:
            result = await self._run_git_command(["status", "--porcelain"])
            return result.stdout
        except:
            return ""

    async def _get_tracked_files(self) -> set[str]:
        """Get list of tracked files."""
        try:
            result = await self._run_git_command(["ls-files"])
            return (
                set(result.stdout.strip().split("\n"))
                if result.stdout.strip()
                else set()
            )
        except:
            return set()

    async def _get_changed_files(
        self, initial_files: set[str], final_files: set[str]
    ) -> list[str]:
        """Get list of files that changed."""
        try:
            # Check git status for modified files
            result = await self._run_git_command(["diff", "--name-only"])
            modified_files = (
                result.stdout.strip().split("\n") if result.stdout.strip() else []
            )

            # Check for new files
            new_files = final_files - initial_files

            # Combine and deduplicate
            all_changed = set(modified_files) | new_files
            return [f for f in all_changed if f]
        except:
            return []

    async def _commit_changes(self, commit_message: str) -> str | None:
        """Commit changes and return commit hash."""
        try:
            # Add all changes
            await self._run_git_command(["add", "."])

            # Commit
            await self._run_git_command(["commit", "-m", commit_message])

            # Get commit hash
            result = await self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = result.stdout.strip()

            logger.info("Changes committed", commit_hash=commit_hash[:8])
            return commit_hash

        except subprocess.CalledProcessError as e:
            logger.error("Failed to commit changes", error=str(e))
            return None

    def _generate_commit_message(
        self, context: CLIExecutionContext, files_changed: list[str], tool_used: str
    ) -> str:
        """Generate commit message."""

        prefix = context.commit_message_prefix
        if not prefix:
            workflow_type = self._determine_workflow_type(context.task_description)
            prefix = {
                WorkflowType.FEATURE_DEVELOPMENT: "feat",
                WorkflowType.BUG_FIX: "fix",
                WorkflowType.REFACTORING: "refactor",
                WorkflowType.DOCUMENTATION: "docs",
                WorkflowType.TESTING: "test",
                WorkflowType.OPTIMIZATION: "perf",
            }.get(workflow_type, "chore")

        # Truncate description for commit message
        description = context.task_description
        if len(description) > 50:
            description = description[:47] + "..."

        commit_message = f"{prefix}: {description}"

        # Add details
        if files_changed:
            commit_message += f"\n\nFiles modified: {', '.join(files_changed[:5])}"
            if len(files_changed) > 5:
                commit_message += f" and {len(files_changed) - 5} more"

        if tool_used:
            commit_message += f"\nTool used: {tool_used}"

        commit_message += "\n\n🤖 Generated with [opencode](https://opencode.ai)\n\nCo-Authored-By: opencode <noreply@opencode.ai>"

        return commit_message

    async def _run_tests(self) -> dict[str, Any]:
        """Run tests and return results."""
        test_results = {"passed": False, "output": "", "error": ""}

        try:
            # Try different test commands
            test_commands = [
                ["make", "test"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test"],
                ["cargo", "test"],
            ]

            for cmd in test_commands:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=self.workspace_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        timeout=300,  # 5 minute timeout
                    )

                    stdout, stderr = await process.communicate()

                    test_results["output"] = stdout.decode()
                    test_results["error"] = stderr.decode()
                    test_results["passed"] = process.returncode == 0
                    test_results["command"] = " ".join(cmd)

                    if process.returncode == 0:
                        logger.info("Tests passed", command=" ".join(cmd))
                    else:
                        logger.warning("Tests failed", command=" ".join(cmd))

                    break  # Found working test command

                except (FileNotFoundError, asyncio.TimeoutError):
                    continue  # Try next command

        except Exception as e:
            test_results["error"] = str(e)
            logger.error("Test execution failed", error=str(e))

        return test_results

    def _evaluate_success_criteria(
        self,
        criteria: list[str],
        cli_result,
        test_results: dict[str, Any],
        files_changed: list[str],
    ) -> bool:
        """Evaluate success criteria."""

        if not criteria:
            # Default criteria: CLI success and tests pass (if run)
            cli_success = cli_result.success
            tests_success = test_results.get("passed", True)  # True if no tests run
            return cli_success and tests_success

        # Evaluate each criterion
        for criterion in criteria:
            criterion_lower = criterion.lower()

            if "cli success" in criterion_lower:
                if not cli_result.success:
                    return False

            elif "tests pass" in criterion_lower:
                if not test_results.get("passed", False):
                    return False

            elif "files changed" in criterion_lower:
                if not files_changed:
                    return False

            elif "no errors" in criterion_lower:
                if cli_result.error or test_results.get("error"):
                    return False

        return True

    async def _run_git_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a git command."""
        return await self.git_manager._run_git_command(args)


# Global instance
_enhanced_cli_executor: EnhancedCLIExecutor | None = None


def get_enhanced_cli_executor(workspace_path: str = None) -> EnhancedCLIExecutor:
    """Get the enhanced CLI executor singleton."""
    global _enhanced_cli_executor

    if _enhanced_cli_executor is None:
        if workspace_path is None:
            workspace_path = os.getcwd()
        _enhanced_cli_executor = EnhancedCLIExecutor(workspace_path)

    return _enhanced_cli_executor
