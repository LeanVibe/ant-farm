"""Safe self-modification system for LeanVibe Agent Hive 2.0."""

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

logger = structlog.get_logger()


class ModificationType(Enum):
    """Types of code modifications."""

    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_REFACTORING = "code_refactoring"
    DOCUMENTATION_UPDATE = "documentation_update"
    TEST_ADDITION = "test_addition"
    SECURITY_IMPROVEMENT = "security_improvement"


class ModificationStatus(Enum):
    """Status of a modification."""

    PROPOSED = "proposed"
    ANALYZING = "analyzing"
    TESTING = "testing"
    VALIDATED = "validated"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CodeChange:
    """Represents a single code change."""

    file_path: str
    original_content: str
    modified_content: str
    change_description: str
    line_numbers: list[int]


@dataclass
class ModificationProposal:
    """A proposed code modification."""

    id: str
    title: str
    description: str
    modification_type: ModificationType
    changes: list[CodeChange]
    test_commands: list[str]
    validation_criteria: list[str]
    risk_level: str  # "low", "medium", "high"
    estimated_impact: str
    created_at: float
    created_by: str


@dataclass
class TestResult:
    """Result of running tests."""

    command: str
    success: bool
    output: str
    error: str
    execution_time: float
    exit_code: int


@dataclass
class ValidationResult:
    """Result of validating a modification."""

    proposal_id: str
    tests_passed: bool
    performance_impact: dict[str, float]
    security_issues: list[str]
    code_quality_score: float
    overall_success: bool
    issues_found: list[str]
    recommendations: list[str]


@dataclass
class ModificationRecord:
    """Record of an applied modification."""

    proposal_id: str
    status: ModificationStatus
    git_branch: str
    git_commit_hash: str | None
    backup_commit_hash: str
    applied_at: float
    validation_result: ValidationResult
    rollback_reason: str | None = None
    rollback_at: float | None = None


class GitManager:
    """Manages Git operations for safe modifications."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.main_branch = "main"

    async def create_backup_branch(self, prefix: str = "backup") -> str:
        """Create a backup branch from current state."""
        timestamp = int(time.time())
        branch_name = f"{prefix}-{timestamp}"

        try:
            # Create and checkout backup branch
            await self._run_git_command(["checkout", "-b", branch_name])

            # Return to main branch
            await self._run_git_command(["checkout", self.main_branch])

            logger.info("Backup branch created", branch_name=branch_name)
            return branch_name

        except Exception as e:
            logger.error("Failed to create backup branch", error=str(e))
            raise

    async def create_feature_branch(self, proposal_id: str) -> str:
        """Create a feature branch for modification."""
        branch_name = f"feature/modification-{proposal_id}"

        try:
            # Ensure we're on main branch
            await self._run_git_command(["checkout", self.main_branch])

            # Only pull if we have a remote origin
            try:
                # Check if origin remote exists
                result = await self._run_git_command(["remote", "get-url", "origin"])
                # Pull latest changes
                await self._run_git_command(["pull", "origin", self.main_branch])
            except subprocess.CalledProcessError:
                # No remote origin, skip pull
                logger.info("No remote origin found, skipping pull")

            # Create feature branch
            await self._run_git_command(["checkout", "-b", branch_name])

            logger.info("Feature branch created", branch_name=branch_name)
            return branch_name

        except Exception as e:
            logger.error("Failed to create feature branch", error=str(e))
            raise

    async def commit_changes(self, message: str) -> str:
        """Commit current changes and return commit hash."""
        try:
            # Stage all changes
            await self._run_git_command(["add", "."])

            # Commit with message
            await self._run_git_command(["commit", "-m", message])

            # Get commit hash
            result = await self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = result.stdout.strip()

            logger.info("Changes committed", commit_hash=commit_hash[:8])
            return commit_hash

        except Exception as e:
            logger.error("Failed to commit changes", error=str(e))
            raise

    async def merge_to_main(self, feature_branch: str) -> str:
        """Merge feature branch to main."""
        try:
            # Switch to main
            await self._run_git_command(["checkout", self.main_branch])

            # Merge feature branch
            await self._run_git_command(["merge", feature_branch, "--no-ff"])

            # Get merge commit hash
            result = await self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = result.stdout.strip()

            # Delete feature branch
            await self._run_git_command(["branch", "-d", feature_branch])

            logger.info(
                "Feature branch merged",
                feature_branch=feature_branch,
                commit_hash=commit_hash[:8],
            )
            return commit_hash

        except Exception as e:
            logger.error("Failed to merge feature branch", error=str(e))
            raise

    async def rollback_to_commit(self, commit_hash: str) -> None:
        """Rollback to a specific commit."""
        try:
            # Hard reset to commit
            await self._run_git_command(["reset", "--hard", commit_hash])

            logger.info("Rolled back to commit", commit_hash=commit_hash[:8])

        except Exception as e:
            logger.error("Failed to rollback", error=str(e))
            raise

    async def get_current_commit(self) -> str:
        """Get current commit hash."""
        result = await self._run_git_command(["rev-parse", "HEAD"])
        return result.stdout.strip()

    async def _run_git_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.repo_path,
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


class SandboxTester:
    """Tests modifications in a sandboxed environment."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)

    async def run_tests(self, test_commands: list[str]) -> list[TestResult]:
        """Run a list of test commands."""
        results = []

        for command in test_commands:
            result = await self._run_test_command(command)
            results.append(result)

            # Stop on first failure for safety
            if not result.success:
                logger.warning("Test failed, stopping execution", command=command)
                break

        return results

    async def _run_test_command(self, command: str) -> TestResult:
        """Run a single test command."""
        start_time = time.time()

        try:
            # Parse command
            cmd_parts = command.split()

            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=self.workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=300,  # 5 minute timeout
            )

            stdout, stderr = await process.communicate()
            execution_time = time.time() - start_time

            result = TestResult(
                command=command,
                success=process.returncode == 0,
                output=stdout.decode(),
                error=stderr.decode(),
                execution_time=execution_time,
                exit_code=process.returncode,
            )

            logger.info(
                "Test command executed",
                command=command,
                success=result.success,
                execution_time=execution_time,
            )

            return result

        except TimeoutError:
            execution_time = time.time() - start_time
            return TestResult(
                command=command,
                success=False,
                output="",
                error="Test timed out after 5 minutes",
                execution_time=execution_time,
                exit_code=-1,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                command=command,
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                exit_code=-1,
            )

    async def run_performance_benchmarks(self) -> dict[str, float]:
        """Run performance benchmarks."""
        benchmarks = {}

        # Example benchmarks (would be customized per project)
        benchmark_commands = [
            (
                "startup_time",
                'python -c "import time; start=time.time(); import src; print(time.time()-start)"',
            ),
            (
                "memory_usage",
                'python -c "import psutil; print(psutil.Process().memory_info().rss / 1024 / 1024)"',
            ),
        ]

        for name, command in benchmark_commands:
            try:
                result = await self._run_test_command(command)
                if result.success:
                    # Extract numeric value from output
                    try:
                        value = float(result.output.strip())
                        benchmarks[name] = value
                    except ValueError:
                        benchmarks[name] = 0.0
                else:
                    benchmarks[name] = float("inf")  # Indicates failure
            except Exception:
                benchmarks[name] = float("inf")

        return benchmarks

    async def check_code_quality(self) -> float:
        """Check code quality using static analysis tools."""
        quality_score = 1.0  # Start with perfect score

        # Run code quality tools
        quality_commands = [
            "ruff check .",
            "mypy src/ --ignore-missing-imports",
            # "pylint src/" # Could add more tools
        ]

        for command in quality_commands:
            try:
                result = await self._run_test_command(command)
                if not result.success:
                    # Deduct points for quality issues
                    quality_score -= 0.2
            except Exception:
                quality_score -= 0.1

        return max(0.0, quality_score)


class SecurityScanner:
    """Scans code for security issues."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)

    async def scan_for_security_issues(self, changes: list[CodeChange]) -> list[str]:
        """Scan code changes for security issues."""
        issues = []

        for change in changes:
            # Check for common security anti-patterns
            content = change.modified_content.lower()

            # Check for hardcoded secrets
            if any(
                keyword in content
                for keyword in ["password", "secret", "key", "token", "api_key"]
            ):
                if any(
                    bad_pattern in content
                    for bad_pattern in ['= "', "= '", 'password="', 'secret="']
                ):
                    issues.append(f"Potential hardcoded secret in {change.file_path}")

            # Check for SQL injection vulnerabilities
            if any(
                keyword in content
                for keyword in ["sql", "select", "insert", "update", "delete"]
            ) and any(
                pattern in content
                for pattern in ["execute(", "query(", ".format(", "% "]
            ):
                issues.append(
                    f"Potential SQL injection vulnerability in {change.file_path}"
                )

            # Check for command injection
            if any(
                func in content
                for func in ["os.system", "subprocess.", "eval(", "exec("]
            ):
                issues.append(f"Potential command injection in {change.file_path}")

            # Check for insecure random usage
            if "random." in content and "import random" in content:
                issues.append(
                    f"Insecure random usage in {change.file_path} (use secrets module)"
                )

        return issues


class SelfModifier:
    """Main self-modification system with safety guarantees."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.git_manager = GitManager(workspace_path)
        self.sandbox_tester = SandboxTester(workspace_path)
        self.security_scanner = SecurityScanner(workspace_path)

        # Modification tracking
        self.modification_records: dict[str, ModificationRecord] = {}
        self.active_proposals: dict[str, ModificationProposal] = {}

        # Safety limits
        self.max_concurrent_modifications = 1
        self.max_modifications_per_hour = 5
        self.modification_history: list[float] = []

        logger.info("Self-modifier initialized", workspace_path=str(workspace_path))

    async def propose_modification(
        self,
        title: str,
        description: str,
        modification_type: ModificationType,
        changes: list[CodeChange],
        created_by: str,
    ) -> str:
        """Propose a code modification."""

        # Check rate limits
        if not self._check_rate_limits():
            raise RuntimeError("Modification rate limit exceeded")

        # Generate proposal ID
        proposal_id = str(uuid.uuid4())[:8]

        # Determine test commands based on project
        test_commands = self._get_default_test_commands()

        # Determine validation criteria
        validation_criteria = self._get_validation_criteria(modification_type)

        # Assess risk level
        risk_level = self._assess_risk_level(changes, modification_type)

        proposal = ModificationProposal(
            id=proposal_id,
            title=title,
            description=description,
            modification_type=modification_type,
            changes=changes,
            test_commands=test_commands,
            validation_criteria=validation_criteria,
            risk_level=risk_level,
            estimated_impact=self._estimate_impact(changes),
            created_at=time.time(),
            created_by=created_by,
        )

        self.active_proposals[proposal_id] = proposal

        logger.info(
            "Modification proposed",
            proposal_id=proposal_id,
            title=title,
            risk_level=risk_level,
            changes_count=len(changes),
        )

        return proposal_id

    async def validate_modification(self, proposal_id: str) -> ValidationResult:
        """Validate a proposed modification."""

        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.active_proposals[proposal_id]

        logger.info("Starting modification validation", proposal_id=proposal_id)

        # Create backup
        await self.git_manager.create_backup_branch(f"backup-{proposal_id}")
        backup_commit = await self.git_manager.get_current_commit()

        try:
            # Create feature branch
            feature_branch = await self.git_manager.create_feature_branch(proposal_id)

            # Apply changes
            await self._apply_changes(proposal.changes)

            # Commit changes
            commit_message = f"{proposal.modification_type.value}: {proposal.title}\n\n{proposal.description}"
            commit_hash = await self.git_manager.commit_changes(commit_message)

            # Run tests
            test_results = await self.sandbox_tester.run_tests(proposal.test_commands)
            tests_passed = all(result.success for result in test_results)

            # Check performance impact
            performance_impact = await self.sandbox_tester.run_performance_benchmarks()

            # Security scan
            security_issues = await self.security_scanner.scan_for_security_issues(
                proposal.changes
            )

            # Code quality check
            code_quality_score = await self.sandbox_tester.check_code_quality()

            # Overall validation
            issues_found = []
            if not tests_passed:
                issues_found.extend(
                    [f"Test failed: {r.command}" for r in test_results if not r.success]
                )

            if security_issues:
                issues_found.extend(security_issues)

            if code_quality_score < 0.7:
                issues_found.append(
                    f"Code quality below threshold: {code_quality_score:.2f}"
                )

            overall_success = (
                tests_passed
                and len(security_issues) == 0
                and code_quality_score >= 0.7
                and self._check_performance_impact(performance_impact)
            )

            validation_result = ValidationResult(
                proposal_id=proposal_id,
                tests_passed=tests_passed,
                performance_impact=performance_impact,
                security_issues=security_issues,
                code_quality_score=code_quality_score,
                overall_success=overall_success,
                issues_found=issues_found,
                recommendations=self._generate_recommendations(proposal, issues_found),
            )

            # Record validation
            record = ModificationRecord(
                proposal_id=proposal_id,
                status=ModificationStatus.VALIDATED
                if overall_success
                else ModificationStatus.FAILED,
                git_branch=feature_branch,
                git_commit_hash=commit_hash,
                backup_commit_hash=backup_commit,
                applied_at=time.time(),
                validation_result=validation_result,
            )

            self.modification_records[proposal_id] = record

            logger.info(
                "Modification validation completed",
                proposal_id=proposal_id,
                overall_success=overall_success,
                issues_count=len(issues_found),
            )

            return validation_result

        except Exception as e:
            logger.error("Validation failed", proposal_id=proposal_id, error=str(e))

            # Rollback to backup
            await self.git_manager.rollback_to_commit(backup_commit)

            # Create failed validation result
            validation_result = ValidationResult(
                proposal_id=proposal_id,
                tests_passed=False,
                performance_impact={},
                security_issues=[],
                code_quality_score=0.0,
                overall_success=False,
                issues_found=[f"Validation exception: {str(e)}"],
                recommendations=["Fix validation errors before retrying"],
            )

            return validation_result

    async def apply_modification(self, proposal_id: str) -> bool:
        """Apply a validated modification."""

        if proposal_id not in self.modification_records:
            raise ValueError(f"No validation record found for {proposal_id}")

        record = self.modification_records[proposal_id]

        if record.status != ModificationStatus.VALIDATED:
            raise ValueError(f"Modification {proposal_id} not validated")

        if not record.validation_result.overall_success:
            raise ValueError(f"Modification {proposal_id} failed validation")

        try:
            # Merge feature branch to main
            merge_commit = await self.git_manager.merge_to_main(record.git_branch)

            # Update record
            record.status = ModificationStatus.APPLIED
            record.git_commit_hash = merge_commit

            # Update rate limiting
            self.modification_history.append(time.time())

            # Clean up
            if proposal_id in self.active_proposals:
                del self.active_proposals[proposal_id]

            logger.info(
                "Modification applied successfully",
                proposal_id=proposal_id,
                commit_hash=merge_commit[:8],
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to apply modification", proposal_id=proposal_id, error=str(e)
            )

            # Mark as failed
            record.status = ModificationStatus.FAILED

            return False

    async def rollback_modification(self, proposal_id: str, reason: str) -> bool:
        """Rollback an applied modification."""

        if proposal_id not in self.modification_records:
            raise ValueError(f"No record found for {proposal_id}")

        record = self.modification_records[proposal_id]

        if record.status != ModificationStatus.APPLIED:
            raise ValueError(f"Modification {proposal_id} not applied")

        try:
            # Rollback to backup commit
            await self.git_manager.rollback_to_commit(record.backup_commit_hash)

            # Update record
            record.status = ModificationStatus.ROLLED_BACK
            record.rollback_reason = reason
            record.rollback_at = time.time()

            logger.info(
                "Modification rolled back", proposal_id=proposal_id, reason=reason
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to rollback modification", proposal_id=proposal_id, error=str(e)
            )
            return False

    async def _apply_changes(self, changes: list[CodeChange]) -> None:
        """Apply code changes to files."""
        for change in changes:
            file_path = self.workspace_path / change.file_path

            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write modified content
            file_path.write_text(change.modified_content)

            logger.debug("Applied change", file_path=str(file_path))

    def _get_default_test_commands(self) -> list[str]:
        """Get default test commands for the project."""
        return [
            "python -m pytest tests/ -q",
            "ruff check .",
            "python -c 'import src; print(\"Import successful\")'",
        ]

    def _get_validation_criteria(
        self, modification_type: ModificationType
    ) -> list[str]:
        """Get validation criteria based on modification type."""
        base_criteria = [
            "All existing tests pass",
            "No security vulnerabilities introduced",
            "Code quality standards maintained",
        ]

        if modification_type == ModificationType.PERFORMANCE_OPTIMIZATION:
            base_criteria.append("Performance improves or maintains current levels")

        elif modification_type == ModificationType.FEATURE_ADDITION:
            base_criteria.extend(
                [
                    "New functionality works as expected",
                    "No breaking changes to existing functionality",
                ]
            )

        elif modification_type == ModificationType.BUG_FIX:
            base_criteria.append("Bug is actually fixed")

        return base_criteria

    def _assess_risk_level(
        self, changes: list[CodeChange], modification_type: ModificationType
    ) -> str:
        """Assess the risk level of a modification."""

        # High risk modifications
        if modification_type in [ModificationType.SECURITY_IMPROVEMENT]:
            return "high"

        # Check for high-risk file patterns
        high_risk_patterns = ["core/", "models.py", "config.py", "__init__.py"]

        for change in changes:
            if any(pattern in change.file_path for pattern in high_risk_patterns):
                return "high"

        # Check change size
        total_lines_changed = sum(
            len(change.modified_content.split("\n")) for change in changes
        )

        if total_lines_changed > 100:
            return "medium"
        elif total_lines_changed > 20:
            return "low"
        else:
            return "low"

    def _estimate_impact(self, changes: list[CodeChange]) -> str:
        """Estimate the impact of changes."""
        total_files = len(changes)
        total_lines = sum(
            len(change.modified_content.split("\n")) for change in changes
        )

        if total_files > 5 or total_lines > 200:
            return "high"
        elif total_files > 2 or total_lines > 50:
            return "medium"
        else:
            return "low"

    def _check_performance_impact(self, performance_impact: dict[str, float]) -> bool:
        """Check if performance impact is acceptable."""
        # Simple check - no performance metric should be infinite (indicating failure)
        return all(value != float("inf") for value in performance_impact.values())

    def _generate_recommendations(
        self, proposal: ModificationProposal, issues: list[str]
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if issues:
            recommendations.append("Address all identified issues before applying")

        if proposal.risk_level == "high":
            recommendations.append(
                "Consider additional review for high-risk modification"
            )

        if len(proposal.changes) > 3:
            recommendations.append("Consider breaking into smaller modifications")

        return recommendations

    def _check_rate_limits(self) -> bool:
        """Check if modification rate limits are respected."""
        current_time = time.time()

        # Clean old entries
        self.modification_history = [
            t
            for t in self.modification_history
            if current_time - t < 3600  # Keep last hour
        ]

        # Check limits
        if len(self.modification_history) >= self.max_modifications_per_hour:
            return False

        # Check concurrent modifications
        active_count = sum(
            1
            for record in self.modification_records.values()
            if record.status
            in [ModificationStatus.ANALYZING, ModificationStatus.TESTING]
        )

        return active_count < self.max_concurrent_modifications

    async def propose_and_apply_change(
        self,
        file_path: str,
        change_description: str,
        agent_id: str = "unknown",
        modification_type: ModificationType = ModificationType.CODE_REFACTORING,
    ) -> dict[str, Any]:
        """
        Main entry point for MetaAgent to propose and apply code changes.

        This is the simplified workflow from PLAN.md:
        1. Read file content
        2. Generate new code using CLI tool (placeholder for now)
        3. Create feature branch
        4. Apply changes
        5. Run tests
        6. Commit if tests pass, rollback if they fail

        Returns dict with success status and details.
        """
        try:
            file_full_path = self.workspace_path / file_path

            # Validate file exists
            if not file_full_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "file_path": file_path,
                }

            logger.info(
                "Starting self-modification",
                file_path=file_path,
                description=change_description[:100],
                agent_id=agent_id,
            )

            # 1. Read current file content
            original_content = file_full_path.read_text()

            # 2. For now, create a simple placeholder modification
            # TODO: Use BaseAgent's execute_with_cli_tool to generate actual changes
            # This is a minimal implementation for bootstrap
            modified_content = (
                original_content + f"\n# Modified by {agent_id}: {change_description}\n"
            )

            # 3. Create a code change
            change = CodeChange(
                file_path=file_path,
                original_content=original_content,
                modified_content=modified_content,
                change_description=change_description,
                line_numbers=[len(original_content.split("\n")) + 1],
            )

            # 4. Create proposal
            proposal_id = await self.propose_modification(
                title=f"Modify {file_path}",
                description=change_description,
                modification_type=modification_type,
                changes=[change],
                created_by=agent_id,
            )

            # 5. Validate the modification (includes testing)
            validation_result = await self.validate_modification(proposal_id)

            if not validation_result.overall_success:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "proposal_id": proposal_id,
                    "validation_result": {
                        "tests_passed": validation_result.tests_passed,
                        "issues_found": validation_result.issues_found,
                        "recommendations": validation_result.recommendations,
                    },
                }

            # 6. Apply the modification
            apply_success = await self.apply_modification(proposal_id)

            if apply_success:
                return {
                    "success": True,
                    "message": "Modification applied successfully",
                    "proposal_id": proposal_id,
                    "file_path": file_path,
                    "change_description": change_description,
                    "validation_result": {
                        "tests_passed": validation_result.tests_passed,
                        "code_quality_score": validation_result.code_quality_score,
                    },
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to apply modification",
                    "proposal_id": proposal_id,
                }

        except Exception as e:
            logger.error(
                "propose_and_apply_change failed",
                file_path=file_path,
                error=str(e),
                agent_id=agent_id,
            )
            return {
                "success": False,
                "error": f"Exception during modification: {str(e)}",
                "file_path": file_path,
            }

    def get_modification_stats(self) -> dict[str, Any]:
        """Get modification statistics."""
        total_proposals = len(self.modification_records)

        if total_proposals == 0:
            return {
                "total_proposals": 0,
                "success_rate": 0.0,
                "by_status": {},
                "by_type": {},
                "average_validation_time": 0.0,
            }

        status_counts = {}
        type_counts = {}

        for record in self.modification_records.values():
            status_counts[record.status.value] = (
                status_counts.get(record.status.value, 0) + 1
            )

            if record.proposal_id in self.active_proposals:
                proposal = self.active_proposals[record.proposal_id]
                type_counts[proposal.modification_type.value] = (
                    type_counts.get(proposal.modification_type.value, 0) + 1
                )

        applied_count = status_counts.get(ModificationStatus.APPLIED.value, 0)
        success_rate = applied_count / total_proposals

        return {
            "total_proposals": total_proposals,
            "success_rate": success_rate,
            "by_status": status_counts,
            "by_type": type_counts,
            "rate_limit_remaining": self.max_modifications_per_hour
            - len(self.modification_history),
            "active_proposals": len(self.active_proposals),
        }


# Global instance
_self_modifier: SelfModifier | None = None


def get_self_modifier(workspace_path: str = None) -> SelfModifier:
    """Get or create the self-modifier singleton."""
    global _self_modifier

    if _self_modifier is None:
        if workspace_path is None:
            workspace_path = os.getcwd()
        _self_modifier = SelfModifier(workspace_path)

    return _self_modifier
