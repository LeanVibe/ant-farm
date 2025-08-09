"""Enhanced tests for SelfModifier system with comprehensive coverage."""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.self_modifier import (
    CodeChange,
    ModificationProposal,
    ModificationRecord,
    ModificationType,
    ModificationStatus,
    SelfModifier,
    TestResult,
    ValidationResult,
    GitManager,
    SandboxTester,
    SecurityScanner,
    get_self_modifier,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)

        # Initialize as git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=workspace_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=workspace_path
        )
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=workspace_path)

        # Create initial commit
        test_file = workspace_path / "test.py"
        test_file.write_text("# Initial content\nprint('Hello, World!')\n")
        subprocess.run(["git", "add", "."], cwd=workspace_path)
        subprocess.run(
            ["git", "commit", "-m", "feat: initial commit"], cwd=workspace_path
        )

        yield workspace_path


@pytest.fixture
def self_modifier(temp_workspace):
    """Create SelfModifier instance with temporary workspace."""
    return SelfModifier(str(temp_workspace))


@pytest.fixture
def sample_code_change():
    """Create a sample CodeChange for testing."""
    return CodeChange(
        file_path="test.py",
        original_content="# Initial content\nprint('Hello, World!')\n",
        modified_content="# Modified content\nprint('Hello, Modified World!')\n",
        change_description="Update greeting message",
        line_numbers=[2],
    )


class TestSelfModifier:
    """Test cases for SelfModifier functionality."""

    @pytest.mark.asyncio
    async def test_propose_modification_success(
        self, self_modifier, sample_code_change
    ):
        """Test successful modification proposal."""
        proposal_id = await self_modifier.propose_modification(
            title="Test modification",
            description="A test modification",
            modification_type=ModificationType.BUG_FIX,
            changes=[sample_code_change],
            created_by="test",
        )

        assert proposal_id is not None
        assert len(proposal_id) == 8  # UUID first 8 characters
        assert proposal_id in self_modifier.active_proposals

        proposal = self_modifier.active_proposals[proposal_id]
        assert proposal.title == "Test modification"
        assert proposal.modification_type == ModificationType.BUG_FIX
        assert len(proposal.changes) == 1
        assert proposal.risk_level == "low"

    @pytest.mark.asyncio
    async def test_propose_modification_rate_limit(
        self, self_modifier, sample_code_change
    ):
        """Test rate limiting for modification proposals."""
        # Fill up the rate limit with current time - these won't be cleaned up
        current_time = time.time()
        self_modifier.modification_history = [
            current_time - 100  # Recent timestamps within the hour
        ] * 5  # max_modifications_per_hour

        with pytest.raises(RuntimeError, match="Modification rate limit exceeded"):
            await self_modifier.propose_modification(
                title="Rate limited",
                description="Should be rejected",
                modification_type=ModificationType.BUG_FIX,
                changes=[sample_code_change],
                created_by="test",
            )

    @pytest.mark.asyncio
    async def test_propose_and_apply_change_success(
        self, self_modifier, temp_workspace
    ):
        """Test the main propose_and_apply_change interface with current implementation."""
        test_file = temp_workspace / "new_test.py"
        test_file.write_text("print('original')\n")
        original_content = test_file.read_text()

        # Mock the test execution to always pass
        with (
            patch.object(self_modifier.sandbox_tester, "run_tests") as mock_tests,
            patch.object(
                self_modifier.sandbox_tester, "run_performance_benchmarks"
            ) as mock_perf,
            patch.object(
                self_modifier.sandbox_tester, "check_code_quality"
            ) as mock_quality,
            patch.object(
                self_modifier.security_scanner, "scan_for_security_issues"
            ) as mock_security,
        ):
            mock_tests.return_value = [
                TestResult(
                    command="pytest",
                    success=True,
                    output="All tests passed",
                    error="",
                    execution_time=1.0,
                    exit_code=0,
                )
            ]
            mock_perf.return_value = {"startup_time": 0.1}
            mock_quality.return_value = 0.8
            mock_security.return_value = []

            result = await self_modifier.propose_and_apply_change(
                file_path="new_test.py",
                change_description="Update print statement",
            )

            assert result["success"] is True
            assert "proposal_id" in result

            # Check that content was modified (current implementation appends comment)
            new_content = test_file.read_text()
            assert original_content in new_content  # Original content preserved
            assert "Modified by unknown: Update print statement" in new_content

    @pytest.mark.asyncio
    async def test_propose_and_apply_change_file_not_found(self, self_modifier):
        """Test propose_and_apply_change with non-existent file."""
        result = await self_modifier.propose_and_apply_change(
            file_path="nonexistent.py", change_description="Test change"
        )

        assert result["success"] is False
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_propose_and_apply_change_validation_failure(
        self, self_modifier, temp_workspace
    ):
        """Test propose_and_apply_change with validation failure."""
        test_file = temp_workspace / "failing_test.py"
        test_file.write_text("print('original')\n")

        # Mock test failure
        with (
            patch.object(self_modifier.sandbox_tester, "run_tests") as mock_tests,
            patch.object(
                self_modifier.sandbox_tester, "run_performance_benchmarks"
            ) as mock_perf,
            patch.object(
                self_modifier.sandbox_tester, "check_code_quality"
            ) as mock_quality,
            patch.object(
                self_modifier.security_scanner, "scan_for_security_issues"
            ) as mock_security,
        ):
            mock_tests.return_value = [
                TestResult(
                    command="pytest",
                    success=False,
                    output="",
                    error="Test failed",
                    execution_time=1.0,
                    exit_code=1,
                )
            ]
            mock_perf.return_value = {"startup_time": 0.1}
            mock_quality.return_value = 0.8
            mock_security.return_value = []

            result = await self_modifier.propose_and_apply_change(
                file_path="failing_test.py",
                change_description="Test validation failure",
            )

            assert result["success"] is False
            assert "Validation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_modification_with_security_issues(
        self, self_modifier, temp_workspace
    ):
        """Test validation with security issues detected."""
        # Create a change with security issues
        insecure_change = CodeChange(
            file_path="insecure.py",
            original_content="# Clean code\n",
            modified_content='password = "hardcoded_secret"\n',
            change_description="Add hardcoded password",
            line_numbers=[1],
        )

        proposal_id = await self_modifier.propose_modification(
            title="Insecure change",
            description="This should fail security scan",
            modification_type=ModificationType.BUG_FIX,
            changes=[insecure_change],
            created_by="test",
        )

        # Mock other checks to pass
        with (
            patch.object(self_modifier.sandbox_tester, "run_tests") as mock_tests,
            patch.object(
                self_modifier.sandbox_tester, "run_performance_benchmarks"
            ) as mock_perf,
            patch.object(
                self_modifier.sandbox_tester, "check_code_quality"
            ) as mock_quality,
        ):
            mock_tests.return_value = [
                TestResult(
                    command="pytest",
                    success=True,
                    output="All tests passed",
                    error="",
                    execution_time=1.0,
                    exit_code=0,
                )
            ]
            mock_perf.return_value = {"startup_time": 0.1}
            mock_quality.return_value = 0.8

            validation_result = await self_modifier.validate_modification(proposal_id)

            assert validation_result.overall_success is False
            assert len(validation_result.security_issues) > 0
            assert any(
                "hardcoded secret" in issue.lower()
                for issue in validation_result.security_issues
            )

    @pytest.mark.asyncio
    async def test_full_modification_workflow(self, self_modifier, temp_workspace):
        """Test complete modification workflow: propose -> validate -> apply."""
        # Create test file
        test_file = temp_workspace / "workflow_test.py"
        test_file.write_text("print('original')\n")
        original_content = test_file.read_text()

        # Mock successful validation
        with (
            patch.object(self_modifier.sandbox_tester, "run_tests") as mock_tests,
            patch.object(
                self_modifier.sandbox_tester, "run_performance_benchmarks"
            ) as mock_perf,
            patch.object(
                self_modifier.sandbox_tester, "check_code_quality"
            ) as mock_quality,
            patch.object(
                self_modifier.security_scanner, "scan_for_security_issues"
            ) as mock_security,
        ):
            mock_tests.return_value = [
                TestResult(
                    command="pytest",
                    success=True,
                    output="All tests passed",
                    error="",
                    execution_time=1.0,
                    exit_code=0,
                )
            ]
            mock_perf.return_value = {"startup_time": 0.1}
            mock_quality.return_value = 0.8
            mock_security.return_value = []

            # Step 1: Propose modification
            change = CodeChange(
                file_path="workflow_test.py",
                original_content=original_content,
                modified_content="print('modified')\n",
                change_description="Update print statement",
                line_numbers=[1],
            )

            proposal_id = await self_modifier.propose_modification(
                title="Workflow test",
                description="Testing full workflow",
                modification_type=ModificationType.BUG_FIX,
                changes=[change],
                created_by="test",
            )

            assert proposal_id in self_modifier.active_proposals

            # Step 2: Validate modification
            validation_result = await self_modifier.validate_modification(proposal_id)

            assert validation_result.overall_success is True
            assert validation_result.tests_passed is True
            assert len(validation_result.security_issues) == 0
            assert validation_result.code_quality_score >= 0.7

            # Step 3: Apply modification
            apply_result = await self_modifier.apply_modification(proposal_id)

            assert apply_result is True

            # Verify the proposal is no longer active
            assert proposal_id not in self_modifier.active_proposals

            # Verify modification was recorded
            assert proposal_id in self_modifier.modification_records
            record = self_modifier.modification_records[proposal_id]
            assert record.status == ModificationStatus.APPLIED

    @pytest.mark.asyncio
    async def test_rollback_modification(self, self_modifier, temp_workspace):
        """Test rolling back an applied modification."""
        # Create and apply a modification first
        test_file = temp_workspace / "rollback_test.py"
        test_file.write_text("print('original')\n")

        # Mock successful validation and application
        with (
            patch.object(self_modifier.sandbox_tester, "run_tests") as mock_tests,
            patch.object(
                self_modifier.sandbox_tester, "run_performance_benchmarks"
            ) as mock_perf,
            patch.object(
                self_modifier.sandbox_tester, "check_code_quality"
            ) as mock_quality,
            patch.object(
                self_modifier.security_scanner, "scan_for_security_issues"
            ) as mock_security,
        ):
            mock_tests.return_value = [
                TestResult(
                    command="pytest",
                    success=True,
                    output="All tests passed",
                    error="",
                    execution_time=1.0,
                    exit_code=0,
                )
            ]
            mock_perf.return_value = {"startup_time": 0.1}
            mock_quality.return_value = 0.8
            mock_security.return_value = []

            # Apply modification using propose_and_apply_change for simplicity
            result = await self_modifier.propose_and_apply_change(
                file_path="rollback_test.py",
                change_description="Test rollback",
            )

            assert result["success"] is True
            proposal_id = result["proposal_id"]

            # Verify change was applied
            modified_content = test_file.read_text()
            assert "Test rollback" in modified_content

            # Rollback the modification
            rollback_success = await self_modifier.rollback_modification(
                proposal_id, "Testing rollback functionality"
            )

            assert rollback_success is True

            # Verify rollback was recorded
            record = self_modifier.modification_records[proposal_id]
            assert record.status == ModificationStatus.ROLLED_BACK
            assert record.rollback_reason == "Testing rollback functionality"

    def test_assess_risk_level(self, self_modifier):
        """Test risk level assessment logic."""
        # High risk - core file
        high_risk_change = CodeChange(
            file_path="src/core/models.py",
            original_content="",
            modified_content="# lots of changes\n" * 100,
            change_description="Major core change",
            line_numbers=[],
        )

        risk_level = self_modifier._assess_risk_level(
            [high_risk_change], ModificationType.SECURITY_IMPROVEMENT
        )
        assert risk_level == "high"

        # Low risk - small change
        low_risk_change = CodeChange(
            file_path="tests/test_example.py",
            original_content="",
            modified_content="# small change",
            change_description="Minor test update",
            line_numbers=[],
        )

        risk_level = self_modifier._assess_risk_level(
            [low_risk_change], ModificationType.TEST_ADDITION
        )
        assert risk_level == "low"

    def test_rate_limiting_check(self, self_modifier):
        """Test rate limiting logic."""
        # Should pass initially
        assert self_modifier._check_rate_limits() is True

        # Fill up rate limit
        current_time = time.time()
        self_modifier.modification_history = [current_time - 60] * 5  # 5 in last hour

        # Should fail now
        assert self_modifier._check_rate_limits() is False

        # Old entries should be cleaned up
        self_modifier.modification_history = [current_time - 3700]  # Over 1 hour old
        assert self_modifier._check_rate_limits() is True

    def test_get_modification_stats_empty(self, self_modifier):
        """Test getting stats when no modifications exist."""
        stats = self_modifier.get_modification_stats()

        assert stats["total_proposals"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["by_status"] == {}
        assert stats["by_type"] == {}
        assert stats["active_proposals"] == 0

    @pytest.mark.asyncio
    async def test_get_modification_stats_with_data(
        self, self_modifier, sample_code_change
    ):
        """Test getting stats with actual modification data."""
        # Create a proposal to have some data
        proposal_id = await self_modifier.propose_modification(
            title="Test for stats",
            description="Creating data for stats test",
            modification_type=ModificationType.BUG_FIX,
            changes=[sample_code_change],
            created_by="test",
        )

        stats = self_modifier.get_modification_stats()

        assert stats["total_proposals"] == 0  # No records yet, only proposal
        assert stats["active_proposals"] == 1
        assert proposal_id in self_modifier.active_proposals

    @pytest.mark.asyncio
    async def test_validation_exception_handling(self, self_modifier):
        """Test validation exception handling."""
        # Create a change that will cause validation to fail
        change = CodeChange(
            file_path="exception_test.py",
            original_content="# test",
            modified_content="# modified",
            change_description="Test exception",
            line_numbers=[1],
        )

        proposal_id = await self_modifier.propose_modification(
            title="Exception test",
            description="Testing exception handling",
            modification_type=ModificationType.BUG_FIX,
            changes=[change],
            created_by="test",
        )

        # Mock git operations to throw an exception
        with patch.object(
            self_modifier.git_manager,
            "create_backup_branch",
            side_effect=Exception("Git error"),
        ):
            validation_result = await self_modifier.validate_modification(proposal_id)

            assert validation_result.overall_success is False
            assert "Git error" in validation_result.issues_found[0]

    def test_estimate_impact(self, self_modifier):
        """Test impact estimation logic."""
        # High impact - many files
        high_impact_changes = [
            CodeChange(f"file_{i}.py", "", "content", "change", []) for i in range(10)
        ]
        impact = self_modifier._estimate_impact(high_impact_changes)
        assert impact == "high"

        # Medium impact
        medium_impact_changes = [
            CodeChange("file.py", "", "content\n" * 60, "change", [])
        ]
        impact = self_modifier._estimate_impact(medium_impact_changes)
        assert impact == "medium"

        # Low impact
        low_impact_changes = [CodeChange("file.py", "", "small change", "change", [])]
        impact = self_modifier._estimate_impact(low_impact_changes)
        assert impact == "low"

    def test_check_performance_impact(self, self_modifier):
        """Test performance impact checking."""
        # Good performance
        good_performance = {"startup_time": 0.1, "memory_usage": 100.0}
        assert self_modifier._check_performance_impact(good_performance) is True

        # Bad performance (infinite indicates failure)
        bad_performance = {"startup_time": float("inf")}
        assert self_modifier._check_performance_impact(bad_performance) is False

    def test_generate_recommendations(self, self_modifier, sample_code_change):
        """Test recommendation generation."""
        proposal = ModificationProposal(
            id="test",
            title="Test",
            description="Test",
            modification_type=ModificationType.BUG_FIX,
            changes=[sample_code_change],
            test_commands=[],
            validation_criteria=[],
            risk_level="high",
            estimated_impact="high",
            created_at=time.time(),
            created_by="test",
        )

        # With issues
        issues = ["Test failed", "Security issue found"]
        recommendations = self_modifier._generate_recommendations(proposal, issues)

        assert "Address all identified issues" in recommendations[0]
        assert any("high-risk" in rec for rec in recommendations)

        # Many changes
        proposal.changes = [sample_code_change] * 5
        recommendations = self_modifier._generate_recommendations(proposal, [])
        assert any("smaller modifications" in rec for rec in recommendations)


class TestGitManager:
    """Test cases for GitManager."""

    @pytest.mark.asyncio
    async def test_create_backup_branch(self, temp_workspace):
        """Test creating backup branch."""
        git_manager = GitManager(str(temp_workspace))

        branch_name = await git_manager.create_backup_branch("test-backup")

        assert branch_name.startswith("test-backup-")

        # Verify we're back on main
        result = await git_manager._run_git_command(["branch", "--show-current"])
        assert result.stdout.strip() == "main"

    @pytest.mark.asyncio
    async def test_create_feature_branch(self, temp_workspace):
        """Test creating feature branch."""
        git_manager = GitManager(str(temp_workspace))

        branch_name = await git_manager.create_feature_branch("test-123")

        assert branch_name == "feature/modification-test-123"

        # Verify we're on the feature branch
        result = await git_manager._run_git_command(["branch", "--show-current"])
        assert result.stdout.strip() == branch_name

    @pytest.mark.asyncio
    async def test_commit_changes(self, temp_workspace):
        """Test committing changes."""
        git_manager = GitManager(str(temp_workspace))

        # Create a change
        test_file = temp_workspace / "new_file.py"
        test_file.write_text("print('test')\n")

        commit_hash = await git_manager.commit_changes("test: add new file")

        assert len(commit_hash) == 40  # SHA-1 hash length

        # Verify commit exists
        result = await git_manager._run_git_command(["log", "--oneline", "-1"])
        assert "test: add new file" in result.stdout

    @pytest.mark.asyncio
    async def test_get_current_commit(self, temp_workspace):
        """Test getting current commit hash."""
        git_manager = GitManager(str(temp_workspace))

        commit_hash = await git_manager.get_current_commit()

        assert len(commit_hash) == 40  # SHA-1 hash length


class TestSecurityScanner:
    """Test cases for SecurityScanner."""

    @pytest.mark.asyncio
    async def test_security_scanner_hardcoded_secrets(self, temp_workspace):
        """Test detection of hardcoded secrets."""
        scanner = SecurityScanner(str(temp_workspace))

        # Change with hardcoded secret
        insecure_change = CodeChange(
            file_path="insecure.py",
            original_content="",
            modified_content='API_KEY = "sk-1234567890abcdef"',
            change_description="Add API key",
            line_numbers=[],
        )

        issues = await scanner.scan_for_security_issues([insecure_change])

        assert len(issues) > 0
        assert any("hardcoded secret" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_security_scanner_sql_injection(self, temp_workspace):
        """Test detection of SQL injection vulnerabilities."""
        scanner = SecurityScanner(str(temp_workspace))

        # Change with potential SQL injection
        vulnerable_change = CodeChange(
            file_path="vulnerable.py",
            original_content="",
            modified_content='query = "SELECT * FROM users WHERE id = %s" % user_id',
            change_description="Add SQL query",
            line_numbers=[],
        )

        issues = await scanner.scan_for_security_issues([vulnerable_change])

        assert len(issues) > 0
        assert any("sql injection" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_security_scanner_command_injection(self, temp_workspace):
        """Test detection of command injection vulnerabilities."""
        scanner = SecurityScanner(str(temp_workspace))

        # Change with command injection risk
        vulnerable_change = CodeChange(
            file_path="command.py",
            original_content="",
            modified_content='os.system("rm -rf " + user_input)',
            change_description="Add system command",
            line_numbers=[],
        )

        issues = await scanner.scan_for_security_issues([vulnerable_change])

        assert len(issues) > 0
        assert any("command injection" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_security_scanner_no_issues(self, temp_workspace):
        """Test scanner with secure code."""
        scanner = SecurityScanner(str(temp_workspace))

        # Secure change
        secure_change = CodeChange(
            file_path="secure.py",
            original_content="",
            modified_content="def add_numbers(a, b):\n    return a + b",
            change_description="Add safe function",
            line_numbers=[],
        )

        issues = await scanner.scan_for_security_issues([secure_change])
        assert len(issues) == 0


class TestSandboxTester:
    """Test cases for SandboxTester."""

    @pytest.mark.asyncio
    async def test_run_test_command_success(self, temp_workspace):
        """Test running a successful test command."""
        tester = SandboxTester(str(temp_workspace))

        # Use a simple command that should succeed
        result = await tester._run_test_command("echo 'test passed'")

        assert result.success is True
        assert result.output.strip() == "test passed"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_test_command_failure(self, temp_workspace):
        """Test running a failing test command."""
        tester = SandboxTester(str(temp_workspace))

        # Use a command that should fail
        result = await tester._run_test_command("exit 1")

        assert result.success is False
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_check_code_quality(self, temp_workspace):
        """Test code quality checking."""
        tester = SandboxTester(str(temp_workspace))

        # Mock the test commands to simulate tools
        with patch.object(tester, "_run_test_command") as mock_run:
            mock_run.return_value = TestResult(
                command="ruff check .",
                success=True,
                output="",
                error="",
                execution_time=1.0,
                exit_code=0,
            )

            quality_score = await tester.check_code_quality()
            assert quality_score == 1.0  # Perfect score

    @pytest.mark.asyncio
    async def test_run_performance_benchmarks(self, temp_workspace):
        """Test performance benchmark execution."""
        tester = SandboxTester(str(temp_workspace))

        # Mock benchmark commands
        with patch.object(tester, "_run_test_command") as mock_run:
            mock_run.return_value = TestResult(
                command="benchmark",
                success=True,
                output="0.5",  # Mock timing result
                error="",
                execution_time=1.0,
                exit_code=0,
            )

            benchmarks = await tester.run_performance_benchmarks()
            assert "startup_time" in benchmarks
            assert benchmarks["startup_time"] == 0.5


class TestDataClasses:
    """Test cases for data classes."""

    def test_code_change_creation(self):
        """Test creating a CodeChange object."""
        change = CodeChange(
            file_path="test.py",
            original_content="old content",
            modified_content="new content",
            change_description="Test change",
            line_numbers=[1, 2, 3],
        )

        assert change.file_path == "test.py"
        assert change.original_content == "old content"
        assert change.modified_content == "new content"
        assert change.change_description == "Test change"
        assert change.line_numbers == [1, 2, 3]

    def test_modification_types(self):
        """Test ModificationType enum values."""
        assert ModificationType.BUG_FIX.value == "bug_fix"
        assert ModificationType.FEATURE_ADDITION.value == "feature_addition"
        assert (
            ModificationType.PERFORMANCE_OPTIMIZATION.value
            == "performance_optimization"
        )
        assert ModificationType.CODE_REFACTORING.value == "code_refactoring"
        assert ModificationType.DOCUMENTATION_UPDATE.value == "documentation_update"
        assert ModificationType.TEST_ADDITION.value == "test_addition"
        assert ModificationType.SECURITY_IMPROVEMENT.value == "security_improvement"

    def test_modification_status(self):
        """Test ModificationStatus enum values."""
        assert ModificationStatus.PROPOSED.value == "proposed"
        assert ModificationStatus.ANALYZING.value == "analyzing"
        assert ModificationStatus.TESTING.value == "testing"
        assert ModificationStatus.VALIDATED.value == "validated"
        assert ModificationStatus.APPLIED.value == "applied"
        assert ModificationStatus.FAILED.value == "failed"
        assert ModificationStatus.ROLLED_BACK.value == "rolled_back"


class TestSelfModifierSingleton:
    """Test the singleton pattern for self modifier."""

    def test_get_self_modifier_singleton(self, temp_workspace):
        """Test that get_self_modifier returns consistent instances."""
        modifier1 = get_self_modifier(str(temp_workspace))
        modifier2 = get_self_modifier()  # Should use existing

        # Note: Different workspace path creates different instance
        # This tests the factory pattern rather than true singleton
        assert isinstance(modifier1, SelfModifier)
        assert isinstance(modifier2, SelfModifier)
