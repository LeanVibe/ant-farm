"""Unit tests for SelfModifier system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.self_modifier import (
    CodeChange,
    ModificationType,
    SelfModifier,
    TestResult,
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
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=workspace_path)

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

    @pytest.mark.asyncio
    async def test_propose_modification_rate_limit(
        self, self_modifier, sample_code_change
    ):
        """Test rate limiting for modification proposals."""
        # Fill up the rate limit
        self_modifier.modification_history = [
            1234567890.0
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
        """Test the main propose_and_apply_change interface."""
        test_file = temp_workspace / "new_test.py"
        test_file.write_text("print('original')\n")

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
                modified_content="print('modified')\n",
            )

            assert result["success"] is True
            assert "proposal_id" in result
            assert test_file.read_text() == "print('modified')\n"

    @pytest.mark.asyncio
    async def test_propose_and_apply_change_no_content(self, self_modifier):
        """Test propose_and_apply_change without modified content."""
        result = await self_modifier.propose_and_apply_change(
            file_path="test.py", change_description="Test change"
        )

        assert result["success"] is False
        assert "Modified content must be provided" in result["error"]

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
                change_description="Breaking change",
                modified_content="invalid syntax here!\n",
            )

            assert result["success"] is False
            assert "Validation failed" in result["error"]

            # Check that we're back on the main branch after validation failure
            import subprocess

            git_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=temp_workspace,
                capture_output=True,
                text=True,
            )
            current_branch = git_result.stdout.strip()
            assert current_branch == "main"

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
    async def test_rollback_modification(
        self, self_modifier, sample_code_change, temp_workspace
    ):
        """Test rolling back an applied modification."""
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

            # Store original content
            test_file = temp_workspace / "test.py"
            original_content = test_file.read_text()

            # Apply modification
            result = await self_modifier.propose_and_apply_change(
                file_path="test.py",
                change_description="Test change for rollback",
                modified_content="print('This will be rolled back')\n",
            )

            assert result["success"] is True
            proposal_id = result["proposal_id"]

            # Verify change was applied
            assert test_file.read_text() == "print('This will be rolled back')\n"

            # Rollback the modification
            rollback_success = await self_modifier.rollback_modification(
                proposal_id, "Testing rollback functionality"
            )

            assert rollback_success is True
            # Verify content was restored
            assert test_file.read_text() == original_content

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
        import time

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

    @pytest.mark.asyncio
    async def test_get_modification_stats_with_data(
        self, self_modifier, sample_code_change
    ):
        """Test getting stats with actual modification data."""
        # Create a proposal to have some data
        await self_modifier.propose_modification(
            title="Test for stats",
            description="Creating data for stats test",
            modification_type=ModificationType.BUG_FIX,
            changes=[sample_code_change],
            created_by="test",
        )

        stats = self_modifier.get_modification_stats()

        assert stats["total_proposals"] == 0  # No records yet, only proposal
        assert stats["active_proposals"] == 1


class TestCodeChange:
    """Test cases for CodeChange dataclass."""

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


class TestModificationTypes:
    """Test cases for modification type enums."""

    def test_modification_type_values(self):
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


class TestSecurityScanner:
    """Test cases for SecurityScanner."""

    @pytest.mark.asyncio
    async def test_security_scanner_hardcoded_secrets(self, temp_workspace):
        """Test detection of hardcoded secrets."""
        from src.core.self_modifier import SecurityScanner

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
        from src.core.self_modifier import SecurityScanner

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
        from src.core.self_modifier import SecurityScanner

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
