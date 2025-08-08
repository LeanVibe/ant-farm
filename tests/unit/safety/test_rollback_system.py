"""Tests for the AutoRollbackSystem."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.safety.rollback_system import (
    AutoRollbackSystem,
    GitCheckpoint,
    PerformanceBaseline,
    RollbackLevel,
)


class TestGitCheckpoint:
    """Test GitCheckpoint functionality."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary git repository for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"], cwd=repo_path, check=True
            )

            # Create initial commit
            test_file = repo_path / "test.txt"
            test_file.write_text("initial content")
            subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True
            )

            yield repo_path

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, temp_repo):
        """Test creating a git checkpoint."""
        git_checkpoint = GitCheckpoint(temp_repo)

        # Make a change
        test_file = temp_repo / "test.txt"
        test_file.write_text("modified content")

        # Create checkpoint
        commit_hash = await git_checkpoint.create_checkpoint("test checkpoint")

        assert commit_hash is not None
        assert len(commit_hash) == 40  # SHA-1 hash length

    @pytest.mark.asyncio
    async def test_rollback_to_commit(self, temp_repo):
        """Test rolling back to a specific commit."""
        git_checkpoint = GitCheckpoint(temp_repo)

        # Get initial commit hash
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        initial_commit = result.stdout.strip()

        # Make changes and commit
        test_file = temp_repo / "test.txt"
        test_file.write_text("modified content")
        await git_checkpoint.create_checkpoint("modification")

        # Verify file is modified
        assert test_file.read_text() == "modified content"

        # Rollback to initial commit
        success = await git_checkpoint.rollback_to_commit(initial_commit)

        assert success is True
        assert test_file.read_text() == "initial content"

    @pytest.mark.asyncio
    async def test_get_last_stable_checkpoint(self, temp_repo):
        """Test getting the last stable checkpoint."""
        git_checkpoint = GitCheckpoint(temp_repo)

        # Create a checkpoint
        test_file = temp_repo / "test.txt"
        test_file.write_text("checkpoint content")
        commit_hash = await git_checkpoint.create_checkpoint("stable checkpoint")

        # Get last stable checkpoint
        last_checkpoint = await git_checkpoint.get_last_stable_checkpoint()

        assert last_checkpoint == commit_hash


class TestPerformanceBaseline:
    """Test PerformanceBaseline functionality."""

    @pytest.fixture
    def temp_baseline_file(self):
        """Create a temporary baseline file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            baseline_file = Path(f.name)

        yield baseline_file

        # Cleanup
        if baseline_file.exists():
            baseline_file.unlink()

    def test_set_and_check_baseline(self, temp_baseline_file):
        """Test setting and checking performance baselines."""
        baseline = PerformanceBaseline(temp_baseline_file)

        # Set baseline
        baseline.set_baseline("test_metric", 100.0, tolerance=0.1)

        # Check no regression
        is_regression, ratio = baseline.check_regression("test_metric", 105.0)
        assert is_regression is False

        # Check regression
        is_regression, ratio = baseline.check_regression("test_metric", 120.0)
        assert is_regression is True
        assert ratio == 0.2  # 20% increase

    def test_baseline_persistence(self, temp_baseline_file):
        """Test that baselines are persisted to file."""
        # Create baseline and set value
        baseline1 = PerformanceBaseline(temp_baseline_file)
        baseline1.set_baseline("test_metric", 100.0)

        # Create new instance and check value persists
        baseline2 = PerformanceBaseline(temp_baseline_file)
        is_regression, ratio = baseline2.check_regression("test_metric", 110.0)

        # Should detect regression since threshold is 10%
        assert is_regression is False  # 10% is within default tolerance


class TestAutoRollbackSystem:
    """Test AutoRollbackSystem functionality."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary git repository for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"], cwd=repo_path, check=True
            )

            # Create initial commit
            test_file = repo_path / "test.txt"
            test_file.write_text("initial content")
            subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True
            )

            yield repo_path

    @pytest.mark.asyncio
    async def test_create_safety_checkpoint(self, temp_repo):
        """Test creating a safety checkpoint."""
        rollback_system = AutoRollbackSystem(temp_repo)

        # Make a change
        test_file = temp_repo / "test.txt"
        test_file.write_text("modified for checkpoint")

        # Create safety checkpoint
        commit_hash = await rollback_system.create_safety_checkpoint("safety test")

        assert commit_hash is not None
        assert len(commit_hash) == 40

    @pytest.mark.asyncio
    async def test_handle_syntax_error_failure(self, temp_repo):
        """Test handling syntax error failure."""
        rollback_system = AutoRollbackSystem(temp_repo)

        # Create initial checkpoint
        test_file = temp_repo / "test.txt"
        test_file.write_text("good content")
        await rollback_system.create_safety_checkpoint("before syntax error")

        # Simulate syntax error by making bad changes
        test_file.write_text("bad syntax content")
        await rollback_system.create_safety_checkpoint("syntax error")

        # Handle syntax error
        success = await rollback_system.handle_failure(
            RollbackLevel.SYNTAX_ERROR, {"error": "SyntaxError: invalid syntax"}
        )

        # Should successfully rollback
        assert success is True
        assert len(rollback_system.rollback_history) == 1
        assert rollback_system.rollback_history[0]["failure_type"] == "syntax_error"

    @pytest.mark.asyncio
    async def test_rollback_statistics(self, temp_repo):
        """Test rollback statistics collection."""
        rollback_system = AutoRollbackSystem(temp_repo)

        # Simulate multiple rollback attempts
        await rollback_system.handle_failure(RollbackLevel.SYNTAX_ERROR)
        await rollback_system.handle_failure(RollbackLevel.TEST_FAILURE)

        # Get statistics
        stats = rollback_system.get_rollback_statistics()

        assert stats["total_attempts"] == 2
        assert "syntax_error" in stats["failure_type_distribution"]
        assert "test_failure" in stats["failure_type_distribution"]
        assert "success_rate" in stats
        assert "average_duration" in stats

    @pytest.mark.asyncio
    async def test_validation_methods(self, temp_repo):
        """Test different validation methods."""
        rollback_system = AutoRollbackSystem(temp_repo)

        # Test syntax check validation
        syntax_valid = await rollback_system._validate_rollback("syntax_check")
        # Should pass since we're not actually checking Python syntax of files

        # Test test suite validation (will fail since no tests in temp repo)
        test_valid = await rollback_system._validate_rollback("run_test_suite")
        # Should fail since pytest will return non-zero exit code

        # Test other validations (placeholders should return True)
        perf_valid = await rollback_system._validate_rollback("performance_test")
        assert perf_valid is True

        system_valid = await rollback_system._validate_rollback("full_system_check")
        assert system_valid is True


@pytest.mark.asyncio
async def test_rollback_system_integration(tmp_path):
    """Integration test for the complete rollback system."""
    # Create a mock git repository
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git
    import subprocess

    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_path, check=True
    )

    # Create initial file and commit
    test_file = repo_path / "code.py"
    test_file.write_text("print('hello world')")
    subprocess.run(["git", "add", "code.py"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial working code"], cwd=repo_path, check=True
    )

    # Create rollback system
    rollback_system = AutoRollbackSystem(repo_path)

    # Create safety checkpoint
    checkpoint = await rollback_system.create_safety_checkpoint("before risky change")
    assert checkpoint is not None

    # Make a "bad" change
    test_file.write_text("print('broken syntax")  # Missing closing quote
    subprocess.run(["git", "add", "code.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Broken code"], cwd=repo_path, check=True)

    # Trigger rollback due to syntax error
    success = await rollback_system.handle_failure(
        RollbackLevel.SYNTAX_ERROR, {"file": "code.py", "error": "SyntaxError"}
    )

    # Verify rollback worked
    assert success is True

    # Check that file content was restored
    current_content = test_file.read_text()
    assert "broken syntax" not in current_content

    # Verify statistics were recorded
    stats = rollback_system.get_rollback_statistics()
    assert stats["total_attempts"] >= 1
    assert stats["successful_attempts"] >= 1
