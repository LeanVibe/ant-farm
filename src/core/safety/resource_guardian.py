"""Resource exhaustion prevention for autonomous development sessions.

This module monitors and prevents resource depletion during extended
autonomous development workflows.
"""

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class ResourceThresholds:
    """Resource usage thresholds."""

    memory_warning: float = 0.7  # 70%
    memory_critical: float = 0.85  # 85%
    cpu_warning: float = 0.8  # 80%
    cpu_critical: float = 0.95  # 95%
    disk_warning: float = 0.8  # 80%
    disk_critical: float = 0.9  # 90%
    test_runtime_multiplier: float = 1.5  # 150% of baseline
    max_processes: int = 100


@dataclass
class ResourceStatus:
    """Current resource usage status."""

    memory_percent: float
    cpu_percent: float
    disk_percent: float
    process_count: int
    test_runtime_ratio: float
    timestamp: float
    warnings: list[str]
    critical_alerts: list[str]


class MemoryOptimizer:
    """Optimizes memory usage during development sessions."""

    def __init__(self):
        self.memory_baseline = psutil.virtual_memory().percent
        self.optimization_history: list[dict[str, Any]] = []

    async def optimize_memory_usage(self) -> dict[str, Any]:
        """Optimize system memory usage."""
        start_memory = psutil.virtual_memory().percent
        optimizations_applied = []

        try:
            # Force garbage collection
            import gc

            gc.collect()
            optimizations_applied.append("garbage_collection")

            # Clear Python import caches
            import sys

            if hasattr(sys, "_clear_type_cache"):
                sys._clear_type_cache()
                optimizations_applied.append("clear_type_cache")

            # Clear pytest cache if it exists
            pytest_cache_dir = Path.cwd() / ".pytest_cache"
            if pytest_cache_dir.exists():
                await self._safe_remove_directory(pytest_cache_dir)
                optimizations_applied.append("clear_pytest_cache")

            # Clear __pycache__ directories
            pycache_dirs = list(Path.cwd().rglob("__pycache__"))
            for cache_dir in pycache_dirs:
                await self._safe_remove_directory(cache_dir)
            if pycache_dirs:
                optimizations_applied.append(f"clear_pycache_{len(pycache_dirs)}_dirs")

            # Clear temporary files
            temp_files = list(Path.cwd().rglob("*.tmp")) + list(
                Path.cwd().rglob("*.temp")
            )
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(
                        "Failed to remove temp file", file=str(temp_file), error=str(e)
                    )
            if temp_files:
                optimizations_applied.append(f"clear_temp_files_{len(temp_files)}")

            end_memory = psutil.virtual_memory().percent
            memory_freed = start_memory - end_memory

            optimization_result = {
                "start_memory_percent": start_memory,
                "end_memory_percent": end_memory,
                "memory_freed_percent": memory_freed,
                "optimizations_applied": optimizations_applied,
                "timestamp": time.time(),
            }

            self.optimization_history.append(optimization_result)

            logger.info("Memory optimization completed", **optimization_result)
            return optimization_result

        except Exception as e:
            logger.error("Memory optimization failed", error=str(e))
            return {"error": str(e), "optimizations_applied": optimizations_applied}

    async def _safe_remove_directory(self, directory: Path) -> bool:
        """Safely remove a directory."""
        try:
            if directory.exists() and directory.is_dir():
                shutil.rmtree(directory)
                return True
        except Exception as e:
            logger.warning(
                "Failed to remove directory", directory=str(directory), error=str(e)
            )
        return False


class CPUThrottler:
    """Manages CPU usage during intensive operations."""

    def __init__(self):
        self.throttle_active = False
        self.throttle_history: list[dict[str, Any]] = []

    async def throttle_operations(
        self, target_cpu_percent: float = 70.0
    ) -> dict[str, Any]:
        """Throttle CPU-intensive operations."""
        if self.throttle_active:
            logger.info("CPU throttling already active")
            return {"status": "already_active"}

        try:
            self.throttle_active = True
            start_time = time.time()
            start_cpu = psutil.cpu_percent(interval=1)

            # Set lower priority for current process
            current_process = psutil.Process()
            original_priority = current_process.nice()

            if os.name == "nt":  # Windows
                current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:  # Unix-like
                current_process.nice(10)  # Lower priority

            # Introduce delays in intensive loops
            await asyncio.sleep(0.1)

            end_cpu = psutil.cpu_percent(interval=1)

            throttle_result = {
                "start_cpu_percent": start_cpu,
                "end_cpu_percent": end_cpu,
                "original_priority": original_priority,
                "new_priority": current_process.nice(),
                "duration": time.time() - start_time,
                "timestamp": time.time(),
            }

            self.throttle_history.append(throttle_result)
            logger.info("CPU throttling applied", **throttle_result)

            return throttle_result

        except Exception as e:
            logger.error("CPU throttling failed", error=str(e))
            return {"error": str(e)}
        finally:
            self.throttle_active = False

    async def release_throttle(self) -> dict[str, Any]:
        """Release CPU throttling."""
        try:
            current_process = psutil.Process()

            if os.name == "nt":  # Windows
                current_process.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:  # Unix-like
                current_process.nice(0)  # Normal priority

            self.throttle_active = False

            result = {
                "status": "released",
                "new_priority": current_process.nice(),
                "timestamp": time.time(),
            }

            logger.info("CPU throttling released", **result)
            return result

        except Exception as e:
            logger.error("Failed to release CPU throttling", error=str(e))
            return {"error": str(e)}


class DiskSpaceManager:
    """Manages disk space during development sessions."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.cleanup_history: list[dict[str, Any]] = []

    async def cleanup_temporary_files(self) -> dict[str, Any]:
        """Clean up temporary files and directories."""
        start_time = time.time()
        start_disk_usage = shutil.disk_usage(self.project_path)

        files_removed = 0
        directories_removed = 0
        space_freed = 0

        try:
            # Clean up patterns
            cleanup_patterns = [
                "*.pyc",
                "*.pyo",
                "*.tmp",
                "*.temp",
                "*.log",
                "*.bak",
                "*~",
                "*.swp",
                "*.swo",
            ]

            # Remove files matching patterns
            for pattern in cleanup_patterns:
                matching_files = list(self.project_path.rglob(pattern))
                for file_path in matching_files:
                    try:
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            space_freed += file_size
                    except Exception as e:
                        logger.warning(
                            "Failed to remove file during cleanup",
                            file=str(file_path),
                            error=str(e),
                        )

            # Remove empty directories
            for dir_path in self.project_path.rglob("*"):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        directories_removed += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to remove empty directory",
                            directory=str(dir_path),
                            error=str(e),
                        )

            # Clean up specific directories
            cleanup_dirs = [
                ".pytest_cache",
                "__pycache__",
                ".mypy_cache",
                ".ruff_cache",
                "node_modules/.cache",
                ".coverage",
            ]

            for cleanup_dir in cleanup_dirs:
                dir_path = self.project_path / cleanup_dir
                if dir_path.exists():
                    try:
                        if dir_path.is_file():
                            space_freed += dir_path.stat().st_size
                            dir_path.unlink()
                            files_removed += 1
                        else:
                            space_freed += sum(
                                f.stat().st_size
                                for f in dir_path.rglob("*")
                                if f.is_file()
                            )
                            shutil.rmtree(dir_path)
                            directories_removed += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to remove directory during cleanup",
                            directory=str(dir_path),
                            error=str(e),
                        )

            end_disk_usage = shutil.disk_usage(self.project_path)

            cleanup_result = {
                "files_removed": files_removed,
                "directories_removed": directories_removed,
                "space_freed_bytes": space_freed,
                "space_freed_mb": space_freed / (1024 * 1024),
                "start_free_space_gb": start_disk_usage.free / (1024**3),
                "end_free_space_gb": end_disk_usage.free / (1024**3),
                "duration": time.time() - start_time,
                "timestamp": time.time(),
            }

            self.cleanup_history.append(cleanup_result)
            logger.info("Disk cleanup completed", **cleanup_result)

            return cleanup_result

        except Exception as e:
            logger.error("Disk cleanup failed", error=str(e))
            return {"error": str(e)}


class TestSuiteOptimizer:
    """Optimizes test suite performance."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.baseline_runtime = None
        self.runtime_history: list[dict[str, Any]] = []

    async def measure_test_runtime(self) -> float:
        """Measure current test suite runtime."""
        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                "pytest",
                "tests/",
                "--tb=no",
                "-q",
                "--maxfail=1",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.communicate()
            runtime = time.time() - start_time

            self.runtime_history.append(
                {
                    "runtime": runtime,
                    "timestamp": time.time(),
                    "exit_code": process.returncode,
                }
            )

            return runtime

        except Exception as e:
            logger.error("Failed to measure test runtime", error=str(e))
            return 0.0

    async def optimize_test_suite(self) -> dict[str, Any]:
        """Optimize test suite performance."""
        try:
            optimizations_applied = []

            # Run tests in parallel if pytest-xdist is available
            try:
                import pytest_xdist

                optimizations_applied.append("parallel_execution_available")
            except ImportError:
                pass

            # Check for slow tests
            slow_test_threshold = 5.0  # seconds

            # This would require more sophisticated test analysis
            # For now, we'll just record the optimization attempt

            optimization_result = {
                "optimizations_applied": optimizations_applied,
                "baseline_runtime": self.baseline_runtime,
                "current_runtime": self.runtime_history[-1]["runtime"]
                if self.runtime_history
                else None,
                "timestamp": time.time(),
            }

            logger.info("Test suite optimization completed", **optimization_result)
            return optimization_result

        except Exception as e:
            logger.error("Test suite optimization failed", error=str(e))
            return {"error": str(e)}

    def set_baseline_runtime(self, runtime: float) -> None:
        """Set baseline test runtime."""
        self.baseline_runtime = runtime
        logger.info("Test runtime baseline set", baseline=runtime)

    def is_runtime_regression(
        self, current_runtime: float, threshold_multiplier: float = 1.5
    ) -> bool:
        """Check if current runtime represents a regression."""
        if self.baseline_runtime is None:
            return False

        return current_runtime > (self.baseline_runtime * threshold_multiplier)


class ResourceGuardian:
    """Main resource monitoring and management system."""

    def __init__(
        self, project_path: Path, thresholds: ResourceThresholds | None = None
    ):
        self.project_path = project_path
        self.thresholds = thresholds or ResourceThresholds()

        self.memory_optimizer = MemoryOptimizer()
        self.cpu_throttler = CPUThrottler()
        self.disk_manager = DiskSpaceManager(project_path)
        self.test_optimizer = TestSuiteOptimizer(project_path)

        self.monitoring_active = False
        self.status_history: list[ResourceStatus] = []

    async def get_current_status(self) -> ResourceStatus:
        """Get current resource usage status."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = shutil.disk_usage(self.project_path)
        process_count = len(psutil.pids())

        # Calculate test runtime ratio
        test_runtime_ratio = 1.0
        if self.test_optimizer.baseline_runtime and self.test_optimizer.runtime_history:
            current_runtime = self.test_optimizer.runtime_history[-1]["runtime"]
            test_runtime_ratio = current_runtime / self.test_optimizer.baseline_runtime

        disk_percent = (disk.total - disk.free) / disk.total

        warnings = []
        critical_alerts = []

        # Check thresholds
        if memory.percent > self.thresholds.memory_critical:
            critical_alerts.append(f"Memory usage critical: {memory.percent:.1f}%")
        elif memory.percent > self.thresholds.memory_warning:
            warnings.append(f"Memory usage high: {memory.percent:.1f}%")

        if cpu_percent > self.thresholds.cpu_critical:
            critical_alerts.append(f"CPU usage critical: {cpu_percent:.1f}%")
        elif cpu_percent > self.thresholds.cpu_warning:
            warnings.append(f"CPU usage high: {cpu_percent:.1f}%")

        if disk_percent > self.thresholds.disk_critical:
            critical_alerts.append(f"Disk usage critical: {disk_percent * 100:.1f}%")
        elif disk_percent > self.thresholds.disk_warning:
            warnings.append(f"Disk usage high: {disk_percent * 100:.1f}%")

        if process_count > self.thresholds.max_processes:
            warnings.append(f"High process count: {process_count}")

        if test_runtime_ratio > self.thresholds.test_runtime_multiplier:
            warnings.append(
                f"Test runtime regression: {test_runtime_ratio:.2f}x baseline"
            )

        status = ResourceStatus(
            memory_percent=memory.percent,
            cpu_percent=cpu_percent,
            disk_percent=disk_percent * 100,
            process_count=process_count,
            test_runtime_ratio=test_runtime_ratio,
            timestamp=time.time(),
            warnings=warnings,
            critical_alerts=critical_alerts,
        )

        self.status_history.append(status)

        # Keep only last 100 status records
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]

        return status

    async def handle_resource_issues(self, status: ResourceStatus) -> dict[str, Any]:
        """Handle resource issues based on current status."""
        actions_taken = []

        try:
            # Handle memory issues
            if status.memory_percent > self.thresholds.memory_critical:
                memory_result = await self.memory_optimizer.optimize_memory_usage()
                actions_taken.append(
                    f"memory_optimization: {memory_result.get('memory_freed_percent', 0):.1f}% freed"
                )

            # Handle CPU issues
            if status.cpu_percent > self.thresholds.cpu_critical:
                cpu_result = await self.cpu_throttler.throttle_operations()
                actions_taken.append(
                    f"cpu_throttling: {cpu_result.get('status', 'unknown')}"
                )

            # Handle disk space issues
            if status.disk_percent > self.thresholds.disk_critical:
                disk_result = await self.disk_manager.cleanup_temporary_files()
                actions_taken.append(
                    f"disk_cleanup: {disk_result.get('space_freed_mb', 0):.1f}MB freed"
                )

            # Handle test runtime issues
            if status.test_runtime_ratio > self.thresholds.test_runtime_multiplier:
                test_result = await self.test_optimizer.optimize_test_suite()
                actions_taken.append(
                    f"test_optimization: {len(test_result.get('optimizations_applied', []))} optimizations"
                )

            return {
                "actions_taken": actions_taken,
                "timestamp": time.time(),
                "success": True,
            }

        except Exception as e:
            logger.error("Failed to handle resource issues", error=str(e))
            return {
                "actions_taken": actions_taken,
                "error": str(e),
                "timestamp": time.time(),
                "success": False,
            }

    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous resource monitoring."""
        self.monitoring_active = True
        logger.info("Resource monitoring started", interval=interval)

        while self.monitoring_active:
            try:
                status = await self.get_current_status()

                if status.warnings or status.critical_alerts:
                    logger.warning(
                        "Resource issues detected",
                        warnings=status.warnings,
                        critical_alerts=status.critical_alerts,
                    )

                    # Handle issues automatically
                    if status.critical_alerts:
                        await self.handle_resource_issues(status)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error("Resource monitoring error", error=str(e))
                await asyncio.sleep(interval)

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        logger.info("Resource monitoring stopped")

    def get_resource_statistics(self) -> dict[str, Any]:
        """Get resource usage statistics."""
        if not self.status_history:
            return {"total_checks": 0}

        recent_status = (
            self.status_history[-10:]
            if len(self.status_history) > 10
            else self.status_history
        )

        avg_memory = sum(s.memory_percent for s in recent_status) / len(recent_status)
        avg_cpu = sum(s.cpu_percent for s in recent_status) / len(recent_status)
        avg_disk = sum(s.disk_percent for s in recent_status) / len(recent_status)

        warning_count = sum(len(s.warnings) for s in self.status_history)
        critical_count = sum(len(s.critical_alerts) for s in self.status_history)

        return {
            "total_checks": len(self.status_history),
            "recent_avg_memory_percent": avg_memory,
            "recent_avg_cpu_percent": avg_cpu,
            "recent_avg_disk_percent": avg_disk,
            "total_warnings": warning_count,
            "total_critical_alerts": critical_count,
            "memory_optimizations": len(self.memory_optimizer.optimization_history),
            "cpu_throttle_events": len(self.cpu_throttler.throttle_history),
            "disk_cleanups": len(self.disk_manager.cleanup_history),
            "latest_status": self.status_history[-1] if self.status_history else None,
        }
