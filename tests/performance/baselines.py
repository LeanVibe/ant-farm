"""Performance baselines and regression detection for LeanVibe Agent Hive.

This module defines performance baselines, tracks historical performance data,
and provides regression detection capabilities.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    operation: str
    timestamp: datetime
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    requests_per_second: float
    error_rate: float
    concurrent_users: int
    environment: str
    commit_sha: Optional[str] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for an operation."""
    operation: str
    baseline_p95: float
    baseline_p99: float
    baseline_rps: float
    tolerance_percent: float
    last_updated: datetime
    sample_size: int


class PerformanceTracker:
    """Tracks performance metrics and detects regressions."""

    BASELINE_FILE = Path("tests/performance/baselines.json")
    HISTORY_FILE = Path("tests/performance/history.json")

    # Default baselines for user profile operations
    DEFAULT_BASELINES = {
        "GET /api/v1/auth/profile": PerformanceBaseline(
            operation="GET /api/v1/auth/profile",
            baseline_p95=200.0,  # ms
            baseline_p99=300.0,  # ms
            baseline_rps=100.0,  # requests/second
            tolerance_percent=10.0,  # 10% tolerance
            last_updated=datetime.now(timezone.utc),
            sample_size=1000,
        ),
        "PUT /api/v1/auth/profile": PerformanceBaseline(
            operation="PUT /api/v1/auth/profile",
            baseline_p95=500.0,  # ms
            baseline_p99=750.0,  # ms
            baseline_rps=50.0,   # requests/second
            tolerance_percent=15.0,  # 15% tolerance
            last_updated=datetime.now(timezone.utc),
            sample_size=500,
        ),
        "POST /api/v1/auth/users": PerformanceBaseline(
            operation="POST /api/v1/auth/users",
            baseline_p95=1000.0,  # ms
            baseline_p99=1500.0,  # ms
            baseline_rps=20.0,    # requests/second
            tolerance_percent=20.0,  # 20% tolerance
            last_updated=datetime.now(timezone.utc),
            sample_size=200,
        ),
        "DELETE /api/v1/auth/profile": PerformanceBaseline(
            operation="DELETE /api/v1/auth/profile",
            baseline_p95=2000.0,  # ms
            baseline_p99=3000.0,  # ms
            baseline_rps=10.0,    # requests/second
            tolerance_percent=25.0,  # 25% tolerance
            last_updated=datetime.now(timezone.utc),
            sample_size=100,
        ),
    }

    @classmethod
    def load_baselines(cls) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from file."""
        if not cls.BASELINE_FILE.exists():
            return cls.DEFAULT_BASELINES.copy()

        try:
            with open(cls.BASELINE_FILE, 'r') as f:
                data = json.load(f)

            baselines = {}
            for op, baseline_data in data.items():
                baseline_data['last_updated'] = datetime.fromisoformat(baseline_data['last_updated'])
                baselines[op] = PerformanceBaseline(**baseline_data)

            return baselines
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not load baselines: {e}")
            return cls.DEFAULT_BASELINES.copy()

    @classmethod
    def save_baselines(cls, baselines: Dict[str, PerformanceBaseline]) -> None:
        """Save performance baselines to file."""
        cls.BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for op, baseline in baselines.items():
            baseline_dict = asdict(baseline)
            baseline_dict['last_updated'] = baseline.last_updated.isoformat()
            data[op] = baseline_dict

        with open(cls.BASELINE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def record_measurement(cls, metric: PerformanceMetric) -> None:
        """Record a performance measurement."""
        cls.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        history = cls.load_history()

        # Add new measurement
        measurement_dict = asdict(metric)
        measurement_dict['timestamp'] = metric.timestamp.isoformat()
        if metric.commit_sha:
            measurement_dict['commit_sha'] = metric.commit_sha

        history.append(measurement_dict)

        # Keep only last 1000 measurements to prevent file from growing too large
        if len(history) > 1000:
            history = history[-1000:]

        # Save history
        with open(cls.HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    @classmethod
    def load_history(cls) -> List[Dict[str, Any]]:
        """Load performance measurement history."""
        if not cls.HISTORY_FILE.exists():
            return []

        try:
            with open(cls.HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    @classmethod
    def detect_regression(
        cls,
        operation: str,
        current_p95: float,
        current_p99: float,
        current_rps: float
    ) -> Dict[str, Any]:
        """Detect performance regression for an operation."""
        baselines = cls.load_baselines()

        if operation not in baselines:
            return {
                "regression_detected": False,
                "reason": f"No baseline found for operation: {operation}",
            }

        baseline = baselines[operation]

        # Check P95 regression
        p95_threshold = baseline.baseline_p95 * (1 + baseline.tolerance_percent / 100)
        p95_regression = current_p95 > p95_threshold

        # Check P99 regression
        p99_threshold = baseline.baseline_p99 * (1 + baseline.tolerance_percent / 100)
        p99_regression = current_p99 > p99_threshold

        # Check RPS regression (lower is worse)
        rps_threshold = baseline.baseline_rps * (1 - baseline.tolerance_percent / 100)
        rps_regression = current_rps < rps_threshold

        regression_detected = p95_regression or p99_regression or rps_regression

        result = {
            "regression_detected": regression_detected,
            "operation": operation,
            "current_p95": current_p95,
            "baseline_p95": baseline.baseline_p95,
            "p95_threshold": p95_threshold,
            "p95_regression": p95_regression,
            "current_p99": current_p99,
            "baseline_p99": baseline.baseline_p99,
            "p99_threshold": p99_threshold,
            "p99_regression": p99_regression,
            "current_rps": current_rps,
            "baseline_rps": baseline.baseline_rps,
            "rps_threshold": rps_threshold,
            "rps_regression": rps_regression,
            "tolerance_percent": baseline.tolerance_percent,
        }

        if regression_detected:
            reasons = []
            if p95_regression:
                reasons.append(".1f")
            if p99_regression:
                reasons.append(".1f")
            if rps_regression:
                reasons.append(".1f")
            result["reason"] = "Performance regression detected: " + ", ".join(reasons)

        return result

    @classmethod
    def update_baseline(
        cls,
        operation: str,
        new_p95: float,
        new_p99: float,
        new_rps: float,
        sample_size: int = 100
    ) -> None:
        """Update performance baseline for an operation."""
        baselines = cls.load_baselines()

        if operation not in baselines:
            baselines[operation] = PerformanceBaseline(
                operation=operation,
                baseline_p95=new_p95,
                baseline_p99=new_p99,
                baseline_rps=new_rps,
                tolerance_percent=10.0,  # Default 10% tolerance
                last_updated=datetime.now(timezone.utc),
                sample_size=sample_size,
            )
        else:
            baseline = baselines[operation]
            # Update baseline with weighted average
            weight = min(sample_size / (baseline.sample_size + sample_size), 0.3)  # Max 30% weight for new data

            baseline.baseline_p95 = baseline.baseline_p95 * (1 - weight) + new_p95 * weight
            baseline.baseline_p99 = baseline.baseline_p99 * (1 - weight) + new_p99 * weight
            baseline.baseline_rps = baseline.baseline_rps * (1 - weight) + new_rps * weight
            baseline.last_updated = datetime.now(timezone.utc)
            baseline.sample_size += sample_size

        cls.save_baselines(baselines)

    @classmethod
    def generate_performance_report(cls) -> str:
        """Generate a performance report."""
        baselines = cls.load_baselines()
        history = cls.load_history()

        report_lines = [
            "# Performance Report - LeanVibe Agent Hive",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Current Baselines",
            "",
        ]

        for operation, baseline in baselines.items():
            report_lines.extend([
                f"### {operation}",
                f"- **P95 Response Time**: {baseline.baseline_p95:.1f}ms",
                f"- **P99 Response Time**: {baseline.baseline_p99:.1f}ms",
                f"- **Target RPS**: {baseline.baseline_rps:.1f}",
                f"- **Tolerance**: {baseline.tolerance_percent:.1f}%",
                f"- **Sample Size**: {baseline.sample_size}",
                f"- **Last Updated**: {baseline.last_updated.isoformat()}",
                "",
            ])

        # Add recent performance history
        if history:
            report_lines.extend([
                "## Recent Performance History",
                "",
                "| Operation | Timestamp | P95 (ms) | P99 (ms) | RPS | Errors |",
                "|-----------|-----------|----------|----------|-----|--------|",
            ])

            # Show last 10 measurements
            for measurement in history[-10:]:
                report_lines.append(
                    f"| {measurement['operation']} | {measurement['timestamp'][:19]} | "
                    f"{measurement['response_time_p95']:.1f} | {measurement['response_time_p99']:.1f} | "
                    f"{measurement['requests_per_second']:.1f} | {measurement['error_rate']:.3f} |"
                )

            report_lines.append("")

        # Add regression analysis
        report_lines.extend([
            "## Regression Analysis",
            "",
            "Performance regression detection is active with the following thresholds:",
            "",
            "- P95/P99 response times can increase by up to the tolerance percentage",
            "- RPS can decrease by up to the tolerance percentage",
            "- Regressions are detected when measurements exceed these thresholds",
            "",
        ])

        return "\n".join(report_lines)


# Utility functions for CI/CD integration
def get_commit_sha() -> Optional[str]:
    """Get current commit SHA for performance tracking."""
    return os.environ.get("GITHUB_SHA")


def get_environment() -> str:
    """Get current environment for performance tracking."""
    return os.environ.get("ENVIRONMENT", "ci")


def should_update_baselines() -> bool:
    """Determine if baselines should be updated (e.g., on main branch)."""
    ref = os.environ.get("GITHUB_REF", "")
    return "main" in ref or "master" in ref


# Export key functions and classes
__all__ = [
    "PerformanceMetric",
    "PerformanceBaseline",
    "PerformanceTracker",
    "get_commit_sha",
    "get_environment",
    "should_update_baselines",
]