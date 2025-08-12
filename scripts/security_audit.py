#!/usr/bin/env python3
"""Security audit script to identify unprotected API endpoints."""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class SecurityAuditor:
    """Audits API endpoints for security compliance."""

    def __init__(self, api_file: Path):
        self.api_file = api_file
        self.unprotected_endpoints: list[dict] = []
        self.protected_endpoints: list[dict] = []
        self.public_endpoints: set[str] = {
            "/health",
            "/api/v1/health",
            "/api/v1/test",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
        }

    def audit_endpoints(self) -> dict:
        """Audit all endpoints and categorize by security status."""
        content = self.api_file.read_text()

        # Find all endpoint definitions using regex
        endpoint_pattern = r'@app\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']'
        endpoints = re.findall(endpoint_pattern, content)

        for method, path in endpoints:
            self._analyze_endpoint(content, method.upper(), path)

        return {
            "total_endpoints": len(endpoints),
            "protected_endpoints": len(self.protected_endpoints),
            "unprotected_endpoints": len(self.unprotected_endpoints),
            "unprotected_details": self.unprotected_endpoints,
            "summary": self._generate_summary(),
        }

    def _analyze_endpoint(self, content: str, method: str, path: str):
        """Analyze a specific endpoint for security decorators."""
        # Skip public endpoints
        if path in self.public_endpoints:
            return

        # Find the function definition for this endpoint
        pattern = (
            rf'@app\.{method.lower()}\(\s*["\']({re.escape(path)})["\'].*?\ndef\s+(\w+)'
        )
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            return

        func_name = match.group(2)

        # Look for protection patterns before the function
        func_start = match.start()

        # Look backwards for decorators up to 20 lines
        lines_before = content[:func_start].split("\n")[-20:]
        decorators_text = "\n".join(lines_before)

        # Check for authentication/authorization decorators
        has_auth = any(
            pattern in decorators_text
            for pattern in [
                "@Permissions.",
                "Depends(get_current_user)",
                "Depends(get_cli_user)",
                "@require_admin",
                "@require_permissions",
            ]
        )

        endpoint_info = {
            "method": method,
            "path": path,
            "function": func_name,
            "has_auth": has_auth,
            "severity": self._assess_severity(method, path),
        }

        if has_auth:
            self.protected_endpoints.append(endpoint_info)
        else:
            self.unprotected_endpoints.append(endpoint_info)

    def _assess_severity(self, method: str, path: str) -> str:
        """Assess the security severity of an unprotected endpoint."""
        # Critical: Write operations that modify system state
        if method in ["POST", "PUT", "DELETE", "PATCH"]:
            if any(
                critical_path in path
                for critical_path in [
                    "/agents",
                    "/system/",
                    "/modifications",
                    "/tasks",
                    "/broadcast",
                ]
            ):
                return "CRITICAL"

        # High: Administrative or sensitive read operations
        if any(
            sensitive_path in path
            for sensitive_path in [
                "/metrics",
                "/context",
                "/workflows",
                "/adw/",
                "/collaboration",
                "/diagnostics",
            ]
        ):
            return "HIGH"

        # Medium: General read operations
        if method == "GET":
            return "MEDIUM"

        return "LOW"

    def _generate_summary(self) -> dict:
        """Generate security audit summary."""
        severity_counts = {}
        for endpoint in self.unprotected_endpoints:
            severity = endpoint["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "critical_vulnerabilities": severity_counts.get("CRITICAL", 0),
            "high_risk_endpoints": severity_counts.get("HIGH", 0),
            "medium_risk_endpoints": severity_counts.get("MEDIUM", 0),
            "low_risk_endpoints": severity_counts.get("LOW", 0),
            "compliance_score": self._calculate_compliance_score(),
        }

    def _calculate_compliance_score(self) -> float:
        """Calculate security compliance score (0-100)."""
        total = len(self.protected_endpoints) + len(self.unprotected_endpoints)
        if total == 0:
            return 100.0

        protected_ratio = len(self.protected_endpoints) / total

        # Penalty for critical vulnerabilities
        critical_count = sum(
            1 for ep in self.unprotected_endpoints if ep["severity"] == "CRITICAL"
        )
        critical_penalty = min(critical_count * 20, 80)  # Max 80% penalty

        score = (protected_ratio * 100) - critical_penalty
        return max(0.0, score)


def main():
    """Run security audit and generate report."""
    api_file = Path("/Users/bogdan/work/leanvibe-dev/ant-farm/src/api/main.py")

    if not api_file.exists():
        print(f"❌ API file not found: {api_file}")
        return

    auditor = SecurityAuditor(api_file)
    results = auditor.audit_endpoints()

    print("🔒 API Security Audit Report")
    print("=" * 50)
    print(f"📊 Total Endpoints: {results['total_endpoints']}")
    print(f"✅ Protected: {results['protected_endpoints']}")
    print(f"❌ Unprotected: {results['unprotected_endpoints']}")
    print(f"🎯 Compliance Score: {results['summary']['compliance_score']:.1f}%")
    print()

    summary = results["summary"]
    if summary["critical_vulnerabilities"] > 0:
        print(f"🚨 CRITICAL: {summary['critical_vulnerabilities']} endpoints")
    if summary["high_risk_endpoints"] > 0:
        print(f"⚠️  HIGH RISK: {summary['high_risk_endpoints']} endpoints")
    if summary["medium_risk_endpoints"] > 0:
        print(f"⚡ MEDIUM RISK: {summary['medium_risk_endpoints']} endpoints")
    if summary["low_risk_endpoints"] > 0:
        print(f"ℹ️  LOW RISK: {summary['low_risk_endpoints']} endpoints")

    print("\n🔍 Unprotected Endpoints Details:")
    print("-" * 40)

    for endpoint in results["unprotected_details"]:
        severity_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "ℹ️"}
        print(
            f"{severity_emoji.get(endpoint['severity'], '?')} {endpoint['severity']}: {endpoint['method']} {endpoint['path']}"
        )

    print("\n📝 Report saved to: security_audit_report.json")

    # Save detailed report
    import json

    with open("security_audit_report.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
