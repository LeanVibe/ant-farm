#!/usr/bin/env python3
"""
Database query performance analysis and optimization tool.

Analyzes slow queries, identifies missing indexes, and provides
optimization recommendations for the LeanVibe Agent Hive database.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg
import structlog
from tabulate import tabulate

logger = structlog.get_logger()


@dataclass
class QueryAnalysis:
    """Analysis result for a database query."""

    query: str
    execution_time_ms: float
    cost: float
    rows: int
    planning_time_ms: float
    execution_plan: dict
    recommendations: list[str]
    severity: str  # 'critical', 'warning', 'info'


@dataclass
class IndexRecommendation:
    """Recommendation for a database index."""

    table: str
    columns: list[str]
    index_type: str
    estimated_benefit: str
    sql: str


class DatabasePerformanceAnalyzer:
    """Analyzes and optimizes database performance."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection: asyncpg.Connection | None = None

    async def connect(self):
        """Connect to the database."""
        self.connection = await asyncpg.connect(self.database_url)

    async def disconnect(self):
        """Disconnect from the database."""
        if self.connection:
            await self.connection.close()

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a specific query's performance."""
        if not self.connection:
            await self.connect()

        # Get execution plan with timing
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

        try:
            start_time = time.time()
            result = await self.connection.fetch(explain_query)
            execution_time = (time.time() - start_time) * 1000

            plan_data = result[0][0][0]  # Extract JSON data

            # Extract metrics
            planning_time = plan_data.get("Planning Time", 0)
            execution_time_actual = plan_data.get("Execution Time", execution_time)

            plan = plan_data.get("Plan", {})
            total_cost = plan.get("Total Cost", 0)
            actual_rows = plan.get("Actual Rows", 0)

            # Generate recommendations
            recommendations = self._generate_recommendations(plan_data, query)

            # Determine severity
            severity = self._determine_severity(execution_time_actual, total_cost)

            return QueryAnalysis(
                query=query,
                execution_time_ms=execution_time_actual,
                cost=total_cost,
                rows=actual_rows,
                planning_time_ms=planning_time,
                execution_plan=plan_data,
                recommendations=recommendations,
                severity=severity,
            )

        except Exception as e:
            logger.error("Query analysis failed", query=query[:100], error=str(e))
            return QueryAnalysis(
                query=query,
                execution_time_ms=0,
                cost=0,
                rows=0,
                planning_time_ms=0,
                execution_plan={},
                recommendations=[f"Query analysis failed: {str(e)}"],
                severity="critical",
            )

    def _generate_recommendations(self, plan_data: dict, query: str) -> list[str]:
        """Generate optimization recommendations based on execution plan."""
        recommendations = []
        plan = plan_data.get("Plan", {})

        # Check for sequential scans
        if self._has_sequential_scan(plan):
            recommendations.append(
                "Consider adding indexes for tables with sequential scans"
            )

        # Check for high cost operations
        if plan.get("Total Cost", 0) > 1000:
            recommendations.append(
                "Query has high cost - consider query optimization or indexing"
            )

        # Check for nested loop joins with high row counts
        if self._has_expensive_nested_loops(plan):
            recommendations.append(
                "Consider optimizing join conditions or adding indexes for join columns"
            )

        # Check for sorts without indexes
        if self._has_expensive_sorts(plan):
            recommendations.append(
                "Consider adding indexes for ORDER BY or GROUP BY columns"
            )

        # Check execution time
        execution_time = plan_data.get("Execution Time", 0)
        if execution_time > 100:
            recommendations.append(
                f"Query execution time ({execution_time:.2f}ms) exceeds target (<100ms)"
            )

        return recommendations or ["No optimization recommendations"]

    def _has_sequential_scan(self, plan: dict) -> bool:
        """Check if plan contains sequential scans."""
        if plan.get("Node Type") == "Seq Scan":
            return True
        for child in plan.get("Plans", []):
            if self._has_sequential_scan(child):
                return True
        return False

    def _has_expensive_nested_loops(self, plan: dict) -> bool:
        """Check for expensive nested loop joins."""
        if plan.get("Node Type") == "Nested Loop" and plan.get("Actual Rows", 0) > 1000:
            return True
        for child in plan.get("Plans", []):
            if self._has_expensive_nested_loops(child):
                return True
        return False

    def _has_expensive_sorts(self, plan: dict) -> bool:
        """Check for expensive sort operations."""
        if plan.get("Node Type") == "Sort" and plan.get("Total Cost", 0) > 100:
            return True
        for child in plan.get("Plans", []):
            if self._has_expensive_sorts(child):
                return True
        return False

    def _determine_severity(self, execution_time: float, cost: float) -> str:
        """Determine severity based on execution metrics."""
        if execution_time > 500 or cost > 10000:
            return "critical"
        elif execution_time > 100 or cost > 1000:
            return "warning"
        else:
            return "info"

    async def analyze_common_queries(self) -> list[QueryAnalysis]:
        """Analyze common application queries."""
        common_queries = [
            # Agent queries
            "SELECT * FROM agents WHERE status = 'active'",
            "SELECT * FROM agents WHERE type = 'meta' AND status = 'active'",
            "SELECT COUNT(*) FROM agents WHERE status IN ('active', 'idle')",
            # Task queries
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY priority DESC, created_at ASC",
            "SELECT * FROM tasks WHERE agent_id = 'test-agent' AND status IN ('pending', 'running')",
            "SELECT COUNT(*) FROM tasks WHERE status = 'completed' AND created_at > NOW() - INTERVAL '24 hours'",
            # Context queries
            "SELECT * FROM contexts WHERE agent_id = 'test-agent' ORDER BY importance_score DESC LIMIT 10",
            "SELECT * FROM contexts WHERE category = 'code' AND created_at > NOW() - INTERVAL '7 days'",
            # Conversation queries
            "SELECT * FROM conversations WHERE agent_id = 'test-agent' ORDER BY created_at DESC LIMIT 50",
            "SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '1 hour'",
            # Vector similarity queries (if applicable)
            "SELECT * FROM contexts ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector LIMIT 5",
            "SELECT * FROM conversations ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector LIMIT 10",
        ]

        results = []
        for query in common_queries:
            try:
                analysis = await self.analyze_query(query)
                results.append(analysis)
            except Exception as e:
                logger.error("Failed to analyze query", query=query[:50], error=str(e))

        return results

    async def identify_missing_indexes(self) -> list[IndexRecommendation]:
        """Identify missing indexes based on query patterns."""
        recommendations = []

        # Check for common filter patterns
        filter_patterns = [
            ("agents", ["status"], "btree", "Fast agent status filtering"),
            ("agents", ["type", "status"], "btree", "Fast agent type+status filtering"),
            ("tasks", ["status"], "btree", "Fast task status filtering"),
            ("tasks", ["priority", "status"], "btree", "Fast task queue ordering"),
            ("tasks", ["agent_id", "status"], "btree", "Fast agent task lookup"),
            ("contexts", ["agent_id"], "btree", "Fast context lookup by agent"),
            (
                "contexts",
                ["importance_score"],
                "btree",
                "Fast importance-based queries",
            ),
            ("contexts", ["category"], "btree", "Fast category filtering"),
            (
                "conversations",
                ["agent_id"],
                "btree",
                "Fast conversation lookup by agent",
            ),
            ("conversations", ["session_id"], "btree", "Fast session-based queries"),
        ]

        for table, columns, index_type, benefit in filter_patterns:
            # Check if index exists
            index_name = f"idx_{table}_{'_'.join(columns)}"
            if not await self._index_exists(index_name):
                sql = f"CREATE INDEX CONCURRENTLY {index_name} ON {table} ({', '.join(columns)}) USING {index_type};"

                recommendations.append(
                    IndexRecommendation(
                        table=table,
                        columns=columns,
                        index_type=index_type,
                        estimated_benefit=benefit,
                        sql=sql,
                    )
                )

        # Check for vector indexes if using embeddings
        vector_patterns = [
            ("contexts", ["embedding"], "hnsw", "Fast vector similarity search"),
            (
                "conversations",
                ["embedding"],
                "hnsw",
                "Fast conversation similarity search",
            ),
        ]

        for table, columns, index_type, benefit in vector_patterns:
            index_name = f"idx_{table}_{'_'.join(columns)}_hnsw"
            if not await self._index_exists(index_name):
                sql = f"CREATE INDEX CONCURRENTLY {index_name} ON {table} USING hnsw ({columns[0]} vector_cosine_ops) WITH (m = 16, ef_construction = 64);"

                recommendations.append(
                    IndexRecommendation(
                        table=table,
                        columns=columns,
                        index_type=index_type,
                        estimated_benefit=benefit,
                        sql=sql,
                    )
                )

        return recommendations

    async def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        query = """
        SELECT EXISTS(
            SELECT 1 FROM pg_indexes 
            WHERE indexname = $1
        );
        """
        result = await self.connection.fetchval(query, index_name)
        return result

    async def get_slow_queries(self, min_duration_ms: float = 100) -> list[dict]:
        """Get slow queries from pg_stat_statements if available."""
        try:
            query = """
            SELECT 
                query,
                calls,
                total_time,
                mean_time,
                rows,
                100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
            FROM pg_stat_statements
            WHERE mean_time > $1
            ORDER BY mean_time DESC
            LIMIT 20;
            """

            results = await self.connection.fetch(query, min_duration_ms)
            return [dict(row) for row in results]

        except Exception as e:
            logger.warning("pg_stat_statements not available", error=str(e))
            return []

    async def generate_report(self) -> dict:
        """Generate comprehensive performance analysis report."""
        logger.info("Generating database performance report...")

        # Analyze common queries
        query_analyses = await self.analyze_common_queries()

        # Identify missing indexes
        index_recommendations = await self.identify_missing_indexes()

        # Get slow queries
        slow_queries = await self.get_slow_queries()

        # Calculate summary statistics
        critical_queries = [q for q in query_analyses if q.severity == "critical"]
        warning_queries = [q for q in query_analyses if q.severity == "warning"]

        avg_execution_time = (
            sum(q.execution_time_ms for q in query_analyses) / len(query_analyses)
            if query_analyses
            else 0
        )

        report = {
            "timestamp": time.time(),
            "summary": {
                "total_queries_analyzed": len(query_analyses),
                "critical_issues": len(critical_queries),
                "warning_issues": len(warning_queries),
                "average_execution_time_ms": round(avg_execution_time, 2),
                "missing_indexes": len(index_recommendations),
                "slow_queries_detected": len(slow_queries),
            },
            "query_analyses": [
                {
                    "query": analysis.query[:100] + "..."
                    if len(analysis.query) > 100
                    else analysis.query,
                    "execution_time_ms": analysis.execution_time_ms,
                    "cost": analysis.cost,
                    "rows": analysis.rows,
                    "severity": analysis.severity,
                    "recommendations": analysis.recommendations,
                }
                for analysis in query_analyses
            ],
            "index_recommendations": [
                {
                    "table": rec.table,
                    "columns": rec.columns,
                    "index_type": rec.index_type,
                    "benefit": rec.estimated_benefit,
                    "sql": rec.sql,
                }
                for rec in index_recommendations
            ],
            "slow_queries": slow_queries[:10],  # Top 10 slowest
        }

        return report


async def main():
    """Run database performance analysis."""
    from src.core.config import settings

    analyzer = DatabasePerformanceAnalyzer(settings.database_url)

    try:
        await analyzer.connect()
        report = await analyzer.generate_report()

        # Print summary
        print("🔍 Database Performance Analysis Report")
        print("=" * 50)

        summary = report["summary"]
        print(f"📊 Queries Analyzed: {summary['total_queries_analyzed']}")
        print(f"🚨 Critical Issues: {summary['critical_issues']}")
        print(f"⚠️  Warning Issues: {summary['warning_issues']}")
        print(
            f"⏱️  Average Execution Time: {summary['average_execution_time_ms']:.2f}ms"
        )
        print(f"📑 Missing Indexes: {summary['missing_indexes']}")
        print(f"🐌 Slow Queries: {summary['slow_queries_detected']}")

        # Print critical issues
        if report["query_analyses"]:
            print("\n🔍 Query Analysis Results:")
            print("-" * 30)

            table_data = []
            for analysis in report["query_analyses"]:
                severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
                table_data.append(
                    [
                        severity_icon.get(analysis["severity"], "?"),
                        analysis["query"][:50] + "...",
                        f"{analysis['execution_time_ms']:.2f}ms",
                        f"{analysis['cost']:.0f}",
                        analysis["rows"],
                    ]
                )

            print(
                tabulate(
                    table_data, headers=["Severity", "Query", "Time", "Cost", "Rows"]
                )
            )

        # Print index recommendations
        if report["index_recommendations"]:
            print("\n📑 Index Recommendations:")
            print("-" * 30)

            for rec in report["index_recommendations"]:
                print(
                    f"• {rec['table']}.{'+'.join(rec['columns'])} ({rec['index_type']})"
                )
                print(f"  Benefit: {rec['benefit']}")
                print(f"  SQL: {rec['sql']}")
                print()

        # Save detailed report
        with open("database_performance_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("📝 Detailed report saved to: database_performance_report.json")

        # Return exit code based on critical issues
        return 1 if summary["critical_issues"] > 0 else 0

    finally:
        await analyzer.disconnect()


if __name__ == "__main__":
    exit(asyncio.run(main()))
