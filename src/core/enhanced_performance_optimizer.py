"""Performance optimization and monitoring for LeanVibe Agent Hive 2.0.

This module provides:
- Database query optimization and monitoring
- Cache performance optimization
- Performance analysis and recommendations
- Automated system optimization

Performance Target: <50ms p95 response time
"""

import time
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import text

from .caching import get_cache_manager
from .models import get_database_manager

logger = structlog.get_logger()


@dataclass
class QueryMetrics:
    """Metrics for a database query."""

    query_hash: str
    query_text: str
    execution_time_ms: float
    rows_returned: int
    rows_examined: int
    cache_hit: bool
    timestamp: float


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    database_health: dict[str, Any]
    cache_health: dict[str, Any]
    slow_queries: list[QueryMetrics]
    performance_recommendations: list[str]
    overall_score: float
    meets_target: bool


class QueryAnalyzer:
    """Analyzes database queries for performance optimization."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.query_log = []
        self.slow_query_threshold_ms = 50.0

    async def analyze_query(
        self, query: str, params: dict | None = None
    ) -> QueryMetrics:
        """Analyze a query's performance."""

        start_time = time.time()
        query_hash = str(hash(query))

        session = self.db_manager.get_session()
        try:
            # Execute query and measure performance
            result = session.execute(text(query), params or {})
            rows = result.fetchall()

            execution_time = (time.time() - start_time) * 1000

            metrics = QueryMetrics(
                query_hash=query_hash,
                query_text=query[:200] + "..." if len(query) > 200 else query,
                execution_time_ms=execution_time,
                rows_returned=len(rows),
                rows_examined=len(rows),  # Simplified - would use EXPLAIN in production
                cache_hit=False,
                timestamp=time.time(),
            )

            # Log slow queries
            if execution_time > self.slow_query_threshold_ms:
                logger.warning(
                    "Slow query detected",
                    execution_time_ms=execution_time,
                    query=metrics.query_text,
                    rows_returned=len(rows),
                )
                self.query_log.append(metrics)

            return metrics

        except Exception as e:
            logger.error("Query analysis failed", query=query[:100], error=str(e))
            raise
        finally:
            session.close()

    def get_slow_queries(self, limit: int = 10) -> list[QueryMetrics]:
        """Get the slowest queries."""
        sorted_queries = sorted(
            self.query_log, key=lambda q: q.execution_time_ms, reverse=True
        )
        return sorted_queries[:limit]

    def generate_recommendations(self) -> list[str]:
        """Generate performance recommendations based on query analysis."""
        recommendations = []

        if not self.query_log:
            return ["No queries analyzed yet"]

        slow_queries = [
            q
            for q in self.query_log
            if q.execution_time_ms > self.slow_query_threshold_ms
        ]

        if slow_queries:
            recommendations.append(
                f"Found {len(slow_queries)} slow queries exceeding {self.slow_query_threshold_ms}ms"
            )

        # Analyze query patterns
        query_patterns = {}
        for query in self.query_log:
            # Simple pattern detection
            if "SELECT * FROM" in query.query_text.upper():
                query_patterns["select_star"] = query_patterns.get("select_star", 0) + 1
            if "ORDER BY" not in query.query_text.upper() and query.rows_returned > 100:
                query_patterns["missing_order"] = (
                    query_patterns.get("missing_order", 0) + 1
                )
            if "LIMIT" not in query.query_text.upper() and query.rows_returned > 1000:
                query_patterns["missing_limit"] = (
                    query_patterns.get("missing_limit", 0) + 1
                )

        if query_patterns.get("select_star", 0) > 0:
            recommendations.append(
                f"Avoid SELECT * queries ({query_patterns['select_star']} found) - specify needed columns"
            )

        if query_patterns.get("missing_order", 0) > 0:
            recommendations.append(
                f"Consider adding ORDER BY to queries returning many rows ({query_patterns['missing_order']} found)"
            )

        if query_patterns.get("missing_limit", 0) > 0:
            recommendations.append(
                f"Consider adding LIMIT to queries returning many rows ({query_patterns['missing_limit']} found)"
            )

        return recommendations


class DatabaseHealthMonitor:
    """Monitors database health and performance."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    async def check_health(self) -> dict[str, Any]:
        """Perform comprehensive database health check."""

        health_data = {
            "status": "healthy",
            "connection_test": False,
            "query_performance": {},
            "table_sizes": {},
            "index_usage": {},
            "recommendations": [],
        }

        session = self.db_manager.get_session()
        try:
            # Test basic connectivity
            start_time = time.time()
            session.execute(text("SELECT 1"))
            connection_time = (time.time() - start_time) * 1000

            health_data["connection_test"] = True
            health_data["connection_time_ms"] = connection_time

            if connection_time > 10:
                health_data["recommendations"].append("Database connection is slow")

            # Check table sizes (PostgreSQL specific)
            try:
                size_query = """
                SELECT
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 10
                """

                result = session.execute(text(size_query))
                tables = result.fetchall()

                for table in tables:
                    health_data["table_sizes"][table[1]] = {
                        "size_pretty": table[2],
                        "size_bytes": table[3],
                    }

            except Exception as e:
                logger.debug("Could not fetch table sizes", error=str(e))

            # Check for missing indexes on foreign keys
            try:
                missing_indexes_query = """
                SELECT DISTINCT
                    t.table_name,
                    kcu.column_name
                FROM information_schema.table_constraints t
                LEFT JOIN information_schema.key_column_usage kcu
                    ON t.constraint_name = kcu.constraint_name
                LEFT JOIN pg_stat_user_indexes i
                    ON i.relname = t.table_name
                    AND i.indexrelname LIKE '%' || kcu.column_name || '%'
                WHERE t.constraint_type = 'FOREIGN KEY'
                    AND t.table_schema = 'public'
                    AND i.indexrelname IS NULL
                """

                result = session.execute(text(missing_indexes_query))
                missing_indexes = result.fetchall()

                if missing_indexes:
                    health_data["recommendations"].extend(
                        [
                            f"Consider adding index on {table}.{column}"
                            for table, column in missing_indexes
                        ]
                    )

            except Exception as e:
                logger.debug("Could not check for missing indexes", error=str(e))

            # Performance test queries
            test_queries = [
                (
                    "Context lookup",
                    "SELECT COUNT(*) FROM contexts WHERE agent_id IS NOT NULL",
                ),
                (
                    "Task status count",
                    "SELECT status, COUNT(*) FROM tasks GROUP BY status",
                ),
                (
                    "Agent performance",
                    "SELECT type, AVG(load_factor) FROM agents GROUP BY type",
                ),
            ]

            for test_name, query in test_queries:
                start_time = time.time()
                session.execute(text(query))
                query_time = (time.time() - start_time) * 1000

                health_data["query_performance"][test_name] = query_time

                if query_time > 50:
                    health_data["recommendations"].append(
                        f"{test_name} query is slow ({query_time:.1f}ms)"
                    )

        except Exception as e:
            health_data["status"] = "unhealthy"
            health_data["error"] = str(e)
            logger.error("Database health check failed", error=str(e))

        finally:
            session.close()

        return health_data


class EnhancedPerformanceOptimizer:
    """Enhanced performance optimization coordinator with caching integration."""

    def __init__(self, database_url: str, redis_url: str = "redis://localhost:6379"):
        self.db_manager = get_database_manager(database_url)
        self.query_analyzer = QueryAnalyzer(self.db_manager)
        self.db_health_monitor = DatabaseHealthMonitor(self.db_manager)
        self.cache_manager = None
        self.redis_url = redis_url

        # Performance targets
        self.target_response_time_ms = 50.0
        self.target_cache_hit_rate = 0.8
        self.target_db_connection_ms = 10.0

    async def initialize(self):
        """Initialize the performance optimizer."""
        try:
            self.cache_manager = await get_cache_manager(self.redis_url)
            logger.info("Enhanced performance optimizer initialized")
        except Exception as e:
            logger.error("Performance optimizer initialization failed", error=str(e))
            raise

    async def generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""

        logger.info("Generating performance report")

        # Get database health
        db_health = await self.db_health_monitor.check_health()

        # Get cache health
        cache_health = (
            await self.cache_manager.health_check() if self.cache_manager else {}
        )

        # Get slow queries
        slow_queries = self.query_analyzer.get_slow_queries()

        # Generate recommendations
        recommendations = []
        recommendations.extend(self.query_analyzer.generate_recommendations())
        recommendations.extend(db_health.get("recommendations", []))

        # Cache-specific recommendations
        if cache_health.get("cache_stats"):
            for namespace, stats in cache_health["cache_stats"].items():
                if stats.hit_rate < self.target_cache_hit_rate:
                    recommendations.append(
                        f"Cache hit rate for {namespace} is {stats.hit_rate:.2%} (target: {self.target_cache_hit_rate:.2%})"
                    )

                if stats.average_response_time_ms > self.target_response_time_ms:
                    recommendations.append(
                        f"Cache response time for {namespace} is {stats.average_response_time_ms:.1f}ms (target: <{self.target_response_time_ms}ms)"
                    )

        # Calculate overall performance score
        score_factors = []

        # Database performance (40% of score)
        if db_health.get("connection_time_ms", 0) <= self.target_db_connection_ms:
            score_factors.append(40)
        else:
            score_factors.append(
                max(
                    0,
                    40
                    - (db_health["connection_time_ms"] - self.target_db_connection_ms),
                )
            )

        # Cache performance (30% of score)
        if cache_health.get("performance_target_met", False):
            score_factors.append(30)
        else:
            score_factors.append(15)  # Partial credit

        # Query performance (30% of score)
        if len(slow_queries) == 0:
            score_factors.append(30)
        else:
            score_factors.append(max(0, 30 - len(slow_queries) * 5))

        overall_score = sum(score_factors)
        meets_target = (
            db_health.get("connection_time_ms", 0) <= self.target_db_connection_ms
            and cache_health.get("performance_target_met", False)
            and len(slow_queries) == 0
        )

        report = PerformanceReport(
            database_health=db_health,
            cache_health=cache_health,
            slow_queries=slow_queries,
            performance_recommendations=recommendations,
            overall_score=overall_score,
            meets_target=meets_target,
        )

        logger.info(
            "Performance report generated",
            overall_score=overall_score,
            meets_target=meets_target,
            recommendations_count=len(recommendations),
        )

        return report

    async def optimize_system(self) -> dict[str, Any]:
        """Perform automatic system optimizations."""

        optimizations_applied = []

        try:
            # Generate performance report first
            report = await self.generate_performance_report()

            # Apply cache optimizations
            if self.cache_manager:
                # Clear low-performing caches
                cache_stats = report.cache_health.get("cache_stats", {})
                for namespace, stats in cache_stats.items():
                    if stats.hit_rate < 0.3:  # Very low hit rate
                        cleared = await self.cache_manager.invalidate_namespace(
                            namespace
                        )
                        if cleared > 0:
                            optimizations_applied.append(
                                f"Cleared {cleared} entries from low-performing cache: {namespace}"
                            )

            # Apply database optimizations
            if report.database_health.get("status") == "healthy":
                session = self.db_manager.get_session()
                try:
                    # Update statistics for query planner
                    session.execute(text("ANALYZE"))
                    session.commit()
                    optimizations_applied.append("Updated database statistics")

                except Exception as e:
                    logger.warning("Could not update database statistics", error=str(e))
                finally:
                    session.close()

            return {
                "optimizations_applied": optimizations_applied,
                "performance_report": report,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error("System optimization failed", error=str(e))
            return {
                "error": str(e),
                "optimizations_applied": optimizations_applied,
                "timestamp": time.time(),
            }

    async def monitor_query(
        self, query: str, params: dict | None = None
    ) -> QueryMetrics:
        """Monitor a specific query's performance."""
        return await self.query_analyzer.analyze_query(query, params)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a quick performance summary."""
        slow_queries_count = len(
            [
                q
                for q in self.query_analyzer.query_log
                if q.execution_time_ms > self.target_response_time_ms
            ]
        )

        return {
            "total_queries_analyzed": len(self.query_analyzer.query_log),
            "slow_queries_count": slow_queries_count,
            "performance_target_ms": self.target_response_time_ms,
            "cache_target_hit_rate": self.target_cache_hit_rate,
        }


# Global performance optimizer instance (enhanced version)
enhanced_performance_optimizer = None


async def get_enhanced_performance_optimizer(
    database_url: str, redis_url: str = "redis://localhost:6379"
) -> EnhancedPerformanceOptimizer:
    """Get the global enhanced performance optimizer instance."""
    global enhanced_performance_optimizer
    if enhanced_performance_optimizer is None:
        enhanced_performance_optimizer = EnhancedPerformanceOptimizer(
            database_url, redis_url
        )
        await enhanced_performance_optimizer.initialize()
    return enhanced_performance_optimizer
