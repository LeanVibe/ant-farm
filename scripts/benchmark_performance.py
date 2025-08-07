"""Performance benchmarking script for LeanVibe Agent Hive 2.0.

This script validates that the system meets the <50ms p95 response time target
for key operations including database queries and cache operations.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

import structlog
from src.core.caching import get_cache_manager, CacheConfig, CacheLevel
from src.core.context_engine import get_context_engine
from src.core.task_queue import TaskQueue
from src.core.enhanced_performance_optimizer import get_enhanced_performance_optimizer

logger = structlog.get_logger()


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""

    operation_name: str
    total_requests: int
    total_time_seconds: float
    average_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_per_second: float
    meets_target: bool
    target_ms: float = 50.0


class PerformanceBenchmark:
    """Performance benchmarking suite."""

    def __init__(self, database_url: str, redis_url: str = "redis://localhost:6379"):
        self.database_url = database_url
        self.redis_url = redis_url
        self.target_p95_ms = 50.0
        self.cache_manager = None
        self.context_engine = None
        self.task_queue = None
        self.performance_optimizer = None

    async def initialize(self):
        """Initialize benchmark components."""
        logger.info("Initializing performance benchmark suite")

        # Initialize cache manager
        self.cache_manager = await get_cache_manager(self.redis_url)

        # Initialize context engine
        self.context_engine = await get_context_engine(self.database_url)

        # Initialize task queue
        self.task_queue = TaskQueue(self.redis_url)
        await self.task_queue.initialize()

        # Initialize performance optimizer
        self.performance_optimizer = await get_enhanced_performance_optimizer(
            self.database_url, self.redis_url
        )

        logger.info("Benchmark suite initialized")

    async def run_benchmark(
        self,
        operation_func,
        operation_name: str,
        num_requests: int = 100,
        concurrent_requests: int = 10,
    ) -> BenchmarkResult:
        """Run a performance benchmark for a specific operation."""

        logger.info(
            f"Starting benchmark: {operation_name}",
            num_requests=num_requests,
            concurrent_requests=concurrent_requests,
        )

        response_times = []
        start_time = time.time()

        # Run requests in batches to control concurrency
        batch_size = concurrent_requests
        for i in range(0, num_requests, batch_size):
            batch_end = min(i + batch_size, num_requests)
            batch_tasks = []

            for j in range(i, batch_end):
                batch_tasks.append(self._time_operation(operation_func, j))

            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect successful results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Operation failed", error=str(result))
                else:
                    response_times.append(result)

        total_time = time.time() - start_time

        if not response_times:
            logger.error(f"No successful operations for {operation_name}")
            return BenchmarkResult(
                operation_name=operation_name,
                total_requests=0,
                total_time_seconds=total_time,
                average_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                throughput_ops_per_second=0,
                meets_target=False,
            )

        # Calculate statistics
        avg_time_ms = statistics.mean(response_times)
        p50_time_ms = np.percentile(response_times, 50)
        p95_time_ms = np.percentile(response_times, 95)
        p99_time_ms = np.percentile(response_times, 99)
        min_time_ms = min(response_times)
        max_time_ms = max(response_times)
        throughput = len(response_times) / total_time
        meets_target = p95_time_ms <= self.target_p95_ms

        result = BenchmarkResult(
            operation_name=operation_name,
            total_requests=len(response_times),
            total_time_seconds=total_time,
            average_time_ms=avg_time_ms,
            p50_time_ms=p50_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            throughput_ops_per_second=throughput,
            meets_target=meets_target,
        )

        logger.info(
            f"Benchmark completed: {operation_name}",
            p95_time_ms=p95_time_ms,
            meets_target=meets_target,
            throughput=throughput,
        )

        return result

    async def _time_operation(self, operation_func, request_id: int) -> float:
        """Time a single operation and return response time in milliseconds."""
        start_time = time.perf_counter()

        try:
            await operation_func(request_id)
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.debug(f"Operation failed for request {request_id}", error=str(e))
            raise

    async def benchmark_cache_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark cache operations."""
        results = {}

        cache = self.cache_manager.get_cache("benchmark")

        # Cache SET operations
        async def cache_set_operation(request_id: int):
            key = f"benchmark_key_{request_id}"
            value = {"data": f"test_value_{request_id}", "timestamp": time.time()}
            await cache.set(key, value)

        results["cache_set"] = await self.run_benchmark(
            cache_set_operation, "Cache SET", num_requests=200, concurrent_requests=20
        )

        # Cache GET operations (with hits)
        async def cache_get_operation(request_id: int):
            key = f"benchmark_key_{request_id % 50}"  # Reuse keys for cache hits
            await cache.get(key)

        results["cache_get"] = await self.run_benchmark(
            cache_get_operation, "Cache GET", num_requests=500, concurrent_requests=50
        )

        # Cache DELETE operations
        async def cache_delete_operation(request_id: int):
            key = f"benchmark_key_{request_id}"
            await cache.delete(key)

        results["cache_delete"] = await self.run_benchmark(
            cache_delete_operation,
            "Cache DELETE",
            num_requests=100,
            concurrent_requests=10,
        )

        return results

    async def benchmark_context_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark context engine operations."""
        results = {}

        # Context storage operations
        async def context_store_operation(request_id: int):
            await self.context_engine.store_context(
                agent_id=f"benchmark_agent_{request_id % 10}",
                content=f"Benchmark context content for request {request_id}. " * 10,
                importance_score=0.5 + (request_id % 5) * 0.1,
                category="benchmark",
                topic=f"topic_{request_id % 5}",
            )

        results["context_store"] = await self.run_benchmark(
            context_store_operation,
            "Context Store",
            num_requests=100,
            concurrent_requests=10,
        )

        # Context retrieval operations
        async def context_retrieve_operation(request_id: int):
            agent_id = f"benchmark_agent_{request_id % 10}"
            query = f"benchmark query {request_id % 5}"
            await self.context_engine.retrieve_context(
                query=query, agent_id=agent_id, limit=10
            )

        results["context_retrieve"] = await self.run_benchmark(
            context_retrieve_operation,
            "Context Retrieve",
            num_requests=200,
            concurrent_requests=20,
        )

        # Memory stats operations
        async def memory_stats_operation(request_id: int):
            agent_id = f"benchmark_agent_{request_id % 10}"
            await self.context_engine.get_memory_stats(agent_id)

        results["memory_stats"] = await self.run_benchmark(
            memory_stats_operation,
            "Memory Stats",
            num_requests=100,
            concurrent_requests=15,
        )

        return results

    async def benchmark_task_queue_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark task queue operations."""
        results = {}

        # Queue stats operations
        async def queue_stats_operation(request_id: int):
            await self.task_queue.get_queue_stats()

        results["queue_stats"] = await self.run_benchmark(
            queue_stats_operation,
            "Queue Stats",
            num_requests=200,
            concurrent_requests=25,
        )

        # Agent task count operations
        async def agent_task_count_operation(request_id: int):
            agent_id = f"benchmark_agent_{request_id % 10}"
            await self.task_queue.get_agent_active_task_count(agent_id)

        results["agent_task_count"] = await self.run_benchmark(
            agent_task_count_operation,
            "Agent Task Count",
            num_requests=300,
            concurrent_requests=30,
        )

        # Total tasks operations
        async def total_tasks_operation(request_id: int):
            await self.task_queue.get_total_tasks()

        results["total_tasks"] = await self.run_benchmark(
            total_tasks_operation,
            "Total Tasks",
            num_requests=150,
            concurrent_requests=20,
        )

        return results

    async def benchmark_performance_optimizer(self) -> Dict[str, BenchmarkResult]:
        """Benchmark performance optimizer operations."""
        results = {}

        # Performance report generation
        async def performance_report_operation(request_id: int):
            await self.performance_optimizer.generate_performance_report()

        results["performance_report"] = await self.run_benchmark(
            performance_report_operation,
            "Performance Report",
            num_requests=20,
            concurrent_requests=5,
        )

        # Health check operations
        async def health_check_operation(request_id: int):
            await self.cache_manager.health_check()

        results["health_check"] = await self.run_benchmark(
            health_check_operation,
            "Health Check",
            num_requests=50,
            concurrent_requests=10,
        )

        return results

    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting full benchmark suite")

        all_results = {}
        summary = {
            "total_operations": 0,
            "operations_meeting_target": 0,
            "overall_target_met": True,
            "slowest_operations": [],
            "fastest_operations": [],
        }

        # Run all benchmark categories
        benchmark_categories = [
            ("cache", self.benchmark_cache_operations),
            ("context", self.benchmark_context_operations),
            ("task_queue", self.benchmark_task_queue_operations),
            ("performance", self.benchmark_performance_optimizer),
        ]

        for category_name, benchmark_func in benchmark_categories:
            logger.info(f"Running {category_name} benchmarks")

            try:
                category_results = await benchmark_func()
                all_results[category_name] = category_results

                # Update summary
                for operation_name, result in category_results.items():
                    summary["total_operations"] += 1

                    if result.meets_target:
                        summary["operations_meeting_target"] += 1
                    else:
                        summary["overall_target_met"] = False

                    # Track slowest and fastest operations
                    operation_summary = {
                        "name": f"{category_name}.{operation_name}",
                        "p95_time_ms": result.p95_time_ms,
                        "throughput": result.throughput_ops_per_second,
                    }

                    summary["slowest_operations"].append(operation_summary)

            except Exception as e:
                logger.error(f"Benchmark category {category_name} failed", error=str(e))
                all_results[category_name] = {"error": str(e)}

        # Sort and limit slowest/fastest operations
        summary["slowest_operations"].sort(key=lambda x: x["p95_time_ms"], reverse=True)
        summary["fastest_operations"] = sorted(
            summary["slowest_operations"], key=lambda x: x["p95_time_ms"]
        )[:5]
        summary["slowest_operations"] = summary["slowest_operations"][:5]

        # Calculate overall success rate
        if summary["total_operations"] > 0:
            summary["success_rate"] = (
                summary["operations_meeting_target"] / summary["total_operations"]
            )
        else:
            summary["success_rate"] = 0.0

        logger.info(
            "Benchmark suite completed",
            total_operations=summary["total_operations"],
            success_rate=summary["success_rate"],
            overall_target_met=summary["overall_target_met"],
        )

        return {
            "results": all_results,
            "summary": summary,
            "timestamp": time.time(),
            "target_p95_ms": self.target_p95_ms,
        }

    def print_benchmark_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 80)
        print("LEANVIBE AGENT HIVE 2.0 - PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)

        summary = results["summary"]
        target_ms = results["target_p95_ms"]

        print(f"\nOVERALL PERFORMANCE SUMMARY:")
        print(f"  Target: <{target_ms}ms p95 response time")
        print(f"  Total Operations: {summary['total_operations']}")
        print(f"  Operations Meeting Target: {summary['operations_meeting_target']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(
            f"  Overall Target Met: {'✅ YES' if summary['overall_target_met'] else '❌ NO'}"
        )

        print(f"\nSLOWEST OPERATIONS:")
        for op in summary["slowest_operations"]:
            status = "✅" if op["p95_time_ms"] <= target_ms else "❌"
            print(
                f"  {status} {op['name']}: {op['p95_time_ms']:.1f}ms p95 ({op['throughput']:.1f} ops/sec)"
            )

        print(f"\nFASTEST OPERATIONS:")
        for op in summary["fastest_operations"]:
            status = "✅" if op["p95_time_ms"] <= target_ms else "❌"
            print(
                f"  {status} {op['name']}: {op['p95_time_ms']:.1f}ms p95 ({op['throughput']:.1f} ops/sec)"
            )

        print(f"\nDETAILED RESULTS BY CATEGORY:")
        for category, category_results in results["results"].items():
            print(f"\n  {category.upper()}:")

            if "error" in category_results:
                print(f"    ❌ Error: {category_results['error']}")
                continue

            for operation, result in category_results.items():
                status = "✅" if result.meets_target else "❌"
                print(f"    {status} {operation}:")
                print(
                    f"      p95: {result.p95_time_ms:.1f}ms | avg: {result.average_time_ms:.1f}ms"
                )
                print(
                    f"      throughput: {result.throughput_ops_per_second:.1f} ops/sec"
                )
                print(f"      requests: {result.total_requests}")

        print("\n" + "=" * 80)


async def main():
    """Main benchmark execution."""

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.LoggerAdapter,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configuration
    database_url = "postgresql://user:password@localhost/hive_test"
    redis_url = "redis://localhost:6379"

    # Create and run benchmark
    benchmark = PerformanceBenchmark(database_url, redis_url)

    try:
        await benchmark.initialize()
        results = await benchmark.run_full_benchmark_suite()
        benchmark.print_benchmark_results(results)

        # Exit with appropriate code
        if results["summary"]["overall_target_met"]:
            print("\n🎉 All performance targets met!")
            exit(0)
        else:
            print(
                "\n⚠️  Some performance targets not met. Review optimization recommendations."
            )
            exit(1)

    except Exception as e:
        logger.error("Benchmark failed", error=str(e))
        print(f"\n❌ Benchmark failed: {e}")
        exit(2)


if __name__ == "__main__":
    asyncio.run(main())
