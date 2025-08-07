#!/usr/bin/env python3
"""
Submit the first live self-improvement task to the API.
"""

import asyncio
import sys
from pathlib import Path

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def submit_self_improvement_task():
    """Submit a self-improvement task to the API and monitor the result."""

    api_url = "http://localhost:8001"
    endpoint = f"{api_url}/api/v1/tasks/self-improvement"

    # Test task: Improve error handling in task queue
    task_data = {
        "title": "Improve error handling in task queue",
        "description": "Add better error logging and retry logic to the task queue system in src/core/task_queue.py. Focus on making error messages more descriptive and adding structured logging for debugging.",
    }

    print("🎯 Submitting First Self-Improvement Task")
    print("=" * 50)
    print(f"API Endpoint: {endpoint}")
    print(f"Task: {task_data['title']}")
    print(f"Description: {task_data['description']}")
    print("=" * 50)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit the task
            print("📤 Submitting task...")
            response = await client.post(endpoint, json=task_data)

            if response.status_code == 200:
                result = response.json()
                print("✅ Task submitted successfully!")
                print(f"Task ID: {result.get('task_id', 'Unknown')}")
                print(f"Status: {result.get('status', 'Unknown')}")
                print(f"Message: {result.get('message', 'No message')}")

                # If we have a task ID, we could monitor progress
                task_id = result.get("task_id")
                if task_id:
                    print(f"\n📊 Task submitted with ID: {task_id}")
                    print(
                        "🤖 MetaAgent will now process this self-improvement task autonomously!"
                    )
                    print("\n🎉 SUCCESS: First self-improvement task is now running!")

                return True

            else:
                print(f"❌ Failed to submit task. Status: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except httpx.ConnectError:
        print("❌ Could not connect to API server.")
        print("💡 Make sure the API server is running on http://localhost:8001")
        print(
            "Start with: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001"
        )
        return False

    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        return False


async def check_api_health():
    """Check if the API server is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8001/health")
            if response.status_code == 200:
                print("✅ API server is healthy and ready")
                return True
            else:
                print(f"⚠️ API server responded with status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ API server health check failed: {e}")
        return False


async def main():
    """Main function to submit the first self-improvement task."""

    print("🔍 Checking API server health...")
    if not await check_api_health():
        print("\n💡 To start the API server, run:")
        print("cd /Users/bogdan/work/leanvibe-dev/ant-farm")
        print("python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001")
        return False

    print("🚀 API server is ready!")

    # Submit the self-improvement task
    success = await submit_self_improvement_task()

    if success:
        print("\n" + "=" * 60)
        print("🎉 LEANVIBE AGENT HIVE 2.0 SELF-IMPROVEMENT ACTIVE!")
        print("=" * 60)
        print("The system is now autonomously improving itself!")
        print("🤖 MetaAgent is processing the improvement task")
        print("📈 Future improvements will build on this foundation")
        print("🔄 The self-improvement loop is now operational!")
        print("=" * 60)
    else:
        print("\n💥 Failed to submit self-improvement task")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
