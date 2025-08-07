Of course. To enable the agent hive system to achieve self-improvement, we must first establish a stable, end-to-end "vertical slice" of functionality. This slice will ensure that a `MetaAgent` can be assigned a task, understand the codebase, modify it, test the changes, and commit them.

Here is a detailed plan outlining the core tasks required to achieve this initial bootstrap.

---

### **Core Objective: The First Self-Improvement Task**

The goal is to create a system where we can give the `MetaAgent` its first task via an API call, and it will successfully modify the codebase on its own.

**The target workflow:**
1.  A user submits a task (e.g., "Refactor a file") to a new API endpoint.
2.  The task is added to the `TaskQueue`.
3.  The `Orchestrator` assigns the task to a running `MetaAgent`.
4.  The `MetaAgent` uses the `ContextEngine` to understand the relevant code.
5.  The `MetaAgent` uses the `SelfModifier` to generate and apply the code changes.
6.  The `SelfModifier` runs tests to validate the changes.
7.  If tests pass, the changes are committed to a new git branch.
8.  The `MetaAgent` marks the task as complete.

---

### **Phase 1: Stabilize the Foundation**

This phase focuses on fixing critical bugs and addressing major technical debt that prevents the system from running.

**Task 1: Fix the `TaskQueue` (`src/core/task_queue.py`)**
*   **Problem:** The queue is non-functional due to two critical bugs. No tasks can be assigned or processed.
*   **Action Plan:**
    1.  **Fix Indentation Error:** In the `get_task` method, the main block of code for processing a retrieved `task_id` is incorrectly indented and will never execute. This block must be moved into the `try` block.
    2.  **Fix Undefined Status:** The `get_agent_active_task_count` method uses `TaskStatus.RUNNING`, which is not defined in the `TaskStatus` class. This will cause a runtime error. It should be corrected to use a valid status like `TaskStatus.IN_PROGRESS`.
    3.  **Write Unit Tests:** Create `tests/unit/test_task_queue.py` to verify that tasks can be submitted, retrieved, assigned, and completed. This will prevent future regressions.

**Task 2: Refactor the `Orchestrator` to Use SQLAlchemy (`src/core/orchestrator.py`)**
*   **Problem:** The `AgentRegistry` uses raw `psycopg2` for database operations, which bypasses the project's ORM (`SQLAlchemy`) and introduces security risks.
*   **Action Plan:**
    1.  **Modify `AgentRegistry`:** Rewrite all methods (`register_agent`, `update_agent_status`, etc.) to use a SQLAlchemy session and the `Agent` model from `src/core/models.py`.
    2.  **Dependency Injection:** Ensure the SQLAlchemy session is properly injected into the `AgentRegistry`.
    3.  **Write Unit Tests:** Create `tests/unit/test_orchestrator.py` to mock the database session and verify that agent registration and status updates work correctly through the ORM.

---

### **Phase 2: Implement Core Self-Improvement Tooling**

With a stable foundation, we now build the tools the `MetaAgent` will use to perform its work.

**Task 3: Implement the `SelfModifier` (`src/core/self_modifier.py`)**
*   **Problem:** The file is currently a placeholder.
*   **Action Plan:**
    1.  **Implement `propose_and_apply_change` function:** This will be the main entry point for the `MetaAgent`.
    2.  **Workflow:**
        *   **Input:** `file_path`, `change_description`.
        *   **Steps:**
            1.  Read the file content.
            2.  Use the `BaseAgent`'s `execute_with_cli_tool` to generate the new code.
            3.  Create a new feature branch in git (e.g., `agent/task-name-timestamp`).
            4.  Write the modified content to the file.
            5.  Execute the project's test suite (`pytest -q`).
            6.  If tests pass, commit the changes to the feature branch.
            7.  If tests fail, use `git reset --hard` to roll back the changes and report the failure.

**Task 4: Implement the `ContextEngine` (`src/core/context_engine.py`)**
*   **Problem:** The file is currently a placeholder. The `MetaAgent` cannot understand the codebase without it.
*   **Action Plan:**
    1.  **Implement `store_context`:** This method will take file content, generate a vector embedding for it, and save it to the `contexts` table using the SQLAlchemy model.
    2.  **Implement `retrieve_context`:** This method will take a query, generate an embedding for the query, and perform a vector similarity search against the `contexts` table to find the most relevant documents.
    3.  **Create a Bootstrapping Script:** Write a one-time script (`scripts/populate_context.py`) that scans the `src` directory, reads all `.py` files, and uses the `ContextEngine` to store their contents. This will provide the initial knowledge base for the `MetaAgent`.

---

### **Phase 3: Activate the `MetaAgent`**

Now we implement the agent itself and the means to give it a task.

**Task 5: Implement the `MetaAgent` (`src/agents/meta_agent.py`)**
*   **Problem:** The file is currently a placeholder.
*   **Action Plan:**
    1.  **Inherit from `BaseAgent`:** Ensure it has all the base functionalities.
    2.  **Implement the `run` loop:**
        *   Continuously try to fetch a task from the `TaskQueue`.
        *   When a task is received:
            1.  Log the task details.
            2.  Use the `ContextEngine` to retrieve context for the files mentioned in the task.
            3.  Construct a detailed prompt for the `SelfModifier`.
            4.  Call the `SelfModifier`'s `propose_and_apply_change` method.
            5.  Mark the task as `completed` or `failed` based on the result.

**Task 6: Create the API Endpoint and Bootstrap Script**
*   **Problem:** There is no way to give the system its first task.
*   **Action Plan:**
    1.  **Implement API Endpoint:** In `src/api/main.py`, create a `POST /api/v1/tasks` endpoint that accepts a JSON payload for a new task and uses `task_queue.submit_task()` to add it to the queue.
    2.  **Create Main Bootstrap Script (`bootstrap.py`):**
        *   This script will be the main entry point to start the system.
        *   It will:
            1.  Initialize the database connection and create tables.
            2.  Run the context population script (`scripts/populate_context.py`).
            3.  Initialize and start the `Orchestrator`.
            4.  Configure the `Orchestrator` to spawn one `MetaAgent` on startup.
            5.  Start the FastAPI application server using `uvicorn`.

After completing these six core tasks, the system will be ready for its first live test of the self-improvement loop.