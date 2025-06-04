BRAINSTORM.md - ResearchAgent Project (v2.6.0 In Progress)
==========================================================

This document tracks current workflow ideas, user feedback, and immediate brainstorming for the ResearchAgent project. For longer-term plans and phased development, please see `ROADMAP.md`.

**Current Version & Focus (v2.6.0 - LangGraph Migration & Core Stability):**

The ResearchAgent is actively undergoing a major architectural shift to LangGraph.

-   **Key Backend Advancement: LangGraph PCEE Loop with Retry Logic:**

    -   The core Plan-Code-Execute-Evaluate (PCEE) workflow has been successfully implemented as a stateful graph using LangGraph. This includes nodes for intent classification, planning, control (tool/LLM selection), execution, step-by-step evaluation, and overall evaluation.

    -   **Crucially, the retry logic within this isolated LangGraph setup has been verified.** The graph correctly identifies recoverable step failures, increments retry counts, uses evaluator feedback for subsequent attempts, and respects the maximum retry limit for a given step. This was validated through the `langgraph_agent.py` test script.

-   **Immediate Backend Focus: Server Integration:**

    -   The top priority is now integrating this fully functional LangGraph agent (from `langgraph_agent.py`) into the main application server (`server.py` and `agent_flow_handlers.py`). This involves replacing the older `AgentExecutor`-based flow with calls to the LangGraph, managing graph execution tasks, and ensuring the `WebSocketCallbackHandler` correctly streams updates to the UI.

**UI/UX Feedback & Ideas (To be revisited/prioritized post-LangGraph integration):**

1.  **Plan Confirmation UI (High Priority - Currently Blocked by Backend Changes):**

    -   **Issue:** The legacy backend sends a `propose_plan_for_confirmation` message type, but the frontend (`script.js`) does not yet handle it.

    -   **Note:** With LangGraph, plan proposal and confirmation might be handled differently, possibly through streamed messages directly from the `PlannerNode` or a dedicated confirmation step within the graph. This UI aspect will be re-evaluated once the LangGraph backend is integrated.

2.  **Chat Clutter & Plan Display Format (High Priority - Post LangGraph Integration):**

    -   **User Feedback:** The desire is for a cleaner interface, distinguishing clearly between direct agent-user messages and status/progress updates. Detailed plans should be less intrusive.

    -   **LangGraph Implication:** LangGraph's event streaming capabilities should allow for more granular control over what gets sent to the main chat versus the monitor log.

3.  **Color-Coding Agent Workspace & LLM Selectors (Medium Priority):**

    -   **User Idea:** Visually differentiate messages in the Agent Workspace (Monitor Log) based on the agent component (Planner, Controller, Executor, Evaluator) and link to LLM selectors.

    -   **LangGraph Implication:** Node-specific outputs from LangGraph can be tagged to facilitate this color-coding in the UI via the callback handler.

**Illustrative Workflow (Conceptual - Post LangGraph Integration):**

1.  User: "Use the Python REPL tool to obtain the literal string 'Python is fun!' (including the single quotes). Then, write this exact string to a file named 'python_quote.txt'."

2.  `IntentClassifierNode` (LangGraph): Classifies as `PLAN`.

3.  `PlannerNode` (LangGraph): Generates a plan:

    -   Step 1: Use `Python_REPL` to get "'Python is fun!'". Expected: The literal string.

    -   Step 2: Use `write_file` to save the string to "python_quote.txt". Expected: File created.

4.  **LangGraph Execution Loop Begins:**

    -   **Step 1 - Attempt 1:**

        -   `ControllerNode`: Decides `Python_REPL` with input, e.g., `"'Python is fun!'"` (or `print(...)`).

        -   `ExecutorNode`: Executes `Python_REPL`. Let's assume the tool initially fails to capture output (e.g., returns empty string due to a subtle tool issue).

        -   `StepEvaluatorNode`: Detects failure (output "" != expected "'Python is fun!'"). Marks as recoverable. Suggests retry, perhaps with `print(repr(...))`. Increments retry count for this step to 1.

    -   **Step 1 - Attempt 2 (Retry):**

        -   `ControllerNode`: Uses feedback. Decides `Python_REPL` with input `print(repr('Python is fun!'))`.

        -   `ExecutorNode`: Executes. `PythonREPLTool` now (hopefully) returns `"'Python is fun!'"`.

        -   `StepEvaluatorNode`: Output matches expected. Marks step as successful. Resets retry count for *next* step. `previous_step_executor_output` now holds `"'Python is fun!'"`.

    -   **Step 2 - Attempt 1:**

        -   `ControllerNode`: Sees plan for writing file. Uses `previous_step_executor_output` (`"'Python is fun!'"`) to formulate input for `write_file` tool (e.g., `python_quote.txt:::'Python is fun!'`).

        -   `ExecutorNode`: Executes `write_file`.

        -   `StepEvaluatorNode`: Confirms file written successfully.

5.  `OverallEvaluatorNode` (LangGraph): Assesses overall success. Sends final assessment.

This document will be updated as we progress with the LangGraph integration and new ideas emerge.