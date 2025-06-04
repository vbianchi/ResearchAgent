ResearchAgent: Project Roadmap (v2.6.0 In Progress)
===================================================

This document outlines the planned development path for the ResearchAgent project. It is a living document and will be updated as the project evolves.

Guiding Principles for Development
----------------------------------

-   Accuracy & Reliability Over Speed

-   User-in-the-Loop (UITL/HITL)

-   Modularity & Maintainability

-   Extensibility

Phase 1: Core Stability & Foundational Tools (Largely Complete - Basis of v2.5.1)
---------------------------------------------------------------------------------

-   **UI Framework:** Three-panel layout.

-   **Backend Infrastructure:** Python, WebSockets, HTTP file server.

-   **Task Management:** Create, select, delete, rename tasks with persistent storage.

-   **Legacy Agent Flow (AgentExecutor-based):** Intent Classification, P-C-E-E pipeline.

    -   Inter-step data flow fixed.

    -   Evaluator message persistence fixed.

-   **Core Tools:** Web search, file I/O, PubMed, web reader, DeepResearchTool v1.

-   **LLM Configuration:** Gemini & Ollama, role-specific.

-   **Frontend & Backend Refactoring (v2.5.1):** Modularized JavaScript and Python message handlers.

-   **Numerous Stability Fixes.**

Phase 2: LangGraph Migration & Core Stability (v2.6.0 - CURRENT & CRITICAL FOCUS)
---------------------------------------------------------------------------------

The primary goal of this phase is to migrate the core agentic backend to LangGraph and integrate it fully into the main application server for enhanced control, state management, and reliability.

1.  **LangGraph PCEE Loop with Retry Logic (Isolated Testing - COMPLETE)**

    -   **Goal:** Develop and test a stateful Plan-Code-Execute-Evaluate (PCEE) graph using LangGraph.

    -   **Details:**

        -   Implemented nodes: `IntentClassifierNode`, `PlannerNode`, `ControllerNode`, `ExecutorNode`, `StepEvaluatorNode`, `OverallEvaluatorNode`.

        -   Implemented iterative step processing with state updates.

        -   **Implemented and verified robust retry logic within the graph:**

            -   `StepEvaluatorNode` correctly identifies recoverable errors and suggests retry actions (tool, input instructions).

            -   `StepEvaluatorNode` correctly increments `retry_count_for_current_step` and maintains `current_step_index` for retries.

            -   `ControllerNode` utilizes feedback from `StepEvaluatorNode` for retry attempts.

            -   The graph respects `MAX_STEP_RETRIES` for a given step.

    -   **Status: COMPLETE** (Tested successfully in isolation via `langgraph_agent.py`).

2.  **CRITICAL: Integrate LangGraph Agent with Main Application (`server.py`) (Top Priority - IN PROGRESS)**

    -   **Goal:** Replace the legacy `AgentExecutor`-based workflow with the new LangGraph-based agent in the live application.

    -   **Details:**

        -   **Modify `backend/message_processing/agent_flow_handlers.py`:**

            -   Update `process_user_message` (and potentially `process_execute_confirmed_plan` or a new handler) to prepare initial state and invoke the `research_agent_graph.astream()` method.

        -   **Adapt `backend/server.py`:**

            -   Manage the lifecycle of the compiled `research_agent_graph` (e.g., instantiate once).

            -   Adapt session data management (`session_data`, `connected_clients`) to support LangGraph executions, including storing and managing the `asyncio.Task` for each graph run.

        -   **Integrate `WebSocketCallbackHandler`:**

            -   Ensure the existing `WebSocketCallbackHandler` (or an adapted/new version) is correctly instantiated per session and passed into the graph's `RunnableConfig`.

            -   Verify/update callback methods to handle events and data streamed from the LangGraph (e.g., node outputs, state changes). Consider using `astream_events` for more granular updates if beneficial.

        -   **Implement Streaming of Graph Outputs:**

            -   Stream node-generated AIMessages, status updates, tool calls, and final evaluations to the UI via the callback handler for real-time feedback.

        -   **Handle `task_id` and LLM Configurations:**

            -   Ensure `task_id` (for workspace and tool context) and session-specific LLM overrides are correctly passed into the LangGraph's initial state or `RunnableConfig`.

    -   **Status: IN PROGRESS (This is the immediate next development task).**

3.  **Implement `DIRECT_QA` Path in Integrated LangGraph (Post-Integration)**

    -   **Goal:** Route "DIRECT_QA" intents to a dedicated `DirectQANode` within the main LangGraph.

    -   **Details:** This node will use an LLM for direct answers, with output flowing to the `OverallEvaluatorNode`.

    -   **Status:** Planned (after core integration).

4.  **CRITICAL: Robust Task Interruption & Cancellation for Integrated LangGraph (Post-Integration)**

    -   **Goal:** Enable reliable stopping of ongoing LangGraph executions via the UI's "STOP" button.

    -   **Details:**

        -   Leverage LangGraph's built-in interrupt mechanisms.

        -   Use `asyncio.Task.cancel()` on the graph execution task managed in `server.py`.

        -   Ensure `_check_cancellation` in `WebSocketCallbackHandler` (or equivalent checks within graph nodes) effectively halts processing.

    -   **Status:** Planned (after core integration and basic functionality is verified).

5.  **Refine Tool Loading and Availability in Integrated LangGraph Nodes (Post-Integration)**

    -   **Status:** Planned.

6.  **Enhance Error Handling and Overall Robustness (Post-Integration)**

    -   **Status:** Planned.

7.  **Comprehensive UI Testing of the Integrated LangGraph Architecture (Post-Integration)**

    -   **Status:** Planned.

8.  **Implement UI for Concise Plan Proposal & Interaction (Deferred - Post LangGraph Integration)**

    -   **Goal:** Create a clean, user-friendly way to confirm plans, reducing chat clutter.

    -   **Status:** Design exists, implementation deferred until LangGraph backend is stable. The current `display_plan_for_confirmation` message will be adapted or replaced by direct streaming from LangGraph nodes.

Phase 3: Advanced Interactivity & Tooling (Mid-Term - Post v2.6.0)
------------------------------------------------------------------

(Items largely remain the same as previous roadmap, but depend on successful LangGraph migration)

-   **Advanced User-in-the-Loop (UITL/HITL) Capabilities:**

    -   Agent-initiated interaction points during plan execution (e.g., asking for clarification, choices).

-   **New Tools & Tool Refinements:**

    -   **Key Strategic Enhancement: `PythonSandboxTool` (CodeAct-Inspired):** A more robust and isolated environment for Python code execution, potentially using Docker or a similar sandboxing technology. This is a major feature.

    -   Data visualization tools.

    -   Version control integration (e.g., Git).

-   **Workspace RAG (Retrieval Augmented Generation):**

    -   Enable the agent to perform semantic search and retrieval over documents within the current task's workspace.

Phase 4: Advanced Agent Autonomy & Specialized Applications (Longer-Term)
-------------------------------------------------------------------------

-   **Advanced Re-planning & Self-Correction within LangGraph.**

-   **User Permission Gateway for Sensitive Tools.**

-   **Full Streaming Output for all Agent Steps/Thoughts to UI.**

-   **Specialized Agent Personas & Fine-tuning.**

-   **Concurrent Task Processing.**

This roadmap will guide our development efforts. Feedback and adjustments are welcome as the project progresses.