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

-   UI Framework: Three-panel layout.

-   Backend Infrastructure: Python, WebSockets, HTTP file server.

-   Task Management: Create, select, delete, rename tasks with persistent storage.

-   Legacy Agent Flow (AgentExecutor-based): Intent Classification, P-C-E-E pipeline.

-   Core Tools & Pydantic v2 Migration.

-   Frontend & Backend Refactoring (v2.5.1).

Phase 2: LangGraph Migration & Core Stability (v2.6.0 - CURRENT & CRITICAL FOCUS)
---------------------------------------------------------------------------------

The primary goal of this phase is to migrate the core agentic backend to LangGraph and integrate it fully into the main application server for enhanced control, state management, and reliability.

1.  LangGraph PCEE Loop with Retry Logic (Isolated Testing - COMPLETE)

    -   Status: COMPLETE (Tested successfully in isolation via `langgraph_agent.py`).

2.  Integrate LangGraph Agent with Main Application (`server.py`) (Partially Complete - Direct Paths Operational)

    -   Goal: Replace the legacy `AgentExecutor`-based workflow with the new LangGraph-based agent in the live application.

    -   Progress & Achieved:

        -   The LangGraph agent is now integrated with `server.py` and `agent_flow_handlers.py`.

        -   `DIRECT_QA` Path: Fully functional. User queries classified for direct answering are processed by the graph (`direct_qa_node` -> `overall_evaluator_node`) and results are streamed to the UI.

        -   `DIRECT_TOOL_REQUEST` Path: Fully functional. User queries explicitly requesting a tool are processed by the graph (`direct_tool_executor_node` -> `overall_evaluator_node`), with dynamic tool loading and execution.

        -   `WebSocketCallbackHandler` is correctly handling events from these direct paths.

        -   Session data management in `server.py` and `agent_flow_handlers.py` supports these LangGraph executions.

    -   Status: IN PROGRESS (Direct paths complete; Full PCEE plan path is next).

3.  CRITICAL: Implement Full PCEE Workflow in Integrated LangGraph (Top Priority - NEXT)

    -   Goal: Replace placeholder nodes in `langgraph_agent.py` with actual implementations for multi-step plan execution.

    -   Details:

        -   Implement `PlannerNode`: Integrate `backend.planner.generate_plan` (or equivalent logic) to generate multi-step plans within the graph.

        -   Implement `ControllerNode`: Integrate `backend.controller.validate_and_prepare_step_action` (or equivalent logic) to select tools/formulate inputs for each plan step, utilizing `previous_step_executor_output` and retry feedback.

        -   Implement `ExecutorNode`: This node will execute the action decided by the `ControllerNode` (either calling a tool or making a direct LLM call for "None" tool steps). It will need access to dynamic tools and appropriate LLM configurations.

        -   Implement `StepEvaluatorNode`: Integrate `backend.evaluator.evaluate_step_outcome_and_suggest_correction` (or equivalent) to assess step success and manage the retry loop (incrementing `retry_count_for_current_step`, using `MAX_STEP_RETRIES`).

        -   Refine `agent_flow_handlers.process_execute_confirmed_plan`: Ensure this function correctly prepares the initial `ResearchAgentState` (including the confirmed plan steps) and invokes the `research_agent_graph` for plan execution.

    -   Status: PLANNED (This is the immediate next major development task).

4.  Refine Intent Classification (Ongoing)

    -   Goal: Improve `intent_classifier.py` to better distinguish `PLAN` intents, especially for meta-queries (e.g., "list available tools").

    -   Status: Ongoing.

5.  CRITICAL: Robust Task Interruption & Cancellation for Full LangGraph (Post-PCEE Implementation)

    -   Goal: Enable reliable stopping of ongoing multi-step LangGraph plan executions.

    -   Status: Planned.

6.  Implement UI for Concise Plan Proposal & Interaction (Post-PCEE Implementation)

    -   Status: Design exists, implementation deferred.

7.  Refine Tool Loading and Availability in Integrated LangGraph Nodes (As part of PCEE Implementation)

    -   Status: Ongoing with PCEE development.

8.  Enhance Error Handling and Overall Robustness (Ongoing)

    -   Status: Ongoing.

9.  Comprehensive UI Testing of the Full LangGraph Architecture (Post-PCEE Implementation)

    -   Status: Planned.

Phase 3: Advanced Interactivity & Tooling (Mid-Term - Post v2.6.0 Stability)
----------------------------------------------------------------------------

(Items largely remain the same as previous roadmap, but depend on successful full LangGraph PCEE migration)

-   Advanced User-in-the-Loop (UITL/HITL) Capabilities.

-   New Tools & Tool Refinements:

    -   Key Strategic Enhancement: `PythonSandboxTool` (CodeAct-Inspired).

-   Workspace RAG (Retrieval Augmented Generation).

Phase 4: Advanced Agent Autonomy & Specialized Applications (Longer-Term)
-------------------------------------------------------------------------

-   Advanced Re-planning & Self-Correction within LangGraph.

-   User Permission Gateway for Sensitive Tools.

-   Concurrent Task Processing.

This roadmap will guide our development efforts.