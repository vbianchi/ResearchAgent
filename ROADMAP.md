ResearchAgent: Project Roadmap (v2.6.0 In Progress)
===================================================

This document outlines the planned development path for the ResearchAgent project.

Guiding Principles for Development
----------------------------------
-   Accuracy & Reliability Over Speed
-   User-in-the-Loop (UITL/HITL)
-   Modularity & Maintainability
-   Extensibility

Phase 1: Core Stability & Foundational Tools (Largely Complete - Basis of v2.5.1)
---------------------------------------------------------------------------------
*This phase is complete.*

Phase 2: LangGraph Migration & Core Stability (v2.6.0 - CURRENT & CRITICAL FOCUS)
---------------------------------------------------------------------------------
The primary goal of this phase is to build a fully functional PCEE (Plan-Code-Execute-Evaluate) agent on a robust LangGraph foundation.

1.  **LangGraph Agent Integration with Main Application (`server.py`)**
    -   **Goal:** Replace the legacy `AgentExecutor` workflow with the new LangGraph agent.
    -   **Details:**
        -   `intent_classifier` is integrated and correctly routes "PLAN" and "DIRECT_QA" intents.
        -   The `DIRECT_QA` path is fully functional, providing immediate answers.
        -   The `PLAN` path correctly generates a plan and presents it to the user for confirmation via the UI.
        -   The graph's control flow has been refined for clarity and robust error handling (e.g., planner failures).
        -   `WebSocketCallbackHandler` is integrated for streaming LLM and node events.
    -   **Status: COMPLETE**

2.  **CRITICAL: Implement Core PCEE Execution Logic (Top Priority - IN PROGRESS)**
    -   **Goal:** Replace the placeholder nodes in the graph with their real, LLM-powered logic.
    -   **Next Steps:**
        1.  Implement `ControllerNode` logic.
        2.  Implement `ExecutorNode` logic (tool execution and direct LLM calls).
        3.  Implement `StepEvaluatorNode` logic (step assessment and retry management).
    -   **Status: IN PROGRESS**

3.  **Robust Task Interruption & Cancellation**
    -   **Goal:** Enable reliable stopping of ongoing LangGraph executions via the UI's "STOP" button.
    -   **Status:** Planned (Post-PCEE Implementation)

Phase 3: Advanced Interactivity & Tooling (Mid-Term - Post v2.6.0)
------------------------------------------------------------------
-   **Key Strategic Enhancement: `PythonSandboxTool` (CodeAct-Inspired)**
-   Data visualization tools.
-   Version control integration (e.g., Git).
-   Enable agent to perform semantic search (RAG) over the task's workspace.

Phase 4: Advanced Agent Autonomy & Specialized Applications (Longer-Term)
-------------------------------------------------------------------------
-   **Implement Dynamic Re-Planning Loop:**
    -   **Goal:** Enhance the agent's resilience by allowing it to modify its plan mid-execution.
    -   **Design:** If the `StepEvaluatorNode` determines a step has failed due to a fundamental flaw in the plan (not just a simple execution error), it will route the agent back to the `PlannerNode`. The Planner will receive the original query plus the new context about the failure, allowing it to generate a new, more informed plan.
-   User Permission Gateway for Sensitive Tools.
-   Specialized Agent Personas & Fine-tuning.
-   Concurrent Task Processing.