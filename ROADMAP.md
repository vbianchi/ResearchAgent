# ResearchAgent: Project Roadmap (Post-Reset)

This document outlines the planned development path for the ResearchAgent project, starting from its new, stable baseline.

## Guiding Principles for Development

-   **Stability First:** Each incremental feature must be fully functional and testable before moving on.
-   **Clarity and Simplicity:** Start with the simplest implementation and add complexity deliberately and only when necessary.
-   **User-in-the-Loop (UITL/HITL):** Maintain and enhance key points for user interaction and confirmation.
-   **Modularity & Maintainability:** Keep logic and configuration separate and well-defined.

### **Phase 1: Stable Core (v2.7 - CURRENT)**

-   **Status:** **COMPLETE**
-   **Description:** The project has been reset to a stable foundation. The backend now runs a standard LangChain `ReAct` agent.
-   **Core Functionality:**
    -   The agent can hold a conversation.
    -   The agent can autonomously choose and use tools from a predefined, dynamically loaded suite (web search, file I/O, etc.).
    -   The agent's internal monologue (thoughts, actions, tool observations) is streamed to the UI monitor in real-time.
    -   The UI is fully functional for task management, chat, and artifact viewing.

### **Phase 2: Re-introducing Advanced Control (Future)**

The goal of this phase is to gradually re-implement the control and planning capabilities that the previous LangGraph attempt aimed for, but in a step-by-step, verifiable manner.

**Step 1: Intent Classification**

-   **Task:** Re-introduce an `intent_classifier` as the first step in processing a user's query.
-   **Goal:** Create a conditional path. Simple questions (`DIRECT_QA`) will be answered directly by the LLM, while complex requests will be passed to the `ReAct` agent. This avoids invoking the full agent for simple tasks.
-   **Verification:** Confirm that simple questions are answered quickly without tool-use thoughts, while complex queries correctly trigger the full ReAct agent loop.
**Step 2: Explicit Planning & Confirmation**

-   **Task:** For complex intents, introduce a `planner` that generates a multi-step plan _before_ execution begins.
-   **Goal:** Present this plan to the user in the UI for confirmation or cancellation. The `ReAct` agent will then be tasked with executing this confirmed plan, one step at a time.
-   **Verification:** Confirm that a plan is generated and displayed correctly in the UI, and that the agent's execution follows the approved steps.
**Step 3: Advanced Execution and Evaluation**

-   **Task:** Explore replacing the single `ReAct` agent with more specialized components for executing and evaluating each step of the plan (revisiting the Controller/Executor/StepEvaluator pattern from the parked code).
-   **Goal:** Achieve a more robust and granular execution loop where the system can evaluate the success of each step and potentially retry or flag failures.
-   **Verification:** Test multi-step plans that involve potential failures to see if the system can handle them gracefully.

### **Phase 3: Advanced Features (Long-Term)**

-   **Task:** Implement a secure `PythonSandboxTool` for safer code execution.
-   **Task:** Enhance the UI for more detailed plan interaction (e.g., step-by-step approval, plan modification).
-   **Task:** Explore data visualization tools and other advanced capabilities.


# LAST UPDATE
# ResearchAgent: Project Roadmap

This document outlines the planned development path for the ResearchAgent project.

## Guiding Principles for Development

-   **Stability First:** Each incremental feature must be stable and testable before moving on.
-   **Clarity and Simplicity:** Start with the simplest implementation and add complexity deliberately.
-   **User-in-the-Loop (UITL/HITL):** Maintain key points for user interaction and confirmation.
-   **Modularity & Maintainability:** Keep logic and configuration separate and well-defined.

### Phase 1: Core UI & Legacy Agent (v2.5)

_This phase is complete._

### Phase 2: LangGraph Migration (v2.6 - Original Plan)

-   **Status:** **ABORTED**
-   **Outcome:** The "big bang" integration of the full PCEE workflow led to cascading errors and an unstable state. The decision has been made to restart this phase with a more granular, incremental approach.

### **Phase 2 (Restart): Foundational PCEE Implementation**

The primary goal of this phase is to methodically build and verify each component of the PCEE (Plan-Code-Execute-Evaluate) agent on the LangGraph foundation, ensuring stability at each step.

**Step 1: Bare-Bones Graph & `direct_qa` Path**

-   \[ \] **Task:** Strip `langgraph_agent.py` to its absolute minimum.
-   \[ \] **Task:** Create a graph with only a `direct_qa` node and an `overall_evaluator` node.
-   \[ \] **Verification:** Confirm that simple, no-tool questions are answered correctly and streamed to the UI.
**Step 2: Add the `Planner` Node**

-   \[ \] **Task:** Add the `planner` node to the graph.
-   \[ \] **Task:** Implement a conditional edge from the entry point to route to either `planner` or `direct_qa`.
-   \[ \] **Task:** The `planner`'s output (the plan) will be sent to the `overall_evaluator` node for display ONLY. No execution will occur.
-   \[ \] **Verification:** Confirm that complex queries generate a plan and that this plan is correctly displayed in the UI for review.
**Step 3: Add the `Controller` Node**

-   \[ \] **Task:** Add the `controller` node after the `planner`.
-   \[ \] **Task:** The `controller` will analyze the first step of the plan and its output (tool name and input) will be sent to the `overall_evaluator` for display.
-   \[ \] **Verification:** Confirm that the controller correctly interprets a plan step and selects a tool.
**Step 4: Add the `Executor` Node (Tool Path)**

-   \[ \] **Task:** Add the `executor` node after the `controller`.
-   \[ \] **Task:** The `executor` will execute a single tool call based on the controller's output.
-   \[ \] **Verification:** Test with single-step plans that use a tool (e.g., `write_file`). Confirm the tool executes successfully.
**Step 5: Add the `StepEvaluator` and Full Loop**

-   \[ \] **Task:** Add the `step_evaluator` node and the retry/next-step conditional logic.
-   \[ \] **Verification:** Test a full, multi-step plan. Verify that the agent can proceed from one step to the next and that the final output is correctly synthesized by the `overall_evaluator`.

### Phase 3: Advanced Features (Post-PCEE Stability)

-   **Task:** Re-implement robust task interruption and cancellation.
-   **Task:** Enhance UI for plan interaction (e.g., step-by-step approval).
-   **Task:** Implement the `PythonSandboxTool`



# PREVIOUS UPDATE
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