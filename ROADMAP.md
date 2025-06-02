# ResearchAgent: Project Roadmap (Transitioning to LangGraph with a Vision for CodeAct)

This document outlines the planned development path for the ResearchAgent project, reflecting a strategic shift towards using LangGraph for core agent orchestration and a future vision for incorporating CodeAct-inspired dynamic code execution.

## Guiding Principles
-   **Robustness & Control:** Prioritize architectures that offer explicit state management and reliable task lifecycle control.
-   **Flexibility & Power:** Enable the agent to tackle a wider range of complex tasks.
-   **User-Centricity:** Improve research efficiency and user experience.
-   **Modularity & Iteration:** Design for maintainability and incremental improvement.

## Phase 1: Initial LangChain-based Implementation & Learnings (Completed)
-   Developed a foundational UI/backend with a PCEE agent using LangChain `AgentExecutor`.
-   Key Learning: Highlighted the need for more robust asynchronous task control, reliable cancellation, and a more flexible action paradigm, leading to the strategic decisions for subsequent phases.

## Phase 2: LangGraph Migration & Core Stability (Current Focus - V1.0 Go-Live Target)
This phase is dedicated to migrating the core agent logic to LangGraph to build a more robust, controllable, and observable foundation.

1.  **CRITICAL: Migrate PCEE Workflow to LangGraph:**
    * **Description:** Re-implement Intent Classification, Planner, Controller, Executor logic, and Evaluators as a stateful graph in LangGraph.
    * **Goal:** A fully functional PCEE loop on LangGraph, retaining existing agent capabilities.

2.  **CRITICAL: Implement Robust Task Interruption & Cancellation on LangGraph:**
    * **Description:** Utilize LangGraph's interrupt features, `asyncio.Task` management, and callback checks to ensure STOP signals and context switches reliably halt or pause graph executions.
    * **Goal:** Responsive and reliable task lifecycle control.

3.  **HIGH: Re-validate Core Features & UI Integration with LangGraph:**
    * **Tool System:** Adapt the "Plug and Play" tool loading for LangGraph nodes.
    * **State Checkpointing:** Implement for resilience and future pause/resume.
    * **UI Feedback:** Leverage LangGraph's streaming for real-time UI updates.
    * **Artifact Management:** Ensure artifact generation and viewer refresh integrate with graph events.
    * **Testing:** Comprehensive testing of the new LangGraph-based architecture.

## Phase 3: V1.x Enhancements (Building on Stable LangGraph Foundation)
Focus on leveraging LangGraph for enhanced user experience and initial advanced features.

1.  **SHOULD HAVE: Implement Basic Concurrent Task Processing:**
    * **Description:** Allow one agent task (LangGraph execution) to run in the background while the user interacts with a new foreground task, using LangGraph's state and checkpointing.
    * **Goal:** Support for "one active + one background" task per session.

2.  **SHOULD HAVE: UI/UX Polish & Feature Completion:**
    * Refine global status indicators, "View [artifact] in Artifacts" links.
    * Address lower-priority UI bugs.

3.  **SHOULD HAVE: "Plug and Play" Tool System Finalization on LangGraph:**
    * Complete formalization of Tool Input/Output Schemas.
    * Finalize the New Tool Integration Guide for the LangGraph context.

4.  **SHOULD HAVE: Make Tavily Search Optional (User Configuration).**

## Phase 4: Future Iterations (Advanced Capabilities & `PythonSandboxTool`)
Focus on significantly expanding agent capabilities and intelligence.

1.  **KEY STRATEGIC FEATURE: Develop `PythonSandboxTool` (CodeAct-Inspired):**
    * **Description:** Implement a specialized tool for LangGraph that allows the agent to request the generation and sandboxed execution of Python code for complex or novel sub-tasks.
        * The tool will internally use a code-generation LLM to translate natural language sub-task descriptions into Python scripts.
        * These scripts will run in a secure sandbox with controlled access to libraries and the task workspace.
    * **Goal:** Provide the agent with powerful, dynamic problem-solving capabilities, reducing the need for many predefined granular tools and enabling more flexible interactions with data and the environment. This aligns with the user's enthusiasm for "Idea 1."
    * **Impact:** Transforms the agent's ability to handle unforeseen challenges and perform complex data manipulations or analyses.

2.  **NICE TO HAVE: Advanced Agent Reasoning & Self-Correction:**
    * Utilize LangGraph's cyclical nature for more sophisticated error handling, code debugging (especially for the `PythonSandboxTool`), and plan refinement loops.

3.  **NICE TO HAVE: Further Tool Ecosystem Expansion:**
    * Rscript Execution Tool, enhanced PDF processing, specialized bioinformatics tools, data visualization tools, etc., built for the LangGraph architecture.

4.  **NICE TO HAVE: Advanced UI/UX & Workspace Enhancements:**
    * Visual graph execution monitor.

## Phase 5: Advanced Agent Autonomy & Specialized Applications (Longer-Term)
(Explore more autonomous agent behaviors, long-term memory solutions, and potential cloud deployment scenarios, building on the robust and flexible LangGraph + CodeAct-inspired foundation.)

This roadmap now strongly reflects the strategic shift to LangGraph and the exciting future potential of integrating CodeAct principles via a specialized sandbox tool.
