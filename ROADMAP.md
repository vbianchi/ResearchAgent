# ResearchAgent: Project Roadmap (Transitioning to LangGraph with a Vision for CodeAct)

This document outlines the planned development path for the ResearchAgent project, reflecting a strategic shift towards using LangGraph for core agent orchestration and a future vision for incorporating CodeAct-inspired dynamic code execution. [cite: 605]

## Guiding Principles
-   **Robustness & Control:** Prioritize architectures that offer explicit state management and reliable task lifecycle control. [cite: 606]
-   **Flexibility & Power:** Enable the agent to tackle a wider range of complex tasks. [cite: 607]
-   **User-Centricity:** Improve research efficiency and user experience. [cite: 608]
-   **Modularity & Iteration:** Design for maintainability and incremental improvement. [cite: 609]

## Phase 1: Initial LangChain-based Implementation & Learnings (Completed)
-   Developed a foundational UI/backend with a PCEE agent using LangChain `AgentExecutor`. [cite: 610]
-   Key Learning: Highlighted the need for more robust asynchronous task control, reliable cancellation, and a more flexible action paradigm, leading to the strategic decisions for subsequent phases. [cite: 611]

## Phase 2: LangGraph Migration & Core Stability (Current Focus - V2.6.0 Target)
This phase is dedicated to migrating the core agent logic to LangGraph to build a more robust, controllable, and observable foundation. [cite: 612]

1.  **CRITICAL: Migrate PCEE Workflow to LangGraph:** (Largely Completed in Test Environment) [cite: 1, 613]
    * **Description:** Re-implemented Intent Classification, Planner, Controller, Executor logic, Step-wise Evaluator, and Overall Evaluator as a stateful graph in LangGraph. Includes iterative processing of plan steps. [cite: 1, 613]
    * **Goal:** A fully functional PCEE loop on LangGraph, retaining and enhancing existing agent capabilities. [cite: 614]
    * **Status:** Core loop functional in tests. Next steps focus on retry logic and full application integration.

2.  **CRITICAL: Implement Robust Task Interruption & Cancellation on LangGraph:** (Next Major Focus after Core Loop Stabilization) [cite: 1, 615]
    * **Description:** Utilize LangGraph's interrupt features, `asyncio.Task` management, and callback checks to ensure STOP signals and context switches reliably halt or pause graph executions. [cite: 615]
    * **Goal:** Responsive and reliable task lifecycle control. [cite: 615]

3.  **HIGH: Refine and Integrate Core LangGraph Agent into Main Application:** (Next Steps)
    * **Implement Robust Retry Logic:** Enhance Controller and loop to use Step Evaluator feedback for retrying failed steps. [cite: 1]
    * **Integrate with `server.py`:** Replace old agent flow, manage WebSocket communication for real-time updates via graph events. [cite: 1]
    * **Implement `DIRECT_QA` Path:** Add a dedicated path in LangGraph for direct question answering. [cite: 1]
    * **Refine Tool Loading:** Optimize if necessary. [cite: 1]
    * **Enhance Error Handling:** Improve overall agent robustness. [cite: 1]
    * **UI Feedback:** Ensure rich UI feedback via LangGraph's streaming capabilities (`astream_events`). [cite: 575, 617]
    * **Re-validate Core Features:** Tool System (Plug and Play for LangGraph nodes [cite: 616]), State Checkpointing (for resilience, future pause/resume [cite: 616]), Artifact Management (integration with graph events [cite: 618]).

4.  **HIGH: Comprehensive Testing of the New LangGraph-based Architecture.** (Ongoing with each integration) [cite: 618]

## Phase 3: V2.x Enhancements (Building on Stable LangGraph Foundation)
Focus on leveraging LangGraph for enhanced user experience and initial advanced features.

1.  **SHOULD HAVE: Implement Basic Concurrent Task Processing:** (Post-Core Stability) [cite: 619]
    * **Description:** Allow one agent task (LangGraph execution) to run in the background while the user interacts with a new foreground task, using LangGraph's state and checkpointing. [cite: 619, 620]
    * **Goal:** Support for "one active + one background" task per session. [cite: 621]
2.  **SHOULD HAVE: UI/UX Polish & Feature Completion:**
    * Refine global status indicators, "View [artifact] in Artifacts" links. [cite: 622]
    * Address lower-priority UI bugs. [cite: 622]
3.  **SHOULD HAVE: "Plug and Play" Tool System Finalization on LangGraph:**
    * Complete formalization of Tool Input/Output Schemas. [cite: 623]
    * Finalize the New Tool Integration Guide for the LangGraph context. [cite: 624]
4.  **SHOULD HAVE: Make Tavily Search Optional (User Configuration).** [cite: 624]

## Phase 4: Future Iterations (Advanced Capabilities & `PythonSandboxTool`)
Focus on significantly expanding agent capabilities and intelligence. [cite: 625]

1.  **KEY STRATEGIC FEATURE: Develop `PythonSandboxTool` (CodeAct-Inspired):** [cite: 625]
    * **Description:** Implement a specialized tool for LangGraph that allows the agent to request the generation and sandboxed execution of Python code for complex or novel sub-tasks. [cite: 626] The tool will internally use a code-generation LLM. [cite: 627] Scripts will run in a secure sandbox. [cite: 628]
    * **Goal:** Provide powerful, dynamic problem-solving capabilities, reducing the need for many predefined granular tools. [cite: 628, 629] This aligns with user enthusiasm. [cite: 629]
    * **Impact:** Transforms the agent's ability to handle unforeseen challenges. [cite: 630]
2.  **NICE TO HAVE: Advanced Agent Reasoning & Self-Correction:**
    * Utilize LangGraph's cyclical nature for more sophisticated error handling, code debugging (especially for the `PythonSandboxTool`), and plan refinement loops. [cite: 631]
3.  **NICE TO HAVE: Further Tool Ecosystem Expansion:**
    * Rscript Execution Tool, enhanced PDF processing, specialized bioinformatics tools, data visualization tools, etc., built for the LangGraph architecture. [cite: 632]
4.  **NICE TO HAVE: Advanced UI/UX & Workspace Enhancements:**
    * Visual graph execution monitor. [cite: 633]

## Phase 5: Advanced Agent Autonomy & Specialized Applications (Longer-Term)
(Explore more autonomous agent behaviors, long-term memory solutions, and potential cloud deployment scenarios, building on the robust and flexible LangGraph + CodeAct-inspired foundation.) [cite: 634]

This roadmap now strongly reflects the strategic shift to LangGraph, our current progress, and the exciting future potential of integrating CodeAct principles. [cite: 634]