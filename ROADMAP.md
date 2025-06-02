# ResearchAgent: Project Roadmap (Targeting LangGraph Architecture)

This document outlines the planned development path for the ResearchAgent project, reflecting a strategic shift towards using LangGraph for core agent orchestration.

## Guiding Principles for Development
-   **Robustness & Control:** Prioritize architectures that offer explicit state management and reliable task lifecycle control.
-   **User-Centricity:** Ensure features improve research efficiency and user experience.
-   **Modularity & Maintainability:** Design for ease of development, testing, and future expansion.
-   **Iterative Improvement:** Incrementally build and refine, gathering feedback.

## Phase 1: Initial LangChain-based Implementation & Learnings (Completed)
-   Developed a foundational UI and backend with a PCEE agent workflow using LangChain's `AgentExecutor`.
-   Implemented dynamic tool loading, Pydantic v2 migration for several models, and significant UI/UX enhancements.
-   **Key Learning:** Encountered challenges with reliable and prompt task interruption/cancellation, highlighting the need for an architecture with more explicit control over asynchronous operations and state. This learning directly informs the strategy for Phase 2.

## Phase 2: LangGraph Migration & Core Stability (Current Focus - V1.0 Go-Live Target)
This phase is dedicated to migrating the core agent logic to LangGraph to build a more robust and controllable foundation.

1.  **CRITICAL: Migrate PCEE Workflow to LangGraph:**
    * **Description:** Redesign and implement the Intent Classifier, Planner, Controller, Executor (ReAct-style logic), and Evaluators (Step & Overall) as a stateful graph in LangGraph.
    * Define a clear state schema for the graph.
    * Adapt existing Pydantic models for compatibility with LangGraph state.
    * **Goal:** A fully functional PCEE loop running on LangGraph.

2.  **CRITICAL: Implement Robust Task Interruption & Cancellation:**
    * **Description:** Leverage LangGraph's interrupt mechanisms and `asyncio` task management to ensure STOP signals (from UI) and context switches reliably and promptly terminate or pause graph executions.
    * **Goal:** User-perceivable responsiveness to STOP commands; clean termination of tasks.

3.  **HIGH: Re-validate and Integrate Core Features on LangGraph:**
    * **Tool System:** Ensure the "Plug and Play" tool loading and custom tool execution work seamlessly within LangGraph nodes.
    * **State Management & Checkpointing:** Implement effective checkpointing for graph state to enable resilience and pave the way for pausable/resumable tasks.
    * **UI Feedback:** Utilize LangGraph's streaming capabilities (e.g., `astream_events`) for detailed real-time updates to the WebSocket UI (status, thoughts, tool calls, logs).
    * **Artifact Viewer:** Ensure artifact generation and viewer refresh are correctly triggered by graph events.
    * **Token Usage:** Adapt callbacks to accurately track LLM token usage within the graph.

4.  **HIGH: Comprehensive Testing of LangGraph Architecture:**
    * **Description:** Develop unit and integration tests specifically for the LangGraph implementation, covering state transitions, node execution, error handling, and cancellation.
    * **Goal:** A stable V1.0 release based on LangGraph.

## Phase 3: V1.x Enhancements (Building on LangGraph Foundation)
With a stable LangGraph core, this phase will focus on user-facing enhancements and leveraging the new architecture.

1.  **SHOULD HAVE: Implement Basic Concurrent Task Processing:**
    * **Description:** Allow one agent task (LangGraph execution) to run in the background while the user interacts with a new foreground task.
    * **Leverages:** LangGraph's state checkpointing and explicit task management.
    * **Goal:** Initial support for "one active + one background" task per session.

2.  **SHOULD HAVE: UI/UX Polish & Feature Completion:**
    * Finalize "View [artifact] in Artifacts" links.
    * Refine global status indicators and thinking updates based on LangGraph streaming.
    * Address lower-priority UI bugs (e.g., copy button placement).

3.  **SHOULD HAVE: "Plug and Play" Tool System Finalization:**
    * Complete formalization of Tool Input/Output Schemas (JSON Schemas from Pydantic).
    * Finalize the New Tool Integration Guide for the LangGraph context.

4.  **SHOULD HAVE: Make Tavily Search Optional (User Configuration).**

5.  **SHOULD HAVE: Advanced User-in-the-Loop (UITL/HITL) Patterns:**
    * Explore using LangGraph interrupts for more sophisticated human approval steps within complex plans.

## Phase 4: Future Iterations (Advanced Capabilities & Ecosystem)
Focus on expanding capabilities and toolset.

1.  **NICE TO HAVE: Advanced Agent Reasoning & Self-Correction:**
    * Utilize LangGraph's cyclical nature for more complex error handling and plan refinement loops.
2.  **NICE TO HAVE: Comprehensive Tool Ecosystem Expansion:**
    * Prioritize development of key tools: Rscript Execution, Data Analysis/Visualization tools, enhanced PDF/Document parsing, specialized Bioinformatics tools (e.g., BLAST wrappers, domain analysis, specific DB searches like ClinVar, KEGG).
3.  **NICE TO HAVE: Advanced UI/UX & Workspace Enhancements:**
    * Visual graph execution monitor.
    * Integrated folder viewer for workspaces.
4.  **NICE TO HAVE: Backend & Architecture (Scalability, Multi-User Personas).**
5.  **NICE TO HAVE: Deployment & DevOps (Streamlined deployment options).**

## Phase 5: Advanced Agent Autonomy & Specialized Applications (Longer-Term)
Explore more autonomous agent behaviors, long-term memory solutions, and potential cloud deployment scenarios.

This updated roadmap reflects the strategic decision to build upon LangGraph for a more robust and controllable ResearchAgent.
