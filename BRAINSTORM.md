# BRAINSTORM.md - ResearchAgent Project (V1.0 Go-Live Target & Beyond)

This document tracks user feedback, and brainstorming ideas for the ResearchAgent project.
## Current Version & State (Targeting V1.0 Go-Live - Embarking on LangGraph Migration)

**Recent Key Advancements (Prior to LangGraph Decision):**
* "Plug and Play" Tool System: Dynamic tool loading via `tool_config.json` achieved.
* Pydantic v2 Migration: Largely completed for core data models and tool arguments.
* UI/UX Enhancements: Significant improvements to chat, plan proposal, token tracking, file uploads.

**Current Strategic Pivot:**
* **Migrating to LangGraph:** The project is shifting its core agent architecture from a LangChain `AgentExecutor`-based model to **LangGraph**.
* **Rationale:** To gain more explicit control over the agent's execution flow, improve state management, enable more robust and responsive task interruption/cancellation, and lay a better foundation for future concurrent task processing (e.g., background tasks).

## Immediate Focus: LangGraph Migration & Core Functionality Refoundation

1.  **CRITICAL: LangGraph Migration - PCEE Workflow Implementation:**
    * **Goal:** Re-implement the core Plan-Code-Execute-Evaluate (PCEE) loop as a stateful graph using LangGraph.
        * Define a clear state schema (likely Pydantic model or TypedDict).
        * Adapt existing components (Intent Classifier, Planner, Controller, Executor logic, Evaluators) to function as nodes within the graph.
        * Implement conditional edges for routing based on state (e.g., intent, step success, retry needed).
    * **Effort & Time Est.:** Significant, forming the bulk of the current development cycle.

2.  **CRITICAL: Robust Task Interruption & Cancellation (within LangGraph):**
    * **Goal:** Ensure that STOP signals from the UI and implicit cancellations (e.g., task switching) can reliably and promptly interrupt the execution of a LangGraph instance.
    * **Approach:**
        * Manage the `asyncio.Task` running the LangGraph execution.
        * Utilize LangGraph's `interrupt` features if applicable between node transitions.
        * Explore passing `asyncio.Event` or similar signals into graph execution for more granular checks within nodes or long-running tools (if tools are refactored to accept them).
        * Ensure `asyncio.CancelledError` is handled gracefully throughout the graph execution.
    * **Effort & Time Est.:** High (intertwined with migration).

3.  **HIGH: Re-Verification of Core Features on LangGraph:**
    * **Tool Integration:** Ensure the "Plug and Play" tool system (`tool_config.json`, `tool_loader.py`) integrates smoothly with LangGraph nodes.
    * **State Persistence (Checkpointing):** Implement and leverage LangGraph's checkpointing for saving and resuming graph state, which is crucial for long-running tasks and future backgrounding.
    * **LLM Integration:** Confirm seamless operation with Gemini and Ollama via LangChain LLM wrappers within graph nodes.
    * **UI Feedback:** Ensure LangGraph's streaming capabilities (e.g., `astream_events`) are used to provide rich, real-time updates to the WebSocket UI (step status, thoughts, tool calls, final answers).
    * **Artifact Viewer Refresh:** Re-integrate or re-design the mechanism for triggering artifact viewer updates based on graph events or state changes indicating file writes.
    * **Step Evaluation & Retry Logic:** Implement the retry loop for plan steps as part of the graph's conditional logic, guided by the Step Evaluator node.
    * **Token Usage Tracking:** Adapt callback handlers to correctly track token usage from LLM calls made within graph nodes.

4.  **HIGH: Comprehensive Testing of the New LangGraph Architecture.**

## Future Brainstorming / Enhanced Capabilities (Post-LangGraph Stability)

**Leveraging LangGraph's Strengths:**

1.  **True Asynchronous Background Task Processing:**
    * With robust state management and checkpointing in LangGraph, implement the ability for a user to switch tasks, allowing the previous agent's graph execution to continue in the background (up to a defined limit, e.g., one background task).
    * Requires UI indicators for background tasks and robust message filtering in the frontend.

2.  **More Complex Agentic Behaviors:**
    * **Advanced Self-Correction Loops:** LangGraph's cyclical nature is ideal for more sophisticated self-correction where the agent can loop back to earlier stages (e.g., re-plan, re-validate) based on evaluation.
    * **Multi-Actor Agents:** Explore scenarios where different specialized LangGraph agents (or sub-graphs) collaborate.
    * **Persistent Agent State / Memory:** Use checkpointing to allow agents to "remember" their state across user sessions or server restarts for specific long-running research tasks.

3.  **Sophisticated Human-in-the-Loop (HITL) Workflows:**
    * Beyond plan confirmation, use LangGraph's interrupt feature to explicitly request human input or approval at critical junctures within a plan's execution.

4.  **Comprehensive Tool Ecosystem Expansion:**
    * The more robust state and flow control of LangGraph should make integrating and managing a wider array of complex tools more feasible. All previously listed tools (Rscript, Data Analysis, Bioinformatics specific) remain relevant here.

5.  **UI/UX & Workspace Enhancements:**
    * Develop a more visual representation of the LangGraph's execution progress in the UI.
    * Integrated Folder Viewer.

## Open Questions / Areas for Investigation with LangGraph

* **Optimal Granularity for Graph Nodes:** How finely should the PCEE loop be broken down into LangGraph nodes for the best balance of control, observability, and complexity?
* **Error Handling and Resilience within Graph Edges:** Best practices for defining conditional edges that handle unexpected errors or tool failures gracefully.
* **Performance/Latency:** Monitor the performance of LangGraph execution, especially with checkpointing enabled.
* **Cancellation Signal Propagation:** Deepest level of cancellation achievable within LangGraph nodes that make external calls (e.g., how to best make tools themselves truly cancellable if they involve long, uninterruptible SDK calls).
* **Managing State for Concurrent Sessions:** If multiple users are active, how to ensure clean isolation and efficient management of their respective LangGraph instances and checkpoints.

This shift to LangGraph is a significant architectural decision aimed at building a more robust, controllable, and future-proof ResearchAgent.
