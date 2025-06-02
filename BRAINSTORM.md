# BRAINSTORM.md - ResearchAgent Project (LangGraph Migration & Future Vision)

This document tracks user feedback and brainstorming ideas for the ResearchAgent project, focusing on the transition to LangGraph and future capabilities.

**Current Version & State (Targeting V2.6.0 - LangGraph PCEE Loop Functional in Tests)**

**Recent Key Advancements:**
* **LangGraph Migration - Core PCEE Loop Achieved:** The primary Plan-Code-Execute-Evaluate (PCEE) workflow, including Intent Classification, Planning, Controller, Executor, Step Evaluator, iterative step processing, and Overall Evaluator, has been successfully implemented and tested as a stateful graph using LangGraph. [cite: 1] This provides explicit control over the agent's execution flow, improved state management, and a solid foundation for robust interruption/cancellation. [cite: 428, 461]
* "Plug and Play" Tool System: Dynamic tool loading via `tool_config.json` remains functional and integrated. [cite: 431, 457]
* Pydantic v2 Migration: Largely completed for core data models, LangGraph state, and tool arguments. [cite: 432, 458]
* UI/UX Enhancements: Previous improvements to chat, plan proposal, token tracking, file uploads are in place. [cite: 433, 459]

**Immediate Focus: Stabilizing LangGraph Agent & Full Application Integration**

1.  **Robust Retry Logic within LangGraph Loop:** [cite: 1]
    * **Goal:** Enable the agent to intelligently retry failed plan steps.
    * **Approach:** Enhance the `ControllerNode` to use feedback from the `StepEvaluatorNode` (e.g., `step_evaluation_suggested_tool`, `step_evaluation_suggested_input_instructions`) when a step is marked as recoverable. Manage `retry_count_for_current_step`.

2.  **Full Integration with Main Application (`server.py` & UI):** [cite: 1]
    * **Goal:** Replace the old agent execution flow with the new LangGraph-based agent.
    * **Approach:**
        * Refactor `backend/message_processing/agent_flow_handlers.py` to invoke the compiled `research_agent_graph`.
        * Ensure seamless communication via WebSockets, passing the `WebSocketCallbackHandler` to the graph for real-time UI updates (e.g., using `astream_events`).
        * Handle `task_id` and session-specific LLM configurations through the graph's initial state or `RunnableConfig`.

3.  **Implement `DIRECT_QA` Path in LangGraph:** [cite: 1]
    * **Goal:** Efficiently handle simple questions that don't require full planning.
    * **Approach:** Add a `DirectQANode` that is triggered if `classified_intent` is "DIRECT\_QA". This node will use an LLM to generate a direct answer, possibly using a simple search tool if needed. Its output can then flow to the `OverallEvaluatorNode` or directly to `END`.

4.  **Refine Tool Loading and Availability in Graph Nodes:** [cite: 1]
    * **Goal:** Optimize tool loading if current dynamic loading in each node proves inefficient.
    * **Approach:** Investigate passing tool instances/summaries via graph state or loading them once per graph invocation if toolset is static for the run.

5.  **Enhance Error Handling and Overall Agent Robustness:** [cite: 1]
    * **Goal:** Make the agent more resilient to unexpected errors.
    * **Approach:** Implement more granular error catching within nodes, define clear error states, and ensure the graph can gracefully go to the `OverallEvaluatorNode` or `END` upon critical failures, providing useful feedback to the user.

6.  **CRITICAL Re-focus: Robust Task Interruption & Cancellation (within LangGraph):**
    * **Goal:** Achieve reliable and prompt stopping/pausing of agent operations. [cite: 464]
    * **Approach:** Now that the graph loop is functional, actively implement and test LangGraph's `interrupt` features, `asyncio.Task.cancel()` on graph execution tasks, and ensure cooperative cancellation within custom nodes/tools. [cite: 465] This is a primary objective of the LangGraph migration.

7.  **Comprehensive Testing of the Integrated LangGraph Architecture.** [cite: 449, 467]

## Future Brainstorming / Enhanced Capabilities (Post-LangGraph Stability)

**Leveraging LangGraph's Strengths:**

1.  **Key Strategic Enhancement: `PythonSandboxTool` (CodeAct-Inspired Integration)**
    * **Concept:** Introduce a powerful, specialized tool within the LangGraph framework that allows the agent to generate and execute Python code on-the-fly to handle complex or novel sub-tasks. [cite: 449, 468, 577] This aligns with the user's enthusiasm for "Idea 1." [cite: 468]
    * **Mechanism:**
        * Invocation: LangGraph Controller node decides to use `PythonSandboxTool`. [cite: 468]
        * Input: Natural language description of the sub-task, relevant context. [cite: 469, 578]
        * Internal LLM (Code Generation): Translates sub-task to Python script. [cite: 470, 579]
        * Sandboxed Execution: Secure, isolated environment (e.g., Docker, restricted interpreter) with controlled library/file access. [cite: 471, 472, 473, 474, 580] May allow calling other atomic tools. [cite: 474]
        * Output: Script output (stdout, stderr), generated files, success/error status. [cite: 475, 581]
    * **Benefits:** Massive flexibility, reduced need for hyper-specific tools, leverages specialized coding models, enhanced problem-solving. [cite: 476, 477, 478, 479, 582, 583]
    * **Challenges & Considerations:** Sandbox security, reliability of generated code (potential for LangGraph-based debug loops), prompt engineering, resource management, observability. [cite: 480, 481, 482, 483, 484, 485, 584]
2.  **True Asynchronous Background Task Processing (Enabled by LangGraph's State/Checkpointing).** [cite: 451, 486, 576]
3.  **More Complex Agentic Behaviors (Advanced Self-Correction, Multi-Actor Agents).** [cite: 452, 486]
4.  **Persistent Agent State / Memory (across sessions for long-running tasks).** [cite: 486]
5.  **Sophisticated Human-in-the-Loop (HITL) Workflows.** [cite: 486]
6.  **Further Tool Ecosystem Expansion (Rscript, specialized DB queries, etc.).** [cite: 453, 486]

## Open Questions / Areas for Investigation with LangGraph & CodeAct Hybrid
* Design of the `PythonSandboxTool`'s internal LLM prompting.
* Security and isolation for the Python sandbox.
* Error handling and debugging for LLM-generated code by `PythonSandboxTool`. [cite: 483]
* Context passing to `PythonSandboxTool` for relevant code generation. [cite: 487]
* Interaction between LangGraph cancellation and sandbox execution. [cite: 488]

This hybrid approach, with LangGraph as the backbone and a CodeAct-inspired `PythonSandboxTool` as a key capability, seems like a very promising direction to achieve the desired power, flexibility, and control. [cite: 489]