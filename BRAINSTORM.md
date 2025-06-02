# BRAINSTORM.md - ResearchAgent Project (LangGraph Migration & Future Vision)

This document tracks user feedback and brainstorming ideas for the ResearchAgent project, focusing on the transition to LangGraph and future capabilities.

## Current Version & State (Targeting V1.0 Go-Live - Embarking on LangGraph Migration)

**Recent Key Advancements (Prior to LangGraph Decision):**
* "Plug and Play" Tool System: Dynamic tool loading via `tool_config.json` achieved.
* Pydantic v2 Migration: Largely completed for core data models and tool arguments.
* UI/UX Enhancements: Significant improvements to chat, plan proposal, token tracking, file uploads.

**Current Strategic Pivot:**
* **Migrating to LangGraph:** The project is shifting its core agent architecture to LangGraph.
* **Rationale:** To gain explicit control over the agent's execution flow, improve state management, enable more robust task interruption/cancellation, and lay a better foundation for future concurrent task processing and advanced agent capabilities like dynamic code execution.

## Immediate Focus: LangGraph Migration & Core Functionality Refoundation

1.  **CRITICAL: LangGraph Migration - PCEE Workflow Implementation:**
    * **Goal:** Re-implement the core Plan-Code-Execute-Evaluate (PCEE) loop as a stateful graph using LangGraph.
    * **Details:** Define state schema; adapt components (Intent Classifier, Planner, Controller, Executor logic, Evaluators) as graph nodes; implement conditional edges.
    * **Effort & Time Est.:** Significant, primary focus.

2.  **CRITICAL: Robust Task Interruption & Cancellation (within LangGraph):**
    * **Goal:** Achieve reliable and prompt stopping/pausing of agent operations.
    * **Approach:** Utilize LangGraph's `interrupt` features, `asyncio.Task.cancel()` on graph execution tasks, and ensure cooperative cancellation within custom nodes/tools.

3.  **HIGH: Re-Verification of Core Features on LangGraph:**
    * Tool Integration (Plug and Play), State Checkpointing, LLM Integration, UI Feedback (via `astream_events`), Artifact Viewer Refresh, Step Evaluation & Retry Logic, Token Usage Tracking.

4.  **HIGH: Comprehensive Testing of the New LangGraph Architecture.**

## Future Brainstorming / Enhanced Capabilities (Post-LangGraph Stability)

**Leveraging LangGraph's Strengths:**

1.  **Key Strategic Enhancement: `PythonSandboxTool` (CodeAct-Inspired Integration)**
    * **Concept:** Introduce a powerful, specialized tool within the LangGraph framework that allows the agent to generate and execute Python code on-the-fly to handle complex or novel sub-tasks. This aligns with the user's enthusiasm for "Idea 1."
    * **Mechanism:**
        * **Invocation:** The LangGraph Controller node would decide to use the `PythonSandboxTool` when a task step is too complex for predefined tools.
        * **Input:** A natural language description of the sub-task, relevant context (e.g., filenames in the workspace).
        * **Internal LLM (Code Generation):** The tool itself would use a dedicated LLM (potentially a strong coding model) to translate the sub-task description into a Python script.
        * **Sandboxed Execution:** The generated script runs in a secure, isolated environment (e.g., Docker container, heavily restricted Python interpreter). The sandbox would have:
            * Controlled access to a curated set of safe Python libraries (e.g., pandas, numpy, matplotlib for data analysis and visualization).
            * Restricted file system access, limited to the current task's workspace.
            * Potentially, an API to call other existing "atomic" ResearchAgent tools in a controlled manner from within the generated script.
        * **Output:** The tool returns the script's output (stdout, stderr), paths to any generated files (like plots or processed data files), or a structured success/error status.
    * **Benefits:**
        * **Massive Flexibility:** Agent can tackle a much wider range of problems by "writing its own solution" for parts of a task.
        * **Reduced Need for Hyper-Specific Tools:** Instead of creating dozens of very granular tools, the agent can generate code for many specific data manipulations or analyses.
        * **Leverages Specialized Models:** Allows using a powerful code generation LLM specifically for the coding sub-task, while other LLMs can handle planning, control, and evaluation.
        * **Enhanced Problem Solving:** Enables the agent to perform more complex, multi-step computations or data transformations as a single "tool call" from the perspective of the main LangGraph.
    * **Challenges & Considerations:**
        * **Sandbox Security:** This is paramount. The sandbox must be extremely robust to prevent malicious or harmful code execution.
        * **Reliability of Generated Code:** LLM-generated code can have bugs. The system might need mechanisms for the agent to test, debug, or refine the generated code (potentially using the iterative loop capabilities of LangGraph for this sub-process).
        * **Prompt Engineering:** Crafting effective prompts for the internal code-generating LLM will be key.
        * **Resource Management:** Code execution can be resource-intensive.
        * **Observability:** Need good logging and insight into what code is generated and how it executes.

2.  **True Asynchronous Background Task Processing (Enabled by LangGraph's State/Checkpointing).**
3.  **More Complex Agentic Behaviors (Advanced Self-Correction, Multi-Actor Agents).**
4.  **Persistent Agent State / Memory (across sessions for long-running tasks).**
5.  **Sophisticated Human-in-the-Loop (HITL) Workflows.**
6.  **Further Tool Ecosystem Expansion (Rscript, specialized DB queries, etc.).**

## Open Questions / Areas for Investigation with LangGraph & CodeAct Hybrid

* **Design of the `PythonSandboxTool`'s internal LLM prompting for reliable and safe code generation.**
* **Security and isolation mechanisms for the Python sandbox.**
* **Error handling and debugging strategies for LLM-generated code executed by the `PythonSandboxTool`.**
* How to best pass context (e.g., available data files, schemas) to the `PythonSandboxTool` for its internal LLM to generate relevant code.
* Interaction between LangGraph's main cancellation mechanisms and long-running code execution within the sandbox tool.

This hybrid approach, with LangGraph as the backbone and a CodeAct-inspired `PythonSandboxTool` as a key capability, seems like a very promising direction to achieve the desired power, flexibility, and control.
