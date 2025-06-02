# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 - LangGraph PCEE Loop Implemented)

This project provides a UI and backend for an AI agent system designed to assist with research tasks. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend.

**Current Strategic Direction (Targeting v2.6.0): Migration to LangGraph for enhanced task control, state management, and robust asynchronous operations. This will also lay the groundwork for advanced capabilities like a CodeAct-inspired Python Sandbox Tool.**

**Recent Developments & Current State:**

-   **Successful LangGraph Migration - Core PCEE Loop:**
    * The core Plan-Code-Execute-Evaluate (PCEE) workflow, including Intent Classification, Planning, Controller logic, Tool/LLM Execution, and per-Step Evaluation, has been successfully re-implemented as a stateful graph using **LangGraph** in a test environment. [cite: 1]
    * The graph now correctly processes a user query, generates a plan, and iterates through each step:
        * **Intent Classification Node:** Determines if a plan is needed.
        * **Planner Node:** Generates a multi-step plan.
        * **Controller Node:** Validates each step, selects tools, and formulates tool inputs.
        * **Executor Node:** Executes the selected tool or a direct LLM call for the current step.
        * **Step Evaluator Node:** Assesses the outcome of each executed step.
        * **Looping Logic:** The graph correctly loops back to the Controller for the next step if the current step is successful.
        * **Overall Evaluator Node:** Assesses the entire plan's outcome after all steps are processed or if an unrecoverable error occurs.
    * This new architecture demonstrates improved state management and clearer control flow.
-   **Previous Architecture (LangChain `AgentExecutor`-based):**
    * Successfully implemented a "Plug and Play" tool system with dynamic loading via `tool_config.json`. [cite: 431, 564]
    * Largely completed Pydantic v2 migration for core data models. [cite: 432, 564]
    * Developed the initial PCEE agent workflow. [cite: 432, 565]
    * Significant UI/UX refinements were made (token counting, file uploads, in-chat tool feedback, plan proposal UI). [cite: 433, 565]
-   **Key Challenge Addressed by LangGraph:** The previous architecture faced difficulties with reliable task interruption/cancellation. LangGraph provides a much stronger foundation to address this. [cite: 434, 567]

**Immediate Next Steps (Focus on Stability & Integration):**

1.  **Implement Robust Retry Logic within the LangGraph Loop:** [cite: 1]
    * Enhance the `ControllerNode` to utilize feedback from the `StepEvaluatorNode` (e.g., `step_evaluation_suggested_tool`, `step_evaluation_suggested_input_instructions`).
    * Manage `retry_count_for_current_step` to attempt recovery for failed but recoverable steps.
2.  **Integrate LangGraph Agent with Main Application (`server.py`):** [cite: 1]
    * Replace the old agent flow in `backend/message_processing/agent_flow_handlers.py` with calls to the new `research_agent_graph`.
    * Ensure the `WebSocketCallbackHandler` is correctly passed into the graph's `RunnableConfig` for real-time UI updates via `astream_events` or `astream`.
    * Manage `task_id` and session-specific LLM configurations through the graph's initial state or `RunnableConfig`.
3.  **Implement `DIRECT_QA` Path in LangGraph:** [cite: 1]
    * If `classified_intent` is "DIRECT\_QA", route to a dedicated `DirectQANode`.
    * This node will use an LLM (potentially with a simple ReAct agent or direct call) to answer straightforward queries.
    * The output will then likely go to the `OverallEvaluatorNode` or `END`.
4.  **Refine Tool Loading and Availability in Nodes:** [cite: 1]
    * Optimize how `get_dynamic_tools()` is called within graph nodes to avoid redundancy if it becomes a performance concern. Consider loading tools once at the graph's start or passing them through the state if appropriate.
5.  **Enhance Error Handling and Overall Robustness:** [cite: 1]
    * Implement more specific error handling within nodes and routing logic.
    * Ensure graceful recovery or reporting for unhandled exceptions during tool execution or LLM calls within the graph.
6.  **Robust Task Interruption & Cancellation on LangGraph (Re-focus):**
    * With the core loop in place, leverage LangGraph's interrupt mechanisms, `asyncio.Task` management, and callback checks to ensure STOP signals and context switches reliably halt or pause graph executions. [cite: 434, 447, 573] This is a core design goal of the migration.
7.  **Comprehensive Testing of the Integrated LangGraph Architecture.** [cite: 449]

**Future Considerations & Enhancements (Post-LangGraph Stability):**

-   **Concurrent Task Processing (Foreground/Background):**
    * LangGraph's state management and checkpointing will be key to implementing the "one active, one background" task model per user session. [cite: 451, 576]
-   **Advanced Agent Reasoning & Self-Correction (Leveraging LangGraph's cyclical capabilities).** [cite: 452, 576]
-   **Comprehensive Tool Ecosystem Expansion:**
    * **Key Planned Feature: `PythonSandboxTool` (CodeAct-Inspired):** [cite: 449, 577]
        * **Concept:** A powerful tool within LangGraph where the agent can request the execution of dynamically generated Python code to solve complex sub-tasks. [cite: 449, 577]
        * **Mechanism:** Input: Natural language description & context. Internal LLM generates Python script. Secure sandboxed execution. Output: Results from script. [cite: 468, 469, 470, 471, 472, 473, 474, 475, 578, 579, 580, 581]
        * **Benefits:** Enhances flexibility, reduces need for granular tools, leverages coding LLMs. [cite: 476, 477, 478, 479, 582, 583]
        * **Considerations:** Sandbox security and reliability of generated code. [cite: 480, 481, 482, 483, 484, 485, 584]
    * Development of other specialized tools (Rscript execution, advanced data analysis, bioinformatics database queries). [cite: 453, 585]
-   **Further UI/UX & Workspace Enhancements (e.g., Visual graph execution monitor, Integrated Folder Viewer).** [cite: 585]
-   **Backend & Architecture (Scalability, Personas).**
-   **Deployment & DevOps.**

## Core Architecture & Workflow (Targeting LangGraph)
The ResearchAgent now employs a stateful, graph-based architecture using LangGraph to manage the Plan-Code-Execute-Evaluate (PCEE) loop, including step-wise evaluation and iteration. [cite: 1, 427, 435, 586]
* **Graph Execution:** The graph execution will stream events (node starts/ends, state changes, LLM tokens) to the backend for UI updates. [cite: 587]
* **Nodes:** Represent components like Intent Classification, Planning, Controller, Tool Execution (including the future `PythonSandboxTool`), Step-wise Evaluation, and Overall Evaluation. [cite: 446, 588]
* **State Management:** A Pydantic model (`ResearchAgentState`) defines the graph's state, updated by nodes and used for conditional routing. Checkpointing will persist this state (future). [cite: 428, 444, 446, 589]
* **Interruption/Cancellation:** Will leverage LangGraph's interrupt mechanisms and `asyncio` task cancellation on the graph execution task. [cite: 447, 590]

## Key Current Capabilities & Features (Partially Re-established/Enhanced on LangGraph)
(Existing UI/UX features and tool functionalities will be adapted and improved on the LangGraph architecture. The core PCEE loop is now functional in tests.)

## Tech Stack
-   **Backend:** Python, **LangGraph**, LangChain (for LLM integrations, tool primitives), WebSockets (`websockets` library), `aiohttp` (for file server), SQLite. [cite: 439, 440, 441, 442, 591]
-   **LLM Support:** Google Gemini, Ollama. [cite: 441, 591]
-   **Frontend:** HTML, CSS, JavaScript (Modular). [cite: 592]
-   **Containerization:** Docker, Docker Compose. [cite: 592]

## Project Structure
(CSS and JS file descriptions updated for clarity on recent enhancements)

```
ResearchAgent/
├── .env # Environment variables (GITIGNORED)
├── .env.example # Example environment variables
├── .gitignore
├── backend/
│ ├── init.py
│ ├── agent.py # Creates ReAct agent executor (OLD - to be replaced/removed)
│ ├── callbacks.py # WebSocket and DB logging callbacks, AgentCancelledException
│ ├── config.py # Application settings
│ ├── controller.py # Controller LLM logic, ControllerOutput Pydantic model
│ ├── db_utils.py # SQLite utilities
│ ├── evaluator.py # Evaluator LLM logic, EvaluationResult/StepCorrection Pydantic models
│ ├── graph_state.py # NEW: Defines the LangGraph state schema
│ ├── intent_classifier.py # Intent Classifier LLM logic, IntentClassificationOutput Pydantic model
│ ├── langgraph_agent.py # NEW: Defines the LangGraph agent, nodes, and compiled graph
│ ├── llm_setup.py # Centralized LLM instantiation
│ ├── message_handlers.py # Main router for WebSocket messages (will call LangGraph agent)
│ ├── message_processing/
│ │ ├── init.py
│ │ ├── agent_flow_handlers.py # Orchestrates PCEE loop (OLD - to be refactored to use LangGraph)
│ │ ├── config_handlers.py
│ │ ├── operational_handlers.py
│ │ └── task_handlers.py
│ ├── planner.py # Planner LLM logic, AgentPlan/PlanStep Pydantic models
│ ├── server.py # Main WebSocket server and aiohttp file server
│ ├── tool_config.json # Central configuration for dynamic tool loading
│ ├── tool_loader.py # Module for loading tools
│ └── tools/
│   ├── init.py
│   ├── standard_tools.py
│   ├── tavily_search_tool.py
│   ├── web_page_reader_tool.py
│   ├── python_package_installer_tool.py
│   ├── pubmed_search_tool.py
│   ├── python_repl_tool.py
│   └── deep_research_tool.py
├── css/
│ └── style.css
├── js/
│ ├── script.js
│ ├── state_manager.js
│ └── ui_modules/
│   ├── artifact_ui.js
│   ├── chat_ui.js
│   ├── file_upload_ui.js
│   ├── llm_selector_ui.js
│   ├── monitor_ui.js
│   ├── task_ui.js
│   └── token_usage_ui.js
├── BRAINSTORM.md
├── Dockerfile
├── docker-compose.yml
├── index.html
├── Pydantic_v1_to_v2_migration_effort.md
├── README.md # This project overview
├── ROADMAP.md
└── UI_UX_style.md

```

Setup & Installation
--------------------

1.  Clone the repository.

2.  Configure environment variables: Copy `.env.example` to `.env` and fill in API keys (GOOGLE_API_KEY, TAVILY_API_KEY, ENTREZ_EMAIL) and any other necessary settings.

3.  Build and run with Docker Compose: `docker compose up --build`

4.  Access: UI at `http://localhost:8000`.

Security Warnings
-----------------

-   Ensure API keys in `.env` are kept secure and not committed to public repositories.

-   Tools like `python_package_installer` and `workspace_shell` execute commands/code in the backend environment; use with caution and be aware of the security implications, especially if exposing the agent to untrusted inputs.

-   Review dependencies for vulnerabilities regularly.

Contributing
------------

(Contributions are welcome! Please discuss significant changes via issues first. Standard PR practices apply.)

License
-------



## Setup Instructions & Running the Application
(No changes)

## Previously Fixed/Implemented in v2.0.0 development cycle:
-   **ENHANCEMENT: In-Chat Tool Feedback & Usability.**
-   **ENHANCEMENT: Chat UI/UX Major Improvements:** Collapsible steps & tool outputs, agent avatar, alignment, widths, font sizes, LLM selector colors, no blue lines on RA/Plan.
-   **BUG FIX: `read_file` output visibility & nesting.**
-   **BUG FIX: Chat scroll jump on expand/collapse.**
-   **BUG FIX: Plan persistence and consistent rendering from history.**
-   **FILE UPLOAD (FIXED).**
-   **TOKEN COUNTER (FIXED & ENHANCED).**

## Security Warnings
(No changes)

## Next Steps & Future Perspectives
The immediate high-priority focus is to **implement robust agent task cancellation and ensure the STOP button is fully functional.** Subsequently, we will address the artifact viewer refresh bug. Longer-term considerations include allowing background task processing. For details, see **`ROADMAP.md`** and **`BRAINSTORM.md`**.

