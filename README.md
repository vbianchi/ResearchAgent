# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 In Progress)

This project provides a functional user interface and backend for an AI agent system designed to assist with research tasks, particularly in fields like bioinformatics and epidemiology. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend. The core agent architecture is built on **LangGraph** for enhanced control and capabilities.

**Current Version & Focus (v2.6.0 - LangGraph Core Integration & Direct Execution Paths)**

We are actively developing the "ResearchAgent" project. The primary focus of v2.6.0 has been the successful integration of the LangGraph agent into the main application server, enabling robust direct question answering and direct tool execution capabilities.

* **LangGraph Integration - Key Achievements:**
    * **Server Integration Complete:** The LangGraph agent is now integrated with the main application server (`server.py` and `agent_flow_handlers.py`).
    * **`DIRECT_QA` Path Functional:** User queries classified as "DIRECT_QA" (simple, no-tool questions) are now processed through a dedicated path in the LangGraph, with the `DirectQANode` providing an answer via an LLM, and the `OverallEvaluatorNode` presenting this answer to the user.
    * **`DIRECT_TOOL_REQUEST` Path Functional:** User queries explicitly requesting a specific tool (e.g., "web search for X", "read file Y") are classified as "DIRECT_TOOL_REQUEST". A `DirectToolExecutorNode` in the LangGraph executes the identified tool with the extracted input, and the `OverallEvaluatorNode` presents the result. This has been tested with tools like `tavily_search_api`, `pubmed_search`, `workspace_shell`, and `read_file`.
    * **Stateful Graph Execution:** The system utilizes LangGraph's state management (`ResearchAgentState`) to pass information between nodes (e.g., user query, classified intent, tool outputs).
    * **Callback Integration:** The `WebSocketCallbackHandler` is correctly integrated with the graph's execution, streaming LLM token usage and final agent messages to the UI in real-time for the direct paths.
    * **Dynamic Tool Loading for Direct Requests:** The `DirectToolExecutorNode` dynamically loads and uses tools based on `tool_config.json` and the current `task_id`.
    * **Pydantic v2 for State & Tool Models:** LangGraph state schemas and most tool models utilize Pydantic v2.

* **UI (Frontend):**
    * The existing three-panel layout and UI functionalities (task management, chat, monitor, artifact viewer, LLM selection, file uploads) are in place and correctly interact with the new LangGraph backend for direct QA and direct tool requests.

* **Backend (LangGraph Architecture):**
    * Python server using WebSockets (`websockets`) and `aiohttp` (for file server).
    * LangGraph is the core for agentic workflows.
    * LangChain is used for LLM integrations and tool primitives within the LangGraph context.
    * Task-specific workspaces and SQLite for persistence are functional.

## Core Architecture & Workflow (LangGraph Based)

The ResearchAgent processes user queries through a LangGraph pipeline:

1.  **Intent Classification (`IntentClassifierNode` context in `agent_flow_handlers.py`):**
    * Determines if a query requires:
        * `DIRECT_QA`: Simple, no-tool answer.
        * `DIRECT_TOOL_REQUEST`: Explicit request for a specific tool.
        * `PLAN`: Complex, multi-step processing.
    * For `DIRECT_TOOL_REQUEST`, it also attempts to identify the tool name and extract its input.

2.  **LangGraph Execution (initiated from `agent_flow_handlers.py`):**
    * **Entry Point (`intent_classifier_node` in graph):** Receives the pre-classified intent and related data.
    * **Conditional Routing:**
        * If `DIRECT_QA`: Routes to `direct_qa_node` (uses LLM for direct answer).
        * If `DIRECT_TOOL_REQUEST`: Routes to `direct_tool_executor_node` (loads and runs the specified tool).
        * If `PLAN` (Future Work for Full PCEE): Will route to `planner_node`.
    * **Output Processing (`overall_evaluator_node`):**
        * Receives output from `direct_qa_node` or `direct_tool_executor_node`.
        * Uses an LLM to format/present this output as the final agent message to the user.
        * Handles large file content from `read_file` by truncation and specific prompting.

3.  **Iterative Step Execution Loop (For `PLAN` intent - Next Major Development Phase):**
    * `PlannerNode`: Generates a sequence of steps. (Currently placeholder)
    * `ControllerNode`: Validates the current step, selects tools/LLMs. (Currently placeholder)
    * `ExecutorNode`: Executes tools or direct LLM calls for a plan step. (Currently placeholder)
    * `StepEvaluatorNode`: Assesses step outcome, suggests retries. (Currently placeholder)
    * This loop will feed into the `OverallEvaluatorNode`.

For more detailed information on the P-C-E-E pipeline vision and task flow, please refer to `BRAINSTORM.md`. For future development plans, see `ROADMAP.md`.

## Key Current Capabilities & Features

1.  **UI & User Interaction:**
    * Task Management with persistent storage.
    * Chat Interface with Markdown rendering and input history.
    * Role-Specific LLM Selection (Intent, Planner, Controller, Executor, Evaluator).
    * Monitor Panel for structured agent logs.
    * Artifact Viewer for text/image/PDF outputs with live updates.
    * Token Usage Tracking (Overall and Per-Role).
    * File upload capability to task workspaces.
2.  **Backend Architecture & Logic (LangGraph Core):**
    * Modular Python backend.
    * **LangGraph** for core agent workflows:
        * Functional `DIRECT_QA` path.
        * Functional `DIRECT_TOOL_REQUEST` path with dynamic tool loading.
    * Task-specific, isolated workspaces with persistent history (SQLite).
3.  **Tool Suite (`backend/tools/` & `tool_config.json`):**
    * Dynamically loaded tools including web search (Tavily), web page reader, file I/O, PubMed search, Python REPL, and a multi-step `DeepResearchTool`. These are usable via `DIRECT_TOOL_REQUEST`.

## Tech Stack

-   **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+) (Modularized)
-   **Backend:** Python 3.10+ (3.12 in Docker), **LangGraph**, LangChain, `aiohttp`, `websockets`.
-   **Containerization:** Docker, Docker Compose.

## Project Structure


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
## Setup Instructions & Running the Application
(No changes from v2.5.1 - these remain the same for now until LangGraph integration impacts server startup)

## Known Issues
* **PLAN Path Implementation:** The full Plan-Code-Execute-Evaluate (PCEE) loop within LangGraph (using `PlannerNode`, `ControllerNode`, `ExecutorNode`, `StepEvaluatorNode`) is the next major development task and currently uses placeholder nodes in the graph.
* **Intent Classification for "List Tools":** Queries like "which tools do you have?" are sometimes misclassified as `DIRECT_QA` instead of `PLAN`. Further refinement of `intent_classifier.py` prompt is ongoing.
* **UI for Plan Confirmation:** The legacy backend sent `propose_plan_for_confirmation` which the frontend does not yet fully handle. This will be revisited and adapted for the LangGraph-generated plans.
* **Agent Cancellation (STOP Button):** Robust task interruption for the full LangGraph (especially for multi-step plans) needs to be fully implemented and tested.


## Next Steps & Future Perspectives (v2.6.0 Focus)

With `DIRECT_QA` and `DIRECT_TOOL_REQUEST` paths functional via LangGraph:

1.  **CRITICAL: Implement Full PCEE Loop in LangGraph:**
    * Replace placeholder nodes (`planner`, `controller`, `executor`, `step_evaluator`) in `backend/langgraph_agent.py` with their actual implementations, incorporating LLM calls, tool execution logic, and state updates for iterative plan processing.
    * This includes integrating the existing Pydantic models for `AgentPlan`, `ControllerOutput`, `StepCorrectionOutcome`, etc., into these nodes.
    * Ensure the robust retry logic (previously tested in isolation) is correctly implemented within the integrated `StepEvaluatorNode` and `ControllerNode`.
2.  **Refine `process_execute_confirmed_plan`:** Update this function in `agent_flow_handlers.py` to correctly prepare the initial state for a `PLAN` intent and invoke the `research_agent_graph.astream_events()` for full plan execution.
3.  **Refine Intent Classification:** Continue improving the `intent_classifier.py` prompt to better distinguish `PLAN` intents, especially for meta-queries about agent capabilities.
4.  **Robust Task Interruption & Cancellation:** Implement and test for the full LangGraph execution flow.
5.  **UI for Plan Proposal & Interaction:** Adapt or implement UI elements to display plans generated by the `PlannerNode` and allow user confirmation/modification before execution by the LangGraph.
6.  **Comprehensive Testing:** Thoroughly test the complete LangGraph-based PCEE workflow.

For a detailed, evolving roadmap and ongoing brainstorming, please see **`ROADMAP.md`** and **`BRAINSTORM.md`**.
