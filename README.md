# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 In Progress)

This project provides a functional user interface and backend for an AI agent system designed to assist with research tasks, particularly in fields like bioinformatics and epidemiology. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend. The core agent architecture is currently being migrated to **LangGraph** for enhanced control and capabilities.

**Current Version & Focus (Targeting v2.6.0 - LangGraph Migration & Core Stability):**

We are actively developing the "ResearchAgent" project, with the primary focus on migrating the backend agent architecture from a LangChain `AgentExecutor`-based system to **LangGraph**.

* **Previous Architecture (LangChain `AgentExecutor`-based):**
    * Successfully implemented a "Plug and Play" tool system with dynamic loading via `tool_config.json`.
    * Largely completed Pydantic v2 migration for data models.
    * Developed and refined a Plan-Code-Execute-Evaluate (PCEE) agent workflow.
    * Achieved significant UI/UX refinements with a modularized frontend.

* **LangGraph Migration Progress (Key Achievements):**
    * **Stateful PCEE Graph Functional in Isolation:** The core Plan-Code-Execute-Evaluate (PCEE) workflow, including nodes for Intent Classification, Planning, Control (tool/LLM selection), Execution (tool use or direct LLM call), Step Evaluation, and Overall Evaluation, has been successfully implemented as a stateful graph using LangGraph.
    * **Retry Logic Verified in Isolation:** The graph's retry mechanism, including correct incrementing of retry counts per step and adherence to a maximum retry limit (`MAX_STEP_RETRIES`), has been tested and confirmed to be working as expected in isolated test runs of the `langgraph_agent.py` script.
    * **Dynamic Tool Loading:** The "Plug and Play" tool system is integrated into the LangGraph nodes, allowing dynamic loading and use of tools based on `tool_config.json`.
    * **Pydantic v2 for State:** LangGraph state schemas and tool models utilize Pydantic v2.

* **UI (Frontend):**
    * The existing three-panel layout and UI functionalities (task management, chat, monitor, artifact viewer, LLM selection, file uploads) are in place but are **not yet connected** to the new LangGraph backend.

* **Backend (Targeting LangGraph Integration):**
    * Python server using WebSockets (`websockets`) and `aiohttp` (for file server).
    * LangChain continues to be used for LLM integrations and tool primitives within the LangGraph context.
    * Task-specific workspaces and SQLite for persistence are defined.

## Core Architecture & Workflow (Transitioning to LangGraph)

The ResearchAgent aims to process user queries through a sophisticated LangGraph-based pipeline:
1.  **Intent Classification**: Determines if a query requires direct answering or multi-step planning.
2.  **Planning (if required)**: An LLM-based Planner node generates a sequence of steps.
3.  **Iterative Step Execution Loop (Controller, Executor, Step Evaluator):**
    * **Controller Node**: Validates the current step, selects the appropriate tool (or "None" for direct LLM action), and formulates the precise input, considering previous step outputs and retry feedback.
    * **Executor Node**: Executes the chosen tool or performs a direct LLM call.
    * **Step Evaluator Node**: Assesses if the step achieved its goal. If not, it determines recoverability and provides suggestions for a retry (new tool, revised input). The graph loops back to the Controller for retries, respecting `MAX_STEP_RETRIES`.
4.  **Overall Evaluation**: After all steps are completed, or if an unrecoverable error/max retries are reached, an Overall Evaluator node assesses the plan's success in addressing the user's query.

For more detailed information on the P-C-E-E pipeline and task flow, please refer to `BRAINSTORM.md`. For future development plans, see `ROADMAP.md`.

## Key Current Capabilities & Features

(Largely same as v2.5.1, with backend transitioning)
1.  **UI & User Interaction:**
    * Task Management with persistent storage and reliable UI updates.
    * Chat Interface with Markdown rendering and input history.
    * Role-Specific LLM Selection.
    * Monitor Panel for structured agent logs.
    * Artifact Viewer for text/image/PDF outputs with live updates.
    * Token Usage Tracking.
    * File upload capability to task workspaces.
2.  **Backend Architecture & Logic (Transitioning):**
    * Modular Python backend (refactored `message_handlers`).
    * **LangGraph** for the core PCEE pipeline (functional in isolation, integration in progress).
    * Task-specific, isolated workspaces with persistent history (SQLite).
3.  **Tool Suite (`backend/tools/` & `tool_config.json`):**
    * Dynamically loaded tools including web search (Tavily, DuckDuckGo), web page reader, file I/O, PubMed search, Python REPL, and a multi-step `DeepResearchTool`.

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
* **UI for Plan Confirmation:** The backend (old flow) sends `propose_plan_for_confirmation` which the frontend does not yet handle. This will be revisited after LangGraph integration.
* **Chat Clutter:** Intermediate agent thoughts and tool outputs can still make the chat verbose. This will be managed via the `WebSocketCallbackHandler` during LangGraph integration.
* **Agent Cancellation (STOP Button):** While the LangGraph architecture supports robust interruption, its integration with the UI's STOP button needs to be fully implemented and tested via `server.py`.

## Security Warnings
(No changes - these remain the same)

## Next Steps & Future Perspectives (v2.6.0 Focus)

The **immediate and critical focus** is the complete integration of the successfully tested standalone LangGraph agent (`langgraph_agent.py`) into the main application server (`server.py` and `agent_flow_handlers.py`). This involves:

1.  **Modifying `backend/message_processing/agent_flow_handlers.py`** to invoke the `research_agent_graph` instead of the legacy `AgentExecutor`.
2.  **Adapting `backend/server.py`** to manage session data and `asyncio.Task`s for LangGraph executions.
3.  **Ensuring the `WebSocketCallbackHandler`** is correctly used with the graph's `RunnableConfig` for real-time UI updates, potentially adapting the handler for graph-specific events.
4.  **Implementing streaming of graph outputs** (node messages, status updates) to the UI.
5.  **Passing `task_id` and session-specific LLM configurations** into the graph's initial state or `RunnableConfig`.
6.  **Implementing the `DIRECT_QA` path** within the main integrated LangGraph.
7.  **Ensuring robust task interruption/cancellation** for the integrated LangGraph via the UI's STOP button and `asyncio.Task.cancel()`.

Once the LangGraph agent is fully integrated and stable within the main application:
* Refine tool loading and availability within the integrated graph.
* Enhance error handling and overall system robustness.
* Conduct comprehensive UI testing of the new architecture.

For a detailed, evolving roadmap and ongoing brainstorming, please see **`ROADMAP.md`** and **`BRAINSTORM.md`**.
