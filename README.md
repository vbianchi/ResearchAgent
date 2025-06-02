# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 - Transitioning to LangGraph)

This project provides a UI and backend for an AI agent system designed to assist with research tasks. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend.

**Current Strategic Direction (Targeting v2.6.0): Migration to LangGraph for enhanced task control, state management, and robust asynchronous operations.**

**Recent Developments (Leading to current state):**

-   **Previous Architecture (LangChain-based):**
    * Successfully implemented a "Plug and Play" tool system with dynamic loading via `tool_config.json`.
    * Largely completed Pydantic v2 migration for data models.
    * Developed a Plan-Code-Execute-Evaluate (PCEE) agent workflow.
    * Significant UI/UX refinements, including token counting, file uploads, in-chat tool feedback, and plan proposal UI.
-   **Key Challenge Identified:** Reliable and immediate interruption/cancellation of agent tasks, especially long-running LLM calls or tool executions, proved consistently difficult within the previous LangChain `AgentExecutor` model. This also impacted UI responsiveness during task switching.
-   **Strategic Decision: Migrating to LangGraph:** To address these core challenges and enable more sophisticated future capabilities (like true background tasks), the project is now transitioning its backend agent architecture to **LangGraph**. LangGraph's explicit state management, checkpointing, and graph-based control flow are expected to provide better primitives for task lifecycle control.

**Known Issues & Immediate Next Steps (Focus on LangGraph Migration):**

1.  **CRITICAL (MUST HAVE): LangGraph Migration & Core PCEE Re-implementation:**
    * **Goal:** Redesign and implement the existing Plan-Code-Execute-Evaluate (PCEE) workflow using LangGraph's stateful graph architecture.
    * This includes migrating:
        * Intent Classification as an entry point.
        * Planner, Controller, Executor, and Evaluator components as nodes or sub-graphs within LangGraph.
        * Tool integration and execution within graph nodes.
        * Explicit state management for the PCEE loop using Pydantic models or TypedDicts.
    * **Effort & Time:** This will be a significant undertaking, estimated to be the primary focus for the next development cycle.

2.  **HIGH (MUST HAVE - To be addressed *within* the LangGraph migration): Robust Task Interruption & Cancellation:**
    * **Goal:** Leverage LangGraph's interrupt mechanisms (e.g., human-in-the-loop patterns adapted for programmatic stops, `asyncio.Event` checks in streaming operations) and `asyncio.Task.cancel()` on graph execution tasks to achieve reliable and prompt stopping of agent operations.
    * This will be a core design consideration during the LangGraph implementation.

3.  **HIGH (MUST HAVE - Post initial LangGraph migration): Artifact Viewer Refresh:**
    * **Goal:** Ensure the artifact viewer auto-updates reliably after file writes, integrated with LangGraph's state updates or events.

4.  **HIGH (MUST HAVE - Post initial LangGraph migration): Comprehensive Testing on New Architecture:**
    * **Goal:** Develop and adapt unit and integration tests for the new LangGraph-based backend and ensure frontend compatibility.

5.  **MEDIUM (SHOULD HAVE - During/After LangGraph Migration):**
    * **Finalize Pydantic v2 Migration:** Ensure all data models (especially for graph state) are Pydantic v2.
    * **"Plug and Play" Tool System on LangGraph:** Adapt the existing dynamic tool loading for seamless use within LangGraph nodes.
    * **UI/UX Integration with LangGraph:**
        * Ensure real-time UI updates (status, thinking, logs, chat messages) are effectively driven by LangGraph's streaming capabilities (e.g., `astream_events`).
        * Refine "View [artifact] in Artifacts" links.

**Future Considerations & Enhancements (Post-LangGraph Stability):**

-   **Concurrent Task Processing (Foreground/Background):**
    * LangGraph's state management and checkpointing are expected to provide a much stronger foundation for implementing the "one active, one background" task model per user session. This will involve managing separate LangGraph instances and their states.
-   **Advanced Agent Reasoning & Self-Correction (Leveraging LangGraph's cyclical capabilities).**
-   **Comprehensive Tool Ecosystem Expansion.**
-   **Further UI/UX & Workspace Enhancements.**
-   **Backend & Architecture (Scalability, Personas).**
-   **Deployment & DevOps.**

## Core Architecture & Workflow (Targeting LangGraph)
The ResearchAgent will employ a stateful, graph-based architecture using LangGraph to manage the Plan-Code-Execute-Evaluate (PCEE) loop.

1.  **User Input & Task Context:** Remains the same.
2.  **Graph Initialization:** For a given user query, an instance of the main LangGraph will be created, defining the nodes (representing agent components like Planner, Controller, Tools, Executor, Evaluators) and edges (defining transitions based on state).
3.  **State Management:** A Pydantic model or TypedDict will define the graph's state, which will be updated by each node and used for conditional routing. Checkpointing will persist this state.
4.  **Execution Flow (PCEE as a Graph):**
    * The graph execution will stream events (node starts/ends, state changes, LLM tokens) to the backend, which will then be relayed to the UI via WebSockets.
    * **Intent Classification Node:** Determines PLAN vs. DIRECT_QA.
    * **Planner Node:** Generates the plan (list of steps).
    * **Controller Node:** For each step, determines the tool/action.
    * **Executor Node(s)/Tool Node(s):** Executes the action.
    * **Step Evaluator Node:** Evaluates step outcome, enabling retries by routing back to the Controller or Executor with modified state/input.
    * **Overall Evaluator Node:** Assesses final outcome.
    * The graph structure will allow for cycles (e.g., for retries) and conditional branching.
5.  **Interruption/Cancellation:**
    * UI STOP requests will signal the backend to interrupt the specific LangGraph execution (e.g., by cancelling its `asyncio.Task` and/or using LangGraph's interrupt mechanisms if applicable between node transitions).
    * Context switching in the UI will also trigger cancellation/interruption of the graph associated with the previous task (unless backgrounding is implemented).
6.  **Output to User:** Streamed from the graph, with final outputs collated as defined by the graph's end state.

## Key Current Capabilities & Features (To be Re-established/Enhanced on LangGraph)
The goal is to retain and improve upon the existing UI/UX features and tool functionalities within the new LangGraph architecture. This includes task management, chat interface, plan proposal/confirmation, LLM configuration, monitor/artifact viewing, token tracking, and file uploads. The "Plug and Play" tool system will be adapted.

## Tech Stack
-   **Backend:** Python, **LangGraph**, LangChain (for LLM integrations, tool primitives), FastAPI (potentially for API endpoints if needed beyond WebSockets), WebSockets (`websockets` library), `aiohttp` (for file server), SQLite.
-   **LLM Support:** Google Gemini, Ollama.
-   **Frontend:** HTML, CSS, JavaScript (Modular).
-   **Containerization:** Docker, Docker Compose.

## Project Structure
(CSS and JS file descriptions updated for clarity on recent enhancements)
```
ResearchAgent/

├── .env # Environment variables (GITIGNORED)

├── .env.example # Example environment variables

├── .gitignore

├── backend/

│ ├── init.py

│ ├── agent.py # Creates ReAct agent executor

│ ├── callbacks.py # WebSocket and DB logging callbacks, AgentCancelledException

│ ├── config.py # Application settings

│ ├── controller.py # Controller LLM logic, ControllerOutput Pydantic model (migrated to v2)

│ ├── db_utils.py # SQLite utilities

│ ├── evaluator.py # Evaluator LLM logic, EvaluationResult/StepCorrection Pydantic models (migrated to v2)

│ ├── intent_classifier.py # Intent Classifier LLM logic, IntentClassificationOutput Pydantic model (migrated to v2)

│ ├── llm_setup.py # Centralized LLM instantiation

│ ├── message_handlers.py # Main router for WebSocket messages

│ ├── message_processing/ # Sub-package for message processing modules

│ │ ├── init.py

│ │ ├── agent_flow_handlers.py # Orchestrates PCEE loop, planning, direct QA

│ │ ├── config_handlers.py # Handles LLM config messages

│ │ ├── operational_handlers.py# Handles non-agent operational messages

│ │ └── task_handlers.py # Handles task CRUD, context switching

│ ├── planner.py # Planner LLM logic, AgentPlan/PlanStep Pydantic models (migrated to v2)

│ ├── server.py # Main WebSocket server and aiohttp file server

│ ├── tool_config.json # Central configuration for dynamic tool loading

│ ├── tool_loader.py # Module for loading tools from tool_config.json, includes workspace utils

│ └── tools/

│ ├── init.py

│ ├── standard_tools.py # ReadFileTool, WriteFileTool, TaskWorkspaceShellTool classes, helper functions

│ ├── tavily_search_tool.py # TavilyAPISearchTool class & TavilySearchInput (migrated to v2)

│ ├── web_page_reader_tool.py# WebPageReaderTool class & WebPageReaderInput (migrated to v2)

│ ├── python_package_installer_tool.py # PythonPackageInstallerTool & Input (migrated to v2)

│ ├── pubmed_search_tool.py # PubMedSearchTool & PubMedSearchInput (migrated to v2)

│ ├── python_repl_tool.py # PythonREPLTool & PythonREPLInput (migrated to v2)

│ └── deep_research_tool.py # DeepResearchTool & DeepResearchToolInput (Input migrated, internal models updated for v2)

├── css/

│ └── style.css # Main stylesheet

├── js/

│ ├── script.js # Main frontend orchestrator

│ ├── state_manager.js # Manages UI and application state

│ └── ui_modules/ # Modular UI components

│ ├── artifact_ui.js

│ ├── chat_ui.js

│ ├── file_upload_ui.js

│ ├── llm_selector_ui.js

│ ├── monitor_ui.js

│ ├── task_ui.js

│ └── token_usage_ui.js

├── BRAINSTORM.md

├── Dockerfile

├── docker-compose.yml

├── index.html

├── README.md # This project overview

├── ROADMAP.md

├── UI_UX_style.md # UI/UX refinement notes

└── simulation_option6.html # UI simulation/sandbox (archival)
(To be specified - MIT License is a common choice for open-source projects.)

├── README.md                      # This file
├── ROADMAP.md                     # Updated with multi-tasking goals
└── simulation_option6.html

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

