# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 - Transitioning to LangGraph)

This project provides a UI and backend for an AI agent system designed to assist with research tasks. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend.

**Current Strategic Direction (Targeting v2.6.0): Migration to LangGraph for enhanced task control, state management, and robust asynchronous operations. This will also lay the groundwork for advanced capabilities like a CodeAct-inspired Python Sandbox Tool.**

**Recent Developments (Leading to current state):**

-   **Previous Architecture (LangChain-based):**
    * Successfully implemented a "Plug and Play" tool system with dynamic loading via `tool_config.json`.
    * Largely completed Pydantic v2 migration for data models.
    * Developed a Plan-Code-Execute-Evaluate (PCEE) agent workflow.
    * Significant UI/UX refinements, including token counting, file uploads, in-chat tool feedback, and plan proposal UI.
-   **Key Challenge Identified:** Reliable and immediate interruption/cancellation of agent tasks, especially long-running LLM calls or tool executions, proved consistently difficult within the previous LangChain `AgentExecutor` model. This also impacted UI responsiveness during task switching.
-   **Strategic Decision: Migrating to LangGraph:** To address these core challenges and enable more sophisticated future capabilities, the project is now transitioning its backend agent architecture to **LangGraph**. LangGraph's explicit state management, checkpointing, and graph-based control flow are expected to provide better primitives for task lifecycle control.

**Known Issues & Immediate Next Steps (Focus on LangGraph Migration):**

1.  **CRITICAL (MUST HAVE): LangGraph Migration & Core PCEE Re-implementation:**
    * **Goal:** Redesign and implement the existing Plan-Code-Execute-Evaluate (PCEE) workflow using LangGraph's stateful graph architecture.
    * This includes migrating: Intent Classification, Planner, Controller, Executor logic, and Evaluators as nodes/sub-graphs. Tool integration within graph nodes. Explicit state management.
    * **Effort & Time:** Primary focus for the current development cycle.

2.  **HIGH (MUST HAVE - To be addressed *within* the LangGraph migration): Robust Task Interruption & Cancellation:**
    * **Goal:** Leverage LangGraph's mechanisms and `asyncio` best practices for reliable and prompt stopping of agent operations.

3.  **HIGH (MUST HAVE - Post initial LangGraph migration): Artifact Viewer Refresh & Comprehensive Testing on New Architecture.**

4.  **MEDIUM (SHOULD HAVE - During/After LangGraph Migration):**
    * Finalize Pydantic v2 Migration.
    * Adapt "Plug and Play" Tool System for LangGraph.
    * Ensure rich UI feedback via LangGraph's streaming.

**Future Considerations & Enhancements (Post-LangGraph Stability):**

-   **Concurrent Task Processing (Foreground/Background):**
    * LangGraph's state management and checkpointing will be key to implementing the "one active, one background" task model per user session.
-   **Advanced Agent Reasoning & Self-Correction (Leveraging LangGraph's cyclical capabilities).**
-   **Comprehensive Tool Ecosystem Expansion:**
    * **Key Planned Feature: `PythonSandboxTool` (CodeAct-Inspired):**
        * **Concept:** A powerful tool within LangGraph where the agent can request the execution of dynamically generated Python code to solve complex sub-tasks.
        * **Mechanism:**
            * Input: Natural language description of the sub-task & relevant context (e.g., filenames).
            * Internal LLM (specialized for coding): Generates Python script to achieve the sub-task.
            * Execution: Runs the generated script in a secure, isolated sandbox environment (e.g., Docker container or restricted Python interpreter) with controlled access to approved libraries, the task's workspace, and potentially other basic/safe tools via an API.
            * Output: Results from the script execution (stdout, stderr, created/modified files, structured data).
        * **Benefits:** Greatly enhances agent flexibility, allowing it to tackle novel problems not covered by predefined tools; reduces the need for numerous granular tools; allows for the use of highly capable coding LLMs for specific parts of a task.
        * **Considerations:** Sandbox security and reliability of LLM-generated code are paramount.
    * Development of other specialized tools (Rscript execution, advanced data analysis, bioinformatics database queries) will also be pursued.
-   **Further UI/UX & Workspace Enhancements (e.g., Visual graph execution monitor, Integrated Folder Viewer).**
-   **Backend & Architecture (Scalability, Personas).**
-   **Deployment & DevOps.**

## Core Architecture & Workflow (Targeting LangGraph)
The ResearchAgent will employ a stateful, graph-based architecture using LangGraph to manage the Plan-Code-Execute-Evaluate (PCEE) loop.
* **Graph Execution:** The graph execution will stream events (node starts/ends, state changes, LLM tokens) to the backend for UI updates.
* **Nodes:** Will represent components like Intent Classification, Planning, Controller, Tool Execution (including the future `PythonSandboxTool`), and Evaluation.
* **State Management:** A Pydantic model will define the graph's state, updated by nodes and used for conditional routing. Checkpointing will persist this state.
* **Interruption/Cancellation:** Will leverage LangGraph's interrupt mechanisms and `asyncio` task cancellation on the graph execution task.

## Key Current Capabilities & Features (To be Re-established/Enhanced on LangGraph)
(Existing UI/UX features and tool functionalities will be adapted and improved on the LangGraph architecture.)

## Tech Stack
-   **Backend:** Python, **LangGraph**, LangChain (for LLM integrations, tool primitives), WebSockets (`websockets` library), `aiohttp` (for file server), SQLite.
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

