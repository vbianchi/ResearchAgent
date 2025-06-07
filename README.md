# ResearchAgent: AI Assistant for Research Workflows (v2.7 - Reset to Core)

This project provides a functional user interface and backend for an AI agent system designed to assist with research tasks. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend.

## Current State: Reset to a Stable Core

**CURRENT STATUS: FUNCTIONAL (Simplified ReAct Agent)**

Following a period of instability caused by an attempted migration to a complex graph-based architecture, this project has been strategically reset to a simpler, more robust foundation.

The backend now runs a standard **LangChain `ReAct` (Reasoning and Acting) agent**. This agent is capable of conversational interaction and can use a suite of dynamically loaded tools to perform tasks like web searches, file I/O, and code execution. All interactions, including the agent's internal "thoughts" and tool usage, are streamed to the UI in real-time.

This reset provides a stable and understandable baseline from which we will gradually re-introduce more advanced features in a controlled, incremental manner.

## Tech Stack

-   **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+) (Modularized)
-   **Backend:** Python 3.12, **LangChain**, `aiohttp`, `websockets`
-   **Containerization:** Docker, Docker Compose

## Key Capabilities (Current)

-   **UI & User Interaction:**
    -   Task Management with persistent storage (SQLite).
    -   Real-time chat interface with Markdown rendering.
    -   Monitor Panel for viewing the agent's live thought process and tool outputs.
    -   Artifact Viewer for generated text, image, and PDF files.
    -   File upload capability to a dedicated workspace for each task.
-   **Backend Architecture (ReAct Core):**
    -   Modular Python backend with event-driven message handling.
    -   A core `ReAct` agent that can reason and choose from available tools.
    -   Task-specific, isolated workspaces on the file system.
-   **Tool Suite (`backend/tools/` & `tool_config.json`):**
    -   Dynamically loaded tools including web search (Tavily), web page reader, file I/O, PubMed search, and a Python REPL.

## Project Structure (Simplified)

```
ResearchAgent/
├── .env
├── backend/
│   ├── agent.py         # <<< NEW CORE: Creates the ReAct agent executor.
│   ├── callbacks.py     # Handles streaming events to the UI.
│   ├── config.py        # Application settings.
│   ├── db_utils.py      # SQLite utilities for task persistence.
│   ├── llm_setup.py     # Centralized LLM instantiation.
│   ├── server.py        # Main WebSocket and file server.
│   ├── tool_loader.py   # Loads tools from the config file.
│   ├── tool_config.json # Defines the available tools.
│   └── tools/           # Directory containing all tool implementations.
├── css/
│   └── style.css
├── js/
│   └── ... (frontend files)
├── langgraph_pre_reset/ # <<< PARKED: The previous complex graph implementation.
├── README.md            # This project overview.
└── ROADMAP.md           # The new, phased development plan.


# LAST UPDATE
# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 In Progress)

This project provides a functional user interface and backend for an AI agent system designed to assist with research tasks. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend.

## Core Architecture & Workflow (LangGraph)

The agent's logic is structured as a stateful graph using **LangGraph**. This architecture is designed for clear and efficient routing of user requests, robust state management, and controllable, iterative execution of complex tasks.

_(This diagram represents the target architecture and will be updated as the graph is rebuilt)_

## Current State & Next Steps (Project Reset)

**CURRENT STATUS: NON-FUNCTIONAL - UNDERGOING REIMPLEMENTATION**

Recent attempts to implement the full LangGraph-based PCEE (Plan-Code-Execute-Evaluate) workflow resulted in a series of cascading integration errors (`KeyError`, `NameError`, `AttributeError`), leading to an unstable and non-functional state.

**The project is now undergoing a strategic reset.**

We are stripping the agent's logic back to a minimal baseline and will be re-implementing the core components in a phased, incremental approach. Each new piece of functionality will be thoroughly tested and verified before proceeding to the next.

### Immediate Next Steps (New Phased Approach)

1.  **Establish a Minimal Baseline:** Create a minimal LangGraph with a single entry and end point to verify the core plumbing.
2.  **Implement `direct_qa` Path:** Add the simplest agent capability—answering a question without tools—and ensure it works end-to-end.
3.  **Incrementally Build PCEE Loop:** Add and verify each node of the PCEE workflow (`planner`, `controller`, `executor`, `step_evaluator`) one at a time.

This methodical process will ensure we build a robust and reliable agent on the LangGraph foundation.

## Tech Stack

-   **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+) (Modularized)
-   **Backend:** Python 3.12, **LangGraph**, LangChain, `aiohttp`, `websockets`
-   **Containerization:** Docker, Docker Compose

## Project Structure

```
ResearchAgent/
├── .env
├── backend/
│ ├── langgraph_agent.py # Core LangGraph agent definition (UNDER REVISION)
│ ├── prompts.py         # Centralized system prompts for all LLM components
│ ├── pydantic_models.py # Centralized Pydantic data models
│ ├── ... (other backend files)
├── css/
│ └── style.css
├── js/
│ └── ... (frontend files)
├── README.md # This project overview
├── ROADMAP.md # The new, phased development plan
└── BRAINSTORM.md # Design decisions and project log
```

# PREVIOUS UPDATE
# ResearchAgent: AI Assistant for Research Workflows (v2.6.0 In Progress)

This project provides a functional user interface and backend for an AI agent system designed to assist with research tasks, particularly in fields like bioinformatics and epidemiology. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend. The core agent architecture is built on **LangGraph** for enhanced control, statefulness, and capabilities.

## Core Architecture & Workflow (LangGraph)

The agent's logic is structured as a stateful graph, providing clear and efficient routing for user requests. After a user submits a query, it is processed through the following flow:

![Research Agent LangGraph Flow](research_agent_graph_definition.md)
*(This diagram is generated by `backend/visualize_graph.py`)*

1.  **Intent Classification (`intent_classifier`)**: The query is first analyzed to determine the most efficient path:
    * **Direct QA Path**: For simple questions that don't require external tools (e.g., "What is a neural network?"), the graph routes to the `direct_qa` node for an immediate LLM-generated answer.
    * **Planning Path**: For complex queries requiring multiple steps (e.g., "Research X and write a report"), the graph routes to the `planner`.

2.  **Planning (`planner`)**: If required, this node generates a detailed, multi-step plan. If the planner fails to create a valid plan, the graph gracefully routes to the `overall_evaluator` to inform the user.

3.  **PCEE Execution Loop (Plan-Code-Execute-Evaluate)**: For a successful plan, the graph enters an iterative execution loop.
    * **Controller (`controller`)**: Examines the current plan step and decides which tool to use and with what specific input.
    * **Executor (`executor`)**: Executes the action from the Controller, either by running a tool or calling an LLM directly.
    * **Step Evaluator (`step_evaluator`)**: Assesses the outcome of the execution. If a step fails but is recoverable, it routes back to the Controller for a retry (respecting `MAX_STEP_RETRIES`). If successful, it proceeds to the next step.

4.  **Overall Evaluation (`overall_evaluator`)**: All paths converge on this final node, which synthesizes all information to provide a single, coherent answer to the user.

## Key Capabilities & Features

1.  **UI & User Interaction:**
    * Task Management with persistent storage.
    * Chat Interface with Markdown rendering and input history.
    * Role-Specific LLM Selection (Intent, Planner, Controller, Executor, Evaluator).
    * Monitor Panel for structured agent logs.
    * Artifact Viewer for text/image/PDF outputs with live updates.
    * Token Usage Tracking (Overall and Per-Role).
    * File upload capability to task workspaces.
2.  **Backend Architecture & Logic (LangGraph Core):**
    * Modular Python backend with event-driven message handlers.
    * **LangGraph** for core agent workflows, enabling clear routing and state management.
    * Task-specific, isolated workspaces with persistent history (SQLite).
3.  **Tool Suite (`backend/tools/` & `tool_config.json`):**
    * Dynamically loaded tools including web search (Tavily), web page reader, file I/O, PubMed search, Python REPL, and a multi-step `DeepResearchTool`.

## Current State & Progress (v2.6.0)

* **Core Graph Integrated**: The LangGraph agent is fully integrated with the application server. The core routing logic based on `intent_classifier` is functional and has been debugged.
* **Direct QA Path Functional**: The "Express Lane" for simple, no-tool questions is fully operational.
* **Planning Path Initiated**: The graph correctly routes complex queries to the `planner` node, which successfully generates a plan. The UI now correctly receives and displays this plan for user confirmation.
* **Placeholder Nodes**: The core execution nodes (`controller`, `executor`, `step_evaluator`) are currently implemented as placeholders. While the graph can flow through them, they do not yet perform their intended logic.
* **Visualization**: A utility script (`visualize_graph.py`) has been created to generate a visual diagram of the agent's architecture, and the graph structure has been subsequently cleaned up for clarity.

## Immediate Next Steps (CRITICAL)

With the foundation and routing logic now stable, the top priority is to implement the core agent capabilities by replacing the placeholders:

1.  **Implement `ControllerNode`**: Replace the placeholder with logic from `backend/controller.py` to enable intelligent tool selection for each plan step.
2.  **Implement `ExecutorNode`**: Replace the placeholder with logic to execute the tool or LLM call selected by the Controller.
3.  **Implement `StepEvaluatorNode`**: Replace the placeholder with logic from `backend/evaluator.py` to assess step outcomes and manage the retry loop.

## Tech Stack

-   **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+) (Modularized)
-   **Backend:** Python 3.12, **LangGraph**, LangChain, `aiohttp`, `websockets`
-   **Containerization:** Docker, Docker Compose

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
│ ├── visualize_graph.py # generate the graph information to plot the structure of the core logic
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
## Immediate Next Steps (CRITICAL)
With the foundation and routing logic now stable, the top priority is to implement the core agent capabilities:

1.  **Implement `ControllerNode`**: Replace the placeholder with logic from `backend/controller.py` to enable intelligent tool selection for each plan step.
2.  **Implement `ExecutorNode`**: Replace the placeholder with logic to execute the tool or LLM call selected by the Controller.
3.  **Implement `StepEvaluatorNode`**: Replace the placeholder with logic from `backend/evaluator.py` to assess step outcomes and manage the retry loop.

## Tech Stack
-   **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+)
-   **Backend:** Python 3.12, **LangGraph**, LangChain, `aiohttp`, `websockets`
-   **Containerization:** Docker, Docker Compose

## Known Issues

* **PLAN Path Implementation:** The full Plan-Code-Execute-Evaluate (PCEE) loop within LangGraph (using `PlannerNode`, `ControllerNode`, `ExecutorNode`, `StepEvaluatorNode`) is the next major development task and currently uses placeholder nodes in the graph.

* **Intent Classification for "List Tools":** Queries like "which tools do you have?" are sometimes misclassified as `DIRECT_QA` instead of `PLAN`. Further refinement of `intent_classifier.py` prompt is ongoing.

* **UI for Plan Confirmation:** The legacy backend sent `propose_plan_for_confirmation` which the frontend does not yet fully handle. This will be revisited and adapted for the LangGraph-generated plans.

* **Agent Cancellation (STOP Button):** Robust task interruption for the full LangGraph (especially for multi-step plans) needs to be fully implemented and tested.

* **UI for Plan Proposal & Interaction:** Adapt or implement UI elements to display plans generated by the `PlannerNode` and allow user confirmation/modification before execution by the LangGraph.

* **Comprehensive Testing:** Thoroughly test the complete LangGraph-based PCEE workflow.


For a detailed, evolving roadmap and ongoing brainstorming, please see **`ROADMAP.md`** and **`BRAINSTORM.md`**.