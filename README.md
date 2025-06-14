# ResearchAgent: An Advanced AI-Powered Research Assistant

## 1. Overview

ResearchAgent is a sophisticated, AI-powered agent designed to handle complex, multi-step tasks in software engineering and scientific research. Built with a Python/LangGraph backend and a modern JavaScript frontend (Vite + Preact), it leverages a unique **Plan-Controller-Executor-Evaluator (PCEE)** architecture to autonomously create, execute, and evaluate structured plans to fulfill high-level user requests.

The core philosophy is built on **transparency, adaptive execution, and security**. The agent first uses a `Router` to determine if a request requires a simple answer or a complex plan. For complex tasks, it generates a detailed blueprint and executes it step-by-step within a secure, sandboxed workspace for each persistent task. It is designed to understand the outcome of its actions and has a foundational architecture for future self-correction and human-in-the-loop collaboration.

## 2. Key Features

-   **Stateful, Multi-Turn Tasks:** The application is built around persistent tasks. Users can create, rename, delete, and switch between tasks, with each one maintaining its own independent chat history and sandboxed workspace.
-   **Advanced PCEE Architecture:** A robust, multi-node graph that separates routing, planning, control, execution, and evaluation for complex task management.
-   **Structured JSON Planning:** The agent's "Chief Architect" (Planner) generates detailed JSON-based plans, which are presented to the user for review before execution begins.
-   **Secure Sandboxed Workspaces:** Every task is assigned a unique, isolated directory, ensuring security and preventing state-collision between different tasks.
-   **Modular & Resilient Tools:** A flexible tool system allows for easy addition of new capabilities. Current tools include web search, a sandboxed file system (read, write, list), and a sandboxed shell.
-   **Interactive & Transparent Frontend:** A responsive user interface built with Preact and Vite, designed to provide clear, real-time visibility into the agent's complex operations.
    -   **Hierarchical Agent Trace:** See the agent's thought process as a clear, threaded conversation. The UI visualizes the handoff from the "Chief Architect's" plan, to the "Site Foreman's" execution log, to the "Editor's" final summary.
    -   **Live Step Execution:** Watch each step of the plan update in real-time from "pending" to "in-progress" to "completed" or "failed".
    -   **Task Management Panel:** A dedicated sidebar for managing the entire lifecycle of your research tasks.
    -   **Dynamic Model Selection:** Configure the LLM for each agent role (Router, Planner, etc.) directly from the UI.
    -   **Interactive Workspace & Artifact Viewer:** Browse, view, and upload files directly within the agent's sandboxed workspace for each task.

## 3. Project Structure

```

.

├── backend/

│ ├── tools/

│ │ ├── ... (Modular tool files)

│ ├── langgraph\_agent.py # Core PCEE agent logic

│ ├── prompts.py # Centralized prompts for all agent nodes

│ └── server.py # WebSocket server entry point

│

├── src/

│ ├── components/

│ │ ├── AgentCards.jsx # Components for each agent's response

│ │ ├── Common.jsx # Shared components like buttons

│ │ └── Icons.jsx # All SVG icon components

│ │

│ ├── App.jsx # Main UI component and state management

│ ├── index.css # Global CSS and Tailwind directives

│ └── main.jsx # Frontend application entry point

│

├── .env.example # Template for environment variables

├── .gitignore # Specifies files to ignore for version control

├── BRAINSTORM.md # Document for future ideas

├── docker-compose.yml # Orchestrates the Docker container

├── Dockerfile # Defines the application's Docker image

├── package.json # Frontend dependencies and scripts

├── PCEE\_ARCHITECTURE.md # Document detailing the agent's design

├── tailwind.config.js # Tailwind CSS configuration

└── ROADMAP.md # Project development plan

```

## 4. Installation & Setup

You will need two separate terminals to run the backend and frontend servers.

### Prerequisites

-   **Docker:** Ensure Docker and Docker Compose are installed.
-   **Node.js & npm:** Ensure Node.js (which includes npm) is installed.

### Step 1: Backend Server

1.  **Configure Environment Variables:** Create a `.env` file from the `.env.example` template and add your API keys (`GOOGLE_API_KEY`, `TAVILY_API_KEY`). You can also specify which models to use for each agent role here.
2.  **Run the Backend:** In your first terminal, run `docker compose up --build`.

### Step 2: Frontend Server

1.  **Install Dependencies:** In your second terminal, run `npm install`. This is crucial to install all dependencies, including the Tailwind CSS typography plugin.
2.  **Run the Frontend:** Run `npm run dev`.
3.  **Access the Application:** Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).


# ROADMAP

This document outlines the phased development plan for the ResearchAgent project.

### ✔️ Phase 0-7: Core Engine & Stateful UI

-   \[x\] **Core Backend & Frontend:** Foundational PCEE architecture and UI are stable.
-   \[x\] **Stateful Task Management:** The application is centered around persistent tasks with independent workspaces and histories.

### 🚀 **UP NEXT:** Phase 8: The "Three-Track Brain" & HITL Refactor

This is a major architectural overhaul to improve the agent's efficiency, robustness, and user interactivity.

-   \[ \] **Implement the Three-Track Router:** Rewrite the `router_node` to intelligently sort user requests into `DIRECT_QA`, `SIMPLE_TOOL_USE`, or `COMPLEX_PROJECT` tracks.
-   \[ \] **Build the "Handyman" Path:** Create the new `handyman_node` to handle single-step tool commands, bypassing the complex planning loop.
-   \[ \] **Unify the Output:** Overhaul the `editor_node` and its prompt to serve as the single, consistent voice of the agent for all three tracks.
-   \[ \] **Implement Conversational HITL:** Build the plan approval loop, allowing the user to conversationally refine the `Chief_Architect`'s plans before execution.

### Phase 9: Advanced Context & Environment

-   \[ \] **Full Conversational History:** Feed the complete chat history back into the prompts to give the agent true multi-turn contextual memory.
-   \[ \] **Python Virtual Environments:** Implement full dependency sandboxing with per-task `.venv` directories.

### Phase 10: Production Readiness

-   \[ \] **Database Integration:** Replace browser `localStorage` with a robust database backend (e.g., SQLite) for persistent, server-side state.
-   \[ \] **Enhanced File Viewer:** Upgrade the workspace file viewer to intelligently render various file types (Markdown, images, etc.).

# BRAINSTORM

This document is a living collection of ideas, architectural concepts, and potential future features for the ResearchAgent project.

## 1\. The "Company Model" Architecture

Our agent operates like a small, efficient company with specialized roles. This separation of concerns enables complex, resilient behavior.

-   **The Router:** The dispatcher who triages all incoming requests.
-   **The Handyman:** A specialist for quick, single-step tool-based tasks.
-   **The Chief Architect:** The strategist who designs high-level project blueprints.
-   **The Site Foreman:** The project manager who oversees the execution and correction of individual plan steps.
-   **The Worker:** The hands-on specialist who executes precise tool commands.
-   **The Project Supervisor:** The quality assurance inspector who validates outcomes.
-   **The Editor:** The unified communications director who formats and delivers all final reports and answers to the user.

## 2\. The "Three-Track Brain" Design Philosophy

To avoid using a "sledgehammer for every nail," the agent routes tasks down one of three paths based on its complexity.

-   **Track 1: Direct Q&A:** For simple questions. The Router sends the task directly to the Editor, which functions as a standard chatbot.
-   **Track 2: Simple Tool Use:** For single commands (e.g., "create a file"). The Router sends the task to the Handyman, who formulates a single-step plan. This is executed by the Worker and summarized by the Editor.
-   **Track 3: Complex Projects:** For multi-step tasks. The Router engages the full "Company," starting with the Chief Architect. This track utilizes our full planning, execution, self-correction, and human-in-the-loop capabilities.

## 3\. Conversational Human-in-the-Loop (HITL)

The user is a collaborator, not just an operator. The HITL cycle for complex projects ensures the user has ultimate authority over the agent's strategy.

-   **The Dialogue Loop:**
    1.  The `Chief_Architect` proposes a plan.
    2.  The system **pauses** and presents the plan to the user.
    3.  The user can either **approve** the plan to begin execution or provide **natural language feedback** for modifications (e.g., "Add a step to zip the files at the end").
    4.  If modifications are requested, the feedback is sent back to the `Chief_Architect`, who generates a new, improved plan.
