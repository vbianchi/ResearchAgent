# ResearchAgent - Brainstorming & Ideas

This document is a living collection of ideas, architectural concepts, and potential future features for the ResearchAgent project.

## Core Agent Architecture: The "Company" Model

Our agent operates like a small, efficient company with specialized roles. This separation of concerns is the key to its complex behavior.

-   **The "Router" (Dispatcher):** Quickly classifies user requests into one of three tracks: Direct Q&A, Simple Tool Use, or Complex Project.
-   **The "Memory Updater" (Librarian):** A critical pre-processing step that analyzes every user message to update the agent's structured JSON "Memory Vault," ensuring all new facts are stored before any action is taken.
-   **The "Handyman" (Simple Executor):** A fast-lane agent that handles simple, single-step tool commands.
-   **The "Chief Architect" (Planner):** A strategic thinker that creates detailed, structured JSON "blueprints" for complex tasks.
-   **The "Site Foreman" (Controller):** The project manager that executes the blueprint step-by-step, managing data piping and correction sub-loops.
-   **The "Worker" (Executor):** The specialist that takes precise instructions and runs the tools.
-   **The "Project Supervisor" (Evaluator):** The quality assurance inspector that validates the outcome of each step in a complex plan.
-   **The "Editor" (Reporter):** The unified voice of the agent, capable of acting as a conversational assistant or a project manager to deliver context-aware final responses.

## ✅ COMPLETED FEATURES

-   **Stateful Task Management:** The application is centered around persistent "Tasks", each with a unique workspace and a complete chat history.
-   **Advanced UI Rendering & Control:** A sophisticated UI visualizes the agent's operations in real-time.
-   **Multi-Level Self-Correction:** The agent can robustly handle errors by retrying steps or creating entirely new plans.
-   **Three-Track Brain & Interactive HITL:** The agent efficiently routes requests and allows users to review and modify complex plans before execution.
-   **Robust Memory & Environments:** The agent uses a "Memory Vault" for persistent knowledge and automatically creates isolated Python virtual environments for each task.
-   **Interactive Workbench v1:** A functional file explorer with structured listing, navigation, create/rename/delete actions, drag-and-drop upload, and a smart previewer for text, images, Markdown, and CSVs.
-   **True Concurrency & Control:** The backend server now correctly handles multiple, simultaneous agent runs without interruption. The architecture is fully decoupled, and users can stop any running task from the UI.

## 🚀 NEXT FOCUS: Phase 13: The "Tool Forge"

_**Vision:** Allow users to create and add their own tools to the ResearchAgent without writing any backend code._

### The "Tool Forge" Plan

-   **Tool Creator UI:** Build a "Tool Forge" section in the UI where users can define a tool's properties via a simple form. This will include:
    -   Tool Name (e.g., `get_weather`)
    -   Tool Description (a clear explanation for the LLM)
    -   Input Arguments (a list of names, types, and descriptions)
-   **API Endpoint for Tool Creation:**
    -   **Endpoint:** `POST /api/tools`
    -   **Request Body:** A structured JSON object containing the user's complete tool definition.
-   **Dynamic Tool Generation Backend:**
    -   The server will receive the JSON definition.
    -   It will validate the input to ensure it's a valid tool structure.
    -   It will then use a template to dynamically generate the complete Python code for a new tool file (e.g., `backend/tools/custom_get_weather.py`).
    -   The server will save this new file to the `backend/tools/` directory.
-   **Live Reloading:** The tool loader in `backend/tools/__init__.py` will need to be made aware of the new tool so that it's immediately available to the agent on the next run without requiring a server restart.
