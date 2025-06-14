# ResearchAgent - Brainstorming & Ideas

This document is a living collection of ideas, architectural concepts, and potential future features for the ResearchAgent project.

## Core Agent Architecture: The "Company" Model

Our agent operates like a small, efficient company with specialized roles, creating a clear separation of concerns that enables complex behavior.

-   **The "Router" (Dispatcher):** The entry point. It quickly analyzes a user's request and decides if it's a simple question that can be answered directly or a complex task that requires the full project team.
-   **The "Librarian" (Direct QA):** The fast-response expert. Handles simple, direct questions without the overhead of the full planning process.
-   **The "Chief Architect" (Planner):** A strategic thinker that creates detailed, structured JSON "blueprints" for each complex task.
-   **The "Site Foreman" (Controller):** The project manager that executes the blueprint step-by-step, managing data piping and managing correction sub-loops.
-   **The "Worker" (Executor):** The specialist that takes precise instructions from the Controller and runs the tools.
-   **The "Project Supervisor" (Evaluator):** The quality assurance inspector with the power to validate steps, halt execution, or trigger a correction loop.

## ✅ COMPLETED: Stateful Task Management & UI

We have successfully evolved the agent from a stateless, single-shot tool into a persistent, multi-turn assistant.

-   **Task as the Core Unit:** The application is now centered around the "Task". Each task has its own unique `task_id`, a sandboxed workspace on the file system, and a complete, ordered chat history that is persisted in the browser's `localStorage`.
-   **Advanced UI Rendering:** The frontend has been refactored into a clean, component-based architecture. It now features a sophisticated and unified UI that clearly visualizes the contributions of each agent in the "Company Model." This includes a visual thread to show the flow of information and real-time status updates (pending, in-progress, completed, failed) for each step of the execution plan.

## 🚀 IN PROGRESS: Autonomy and Intelligence

With the stateful foundation in place, our current focus is on elevating the agent's intelligence by giving it memory and the ability to recover from errors.

### 1\. Self-Correction Sub-Loop (Implemented)

The agent no longer fails on the first error. We have implemented a foundational self-healing mechanism.

-   **The Trigger:** When the `Project_Supervisor` evaluates a step and returns a `failure` status, the agent does not give up.
-   **The `Correction_Planner` Node:** We introduced a new agent node whose sole job is to fix mistakes.
    -   **Input:** The failed step's instruction and the Supervisor's evaluation feedback (e.g., "The file content is a placeholder.").
    -   **Action:** It formulates a _new, revised plan step_ to fix the immediate problem. In our successful test, it replaced a placeholder with the actual, required information from a previous step.
    -   **Output:** A revised plan step.
-   **The Loop:** The graph now routes to this new node upon failure. It then sends the corrected plan step back to the `Site_Foreman` to be re-executed. This loop can be attempted multiple times before the agent escalates the failure.

### 2\. Data Piping & Conversational Memory (Up Next)

This is our immediate priority. The agent's real power will come from proactively understanding the flow of data and the context of a conversation.

-   **Goal:** For every new request within a task, the `Site_Foreman` must be able to use the output of previous steps as input for subsequent steps.
-   **Mechanism:** The `Chief_Architect` will create plans with placeholders (e.g., `"content": "{step_1_output}"`). The `Site_Foreman` will then be responsible for replacing these placeholders with the actual data from the `step_outputs` dictionary in the agent's state before calling a tool.
-   **Impact:** This will unlock more complex, multi-stage workflows and prevent the kind of self-correction we just observed by ensuring the correct data is used in the first place. Once this is complete, we will be able to feed the entire chat history back into the prompts to achieve true conversational memory.
