BRAINSTORM.md - ResearchAgent Project (v2.6.0 In Progress)
==========================================================

This document tracks current workflow ideas, user feedback, and recent design decisions for the ResearchAgent project.

**Recent Breakthroughs & Refinements (v2.6.0)**

1.  **Graph Visualization & Cleanup:**
    -   **Status:** Implemented a `visualize_graph.py` script to generate a MermaidJS definition of our LangGraph architecture.
    -   **Outcome:** The visualization helped identify a confusing, redundant error path for planner failures. We have now refactored the graph logic in `langgraph_agent.py` so that the `planner` node is solely responsible for handling its own success or failure. This makes the architecture cleaner and more intuitive.

2.  **Intent Classification Fixed & Refined:**
    -   **Status:** Resolved a critical `KeyError` bug in `intent_classifier.py` that was causing all queries to default to a "PLAN".
    -   **Outcome:** The bug was traced to unescaped curly braces in the prompt's examples. The prompt has been significantly improved to provide a clearer decision-making hierarchy for the LLM, sharpening the distinction between `DIRECT_QA` (no tools), `DIRECT_TOOL_REQUEST` (single tool), and `PLAN` (multiple steps). The classifier is now functioning correctly.

3.  **Plan Confirmation UI:**
    -   **Status:** Resolved.
    -   **Outcome:** The UI now correctly displays plan proposals for user confirmation. A previous issue was traced to a backend/frontend mismatch in the WebSocket message type (`display_plan_for_confirmation` vs. `propose_plan_for_confirmation`), which has been fixed in `agent_flow_handlers.py`.

**UI/UX Feedback & Ideas (To be revisited/prioritized post-PCEE implementation):**

1.  **Chat Clutter & Plan Display Format:**
    -   **User Feedback:** The desire is for a cleaner interface, distinguishing clearly between direct agent-user messages and status/progress updates.
    -   **LangGraph Implication:** LangGraph's event streaming gives us granular control over what gets sent to the main chat versus the monitor log. We can leverage this to create a much cleaner chat experience once the PCEE nodes are implemented.

2.  **Color-Coding Agent Workspace:**
    -   **User Idea:** Visually differentiate messages in the Monitor Log based on the agent component (Planner, Controller, etc.).
    -   **LangGraph Implication:** Node-specific outputs from LangGraph can be tagged via the callback handler to facilitate this color-coding in the UI.