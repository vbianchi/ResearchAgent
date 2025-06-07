# LAST UPDATE
# BRAINSTORM.md - ResearchAgent Project

This document tracks current workflow ideas, user feedback, and recent design decisions for the ResearchAgent project.

### **Project Reset and New Phased Approach (June 2025)**

**Problem Analysis:** Recent development efforts (v2.6.0) to migrate the agent logic to a full LangGraph PCEE (Plan-Code-Execute-Evaluate) workflow resulted in a non-functional and unstable state. The "big bang" approach of implementing all nodes at once led to a series of cascading, hard-to-debug errors, including:

-   `KeyError` from prompt formatting issues.
-   `NameError` from incorrect function references during refactoring.
-   `AttributeError` from improper data handling between components.

Most importantly, the core `intent_classifier` logic regressed, failing to distinguish simple queries from complex ones, and the interactive "propose and confirm plan" workflow was compromised.

**New Strategic Decision:** The project is being reset to a stable baseline to allow for a more robust and transparent development process. The new guiding principles are:

1.  **Simplicity First:** Start with the absolute simplest working implementation.
2.  **Incremental Builds:** Add only one new component or piece of logic at a time.
3.  **Verify at Each Step:** Thoroughly test and validate each new addition before proceeding.

This approach, detailed in the updated `ROADMAP.md`, will ensure we build a reliable and well-understood agent on the LangGraph architecture, preventing the integration issues that halted progress.

**Recent Breakthroughs & Refinements (Pre-Reset)**

1.  **Graph Visualization & Cleanup:**
    -   **Status:** Implemented. The `visualize_graph.py` script remains useful for understanding the target architecture.
2.  **Intent Classification Fixed & Refined:**
    -   **Status:** **REGRESSED.** While a `KeyError` was fixed, the overall logic proved unstable during the migration. This will be one of the first components to be carefully re-implemented and verified.
3.  **Plan Confirmation UI:**
    -   **Status:**


# PREVIOUS UPDATE
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