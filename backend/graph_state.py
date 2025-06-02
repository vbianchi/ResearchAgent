# backend/graph_state.py
from typing import TypedDict, Optional, List, Dict, Any # Removed Annotated, operator
from langchain_core.messages import BaseMessage

# add_messages helper is no longer needed here if we append manually in nodes

class ResearchAgentState(TypedDict):
    """
    Represents the state of the ResearchAgent LangGraph.
    """
    user_query: str
    # MODIFIED: Simplified messages field for manual accumulation in nodes
    messages: List[BaseMessage] 

    # --- Task Identification ---
    task_id: Optional[str] 

    # --- Fields populated by IntentClassifierNode ---
    classified_intent: Optional[str]
    intent_classifier_reasoning: Optional[str]

    # --- Fields for PlannerNode ---
    plan_summary: Optional[str]
    plan_steps: Optional[List[Dict[str, Any]]] 
    plan_generation_error: Optional[str]

    # --- Fields for ControllerNode & Executor (example) ---
    # current_step_index: Optional[int]
    # ... (other fields remain the same)

    # General operational fields
    error_message: Optional[str] 
