# backend/graph_state.py
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage

class ResearchAgentState(TypedDict):
    """
    Represents the state of the ResearchAgent LangGraph.
    """
    user_query: str
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

    # --- Fields for Iterating and Controlling Plan Steps ---
    current_step_index: Optional[int] 
    previous_step_executor_output: Optional[str] 

    # --- Fields for ControllerNode output (for the current step) ---
    controller_tool_name: Optional[str]
    controller_tool_input: Optional[str] 
    controller_reasoning: Optional[str]
    controller_confidence: Optional[float]
    controller_error: Optional[str] 

    # --- Fields for ExecutorNode output (for the current step) ---
    current_executor_output: Optional[str] # Output from the tool/LLM execution
    executor_error_message: Optional[str] # Specific errors from the executor node/tool run

    # --- Fields for EvaluatorNode (example, to be added later) ---
    # step_evaluation_achieved_goal: Optional[bool]
    # ... (other evaluation fields)

    # General operational fields
    error_message: Optional[str] # For critical graph/node errors not specific to a component's logic
