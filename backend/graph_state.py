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
    previous_step_executor_output: Optional[str] # Output from the *previous executed* step (used by Controller)
    retry_count_for_current_step: Optional[int] # To manage retries for the current step

    # --- Fields for ControllerNode output (for the current step) ---
    controller_tool_name: Optional[str]
    controller_tool_input: Optional[str] 
    controller_reasoning: Optional[str]
    controller_confidence: Optional[float]
    controller_error: Optional[str] 

    # --- Fields for ExecutorNode output (for the current step) ---
    current_executor_output: Optional[str] 
    executor_error_message: Optional[str] 

    # --- Fields for StepEvaluatorNode output (for the current step) ---
    step_evaluation_achieved_goal: Optional[bool]
    step_evaluation_assessment: Optional[str]
    step_evaluation_is_recoverable: Optional[bool]
    # Suggestions for retry if is_recoverable is True:
    step_evaluation_suggested_tool: Optional[str] 
    step_evaluation_suggested_input_instructions: Optional[str] # Instructions for Controller
    step_evaluation_confidence_in_correction: Optional[float]
    step_evaluation_error: Optional[str] # For errors occurring within the StepEvaluatorNode itself
    
    # --- Fields for OverallEvaluatorNode (example, to be added later) ---
    # overall_evaluation_success: Optional[bool]
    # overall_evaluation_assessment: Optional[str]
    # overall_evaluation_final_answer: Optional[str]

    # General operational fields
    error_message: Optional[str] 
