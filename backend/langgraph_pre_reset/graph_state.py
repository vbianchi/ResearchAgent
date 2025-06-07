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
    retry_count_for_current_step: Optional[int] 

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
    step_evaluation_suggested_tool: Optional[str] 
    step_evaluation_suggested_input_instructions: Optional[str] 
    step_evaluation_confidence_in_correction: Optional[float]
    step_evaluation_error: Optional[str] 
    
    # --- Fields for OverallEvaluatorNode output ---
    overall_evaluation_success: Optional[bool]
    overall_evaluation_assessment: Optional[str] # User-facing summary of plan outcome
    overall_evaluation_final_answer_content: Optional[str] # The actual content to show user if successful
    overall_evaluation_suggestions_for_replan: Optional[List[str]] # If plan failed
    overall_evaluation_error: Optional[str] # For errors within the OverallEvaluatorNode itself

    # General operational fields
    error_message: Optional[str] 
