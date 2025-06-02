# backend/graph_state.py
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator

# Helper for adding messages to the state.
# LangGraph uses this to allow nodes to append to the 'messages' list
# rather than overwriting it.
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Concatenates two lists of BaseMessage objects."""
    if not isinstance(left, list) or not isinstance(right, list):
        # Handle cases where one might be None or not a list, though state should initialize it as a list
        new_left = left if isinstance(left, list) else []
        new_right = right if isinstance(right, list) else []
        return new_left + new_right
    return left + right

class ResearchAgentState(TypedDict):
    """
    Represents the state of the ResearchAgent LangGraph.
    All fields that might be updated by a node should be Optional or have a default
    if they aren't guaranteed to be set by the entry point.
    """
    # Input from the user, will be the first message
    user_query: str

    # Standard LangGraph field for accumulating messages (e.g., chat history, system messages)
    # The `add_messages` operator ensures new messages are appended.
    # It's crucial that 'messages' is initialized as an empty list or with initial messages
    # when the graph run starts.
    messages: Annotated[List[BaseMessage], add_messages]

    # --- Fields populated by IntentClassifierNode ---
    classified_intent: Optional[str]
    intent_classifier_reasoning: Optional[str]

    # --- Fields for PlannerNode (example) ---
    # plan_summary: Optional[str]
    # plan_steps: Optional[List[Dict[str, Any]]] # Or List[PlanStep Pydantic model]
    # plan_generation_error: Optional[str] # If planner fails

    # --- Fields for ControllerNode & Executor (example) ---
    # current_step_index: Optional[int]
    # current_step_description: Optional[str] # From the plan
    # current_step_expected_outcome: Optional[str] # From the plan
    # controller_tool_name: Optional[str]
    # controller_tool_input: Optional[str] # Could be JSON string or direct input
    # controller_reasoning: Optional[str]
    # controller_confidence: Optional[float]
    # executor_output: Optional[str] # Output from the ReAct agent or tool
    # executor_error: Optional[str]

    # --- Fields for EvaluatorNode (example) ---
    # step_evaluation_achieved_goal: Optional[bool]
    # step_evaluation_assessment: Optional[str]
    # step_evaluation_is_recoverable: Optional[bool]
    # step_evaluation_suggested_tool: Optional[str]
    # step_evaluation_suggested_input: Optional[str]
    # step_evaluation_confidence: Optional[float]
    # overall_evaluation_success: Optional[bool]
    # overall_evaluation_assessment: Optional[str]
    # overall_evaluation_final_answer: Optional[str]

    # General operational fields
    error_message: Optional[str] # For critical errors that halt the graph or specific node errors
    # task_id: Optional[str] # If needed within the graph execution directly
    # llm_config_overrides: Optional[Dict[str, Any]] # For dynamic LLM settings per run
    # retry_count_for_step: Optional[int] # For managing retries
