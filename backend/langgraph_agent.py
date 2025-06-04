# backend/langgraph_agent.py (Part 1/2)
import logging
from typing import TypedDict, Optional, List, Annotated, Dict, Any

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # Commented out

from backend.config import settings

logger = logging.getLogger(__name__)

# --- Define State ---
class ResearchAgentState(TypedDict, total=False):
    user_query: str
    classified_intent: Optional[str]
    plan_steps: Optional[List[Dict[str, Any]]]
    current_step_index: int
    current_task_id: Optional[str]
    chat_history: Optional[List[BaseMessage]]
    controller_output_tool_name: Optional[str]
    controller_output_tool_input: Optional[str]
    executor_output: Optional[str]
    previous_step_executor_output: Optional[str]
    step_evaluator_output: Optional[Dict[str, Any]]
    overall_evaluator_output: Optional[Dict[str, Any]]
    retry_count_for_current_step: int
    accumulated_plan_summary: str
    tool_config: Optional[Dict[str, Any]]

# --- Placeholder Node Functions ---
async def placeholder_intent_classifier_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Intent Classifier (Placeholder)")
    intent = state.get("classified_intent", "DIRECT_QA")
    logger.info(f"Intent classified as: {intent}")
    return {"classified_intent": intent, "current_step_index": 0, "retry_count_for_current_step": 0}

async def placeholder_direct_qa_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Direct QA (Placeholder)")
    user_query = state.get("user_query", "No query in state")
    answer = f"Placeholder direct answer to: {user_query}"
    logger.info(f"Direct QA generated answer: {answer}")
    return {"overall_evaluator_output": {"assessment": answer, "overall_success": True, "is_direct_qa": True}}

async def placeholder_planner_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Planner (Placeholder)")
    plan = [{"step_id": 1, "description": "Placeholder: Do step 1", "tool_to_use": "some_tool", "expected_outcome": "Step 1 done"}]
    logger.info(f"Plan generated: {plan}")
    return {"plan_steps": plan, "current_step_index": 0, "retry_count_for_current_step": 0, "accumulated_plan_summary": "Plan:\n1. Do step 1\n"}

async def placeholder_controller_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Controller (Placeholder)")
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    step_info = plan_steps[current_idx] if plan_steps and 0 <= current_idx < len(plan_steps) else {}
    logger.info(f"Controller for step {current_idx + 1}: {step_info.get('description')}")
    return {"controller_output_tool_name": step_info.get("tool_to_use", "None"), 
            "controller_output_tool_input": "placeholder_input_for_tool"}

async def placeholder_executor_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Executor (Placeholder)")
    tool_name = state.get("controller_output_tool_name", "None")
    tool_input = state.get("controller_output_tool_input", "")
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
    output = f"Placeholder output from {tool_name}."
    return {"executor_output": output}

async def placeholder_step_evaluator_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Step Evaluator (Placeholder)")
    current_idx = state.get("current_step_index", 0)
    executor_out = state.get("executor_output", "")
    # Simulate success for simplicity in this placeholder
    eval_output = {"step_achieved_goal": True, "assessment_of_step": "Placeholder: Step looks good.", "is_recoverable_via_retry": False}
    logger.info(f"Step {current_idx + 1} evaluation: Achieved={eval_output['step_achieved_goal']}")
    
    new_accumulated_summary = state.get("accumulated_plan_summary", "") + f"Step {current_idx + 1} Output: {executor_out[:100]}\n"

    if eval_output["step_achieved_goal"]:
        return {
            "step_evaluator_output": eval_output,
            "previous_step_executor_output": executor_out,
            "retry_count_for_current_step": 0, 
            "accumulated_plan_summary": new_accumulated_summary
        }
    else: 
        return {
            "step_evaluator_output": eval_output,
            "previous_step_executor_output": None,
            "accumulated_plan_summary": new_accumulated_summary + f"Step {current_idx + 1} Failed Assessment: {eval_output['assessment_of_step']}\n"
        }

async def placeholder_overall_evaluator_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> NODE: Overall Evaluator (Placeholder)")
    final_assessment = state.get("overall_evaluator_output", {}).get("assessment", "Placeholder: Plan execution finished.")
    if state.get("overall_evaluator_output", {}).get("is_direct_qa"):
        logger.info(f"Overall evaluation for Direct QA: {final_assessment}")
    else:
        logger.info(f"Overall evaluation for Plan: {final_assessment}")
        logger.info(f"Accumulated Plan Summary for Eval:\n{state.get('accumulated_plan_summary')}")
    return {"overall_evaluator_output": {"assessment": final_assessment, "overall_success": True}}

# --- Conditional Edge Logic ---
def should_proceed_to_plan_or_qa(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Plan or Direct QA? ---")
    intent = state.get("classified_intent")
    if intent == "PLAN":
        logger.info(f"Intent is '{intent}', proceeding to Planner.")
        return "planner"
    elif intent == "DIRECT_QA":
        logger.info(f"Intent is '{intent}', proceeding to Direct QA.")
        return "direct_qa"
    else:
        logger.warning(f"Unknown intent '{intent}', defaulting to Direct QA.")
        return "direct_qa"

# MODIFIED: Merged has_more_steps_in_plan logic into this function
def should_retry_step_or_proceed(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Step Outcome - Retry, Next Step, or Evaluate Overall? ---")
    step_eval = state.get("step_evaluator_output", {})
    current_retries = state.get("retry_count_for_current_step", 0)
    max_retries = settings.agent_max_step_retries

    if not step_eval.get("step_achieved_goal", False):  # Step failed
        if step_eval.get("is_recoverable_via_retry", False) and current_retries < max_retries:
            logger.info(f"Step failed but is recoverable. Will attempt retry {current_retries + 1} of {max_retries}.")
            return "retry_step"  # This will lead to increment_retry_count_node
        else:
            logger.info("Step failed and is not recoverable or retries exhausted. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"
    else:  # Step succeeded
        logger.info("Step succeeded. Checking for more steps in the plan.")
        plan = state.get("plan_steps", [])
        current_idx = state.get("current_step_index", -1)  # current_step_index is 0-based
        
        if current_idx + 1 < len(plan):  # If there's a next step
            logger.info(f"More steps exist. Will proceed to step {current_idx + 2} (index {current_idx + 1}).")
            return "next_step"  # This will lead to advance_to_next_step_node
        else:
            logger.info("No more steps in plan. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"

# REMOVED: has_more_steps_in_plan function (its logic is now in should_retry_step_or_proceed)

# --- Utility nodes for state updates before looping ---
def increment_retry_count_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Incrementing Retry Count")
    current_retries = state.get("retry_count_for_current_step", 0)
    return {"retry_count_for_current_step": current_retries + 1}

def advance_to_next_step_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Advancing to Next Step")
    current_idx = state.get("current_step_index", -1)
    return {
        "current_step_index": current_idx + 1,
        "retry_count_for_current_step": 0, 
    }

# backend/langgraph_agent.py (Part 2/2)
# (Continued from Part 1)
import asyncio # For the __main__ test block

# --- Graph Definition Function ---
def create_research_agent_graph():
    """
    Builds and compiles the LangGraph research agent.
    This version uses placeholder node functions and corrected edge logic.
    """
    logger.info("Building Research Agent Graph with placeholder nodes...")
    workflow = StateGraph(ResearchAgentState)

    # Add Nodes using the placeholder functions defined in Part 1
    workflow.add_node("intent_classifier", placeholder_intent_classifier_node)
    workflow.add_node("direct_qa", placeholder_direct_qa_node)
    workflow.add_node("planner", placeholder_planner_node)
    workflow.add_node("controller", placeholder_controller_node)
    workflow.add_node("executor", placeholder_executor_node)
    workflow.add_node("step_evaluator", placeholder_step_evaluator_node)
    workflow.add_node("overall_evaluator", placeholder_overall_evaluator_node)
    
    # Utility nodes for state updates
    workflow.add_node("increment_retry_count", increment_retry_count_node)
    workflow.add_node("advance_to_next_step", advance_to_next_step_node)

    # --- Define Edges ---
    workflow.set_entry_point("intent_classifier")

    # From Intent Classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        should_proceed_to_plan_or_qa, # Defined in Part 1
        {
            "planner": "planner",
            "direct_qa": "direct_qa"
        }
    )

    # Path for Direct QA
    workflow.add_edge("direct_qa", "overall_evaluator") 

    # Path for Planning
    workflow.add_edge("planner", "controller") 

    # Main P-C-E-E loop
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")

    # MODIFIED: Conditional logic after step evaluation
    workflow.add_conditional_edges(
        "step_evaluator",
        should_retry_step_or_proceed, # This function now handles all outcomes
        {
            "retry_step": "increment_retry_count",    # Leads to utility node then back to controller
            "next_step": "advance_to_next_step",     # Leads to utility node then back to controller
            "evaluate_overall_plan": "overall_evaluator" # Step failed definitively or plan finished
        }
    )
    
    # Edge from utility node for incrementing retry count back to controller
    workflow.add_edge("increment_retry_count", "controller")

    # Edge from utility node for advancing step index back to controller
    workflow.add_edge("advance_to_next_step", "controller")

    # REMOVED: The problematic second conditional_edges call that started from "check_for_more_steps"

    # Final evaluation leads to END
    workflow.add_edge("overall_evaluator", END)

    logger.info("Compiling Research Agent Graph...")
    app = workflow.compile()
    logger.info("Research Agent Graph (with placeholders and corrected edges) compiled successfully.")
    return app

# --- Create and export the compiled graph instance ---
research_agent_graph = create_research_agent_graph()
logger.info(f"Module-level variable 'research_agent_graph' (compiled graph app) created. Type: {type(research_agent_graph)}")


# Example of how to run (for testing within this file if needed)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
    )
    logger.info("Running langgraph_agent.py directly for testing graph compilation and structure.")
    
    # (Optional test invocation code can go here if needed)

    logger.info("langgraph_agent.py (Part 2) finished execution if run as __main__.")
