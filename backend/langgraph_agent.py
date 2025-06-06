# backend/langgraph_agent.py
import logging
from typing import TypedDict, Optional, List, Annotated, Dict, Any

import asyncio 

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.planner import generate_plan

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
    
    intent_classifier_llm_id: Optional[str]
    planner_llm_id: Optional[str]
    controller_llm_id: Optional[str]
    executor_llm_id: Optional[str]
    evaluator_llm_id: Optional[str]

    is_direct_qa_flow: bool

    plan_summary: Optional[str]
    plan_generation_error: Optional[str]

# --- Node Implementations ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Intent Classifier")
    configurable_settings = config.get("configurable", {})

    session_intent_llm_id = state.get("intent_classifier_llm_id") or configurable_settings.get("intent_classifier_llm_id")
    session_planner_llm_id = state.get("planner_llm_id") or configurable_settings.get("planner_llm_id")
    session_controller_llm_id = state.get("controller_llm_id") or configurable_settings.get("controller_llm_id")
    session_executor_llm_id = state.get("executor_llm_id") or configurable_settings.get("executor_llm_id")
    session_evaluator_llm_id = state.get("evaluator_llm_id") or configurable_settings.get("evaluator_llm_id")
    
    return {
        "classified_intent": state.get("classified_intent", "DIRECT_QA"),
        "current_step_index": 0,
        "retry_count_for_current_step": 0,
        "is_direct_qa_flow": state.get("classified_intent") == "DIRECT_QA",
        "intent_classifier_llm_id": session_intent_llm_id,
        "planner_llm_id": session_planner_llm_id,
        "controller_llm_id": session_controller_llm_id,
        "executor_llm_id": session_executor_llm_id,
        "evaluator_llm_id": session_evaluator_llm_id,
    }

async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Direct QA")
    user_query = state.get("user_query", "No query provided.")
    chat_history = state.get("chat_history", [])
    
    llm_id_str = state.get("executor_llm_id")
    if not llm_id_str:
        logger.warning("DirectQANode: executor_llm_id not found in state. Using system default executor LLM.")
        provider = settings.executor_default_provider
        model_name = settings.executor_default_model_name
    else:
        try:
            provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"DirectQANode: Invalid LLM ID format '{llm_id_str}'. Using system default executor LLM.")
            provider = settings.executor_default_provider
            model_name = settings.executor_default_model_name
    
    logger.info(f"DirectQANode: Using LLM {provider}::{model_name}")
    
    try:
        llm = get_llm(settings, provider=provider, model_name=model_name, 
                      callbacks=config.get("callbacks"), 
                      requested_for_role="DirectQA_Node")
    except Exception as e:
        logger.error(f"DirectQANode: Failed to initialize LLM: {e}")
        return {"executor_output": f"Error: Could not initialize LLM for Direct QA. {e}", "is_direct_qa_flow": True}

    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's query directly and concisely. Consider the chat history for context if relevant."),
        ("human", "Chat History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:")
    ])
    
    chain = prompt_template | llm
    
    logger.info(f"DirectQANode: Invoking LLM for query: '{user_query}'")
    try:
        response = await chain.ainvoke({"user_query": user_query, "chat_history": history_str}, config=config)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"DirectQANode: LLM produced answer (raw): {answer_text[:200]}...")
        return {"executor_output": answer_text, "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"executor_output": f"Error processing Direct QA: {e}", "is_direct_qa_flow": True}

async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Planner")
    user_query = state.get("user_query")
    current_task_id = state.get("current_task_id")
    planner_llm_id_str = state.get("planner_llm_id")

    if not user_query:
        logger.error("PlannerNode: User query is missing from state.")
        return {"plan_steps": None, "plan_summary": None, "plan_generation_error": "User query is missing."}

    provider = settings.planner_provider
    model_name = settings.planner_model_name
    if planner_llm_id_str:
        try:
            parsed_provider, parsed_model_name = planner_llm_id_str.split("::", 1)
            provider = parsed_provider
            model_name = parsed_model_name
            logger.info(f"PlannerNode: Using LLM ID from state: {planner_llm_id_str}")
        except ValueError:
            logger.warning(f"PlannerNode: Invalid planner_llm_id format '{planner_llm_id_str}'. Falling back to system default for Planner.")
    else:
        logger.info(f"PlannerNode: planner_llm_id not found in state. Using system default for Planner: {provider}::{model_name}")

    try:
        planner_llm_instance = get_llm(
            settings,
            provider=provider,
            model_name=model_name,
            callbacks=config.get("callbacks"),
            requested_for_role="PlannerNode_LLM"
        )
    except Exception as e:
        logger.error(f"PlannerNode: Failed to initialize Planner LLM ({provider}::{model_name}): {e}", exc_info=True)
        return {
            "plan_steps": None, "plan_summary": None, 
            "plan_generation_error": f"Failed to initialize Planner LLM: {e}",
        }

    available_tools = get_dynamic_tools(current_task_id=current_task_id)
    available_tools_summary = "\n".join(
        [f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools]
    )
    if not available_tools_summary:
        available_tools_summary = "No tools are currently available."
        logger.warning("PlannerNode: No tools available for planning.")
    
    session_data_for_planner = {"session_planner_llm_id": planner_llm_id_str}

    human_summary, structured_steps = await generate_plan(
        user_query=user_query,
        available_tools_summary=available_tools_summary,
        session_data_entry=session_data_for_planner,
        llm_instance=planner_llm_instance
    )

    if human_summary and structured_steps:
        logger.info(f"PlannerNode: Plan generated successfully. Summary: {human_summary[:100]}...")
        initial_accumulated_summary = (
            f"Original Query: {user_query}\n"
            f"Overall Plan Summary: {human_summary}\n"
            f"--- Plan Steps ---\n"
        )
        for i, step in enumerate(structured_steps):
            initial_accumulated_summary += f"{i+1}. {step.get('description', 'N/A')}\n"
        
        return {
            "plan_steps": structured_steps,
            "plan_summary": human_summary,
            "plan_generation_error": None,
            "current_step_index": 0,
            "retry_count_for_current_step": 0,
            "accumulated_plan_summary": initial_accumulated_summary,
            "is_direct_qa_flow": False
        }
    else:
        logger.error(f"PlannerNode: Failed to generate plan for query: {user_query}")
        error_msg = "Failed to generate a plan."
        return {
            "plan_steps": None,
            "plan_summary": None,
            "plan_generation_error": error_msg,
        }

async def placeholder_controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Controller (Placeholder)")
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    step_info = plan_steps[current_idx] if plan_steps and 0 <= current_idx < len(plan_steps) else {}
    logger.info(f"Controller for step {current_idx + 1}: {step_info.get('description')}")
    return {"controller_output_tool_name": step_info.get("tool_to_use", "None"), 
            "controller_output_tool_input": "placeholder_input_for_tool"}

async def placeholder_executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Executor (Placeholder)")
    tool_name = state.get("controller_output_tool_name", "None")
    tool_input = state.get("controller_output_tool_input", "")
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
    output = f"Placeholder output from {tool_name}."
    return {"executor_output": output}

async def placeholder_step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Step Evaluator (Placeholder)")
    current_idx = state.get("current_step_index", 0)
    executor_out = state.get("executor_output", "")
    eval_output = {"step_achieved_goal": True, "assessment_of_step": "Placeholder: Step looks good.", "is_recoverable_via_retry": False}
    logger.info(f"Step {current_idx + 1} evaluation: Achieved={eval_output['step_achieved_goal']}")
    new_accumulated_summary = state.get("accumulated_plan_summary", "") + f"Step {current_idx + 1} Output: {executor_out[:100]}\n"
    if eval_output["step_achieved_goal"]:
        return {"step_evaluator_output": eval_output, "previous_step_executor_output": executor_out, "retry_count_for_current_step": 0, "accumulated_plan_summary": new_accumulated_summary}
    else: 
        return {"step_evaluator_output": eval_output, "previous_step_executor_output": None, "accumulated_plan_summary": new_accumulated_summary + f"Step {current_idx + 1} Failed Assessment: {eval_output['assessment_of_step']}\n"}

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Overall Evaluator")
    
    final_assessment_text = "Evaluation: Processing complete."
    is_direct_qa = state.get("is_direct_qa_flow", False)
    plan_generation_error = state.get("plan_generation_error")
    
    if plan_generation_error:
        logger.warning(f"OverallEvaluatorNode: Processing a plan generation failure: {plan_generation_error}")
        final_assessment_text = f"I'm sorry, I was unable to create a plan for your request. Error: {plan_generation_error}"
    elif is_direct_qa:
        raw_answer = state.get("executor_output", "No direct answer was generated.")
        logger.info(f"OverallEvaluatorNode: Processing direct QA response: {raw_answer[:200]}...")
        final_assessment_text = raw_answer
    else:
        logger.info(f"OverallEvaluatorNode: Processing plan execution outcome.")
        logger.info(f"Accumulated Plan Summary for Eval:\n{state.get('accumulated_plan_summary')}")
        final_assessment_text = state.get("executor_output", "Plan execution completed. See logs for details.")

    llm_id_str = state.get("evaluator_llm_id")
    if not llm_id_str:
        logger.warning("OverallEvaluatorNode: evaluator_llm_id not found in state. Using system default.")
        provider = settings.evaluator_provider
        model_name = settings.evaluator_model_name
    else:
        try: provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"OverallEvaluatorNode: Invalid LLM ID format '{llm_id_str}'. Using system default.")
            provider = settings.evaluator_provider
            model_name = settings.evaluator_model_name
            
    logger.info(f"OverallEvaluatorNode: Using LLM {provider}::{model_name} to finalize output.")
    try:
        finalizing_llm = get_llm(
            settings, provider=provider, model_name=model_name, 
            callbacks=config.get("callbacks"),
            requested_for_role="OverallEvaluator_Finalize"
        )
        
        prompt = ChatPromptTemplate.from_template("Present the following information as the agent's final response: {assessment_text}")
        chain = prompt | finalizing_llm
        
        logger.info(f"OverallEvaluatorNode: Invoking LLM to finalize assessment: '{final_assessment_text[:100]}...'")
        final_response_message_obj = await chain.ainvoke({"assessment_text": final_assessment_text}, config=config)
        
        final_content_for_state = final_response_message_obj.content if hasattr(final_response_message_obj, 'content') else str(final_response_message_obj)
        logger.info(f"OverallEvaluatorNode: Finalized response for state: {final_content_for_state[:200]}...")
        return {"overall_evaluator_output": {"assessment": final_content_for_state, "overall_success": True}}

    except Exception as e:
        logger.error(f"OverallEvaluatorNode: Error during final LLM invocation: {e}", exc_info=True)
        return {"overall_evaluator_output": {"assessment": f"Error finalizing response: {final_assessment_text}", "overall_success": False}}


# --- Conditional Edge Logic ---

# <<< START MODIFICATION: Simplified intent classifier edge >>>
def should_proceed_to_plan_or_qa(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Plan or Direct QA? ---")
    intent = state.get("classified_intent", "DIRECT_QA") # Default to QA if intent is missing
    if intent == "PLAN":
        logger.info(f"Intent is '{intent}', proceeding to Planner.")
        return "planner"
    else:
        # This now covers "DIRECT_QA" and "DIRECT_TOOL_REQUEST"
        logger.info(f"Intent is '{intent}', proceeding to Direct QA / Tool execution path.")
        return "direct_qa" # Both direct paths start at a common node for simplicity now
# <<< END MODIFICATION >>>

# <<< START NEW CONDITIONAL EDGE for planner success/failure >>>
def did_planning_succeed(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Planning Success or Failure? ---")
    if state.get("plan_generation_error"):
        logger.warning(f"Planning failed: {state['plan_generation_error']}. Routing to overall_evaluator.")
        return "failure"
    else:
        logger.info("Planning succeeded. Routing to controller.")
        return "success"
# <<< END NEW CONDITIONAL EDGE >>>

def should_retry_step_or_proceed(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Step Outcome - Retry, Next Step, or Evaluate Overall? ---")
    step_eval = state.get("step_evaluator_output", {})
    current_retries = state.get("retry_count_for_current_step", 0)
    max_retries = settings.agent_max_step_retries

    if not step_eval.get("step_achieved_goal", False):
        if step_eval.get("is_recoverable_via_retry", False) and current_retries < max_retries:
            logger.info(f"Step failed but is recoverable. Will attempt retry {current_retries + 1} of {max_retries}.")
            return "retry_step"
        else:
            logger.info("Step failed and is not recoverable or retries exhausted. Proceeding to overall plan evaluation.")
            state["executor_output"] = state.get("accumulated_plan_summary", "") + \
                                     f"\nStep {state.get('current_step_index', 0) + 1} ultimately failed. " + \
                                     (step_eval.get('assessment_of_step') or "No specific assessment provided.")
            return "evaluate_overall_plan"
    else:
        logger.info("Step succeeded. Checking for more steps in the plan.")
        plan_steps = state.get("plan_steps", [])
        current_idx = state.get("current_step_index", -1)
        if current_idx + 1 < len(plan_steps):
            logger.info(f"More steps exist. Will proceed to step {current_idx + 2} (index {current_idx + 1}).")
            return "next_step"
        else:
            logger.info("No more steps in plan. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"

# --- Utility nodes for state updates ---
def increment_retry_count_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Incrementing Retry Count")
    current_retries = state.get("retry_count_for_current_step", 0)
    return {"retry_count_for_current_step": current_retries + 1}

def advance_to_next_step_node(state: ResearchAgentState) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Advancing to Next Step")
    current_idx = state.get("current_step_index", -1)
    return {"current_step_index": current_idx + 1, "retry_count_for_current_step": 0}

# --- Graph Definition Function ---
def create_research_agent_graph():
    logger.info("Building Research Agent Graph...")
    workflow = StateGraph(ResearchAgentState)

    # Add Nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_qa", direct_qa_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("controller", placeholder_controller_node)
    workflow.add_node("executor", placeholder_executor_node)
    workflow.add_node("step_evaluator", placeholder_step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    workflow.add_node("increment_retry_count", increment_retry_count_node)
    workflow.add_node("advance_to_next_step", advance_to_next_step_node)

    # --- Define Edges ---
    workflow.set_entry_point("intent_classifier")

    # <<< START MODIFICATION: Update graph edges for clarity >>>
    # 1. From intent classifier, only go to the planner or the direct_qa node.
    workflow.add_conditional_edges(
        "intent_classifier",
        should_proceed_to_plan_or_qa,
        {"planner": "planner", "direct_qa": "direct_qa"}
    )
    
    # 2. From planner, check if planning succeeded.
    workflow.add_conditional_edges(
        "planner",
        did_planning_succeed,
        {
            "success": "controller",           # On success, go to controller
            "failure": "overall_evaluator"     # On failure, go to the end to report error
        }
    )
    # <<< END MODIFICATION >>>

    workflow.add_edge("direct_qa", "overall_evaluator") 

    # Path for Planning execution loop
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")

    workflow.add_conditional_edges(
        "step_evaluator",
        should_retry_step_or_proceed,
        {
            "retry_step": "increment_retry_count",
            "next_step": "advance_to_next_step",
            "evaluate_overall_plan": "overall_evaluator"
        }
    )
    
    workflow.add_edge("increment_retry_count", "controller")
    workflow.add_edge("advance_to_next_step", "controller")

    workflow.add_edge("overall_evaluator", END)

    logger.info("Compiling Research Agent Graph...")
    app = workflow.compile()
    logger.info("Research Agent Graph compiled successfully.")
    return app

# --- Create and export the compiled graph instance ---
research_agent_graph = create_research_agent_graph()
logger.info(f"Module-level variable 'research_agent_graph' (compiled graph app) created. Type: {type(research_agent_graph)}")

if __name__ == "__main__":
    # Test block remains the same
    pass