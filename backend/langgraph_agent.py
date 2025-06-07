# backend/langgraph_agent.py
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from backend.config import settings
from backend.graph_state import ResearchAgentState
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.planner import generate_plan
from backend.controller import validate_and_prepare_step_action
from backend.evaluator import evaluate_step_outcome_and_suggest_correction, evaluate_plan_outcome
from backend.pydantic_models import PlanStep
from backend.prompts import EXECUTOR_DIRECT_LLM_PROMPT_TEMPLATE

# MODIFIED: Import the intent classifier function
from backend.intent_classifier import classify_intent

logger = logging.getLogger(__name__)


# --- Node Implementations ---

# MODIFIED: Restored the intent_classifier_node
async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Classifies the user's intent to decide the agent's path.
    """
    logger.info(">>> NODE: Intent Classifier")
    user_query = state.get("user_query")
    available_tools = get_dynamic_tools(current_task_id=state.get("current_task_id"))
    tool_names = ", ".join([f"'{tool.name}'" for tool in available_tools])

    classification_output = await classify_intent(
        user_query=user_query,
        tool_names_for_prompt=tool_names,
        session_data_entry=state,
        config=config
    )
    return {
        "classified_intent": classification_output.intent,
        "intent_classifier_reasoning": classification_output.reasoning
    }

# MODIFIED: Restored the direct_qa_node
async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handles simple questions by directly calling an LLM.
    """
    logger.info(">>> NODE: Direct QA")
    user_query = state.get("user_query", "No query provided.")
    llm_id_str = state.get("executor_llm_id")
    provider, model_name = (llm_id_str.split("::", 1) if "::" in (llm_id_str or "")
                           else (settings.executor_default_provider, settings.executor_default_model_name))
    try:
        llm = get_llm(settings, provider, model_name, callbacks=config.get("callbacks"), requested_for_role="DirectQA_Node")
        prompt = ChatPromptTemplate.from_template("You are a helpful AI assistant. Answer the following question directly and concisely:\n\n{question}")
        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"question": user_query}, config=config)
        # The output is placed in a field that the overall_evaluator will use to generate the final response.
        return {"overall_evaluation_final_answer_content": answer}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"overall_evaluation_error": f"Error processing Direct QA: {e}"}


async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a multi-step plan for complex user queries.
    """
    logger.info(">>> NODE: Planner")
    user_query = state.get("user_query")
    llm_id_str = state.get("planner_llm_id")
    provider, model_name = (llm_id_str.split("::", 1) if "::" in (llm_id_str or "")
                           else (settings.planner_provider, settings.planner_model_name))
    try:
        planner_llm = get_llm(settings, provider, model_name, config.get("callbacks"), "PlannerNode")
        available_tools = get_dynamic_tools(state.get("current_task_id"))
        tools_summary = "\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools])
        
        plan = await generate_plan(user_query, tools_summary, planner_llm, config)
        
        if plan:
            summary = f"Original Query: {user_query}\nPlan Summary: {plan.human_readable_summary}\n"
            return {"plan_steps": [s.model_dump() for s in plan.steps], "plan_summary": plan.human_readable_summary, "accumulated_plan_summary": summary}
        else:
            return {"plan_generation_error": "Failed to generate a valid plan."}
    except Exception as e:
        return {"plan_generation_error": str(e)}


async def controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines the precise tool and input for the current step.
    """
    logger.info(">>> NODE: Controller")
    current_idx = state.get("current_step_index", 0)
    step_info_dict = state.get("plan_steps", [])[current_idx]
    
    try:
        current_plan_step = PlanStep(**step_info_dict)
        if state.get("retry_count_for_current_step", 0) > 0:
            current_plan_step.tool_to_use = state.get("step_evaluation_suggested_tool", current_plan_step.tool_to_use)
            current_plan_step.tool_input_instructions = state.get("step_evaluation_suggested_input_instructions", current_plan_step.tool_input_instructions)

        controller_output = await validate_and_prepare_step_action(
            original_user_query=state.get("user_query"),
            plan_step=current_plan_step,
            available_tools=get_dynamic_tools(state.get("current_task_id")),
            session_data_entry=state,
            previous_step_output=state.get("previous_step_executor_output"),
            config=config
        )
        return {
            "controller_tool_name": controller_output.tool_name,
            "controller_tool_input": controller_output.tool_input
        }
    except Exception as e:
        return {"controller_error": str(e)}

async def executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes the action determined by the Controller.
    """
    logger.info(">>> NODE: Executor")
    tool_name = state.get("controller_tool_name")
    tool_input = state.get("controller_tool_input")
    
    try:
        if tool_name and tool_name.lower() != "none":
            tool_to_execute = next(t for t in get_dynamic_tools(state.get("current_task_id")) if t.name == tool_name)
            output = await tool_to_execute.arun(tool_input, callbacks=config.get("callbacks"))
            return {"executor_output": output}
        else:
            llm_id_str = state.get("executor_llm_id")
            provider, model_name = (llm_id_str.split("::", 1) if "::" in (llm_id_str or "") 
                                   else (settings.executor_default_provider, settings.executor_default_model_name))
            llm = get_llm(settings, provider, model_name, config.get("callbacks"), "ExecutorNode_LLM")
            prompt = ChatPromptTemplate.from_template(EXECUTOR_DIRECT_LLM_PROMPT_TEMPLATE)
            chain = prompt | llm | StrOutputParser()
            response = await chain.ainvoke({"previous_step_output": state.get("previous_step_executor_output", ""), "instruction": tool_input}, config)
            return {"executor_output": response}
    except Exception as e:
        return {"executor_error_message": str(e)}

async def step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Evaluates the executor's output against the plan's expectations.
    """
    logger.info(">>> NODE: Step Evaluator")
    current_idx = state.get("current_step_index", 0)
    step_info_dict = state.get("plan_steps", [])[current_idx]
    
    try:
        current_plan_step = PlanStep(**step_info_dict)
        eval_outcome = await evaluate_step_outcome_and_suggest_correction(
            original_user_query=state.get("user_query"),
            plan_step_being_evaluated=current_plan_step,
            controller_tool_used=state.get("controller_tool_name"),
            controller_tool_input=state.get("controller_tool_input"),
            step_executor_output=state.get("executor_output", ""),
            available_tools=get_dynamic_tools(state.get("current_task_id")),
            session_data_entry=state,
            config=config
        )
        summary = state.get("accumulated_plan_summary", "") + f"\nStep {current_idx+1} Output: {str(state.get('executor_output'))[:200]}..."
        return {
            "step_evaluation_achieved_goal": eval_outcome.step_achieved_goal,
            "step_evaluation_is_recoverable": eval_outcome.is_recoverable_via_retry,
            "step_evaluation_suggested_tool": eval_outcome.suggested_new_tool_for_retry,
            "step_evaluation_suggested_input_instructions": eval_outcome.suggested_new_input_instructions_for_retry,
            "previous_step_executor_output": state.get("executor_output") if eval_outcome.step_achieved_goal else None,
            "accumulated_plan_summary": summary
        }
    except Exception as e:
        return {"step_evaluation_error": str(e)}

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Synthesizes the final answer after the plan is complete or has failed.
    """
    logger.info(">>> NODE: Overall Evaluator")
    
    final_answer = state.get("overall_evaluation_final_answer_content", "Plan execution finished.")
        
    return {"overall_evaluation_final_answer_content": final_answer}


# --- Conditional Edge Logic ---
def should_proceed_to_plan_or_qa(state: ResearchAgentState) -> str:
    """Decision point after intent classification."""
    logger.info("--- DECISION: Plan or Direct QA? ---")
    return "planner" if state.get("classified_intent") == "PLAN" else "direct_qa"

def did_planning_succeed(state: ResearchAgentState) -> str:
    """Decision point after the planner runs."""
    logger.info("--- DECISION: Planning Success or Failure? ---")
    return "failure" if state.get("plan_generation_error") else "success"

def should_retry_step_or_proceed(state: ResearchAgentState) -> str:
    """Decision point after a step is evaluated."""
    logger.info("--- DECISION: Step Outcome - Retry, Next Step, or Evaluate Overall? ---")
    if not state.get("step_evaluation_achieved_goal"):
        if state.get("step_evaluation_is_recoverable") and state.get("retry_count_for_current_step", 0) < settings.agent_max_step_retries:
            return "retry_step"
        return "evaluate_overall_plan"
    return "next_step" if state.get("current_step_index", -1) + 1 < len(state.get("plan_steps", [])) else "evaluate_overall_plan"


# --- Utility Nodes ---
def increment_retry_count(state: ResearchAgentState) -> Dict[str, Any]:
    """Increments the retry counter for the current step."""
    return {"retry_count_for_current_step": state.get("retry_count_for_current_step", 0) + 1}

def advance_to_next_step(state: ResearchAgentState) -> Dict[str, Any]:
    """Advances the step index and resets the retry counter."""
    return {"current_step_index": state.get("current_step_index", -1) + 1, "retry_count_for_current_step": 0}


# --- Graph Definition ---
def create_research_agent_graph():
    """Builds and compiles the LangGraph StateGraph."""
    workflow = StateGraph(ResearchAgentState)
    
    # MODIFIED: Add all nodes, including the restored ones
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_qa", direct_qa_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("controller", controller_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("step_evaluator", step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    workflow.add_node("increment_retry_count", increment_retry_count)
    workflow.add_node("advance_to_next_step", advance_to_next_step)

    # MODIFIED: Set the correct entry point
    workflow.set_entry_point("intent_classifier")
    
    # MODIFIED: Define all edges from the entry point
    workflow.add_conditional_edges("intent_classifier", should_proceed_to_plan_or_qa, {
        "planner": "planner", 
        "direct_qa": "direct_qa"
    })
    
    workflow.add_conditional_edges("planner", did_planning_succeed, {
        "success": "controller", 
        "failure": "overall_evaluator"
    })
    
    workflow.add_edge("direct_qa", "overall_evaluator") 
    
    # Edges for the PCEE loop
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")
    workflow.add_conditional_edges("step_evaluator", should_retry_step_or_proceed, {
        "retry_step": "increment_retry_count",
        "next_step": "advance_to_next_step",
        "evaluate_overall_plan": "overall_evaluator"
    })
    workflow.add_edge("increment_retry_count", "controller")
    workflow.add_edge("advance_to_next_step", "controller")

    # The final node
    workflow.add_edge("overall_evaluator", END)

    return workflow.compile()

research_agent_graph = create_research_agent_graph()
