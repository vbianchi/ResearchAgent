# backend/langgraph_agent.py
import logging
from typing import TypedDict, Optional, List, Annotated, Dict, Any

import asyncio 

# --- Pydantic and LangChain Core Imports ---
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# --- Project Imports ---
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.planner import generate_plan, PlanStep
from backend.graph_state import ResearchAgentState
from backend.controller import validate_and_prepare_step_action
from backend.evaluator import evaluate_step_outcome_and_suggest_correction # MODIFIED: Added evaluator import


logger = logging.getLogger(__name__)


# --- Node Implementations ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Initial node that classifies the user's intent and initializes the graph state.
    """
    logger.info(">>> NODE: Intent Classifier")
    configurable_settings = config.get("configurable", {})

    session_intent_llm_id = state.get("intent_classifier_llm_id") or configurable_settings.get("intent_classifier_llm_id")
    session_planner_llm_id = state.get("planner_llm_id") or configurable_settings.get("planner_llm_id")
    session_controller_llm_id = state.get("controller_llm_id") or configurable_settings.get("controller_llm_id")
    session_executor_llm_id = state.get("executor_llm_id") or configurable_settings.get("executor_llm_id")
    session_evaluator_llm_id = state.get("evaluator_llm_id") or configurable_settings.get("evaluator_llm_id")
    
    classified_intent = configurable_settings.get("classified_intent", state.get("classified_intent", "DIRECT_QA"))
    
    return {
        "classified_intent": classified_intent,
        "is_direct_qa_flow": classified_intent == "DIRECT_QA",
        "intent_classifier_llm_id": session_intent_llm_id,
        "planner_llm_id": session_planner_llm_id,
        "controller_llm_id": session_controller_llm_id,
        "executor_llm_id": session_executor_llm_id,
        "evaluator_llm_id": session_evaluator_llm_id,
        "current_step_index": 0,
        "retry_count_for_current_step": 0,
    }

async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handles simple questions that don't require tools by directly calling an LLM.
    """
    logger.info(">>> NODE: Direct QA")
    user_query = state.get("user_query", "No query provided.")
    chat_history = state.get("chat_history", [])
    
    llm_id_str = state.get("executor_llm_id")
    provider, model_name = (llm_id_str.split("::", 1) if "::" in (llm_id_str or "") 
                           else (settings.executor_default_provider, settings.executor_default_model_name))

    logger.info(f"DirectQANode: Using LLM {provider}::{model_name}")
    try:
        llm = get_llm(settings, provider=provider, model_name=model_name, 
                      callbacks=config.get("callbacks"), requested_for_role="DirectQA_Node")
    except Exception as e:
        logger.error(f"DirectQANode: Failed to initialize LLM: {e}")
        return {"overall_evaluation_error": f"Error: Could not initialize LLM for Direct QA. {e}"}

    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's query directly and concisely. Consider the chat history for context if relevant."),
        ("human", "Chat History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:")
    ])
    chain = prompt_template | llm
    
    try:
        response = await chain.ainvoke({"user_query": user_query, "chat_history": history_str}, config=config)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        return {"overall_evaluation_final_answer_content": answer_text, "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"overall_evaluation_error": f"Error processing Direct QA: {e}"}

async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a multi-step plan for complex user queries.
    """
    logger.info(">>> NODE: Planner")
    user_query = state.get("user_query")
    current_task_id = state.get("current_task_id")
    planner_llm_id_str = state.get("planner_llm_id")

    if not user_query: return {"plan_generation_error": "User query is missing."}

    provider, model_name = (planner_llm_id_str.split("::", 1) if "::" in (planner_llm_id_str or "") 
                           else (settings.planner_provider, settings.planner_model_name))
    try:
        planner_llm_instance = get_llm(settings, provider=provider, model_name=model_name,
                                       callbacks=config.get("callbacks"), requested_for_role="PlannerNode_LLM")
    except Exception as e:
        return {"plan_generation_error": f"Failed to initialize Planner LLM: {e}"}

    available_tools = get_dynamic_tools(current_task_id=current_task_id)
    available_tools_summary = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools]) or "No tools are currently available."
    
    session_data_for_planner = {"session_planner_llm_id": planner_llm_id_str}
    human_summary, structured_steps = await generate_plan(
        user_query, available_tools_summary, session_data_for_planner, planner_llm_instance
    )

    if human_summary and structured_steps:
        initial_summary = f"Original Query: {user_query}\nOverall Plan Summary: {human_summary}\n--- Plan Steps ---\n" + \
                          "".join([f"{i+1}. {step.get('description', 'N/A')}\n" for i, step in enumerate(structured_steps)])
        return {"plan_steps": structured_steps, "plan_summary": human_summary, "accumulated_plan_summary": initial_summary}
    else:
        return {"plan_generation_error": "Failed to generate a valid plan."}

async def controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines the precise tool and input for the current step, considering retries.
    """
    logger.info(">>> NODE: Controller")
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)

    if not (0 <= current_idx < len(plan_steps)):
        return {"controller_error": "Controller error: Invalid plan or step index."}

    step_info_dict = plan_steps[current_idx]
    try: current_plan_step = PlanStep(**step_info_dict)
    except Exception as e: return {"controller_error": f"Could not parse plan step {current_idx + 1}: {e}"}

    logger.info(f"Controller for step {current_idx + 1}: {current_plan_step.description}")

    # MODIFIED: If it's a retry, incorporate suggestions from the evaluator
    if state.get("retry_count_for_current_step", 0) > 0:
        logger.info("This is a retry attempt. Incorporating suggestions from evaluator.")
        suggested_tool = state.get("step_evaluation_suggested_tool")
        suggested_input = state.get("step_evaluation_suggested_input_instructions")
        if suggested_tool:
            current_plan_step.tool_to_use = suggested_tool
            logger.info(f"Using suggested tool for retry: {suggested_tool}")
        if suggested_input:
            current_plan_step.tool_input_instructions = suggested_input
            logger.info(f"Using suggested input instructions for retry: {suggested_input}")

    available_tools = get_dynamic_tools(current_task_id=state.get("current_task_id"))
    session_data_for_controller = {"session_controller_llm_id": state.get("controller_llm_id")}

    controller_output = await validate_and_prepare_step_action(
        state.get("user_query"), current_plan_step, available_tools, 
        session_data_for_controller, state.get("previous_step_executor_output")
    )

    if controller_output:
        return {
            "controller_tool_name": controller_output.tool_name,
            "controller_tool_input": controller_output.tool_input,
            "controller_reasoning": controller_output.reasoning,
            "controller_confidence": controller_output.confidence_score,
        }
    else:
        return {"controller_error": "Controller failed to produce a valid action."}

async def executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes the action determined by the Controller.
    """
    logger.info(">>> NODE: Executor")
    tool_name = state.get("controller_tool_name")
    tool_input = state.get("controller_tool_input")
    
    try:
        if tool_name and tool_name.lower() != "none":
            logger.info(f"ExecutorNode: Executing tool '{tool_name}'")
            available_tools = get_dynamic_tools(current_task_id=state.get("current_task_id"))
            tool_to_execute = next((t for t in available_tools if t.name == tool_name), None)
            if not tool_to_execute:
                return {"executor_error_message": f"Tool '{tool_name}' not found."}
            output = await tool_to_execute.arun(tool_input, callbacks=config.get("callbacks"))
            return {"executor_output": output}
        else:
            logger.info("ExecutorNode: No tool specified, executing direct LLM call.")
            llm_id_str = state.get("executor_llm_id")
            provider, model_name = (llm_id_str.split("::", 1) if "::" in (llm_id_str or "") 
                                   else (settings.executor_default_provider, settings.executor_default_model_name))
            executor_llm = get_llm(settings, provider, model_name, config.get("callbacks"), "ExecutorNode_LLM")
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Execute the instruction using the provided 'Previous Step Output' as context."),
                ("human", "Previous Step Output:\n---\n{previous_step_output}\n---\n\nInstruction:\n{instruction}")
            ])
            chain = prompt | executor_llm
            response = await chain.ainvoke({
                "previous_step_output": state.get("previous_step_executor_output", "N/A"),
                "instruction": tool_input
            }, config)
            return {"executor_output": response.content if hasattr(response, 'content') else str(response)}
    except Exception as e:
        logger.error(f"ExecutorNode: Error during execution (Tool: {tool_name}): {e}", exc_info=True)
        return {"executor_error_message": f"Error during execution: {e}"}

async def step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Evaluates the outcome of the executor's action against the plan's expectations.
    This is the full implementation, replacing the placeholder.
    """
    logger.info(">>> NODE: Step Evaluator")
    current_idx = state.get("current_step_index", 0)
    plan_steps = state.get("plan_steps", [])
    
    if not (0 <= current_idx < len(plan_steps)):
        return {"step_evaluation_error": "Step evaluator error: Invalid plan or step index."}

    step_info_dict = plan_steps[current_idx]
    try: current_plan_step = PlanStep(**step_info_dict)
    except Exception as e: return {"step_evaluation_error": f"Could not parse plan step {current_idx + 1}: {e}"}

    logger.info(f"Evaluating outcome of step {current_idx + 1}: {current_plan_step.description}")

    available_tools = get_dynamic_tools(current_task_id=state.get("current_task_id"))
    session_data_for_evaluator = {"session_evaluator_llm_id": state.get("evaluator_llm_id")}
    
    eval_outcome = await evaluate_step_outcome_and_suggest_correction(
        original_user_query=state.get("user_query"),
        plan_step_being_evaluated=current_plan_step,
        controller_tool_used=state.get("controller_tool_name"),
        controller_tool_input=state.get("controller_tool_input"),
        step_executor_output=state.get("executor_output", "No output from executor."),
        available_tools=available_tools,
        session_data_entry=session_data_for_evaluator
    )

    if not eval_outcome:
        return {"step_evaluation_error": "Step evaluation failed to produce a result."}

    logger.info(f"Step {current_idx + 1} Evaluation: Goal Achieved = {eval_outcome.step_achieved_goal}, Recoverable = {eval_outcome.is_recoverable_via_retry}")
    
    # Update the accumulated summary with the step's results
    accumulated_summary = state.get("accumulated_plan_summary", "")
    accumulated_summary += f"\n--- Step {current_idx + 1}: {current_plan_step.description} ---\n"
    accumulated_summary += f"Action: Tool='{state.get('controller_tool_name', 'N/A')}', Input='{str(state.get('controller_tool_input', 'N/A'))[:150]}...'\n"
    accumulated_summary += f"Output: {str(state.get('executor_output', 'N/A'))[:200]}...\n"
    accumulated_summary += f"Assessment: {eval_outcome.assessment_of_step}\n"
    
    return {
        "step_evaluation_achieved_goal": eval_outcome.step_achieved_goal,
        "step_evaluation_is_recoverable": eval_outcome.is_recoverable_via_retry,
        "step_evaluation_suggested_tool": eval_outcome.suggested_new_tool_for_retry,
        "step_evaluation_suggested_input_instructions": eval_outcome.suggested_new_input_instructions_for_retry,
        "accumulated_plan_summary": accumulated_summary,
        "previous_step_executor_output": state.get("executor_output") if eval_outcome.step_achieved_goal else None
    }

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Final node that synthesizes all information to provide a coherent answer to the user.
    """
    logger.info(">>> NODE: Overall Evaluator")
    
    final_text = "Processing complete."
    if state.get("plan_generation_error"):
        final_text = f"I'm sorry, I was unable to create a plan. Error: {state.get('plan_generation_error')}"
    elif state.get("is_direct_qa_flow"):
        final_text = state.get("overall_evaluation_final_answer_content", "I was unable to find a direct answer.")
    else:
        final_text = state.get("executor_output", "The plan has finished, but there is no final output to display.")

    llm_id_str = state.get("evaluator_llm_id")
    provider, model_name = (llm_id_str.split("::", 1) if "::" in (llm_id_str or "") 
                           else (settings.evaluator_provider, settings.evaluator_model_name))
    try:
        finalizing_llm = get_llm(settings, provider, model_name, config.get("callbacks"), "OverallEvaluatorNode")
        prompt = ChatPromptTemplate.from_template("Present the following information as the agent's final response, without adding any conversational phrases like 'Here is the answer'. Just provide the information directly:\n\n{assessment_text}")
        chain = prompt | finalizing_llm
        await chain.ainvoke({"assessment_text": final_text}, config)
        return {}
    except Exception as e:
        return {"overall_evaluation_error": f"Error finalizing response: {e}"}

# --- Conditional Edge Logic ---

def should_proceed_to_plan_or_qa(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Plan or Direct QA? ---")
    return "planner" if state.get("classified_intent") == "PLAN" else "direct_qa"
        
def did_planning_succeed(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Planning Success or Failure? ---")
    return "failure" if state.get("plan_generation_error") else "success"

def should_retry_step_or_proceed(state: ResearchAgentState) -> str:
    logger.info("--- DECISION: Step Outcome - Retry, Next Step, or Evaluate Overall? ---")
    if not state.get("step_evaluation_achieved_goal"):
        if state.get("step_evaluation_is_recoverable") and state.get("retry_count_for_current_step", 0) < settings.agent_max_step_retries:
            return "retry_step"
        else:
            return "evaluate_overall_plan"
    else:
        return "next_step" if state.get("current_step_index", -1) + 1 < len(state.get("plan_steps", [])) else "evaluate_overall_plan"

# --- Utility nodes for state updates ---
def increment_retry_count_node(state: ResearchAgentState) -> Dict[str, Any]:
    return {"retry_count_for_current_step": state.get("retry_count_for_current_step", 0) + 1}

def advance_to_next_step_node(state: ResearchAgentState) -> Dict[str, Any]:
    return {"current_step_index": state.get("current_step_index", -1) + 1, "retry_count_for_current_step": 0}

# --- Graph Definition Function ---
def create_research_agent_graph():
    logger.info("Building Research Agent Graph...")
    workflow = StateGraph(ResearchAgentState)

    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_qa", direct_qa_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("controller", controller_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("step_evaluator", step_evaluator_node) # MODIFIED: Using new evaluator
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    workflow.add_node("increment_retry_count", increment_retry_count_node)
    workflow.add_node("advance_to_next_step", advance_to_next_step_node)

    workflow.set_entry_point("intent_classifier")
    workflow.add_conditional_edges("intent_classifier", should_proceed_to_plan_or_qa, {"planner": "planner", "direct_qa": "direct_qa"})
    workflow.add_conditional_edges("planner", did_planning_succeed, {"success": "controller", "failure": "overall_evaluator"})
    workflow.add_edge("direct_qa", "overall_evaluator") 
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")
    workflow.add_conditional_edges("step_evaluator", should_retry_step_or_proceed, {"retry_step": "increment_retry_count", "next_step": "advance_to_next_step", "evaluate_overall_plan": "overall_evaluator"})
    workflow.add_edge("increment_retry_count", "controller")
    workflow.add_edge("advance_to_next_step", "controller")
    workflow.add_edge("overall_evaluator", END)

    logger.info("Compiling Research Agent Graph...")
    return workflow.compile()

research_agent_graph = create_research_agent_graph()
logger.info(f"Module-level variable 'research_agent_graph' (compiled graph app) created.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running langgraph_agent.py directly for structural check...")
    
    print("\n--- Graph Definition (ASCII) ---")
    try: research_agent_graph.get_graph().print_ascii()
    except Exception as e: print(f"Could not print ASCII graph: {e}")

    print("\n--- State Schema (JSON) ---")
    try:
        if hasattr(ResearchAgentState, 'schema_json'): print(ResearchAgentState.schema_json(indent=2))
    except Exception as e: print(f"Could not print state schema: {e}")
    
    print("\nScript finished.")
