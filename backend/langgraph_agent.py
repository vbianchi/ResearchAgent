# backend/langgraph_agent.py
import logging
from typing import TypedDict, Optional, List, Dict, Any
import asyncio 

from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools, ToolLoadingError 

logger = logging.getLogger(__name__)

# --- Define State ---
class ResearchAgentState(TypedDict, total=False):
    user_query: str
    classified_intent: Optional[str]
    identified_tool_name: Optional[str]
    extracted_tool_input: Optional[str]
    
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
    direct_tool_request_error: Optional[str] 


# --- Node Implementations ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Intent Classifier (Graph Node)")
    configurable_settings = config.get("configurable", {})
    
    intent = state.get("classified_intent", "DIRECT_QA") 
    is_direct_flow_flag = intent in ["DIRECT_QA", "DIRECT_TOOL_REQUEST"]
    
    update_dict = {
        "classified_intent": intent,
        "identified_tool_name": state.get("identified_tool_name"), 
        "extracted_tool_input": state.get("extracted_tool_input"), 
        "current_step_index": 0, 
        "retry_count_for_current_step": 0, 
        "is_direct_qa_flow": is_direct_flow_flag, 
        "intent_classifier_llm_id": configurable_settings.get("intent_classifier_llm_id", state.get("intent_classifier_llm_id")),
        "planner_llm_id": configurable_settings.get("planner_llm_id", state.get("planner_llm_id")),
        "controller_llm_id": configurable_settings.get("controller_llm_id", state.get("controller_llm_id")),
        "executor_llm_id": configurable_settings.get("executor_llm_id", state.get("executor_llm_id")),
        "evaluator_llm_id": configurable_settings.get("evaluator_llm_id", state.get("evaluator_llm_id")),
    }
    logger.info(f"Intent Classifier Node: Outputting state update: { {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in update_dict.items()} }")
    return update_dict


async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Direct QA (LLM Only)")
    user_query = state.get("user_query", "No query provided.")
    chat_history_messages = state.get("chat_history", [])
    
    llm_id_str = state.get("executor_llm_id") 
    provider = settings.executor_default_provider 
    model_name = settings.executor_default_model_name 

    if llm_id_str:
        try:
            provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"DirectQANode: Invalid LLM ID format '{llm_id_str}'. Using system default.")
    else:
        logger.warning("DirectQANode: executor_llm_id not found in state. Using system default.")

    logger.info(f"DirectQANode: Using LLM {provider}::{model_name}")
    
    try:
        llm = get_llm(settings, provider=provider, model_name=model_name, requested_for_role="DirectQA_Node")
    except Exception as e:
        logger.error(f"DirectQANode: Failed to initialize LLM: {e}")
        return {"executor_output": f"Error: Could not initialize LLM for Direct QA. {e}", "is_direct_qa_flow": True}

    history_str_parts = []
    for msg in chat_history_messages:
        if isinstance(msg, HumanMessage):
            history_str_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_str_parts.append(f"AI: {msg.content}")
    history_str = "\n".join(history_str_parts)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's query directly and concisely. Consider the chat history for context if relevant."),
        ("human", "Chat History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:")
    ])
    
    chain = prompt_template | llm
    
    logger.info(f"DirectQANode: Invoking LLM for query: '{user_query}' with history.")
    try:
        response = await chain.ainvoke({"user_query": user_query, "chat_history": history_str}, config=config) 
        answer_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"DirectQANode: LLM produced answer (raw): {answer_text[:200]}...")
        return {"executor_output": answer_text, "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"executor_output": f"Error processing Direct QA: {e}", "is_direct_qa_flow": True}


async def direct_tool_executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Direct Tool Executor")
    task_id = state.get("current_task_id")
    tool_name = state.get("identified_tool_name")
    tool_input_str = state.get("extracted_tool_input")

    if not task_id:
        logger.error("DirectToolExecutorNode: Missing task_id in state. Cannot load tools.")
        return {"executor_output": "Error: Task context missing for tool execution.", "is_direct_qa_flow": True, "direct_tool_request_error": "Task ID missing."}
    if not tool_name:
        logger.error("DirectToolExecutorNode: Missing identified_tool_name in state.")
        return {"executor_output": "Error: Tool name not identified for execution.", "is_direct_qa_flow": True, "direct_tool_request_error": "Tool name missing."}

    logger.info(f"DirectToolExecutorNode: Attempting to execute tool '{tool_name}' with input '{str(tool_input_str)[:100]}...' for task '{task_id}'.")

    try:
        available_tools = get_dynamic_tools(current_task_id=task_id)
        logger.debug(f"DirectToolExecutorNode: Loaded {len(available_tools)} tools for task '{task_id}'.")
    except ToolLoadingError as e:
        logger.error(f"DirectToolExecutorNode: Failed to load tools for task '{task_id}': {e}", exc_info=True)
        return {"executor_output": f"Error: Could not load tools for task. {e}", "is_direct_qa_flow": True, "direct_tool_request_error": f"Tool loading failed: {e}"}

    selected_tool = next((t for t in available_tools if t.name == tool_name), None)
    
    if not selected_tool:
        logger.error(f"DirectToolExecutorNode: Tool '{tool_name}' not found in available tools for task '{task_id}'.")
        tool_names = [t.name for t in available_tools]
        return {"executor_output": f"Error: Tool '{tool_name}' is not available. Available tools: {tool_names}", "is_direct_qa_flow": True, "direct_tool_request_error": f"Tool '{tool_name}' not found."}

    try:
        logger.info(f"DirectToolExecutorNode: Executing tool '{selected_tool.name}' with input: {tool_input_str}")
        tool_result = await selected_tool.arun(tool_input_str, config=config) 
        logger.info(f"DirectToolExecutorNode: Tool '{tool_name}' executed. Output length: {len(str(tool_result))}. Output snippet: {str(tool_result)[:200]}...")
        return {"executor_output": str(tool_result), "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectToolExecutorNode: Error executing tool '{tool_name}': {e}", exc_info=True)
        return {"executor_output": f"Error during execution of tool '{tool_name}': {e}", "is_direct_qa_flow": True, "direct_tool_request_error": f"Tool execution error: {e}"}


async def placeholder_planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Planner (Placeholder)")
    plan = [{"step_id": 1, "description": "Placeholder: Do step 1", "tool_to_use": "some_tool", "expected_outcome": "Step 1 done"}]
    logger.info(f"Plan generated: {plan}")
    return {"plan_steps": plan, "current_step_index": 0, "retry_count_for_current_step": 0, "accumulated_plan_summary": "Plan:\n1. Do step 1\n", "is_direct_qa_flow": False}

async def placeholder_controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Controller (Placeholder)")
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    step_info = plan_steps[current_idx] if plan_steps and 0 <= current_idx < len(plan_steps) else {}
    logger.info(f"Controller for step {current_idx + 1}: {step_info.get('description')}")
    return {"controller_output_tool_name": step_info.get("tool_to_use", "None"), 
            "controller_output_tool_input": "placeholder_input_for_tool"}

async def placeholder_executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Executor (Placeholder - For PLAN path)")
    tool_name = state.get("controller_output_tool_name", "None")
    tool_input = state.get("controller_output_tool_input", "")
    logger.info(f"Executing tool (Plan Path): {tool_name} with input: {tool_input}")
    output = f"Placeholder output from {tool_name} (Plan Path)."
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
    
    is_direct_flow = state.get("is_direct_qa_flow", False) 
    final_content_to_present = state.get("executor_output", "No direct answer or tool output was generated.")

    if is_direct_flow and state.get("direct_tool_request_error"):
        final_content_to_present = f"Error in direct tool request: {state.get('direct_tool_request_error')}\nFallback output: {final_content_to_present}"
        logger.warning(f"OverallEvaluatorNode: Direct tool request had an error: {state.get('direct_tool_request_error')}")
    elif not is_direct_flow : # Plan execution flow
        logger.info(f"OverallEvaluatorNode: Processing plan execution outcome.")
        logger.info(f"Accumulated Plan Summary for Eval:\n{state.get('accumulated_plan_summary')}")
        # For plan, final_content_to_present is already the last step's output or overall summary
    
    logger.info(f"OverallEvaluatorNode: Content to potentially present to LLM: {final_content_to_present[:200]}...")
    
    llm_id_str = state.get("evaluator_llm_id") 
    provider = settings.evaluator_provider 
    model_name = settings.evaluator_model_name 
    if llm_id_str:
        try: provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"OverallEvaluatorNode: Invalid LLM ID format '{llm_id_str}'. Using system default.")
    else:
        logger.warning("OverallEvaluatorNode: evaluator_llm_id not found in state. Using system default.")
            
    logger.info(f"OverallEvaluatorNode: Using LLM {provider}::{model_name} to finalize output.")
    try:
        finalizing_llm = get_llm(settings, provider=provider, model_name=model_name, requested_for_role="OverallEvaluator_Finalize")
        
        # --- MODIFICATION START: Conditional Prompting ---
        prompt_template_str = "Present the following information as the agent's final response: {assessment_text}" # Default
        
        # Check if the content came from a 'read_file' tool in a direct tool request
        is_read_file_direct_request = (
            state.get("classified_intent") == "DIRECT_TOOL_REQUEST" and
            state.get("identified_tool_name") == "read_file"
        )

        if is_read_file_direct_request:
            logger.info("OverallEvaluatorNode: Detected 'read_file' output for direct tool request. Using specific presentation prompt.")
            
            MAX_FILE_CONTENT_DISPLAY_LENGTH = 15000 # Max characters to show from file directly to LLM for presentation
            original_length = len(final_content_to_present)

            if original_length > MAX_FILE_CONTENT_DISPLAY_LENGTH:
                final_content_to_present = final_content_to_present[:MAX_FILE_CONTENT_DISPLAY_LENGTH] + \
                                           f"\n\n[... Content truncated by agent. Original length: {original_length} characters. Full content available in workspace file.]"
                logger.info(f"OverallEvaluatorNode: Truncated 'read_file' content for LLM presentation. Original: {original_length}, Truncated: {len(final_content_to_present)}")

            prompt_template_str = (
                "The user requested to read a file. The content (or a truncated version if it was very long) is provided below. "
                "Present this content directly as the agent's response. Avoid conversational additions unless the content itself is conversational. "
                "If the content is code or structured data, preserve its formatting as much as possible using Markdown code blocks if appropriate.\n\n"
                "File Content:\n---\n{assessment_text}\n---"
            )
        # --- MODIFICATION END ---
        
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        chain = prompt | finalizing_llm
        
        logger.info(f"OverallEvaluatorNode: Invoking LLM to finalize assessment (snippet): '{final_content_to_present[:100]}...'")
        final_response_message_obj = await chain.ainvoke({"assessment_text": final_content_to_present}, config=config) 
        
        final_content_for_state = final_response_message_obj.content if hasattr(final_response_message_obj, 'content') else str(final_response_message_obj)
        logger.info(f"OverallEvaluatorNode: Finalized response for state (snippet): {final_content_for_state[:200]}...")
        return {"overall_evaluator_output": {"assessment": final_content_for_state, "overall_success": True}}

    except Exception as e:
        logger.error(f"OverallEvaluatorNode: Error during final LLM invocation: {e}", exc_info=True)
        # Fallback: use the raw content without LLM finalization if LLM fails
        return {"overall_evaluator_output": {"assessment": f"Error finalizing response. Original content: {final_content_to_present}", "overall_success": False}}


# --- Conditional Edge Logic ---
def should_proceed_to_plan_or_qa_or_tool(state: ResearchAgentState, config: RunnableConfig) -> str: 
    logger.info("--- DECISION: Plan, Direct QA, or Direct Tool Request? ---")
    intent = state.get("classified_intent")
    if intent == "PLAN":
        logger.info(f"Intent is '{intent}', proceeding to Planner.")
        return "planner"
    elif intent == "DIRECT_TOOL_REQUEST":
        logger.info(f"Intent is '{intent}', proceeding to Direct Tool Executor.")
        return "direct_tool_executor" 
    elif intent == "DIRECT_QA":
        logger.info(f"Intent is '{intent}', proceeding to Direct QA (LLM only).")
        return "direct_qa"
    else:
        logger.warning(f"Unknown intent '{intent}', defaulting to Direct QA.")
        return "direct_qa" 

def should_retry_step_or_proceed(state: ResearchAgentState, config: RunnableConfig) -> str:
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
            return "evaluate_overall_plan"
    else:
        logger.info("Step succeeded. Checking for more steps in the plan.")
        plan_steps = state.get("plan_steps", []) # Ensure plan_steps is treated as a list
        current_idx = state.get("current_step_index", -1)
        # Check if plan_steps is not None and current_idx is valid before accessing length
        if plan_steps and current_idx + 1 < len(plan_steps):
            logger.info(f"More steps exist. Will proceed to step {current_idx + 2} (index {current_idx + 1}).")
            return "next_step"
        else:
            logger.info("No more steps in plan or plan_steps is empty/None. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"

# --- Utility nodes for state updates (for PLAN path) ---
def increment_retry_count_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Incrementing Retry Count")
    current_retries = state.get("retry_count_for_current_step", 0)
    return {"retry_count_for_current_step": current_retries + 1}

def advance_to_next_step_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Advancing to Next Step")
    current_idx = state.get("current_step_index", -1)
    return {"current_step_index": current_idx + 1, "retry_count_for_current_step": 0}

# --- Graph Definition Function ---
def create_research_agent_graph():
    logger.info("Building Research Agent Graph...")
    workflow = StateGraph(ResearchAgentState)

    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_qa", direct_qa_node) 
    workflow.add_node("direct_tool_executor", direct_tool_executor_node) 
    workflow.add_node("planner", placeholder_planner_node) 
    workflow.add_node("controller", placeholder_controller_node) 
    workflow.add_node("executor", placeholder_executor_node) 
    workflow.add_node("step_evaluator", placeholder_step_evaluator_node) 
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    
    workflow.add_node("increment_retry_count", increment_retry_count_node)
    workflow.add_node("advance_to_next_step", advance_to_next_step_node)

    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        should_proceed_to_plan_or_qa_or_tool, 
        {
            "planner": "planner", 
            "direct_tool_executor": "direct_tool_executor", 
            "direct_qa": "direct_qa"
        }
    )

    workflow.add_edge("direct_qa", "overall_evaluator") 
    workflow.add_edge("direct_tool_executor", "overall_evaluator") 

    workflow.add_edge("planner", "controller")
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

research_agent_graph = create_research_agent_graph()
logger.info(f"Module-level variable 'research_agent_graph' (compiled graph app) created. Type: {type(research_agent_graph)}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
    )
    logger.info("Running langgraph_agent.py directly for testing graph compilation and structure.")
    logger.info("langgraph_agent.py finished execution if run as __main__.")

