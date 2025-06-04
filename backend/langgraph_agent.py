# backend/langgraph_agent.py
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, TypedDict, Annotated, Union
from uuid import uuid4
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # For persistence - Commented out due to ModuleNotFoundError
from langchain_core.runnables import RunnableConfig


# Project specific imports
from backend.config import settings
from backend.intent_classifier import classify_intent 
from backend.planner import generate_plan, PlanStep
from backend.controller import validate_and_prepare_step_action, ControllerOutput
from backend.evaluator import (
    evaluate_step_outcome_and_suggest_correction, StepCorrectionOutcome,
    evaluate_plan_outcome, EvaluationResult
)
from backend.tools.standard_tools import get_dynamic_tools
from backend.tools.standard_tools import get_task_workspace_path 

# --- Logging Setup ---
logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - Line %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


# --- Constants ---
MAX_STEP_RETRIES = settings.agent_max_step_retries

# --- State Definition for the Graph ---
class ResearchAgentState(TypedDict):
    user_query: str
    messages: Annotated[List[BaseMessage], operator.add]
    task_id: str
    classified_intent: Optional[str]
    intent_classifier_reasoning: Optional[str]
    plan_summary: Optional[str]
    plan_steps: Optional[List[PlanStep]]
    plan_generation_error: Optional[str]
    current_step_index: int
    previous_step_executor_output: Optional[str]
    retry_count_for_current_step: int
    controller_tool_name: Optional[str]
    controller_tool_input: Optional[Union[str, Dict[str, Any]]]
    controller_reasoning: Optional[str]
    controller_confidence: Optional[float]
    controller_error: Optional[str]
    current_executor_output: Optional[str]
    executor_error_message: Optional[str]
    step_evaluation_achieved_goal: Optional[bool]
    step_evaluation_assessment: Optional[str]
    step_evaluation_is_recoverable: Optional[bool]
    step_evaluation_suggested_tool: Optional[str]
    step_evaluation_suggested_input_instructions: Optional[str]
    step_evaluation_confidence_in_correction: Optional[float]
    step_evaluation_error: Optional[str]
    overall_evaluation_success: Optional[bool]
    overall_evaluation_assessment: Optional[str]
    overall_evaluation_final_answer_content: Optional[str]
    overall_evaluation_suggestions_for_replan: Optional[List[str]]
    overall_evaluation_error: Optional[str]
    error_message: Optional[str]
    session_intent_classifier_llm_id: Optional[str]
    session_planner_llm_id: Optional[str]
    session_controller_llm_id: Optional[str]
    session_executor_llm_id: Optional[str]
    session_evaluator_llm_id: Optional[str]


# --- Node Functions ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- Entering Intent Classifier Node ---")
    user_query = state["user_query"]
    task_id = state.get("task_id", "default_task_for_intent") 

    available_tools = get_dynamic_tools(task_id)
    tools_summary_for_intent = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools])
    
    intent_str_output = await classify_intent(
        user_query=user_query,
        available_tools_summary=tools_summary_for_intent
    )
    
    classified_intent_str: str
    reasoning_str: str 

    if isinstance(intent_str_output, str) and intent_str_output in ["PLAN", "DIRECT_QA"]:
        classified_intent_str = intent_str_output
        reasoning_str = f"Intent classified as {classified_intent_str} by intent_classifier module (detailed reasoning is logged by that module)."
    else: 
        # This case should ideally not be hit if classify_intent is robust.
        # The warning from the log: "classify_intent returned unexpected value: 'intent='PLAN' reasoning=..."
        # indicates that classify_intent might be returning a formatted string instead of just "PLAN" or "DIRECT_QA".
        # For robustness, let's try to parse it if it looks like the log string.
        raw_output_str = str(intent_str_output)
        logger.warning(f"intent_classifier_node: classify_intent returned an unexpected value/type: '{raw_output_str}'. Attempting to parse.")
        if "intent='PLAN'" in raw_output_str:
            classified_intent_str = "PLAN"
            reasoning_str = raw_output_str # Store the full string as reasoning for now
        elif "intent='DIRECT_QA'" in raw_output_str:
            classified_intent_str = "DIRECT_QA"
            reasoning_str = raw_output_str
        else:
            classified_intent_str = "PLAN" # Default
            reasoning_str = f"classify_intent returned unparsable value '{raw_output_str}', defaulted to PLAN."

    logger.info(f"Intent classification result in langgraph_agent: intent='{classified_intent_str}', reasoning_detail='{reasoning_str}'")
    return {
        "classified_intent": classified_intent_str,
        "intent_classifier_reasoning": reasoning_str, 
        "messages": [AIMessage(content=f"Intent Classified as: {classified_intent_str}\n(Reasoning: {reasoning_str})")]
    }


async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- Entering Planner Node ---")
    if state["classified_intent"] != "PLAN":
        logger.info("Planner Node: Intent is not PLAN, skipping plan generation.")
        return {"plan_summary": "No plan needed.", "plan_steps": []}

    user_query = state["user_query"]
    task_id = state.get("task_id", "default_task_for_planner")
    logger.info(f"Planner Node: current_task_id_for_tools from state: {task_id}")

    available_tools = get_dynamic_tools(task_id)
    tools_summary = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools])

    plan_result = await generate_plan(
        user_query=user_query,
        available_tools_summary=tools_summary,
    )
    logger.info(f"Planner Node (DEBUG): Raw result from generate_plan - Type: {type(plan_result)}, Value: '{str(plan_result)[:500]}...'")

    if plan_result and isinstance(plan_result, tuple) and len(plan_result) == 2:
        summary, steps_dicts = plan_result
        if summary and steps_dicts is not None: 
            plan_steps_models = [PlanStep(**step_data) for step_data in steps_dicts]
            logger.info(f"Planner Node: Plan generated. Summary: {summary}")
            return {
                "plan_summary": summary,
                "plan_steps": plan_steps_models,
                "current_step_index": 0, 
                "retry_count_for_current_step": 0,
                "messages": [AIMessage(content=f"Plan Generated:\nSummary: {summary}\nSteps:\n" + "\n".join(f"{i+1}. {s.description}" for i, s in enumerate(plan_steps_models)))]
            }
    
    logger.error(f"Planner Node: Failed to generate plan or generate_plan returned unexpected structure. Result: {plan_result}")
    return {
        "plan_summary": "Failed to generate a plan.",
        "plan_steps": [],
        "plan_generation_error": "Planner failed to produce a valid plan structure or returned unexpected data.",
        "messages": [AIMessage(content="Error: Failed to generate a plan.")]
    }

async def controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- Entering Controller Node ---")
    current_step_idx = state["current_step_index"]
    plan_steps = state.get("plan_steps", [])
    task_id = state.get("task_id", "default_task_for_controller")
    
    if not plan_steps or current_step_idx >= len(plan_steps):
        logger.warning("Controller Node: No plan steps or index out of bounds. Cannot proceed.")
        return {"controller_error": "No plan steps or index out of bounds."}

    current_plan_step = plan_steps[current_step_idx]
    attempt_number = state.get("retry_count_for_current_step", 0) + 1
    logger.info(f"Controller Node: Processing Step {current_step_idx + 1}/{len(plan_steps)}: '{current_plan_step.description}' (Attempt: {attempt_number})")

    if state.get("retry_count_for_current_step", 0) >= MAX_STEP_RETRIES and attempt_number > MAX_STEP_RETRIES : 
        logger.warning(f"Controller Node: Entered for Step {current_step_idx + 1} which has already reached max retries ({state.get('retry_count_for_current_step', 0)} >= {MAX_STEP_RETRIES}). This indicates a potential routing issue if this is not the final allowed attempt.")

    available_tools = get_dynamic_tools(task_id)
    
    plan_step_for_controller_call = current_plan_step.copy(deep=True)
    if state.get("retry_count_for_current_step", 0) > 0:
        logger.info(f"Controller Node: This is a retry (attempt {attempt_number}). Using evaluator suggestions if available.")
        suggested_tool = state.get("step_evaluation_suggested_tool")
        suggested_input_instr = state.get("step_evaluation_suggested_input_instructions")

        if suggested_tool is not None: 
            plan_step_for_controller_call.tool_to_use = suggested_tool if suggested_tool != "None" else None
            logger.info(f"  Retry: Overriding tool to: '{plan_step_for_controller_call.tool_to_use}'")
        if suggested_input_instr is not None:
            plan_step_for_controller_call.tool_input_instructions = suggested_input_instr
            logger.info(f"  Retry: Overriding input instructions to: '{suggested_input_instr[:100]}...'")
            
    # Using positional arguments as a diagnostic for the TypeError
    controller_output_model_or_tuple: Optional[Union[ControllerOutput, Tuple]]
    controller_output_model_or_tuple = await validate_and_prepare_step_action(
        state["user_query"],                       
        plan_step_for_controller_call,             
        available_tools,                           
        state,                                     
        state.get("previous_step_executor_output") 
    )

    # Ensure controller_output_model is of ControllerOutput type
    controller_output_model: Optional[ControllerOutput] = None
    if isinstance(controller_output_model_or_tuple, ControllerOutput):
        controller_output_model = controller_output_model_or_tuple
    elif isinstance(controller_output_model_or_tuple, tuple) and len(controller_output_model_or_tuple) == 4:
        # This path is if validate_and_prepare_step_action was an older version returning a tuple
        # For safety, try to parse it into the Pydantic model. This may not be hit if it's truly None.
        try:
            tool_name, tool_input, reasoning, confidence = controller_output_model_or_tuple
            controller_output_model = ControllerOutput(
                tool_name=tool_name, 
                tool_input=tool_input, 
                reasoning=reasoning, 
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Controller Node: Error converting tuple output from validate_and_prepare_step_action to ControllerOutput: {e}")
            controller_output_model = None # Failed to parse
    elif controller_output_model_or_tuple is None:
        logger.info("Controller Node: validate_and_prepare_step_action returned None.")
        # controller_output_model remains None
    else:
        logger.error(f"Controller Node: Unexpected return type from validate_and_prepare_step_action: {type(controller_output_model_or_tuple)}")
        # controller_output_model remains None


    if controller_output_model:
        logger.info(f"Controller LLM decided: Tool='{controller_output_model.tool_name}', Input (summary)='{str(controller_output_model.tool_input)[:100]}...', Confidence={controller_output_model.confidence_score:.2f}")
        return {
            "controller_tool_name": controller_output_model.tool_name,
            "controller_tool_input": controller_output_model.tool_input,
            "controller_reasoning": controller_output_model.reasoning,
            "controller_confidence": controller_output_model.confidence_score,
            "messages": [AIMessage(content=f"Controller for Step {current_step_idx + 1} (Attempt {attempt_number}):\n  Tool: {controller_output_model.tool_name or 'None'}\n  Input (summary): {str(controller_output_model.tool_input)[:100]}...\n  Reasoning: {controller_output_model.reasoning[:100]}...")]
        }
    else:
        logger.error("Controller Node: Failed to get valid output from controller LLM or validate_and_prepare_step_action.")
        return {"controller_error": "Controller LLM failed to produce valid structured output or encountered an internal error."}


async def executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- Entering Executor Node ---")
    tool_name = state.get("controller_tool_name")
    tool_input = state.get("controller_tool_input")
    task_id = state.get("task_id", "default_task_for_executor")
    current_step_idx = state["current_step_index"]
    plan_steps = state.get("plan_steps", [])
    current_plan_step_desc = plan_steps[current_step_idx].description if plan_steps and current_step_idx < len(plan_steps) else "N/A"
    attempt_number = state.get("retry_count_for_current_step", 0) + 1

    logger.info(f"Executor Node: Task ID '{task_id}', Step {current_step_idx + 1} ('{current_plan_step_desc}'), Attempt {attempt_number}. Tool: '{tool_name}', Input: '{str(tool_input)[:100]}...'")

    available_tools = get_dynamic_tools(task_id)
    tool_map = {tool.name: tool for tool in available_tools}

    executor_output: str
    if tool_name and tool_name != "None":
        if tool_name in tool_map:
            selected_tool = tool_map[tool_name]
            logger.info(f"Executor Node: Executing tool '{tool_name}'")
            try:
                if asyncio.iscoroutinefunction(selected_tool.arun):
                    tool_result = await selected_tool.arun(tool_input, config=config)
                else: 
                    tool_result = await asyncio.to_thread(selected_tool.run, tool_input, config=config) 
                executor_output = str(tool_result)
                logger.info(f"Executor Node: Tool '{tool_name}' executed. Output length: {len(executor_output)}")
            except Exception as e:
                logger.error(f"Executor Node: Error executing tool '{tool_name}': {e}", exc_info=True)
                executor_output = f"Error executing tool {tool_name}: {type(e).__name__} - {str(e)}"
                return {"current_executor_output": executor_output, "executor_error_message": executor_output}
        else:
            logger.error(f"Executor Node: Tool '{tool_name}' not found in available tools.")
            executor_output = f"Error: Tool '{tool_name}' not found."
            return {"current_executor_output": executor_output, "executor_error_message": executor_output}
    else: 
        logger.info(f"Executor Node: 'None' tool specified. Executing direct LLM call with input: {str(tool_input)[:100]}...")
        try:
            from backend.llm_setup import get_llm 
            
            executor_llm_provider = state.get("session_executor_llm_id", settings.executor_default_provider)
            executor_llm_model = settings.executor_default_model_name 
            
            if state.get("session_executor_llm_id"):
                try:
                    provider, model = state.get("session_executor_llm_id").split("::",1)
                    executor_llm_provider = provider
                    executor_llm_model = model
                except ValueError:
                    logger.warning(f"Could not parse session_executor_llm_id: {state.get('session_executor_llm_id')}. Using defaults.")

            direct_llm = get_llm(settings, provider=executor_llm_provider, model_name=executor_llm_model, requested_for_role="EXECUTOR_DirectLLM")
            
            prompt_messages = []
            if state.get("previous_step_executor_output"):
                prompt_messages.append(HumanMessage(content=f"Context from previous step's output:\n{state['previous_step_executor_output']}"))
            
            prompt_messages.append(HumanMessage(content=str(tool_input))) 

            response = await direct_llm.ainvoke(prompt_messages, config=config)
            executor_output = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Executor Node: Direct LLM call completed. Output length: {len(executor_output)}")
        except Exception as e:
            logger.error(f"Executor Node: Error during direct LLM call: {e}", exc_info=True)
            executor_output = f"Error during direct LLM call: {type(e).__name__} - {str(e)}"
            return {"current_executor_output": executor_output, "executor_error_message": executor_output}

    return {
        "current_executor_output": executor_output,
        "messages": [AIMessage(content=f"Executor (Step {current_step_idx + 1}, Attempt {attempt_number}):\nTool: {tool_name or 'None'}\nOutput: {executor_output[:200]}...")]
    }


async def step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- Entering Step Evaluator Node ---")
    current_step_idx = state["current_step_index"]
    plan_steps = state.get("plan_steps", [])
    task_id = state.get("task_id", "default_task_for_evaluator")
    attempt_number = state.get("retry_count_for_current_step", 0) + 1 

    if not plan_steps or current_step_idx >= len(plan_steps):
        logger.error("Step Evaluator Node: No plan steps or index out of bounds.")
        return {"step_evaluation_error": "No plan steps or index out of bounds."}

    current_plan_step = plan_steps[current_step_idx]
    logger.info(f"Step Evaluator Node: Evaluating Step {current_step_idx + 1} (Attempt {attempt_number}): {current_plan_step.description}")

    available_tools = get_dynamic_tools(task_id)
    
    # --- MODIFIED: Call evaluate_step_outcome_and_suggest_correction with positional arguments ---
    evaluation_result_model: Optional[StepCorrectionOutcome] = await evaluate_step_outcome_and_suggest_correction(
        state["user_query"],                                # original_user_query
        current_plan_step,                                  # plan_step_being_evaluated
        state.get("controller_tool_name"),                  # controller_tool_used
        state.get("controller_tool_input"),                 # controller_tool_input
        state.get("current_executor_output", "No output from executor."), # step_executor_output
        available_tools,                                    # available_tools
        state                                               # session_data_entry (passing whole state)
    )
    # --- END MODIFIED SECTION ---


    if not evaluation_result_model:
        logger.error("Step Evaluator Node: Failed to get evaluation result from LLM.")
        return {
            "step_evaluation_achieved_goal": False,
            "step_evaluation_assessment": "Evaluator LLM failed to produce an assessment.",
            "step_evaluation_is_recoverable": False, 
            "step_evaluation_error": "Evaluator LLM failed.",
            "messages": [AIMessage(content=f"Step Evaluator (Step {current_step_idx + 1}, Attempt {attempt_number}):\n  Error: Evaluation LLM failed.")]
        }

    logger.info(f"Evaluator (Step): Achieved: {evaluation_result_model.step_achieved_goal}, Recoverable: {evaluation_result_model.is_recoverable_via_retry}, Assessment: {evaluation_result_model.assessment_of_step[:100]}...")

    patch: Dict[str, Any] = {
        "step_evaluation_achieved_goal": evaluation_result_model.step_achieved_goal,
        "step_evaluation_assessment": evaluation_result_model.assessment_of_step,
        "step_evaluation_is_recoverable": evaluation_result_model.is_recoverable_via_retry,
        "step_evaluation_suggested_tool": evaluation_result_model.suggested_new_tool_for_retry,
        "step_evaluation_suggested_input_instructions": evaluation_result_model.suggested_new_input_instructions_for_retry,
        "step_evaluation_confidence_in_correction": evaluation_result_model.confidence_in_correction,
        "messages": [AIMessage(content=f"Step Evaluator (Step {current_step_idx + 1}, Attempt {attempt_number}):\n  Achieved Goal: {evaluation_result_model.step_achieved_goal}\n  Assessment: {evaluation_result_model.assessment_of_step[:150]}...")]
    }

    if evaluation_result_model.step_achieved_goal:
        logger.info(f"Step Evaluator: Step {current_step_idx + 1} successful. Preparing for next step {current_step_idx + 2}.")
        patch["current_step_index"] = current_step_idx + 1
        patch["previous_step_executor_output"] = state.get("current_executor_output") 
        patch["retry_count_for_current_step"] = 0 
    else:
        logger.warning(f"Step Evaluator: Step {current_step_idx + 1} (Attempt {attempt_number}) failed.")
        current_retry_count_for_this_step_before_this_attempt = state.get("retry_count_for_current_step", 0)
        patch["retry_count_for_current_step"] = current_retry_count_for_this_step_before_this_attempt + 1
        patch["previous_step_executor_output"] = None 

        if evaluation_result_model.is_recoverable_via_retry:
            logger.info(f"  Step Evaluator: Preparing for retry attempt {patch['retry_count_for_current_step']} for step {current_step_idx + 1}.")
        else:
            logger.warning(f"  Step Evaluator: Step {current_step_idx + 1} failed and is not recoverable.")

    return patch


async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- Entering Overall Evaluator Node ---")
    user_query = state["user_query"]
    plan_summary = state.get("plan_summary", "N/A")
    plan_steps = state.get("plan_steps", [])
    
    executed_plan_summary_str = f"Plan Summary: {plan_summary}\n"
    if plan_steps:
        executed_plan_summary_str += "Steps Attempted:\n"
        for i, step in enumerate(plan_steps):
            executed_plan_summary_str += f"  {i+1}. {step.description}\n"
            if i == state["current_step_index"] -1 : 
                 executed_plan_summary_str += f"    Last Status: {'Successful' if state.get('step_evaluation_achieved_goal') else 'Failed'}. Assessment: {state.get('step_evaluation_assessment', 'N/A')}\n"

    final_agent_answer_for_eval = state.get("current_executor_output") 
    if state.get("executor_error_message"):
        final_agent_answer_for_eval = f"Execution failed with error: {state.get('executor_error_message')}"
    elif state.get("step_evaluation_error"):
         final_agent_answer_for_eval = f"Step evaluation failed: {state.get('step_evaluation_error')}"
    elif not state.get("step_evaluation_achieved_goal") and state.get("step_evaluation_assessment"):
        final_agent_answer_for_eval = f"Last step failed. Assessment: {state.get('step_evaluation_assessment')}"
    elif not final_agent_answer_for_eval: 
        final_agent_answer_for_eval = state.get("step_evaluation_assessment", "Plan concluded without a specific final output from the last step.")

    logger.info(f"Overall Evaluator: Evaluating plan for query: '{user_query[:100]}...' ")
    logger.debug(f"Overall Evaluator: Plan Summary for Eval: {executed_plan_summary_str[:300]}...")
    logger.debug(f"Overall Evaluator: Final Answer for Eval: {str(final_agent_answer_for_eval)[:300]}...")

    overall_eval_result: Optional[EvaluationResult] = await evaluate_plan_outcome(
        original_user_query=user_query,
        executed_plan_summary=executed_plan_summary_str,
        final_agent_answer=str(final_agent_answer_for_eval), 
        session_data_entry=state 
    )

    if overall_eval_result:
        logger.info(f"Evaluator (Overall): Success: {overall_eval_result.overall_success}, Assessment: '{overall_eval_result.assessment}'")
        return {
            "overall_evaluation_success": overall_eval_result.overall_success,
            "overall_evaluation_assessment": overall_eval_result.assessment,
            "overall_evaluation_final_answer_content": overall_eval_result.assessment, 
            "overall_evaluation_suggestions_for_replan": overall_eval_result.suggestions_for_replan,
            "messages": [AIMessage(content=f"Overall Plan Evaluation:\n  Success: {overall_eval_result.overall_success}\n  Assessment: {overall_eval_result.assessment}")]
        }
    else:
        logger.error("Overall Evaluator Node: Failed to get evaluation result from LLM.")
        final_assessment = "Overall evaluation failed to produce a result. Plan outcome uncertain."
        return {
            "overall_evaluation_success": False, 
            "overall_evaluation_assessment": final_assessment,
            "overall_evaluation_final_answer_content": final_assessment,
            "overall_evaluation_error": "Overall Evaluator LLM failed.",
            "messages": [AIMessage(content=f"Overall Plan Evaluation:\n  Error: Evaluation LLM failed.")]
        }

# --- Conditional Edges ---

def should_plan(state: ResearchAgentState) -> str:
    logger.info("--- Routing Decision: Should Plan? ---")
    actual_intent = state.get("classified_intent")
    
    if actual_intent == "PLAN":
        logger.info("Routing: Intent is PLAN, proceeding to planner.")
        return "planner"
    else: 
        logger.info(f"Routing: Intent is '{actual_intent}' (NOT PLAN), proceeding to overall_evaluator.")
        return "overall_evaluator" 

def should_continue_after_step_eval(state: ResearchAgentState) -> str:
    logger.info("--- Routing Decision after Step Evaluation ---")
    step_achieved_goal = state.get("step_evaluation_achieved_goal")
    is_recoverable = state.get("step_evaluation_is_recoverable")
    
    idx_for_next_action = state["current_step_index"] 
    plan_steps = state.get("plan_steps", [])
    retry_for_next_attempt = state.get("retry_count_for_current_step", 0)
    
    logger.info(f"Router after Step Eval: achieved_goal={step_achieved_goal}, is_recoverable={is_recoverable}, idx_for_next_action={idx_for_next_action}, plan_len={len(plan_steps)}, retry_for_next_attempt={retry_for_next_attempt}")

    if step_achieved_goal:
        if idx_for_next_action < len(plan_steps):
            logger.info(f"Routing: Step successful. Proceeding to Controller for next step: {idx_for_next_action + 1} (index {idx_for_next_action})")
            return "controller" 
        else:
            logger.info(f"Routing: All plan steps completed successfully. Index {idx_for_next_action} >= Plan length {len(plan_steps)}. Proceeding to Overall Evaluator.")
            return "overall_evaluator"
    else: 
        assessment = state.get('step_evaluation_assessment', 'Unknown failure reason.')
        logger.warning(f"Routing: Step (index {idx_for_next_action}) failed. Assessment: {assessment}") 
        
        if is_recoverable and retry_for_next_attempt <= MAX_STEP_RETRIES:
            logger.info(f"Routing: Step failed, but is recoverable. Proceeding to Controller for retry attempt {retry_for_next_attempt} of step {idx_for_next_action + 1}.")
            return "controller" 
        else:
            if not is_recoverable:
                logger.warning(f"Routing: Step failed and is not recoverable. Proceeding to Overall Evaluator.")
            else: 
                logger.warning(f"Routing: Step failed and retries exhausted ({retry_for_next_attempt-1} >= {MAX_STEP_RETRIES}). Proceeding to Overall Evaluator.") 
            return "overall_evaluator"

# --- Build the Graph ---
def create_research_agent_graph() -> StateGraph:
    workflow = StateGraph(ResearchAgentState)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("controller", controller_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("step_evaluator", step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    workflow.set_entry_point("intent_classifier")
    workflow.add_conditional_edges(
        "intent_classifier",
        should_plan,
        {"planner": "planner", "overall_evaluator": "overall_evaluator"}
    )
    workflow.add_edge("planner", "controller")
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")
    workflow.add_conditional_edges(
        "step_evaluator",
        should_continue_after_step_eval,
        {"controller": "controller", "overall_evaluator": "overall_evaluator"}
    )
    workflow.add_edge("overall_evaluator", END)
    research_agent_graph = workflow.compile()
    logger.info("ResearchAgent LangGraph compiled successfully with Overall Evaluator and full loop logic.")
    return research_agent_graph

# --- Main Test Execution (Example) ---
async def run_graph_example(query: str, task_id_override: Optional[str] = None):
    research_agent_graph = create_research_agent_graph()
    current_task_id = task_id_override if task_id_override else f"test_task_for_overall_eval"
    get_task_workspace_path(current_task_id, create_if_not_exists=True)

    initial_state: ResearchAgentState = {
        "user_query": query, "messages": [HumanMessage(content=query)], "task_id": current_task_id,
        "classified_intent": None, "intent_classifier_reasoning": None, "plan_summary": None,
        "plan_steps": None, "plan_generation_error": None, "current_step_index": 0,
        "previous_step_executor_output": None, "retry_count_for_current_step": 0,
        "controller_tool_name": None, "controller_tool_input": None, "controller_reasoning": None,
        "controller_confidence": None, "controller_error": None, "current_executor_output": None,
        "executor_error_message": None, "step_evaluation_achieved_goal": None,
        "step_evaluation_assessment": None, "step_evaluation_is_recoverable": None,
        "step_evaluation_suggested_tool": None, "step_evaluation_suggested_input_instructions": None,
        "step_evaluation_confidence_in_correction": None, "step_evaluation_error": None,
        "overall_evaluation_success": None, "overall_evaluation_assessment": None,
        "overall_evaluation_final_answer_content": None, "overall_evaluation_suggestions_for_replan": None,
        "overall_evaluation_error": None, "error_message": None,
        "session_intent_classifier_llm_id": None, "session_planner_llm_id": None,
        "session_controller_llm_id": None, "session_executor_llm_id": None,
        "session_evaluator_llm_id": None,
    }
    config_for_run = RunnableConfig(configurable={"thread_id": f"test_run_{uuid4()}"})

    logger.info(f"Streaming LangGraph execution for query: '{query}'")
    final_state = None
    async for chunk in research_agent_graph.astream(initial_state, config=config_for_run):
        logger.info("--- Graph Stream Chunk ---")
        for key, value in chunk.items():
            logger.info(f"State after Node '{key}':")
            node_state = value 
            for k, v in node_state.items():
                if isinstance(v, list) and len(v) > 3 and k != "messages": 
                    logger.info(f"    {k.replace('_', ' ').title()}: List with {len(v)} items (first 3 shown)...")
                    for i, item in enumerate(v[:3]): logger.info(f"        Item {i}: {str(item)[:100]}...")
                elif isinstance(v, str) and len(v) > 150: logger.info(f"    {k.replace('_', ' ').title()}: {v[:150]}...")
                elif v is not None : logger.info(f"    {k.replace('_', ' ').title()}: {v}")
            final_state = value 
    logger.info("--- End of Graph Stream ---")
    if final_state:
        logger.info("--- Final Accumulated Graph State ---")
        for key, value in final_state.items():
            if isinstance(value, list) and len(value) > 3 and key != "messages":
                 logger.info(f"{key.replace('_', ' ').title()}: List with {len(value)} items (first 3 shown)...")
                 for i, item_in_list in enumerate(value[:3]): logger.info(f"    Item {i}: {str(item_in_list)[:100]}...")
            elif isinstance(value, str) and len(value) > 200 and key != "user_query": 
                 logger.info(f"{key.replace('_', ' ').title()}: {value[:200]}...")
            else: logger.info(f"{key.replace('_', ' ').title()}: {value}")
    return final_state

async def run_main_test():
    test_query_python_repl_retry = "Use the Python REPL tool to obtain the literal string 'Python is fun!' (including the single quotes). Then, write this exact string to a file named 'python_quote.txt'."
    print(f"\n--- Running LangGraph Test (PCEE Retry Test) for: '{test_query_python_repl_retry}' ---")
    final_state_repl = await run_graph_example(test_query_python_repl_retry, task_id_override="test_task_repl_retry_002")
    if final_state_repl:
        print(f"\n--- Test Run Complete (PCEE Retry Test) ---")
        print(f"Overall Evaluation Success: {final_state_repl.get('overall_evaluation_success')}")
        print(f"Overall Evaluation Assessment: {final_state_repl.get('overall_evaluation_assessment')}")
        print(f"Final number of messages: {len(final_state_repl.get('messages', []))}")
    else:
        print("\n--- Test Run Failed (Python REPL Retry Query) ---")

if __name__ == "__main__":
    asyncio.run(run_main_test())
