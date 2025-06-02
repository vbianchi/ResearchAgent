# backend/langgraph_agent.py
import logging
import asyncio 

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END 
from typing import TypedDict, Optional, List, Dict as TypingDict, Union

from backend.config import settings 
from backend.tools import get_dynamic_tools 
from backend.llm_setup import get_llm 
from langchain_core.language_models.chat_models import BaseChatModel 

from .graph_state import ResearchAgentState 
from .intent_classifier import classify_intent, IntentClassificationOutput 
from .planner import generate_plan 
from .controller import validate_and_prepare_step_action, ControllerOutput 
from .evaluator import evaluate_step_outcome_and_suggest_correction, evaluate_plan_outcome, StepCorrectionOutcome, EvaluationResult
from backend.callbacks import WebSocketCallbackHandler, LOG_SOURCE_EXECUTOR, LOG_SOURCE_EVALUATOR_STEP, LOG_SOURCE_EVALUATOR_OVERALL, LOG_SOURCE_CONTROLLER

logger = logging.getLogger(__name__)

# --- Node Definitions ---
# intent_classifier_node, planner_node (as in Canvas 17)
# executor_node, overall_evaluator_node (as in Canvas 17)
# For brevity, only modified nodes (controller_node, step_evaluator_node) and routing are shown fully.

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Intent Classifier Node ---")
    user_query = state.get("user_query")
    existing_messages = state.get("messages", []) 
    if not isinstance(existing_messages, list):
        logger.warning("intent_classifier_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    if not user_query:
        logger.error("Intent Classifier Node: User query is missing from state.")
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content="System Error: User query was not found for intent classification.")]
        return {**state, "error_message": "User query is missing for intent classification.", "messages": new_messages}
    callback_handler_from_config = config.get("callbacks")
    try:
        classification_output: IntentClassificationOutput = await classify_intent(
            user_query=user_query, available_tools_summary=None, callback_handler=callback_handler_from_config)
        logger.info(f"Intent classification result: {classification_output.intent}, Reasoning: {classification_output.reasoning}")
        ai_message_content = f"Intent Classified as: {classification_output.intent}\nReasoning: {classification_output.reasoning or 'N/A'}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=ai_message_content)]
        return {**state, "classified_intent": classification_output.intent, "intent_classifier_reasoning": classification_output.reasoning, "messages": new_messages, "error_message": None }
    except Exception as e:
        logger.error(f"Intent Classifier Node: Error during intent classification: {e}", exc_info=True)
        error_msg_content = f"System Error during intent classification: {str(e)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "error_message": f"Error in intent classification: {str(e)}", "classified_intent": "PLAN", "intent_classifier_reasoning": f"Error during intent classification: {str(e)}", "messages": new_messages}

async def planner_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Planner Node ---")
    user_query = state.get("user_query")
    current_task_id_for_tools = state.get("task_id")
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("planner_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    logger.info(f"Planner Node: current_task_id_for_tools from state: {current_task_id_for_tools}")
    if not user_query:
        logger.error("Planner Node: User query is missing from state.")
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content="System Error: User query was not found for planning.")]
        # Initialize loop counters even on planning failure to ensure state schema consistency for router
        return {**state, "plan_generation_error": "User query is missing for planning.", "messages": new_messages, "current_step_index": 0, "retry_count_for_current_step": 0}
    callback_handler_from_config = config.get("callbacks")
    try:
        tools = get_dynamic_tools(current_task_id_for_tools) 
        available_tools_summary = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        if not available_tools_summary: available_tools_summary = "No specific tools loaded."
    except Exception as e:
        logger.error(f"Planner Node: Failed to load tools for summary: {e}", exc_info=True)
        available_tools_summary = "Error loading tool information."
    try:
        plan_result_dict = await generate_plan(
            user_query=user_query, available_tools_summary=available_tools_summary, callback_handler=callback_handler_from_config)
        if plan_result_dict.get("plan_generation_error"):
            error_content = f"Planning Error: {plan_result_dict['plan_generation_error']}"
            logger.error(f"Planner Node: Plan generation failed: {error_content}")
            safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
            new_messages = safe_existing_messages + [AIMessage(content=error_content)]
            return {**state, "plan_generation_error": plan_result_dict["plan_generation_error"], "messages": new_messages, "current_step_index": 0, "retry_count_for_current_step": 0}
        plan_summary = plan_result_dict.get("plan_summary")
        plan_steps = plan_result_dict.get("plan_steps", [])
        logger.info(f"Planner Node: Plan generated. Summary: {plan_summary}")
        ai_message_content = f"Plan Generated:\nSummary: {plan_summary}\nNumber of steps: {len(plan_steps)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=ai_message_content)]
        return {**state, "plan_summary": plan_summary, "plan_steps": plan_steps, 
                "plan_generation_error": None, "messages": new_messages, 
                "error_message": None, "current_step_index": 0, "retry_count_for_current_step": 0 } 
    except Exception as e:
        logger.error(f"Planner Node: Unexpected error during planning: {e}", exc_info=True)
        error_msg_content = f"System Error during planning: {str(e)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "plan_generation_error": f"Unexpected error in planner: {str(e)}", "messages": new_messages, "current_step_index": 0, "retry_count_for_current_step": 0}

async def controller_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Controller Node ---")
    original_user_query = state.get("user_query")
    plan_steps = state.get("plan_steps")
    current_step_idx = state.get("current_step_index") 
    retry_count = state.get("retry_count_for_current_step", 0) # Get current retry count

    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("controller_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []

    if current_step_idx is None: 
        logger.error("Controller Node: current_step_index is None.")
        new_messages = existing_messages + [AIMessage(content="System Error: Step index not initialized for Controller.")]
        return {**state, "controller_error": "Step index not initialized, plan likely ended.", "messages": new_messages}
        
    if not original_user_query:
        logger.error("Controller Node: Original user query missing from state.")
        new_messages = existing_messages + [AIMessage(content="System Error: Original query missing for Controller.")]
        return {**state, "controller_error": "Original user query missing.", "messages": new_messages}

    if not plan_steps or not isinstance(plan_steps, list) or current_step_idx >= len(plan_steps):
        logger.warning(f"Controller Node: current_step_idx ({current_step_idx}) out of bounds for plan_steps (len: {len(plan_steps) if plan_steps else 0}). Plan likely complete or error.")
        new_messages = existing_messages + [AIMessage(content="System Info: Controller determined no more valid steps to process.")]
        return {**state, "controller_error": "No valid next step (plan likely complete).", "messages": new_messages} 
    
    current_plan_step_dict = plan_steps[current_step_idx]
    logger.info(f"Controller Node: Processing Step {current_step_idx + 1}/{len(plan_steps)}: '{current_plan_step_dict.get('description')}' (Attempt: {retry_count + 1})")

    previous_executor_output = state.get("previous_step_executor_output") 
    current_task_id_for_tools = state.get("task_id")
    try: tools = get_dynamic_tools(current_task_id_for_tools)
    except Exception as e:
        logger.error(f"Controller Node: Failed to load tools: {e}", exc_info=True)
        new_messages = existing_messages + [AIMessage(content=f"System Error: Failed to load tools for Controller: {e}")]
        return {**state, "controller_error": f"Failed to load tools: {e}", "messages": new_messages}

    callback_handler_from_config = config.get("callbacks")
    
    step_to_validate = current_plan_step_dict.copy() # Start with the original step definition
    if retry_count > 0:
        logger.info(f"Controller Node: This is a retry (attempt {retry_count + 1}). Using evaluator suggestions if available.")
        evaluator_suggested_tool = state.get("step_evaluation_suggested_tool")
        evaluator_suggested_instructions = state.get("step_evaluation_suggested_input_instructions")
        
        if evaluator_suggested_tool is not None: # Can be "None" (string) or a tool name
            step_to_validate["tool_to_use"] = evaluator_suggested_tool 
            logger.info(f"  Retry: Overriding tool to: '{evaluator_suggested_tool}'")
        elif "tool_to_use" not in step_to_validate : # Ensure it exists from original plan
             step_to_validate["tool_to_use"] = current_plan_step_dict.get("tool_to_use")

        if evaluator_suggested_instructions is not None:
            step_to_validate["tool_input_instructions"] = evaluator_suggested_instructions
            logger.info(f"  Retry: Overriding input instructions to: '{evaluator_suggested_instructions[:100]}...'")
        elif "tool_input_instructions" not in step_to_validate: # Ensure it exists
             step_to_validate["tool_input_instructions"] = current_plan_step_dict.get("tool_input_instructions")
    
    try:
        controller_result: Dict[str, Any] = await validate_and_prepare_step_action(
            original_user_query=original_user_query, 
            current_plan_step=step_to_validate, # Use the (potentially modified for retry) step
            available_tools=tools, 
            previous_step_executor_output=previous_executor_output,
            controller_llm_id_override=None, # TODO: Get from state if session override exists
            callback_handler=callback_handler_from_config
        )

        if controller_result.get("controller_error"):
            error_msg = f"Controller Error for Step {current_step_idx + 1} (Attempt {retry_count + 1}): {controller_result['controller_error']}"
            logger.error(error_msg)
            new_messages = existing_messages + [AIMessage(content=error_msg)]
            # Preserve retry_count for the router to decide if it was a controller error during a retry
            return {**state, **controller_result, "messages": new_messages, "retry_count_for_current_step": retry_count}

        ai_message_content = (f"Controller for Step {current_step_idx + 1} (Attempt {retry_count + 1}):\n"
            f"  Tool: {controller_result.get('controller_tool_name', 'None')}\n"
            f"  Input (summary): {str(controller_result.get('controller_tool_input', 'N/A'))[:100]}...\n"
            f"  Reasoning: {controller_result.get('controller_reasoning', 'N/A')}")
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        
        updated_state_fields = {
            **controller_result, 
            "current_step_index": current_step_idx, # Keep current step index
            "retry_count_for_current_step": retry_count, # Pass along current retry count
            "messages": new_messages, 
            "current_executor_output": None, 
            "executor_error_message": None,
            "step_evaluation_achieved_goal": None, "step_evaluation_assessment": None,
            "step_evaluation_is_recoverable": None, "step_evaluation_suggested_tool": None,
            "step_evaluation_suggested_input_instructions": None, 
            "step_evaluation_confidence_in_correction": None, "step_evaluation_error": None,
            "error_message": None 
        }
        return {**state, **updated_state_fields}
    except Exception as e:
        logger.error(f"Controller Node: Unexpected error for step {current_step_idx + 1} (Attempt {retry_count + 1}): {e}", exc_info=True)
        error_msg_content = f"System Error in Controller Node for step {current_step_idx + 1} (Attempt {retry_count + 1}): {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "controller_error": f"Unexpected error in Controller: {str(e)}", "messages": new_messages, "retry_count_for_current_step": retry_count}

async def executor_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Executor Node ---") # ... (executor_node remains the same as Canvas 16)
    tool_name = state.get("controller_tool_name")
    tool_input_str = state.get("controller_tool_input")
    task_id = state.get("task_id")
    current_step_idx = state.get("current_step_index", 0) 
    retry_count = state.get("retry_count_for_current_step", 0)
    plan_steps = state.get("plan_steps", [])
    step_description = "N/A"
    if plan_steps and current_step_idx < len(plan_steps):
        step_description = plan_steps[current_step_idx].get("description", "N/A")
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("executor_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    logger.info(f"Executor Node: Task ID '{task_id}', Step {current_step_idx + 1} ('{step_description}'), Attempt {retry_count + 1}. Tool: '{tool_name}', Input: '{str(tool_input_str)[:100]}...'")
    output_str_from_execution = "" 
    error_msg_from_execution = None 
    callback_handler_from_config = config.get("callbacks")
    if tool_name and tool_name.lower() != "none":
        try:
            available_tools = get_dynamic_tools(task_id)
            selected_tool = next((t for t in available_tools if t.name == tool_name), None)
            if selected_tool:
                logger.info(f"Executor Node: Executing tool '{selected_tool.name}'")
                output_str_from_execution = await selected_tool.arun(tool_input=tool_input_str, callbacks=callback_handler_from_config) 
                logger.info(f"Executor Node: Tool '{selected_tool.name}' executed. Output length: {len(str(output_str_from_execution))}")
            else:
                error_msg_from_execution = f"Tool '{tool_name}' not found in available tools."
                logger.error(f"Executor Node: {error_msg_from_execution}")
                output_str_from_execution = f"Error: Tool '{tool_name}' not found." 
        except Exception as e:
            logger.error(f"Executor Node: Error executing tool '{tool_name}': {e}", exc_info=True)
            error_msg_from_execution = f"Error executing tool '{tool_name}': {str(e)}"
            output_str_from_execution = f"Tool Execution Error: {str(e)}" 
    elif tool_input_str: 
        logger.info(f"Executor Node: 'None' tool specified. Executing direct LLM call with input: {tool_input_str[:100]}...")
        try:
            executor_llm: BaseChatModel = get_llm(
                settings, provider=settings.executor_default_provider, 
                model_name=settings.executor_default_model_name, 
                requested_for_role=LOG_SOURCE_EXECUTOR + "_DirectLLM",
                callbacks=callback_handler_from_config)
            llm_response = await executor_llm.ainvoke(
                [HumanMessage(content=tool_input_str)], 
                config=RunnableConfig(callbacks=callback_handler_from_config, metadata={"component_name": LOG_SOURCE_EXECUTOR + "_DirectLLM"}))
            output_str_from_execution = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            logger.info(f"Executor Node: Direct LLM call completed. Output length: {len(output_str_from_execution)}")
        except Exception as e:
            logger.error(f"Executor Node: Error during direct LLM call: {e}", exc_info=True)
            error_msg_from_execution = f"Error during direct LLM call: {str(e)}"
            output_str_from_execution = f"LLM Execution Error: {str(e)}"
    else:
        logger.warning("Executor Node: No tool name and no tool input provided for step.")
        output_str_from_execution = "Executor: No specific action taken as no tool or direct LLM input was provided by the controller for this step."
    ai_message_content = f"Executor (Step {current_step_idx + 1}, Attempt {retry_count + 1}):\nTool: {tool_name or 'None'}\nOutput: {str(output_str_from_execution)[:500]}{'...' if len(str(output_str_from_execution)) > 500 else ''}"
    if error_msg_from_execution: ai_message_content += f"\nError: {error_msg_from_execution}"
    new_messages = existing_messages + [AIMessage(content=ai_message_content)]
    return {**state, "current_executor_output": str(output_str_from_execution), "executor_error_message": error_msg_from_execution, "messages": new_messages, "error_message": None}


async def step_evaluator_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Step Evaluator Node ---")
    original_user_query = state.get("user_query")
    plan_steps = state.get("plan_steps")
    current_step_idx = state.get("current_step_index", 0) 
    retry_count_for_step_just_attempted = state.get("retry_count_for_current_step", 0)
    controller_tool_used = state.get("controller_tool_name")
    controller_tool_input_str = state.get("controller_tool_input")
    executor_output_str = state.get("current_executor_output", "") 
    task_id = state.get("task_id")
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("step_evaluator_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []

    if not original_user_query or not plan_steps or not isinstance(plan_steps, list) or current_step_idx >= len(plan_steps):
        error_msg = "Step Evaluator: Missing critical info (query, plan, or valid step index)."
        logger.error(error_msg + f" Index: {current_step_idx}, Plan len: {len(plan_steps) if plan_steps else 'N/A'}")
        new_messages = existing_messages + [AIMessage(content=f"System Error: {error_msg}")]
        # Ensure essential fields for routing are set even on this early error
        return {**state, "step_evaluation_error": error_msg, "step_evaluation_achieved_goal": False, 
                "step_evaluation_is_recoverable": False, "messages": new_messages}

    current_plan_step_dict = plan_steps[current_step_idx]
    logger.info(f"Step Evaluator Node: Evaluating Step {current_step_idx + 1} (Attempt {retry_count_for_step_just_attempted + 1}): {current_plan_step_dict.get('description')}")
    try: tools = get_dynamic_tools(task_id)
    except Exception as e:
        logger.error(f"Step Evaluator Node: Failed to load tools: {e}", exc_info=True)
        new_messages = existing_messages + [AIMessage(content=f"System Error: Failed to load tools for Step Evaluator: {e}")]
        return {**state, "step_evaluation_error": f"Failed to load tools: {e}", "step_evaluation_achieved_goal": False, "messages": new_messages}
    
    callback_handler_from_config = config.get("callbacks")
    updated_state_for_return = {**state} 

    try:
        evaluation_result_dict: Dict[str, Any] = await evaluate_step_outcome_and_suggest_correction(
            original_user_query=original_user_query, current_plan_step=current_plan_step_dict,
            controller_tool_used=controller_tool_used, controller_tool_input=controller_tool_input_str,
            step_executor_output=executor_output_str, available_tools=tools,
            evaluator_llm_id_override=None, callback_handler=callback_handler_from_config)
        
        for key, value in evaluation_result_dict.items(): updated_state_for_return[key] = value
        if "step_evaluation_error" not in evaluation_result_dict: updated_state_for_return["step_evaluation_error"] = None
        
        assessment = updated_state_for_return.get('step_evaluation_assessment', 'N/A')
        achieved = updated_state_for_return.get('step_evaluation_achieved_goal', False)
        ai_message_content = (f"Step Evaluator (Step {current_step_idx + 1}, Attempt {retry_count_for_step_just_attempted + 1}):\n"
            f"  Achieved Goal: {achieved}\n"
            f"  Assessment: {assessment[:300]}{'...' if len(assessment) > 300 else ''}")
        if not achieved and updated_state_for_return.get('step_evaluation_is_recoverable'):
            ai_message_content += f"\n  Retry Suggested: Tool='{updated_state_for_return.get('step_evaluation_suggested_tool', 'N/A')}', Instructions='{str(updated_state_for_return.get('step_evaluation_suggested_input_instructions', 'N/A'))[:100]}...'"
        
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        updated_state_for_return["messages"] = new_messages
        updated_state_for_return["error_message"] = None 

        if achieved:
            next_step_idx_to_prepare_for = current_step_idx + 1
            plan_steps_list = state.get("plan_steps", []) 
            if next_step_idx_to_prepare_for < len(plan_steps_list):
                logger.info(f"Step Evaluator: Step {current_step_idx + 1} successful. Preparing for next step {next_step_idx_to_prepare_for + 1}.")
                updated_state_for_return["current_step_index"] = next_step_idx_to_prepare_for
                updated_state_for_return["retry_count_for_current_step"] = 0 
                updated_state_for_return["previous_step_executor_output"] = state.get("current_executor_output") 
                # Clear fields for the next fresh step
                for key_to_clear in ["controller_tool_name", "controller_tool_input", "controller_reasoning", 
                                     "controller_confidence", "controller_error", "current_executor_output", 
                                     "executor_error_message", 
                                     # Keep current step_evaluation fields as they are from this evaluation
                                     # They will be overwritten by the next step's evaluation or cleared by controller if needed
                                     # "step_evaluation_achieved_goal", "step_evaluation_assessment", etc.
                                     "step_evaluation_suggested_tool", # Clear retry suggestions
                                     "step_evaluation_suggested_input_instructions",
                                     "step_evaluation_confidence_in_correction"
                                     ]: 
                    updated_state_for_return[key_to_clear] = None 
            else: 
                logger.info(f"Step Evaluator: Step {current_step_idx + 1} successful. All plan steps completed.")
                updated_state_for_return["current_step_index"] = len(plan_steps_list) 
        else: # Step failed
            logger.warning(f"Step Evaluator: Step {current_step_idx + 1} (Attempt {retry_count_for_step_just_attempted + 1}) failed.")
            updated_state_for_return["current_step_index"] = current_step_idx # Stay on the same step
            if updated_state_for_return.get("step_evaluation_is_recoverable") and retry_count_for_step_just_attempted < settings.agent_max_step_retries:
                 updated_state_for_return["retry_count_for_current_step"] = retry_count_for_step_just_attempted + 1
                 logger.info(f"  Step Evaluator: Preparing for retry attempt {updated_state_for_return['retry_count_for_current_step']} for step {current_step_idx + 1}.")
                 # Clear executor output for retry. Controller fields will be re-evaluated.
                 # Retry suggestions (step_evaluation_suggested_tool etc.) are already in updated_state_for_return from evaluation_result_dict.
                 updated_state_for_return["current_executor_output"] = None
                 updated_state_for_return["executor_error_message"] = None
                 # Clear previous controller decision for the retry
                 updated_state_for_return["controller_tool_name"] = None
                 updated_state_for_return["controller_tool_input"] = None
                 updated_state_for_return["controller_reasoning"] = None
                 updated_state_for_return["controller_confidence"] = None
                 updated_state_for_return["controller_error"] = None
            else: 
                 # Not recoverable or max retries hit, keep current retry_count for router to see
                 updated_state_for_return["retry_count_for_current_step"] = retry_count_for_step_just_attempted
        return updated_state_for_return
    except Exception as e:
        logger.error(f"Step Evaluator Node: Unexpected error for step {current_step_idx + 1}: {e}", exc_info=True)
        error_msg_content = f"System Error in Step Evaluator Node for step {current_step_idx + 1}: {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "step_evaluation_error": f"Unexpected error in Step Evaluator: {str(e)}", "step_evaluation_achieved_goal": False, "messages": new_messages}

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Overall Evaluator Node ---") 
    original_user_query = state.get("user_query", "N/A")
    messages_history = state.get("messages", [])
    # Use previous_step_executor_output if current_executor_output is None (e.g. after a loop)
    # This assumes previous_step_executor_output holds the last *meaningful* step output if plan completed.
    final_agent_answer_candidate = state.get("current_executor_output") or state.get("previous_step_executor_output")
    
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("overall_evaluator_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    plan_execution_summary_parts = []
    for msg in messages_history:
        if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
            if "Plan Generated:" in msg.content or "Controller for Step" in msg.content or \
               "Executor (Step" in msg.content or "Step Evaluator (Step" in msg.content or \
               "Planning Error:" in msg.content or "Controller Error" in msg.content or \
               "Tool Execution Error:" in msg.content or "LLM Execution Error:" in msg.content or \
               "Step Evaluation Error" in msg.content: # Added more error types
                plan_execution_summary_parts.append(msg.content)
    executed_plan_summary_str = "\n---\n".join(plan_execution_summary_parts)
    
    # Add overall plan status to summary string
    if state.get("plan_generation_error"): executed_plan_summary_str += f"\n\nFINAL STATUS: PLANNING FAILED: {state.get('plan_generation_error')}"
    elif state.get("controller_error"): executed_plan_summary_str += f"\n\nFINAL STATUS: CONTROLLER FAILED on step {state.get('current_step_index', -1)+1}: {state.get('controller_error')}"
    elif state.get("executor_error_message"): executed_plan_summary_str += f"\n\nFINAL STATUS: EXECUTOR FAILED on step {state.get('current_step_index', -1)+1}: {state.get('executor_error_message')}"
    elif state.get("step_evaluation_error"): executed_plan_summary_str += f"\n\nFINAL STATUS: STEP EVALUATION FAILED for step {state.get('current_step_index', -1)+1}: {state.get('step_evaluation_error')}"
    elif state.get("current_step_index", 0) >= len(state.get("plan_steps",[])) and state.get("step_evaluation_achieved_goal") :
        executed_plan_summary_str += "\n\nFINAL STATUS: ALL PLANNED STEPS COMPLETED SUCCESSFULLY."
    elif not state.get("step_evaluation_achieved_goal", True) and state.get("plan_steps"): # Plan attempted but a step failed unrecoverably
        executed_plan_summary_str += f"\n\nFINAL STATUS: PLAN EXECUTION FAILED at step {state.get('current_step_index', -1)+1}. Last assessment: {state.get('step_evaluation_assessment')}"
    elif state.get("classified_intent") != "PLAN":
        executed_plan_summary_str = f"INTENT: {state.get('classified_intent')}.\n" + (messages_history[-1].content if messages_history and isinstance(messages_history[-1], AIMessage) else "No direct answer found in messages.")
        if not final_agent_answer_candidate and messages_history and isinstance(messages_history[-1], AIMessage): # For DIRECT_QA, last AI message might be the answer
            final_agent_answer_candidate = messages_history[-1].content


    if not executed_plan_summary_str: executed_plan_summary_str = "No detailed execution summary could be constructed. Plan may have failed very early or was not a planning intent."

    logger.info(f"Overall Evaluator: Evaluating plan for query: '{original_user_query[:100]}...'")
    callback_handler_from_config = config.get("callbacks")
    return_state_update = {**state}
    try:
        overall_eval_result_dict: Dict[str, Any] = await evaluate_plan_outcome(
            original_user_query=original_user_query, executed_plan_summary_str=executed_plan_summary_str,
            final_agent_answer_from_last_step=final_agent_answer_candidate, 
            evaluator_llm_id_override=None, callback_handler=callback_handler_from_config)
        for key, value in overall_eval_result_dict.items(): return_state_update[key] = value
        if "overall_evaluation_error" not in overall_eval_result_dict: return_state_update["overall_evaluation_error"] = None
        assessment = overall_eval_result_dict.get('overall_evaluation_assessment', 'N/A')
        success = overall_eval_result_dict.get('overall_evaluation_success', False)
        final_content = overall_eval_result_dict.get('overall_evaluation_final_answer_content')
        ai_message_content = f"Overall Plan Evaluation:\n  Success: {success}\n  Assessment: {assessment}"
        if success and final_content: ai_message_content += f"\n  Final Answer:\n{final_content[:300]}{'...' if len(final_content) > 300 else ''}"
        elif not success and overall_eval_result_dict.get('overall_evaluation_suggestions_for_replan'):
            suggestions = "; ".join(overall_eval_result_dict['overall_evaluation_suggestions_for_replan'])
            ai_message_content += f"\n  Suggestions for Replan: {suggestions}"
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        return_state_update["messages"] = new_messages
        return_state_update["error_message"] = None 
        return return_state_update
    except Exception as e:
        logger.error(f"Overall Evaluator Node: Unexpected error: {e}", exc_info=True)
        error_msg_content = f"System Error in Overall Evaluator Node: {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "overall_evaluation_error": f"Unexpected error in Overall Evaluator: {str(e)}", "overall_evaluation_success": False, "messages": new_messages}

# --- Graph Definition ---
# (Graph Definition and Edges remain the same as Canvas 16)
workflow_builder = StateGraph(ResearchAgentState)
workflow_builder.add_node("intent_classifier", intent_classifier_node)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("controller", controller_node)
workflow_builder.add_node("executor", executor_node) 
workflow_builder.add_node("step_evaluator", step_evaluator_node)
workflow_builder.add_node("overall_evaluator", overall_evaluator_node)

workflow_builder.set_entry_point("intent_classifier")

def route_after_intent_classification(state: ResearchAgentState):
    intent = state.get("classified_intent")
    if state.get("error_message"): 
        logger.error(f"Routing from Intent: Error detected ({state['error_message']}). Proceeding to Overall Evaluator.")
        return "overall_evaluator"
    if intent == "PLAN":
        logger.info("Routing from Intent: Intent is PLAN, proceeding to planner.")
        return "planner"
    else: 
        logger.info(f"Routing from Intent: Intent is '{intent}', proceeding to Overall Evaluator.")
        return "overall_evaluator"

def route_after_planner(state: ResearchAgentState):
    if state.get("plan_generation_error") or not state.get("plan_steps"):
        logger.info("Routing from Planner: Plan error or no steps, proceeding to Overall Evaluator.")
        return "overall_evaluator" 
    logger.info("Routing from Planner: Plan generated, proceeding to controller for first step.")
    return "controller"

def route_after_controller(state: ResearchAgentState):
    if state.get("controller_error"): 
        logger.error(f"Routing from Controller: Controller error: {state['controller_error']}. Proceeding to Overall Evaluator.")
        return "overall_evaluator" 
    plan_steps = state.get("plan_steps", [])
    current_step_idx = state.get("current_step_index")
    # This condition means the controller itself determined there are no more valid steps
    if current_step_idx is not None and current_step_idx >= len(plan_steps) and "No valid next step" in str(state.get("controller_error")):
        logger.info("Routing from Controller: No more steps indicated by controller. Proceeding to Overall Evaluator.")
        return "overall_evaluator"
    logger.info("Routing from Controller: Controller finished, proceeding to executor.")
    return "executor"

def route_after_executor(state: ResearchAgentState):
    logger.info("Routing from Executor: Executor finished, proceeding to step_evaluator.")
    return "step_evaluator"

def route_after_step_evaluation(state: ResearchAgentState):
    logger.info("--- Routing Decision after Step Evaluation ---")
    achieved_goal = state.get("step_evaluation_achieved_goal")
    is_recoverable = state.get("step_evaluation_is_recoverable")
    # current_step_index from state is the index that was JUST evaluated (if failed and preparing for retry)
    # OR it's the index of the NEXT step (if current step succeeded).
    # This is because step_evaluator_node updates current_step_index based on outcome.
    idx_for_next_action = state.get("current_step_index", 0) 
    plan_steps = state.get("plan_steps", [])
    # retry_count_for_current_step in state is the count for the *next* attempt of the *current failed step*.
    retry_count_for_next_attempt = state.get("retry_count_for_current_step", 0) 
    max_retries = settings.agent_max_step_retries 

    # This log needs to be careful about what current_step_idx means here
    logger.info(f"Router after Step Eval: achieved_goal={achieved_goal}, is_recoverable={is_recoverable}, idx_for_next_action={idx_for_next_action}, plan_len={len(plan_steps)}, retry_for_next_attempt={retry_count_for_next_attempt}")

    if state.get("step_evaluation_error"):
        logger.error(f"Routing: Step evaluation itself failed: {state['step_evaluation_error']}. Proceeding to Overall Evaluator.")
        return "overall_evaluator" 

    if achieved_goal:
        # If goal was achieved, step_evaluator_node would have set idx_for_next_action to current_step_idx + 1 or len(plan_steps)
        if idx_for_next_action < len(plan_steps):
            logger.info(f"Routing: Step successful. Proceeding to Controller for next step: {idx_for_next_action + 1} (index {idx_for_next_action})")
            return "controller"
        else: 
            logger.info(f"Routing: All plan steps completed successfully. Index {idx_for_next_action} >= Plan length {len(plan_steps)}. Proceeding to Overall Evaluator.")
            return "overall_evaluator" 
    else: # Step failed
        # idx_for_next_action is still the index of the FAILED step because step_evaluator_node kept it same for retry.
        logger.warning(f"Routing: Step (index {idx_for_next_action}) failed. Assessment: {state.get('step_evaluation_assessment')}")
        if is_recoverable and retry_count_for_next_attempt <= max_retries:
            logger.info(f"Routing: Step failed, but is recoverable. Proceeding to Controller for retry attempt {retry_count_for_next_attempt} of step {idx_for_next_action + 1}.")
            return "controller" 
        else:
            if not is_recoverable:
                logger.error(f"Routing: Step {idx_for_next_action + 1} failed and is not recoverable. Proceeding to Overall Evaluator.")
            else: 
                logger.error(f"Routing: Step {idx_for_next_action + 1} failed. Retries exhausted ({retry_count_for_next_attempt-1 if retry_count_for_next_attempt > 0 else 0} attempts made, max {max_retries}). Proceeding to Overall Evaluator.")
            return "overall_evaluator"

workflow_builder.add_conditional_edges("intent_classifier", route_after_intent_classification, {"planner": "planner", "overall_evaluator": "overall_evaluator"})
workflow_builder.add_conditional_edges("planner", route_after_planner, {"controller": "controller", "overall_evaluator": "overall_evaluator"})
workflow_builder.add_conditional_edges("controller", route_after_controller, {"executor": "executor", "overall_evaluator": "overall_evaluator"}) 
workflow_builder.add_conditional_edges("executor", route_after_executor, {"step_evaluator": "step_evaluator"})
workflow_builder.add_conditional_edges("step_evaluator", route_after_step_evaluation, {"controller": "controller", "overall_evaluator": "overall_evaluator"})
workflow_builder.add_edge("overall_evaluator", END) 

try:
    research_agent_graph = workflow_builder.compile()
    logger.info("ResearchAgent LangGraph compiled successfully with Overall Evaluator and full loop logic.")
except Exception as e:
    logger.critical(f"Failed to compile ResearchAgent LangGraph: {e}", exc_info=True)
    research_agent_graph = None

# --- Test Runner ---
# (Unchanged from Canvas 16)
async def run_graph_example(user_input: str, ws_callback_handler: Optional[WebSocketCallbackHandler] = None):
    if not research_agent_graph: logger.error("Graph not compiled."); return None
    initial_state: ResearchAgentState = {
        "user_query": user_input, "messages": [HumanMessage(content=user_input)], 
        "task_id": "test_task_for_overall_eval", "classified_intent": None, 
        "intent_classifier_reasoning": None, "plan_summary": None, "plan_steps": None,
        "plan_generation_error": None, "current_step_index": 0, 
        "previous_step_executor_output": None, "retry_count_for_current_step": 0, 
        "controller_tool_name": None, "controller_tool_input": None, 
        "controller_reasoning": None, "controller_confidence": None, "controller_error": None,
        "current_executor_output": None, "executor_error_message": None,
        "step_evaluation_achieved_goal": None, "step_evaluation_assessment": None,
        "step_evaluation_is_recoverable": None, "step_evaluation_suggested_tool": None,
        "step_evaluation_suggested_input_instructions": None, 
        "step_evaluation_confidence_in_correction": None, "step_evaluation_error": None,
        "overall_evaluation_success": None, "overall_evaluation_assessment": None,
        "overall_evaluation_final_answer_content": None, 
        "overall_evaluation_suggestions_for_replan": None, "overall_evaluation_error": None,
        "error_message": None 
    }
    callbacks_to_use = [ws_callback_handler] if ws_callback_handler else []
    config_for_run = RunnableConfig(callbacks=callbacks_to_use, recursion_limit=100) 
    logger.info(f"Streaming LangGraph execution for query: '{user_input}'")
    accumulated_state: Optional[ResearchAgentState] = None 
    async for chunk in research_agent_graph.astream(initial_state, config=config_for_run):
        logger.info(f"--- Graph Stream Chunk ---")
        for node_name, state_after_run in chunk.items():
            logger.info(f"State after Node '{node_name}':")
            accumulated_state = state_after_run
            keys_to_log = ["classified_intent", "plan_summary", "current_step_index", 
                           "controller_tool_name", "current_executor_output", 
                           "step_evaluation_achieved_goal", "step_evaluation_assessment",
                           "retry_count_for_current_step", "overall_evaluation_success", "overall_evaluation_assessment",
                           "executor_error_message", "error_message", 
                           "plan_generation_error", "controller_error", "step_evaluation_error", "overall_evaluation_error"]
            for key in keys_to_log:
                if key in accumulated_state and accumulated_state[key] is not None:
                    logger.info(f"    {key.replace('_', ' ').title()}: {str(accumulated_state[key])[:100]}...")
            if "messages" in accumulated_state and accumulated_state["messages"]:
                 logger.info(f"    Last Message: {str(accumulated_state['messages'][-1].content)[:70]}...")
    logger.info("--- End of Graph Stream ---")
    if accumulated_state:
        logger.info("--- Final Accumulated Graph State ---")
        for key, value in accumulated_state.items():
            if key == "messages": logger.info(f"Final Messages ({len(value)}): {[f'({type(m).__name__}) {m.content[:70]}...' for m in value]}")
            elif key == "plan_steps" and value: logger.info(f"Final Plan Steps ({len(value)}): {[s.get('description') for s in value]}")
            else: logger.info(f"{key.replace('_', ' ').title()}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
        return accumulated_state
    else: logger.warning("Graph stream did not yield final state."); return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    logger_langgraph = logging.getLogger("langgraph")
    logger_langgraph.setLevel(logging.INFO) 
    async def run_main_test():
        try:
            from backend.config import settings as app_settings 
            if not app_settings.google_api_key: print("WARNING: API keys might not be loaded.")
        except ImportError: print("ERROR: Could not import app_settings."); return
        test_query_plan_tool = "Research the benefits of solar power and write a short summary file called 'solar_benefits.txt'."
        print(f"\n--- Running LangGraph Test (Full PCEE Loop) for: '{test_query_plan_tool}' ---")
        final_state_tool = await run_graph_example(test_query_plan_tool) 
        if final_state_tool:
            print(f"\n--- Test Run Complete (Full PCEE Loop) ---")
            assert final_state_tool.get('classified_intent') == "PLAN"
            assert final_state_tool.get('plan_steps') is not None
            print(f"Overall Evaluation Success: {final_state_tool.get('overall_evaluation_success')}")
            print(f"Overall Evaluation Assessment: {final_state_tool.get('overall_evaluation_assessment')}")
            print(f"Final number of messages: {len(final_state_tool.get('messages', []))}")
        else: print("\n--- Test Run Failed (Full PCEE Loop) ---")
    asyncio.run(run_main_test())
