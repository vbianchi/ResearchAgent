# backend/langgraph_agent.py
import logging
import asyncio 

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
from .evaluator import evaluate_step_outcome_and_suggest_correction, StepCorrectionOutcome 
from backend.callbacks import WebSocketCallbackHandler, LOG_SOURCE_EXECUTOR, LOG_SOURCE_EVALUATOR_STEP

logger = logging.getLogger(__name__)

# --- Node Definitions ---
# intent_classifier_node, planner_node, controller_node, executor_node (as in Canvas 12)
# For brevity, these are assumed to be correct from the previous version.
# Ensure their return statements use {**state, ...} to pass through and update.

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
        # Initialize step_index and retry_count here for the start of the loop
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
    
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("controller_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []

    if current_step_idx is None: # Should be set by planner or loop logic
        logger.error("Controller Node: current_step_index is None. Cannot proceed.")
        new_messages = existing_messages + [AIMessage(content="System Error: Step index not initialized for Controller.")]
        return {**state, "controller_error": "Step index not initialized.", "messages": new_messages}
        
    if not original_user_query:
        logger.error("Controller Node: Original user query missing from state.")
        new_messages = existing_messages + [AIMessage(content="System Error: Original query missing for Controller.")]
        return {**state, "controller_error": "Original user query missing.", "messages": new_messages}

    if not plan_steps or not isinstance(plan_steps, list) or current_step_idx >= len(plan_steps):
        logger.error(f"Controller Node: Invalid plan_steps or current_step_idx ({current_step_idx}) out of bounds. Plan steps count: {len(plan_steps) if plan_steps else 'None'}")
        new_messages = existing_messages + [AIMessage(content="System Error: Invalid plan or step index for Controller. Plan may be complete or broken.")]
        # This state indicates the loop should probably end.
        return {**state, "controller_error": "Invalid plan or step index (likely plan completion).", "messages": new_messages}


    current_plan_step_dict = plan_steps[current_step_idx]
    logger.info(f"Controller Node: Processing Step {current_step_idx + 1}/{len(plan_steps)}: {current_plan_step_dict.get('description')}")

    previous_executor_output = state.get("previous_step_executor_output") 
    current_task_id_for_tools = state.get("task_id")
    try:
        tools = get_dynamic_tools(current_task_id_for_tools)
    except Exception as e:
        logger.error(f"Controller Node: Failed to load tools: {e}", exc_info=True)
        new_messages = existing_messages + [AIMessage(content=f"System Error: Failed to load tools for Controller: {e}")]
        return {**state, "controller_error": f"Failed to load tools: {e}", "messages": new_messages}

    callback_handler_from_config = config.get("callbacks")
    try:
        controller_result: Dict[str, Any] = await validate_and_prepare_step_action(
            original_user_query=original_user_query, current_plan_step=current_plan_step_dict,
            available_tools=tools, previous_step_executor_output=previous_executor_output,
            controller_llm_id_override=None, callback_handler=callback_handler_from_config)

        if controller_result.get("controller_error"):
            error_msg = f"Controller Error for Step {current_step_idx + 1}: {controller_result['controller_error']}"
            logger.error(error_msg)
            new_messages = existing_messages + [AIMessage(content=error_msg)]
            return {**state, **controller_result, "messages": new_messages} # Merge controller_result which includes the error

        ai_message_content = (f"Controller for Step {current_step_idx + 1}:\n"
            f"  Tool: {controller_result.get('controller_tool_name', 'None')}\n"
            f"  Input (summary): {str(controller_result.get('controller_tool_input', 'N/A'))[:100]}...\n"
            f"  Reasoning: {controller_result.get('controller_reasoning', 'N/A')}")
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        
        # Clear previous step's executor output before this step's execution
        # and clear previous step's evaluation fields
        updated_state_before_executor = {
            **state, 
            **controller_result, 
            "current_step_index": current_step_idx, 
            "messages": new_messages,
            "current_executor_output": None, # Clear for the upcoming execution
            "executor_error_message": None,
            "step_evaluation_achieved_goal": None,
            "step_evaluation_assessment": None,
            "step_evaluation_is_recoverable": None,
            "step_evaluation_suggested_tool": None,
            "step_evaluation_suggested_input_instructions": None,
            "step_evaluation_confidence_in_correction": None,
            "step_evaluation_error": None,
            "error_message": None 
        }
        return updated_state_before_executor
    except Exception as e:
        logger.error(f"Controller Node: Unexpected error for step {current_step_idx + 1}: {e}", exc_info=True)
        error_msg_content = f"System Error in Controller Node for step {current_step_idx + 1}: {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "controller_error": f"Unexpected error in Controller: {str(e)}", "messages": new_messages}

async def executor_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Executor Node ---")
    tool_name = state.get("controller_tool_name")
    tool_input_str = state.get("controller_tool_input")
    task_id = state.get("task_id")
    current_step_idx = state.get("current_step_index", 0) 
    plan_steps = state.get("plan_steps", [])
    step_description = "N/A"
    if plan_steps and current_step_idx < len(plan_steps):
        step_description = plan_steps[current_step_idx].get("description", "N/A")
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("executor_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    logger.info(f"Executor Node: Task ID '{task_id}', Step {current_step_idx + 1} ('{step_description}'). Tool: '{tool_name}', Input: '{str(tool_input_str)[:100]}...'")
    output_str_from_execution = "" # Renamed to avoid conflict with 'output' parameter name
    error_msg_from_execution = None # Renamed
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
        logger.warning("Executor Node: No tool name and no tool input provided.")
        output_str_from_execution = "No action taken by executor (no tool or input)."
    ai_message_content = f"Executor (Step {current_step_idx + 1}):\nTool: {tool_name or 'None'}\nOutput: {str(output_str_from_execution)[:500]}{'...' if len(str(output_str_from_execution)) > 500 else ''}"
    if error_msg_from_execution: ai_message_content += f"\nError: {error_msg_from_execution}"
    new_messages = existing_messages + [AIMessage(content=ai_message_content)]
    return {**state, "current_executor_output": str(output_str_from_execution), "executor_error_message": error_msg_from_execution, "messages": new_messages, "error_message": None}

async def step_evaluator_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Step Evaluator Node ---")
    original_user_query = state.get("user_query")
    plan_steps = state.get("plan_steps")
    current_step_idx = state.get("current_step_index", 0) # Should be correctly set by now
    controller_tool_used = state.get("controller_tool_name")
    controller_tool_input_str = state.get("controller_tool_input")
    executor_output_str = state.get("current_executor_output", "") 
    task_id = state.get("task_id")
    
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("step_evaluator_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []

    if not original_user_query or not plan_steps or current_step_idx >= len(plan_steps):
        error_msg = "Missing critical info for step evaluation (query, plan, or valid step index)."
        logger.error(f"Step Evaluator Node: {error_msg}")
        new_messages = existing_messages + [AIMessage(content=f"System Error: {error_msg}")]
        # Return a state indicating evaluation error and that goal was not achieved
        return {**state, 
                "step_evaluation_error": error_msg, 
                "step_evaluation_achieved_goal": False, # Critical for routing
                "messages": new_messages}

    current_plan_step_dict = plan_steps[current_step_idx]
    logger.info(f"Step Evaluator Node: Evaluating Step {current_step_idx + 1}: {current_plan_step_dict.get('description')}")

    try:
        tools = get_dynamic_tools(task_id)
    except Exception as e:
        logger.error(f"Step Evaluator Node: Failed to load tools: {e}", exc_info=True)
        new_messages = existing_messages + [AIMessage(content=f"System Error: Failed to load tools for Step Evaluator: {e}")]
        return {**state, "step_evaluation_error": f"Failed to load tools: {e}", "step_evaluation_achieved_goal": False, "messages": new_messages}

    callback_handler_from_config = config.get("callbacks")
    
    # Prepare the dictionary to be returned, starting with all existing state
    # This ensures fields not touched by the evaluator are preserved.
    return_state_update = {**state} 

    try:
        evaluation_result_dict: Dict[str, Any] = await evaluate_step_outcome_and_suggest_correction(
            original_user_query=original_user_query, current_plan_step=current_plan_step_dict,
            controller_tool_used=controller_tool_used, controller_tool_input=controller_tool_input_str,
            step_executor_output=executor_output_str, available_tools=tools,
            evaluator_llm_id_override=None, callback_handler=callback_handler_from_config)

        # Update the return_state_update with results from evaluation
        for key, value in evaluation_result_dict.items():
            # Keys from evaluation_result_dict are like "step_evaluation_achieved_goal", etc.
            # These directly match the keys in ResearchAgentState.
            return_state_update[key] = value
        
        # Ensure step_evaluation_error is None if no specific error from evaluator function
        if "step_evaluation_error" not in evaluation_result_dict:
             return_state_update["step_evaluation_error"] = None


        assessment = evaluation_result_dict.get('step_evaluation_assessment', 'N/A')
        achieved = evaluation_result_dict.get('step_evaluation_achieved_goal', False)
        ai_message_content = (f"Step Evaluator (Step {current_step_idx + 1}):\n"
            f"  Achieved Goal: {achieved}\n"
            f"  Assessment: {assessment[:300]}{'...' if len(assessment) > 300 else ''}")
        if not achieved and evaluation_result_dict.get('step_evaluation_is_recoverable'):
            ai_message_content += f"\n  Retry Suggested: Tool='{evaluation_result_dict.get('step_evaluation_suggested_tool', 'N/A')}', Instructions='{str(evaluation_result_dict.get('step_evaluation_suggested_input_instructions', 'N/A'))[:100]}...'"
        
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        return_state_update["messages"] = new_messages
        return_state_update["error_message"] = None # Clear general error if evaluator succeeded

        # MODIFIED: Logic for advancing step_index or preparing for retry
        if achieved:
            next_step_idx = current_step_idx + 1
            plan_steps_list = state.get("plan_steps", [])
            if next_step_idx < len(plan_steps_list):
                logger.info(f"Step Evaluator: Step {current_step_idx + 1} successful. Preparing for next step {next_step_idx + 1}.")
                return_state_update["current_step_index"] = next_step_idx
                return_state_update["retry_count_for_current_step"] = 0
                return_state_update["previous_step_executor_output"] = state.get("current_executor_output")
                # Clear fields for the next iteration of controller/executor/evaluator
                for key_to_clear in ["controller_tool_name", "controller_tool_input", "controller_reasoning", 
                                     "controller_confidence", "controller_error", "current_executor_output", 
                                     "executor_error_message", "step_evaluation_achieved_goal", 
                                     "step_evaluation_assessment", "step_evaluation_is_recoverable", 
                                     "step_evaluation_suggested_tool", "step_evaluation_suggested_input_instructions",
                                     "step_evaluation_confidence_in_correction", "step_evaluation_error"]:
                    return_state_update[key_to_clear] = None
            else: # All steps done
                logger.info(f"Step Evaluator: Step {current_step_idx + 1} successful. All plan steps completed.")
                # current_step_index remains, or set to len(plan_steps) to indicate completion
                return_state_update["current_step_index"] = len(plan_steps_list) 
        else: # Step failed
            logger.warning(f"Step Evaluator: Step {current_step_idx + 1} failed.")
            # Retry logic will be handled by the router based on is_recoverable and retry_count
            # No change to current_step_index here if retrying the same step.
            # previous_step_executor_output should remain from the failed attempt for context if needed.
            pass # Router will decide if it's a retry or end.

        return return_state_update

    except Exception as e:
        logger.error(f"Step Evaluator Node: Unexpected error for step {current_step_idx + 1}: {e}", exc_info=True)
        error_msg_content = f"System Error in Step Evaluator Node for step {current_step_idx + 1}: {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {
            **state, # Return previous state on unexpected error
            "step_evaluation_error": f"Unexpected error in Step Evaluator: {str(e)}",
            "step_evaluation_achieved_goal": False, 
            "messages": new_messages
        }

# --- Graph Definition ---
workflow_builder = StateGraph(ResearchAgentState)
workflow_builder.add_node("intent_classifier", intent_classifier_node)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("controller", controller_node)
workflow_builder.add_node("executor", executor_node) 
workflow_builder.add_node("step_evaluator", step_evaluator_node)

workflow_builder.set_entry_point("intent_classifier")

# --- Routing Functions ---
def route_after_intent_classification(state: ResearchAgentState):
    intent = state.get("classified_intent")
    if intent == "PLAN": return "planner"
    else: return END

def route_after_planner(state: ResearchAgentState):
    if state.get("plan_generation_error") or not state.get("plan_steps"):
        logger.info("Routing from Planner: Plan error or no steps, ending.")
        return END
    # State updates for current_step_index etc. are now done in planner_node or step_evaluator_node
    logger.info("Routing from Planner: Plan generated, proceeding to controller for first step.")
    return "controller"

def route_after_controller(state: ResearchAgentState):
    if state.get("controller_error"):
        logger.error(f"Routing from Controller: Controller error: {state['controller_error']}. Ending.")
        return END
    logger.info("Routing from Controller: Controller finished, proceeding to executor.")
    return "executor"

def route_after_executor(state: ResearchAgentState):
    logger.info("Routing from Executor: Executor finished, proceeding to step_evaluator.")
    return "step_evaluator"

def route_after_step_evaluation(state: ResearchAgentState):
    logger.info("--- Routing Decision after Step Evaluation ---")
    achieved_goal = state.get("step_evaluation_achieved_goal")
    is_recoverable = state.get("step_evaluation_is_recoverable")
    current_step_idx = state.get("current_step_index", 0) # This index is what was *just* evaluated
    plan_steps = state.get("plan_steps", [])
    retry_count = state.get("retry_count_for_current_step", 0)
    max_retries = settings.agent_max_step_retries 

    if state.get("step_evaluation_error"):
        logger.error(f"Routing: Step evaluation itself failed: {state['step_evaluation_error']}. Ending plan.")
        return END 

    if achieved_goal:
        logger.info(f"Routing: Step {current_step_idx + 1} achieved goal.")
        # current_step_index should have been updated by step_evaluator_node if proceeding
        next_step_to_process_idx = state.get("current_step_index") # This is the *new* index for the *next* step
        
        if next_step_to_process_idx is not None and next_step_to_process_idx < len(plan_steps):
            logger.info(f"Routing: Proceeding to Controller for next step: {next_step_to_process_idx + 1}")
            return "controller"
        else: # All steps done
            logger.info("Routing: All plan steps completed successfully or last step was processed.")
            # TODO: Route to OverallEvaluatorNode later
            return END 
    else: # Step failed
        logger.warning(f"Routing: Step {current_step_idx + 1} failed. Assessment: {state.get('step_evaluation_assessment')}")
        if is_recoverable and retry_count < max_retries:
            logger.info(f"Routing: Attempting retry {retry_count + 1}/{max_retries} for step {current_step_idx + 1}.")
            # The state for retry_count_for_current_step needs to be updated for the *next* controller call.
            # This should ideally be done by a node or by the evaluator returning it.
            # For now, we assume step_evaluator_node doesn't increment retry_count, so we do it here conceptually
            # but the state update must happen IN a node.
            # Let's assume step_evaluator_node will return the incremented retry_count if it suggests a retry.
            # If not, this retry logic needs a dedicated state update node.
            # For now, if it's recoverable, we just go back to controller.
            # The controller will re-process the *same* current_step_idx.
            # The step_evaluator_node should return the *next* retry_count.
            # Let's make step_evaluator_node responsible for setting retry_count for next attempt.
            # If step_evaluator_node already set `retry_count_for_current_step` to `retry_count + 1`
            # then we just route to controller.
            # If it didn't, we need a small node to update it.
            # Let's assume for now step_evaluator_node returns the updated retry count.
            return "controller" 
        else:
            logger.error(f"Routing: Step {current_step_idx + 1} failed and is not recoverable or retries exhausted. Ending plan.")
            return END

workflow_builder.add_conditional_edges("intent_classifier", route_after_intent_classification, {"planner": "planner", END: END})
workflow_builder.add_conditional_edges("planner", route_after_planner, {"controller": "controller", END: END})
workflow_builder.add_conditional_edges("controller", route_after_controller, {"executor": "executor", END: END}) 
workflow_builder.add_conditional_edges("executor", route_after_executor, {"step_evaluator": "step_evaluator"})
workflow_builder.add_conditional_edges("step_evaluator", route_after_step_evaluation, {"controller": "controller", END: END})

try:
    research_agent_graph = workflow_builder.compile()
    logger.info("ResearchAgent LangGraph compiled successfully with Step Evaluator and loop logic.")
except Exception as e:
    logger.critical(f"Failed to compile ResearchAgent LangGraph: {e}", exc_info=True)
    research_agent_graph = None

# --- Test Runner ---
# (run_graph_example and __main__ block remain the same as Canvas 12)
async def run_graph_example(user_input: str, ws_callback_handler: Optional[WebSocketCallbackHandler] = None):
    if not research_agent_graph: logger.error("Graph not compiled."); return None
    initial_state: ResearchAgentState = {
        "user_query": user_input, "messages": [HumanMessage(content=user_input)], 
        "task_id": "test_task_for_eval_loop", "classified_intent": None, 
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
        "error_message": None 
    }
    callbacks_to_use = [ws_callback_handler] if ws_callback_handler else []
    config_for_run = RunnableConfig(callbacks=callbacks_to_use, recursion_limit=50) 
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
                           "retry_count_for_current_step",
                           "executor_error_message", "error_message", 
                           "plan_generation_error", "controller_error", "step_evaluation_error"]
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
        print(f"\n--- Running LangGraph Test (Full Loop - 1st step) for: '{test_query_plan_tool}' ---")
        final_state_tool = await run_graph_example(test_query_plan_tool) 
        if final_state_tool:
            print(f"\n--- Test Run Complete (Full Loop - 1st step) ---")
            assert final_state_tool.get('classified_intent') == "PLAN"
            assert final_state_tool.get('plan_steps') is not None and len(final_state_tool['plan_steps']) > 0
            # This assertion might be too strict if the plan finishes early due to an error or if only one step is successful
            # For a plan of 4-5 steps, and if it completes all, messages would be > 5 (Human, Intent, Plan, C, E, S for each step)
            # Let's check for at least one full loop: Human, Intent, Plan, Controller, Executor, Evaluator
            assert len(final_state_tool.get('messages', [])) >= 6, f"Expected at least 6 messages for one full step, got {len(final_state_tool.get('messages', []))}"
            print(f"Final number of messages: {len(final_state_tool.get('messages', []))}")
            print(f"Last step evaluated achieved goal: {final_state_tool.get('step_evaluation_achieved_goal')}")
            print(f"Final current_step_index: {final_state_tool.get('current_step_index')}")
        else: print("\n--- Test Run Failed (Full Loop - 1st step) ---")
    asyncio.run(run_main_test())
