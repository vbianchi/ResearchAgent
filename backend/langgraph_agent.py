# backend/langgraph_agent.py
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, TypedDict, Annotated, Union
from uuid import uuid4
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # For persistence
from langchain_core.runnables import RunnableConfig


# Project specific imports - assuming these are in the parent directory or correctly pathed
from backend.config import settings
from backend.intent_classifier import classify_intent
from backend.planner import generate_plan, PlanStep
from backend.controller import validate_and_prepare_step_action, ControllerOutput
from backend.evaluator import (
    evaluate_step_outcome_and_suggest_correction, StepCorrectionOutcome,
    evaluate_plan_outcome, EvaluationResult
)
from backend.tools.tool_loader import get_dynamic_tools_from_config # MODIFIED: Using new tool loader
from backend.tools.standard_tools import get_task_workspace_path # For workspace access

# --- Logging Setup ---
logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - Line %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


# --- Constants ---
MAX_STEP_RETRIES = settings.agent_max_step_retries # Defined in config.py

# --- State Definition for the Graph ---
class ResearchAgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.
    It is passed between nodes and updated by each node.
    """
    user_query: str
    messages: Annotated[List[BaseMessage], operator.add]
    task_id: str # For tool usage, workspace management, and logging

    # Intent Classification
    classified_intent: Optional[str]
    intent_classifier_reasoning: Optional[str]

    # Planning
    plan_summary: Optional[str]
    plan_steps: Optional[List[PlanStep]] # List of Pydantic models
    plan_generation_error: Optional[str]

    # Execution Control
    current_step_index: int
    previous_step_executor_output: Optional[str] # Output from the last successfully executed step
    retry_count_for_current_step: int

    # Controller
    controller_tool_name: Optional[str]
    controller_tool_input: Optional[Union[str, Dict[str, Any]]] # Can be string or dict for some tools
    controller_reasoning: Optional[str]
    controller_confidence: Optional[float]
    controller_error: Optional[str]

    # Executor
    current_executor_output: Optional[str] # Output from the current step's execution attempt
    executor_error_message: Optional[str] # Error message if executor fails

    # Step Evaluation
    step_evaluation_achieved_goal: Optional[bool]
    step_evaluation_assessment: Optional[str]
    step_evaluation_is_recoverable: Optional[bool]
    step_evaluation_suggested_tool: Optional[str]
    step_evaluation_suggested_input_instructions: Optional[str]
    step_evaluation_confidence_in_correction: Optional[float]
    step_evaluation_error: Optional[str]

    # Overall Evaluation
    overall_evaluation_success: Optional[bool]
    overall_evaluation_assessment: Optional[str]
    overall_evaluation_final_answer_content: Optional[str] # The final answer to present to user
    overall_evaluation_suggestions_for_replan: Optional[List[str]]
    overall_evaluation_error: Optional[str]

    # General error tracking if something goes wrong outside specific components
    error_message: Optional[str]
    
    # LLM Configuration Overrides (per session, if any)
    session_intent_classifier_llm_id: Optional[str]
    session_planner_llm_id: Optional[str]
    session_controller_llm_id: Optional[str]
    session_executor_llm_id: Optional[str] # For the ReAct agent or direct LLM calls in Executor
    session_evaluator_llm_id: Optional[str]


# --- Node Functions ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classifies the user's intent."""
    logger.info("--- Entering Intent Classifier Node ---")
    user_query = state["user_query"]
    task_id = state.get("task_id", "default_task_for_intent") # Fallback if task_id not set

    # MODIFIED: Use get_dynamic_tools_from_config
    available_tools = get_dynamic_tools_from_config(task_id)
    tools_summary_for_intent = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools])

    # Pass session-specific LLM override if available
    # Note: classify_intent and other components will need to be updated
    # to accept and use these session_*.llm_id fields if they are to be truly dynamic per session.
    # For now, this state field is present, but classify_intent uses settings directly.
    # This will be part of the integration with server.py
    
    # The classify_intent function now fetches its own LLM based on settings
    # and the session_intent_classifier_llm_id (if we modify it to accept this from state)
    intent_output = await classify_intent(
        user_query=user_query,
        available_tools_summary=tools_summary_for_intent,
        # session_llm_id_override=state.get("session_intent_classifier_llm_id") # Future enhancement
    )
    
    # Assuming classify_intent returns a dict like {'intent': 'PLAN', 'reasoning': '...'}
    # or just the intent string. Adapt as needed.
    # For now, assuming it returns the intent string directly, and reasoning is logged internally.
    # Let's refine this based on the actual classify_intent output structure.
    # Based on current intent_classifier.py, it returns a string. Reasoning is logged.
    # We need to update intent_classifier.py to return a dict or tuple if we want reasoning in state.
    # For now, let's assume it's just the intent string.

    classified_intent = intent_output # Directly from classify_intent
    reasoning = "Reasoning logged internally by intent_classifier." # Placeholder
    
    # For the demo, let's simulate reasoning being part of the output for logging
    if isinstance(intent_output, tuple) and len(intent_output) == 2: # If it returns (intent, reasoning_str)
        classified_intent_str = intent_output[0]
        reasoning_str = intent_output[1]
    elif isinstance(intent_output, dict) and "intent" in intent_output: # If it returns a dict
        classified_intent_str = intent_output["intent"]
        reasoning_str = intent_output.get("reasoning", "No reasoning provided by classifier.")
    else: # Assuming it's just the intent string
        classified_intent_str = str(intent_output)
        # Attempt to find reasoning from logs if possible (conceptual, not practical here)
        # For the test, we'll use a placeholder if not directly returned.
        reasoning_str = "Reasoning is typically logged by the intent_classifier module."
        # Example of how it might be structured if intent_classifier.py is updated:
        # logger.info(f"Intent classification result: {classified_intent_str}, Reasoning: {reasoning_str}")


    logger.info(f"Intent classification result: {classified_intent_str}, Reasoning: {reasoning_str}")
    return {
        "classified_intent": classified_intent_str,
        "intent_classifier_reasoning": reasoning_str,
        "messages": [AIMessage(content=f"Intent Classified as: {classified_intent_str}\nReasoning: {reasoning_str}")]
    }

async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generates a plan if intent is 'PLAN'."""
    logger.info("--- Entering Planner Node ---")
    if state["classified_intent"] != "PLAN":
        logger.info("Planner Node: Intent is not PLAN, skipping plan generation.")
        return {"plan_summary": "No plan needed.", "plan_steps": []}

    user_query = state["user_query"]
    task_id = state.get("task_id", "default_task_for_planner")
    logger.info(f"Planner Node: current_task_id_for_tools from state: {task_id}")

    # MODIFIED: Use get_dynamic_tools_from_config
    available_tools = get_dynamic_tools_from_config(task_id)
    tools_summary = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools])

    # generate_plan now fetches its own LLM based on settings
    # and session_planner_llm_id (if updated to accept from state)
    summary, steps_dicts = await generate_plan(
        user_query=user_query,
        available_tools_summary=tools_summary,
        # session_llm_id_override=state.get("session_planner_llm_id") # Future enhancement
    )

    if summary and steps_dicts:
        # Convert list of dicts to list of PlanStep Pydantic models
        plan_steps_models = [PlanStep(**step_data) for step_data in steps_dicts]
        logger.info(f"Planner Node: Plan generated. Summary: {summary}")
        return {
            "plan_summary": summary,
            "plan_steps": plan_steps_models,
            "current_step_index": 0, # Initialize for execution
            "retry_count_for_current_step": 0,
            "messages": [AIMessage(content=f"Plan Generated:\nSummary: {summary}\nSteps:\n" + "\n".join(f"{i+1}. {s.description}" for i, s in enumerate(plan_steps_models)))]
        }
    else:
        logger.error("Planner Node: Failed to generate plan.")
        return {
            "plan_summary": "Failed to generate a plan.",
            "plan_steps": [],
            "plan_generation_error": "Planner failed to produce a valid plan structure.",
            "messages": [AIMessage(content="Error: Failed to generate a plan.")]
        }

async def controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Determines the tool and input for the current step, or if direct LLM call is needed."""
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

    # MODIFIED: Check for MAX_STEP_RETRIES exceeded (precautionary, router should handle this)
    if state.get("retry_count_for_current_step", 0) >= MAX_STEP_RETRIES and attempt_number > MAX_STEP_RETRIES : # Check if current retries already hit max
        # This situation means the router might have let it through when it shouldn't have.
        # This primarily applies if this is a *new* attempt beyond the max.
        # If retry_count is already at MAX_STEP_RETRIES, this is the *last allowed attempt*.
        logger.warning(f"Controller Node: Entered for Step {current_step_idx + 1} which has already reached max retries ({state.get('retry_count_for_current_step', 0)} >= {MAX_STEP_RETRIES}). This indicates a potential routing issue if this is not the final allowed attempt.")
        # Let it proceed for this final attempt if the router allowed it.

    # MODIFIED: Use get_dynamic_tools_from_config
    available_tools = get_dynamic_tools_from_config(task_id)
    
    # Prepare plan_step_for_controller by applying retry suggestions if this is a retry
    plan_step_for_controller_call = current_plan_step.copy(deep=True)
    if state.get("retry_count_for_current_step", 0) > 0:
        logger.info(f"Controller Node: This is a retry (attempt {attempt_number}). Using evaluator suggestions if available.")
        # Apply suggestions from StepEvaluator if this is a retry
        suggested_tool = state.get("step_evaluation_suggested_tool")
        suggested_input_instr = state.get("step_evaluation_suggested_input_instructions")

        if suggested_tool is not None: # Can be "None" string or an actual tool name
            plan_step_for_controller_call.tool_to_use = suggested_tool if suggested_tool != "None" else None
            logger.info(f"  Retry: Overriding tool to: '{plan_step_for_controller_call.tool_to_use}'")
        if suggested_input_instr is not None:
            plan_step_for_controller_call.tool_input_instructions = suggested_input_instr
            logger.info(f"  Retry: Overriding input instructions to: '{suggested_input_instr[:100]}...'")
            
    # validate_and_prepare_step_action now fetches its own LLM
    # It will use session_controller_llm_id from state if we pass the whole state or relevant part.
    controller_output_model: Optional[ControllerOutput] = await validate_and_prepare_step_action(
        original_user_query=state["user_query"],
        plan_step=plan_step_for_controller_call, # Use potentially modified plan step
        available_tools=available_tools,
        # Pass relevant parts of the state for LLM selection and context
        session_llm_id_override=state.get("session_controller_llm_id"),
        previous_step_output=state.get("previous_step_executor_output")
    )

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
        logger.error("Controller Node: Failed to get valid output from controller LLM.")
        return {"controller_error": "Controller LLM failed to produce valid structured output."}


async def executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Executes the tool or direct LLM call decided by the Controller."""
    logger.info("--- Entering Executor Node ---")
    tool_name = state.get("controller_tool_name")
    tool_input = state.get("controller_tool_input")
    task_id = state.get("task_id", "default_task_for_executor")
    current_step_idx = state["current_step_index"]
    plan_steps = state.get("plan_steps", [])
    current_plan_step_desc = plan_steps[current_step_idx].description if plan_steps and current_step_idx < len(plan_steps) else "N/A"
    attempt_number = state.get("retry_count_for_current_step", 0) + 1

    logger.info(f"Executor Node: Task ID '{task_id}', Step {current_step_idx + 1} ('{current_plan_step_desc}'), Attempt {attempt_number}. Tool: '{tool_name}', Input: '{str(tool_input)[:100]}...'")

    # MODIFIED: Use get_dynamic_tools_from_config
    available_tools = get_dynamic_tools_from_config(task_id)
    tool_map = {tool.name: tool for tool in available_tools}

    executor_output: str
    if tool_name and tool_name != "None":
        if tool_name in tool_map:
            selected_tool = tool_map[tool_name]
            logger.info(f"Executor Node: Executing tool '{tool_name}'")
            try:
                # Tool invocation might be sync or async depending on the tool's implementation
                if asyncio.iscoroutinefunction(selected_tool.arun):
                    tool_result = await selected_tool.arun(tool_input, config=config)
                else: # Fallback for synchronous tools if any are still structured that way
                    tool_result = await asyncio.to_thread(selected_tool.run, tool_input, config=config) # type: ignore
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
    else: # "None" tool - direct LLM call
        logger.info(f"Executor Node: 'None' tool specified. Executing direct LLM call with input: {str(tool_input)[:100]}...")
        # Here, tool_input is the directive for the LLM
        # This requires an LLM instance. For now, using the "Executor" role LLM.
        # This part needs to be more robust, possibly using a ReAct agent or a simpler LLMChain.
        # For this test, a simple LLM call.
        try:
            # Get LLM for direct execution (e.g., from session_executor_llm_id or default)
            from backend.llm_setup import get_llm # Local import to avoid circularity if not careful
            
            executor_llm_provider = state.get("session_executor_llm_id", settings.executor_default_provider)
            executor_llm_model = settings.executor_default_model_name # Default, needs splitting if session_executor_llm_id is full
            
            if state.get("session_executor_llm_id"):
                try:
                    provider, model = state.get("session_executor_llm_id").split("::",1)
                    executor_llm_provider = provider
                    executor_llm_model = model
                except ValueError:
                    logger.warning(f"Could not parse session_executor_llm_id: {state.get('session_executor_llm_id')}. Using defaults.")


            direct_llm = get_llm(settings, provider=executor_llm_provider, model_name=executor_llm_model, requested_for_role="EXECUTOR_DirectLLM")
            
            # Construct a prompt for the LLM
            # The 'tool_input' here is the directive from the Controller for the 'None' tool case.
            # We also need to provide the previous step's output if relevant and available.
            prompt_messages = []
            if state.get("previous_step_executor_output"):
                prompt_messages.append(HumanMessage(content=f"Context from previous step's output:\n{state['previous_step_executor_output']}"))
            
            prompt_messages.append(HumanMessage(content=str(tool_input))) # The directive from Controller

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
    """Evaluates the outcome of the current step and suggests corrections if needed."""
    logger.info("--- Entering Step Evaluator Node ---")
    current_step_idx = state["current_step_index"]
    plan_steps = state.get("plan_steps", [])
    task_id = state.get("task_id", "default_task_for_evaluator")
    attempt_number = state.get("retry_count_for_current_step", 0) + 1 # This is the attempt that was just made

    if not plan_steps or current_step_idx >= len(plan_steps):
        logger.error("Step Evaluator Node: No plan steps or index out of bounds.")
        return {"step_evaluation_error": "No plan steps or index out of bounds."}

    current_plan_step = plan_steps[current_step_idx]
    logger.info(f"Step Evaluator Node: Evaluating Step {current_step_idx + 1} (Attempt {attempt_number}): {current_plan_step.description}")

    # MODIFIED: Use get_dynamic_tools_from_config
    available_tools = get_dynamic_tools_from_config(task_id)
    
    # evaluate_step_outcome_and_suggest_correction now fetches its own LLM
    # It will use session_evaluator_llm_id from state if we pass the whole state or relevant part.
    evaluation_result_model: Optional[StepCorrectionOutcome] = await evaluate_step_outcome_and_suggest_correction(
        original_user_query=state["user_query"],
        plan_step_being_evaluated=current_plan_step,
        controller_tool_used=state.get("controller_tool_name"),
        controller_tool_input=state.get("controller_tool_input"),
        step_executor_output=state.get("current_executor_output", "No output from executor."),
        available_tools=available_tools,
        session_llm_id_override=state.get("session_evaluator_llm_id")
    )

    if not evaluation_result_model:
        logger.error("Step Evaluator Node: Failed to get evaluation result from LLM.")
        # Default to step failed, not recoverable, to avoid infinite loops on evaluator failure
        return {
            "step_evaluation_achieved_goal": False,
            "step_evaluation_assessment": "Evaluator LLM failed to produce an assessment.",
            "step_evaluation_is_recoverable": False, # Critical: default to not recoverable
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
        patch["previous_step_executor_output"] = state.get("current_executor_output") # Save successful output
        patch["retry_count_for_current_step"] = 0 # Reset for the next step
    else:
        logger.warning(f"Step Evaluator: Step {current_step_idx + 1} (Attempt {attempt_number}) failed.")
        # **MODIFIED START: Correctly increment retry_count_for_current_step for the *next* attempt of the *same* step**
        current_retry_count = state.get("retry_count_for_current_step", 0)
        patch["retry_count_for_current_step"] = current_retry_count + 1
        # current_step_index remains the same, so the same step is retried
        patch["previous_step_executor_output"] = None # Clear previous output as it was from a failed attempt
        # **MODIFIED END**

        if evaluation_result_model.is_recoverable_via_retry:
            logger.info(f"  Step Evaluator: Preparing for retry attempt {patch['retry_count_for_current_step']} for step {current_step_idx + 1}.")
        else:
            logger.warning(f"  Step Evaluator: Step {current_step_idx + 1} failed and is not recoverable.")
            # The router will handle moving to overall_evaluator if retries are exhausted or not recoverable

    return patch


async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Evaluates the overall plan outcome after all steps or if a step fails unrecoverably."""
    logger.info("--- Entering Overall Evaluator Node ---")
    user_query = state["user_query"]
    plan_summary = state.get("plan_summary", "N/A")
    plan_steps = state.get("plan_steps", [])
    
    # Construct a summary of executed steps and their outcomes for the evaluator
    # This needs to be built up during the execution loop or reconstructed from messages.
    # For this example, we'll use the last executor output and step assessment.
    
    executed_plan_summary_str = f"Plan Summary: {plan_summary}\n"
    if plan_steps:
        executed_plan_summary_str += "Steps Attempted:\n"
        for i, step in enumerate(plan_steps):
            executed_plan_summary_str += f"  {i+1}. {step.description}\n"
            # Ideally, we'd have a more detailed log of each step's attempt(s) here.
            # For now, using the last recorded step assessment if available.
            if i == state["current_step_index"] -1 : # If this was the last attempted/completed step
                 executed_plan_summary_str += f"    Last Status: {'Successful' if state.get('step_evaluation_achieved_goal') else 'Failed'}. Assessment: {state.get('step_evaluation_assessment', 'N/A')}\n"

    final_agent_answer_for_eval = state.get("current_executor_output") # Output of the last executed step
    if state.get("executor_error_message"):
        final_agent_answer_for_eval = f"Execution failed with error: {state.get('executor_error_message')}"
    elif state.get("step_evaluation_error"):
         final_agent_answer_for_eval = f"Step evaluation failed: {state.get('step_evaluation_error')}"
    elif not state.get("step_evaluation_achieved_goal") and state.get("step_evaluation_assessment"):
        final_agent_answer_for_eval = f"Last step failed. Assessment: {state.get('step_evaluation_assessment')}"
    elif not final_agent_answer_for_eval: # If no output from last step, use assessment
        final_agent_answer_for_eval = state.get("step_evaluation_assessment", "Plan concluded without a specific final output from the last step.")


    logger.info(f"Overall Evaluator: Evaluating plan for query: '{user_query[:100]}...' ")
    logger.debug(f"Overall Evaluator: Plan Summary for Eval: {executed_plan_summary_str[:300]}...")
    logger.debug(f"Overall Evaluator: Final Answer for Eval: {str(final_agent_answer_for_eval)[:300]}...")

    # evaluate_plan_outcome now fetches its own LLM
    # It will use session_evaluator_llm_id from state if we pass the whole state or relevant part.
    overall_eval_result: Optional[EvaluationResult] = await evaluate_plan_outcome(
        original_user_query=user_query,
        executed_plan_summary=executed_plan_summary_str,
        final_agent_answer=str(final_agent_answer_for_eval), # Ensure it's a string
        session_llm_id_override=state.get("session_evaluator_llm_id")
    )

    if overall_eval_result:
        logger.info(f"Evaluator (Overall): Success: {overall_eval_result.overall_success}, Assessment: '{overall_eval_result.assessment}'")
        return {
            "overall_evaluation_success": overall_eval_result.overall_success,
            "overall_evaluation_assessment": overall_eval_result.assessment,
            "overall_evaluation_final_answer_content": overall_eval_result.assessment, # For now, use assessment as final answer
            "overall_evaluation_suggestions_for_replan": overall_eval_result.suggestions_for_replan,
            "messages": [AIMessage(content=f"Overall Plan Evaluation:\n  Success: {overall_eval_result.overall_success}\n  Assessment: {overall_eval_result.assessment}")]
        }
    else:
        logger.error("Overall Evaluator Node: Failed to get evaluation result from LLM.")
        final_assessment = "Overall evaluation failed to produce a result. Plan outcome uncertain."
        return {
            "overall_evaluation_success": False, # Default to failure if evaluator fails
            "overall_evaluation_assessment": final_assessment,
            "overall_evaluation_final_answer_content": final_assessment,
            "overall_evaluation_error": "Overall Evaluator LLM failed.",
            "messages": [AIMessage(content=f"Overall Plan Evaluation:\n  Error: Evaluation LLM failed.")]
        }

# --- Conditional Edges ---

def should_plan(state: ResearchAgentState) -> str:
    """Determines if planning is needed based on intent."""
    logger.info("--- Routing Decision: Should Plan? ---")
    if state.get("classified_intent") == "PLAN":
        logger.info("Routing: Intent is PLAN, proceeding to planner.")
        return "planner"
    else: # DIRECT_QA or other
        logger.info("Routing: Intent is NOT PLAN (e.g., DIRECT_QA), proceeding to overall_evaluator (or a dedicated QA node in future).")
        # For now, DIRECT_QA will also go to overall_evaluator which will use the intent classification message.
        # Later, add a dedicated direct_qa_node.
        return "overall_evaluator" # Or "direct_qa_node" in a more complex graph

def should_continue_after_step_eval(state: ResearchAgentState) -> str:
    """
    Determines the next step after step evaluation:
    - If step successful and more steps exist: continue to controller for next step.
    - If step successful and no more steps: go to overall_evaluator.
    - If step failed and recoverable and retries left: go back to controller for retry.
    - If step failed and not recoverable OR retries exhausted: go to overall_evaluator.
    """
    logger.info("--- Routing Decision after Step Evaluation ---")
    step_achieved_goal = state.get("step_evaluation_achieved_goal")
    is_recoverable = state.get("step_evaluation_is_recoverable")
    current_step_idx = state["current_step_index"] # This is the index of the *next* step if current one succeeded, or *same* if failed.
    plan_steps = state.get("plan_steps", [])
    
    # **MODIFIED: Use the retry_count that was *just set* by the step_evaluator for the current step's attempt**
    # The retry_count_for_current_step in the state *after* step_evaluator has run reflects the count *for the next attempt* if a retry is happening for the *same step_index*.
    # If the step succeeded, step_evaluator should have reset retry_count_for_current_step to 0 and incremented current_step_index.
    
    retry_count_for_next_attempt_of_this_step = state.get("retry_count_for_current_step", 0)
    
    logger.info(f"Router after Step Eval: achieved_goal={step_achieved_goal}, is_recoverable={is_recoverable}, idx_for_next_action={current_step_idx}, plan_len={len(plan_steps)}, retry_for_next_attempt={retry_count_for_next_attempt_of_this_step}")

    if step_achieved_goal:
        if current_step_idx < len(plan_steps):
            logger.info(f"Routing: Step successful. Proceeding to Controller for next step: {current_step_idx + 1} (index {current_step_idx})")
            return "controller" # Continue to next step
        else:
            logger.info(f"Routing: All plan steps completed successfully. Index {current_step_idx} >= Plan length {len(plan_steps)}. Proceeding to Overall Evaluator.")
            return "overall_evaluator" # All steps done
    else: # Step failed
        assessment = state.get('step_evaluation_assessment', 'Unknown failure reason.')
        logger.warning(f"Routing: Step (index {current_step_idx}) failed. Assessment: {assessment}")
        # **MODIFIED: Check if retries for the *current* step (which is current_step_idx as it wasn't incremented) are exhausted.**
        # retry_count_for_next_attempt_of_this_step already reflects the incremented count for the *next* try of the *same* step.
        if is_recoverable and retry_count_for_next_attempt_of_this_step <= MAX_STEP_RETRIES:
            logger.info(f"Routing: Step failed, but is recoverable. Proceeding to Controller for retry attempt {retry_count_for_next_attempt_of_this_step} of step {current_step_idx + 1}.")
            return "controller" # Retry the same step (current_step_idx hasn't changed)
        else:
            if not is_recoverable:
                logger.warning(f"Routing: Step failed and is not recoverable. Proceeding to Overall Evaluator.")
            else: # Retries exhausted
                logger.warning(f"Routing: Step failed and retries exhausted ({retry_count_for_next_attempt_of_this_step-1} >= {MAX_STEP_RETRIES}). Proceeding to Overall Evaluator.")
            return "overall_evaluator" # Unrecoverable or retries exhausted

# --- Build the Graph ---
def create_research_agent_graph() -> StateGraph:
    """Creates and compiles the LangGraph for the Research Agent."""
    workflow = StateGraph(ResearchAgentState)

    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("controller", controller_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("step_evaluator", step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node)

    # Define edges
    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        should_plan,
        {
            "planner": "planner",
            "overall_evaluator": "overall_evaluator" # Or a dedicated "direct_qa_node"
        }
    )
    # After planner, always go to controller for the first step (or if no plan, this path isn't taken)
    # If planner fails to generate a plan, it should ideally route to overall_evaluator or END.
    # For now, assuming planner always produces something or the should_plan handles empty plan.
    # Let's refine planner_node to handle its own failure case and set state for routing.
    # If plan_generation_error is set by planner, we need a condition.
    # For now, simple:
    workflow.add_edge("planner", "controller")


    # Loop: Controller -> Executor -> StepEvaluator -> (back to Controller or to OverallEvaluator)
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")

    workflow.add_conditional_edges(
        "step_evaluator",
        should_continue_after_step_eval,
        {
            "controller": "controller", # For next step or retry
            "overall_evaluator": "overall_evaluator" # All steps done or unrecoverable failure
        }
    )

    workflow.add_edge("overall_evaluator", END)

    # Compile the graph
    # Add memory for persistence (optional, good for long-running tasks)
    # memory = SqliteSaver.from_conn_string(":memory:") # In-memory for now
    # research_agent_graph = workflow.compile(checkpointer=memory)
    research_agent_graph = workflow.compile()
    
    logger.info("ResearchAgent LangGraph compiled successfully with Overall Evaluator and full loop logic.")
    return research_agent_graph


# --- Main Test Execution (Example) ---
async def run_graph_example(query: str, task_id_override: Optional[str] = None):
    """Helper function to run a test query through the graph."""
    research_agent_graph = create_research_agent_graph()
    
    # Create a unique task_id for this run if not provided
    current_task_id = task_id_override if task_id_override else f"test_task_for_overall_eval"
    
    # Ensure workspace for this task_id exists (tool_loader might also do this)
    get_task_workspace_path(current_task_id, create_if_not_exists=True)

    initial_state: ResearchAgentState = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "task_id": current_task_id,
        "classified_intent": None,
        "intent_classifier_reasoning": None,
        "plan_summary": None,
        "plan_steps": None,
        "plan_generation_error": None,
        "current_step_index": 0,
        "previous_step_executor_output": None,
        "retry_count_for_current_step": 0,
        "controller_tool_name": None,
        "controller_tool_input": None,
        "controller_reasoning": None,
        "controller_confidence": None,
        "controller_error": None,
        "current_executor_output": None,
        "executor_error_message": None,
        "step_evaluation_achieved_goal": None,
        "step_evaluation_assessment": None,
        "step_evaluation_is_recoverable": None,
        "step_evaluation_suggested_tool": None,
        "step_evaluation_suggested_input_instructions": None,
        "step_evaluation_confidence_in_correction": None,
        "step_evaluation_error": None,
        "overall_evaluation_success": None,
        "overall_evaluation_assessment": None,
        "overall_evaluation_final_answer_content": None,
        "overall_evaluation_suggestions_for_replan": None,
        "overall_evaluation_error": None,
        "error_message": None,
        "session_intent_classifier_llm_id": None, # Example, set if testing overrides
        "session_planner_llm_id": None,
        "session_controller_llm_id": None,
        "session_executor_llm_id": None,
        "session_evaluator_llm_id": None,
    }

    # For testing, we can use a unique config for each run if using persistence
    config_for_run = RunnableConfig(configurable={"thread_id": f"test_run_{uuid4()}"})

    logger.info(f"Streaming LangGraph execution for query: '{query}'")
    final_state = None
    async for chunk in research_agent_graph.astream(initial_state, config=config_for_run):
        logger.info("--- Graph Stream Chunk ---")
        # Print keys that have changed or are significant for this chunk
        # This is a simplified view; Langfuse or other tracing would be better for deep inspection.
        for key, value in chunk.items():
            # Log the full state of the node that just ran
            logger.info(f"State after Node '{key}':")
            node_state = value # The 'value' here is the full state dict after the node 'key' ran
            for k, v in node_state.items():
                if isinstance(v, list) and len(v) > 3 and k != "messages": # Avoid overly long lists in basic log
                    logger.info(f"    {k.replace('_', ' ').title()}: List with {len(v)} items (first 3 shown)...")
                    for i, item in enumerate(v[:3]):
                        logger.info(f"        Item {i}: {str(item)[:100]}...")
                elif isinstance(v, str) and len(v) > 150:
                    logger.info(f"    {k.replace('_', ' ').title()}: {v[:150]}...")
                elif v is not None : # Only print if not None, to keep logs cleaner
                    logger.info(f"    {k.replace('_', ' ').title()}: {v}")
            final_state = value # Keep track of the latest full state

    logger.info("--- End of Graph Stream ---")
    
    if final_state:
        logger.info("--- Final Accumulated Graph State ---")
        for key, value in final_state.items():
            if isinstance(value, list) and len(value) > 3 and key != "messages":
                 logger.info(f"{key.replace('_', ' ').title()}: List with {len(value)} items (first 3 shown)...")
                 for i, item_in_list in enumerate(value[:3]):
                     logger.info(f"    Item {i}: {str(item_in_list)[:100]}...")
            elif isinstance(value, str) and len(value) > 200 and key != "user_query": # Don't truncate user_query
                 logger.info(f"{key.replace('_', ' ').title()}: {value[:200]}...")
            else:
                 logger.info(f"{key.replace('_', ' ').title()}: {value}")
    return final_state


async def run_main_test():
    # Test Case 1: Successful multi-step plan (previous solar example)
    # test_query_solar = "Research the benefits of solar power and write a short summary file called 'solar_benefits.txt'."
    # print(f"\n--- Running LangGraph Test (Full PCEE Loop) for: '{test_query_solar}' ---")
    # final_state_solar = await run_graph_example(test_query_solar, task_id_override="test_task_solar_001")
    # if final_state_solar:
    #     print(f"\n--- Test Run Complete (Full PCEE Loop) ---")
    #     print(f"Overall Evaluation Success: {final_state_solar.get('overall_evaluation_success')}")
    #     print(f"Overall Evaluation Assessment: {final_state_solar.get('overall_evaluation_assessment')}")
    #     print(f"Final number of messages: {len(final_state_solar.get('messages', []))}")
    # else:
    #     print("\n--- Test Run Failed (Solar Query) ---")

    # Test Case 2: Designed to test retry for Python_REPL
    test_query_python_repl_retry = "Use the Python REPL tool to obtain the literal string 'Python is fun!' (including the single quotes). Then, write this exact string to a file named 'python_quote.txt'."
    print(f"\n--- Running LangGraph Test (PCEE Retry Test) for: '{test_query_python_repl_retry}' ---")
    final_state_repl = await run_graph_example(test_query_python_repl_retry, task_id_override="test_task_repl_retry_002")
    if final_state_repl:
        print(f"\n--- Test Run Complete (PCEE Retry Test) ---")
        print(f"Overall Evaluation Success: {final_state_repl.get('overall_evaluation_success')}")
        print(f"Overall Evaluation Assessment: {final_state_repl.get('overall_evaluation_assessment')}")
        print(f"Final number of messages: {len(final_state_repl.get('messages', []))}")
        # Specific check for retry count for the first step if it's available in the final plan_steps detail (if we add such logging)
        # For now, we rely on the logs during execution to show retry_count_for_current_step incrementing.
    else:
        print("\n--- Test Run Failed (Python REPL Retry Query) ---")


if __name__ == "__main__":
    # This allows running the test directly
    # Ensure your .env file is set up with necessary API keys (e.g., GOOGLE_API_KEY)
    asyncio.run(run_main_test())

