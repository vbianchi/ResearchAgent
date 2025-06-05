# backend/message_processing/agent_flow_handlers.py
import logging
import json
import datetime
from typing import Dict, Any, Callable, Coroutine, Optional, List
import asyncio
from pathlib import Path
import aiofiles
import re

# LangChain Imports
from langchain_core.messages import AIMessage, HumanMessage
# BaseChatModel and BaseTool might not be directly used here but are good for context
from langchain_core.language_models.chat_models import BaseChatModel 
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig

# Pydantic v2 import
from pydantic import BaseModel 

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm # Used by planner indirectly
from backend.tools import get_dynamic_tools, get_task_workspace_path
from backend.planner import generate_plan, PlanStep # PlanStep is the Pydantic v2 model
from backend.callbacks import AgentCancelledException # WebSocketCallbackHandler is in server.py
from backend.intent_classifier import classify_intent, IntentClassificationOutput 
from backend.langgraph_agent import research_agent_graph, ResearchAgentState # research_agent_graph is the compiled app

logger = logging.getLogger(__name__)

# Type Hints for Passed-in Functions from server.py
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]] # (text, log_source)
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]] # (task_id, session_id, msg_type, content)


async def _update_plan_file_step_status(
    task_workspace_path: Path,
    plan_filename: str,
    step_number: int,
    status_char: str # 'x' for failed, '!' for cancelled, ' ' for pending (or revert)
) -> None:
    """Helper to update the status checkbox of a step in the plan markdown file."""
    if not plan_filename:
        logger.warning(f"Cannot update plan file in {task_workspace_path.name}: no active plan filename provided to _update_plan_file_step_status.")
        return

    plan_file_path = task_workspace_path / plan_filename
    if not await asyncio.to_thread(plan_file_path.exists):
        logger.warning(f"Plan file {plan_file_path.name} not found in workspace {task_workspace_path.name} for updating step {step_number}.")
        return

    try:
        async with aiofiles.open(plan_file_path, 'r+', encoding='utf-8') as f:
            lines = await f.readlines()
            updated_lines = []
            found_step = False
            # Regex to match: optional leading space, '-', optional space, '[', any char or space, ']', optional space, number, '.', space, rest
            # This is made more flexible to match various states of the checkbox.
            step_pattern = re.compile(rf"^\s*-\s*\[\s*.\s*\]\s*{re.escape(str(step_number))}\.\s+.*", re.IGNORECASE)
            # Regex to replace the char inside the checkbox:
            checkbox_pattern = re.compile(r"(\s*-\s*\[)\s*.\s*(\])")

            for line_content in lines:
                if not found_step and step_pattern.match(line_content):
                    # Replace only the character inside the brackets
                    updated_line = checkbox_pattern.sub(rf"\g<1>{status_char}\g<2>", line_content, count=1)
                    updated_lines.append(updated_line)
                    found_step = True
                    logger.info(f"Updated plan file '{plan_filename}' for step {step_number} to status '[{status_char}]'. Line: {updated_line.strip()}")
                else:
                    updated_lines.append(line_content)
            
            if found_step:
                await f.seek(0) # Go to the beginning of the file
                await f.writelines(updated_lines)
                await f.truncate() # Remove any trailing content if new content is shorter
            else:
                logger.warning(f"Step {step_number} pattern not found in plan file '{plan_filename}' for status update. Regex was: {step_pattern.pattern}")

    except Exception as e:
        logger.error(f"Error updating plan file '{plan_filename}' for step {step_number}: {e}", exc_info=True)


async def process_user_message(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc, db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any 
) -> None:
    """Handles incoming user messages, classifies intent, and initiates agent actions."""
    user_input_content = ""
    content_payload = data.get("content")
    if isinstance(content_payload, str):
        user_input_content = content_payload
    elif isinstance(content_payload, dict) and 'content' in content_payload and isinstance(content_payload['content'], str):
        user_input_content = content_payload['content']
    else:
        logger.warning(f"[{session_id}] Received non-string or unexpected content for user_message: {type(content_payload)}. Ignoring.")
        await send_ws_message_func("status_message", {"text": "Error: Invalid message content received.", "isError": True})
        return

    active_task_id = session_data_entry.get("current_task_id")
    if not active_task_id:
        logger.warning(f"[{session_id}] User message received but no task active.")
        await send_ws_message_func("status_message", "Please select or create a task first.")
        return

    if (connected_clients_entry.get("agent_task") and not connected_clients_entry["agent_task"].done()) or \
       session_data_entry.get("plan_execution_active"):
        logger.warning(f"[{session_id}] User message received while agent/plan is already running for task {active_task_id}.")
        await send_ws_message_func("status_message", "Agent is busy. Please wait or stop the current process.")
        return

    await db_add_message_func(active_task_id, session_id, "user_input", user_input_content)
    await add_monitor_log_func(f"User Input: {user_input_content}", "monitor_user_input")

    session_data_entry['original_user_query'] = user_input_content
    session_data_entry['cancellation_requested'] = False
    session_data_entry['active_plan_filename'] = None 

    await send_ws_message_func("agent_thinking_update", {"status": "Classifying intent..."})

    try:
        dynamic_tools = get_dynamic_tools(active_task_id)
        tools_summary_for_intent = "\n".join(
            [f"- {tool.name}: {tool.description.split('.')[0]}" for tool in dynamic_tools]
        )
        if not tools_summary_for_intent:
             tools_summary_for_intent = "No tools seem to be available for this task."
    except Exception as e:
        logger.error(f"[{session_id}] Failed to load dynamic tools for intent classification: {e}", exc_info=True)
        await send_ws_message_func("status_message", {"text":"Error: Could not load agent tools. Cannot proceed.", "isError": True})
        await send_ws_message_func("agent_thinking_update", {"status": "Error."})
        return

    intent_classification_result: IntentClassificationOutput = await classify_intent(
        user_query=user_input_content, 
        session_data_entry=session_data_entry, 
        available_tools_summary=tools_summary_for_intent
    )
    
    classified_intent_value = intent_classification_result.intent.upper() # Ensure uppercase
    identified_tool_name = intent_classification_result.identified_tool_name
    extracted_tool_input = intent_classification_result.extracted_tool_input

    await add_monitor_log_func(
        f"Intent classified as: {classified_intent_value}. "
        f"Tool: {identified_tool_name or 'N/A'}. Input: {str(extracted_tool_input)[:50] if extracted_tool_input else 'N/A'}. "
        f"Reason: {intent_classification_result.reasoning or 'N/A'}",
        "system_intent_classified"
    )

    initial_llm_ids_for_state = {
        "intent_classifier_llm_id": session_data_entry.get("session_intent_classifier_llm_id"),
        "planner_llm_id": session_data_entry.get("session_planner_llm_id"),
        "controller_llm_id": session_data_entry.get("session_controller_llm_id"),
        "executor_llm_id": f"{session_data_entry.get('selected_llm_provider', settings.executor_default_provider)}::{session_data_entry.get('selected_llm_model_name', settings.executor_default_model_name)}",
        "evaluator_llm_id": session_data_entry.get("session_evaluator_llm_id"),
    }
    filtered_initial_llm_ids = {k: v for k,v in initial_llm_ids_for_state.items() if v is not None}

    initial_graph_input_dict: ResearchAgentState = {
        "user_query": user_input_content,
        "classified_intent": classified_intent_value,
        "identified_tool_name": identified_tool_name, 
        "extracted_tool_input": extracted_tool_input, 
        "current_task_id": active_task_id,
        "chat_history": session_data_entry["memory"].chat_memory.messages, 
        "plan_steps": [], 
        "current_step_index": 0,
        "retry_count_for_current_step": 0,
        "accumulated_plan_summary": "",
        **filtered_initial_llm_ids 
    }

    session_data_entry["callback_handler"].set_task_id(active_task_id)
    configurable_fields_for_run = {
        "task_id": active_task_id, "session_id": session_id, **filtered_initial_llm_ids 
    }
    runnable_config = RunnableConfig(
        callbacks=[session_data_entry["callback_handler"]],
        configurable=configurable_fields_for_run
    )

    if classified_intent_value == "PLAN":
        await send_ws_message_func("agent_thinking_update", {"status": "Generating plan..."})
        # generate_plan expects session_data_entry for LLM overrides for the planner
        human_plan_summary, structured_plan_steps_models = await generate_plan(
            user_query=user_input_content,
            available_tools_summary=tools_summary_for_intent,
            session_data_entry=session_data_entry 
        )

        if human_plan_summary and structured_plan_steps_models:
            session_data_entry["current_plan_human_summary"] = human_plan_summary
            session_data_entry["current_plan_structured"] = structured_plan_steps_models
            
            # Convert PlanStep Pydantic v2 models to dicts for sending via JSON
            plan_steps_for_ui = [step.model_dump() if isinstance(step, BaseModel) else step for step in structured_plan_steps_models]

            await send_ws_message_func("display_plan_for_confirmation", {
                "human_summary": human_plan_summary,
                "structured_plan": plan_steps_for_ui 
            })
            await add_monitor_log_func(f"Plan generated. Summary: {human_plan_summary}. Steps: {len(structured_plan_steps_models)}. Awaiting user confirmation.", "system_plan_generated")
            await send_ws_message_func("status_message", "Plan generated. Please review and confirm.")
            await send_ws_message_func("agent_thinking_update", {"status": "Awaiting plan confirmation..."})
        else: # Plan generation failed
            logger.error(f"[{session_id}] Failed to generate a plan for query: {user_input_content}")
            await add_monitor_log_func(f"Error: Failed to generate a plan.", "error_system")
            await send_ws_message_func("status_message", {"text":"Error: Could not generate a plan for your request.", "isError": True})
            await send_ws_message_func("agent_message", "I'm sorry, I couldn't create a plan for that request. Please try rephrasing or breaking it down.")
            await send_ws_message_func("agent_thinking_update", {"status": "Planning failed."})
    
    elif classified_intent_value in ["DIRECT_QA", "DIRECT_TOOL_REQUEST"]:
        log_msg = f"Handling as {classified_intent_value} with LangGraph."
        if classified_intent_value == "DIRECT_TOOL_REQUEST":
            log_msg += f" Tool: {identified_tool_name}, Input: {str(extracted_tool_input)[:50]}..."
        
        await send_ws_message_func("agent_thinking_update", {"status": f"Processing as {classified_intent_value.replace('_', ' ')}..."})
        await add_monitor_log_func(log_msg, f"system_{classified_intent_value.lower()}")

        logger.info(f"[{session_id}] Invoking research_agent_graph.astream_events for {classified_intent_value}. Input State (partial): user_query='{user_input_content}'. Config: {runnable_config.get('configurable')}")

        async def graph_streaming_task():
            try:
                async for event in research_agent_lg_graph.astream_events(
                    initial_graph_input_dict, config=runnable_config, version="v1" 
                ):
                    event_node_name = event.get("name", "graph") 
                    logger.debug(f"[{session_id}] Graph Event: {event['event']} for Node: {event_node_name}, Tags: {event.get('tags')}")
                    if event["event"] == "on_chain_end" and event_node_name == "ResearchAgentGraph": 
                        logger.info(f"[{session_id}] LangGraph stream finished for {classified_intent_value}.")
            except AgentCancelledException:
                logger.warning(f"[{session_id}] {classified_intent_value} execution cancelled by user (LangGraph).")
                await send_ws_message_func("status_message", f"{classified_intent_value.replace('_', ' ')} cancelled.")
                await add_monitor_log_func(f"{classified_intent_value.replace('_', ' ')} cancelled by user (LangGraph).", "system_cancel")
            except Exception as e:
                logger.error(f"[{session_id}] Error during {classified_intent_value} execution (LangGraph): {e}", exc_info=True)
                await add_monitor_log_func(f"Error during {classified_intent_value} (LangGraph): {e}", f"error_{classified_intent_value.lower()}")
                await send_ws_message_func("agent_message", f"Sorry, I encountered an error trying to process your request: {e}")
                await send_ws_message_func("status_message", {"text": f"Error during {classified_intent_value.replace('_', ' ')} processing.", "isError": True})
            finally:
                connected_clients_entry["agent_task"] = None
                session_data_entry["plan_execution_active"] = False 
                await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

        current_graph_task = asyncio.create_task(graph_streaming_task())
        connected_clients_entry["agent_task"] = current_graph_task
    else: # Should not happen with new validation in classify_intent
        logger.error(f"[{session_id}] Fallback: classify_intent returned '{classified_intent_value}', which is not 'PLAN', 'DIRECT_QA', or 'DIRECT_TOOL_REQUEST'. This should not happen with new validation.")
        await add_monitor_log_func(f"Error: classify_intent returned unknown value '{classified_intent_value}'. Defaulting to PLAN.", "error_system")
        # Fallback to PLAN logic (defensive)
        await send_ws_message_func("agent_thinking_update", {"status": "Generating plan (fallback)..."})
        human_plan_summary, structured_plan_steps_models = await generate_plan(
            user_query=user_input_content, 
            available_tools_summary=tools_summary_for_intent,
            session_data_entry=session_data_entry 
        )
        if human_plan_summary and structured_plan_steps_models:
            session_data_entry["current_plan_human_summary"] = human_plan_summary
            session_data_entry["current_plan_structured"] = structured_plan_steps_models
            plan_steps_for_ui = [step.model_dump() if isinstance(step, BaseModel) else step for step in structured_plan_steps_models]
            await send_ws_message_func("display_plan_for_confirmation", {"human_summary": human_plan_summary, "structured_plan": plan_steps_for_ui})
            await add_monitor_log_func(f"Plan generated (fallback). Summary: {human_plan_summary}. Steps: {len(structured_plan_steps_models)}. Awaiting user confirmation.", "system_plan_generated")
            await send_ws_message_func("status_message", "Plan generated. Please review and confirm.")
            await send_ws_message_func("agent_thinking_update", {"status": "Awaiting plan confirmation..."})
        else: # Fallback plan generation failed
            logger.error(f"[{session_id}] Failed to generate a plan (fallback) for query: {user_input_content}")
            await add_monitor_log_func(f"Error: Failed to generate a plan (fallback).", "error_system")
            await send_ws_message_func("status_message", {"text":"Error: Could not generate a plan for your request (fallback).", "isError": True})
            await send_ws_message_func("agent_message", "I'm sorry, I couldn't create a plan for that request. Please try rephrasing or breaking it down.")
            await send_ws_message_func("agent_thinking_update", {"status": "Planning failed."})


async def process_execute_confirmed_plan(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc, db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any 
) -> None:
    """Handles the execution of a user-confirmed plan using LangGraph."""
    logger.info(f"[{session_id}] Received 'execute_confirmed_plan'.")
    active_task_id = session_data_entry.get("current_task_id")
    original_user_query_for_plan = session_data_entry.get('original_user_query')

    if not active_task_id:
        logger.warning(f"[{session_id}] 'execute_confirmed_plan' received but no active task.")
        await send_ws_message_func("status_message", "Error: No active task to execute plan for.")
        return
    if not original_user_query_for_plan:
        logger.error(f"[{session_id}] 'execute_confirmed_plan' but original_user_query is missing from session data.")
        await send_ws_message_func("status_message", "Error: Original query context missing for plan execution.")
        return

    # Plan steps could be dicts from UI or PlanStep models if directly from generate_plan
    confirmed_plan_steps = session_data_entry.get("current_plan_structured")
    if not confirmed_plan_steps or not isinstance(confirmed_plan_steps, list):
        confirmed_plan_steps = data.get("confirmed_plan") 
        if not confirmed_plan_steps or not isinstance(confirmed_plan_steps, list):
            logger.error(f"[{session_id}] Invalid or missing plan in 'execute_confirmed_plan'. Data: {data}")
            await send_ws_message_func("status_message", "Error: Invalid plan received for execution.")
            return
        session_data_entry["current_plan_structured"] = confirmed_plan_steps # Store if from data

    # Ensure plan_steps are dictionaries for the graph state and for plan file
    plan_steps_for_graph_and_file = [
        step.model_dump() if isinstance(step, BaseModel) else step 
        for step in confirmed_plan_steps
    ]

    session_data_entry["current_plan_step_index"] = 0 
    session_data_entry["plan_execution_active"] = True
    session_data_entry['cancellation_requested'] = False

    await add_monitor_log_func(f"User confirmed plan. Starting execution of {len(plan_steps_for_graph_and_file)} steps.", "system_plan_confirmed")
    await send_ws_message_func("status_message", "Plan confirmed. Executing steps with LangGraph...")
    await send_ws_message_func("agent_thinking_update", {"status": "Starting plan execution..."})

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plan_filename = f"_plan_{timestamp_str}.md"
    session_data_entry['active_plan_filename'] = plan_filename
    plan_markdown_content = [f"# Agent Plan for Task: {active_task_id}\n", f"## Plan ID: {timestamp_str}\n"]
    plan_markdown_content.append(f"## Original User Query:\n{original_user_query_for_plan}\n")
    plan_markdown_content.append(f"## Plan Summary (from Planner):\n{session_data_entry.get('current_plan_human_summary', 'N/A')}\n")
    plan_markdown_content.append("## Steps:\n")
    for i, step_data_dict in enumerate(plan_steps_for_graph_and_file):
        desc = step_data_dict.get('description', 'N/A')
        tool_sugg = step_data_dict.get('tool_to_use', 'None')
        input_instr = step_data_dict.get('tool_input_instructions', 'None')
        expected_out = step_data_dict.get('expected_outcome', 'N/A')
        plan_markdown_content.append(f"- [ ] {i+1}. **{desc}**")
        plan_markdown_content.append(f"    - Tool Suggestion (Planner): `{tool_sugg}`")
        plan_markdown_content.append(f"    - Input Instructions (Planner): `{input_instr}`")
        plan_markdown_content.append(f"    - Expected Outcome (Planner): `{expected_out}`\n")
    
    task_workspace_path = get_task_workspace_path(active_task_id)
    try:
        plan_file_path = task_workspace_path / plan_filename
        async with aiofiles.open(plan_file_path, 'w', encoding='utf-8') as f:
            await f.write("\n".join(plan_markdown_content))
        logger.info(f"[{session_id}] Saved confirmed plan to {plan_file_path}")
        await add_monitor_log_func(f"Confirmed plan saved to artifact: {plan_filename}", "system_info")
        await send_ws_message_func("trigger_artifact_refresh", {"taskId": active_task_id})
    except Exception as e:
        logger.error(f"[{session_id}] Failed to save plan to file '{plan_filename}': {e}", exc_info=True)
        await add_monitor_log_func(f"Error saving plan to file '{plan_filename}': {e}", "error_system")

    initial_llm_ids_for_state = {
        "intent_classifier_llm_id": session_data_entry.get("session_intent_classifier_llm_id"),
        "planner_llm_id": session_data_entry.get("session_planner_llm_id"),
        "controller_llm_id": session_data_entry.get("session_controller_llm_id"),
        "executor_llm_id": f"{session_data_entry.get('selected_llm_provider', settings.executor_default_provider)}::{session_data_entry.get('selected_llm_model_name', settings.executor_default_model_name)}",
        "evaluator_llm_id": session_data_entry.get("session_evaluator_llm_id"),
    }
    filtered_initial_llm_ids = {k: v for k,v in initial_llm_ids_for_state.items() if v is not None}

    initial_graph_input_for_plan: ResearchAgentState = {
        "user_query": original_user_query_for_plan,
        "classified_intent": "PLAN", 
        "plan_steps": plan_steps_for_graph_and_file, 
        "current_task_id": active_task_id,
        "chat_history": session_data_entry["memory"].chat_memory.messages,
        "current_step_index": 0, 
        "retry_count_for_current_step": 0,
        "accumulated_plan_summary": f"Executing Plan for query: '{original_user_query_for_plan}'\n", 
        "is_direct_qa_flow": False, 
        **filtered_initial_llm_ids
    }

    session_data_entry["callback_handler"].set_task_id(active_task_id)
    configurable_fields_for_run = {
        "task_id": active_task_id, "session_id": session_id, **filtered_initial_llm_ids
    }
    runnable_config = RunnableConfig(
        callbacks=[session_data_entry["callback_handler"]],
        configurable=configurable_fields_for_run
    )

    logger.info(f"[{session_id}] Invoking research_agent_graph.astream_events for PLAN execution. Initial State (partial): Plan steps count = {len(plan_steps_for_graph_and_file)}. Config: {runnable_config.get('configurable')}")

    async def graph_streaming_task_plan():
        try:
            async for event in research_agent_lg_graph.astream_events(
                initial_graph_input_for_plan, config=runnable_config, version="v1" 
            ):
                event_node_name = event.get("name", "graph")
                logger.debug(f"[{session_id}] Plan Graph Event: {event['event']} for Node: {event_node_name}, Tags: {event.get('tags')}")
                if event["event"] == "on_chain_end" and event_node_name == "ResearchAgentGraph":
                    logger.info(f"[{session_id}] LangGraph stream finished for PLAN execution.")
        
        except AgentCancelledException:
            logger.warning(f"[{session_id}] PLAN execution cancelled by user (LangGraph).")
            await send_ws_message_func("status_message", "Plan execution cancelled.")
            await add_monitor_log_func("Plan execution cancelled by user (LangGraph).", "system_cancel")
            if session_data_entry.get('active_plan_filename'):
                await _update_plan_file_step_status(task_workspace_path, session_data_entry['active_plan_filename'], session_data_entry.get('current_plan_step_index', 0) + 1, '!') 
        except Exception as e:
            logger.error(f"[{session_id}] Error during PLAN execution (LangGraph): {e}", exc_info=True)
            await add_monitor_log_func(f"Error during PLAN execution (LangGraph): {e}", "error_plan_execution")
            await send_ws_message_func("agent_message", f"Sorry, I encountered an error during plan execution: {e}")
            await send_ws_message_func("status_message", {"text": "Error during plan execution.", "isError": True})
            if session_data_entry.get('active_plan_filename'):
                 await _update_plan_file_step_status(task_workspace_path, session_data_entry['active_plan_filename'], session_data_entry.get('current_plan_step_index', 0) + 1, 'x') 
        finally:
            connected_clients_entry["agent_task"] = None
            session_data_entry["plan_execution_active"] = False
            await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

    current_graph_task_plan = asyncio.create_task(graph_streaming_task_plan())
    connected_clients_entry["agent_task"] = current_graph_task_plan


async def process_cancel_plan_proposal(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc
) -> None:
    logger.info(f"[{session_id}] Received 'cancel_plan_proposal'.")
    plan_id_to_cancel = data.get("plan_id") 

    session_data_entry["current_plan_human_summary"] = None
    session_data_entry["current_plan_structured"] = None
    session_data_entry["current_plan_step_index"] = -1 
    session_data_entry["plan_execution_active"] = False 

    await add_monitor_log_func(f"User cancelled plan proposal (ID: {plan_id_to_cancel or 'N/A'}).", "system_plan_cancelled")
    await send_ws_message_func("status_message", "Plan proposal cancelled by user.")
    await send_ws_message_func("agent_message", "Okay, the proposed plan has been cancelled.")
    await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

