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
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools, get_task_workspace_path
from backend.planner import generate_plan, PlanStep # PlanStep might still be used by generate_plan
from backend.callbacks import AgentCancelledException
from backend.intent_classifier import classify_intent
from backend.langgraph_agent import research_agent_graph, ResearchAgentState

logger = logging.getLogger(__name__)

# Type Hints for Passed-in Functions
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]


async def _update_plan_file_step_status(
    task_workspace_path: Path,
    plan_filename: str,
    step_number: int,
    status_char: str
) -> None:
    """Helper to update a step's status in the plan Markdown file."""
    if not plan_filename:
        logger.warning("Cannot update plan file: no active plan filename.")
        return

    plan_file_path = task_workspace_path / plan_filename
    if not await asyncio.to_thread(plan_file_path.exists):
        logger.warning(f"Plan file {plan_file_path} not found for updating step {step_number}.")
        return

    try:
        async with aiofiles.open(plan_file_path, 'r', encoding='utf-8') as f_read:
            lines = await f_read.readlines()

        updated_lines = []
        found_step = False
        step_pattern = re.compile(rf"^\s*-\s*\[\s*[ x!-]?\s*\]\s*{re.escape(str(step_number))}\.\s+.*", re.IGNORECASE)
        checkbox_pattern = re.compile(r"(\s*-\s*\[)\s*[ x!-]?\s*(\])")

        for line_no, line_content in enumerate(lines):
            if not found_step and step_pattern.match(line_content):
                updated_line = checkbox_pattern.sub(rf"\g<1>{status_char}\g<2>", line_content, count=1)
                updated_lines.append(updated_line)
                found_step = True
                logger.info(f"Updated plan file for step {step_number} to status '[{status_char}]'. Line: {updated_line.strip()}")
            else:
                updated_lines.append(line_content)

        if found_step:
            async with aiofiles.open(plan_file_path, 'w', encoding='utf-8') as f_write:
                await f_write.writelines(updated_lines)
        else:
            logger.warning(f"Step {step_number} pattern not found in plan file {plan_file_path} for status update. Regex was: {step_pattern.pattern}")
            if len(lines) > 0:
                logger.debug("First few lines of plan file for debugging update failure:")
                for i, l in enumerate(lines[:5]):
                    logger.debug(f"  Line {i+1}: {l.strip()}")
    except Exception as e:
        logger.error(f"Error updating plan file {plan_file_path} for step {step_number}: {e}", exc_info=True)


async def process_user_message(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc, db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any
) -> None:
    user_input_content = ""
    content_payload = data.get("content")
    if isinstance(content_payload, str):
        user_input_content = content_payload
    elif isinstance(content_payload, dict) and 'content' in content_payload and isinstance(content_payload['content'], str):
        user_input_content = content_payload['content']
    else:
        logger.warning(f"[{session_id}] Received non-string or unexpected content for user_message: {type(content_payload)}. Ignoring.")
        return

    active_task_id = session_data_entry.get("current_task_id")
    if not active_task_id:
        logger.warning(f"[{session_id}] User message received but no task active.")
        await send_ws_message_func("status_message", "Please select or create a task first.")
        return

    if (connected_clients_entry.get("agent_task") or
        session_data_entry.get("plan_execution_active")):
        logger.warning(f"[{session_id}] User message received while agent/plan is already running for task {active_task_id}.")
        await send_ws_message_func("status_message", "Agent is busy. Please wait or stop the current process.")
        return

    await db_add_message_func(active_task_id, session_id, "user_input", user_input_content)
    await add_monitor_log_func(f"User Input: {user_input_content}", "monitor_user_input")

    session_data_entry['original_user_query'] = user_input_content
    session_data_entry['cancellation_requested'] = False
    session_data_entry['active_plan_filename'] = None

    await send_ws_message_func("agent_thinking_update", {"status": "Classifying intent..."})

    dynamic_tools = get_dynamic_tools(active_task_id)
    tools_summary_for_intent = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in dynamic_tools])

    classified_intent_value = await classify_intent(user_input_content, session_data_entry, tools_summary_for_intent)
    await add_monitor_log_func(f"Intent classified as: {classified_intent_value}", "system_intent_classified")


    if classified_intent_value == "PLAN":
        await send_ws_message_func("agent_thinking_update", {"status": "Generating plan..."})
        human_plan_summary, structured_plan_steps = await generate_plan(
            user_query=user_input_content,
            session_data_entry=session_data_entry,
            available_tools_summary=tools_summary_for_intent
        )
        if human_plan_summary and structured_plan_steps:
            session_data_entry["current_plan_human_summary"] = human_plan_summary
            session_data_entry["current_plan_structured"] = structured_plan_steps
            session_data_entry["current_plan_step_index"] = 0
            session_data_entry["plan_execution_active"] = False
            await send_ws_message_func("display_plan_for_confirmation", {
                "human_summary": human_plan_summary,
                "structured_plan": structured_plan_steps
            })
            await add_monitor_log_func(f"Plan generated. Summary: {human_plan_summary}. Steps: {len(structured_plan_steps)}. Awaiting user confirmation.", "system_plan_generated")
            await send_ws_message_func("status_message", "Plan generated. Please review and confirm.")
            await send_ws_message_func("agent_thinking_update", {"status": "Awaiting plan confirmation..."})
        else:
            logger.error(f"[{session_id}] Failed to generate a plan for query: {user_input_content}")
            await add_monitor_log_func(f"Error: Failed to generate a plan.", "error_system")
            await send_ws_message_func("status_message", "Error: Could not generate a plan for your request.")
            await send_ws_message_func("agent_message", "I'm sorry, I couldn't create a plan for that request. Please try rephrasing or breaking it down.")
            await send_ws_message_func("agent_thinking_update", {"status": "Planning failed."})

    elif classified_intent_value == "DIRECT_QA":
        await send_ws_message_func("agent_thinking_update", {"status": "Processing directly (LangGraph)..."})
        await add_monitor_log_func(f"Handling as DIRECT_QA with LangGraph.", "system_direct_qa")

        # MODIFIED: Removed .dict() as ResearchAgentState is a TypedDict and already a dictionary
        initial_graph_input_dict = ResearchAgentState(
            user_query=user_input_content,
            classified_intent="DIRECT_QA",
            current_task_id=active_task_id,
            chat_history=session_data_entry["memory"].chat_memory.messages,
            plan_steps=[],
            current_step_index=0,
            retry_count_for_current_step=0,
            accumulated_plan_summary=""
        )

        session_data_entry["callback_handler"].set_task_id(active_task_id)
        
        configurable_fields = {
            "task_id": active_task_id,
            "session_id": session_id,
            "intent_classifier_llm_id": session_data_entry.get("session_intent_classifier_llm_id"),
            "planner_llm_id": session_data_entry.get("session_planner_llm_id"),
            "controller_llm_id": session_data_entry.get("session_controller_llm_id"),
            "evaluator_llm_id": session_data_entry.get("session_evaluator_llm_id"),
            "executor_llm_id": f"{session_data_entry.get('selected_llm_provider', settings.executor_default_provider)}::{session_data_entry.get('selected_llm_model_name', settings.executor_default_model_name)}",
        }
        filtered_configurable_fields = {k: v for k, v in configurable_fields.items() if v is not None}

        runnable_config = RunnableConfig(
            callbacks=[session_data_entry["callback_handler"]],
            configurable=filtered_configurable_fields
        )

        logger.info(f"[{session_id}] Invoking research_agent_graph.astream_events for DIRECT_QA. Input: {initial_graph_input_dict}, Config: {filtered_configurable_fields}")

        async def graph_streaming_task_direct_qa():
            try:
                async for event in research_agent_lg_graph.astream_events(
                    initial_graph_input_dict, # Pass the dictionary directly
                    config=runnable_config,
                    version="v1"
                ):
                    event_name = event.get("name", "graph")
                    logger.debug(f"[{session_id}] Graph Event: {event['event']} for Node: {event_name}, Tags: {event.get('tags')}")
                    if event["event"] == "on_chain_end" and event.get("name") == "ResearchAgentGraph":
                        logger.info(f"[{session_id}] LangGraph stream finished for DIRECT_QA.")
            except AgentCancelledException:
                logger.warning(f"[{session_id}] DIRECT_QA execution cancelled by user (LangGraph).")
                await send_ws_message_func("status_message", "Direct QA cancelled.")
                await add_monitor_log_func("Direct QA cancelled by user (LangGraph).", "system_cancel")
            except Exception as e:
                logger.error(f"[{session_id}] Error during DIRECT_QA execution (LangGraph): {e}", exc_info=True)
                await add_monitor_log_func(f"Error during Direct QA (LangGraph): {e}", "error_direct_qa")
                await send_ws_message_func("agent_message", f"Sorry, I encountered an error trying to answer directly: {e}")
                await send_ws_message_func("status_message", "Error during direct processing.")
            finally:
                connected_clients_entry["agent_task"] = None
                await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

        current_graph_task = asyncio.create_task(graph_streaming_task_direct_qa())
        connected_clients_entry["agent_task"] = current_graph_task
    else: 
        logger.error(f"[{session_id}] Fallback: classify_intent returned '{classified_intent_value}', which is not 'PLAN' or 'DIRECT_QA'. Defaulting to planning.")
        await add_monitor_log_func(f"Error: classify_intent returned unknown value '{classified_intent_value}'. Defaulting to PLAN.", "error_system")
        await send_ws_message_func("agent_thinking_update", {"status": "Generating plan (fallback)..."})
        human_plan_summary, structured_plan_steps = await generate_plan(
            user_query=user_input_content, 
            session_data_entry=session_data_entry,
            available_tools_summary=tools_summary_for_intent
        )
        if human_plan_summary and structured_plan_steps:
            session_data_entry["current_plan_human_summary"] = human_plan_summary
            session_data_entry["current_plan_structured"] = structured_plan_steps
            session_data_entry["current_plan_step_index"] = 0
            session_data_entry["plan_execution_active"] = False
            await send_ws_message_func("display_plan_for_confirmation", {"human_summary": human_plan_summary, "structured_plan": structured_plan_steps})
            await add_monitor_log_func(f"Plan generated (fallback). Summary: {human_plan_summary}. Steps: {len(structured_plan_steps)}. Awaiting user confirmation.", "system_plan_generated")
            await send_ws_message_func("status_message", "Plan generated. Please review and confirm.")
            await send_ws_message_func("agent_thinking_update", {"status": "Awaiting plan confirmation..."})
        else:
            logger.error(f"[{session_id}] Failed to generate a plan (fallback) for query: {user_input_content}")
            await add_monitor_log_func(f"Error: Failed to generate a plan (fallback).", "error_system")
            await send_ws_message_func("status_message", "Error: Could not generate a plan for your request (fallback).")
            await send_ws_message_func("agent_message", "I'm sorry, I couldn't create a plan for that request. Please try rephrasing or breaking it down.")
            await send_ws_message_func("agent_thinking_update", {"status": "Planning failed."})


async def process_execute_confirmed_plan(
    session_id: str,
    data: Dict[str, Any],
    session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc,
    db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any
) -> None:
    logger.info(f"[{session_id}] Received 'execute_confirmed_plan'.")
    active_task_id = session_data_entry.get("current_task_id")
    if not active_task_id:
        logger.warning(f"[{session_id}] 'execute_confirmed_plan' received but no active task.")
        await send_ws_message_func("status_message", "Error: No active task to execute plan for.")
        return

    confirmed_plan_steps_dicts = data.get("confirmed_plan")
    if not confirmed_plan_steps_dicts or not isinstance(confirmed_plan_steps_dicts, list):
        logger.error(f"[{session_id}] Invalid or missing plan in 'execute_confirmed_plan' message. Data received: {data}")
        await send_ws_message_func("status_message", "Error: Invalid plan received for execution.")
        return

    session_data_entry["current_plan_structured"] = confirmed_plan_steps_dicts
    session_data_entry["current_plan_step_index"] = 0
    session_data_entry["plan_execution_active"] = True
    session_data_entry['cancellation_requested'] = False

    await add_monitor_log_func(f"User confirmed plan. Starting execution of {len(confirmed_plan_steps_dicts)} steps.", "system_plan_confirmed")
    await send_ws_message_func("status_message", "Plan confirmed. Executing steps...")

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plan_filename = f"_plan_{timestamp_str}.md"
    session_data_entry['active_plan_filename'] = plan_filename
    plan_markdown_content = [f"# Agent Plan for Task: {active_task_id}\n", f"## Plan ID: {timestamp_str}\n"]
    original_query_for_plan_file = session_data_entry.get('original_user_query', 'N/A')
    plan_markdown_content.append(f"## Original User Query:\n{original_query_for_plan_file}\n")
    plan_markdown_content.append(f"## Plan Summary (from Planner):\n{session_data_entry.get('current_plan_human_summary', 'N/A')}\n")
    plan_markdown_content.append("## Steps:\n")
    for i, step_data_dict in enumerate(confirmed_plan_steps_dicts):
        desc = step_data_dict.get('description', 'N/A') if isinstance(step_data_dict, dict) else 'N/A (Invalid Step Format)'
        tool_sugg = step_data_dict.get('tool_to_use', 'None') if isinstance(step_data_dict, dict) else 'N/A'
        input_instr = step_data_dict.get('tool_input_instructions', 'None') if isinstance(step_data_dict, dict) else 'N/A'
        expected_out = step_data_dict.get('expected_outcome', 'N/A') if isinstance(step_data_dict, dict) else 'N/A'
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

    await add_monitor_log_func(f"LangGraph plan execution for {len(confirmed_plan_steps_dicts)} steps is PENDING IMPLEMENTATION in 'process_execute_confirmed_plan'.", "warning_system")
    await send_ws_message_func("status_message", "Plan execution with LangGraph is under development.")
    await send_ws_message_func("agent_message", "The plan has been acknowledged. Full execution using the new graph system is coming soon.")
    await send_ws_message_func("agent_thinking_update", {"status": "Idle."})
    session_data_entry["plan_execution_active"] = False


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
