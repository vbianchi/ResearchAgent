# backend/message_processing/agent_flow_handlers.py
import logging
import json
import datetime
from typing import Dict, Any, Callable, Coroutine, Optional, List
import asyncio
from pathlib import Path
import aiofiles
import re
import uuid

# LangChain Imports
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools, get_task_workspace_path
from backend.planner import generate_plan
from backend.callbacks import AgentCancelledException
from backend.intent_classifier import classify_intent
from backend.langgraph_agent import research_agent_graph, ResearchAgentState

from backend.pydantic_models import PlanStep, IntentClassificationOutput 

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
            else:
                updated_lines.append(line_content)

        if found_step:
            async with aiofiles.open(plan_file_path, 'w', encoding='utf-8') as f_write:
                await f_write.writelines(updated_lines)
        else:
            logger.warning(f"Step {step_number} pattern not found in plan file {plan_file_path} for status update. Regex was: {step_pattern.pattern}")
    except Exception as e:
        logger.error(f"Error updating plan file {plan_file_path} for step {step_number}: {e}", exc_info=True)


async def process_user_message(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc, db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any
) -> None:
    # --- MODIFIED: Corrected data extraction logic ---
    user_input_content = data.get("content", "")
    # --- END MODIFICATION ---

    active_task_id = session_data_entry.get("current_task_id")

    if not active_task_id:
        await send_ws_message_func("status_message", "Please select or create a task first.")
        return

    if connected_clients_entry.get("agent_task") or session_data_entry.get("plan_execution_active"):
        await send_ws_message_func("status_message", "Agent is busy. Please wait or stop the current process.")
        return

    await db_add_message_func(active_task_id, session_id, "user_input", user_input_content)
    session_data_entry['original_user_query'] = user_input_content
    session_data_entry['cancellation_requested'] = False

    await send_ws_message_func("agent_thinking_update", {"status": "Classifying intent..."})

    dynamic_tools = get_dynamic_tools(active_task_id)
    tool_names_for_prompt = ", ".join([f"'{tool.name}'" for tool in dynamic_tools])

    runnable_config = RunnableConfig(
        callbacks=[session_data_entry["callback_handler"]],
        configurable={
            "task_id": active_task_id, 
            "session_id": session_id,
            "intent_classifier_llm_id": session_data_entry.get("session_intent_classifier_llm_id"),
            "planner_llm_id": session_data_entry.get("session_planner_llm_id"),
            "controller_llm_id": session_data_entry.get("session_controller_llm_id"),
            "evaluator_llm_id": session_data_entry.get("session_evaluator_llm_id"),
            "executor_llm_id": f"{session_data_entry.get('selected_llm_provider', settings.executor_default_provider)}::{session_data_entry.get('selected_llm_model_name', settings.executor_default_model_name)}",
        }
    )
    
    classification_output = await classify_intent(
        user_query=user_input_content,
        tool_names_for_prompt=tool_names_for_prompt,
        session_data_entry=session_data_entry,
        config=runnable_config
    )
    
    await add_monitor_log_func(f"Intent classified as: {classification_output.intent}. Reason: {classification_output.reasoning}", "system_intent_classified")

    initial_graph_input = ResearchAgentState(
        user_query=user_input_content,
        classified_intent=classification_output.intent,
        current_task_id=active_task_id,
        chat_history=session_data_entry["memory"].chat_memory.messages,
    )

    async def graph_streaming_task():
        try:
            async for event in research_agent_lg_graph.astream_events(initial_graph_input, config=runnable_config, version="v1"):
                pass
        except AgentCancelledException:
            logger.warning(f"[{session_id}] Execution cancelled by user (LangGraph).")
            await send_ws_message_func("status_message", "Operation cancelled.")
        except Exception as e:
            logger.error(f"[{session_id}] Error during graph execution: {e}", exc_info=True)
            await send_ws_message_func("agent_message", f"Sorry, an error occurred: {e}")
        finally:
            connected_clients_entry["agent_task"] = None
            session_data_entry["plan_execution_active"] = False
            await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

    current_graph_task = asyncio.create_task(graph_streaming_task())
    connected_clients_entry["agent_task"] = current_graph_task


async def process_execute_confirmed_plan(*args, **kwargs):
    logger.warning("process_execute_confirmed_plan is deprecated and should not be called.")
    pass

async def process_cancel_plan_proposal(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc, add_monitor_log_func: AddMonitorLogFunc, **kwargs
) -> None:
    logger.info(f"[{session_id}] User cancelled plan proposal.")
    await add_monitor_log_func("User cancelled plan proposal.", "system_plan_cancelled")
    await send_ws_message_func("status_message", "Plan proposal cancelled.")
    await send_ws_message_func("agent_message", "Okay, the proposed plan has been cancelled.")
    await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

