# backend/message_processing/task_handlers.py
import logging
import datetime
import shutil
from pathlib import Path
import asyncio
import json

from typing import Dict, Any, Callable, Coroutine, Optional, List

from langchain_core.messages import AIMessage, HumanMessage

from backend.config import settings
from backend.tool_loader import get_task_workspace_path, BASE_WORKSPACE_ROOT

logger = logging.getLogger(__name__)

# Type definitions for callback functions
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
DBAddTaskFunc = Callable[[str, str, str], Coroutine[Any, Any, None]]
DBGetMessagesFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, Any]]]]
DBDeleteTaskFunc = Callable[[str], Coroutine[Any, Any, bool]]
DBRenameTaskFunc = Callable[[str, str], Coroutine[Any, Any, bool]]
GetArtifactsFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, str]]]]

# Define message types that are for monitor replay only
INTERNAL_DB_MESSAGE_TYPES_FOR_MONITOR_REPLAY_ONLY = {
    "SYSTEM_INTENT_CLASSIFIED", "agent_executor_step_finish", "artifact_generated",
    "llm_token_usage", "monitor_log", "monitor_stdout", "monitor_stderr",
    "monitor_direct_cmd_start", "monitor_direct_cmd_end", "system_connect",
    "system_context_switch", "system_new_task_signal", "system_task_deleted",
    "system_task_renamed", "system_llm_set", "system_plan_generated",
    "system_plan_confirmed", "system_plan_cancelled", "system_direct_qa",
    "system_direct_qa_finish", "system_direct_tool_request", "system_plan_step_start",
    "system_plan_step_end", "system_plan_end", "error_json", "error_unknown_msg",
    "error_processing", "monitor_user_input", "agent_thought", "tool_input", "tool_output",
    "error_llm", "error_tool"
}

async def process_context_switch(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    db_add_task_func: DBAddTaskFunc,
    db_get_messages_func: DBGetMessagesFunc,
    get_artifacts_func: GetArtifactsFunc,
    **kwargs
) -> None:
    task_id = data.get("taskId")
    task_title = data.get("taskTitle", "Untitled Task")

    logger.info(f"[{session_id}] Switching context to Task ID: {task_id}")
    session_data_entry['current_task_id'] = task_id
    if session_data_entry.get("callback_handler"):
        session_data_entry["callback_handler"].set_task_id(task_id)

    # Reset state for the new task
    session_data_entry['cancellation_requested'] = False
    if "memory" in session_data_entry: session_data_entry["memory"].clear()
    
    # Ensure task exists in DB and workspace exists
    await db_add_task_func(task_id, task_title, datetime.datetime.now(datetime.timezone.utc).isoformat())
    get_task_workspace_path(task_id, create_if_not_exists=True)
    
    # Process and send message history
    history_messages = await db_get_messages_func(task_id)
    chat_history_for_memory = []

    logger.critical("CRITICAL_DEBUG: About to send 'history_start'.")
    await send_ws_message_func("history_start", {"text": f"Loading {len(history_messages)} messages..."})
    logger.critical("CRITICAL_DEBUG: 'history_start' message has been sent.")

    if history_messages:
        for msg in history_messages:
            msg_type = msg.get('message_type')
            content = msg.get('content', '')
            
            # Populate LangChain memory
            if msg_type == "user_input":
                chat_history_for_memory.append(HumanMessage(content=content))
            elif msg_type == "agent_message":
                chat_history_for_memory.append(AIMessage(content=content))

            # Send relevant history to UI (simplified for brevity)
            if msg_type == "user_input":
                await send_ws_message_func("user", {"content": content})
            elif msg_type == "agent_message":
                await send_ws_message_func("agent_message", content)
            elif msg_type in INTERNAL_DB_MESSAGE_TYPES_FOR_MONITOR_REPLAY_ONLY:
                 await send_ws_message_func("monitor_log", {"text": f"[{msg.get('timestamp')}][History] {content}", "log_source": msg_type})

    # Repopulate memory
    if "memory" in session_data_entry:
        session_data_entry["memory"].chat_memory.messages = chat_history_for_memory
        logger.info(f"[{session_id}] Repopulated agent memory with {len(chat_history_for_memory)} messages.")

    logger.critical("CRITICAL_DEBUG: History processing loop finished. About to send 'history_end'.")
    await send_ws_message_func("history_end", {"text": "History loaded."})
    logger.critical("CRITICAL_DEBUG: 'history_end' message has been sent.")
    
    # Load and send artifacts
    artifacts = await get_artifacts_func(task_id)
    await send_ws_message_func("update_artifacts", artifacts)


async def process_new_task(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    db_add_task_func: DBAddTaskFunc,
    **kwargs
) -> None:
    # This handler's logic remains simple and is likely correct.
    logger.info(f"[{session_id}] Received 'new_task' signal. Clearing context for UI.")
    session_data_entry['current_task_id'] = None
    if "callback_handler" in session_data_entry:
        session_data_entry["callback_handler"].set_task_id(None)
    if "memory" in session_data_entry:
        session_data_entry["memory"].clear()
    await send_ws_message_func("update_artifacts", [])


async def process_delete_task(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    db_delete_task_func: DBDeleteTaskFunc,
    **kwargs
) -> None:
    # This handler is likely correct.
    task_id_to_delete = data.get("taskId")
    if not task_id_to_delete: return
    logger.warning(f"[{session_id}] Received request to delete task: {task_id_to_delete}")
    deleted_from_db = await db_delete_task_func(task_id_to_delete)
    if deleted_from_db:
        try:
            task_workspace_to_delete = get_task_workspace_path(task_id_to_delete, create_if_not_exists=False)
            if task_workspace_to_delete.exists():
                shutil.rmtree(task_workspace_to_delete)
        except Exception as e:
            logger.error(f"Error during workspace deletion for task {task_id_to_delete}: {e}", exc_info=True)
    if session_data_entry.get("current_task_id") == task_id_to_delete:
        session_data_entry['current_task_id'] = None # Clear active task
        await send_ws_message_func("update_artifacts", [])

async def process_rename_task(
    session_id: str, data: Dict[str, Any],
    db_rename_task_func: DBRenameTaskFunc,
    **kwargs
) -> None:
    # This handler is likely correct.
    task_id_to_rename = data.get("taskId")
    new_name = data.get("newName")
    if not task_id_to_rename or not new_name: return
    await db_rename_task_func(task_id_to_rename, new_name)

