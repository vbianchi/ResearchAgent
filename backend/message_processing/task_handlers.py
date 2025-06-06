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

# <<< START MODIFICATION: Remove faulty import and define constants locally >>>
# Constants for message types stored in the DB, previously imported
DB_MSG_TYPE_SUB_STATUS = "agent_sub_status"
DB_MSG_TYPE_THOUGHT = "agent_thought"
DB_MSG_TYPE_TOOL_RESULT_FOR_CHAT = "tool_result_for_chat"
DB_MSG_TYPE_MAJOR_STEP = "db_major_step_announcement"

# Constants for UI subtype rendering, previously imported
# Now using string literals directly in the code where needed.
# SUB_TYPE_SUB_STATUS = "sub_status"
# SUB_TYPE_THOUGHT = "thought"
# <<< END MODIFICATION >>>


logger = logging.getLogger(__name__)

SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
DBAddTaskFunc = Callable[[str, str, str], Coroutine[Any, Any, None]]
DBGetMessagesFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, Any]]]]
DBDeleteTaskFunc = Callable[[str], Coroutine[Any, Any, bool]]
DBRenameTaskFunc = Callable[[str, str], Coroutine[Any, Any, bool]]
GetArtifactsFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, str]]]]


INTERNAL_DB_MESSAGE_TYPES_FOR_MONITOR_REPLAY_ONLY = {
    "SYSTEM_INTENT_CLASSIFIED",
    "agent_executor_step_finish",
    "artifact_generated",
    "llm_token_usage",
    "monitor_log",
    "monitor_stdout",
    "monitor_stderr",
    "monitor_direct_cmd_start",
    "monitor_direct_cmd_end",
    "system_connect",
    "system_context_switch",
    "system_new_task_signal",
    "system_task_deleted",
    "system_task_renamed",
    "system_llm_set",
    "system_plan_generated",
    "system_plan_confirmed",
    "system_plan_cancelled",
    "system_direct_qa",
    "system_direct_qa_finish",
    "system_direct_tool_request",
    "system_plan_step_start",
    "system_plan_step_end",
    "system_plan_end",
    "error_json",
    "error_unknown_msg",
    "error_processing",
    "monitor_user_input",
}


async def process_context_switch(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc,
    db_add_task_func: DBAddTaskFunc,
    db_get_messages_func: DBGetMessagesFunc,
    get_artifacts_func: GetArtifactsFunc,
    db_add_message_func: Optional[DBAddMessageFunc] = None
) -> None:
    task_id_from_frontend = data.get("taskId")
    task_title_from_frontend = data.get("taskTitle", data.get("task"))

    logger.info(f"[{session_id}] Switching context to Task ID: {task_id_from_frontend}")

    session_data_entry['cancellation_requested'] = False
    session_data_entry['current_plan_structured'] = None
    session_data_entry['current_plan_human_summary'] = None
    session_data_entry['current_plan_step_index'] = -1
    session_data_entry['plan_execution_active'] = False
    session_data_entry['original_user_query'] = None
    session_data_entry['active_plan_filename'] = None
    session_data_entry['current_plan_proposal_id_backend'] = None

    existing_agent_task = connected_clients_entry.get("agent_task")
    if existing_agent_task and not existing_agent_task.done():
        logger.warning(f"[{session_id}] Cancelling active agent/plan task due to context switch.")
        existing_agent_task.cancel()
        await send_ws_message_func("status_message", {"text": "Operation cancelled due to task switch.", "component_hint": "SYSTEM"})
        connected_clients_entry["agent_task"] = None

    session_data_entry["current_task_id"] = task_id_from_frontend
    if "callback_handler" in session_data_entry and hasattr(session_data_entry["callback_handler"], 'set_task_id'):
        session_data_entry["callback_handler"].set_task_id(task_id_from_frontend)

    await db_add_task_func(task_id_from_frontend, task_title_from_frontend or f"Task {task_id_from_frontend}", datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        _ = get_task_workspace_path(task_id_from_frontend, create_if_not_exists=True)
        logger.info(f"[{session_id}] Ensured workspace directory exists for task: {task_id_from_frontend}")
    except (ValueError, OSError) as ws_path_e:
        logger.error(f"[{session_id}] Failed to get/create workspace path for task {task_id_from_frontend}: {ws_path_e}")

    await send_ws_message_func("monitor_log", {
        "text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Switched context to task ID: {task_id_from_frontend} ('{task_title_from_frontend}')",
        "log_source": "SYSTEM_CONTEXT_SWITCH"
    })

    if "memory" in session_data_entry and hasattr(session_data_entry["memory"], 'clear'):
        try:
            session_data_entry["memory"].clear()
            logger.info(f"[{session_id}] Cleared agent memory for new task context.")
        except Exception as mem_e:
            logger.error(f"[{session_id}] Failed to clear memory on context switch: {mem_e}")

    history_messages = await db_get_messages_func(task_id_from_frontend)
    chat_history_for_langchain_memory = []

    if history_messages:
        logger.info(f"[{session_id}] Loading {len(history_messages)} history messages for task {task_id_from_frontend}.")
        await send_ws_message_func("history_start", {"text": f"Loading {len(history_messages)} messages..."})

        for i, msg_hist in enumerate(history_messages):
            db_msg_type = msg_hist.get('message_type', 'unknown')
            db_content_hist_str = msg_hist.get('content', '')
            db_timestamp = msg_hist.get('timestamp', datetime.datetime.now().isoformat())

            ws_message_type_to_send = None
            ws_payload_to_send = None

            try:
                if db_msg_type == "user_input":
                    ws_message_type_to_send = "user"
                    ws_payload_to_send = {"content": db_content_hist_str}
                    chat_history_for_langchain_memory.append(HumanMessage(content=db_content_hist_str))

                elif db_msg_type in ["agent_message", "agent_final_assessment"]:
                    ws_message_type_to_send = "agent_message"
                    try:
                        parsed_content = json.loads(db_content_hist_str)
                        if isinstance(parsed_content, dict) and "content" in parsed_content:
                             ws_payload_to_send = parsed_content
                        else:
                            ws_payload_to_send = {"content": db_content_hist_str, "component_hint": "DEFAULT"}
                    except json.JSONDecodeError:
                        ws_payload_to_send = {"content": db_content_hist_str, "component_hint": "DEFAULT"}
                    chat_history_for_langchain_memory.append(AIMessage(content=ws_payload_to_send["content"]))

                elif db_msg_type == "confirmed_plan_log":
                    ws_message_type_to_send = "confirmed_plan_log"
                    ws_payload_to_send = {"content": db_content_hist_str}

                elif db_msg_type == DB_MSG_TYPE_MAJOR_STEP:
                    ws_message_type_to_send = "agent_major_step_announcement"
                    try:
                        ws_payload_to_send = json.loads(db_content_hist_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"[{session_id}] Error parsing DB_MSG_TYPE_MAJOR_STEP content: {e}. Data: {db_content_hist_str}")
                        continue

                elif db_msg_type == DB_MSG_TYPE_SUB_STATUS:
                    ws_message_type_to_send = "agent_thinking_update"
                    try:
                        parsed_db_content = json.loads(db_content_hist_str)
                        ws_payload_to_send = {
                            "message": parsed_db_content.get("message_text", "Sub-status message missing"),
                            "component_hint": parsed_db_content.get("component_hint", "SYSTEM"),
                            "sub_type": "sub_status", # <<< MODIFIED: Use string literal
                            "status_key": "HISTORICAL_SUB_STATUS"
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"[{session_id}] Error parsing DB_MSG_TYPE_SUB_STATUS content: {e}. Data: {db_content_hist_str}")
                        continue

                elif db_msg_type == DB_MSG_TYPE_THOUGHT:
                    ws_message_type_to_send = "agent_thinking_update"
                    try:
                        parsed_db_content = json.loads(db_content_hist_str)
                        ws_payload_to_send = {
                            "message": {
                                "label": parsed_db_content.get("thought_label", "Agent thought:"),
                                "content_markdown": parsed_db_content.get("thought_content_markdown", "Thought content missing")
                            },
                            "component_hint": parsed_db_content.get("component_hint", "SYSTEM"),
                            "sub_type": "thought", # <<< MODIFIED: Use string literal
                            "status_key": "HISTORICAL_THOUGHT"
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"[{session_id}] Error parsing DB_MSG_TYPE_THOUGHT content: {e}. Data: {db_content_hist_str}")
                        continue

                elif db_msg_type == DB_MSG_TYPE_TOOL_RESULT_FOR_CHAT:
                    ws_message_type_to_send = "tool_result_for_chat"
                    try:
                        ws_payload_to_send = json.loads(db_content_hist_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"[{session_id}] Error parsing DB_MSG_TYPE_TOOL_RESULT_FOR_CHAT content: {e}. Data: {db_content_hist_str}")
                        ws_payload_to_send = {"error": "Corrupted tool result in history.", "raw_content": db_content_hist_str[:200]}

                elif db_msg_type == "status_message":
                    ws_message_type_to_send = "status_message"
                    try:
                        parsed_status = json.loads(db_content_hist_str)
                        if isinstance(parsed_status, dict) and "text" in parsed_status:
                            ws_payload_to_send = parsed_status
                        else:
                            ws_payload_to_send = {"text": db_content_hist_str, "component_hint": "SYSTEM"}
                    except json.JSONDecodeError:
                        ws_payload_to_send = {"text": db_content_hist_str, "component_hint": "SYSTEM"}

                elif db_msg_type in INTERNAL_DB_MESSAGE_TYPES_FOR_MONITOR_REPLAY_ONLY or \
                     db_msg_type.startswith("error_llm_") or db_msg_type.startswith("error_tool_"):
                    log_prefix_hist = f"[{db_timestamp}][{session_id[:8]}]"
                    type_indicator_hist = f"[{db_msg_type.replace('monitor_', '').replace('error_', 'ERR_').replace('system_', 'SYS_').upper()}]"
                    monitor_log_text = f"{log_prefix_hist} [History]{type_indicator_hist} {db_content_hist_str}"
                    await send_ws_message_func("monitor_log", {"text": monitor_log_text, "log_source": db_msg_type})

                else:
                    logger.warning(f"[{session_id}] Unknown history message type '{db_msg_type}' encountered during history load. Content: {db_content_hist_str[:100]}")
                    await send_ws_message_func("monitor_log", {
                        "text": f"[{db_timestamp}][{session_id[:8]}] [History][UNHANDLED_TYPE: {db_msg_type}] {db_content_hist_str}",
                        "log_source": "UNHANDLED_HISTORY_TYPE"
                    })

                if ws_message_type_to_send and ws_payload_to_send:
                    await send_ws_message_func(ws_message_type_to_send, ws_payload_to_send)
                    await asyncio.sleep(0.005)

            except Exception as e:
                logger.error(f"[{session_id}] Error processing history message (Type: {db_msg_type}): {e}. Data: {db_content_hist_str[:100]}", exc_info=True)
                continue

        await send_ws_message_func("history_end", {"text": "History loaded."})
        logger.info(f"[{session_id}] Finished sending {len(history_messages)} history messages.")

        MAX_MEMORY_RELOAD = settings.agent_memory_window_k
        if "memory" in session_data_entry and hasattr(session_data_entry["memory"], 'chat_memory'):
            try:
                relevant_memory_messages = [m for m in chat_history_for_langchain_memory if isinstance(m, (HumanMessage, AIMessage))]
                session_data_entry["memory"].chat_memory.messages = relevant_memory_messages[-MAX_MEMORY_RELOAD:]
                logger.info(f"[{session_id}] Repopulated agent memory with last {len(session_data_entry['memory'].chat_memory.messages)} relevant messages.")
            except Exception as mem_load_e:
                logger.error(f"[{session_id}] Failed to repopulate memory from history: {mem_load_e}")
    else:
        await send_ws_message_func("history_end", {"text": "No history found."})
        logger.info(f"[{session_id}] No history found for task {task_id_from_frontend}.")

    logger.info(f"[{session_id}] Getting current artifacts from filesystem for task {task_id_from_frontend}...")
    current_artifacts = await get_artifacts_func(task_id_from_frontend)
    await send_ws_message_func("update_artifacts", current_artifacts)
    logger.info(f"[{session_id}] Sent current artifact list ({len(current_artifacts)} items) for task {task_id_from_frontend}.")


async def process_new_task(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc,
    get_artifacts_func: GetArtifactsFunc
) -> None:
    # ... (function content is unchanged) ...
    logger.info(f"[{session_id}] Received 'new_task' signal. Clearing context for UI.")
    session_data_entry['cancellation_requested'] = False
    session_data_entry['current_plan_structured'] = None
    session_data_entry['current_plan_human_summary'] = None
    session_data_entry['current_plan_step_index'] = -1
    session_data_entry['plan_execution_active'] = False
    session_data_entry['original_user_query'] = None
    session_data_entry['active_plan_filename'] = None
    session_data_entry['current_plan_proposal_id_backend'] = None
    existing_agent_task = connected_clients_entry.get("agent_task")
    if existing_agent_task and not existing_agent_task.done():
        logger.warning(f"[{session_id}] Cancelling active agent/plan task due to new task signal.")
        existing_agent_task.cancel()
        await send_ws_message_func("status_message", {"text": "Operation cancelled for new task.", "component_hint": "SYSTEM"})
        connected_clients_entry["agent_task"] = None
    session_data_entry["current_task_id"] = None
    if "callback_handler" in session_data_entry and hasattr(session_data_entry["callback_handler"], 'set_task_id'):
        session_data_entry["callback_handler"].set_task_id(None)
    if "memory" in session_data_entry and hasattr(session_data_entry["memory"], 'clear'):
        session_data_entry["memory"].clear()
    await send_ws_message_func("monitor_log", {
        "text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Cleared context for new task signal. Awaiting new task context switch from client.",
        "log_source": "SYSTEM_NEW_TASK_SIGNAL"
    })
    await send_ws_message_func("update_artifacts", [])


async def process_delete_task(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc,
    db_delete_task_func: DBDeleteTaskFunc,
    get_artifacts_func: GetArtifactsFunc
) -> None:
    # ... (function content is unchanged) ...
    task_id_to_delete = data.get("taskId")
    if not task_id_to_delete:
        logger.warning(f"[{session_id}] 'delete_task' message missing taskId.")
        await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_ERROR] 'delete_task' received without taskId.", "log_source": "SYSTEM_ERROR"})
        return
    logger.warning(f"[{session_id}] Received request to delete task: {task_id_to_delete}")
    await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Received request to delete task: {task_id_to_delete}", "log_source": "SYSTEM_DELETE_REQUEST"})
    deleted_from_db = await db_delete_task_func(task_id_to_delete)
    if deleted_from_db:
        await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Task {task_id_to_delete} DB entries deleted.", "log_source": "SYSTEM_DELETE_SUCCESS"})
        task_workspace_to_delete: Optional[Path] = None
        try:
            task_workspace_to_delete = get_task_workspace_path(task_id_to_delete, create_if_not_exists=False)
            if await asyncio.to_thread(task_workspace_to_delete.exists):
                if task_workspace_to_delete.resolve().is_relative_to(BASE_WORKSPACE_ROOT.resolve()) and \
                   BASE_WORKSPACE_ROOT.resolve() != task_workspace_to_delete.resolve():
                    logger.info(f"[{session_id}] Attempting to delete workspace directory: {task_workspace_to_delete}")
                    await asyncio.to_thread(shutil.rmtree, task_workspace_to_delete)
                    logger.info(f"[{session_id}] Successfully deleted workspace directory: {task_workspace_to_delete}")
                    await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Workspace directory deleted: {task_workspace_to_delete.name}", "log_source": "SYSTEM_DELETE_SUCCESS"})
                else:
                    logger.warning(f"[{session_id}] Workspace directory {task_workspace_to_delete} deletion skipped (security). Path not relative or is base.")
            else:
                logger.info(f"[{session_id}] Workspace directory not found for task {task_id_to_delete}, no deletion needed.")
        except Exception as ws_del_e:
            logger.error(f"[{session_id}] Error during workspace deletion for task {task_id_to_delete}: {ws_del_e}", exc_info=True)
            await send_ws_message_func("status_message", {"text": f"Task DB deleted, but workspace for {task_id_to_delete[:8]} failed to delete.", "component_hint": "SYSTEM", "isError": True})
        if session_data_entry.get("current_task_id") == task_id_to_delete:
            logger.info(f"[{session_id}] Active task {task_id_to_delete} was deleted. Clearing session context.")
            session_data_entry['cancellation_requested'] = False
            session_data_entry['current_task_id'] = None
            if "callback_handler" in session_data_entry: session_data_entry["callback_handler"].set_task_id(None)
            if "memory" in session_data_entry: session_data_entry["memory"].clear()
            await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Cleared context as active task was deleted.", "log_source": "SYSTEM_CONTEXT_CLEAR"})
            await send_ws_message_func("update_artifacts", [])
    else:
        await send_ws_message_func("status_message", {"text": f"Failed to delete task {task_id_to_delete[:8]} from database.", "component_hint": "SYSTEM", "isError": True})


async def process_rename_task(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc,
    db_rename_task_func: DBRenameTaskFunc
) -> None:
    # ... (function content is unchanged) ...
    task_id_to_rename = data.get("taskId")
    new_name = data.get("newName")
    if not task_id_to_rename or not new_name:
        logger.warning(f"[{session_id}] Received invalid rename_task message: {data}")
        await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_ERROR] Invalid rename request.", "log_source": "SYSTEM_ERROR"})
        return
    logger.info(f"[{session_id}] Received request to rename task {task_id_to_rename} to '{new_name}'.")
    renamed_in_db = await db_rename_task_func(task_id_to_rename, new_name)
    if renamed_in_db:
        logger.info(f"[{session_id}] Successfully renamed task {task_id_to_rename} in database.")
        await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_EVENT] Task {task_id_to_rename} renamed to '{new_name}' in DB.", "log_source": "SYSTEM_RENAME_SUCCESS"})
    else:
        logger.error(f"[{session_id}] Failed to rename task {task_id_to_rename} in database.")
        await send_ws_message_func("monitor_log", {"text": f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}][{session_id[:8]}] [SYSTEM_ERROR] Failed to rename task {task_id_to_rename} in DB.", "log_source": "DB_ERROR"})