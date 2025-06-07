# backend/message_processing/operational_handlers.py
import logging
from typing import Dict, Any, Callable, Coroutine, Optional, List
import asyncio

from backend.callbacks import AgentCancelledException

logger = logging.getLogger(__name__)

# Type Hints for Passed-in Functions
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
GetArtifactsFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, Any]]]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
ExecuteShellCommandFunc = Callable[[str, str, SendWSMessageFunc, DBAddMessageFunc, Optional[str]], Coroutine[Any, Any, bool]]


async def process_cancel_agent(
    session_id: str,
    session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any],
    **kwargs
) -> None:
    """
    Handles a request to cancel the currently running agent task.
    """
    logger.warning(f"[{session_id}] Received request to cancel current operation.")
    session_data_entry['cancellation_requested'] = True
    
    agent_task_to_cancel = connected_clients_entry.get("agent_task")
    
    if agent_task_to_cancel and not agent_task_to_cancel.done():
        logger.info(f"[{session_id}] Found active agent task. Attempting cancellation...")
        agent_task_to_cancel.cancel()
        try:
            # Wait briefly for the cancellation to be processed by the running task
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass # This is expected
        logger.info(f"[{session_id}] Cancellation request sent to task.")
    else:
        logger.warning(f"[{session_id}] No active (or already finished) agent task found to cancel.")


async def process_get_artifacts_for_task(
    data: Dict[str, Any],
    session_data_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    get_artifacts_func: GetArtifactsFunc,
    **kwargs
) -> None:
    task_id_to_refresh = data.get("taskId")
    if task_id_to_refresh and task_id_to_refresh == session_data_entry.get("current_task_id"):
        artifacts = await get_artifacts_func(task_id_to_refresh)
        await send_ws_message_func("update_artifacts", artifacts)


async def process_run_command(
    session_id: str,
    data: Dict[str, Any],
    session_data_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    db_add_message_func: DBAddMessageFunc,
    execute_shell_command_func: ExecuteShellCommandFunc,
    **kwargs
) -> None:
    command_to_run = data.get("command")
    if command_to_run and isinstance(command_to_run, str):
        active_task_id_for_cmd = session_data_entry.get("current_task_id")
        await execute_shell_command_func(
            command_to_run, session_id, send_ws_message_func,
            db_add_message_func, active_task_id_for_cmd
        )

async def process_action_command(
    session_id: str, data: Dict[str, Any],
    add_monitor_log_func: AddMonitorLogFunc,
    **kwargs
) -> None:
    action = data.get("command")
    if action and isinstance(action, str):
        await add_monitor_log_func(f"Received action command: {action} (Handler is a placeholder).", "system_action_cmd")
