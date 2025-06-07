# backend/message_processing/agent_flow_handlers.py
import logging
import asyncio
from typing import Dict, Any, Callable, Coroutine

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.callbacks import AgentCancelledException
from backend.agent import create_agent_executor

logger = logging.getLogger(__name__)

# Define the types for the functions passed from server.py for clarity
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]

async def process_user_message(
    session_id: str,
    data: Dict[str, Any],
    session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    db_add_message_func: DBAddMessageFunc,
    **kwargs
) -> None:
    """
    Handles an incoming user message by creating and running the ReAct agent.
    This version includes enhanced logging and robust state management.
    """
    user_input = data.get("content", "")
    active_task_id = session_data_entry.get("current_task_id")

    # Pre-check: Ensure a task is active and the agent is not already running
    if not active_task_id:
        await send_ws_message_func("status_message", {"text": "Please select or create a task first.", "isError": True})
        return

    if connected_clients_entry.get("agent_task") is not None:
        logger.warning(f"[{session_id}] User message rejected: An agent task is already running.")
        await send_ws_message_func("status_message", {"text": "Agent is busy. Please wait.", "isError": True})
        return

    # --- Start Processing ---
    await db_add_message_func(active_task_id, session_id, "user_input", user_input)
    session_data_entry['cancellation_requested'] = False
    await send_ws_message_func("agent_thinking_update", {"status": "Thinking..."})

    callback_handler = session_data_entry.get("callback_handler")
    if not callback_handler:
        logger.critical(f"[{session_id}] CRITICAL_ERROR: Callback handler not found in session data.")
        await send_ws_message_func("status_message", {"text": "Critical Error: Session callback handler is missing.", "isError": True})
        return

    try:
        # --- Agent Setup ---
        memory = session_data_entry["memory"]
        llm_provider = session_data_entry.get("selected_llm_provider", settings.executor_default_provider)
        llm_model_name = session_data_entry.get("selected_llm_model_name", settings.executor_default_model_name)
        llm = get_llm(settings, llm_provider, llm_model_name, callbacks=[callback_handler], requested_for_role="Executor")
        tools = get_dynamic_tools(current_task_id=active_task_id)

        # Create the agent executor, passing callbacks directly to its constructor
        logger.info(f"CRITICAL_DEBUG: [{session_id}] Creating AgentExecutor...")
        agent_executor = create_agent_executor(
            llm=llm,
            tools=tools,
            memory=memory,
            max_iterations=settings.agent_max_iterations,
            callbacks=[callback_handler]
        )
        logger.info(f"CRITICAL_DEBUG: [{session_id}] AgentExecutor created successfully.")

    except Exception as e:
        logger.critical(f"[{session_id}] CRITICAL_ERROR: Failed to create agent executor: {e}", exc_info=True)
        await send_ws_message_func("agent_message", f"Sorry, I could not be initialized correctly. Error: {e}")
        return

    # --- Agent Execution ---
    async def agent_task_coroutine():
        try:
            logger.info(f"CRITICAL_DEBUG: [{session_id}] Agent coroutine started. Invoking agent...")
            await agent_executor.ainvoke({"input": user_input})
            logger.info(f"CRITICAL_DEBUG: [{session_id}] Agent invocation finished without error.")
        except AgentCancelledException:
            logger.warning(f"[{session_id}] Agent execution was cancelled by user.")
            await send_ws_message_func("status_message", {"text": "Operation cancelled."})
        except Exception as e:
            logger.error(f"[{session_id}] An unhandled error occurred in the agent task: {e}", exc_info=True)
            await send_ws_message_func("agent_message", f"An error occurred: {e}")
        finally:
            # This is the key state management fix.
            # Ensure the task is marked as completed regardless of how the coroutine exits.
            logger.info(f"CRITICAL_DEBUG: [{session_id}] Agent coroutine's 'finally' block reached. Clearing agent_task flag.")
            connected_clients_entry["agent_task"] = None
            # The on_agent_finish callback is now responsible for the final "Idle" status update.
            # This prevents a race condition where this "Idle" message could arrive before the final answer.

    # Start the agent task and store its handle
    agent_instance_task = asyncio.create_task(agent_task_coroutine())
    connected_clients_entry["agent_task"] = agent_instance_task
    logger.info(f"CRITICAL_DEBUG: [{session_id}] asyncio task for agent created and stored. Waiting for execution to complete.")

