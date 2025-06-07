# backend/message_processing/agent_flow_handlers.py
import logging
import json
import asyncio
from typing import Dict, Any, Callable, Coroutine

# LangChain Imports
from langchain_core.runnables import RunnableConfig

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.callbacks import AgentCancelledException
# This is the crucial import for our new, simplified agent core
from backend.agent import create_agent_executor

logger = logging.getLogger(__name__)

# Type Hints for Passed-in Functions
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]


async def process_user_message(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc, db_add_message_func: DBAddMessageFunc,
    **kwargs # Absorb any other arguments that might be passed
) -> None:
    """
    Handles an incoming user message by invoking the simple ReAct agent executor.
    This is the new core logic for the simplified chat system.
    """
    user_input = data.get("content", "")
    active_task_id = session_data_entry.get("current_task_id")

    if not active_task_id:
        await send_ws_message_func("status_message", {"text": "Please select or create a task first.", "isError": True})
        return

    if connected_clients_entry.get("agent_task"):
        await send_ws_message_func("status_message", {"text": "Agent is busy. Please wait.", "isError": True})
        return

    # Log user input to DB and reset cancellation flag
    await db_add_message_func(active_task_id, session_id, "user_input", user_input)
    session_data_entry['cancellation_requested'] = False
    await send_ws_message_func("agent_thinking_update", {"status": "Thinking..."})

    # 1. Get the session-specific callback handler and memory
    callback_handler = session_data_entry.get("callback_handler")
    if not callback_handler:
        logger.error(f"[{session_id}] CRITICAL: Callback handler not found in session data.")
        await send_ws_message_func("status_message", {"text": "Critical Error: Session callback handler is missing.", "isError": True})
        return
        
    memory = session_data_entry["memory"]

    # 2. Get the LLM selected for the Executor role in the UI
    llm_provider = session_data_entry.get("selected_llm_provider", settings.executor_default_provider)
    llm_model_name = session_data_entry.get("selected_llm_model_name", settings.executor_default_model_name)
    llm = get_llm(settings, llm_provider, llm_model_name, callbacks=[callback_handler], requested_for_role="Executor")

    # 3. Load all available tools for the current task context
    tools = get_dynamic_tools(current_task_id=active_task_id)

    # 4. Create the ReAct agent executor from backend/agent.py
    try:
        agent_executor = create_agent_executor(
            llm=llm,
            tools=tools,
            memory=memory,
            max_iterations=settings.agent_max_iterations
        )
    except Exception as e:
        logger.error(f"[{session_id}] Failed to create agent executor: {e}", exc_info=True)
        await send_ws_message_func("agent_message", f"Sorry, I could not be initialized correctly. Error: {e}")
        return

    # 5. Create and run the agent as an asyncio Task to avoid blocking the server
    async def agent_task_coroutine():
        try:
            # The agent will now run its ReAct loop. Callbacks will stream thoughts/actions to the UI.
            await agent_executor.ainvoke(
                {"input": user_input},
                config=RunnableConfig(callbacks=[callback_handler]) # Pass callbacks via config
            )
        except AgentCancelledException:
            logger.warning(f"[{session_id}] Agent execution was cancelled by user.")
            await send_ws_message_func("status_message", {"text": "Operation cancelled."})
        except Exception as e:
            logger.error(f"[{session_id}] An unhandled error occurred in the agent task: {e}", exc_info=True)
            # Try to send a user-facing error message
            await send_ws_message_func("agent_message", f"I'm sorry, an unexpected error occurred: {e}")
        finally:
            # Clean up the task entry in the session
            connected_clients_entry["agent_task"] = None
            await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

    agent_instance_task = asyncio.create_task(agent_task_coroutine())
    connected_clients_entry["agent_task"] = agent_instance_task
    logger.info(f"[{session_id}] Started ReAct agent task for user input: '{user_input[:50]}...'")