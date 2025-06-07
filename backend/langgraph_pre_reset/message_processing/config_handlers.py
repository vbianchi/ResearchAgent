# backend/message_processing/config_handlers.py
import logging
from typing import Dict, Any, Callable, Coroutine, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# Type Hints for Passed-in Functions
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]


async def process_set_llm(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    add_monitor_log_func: AddMonitorLogFunc, **kwargs
) -> None:
    # This function's logic is correct and remains unchanged.
    llm_id = data.get("llm_id")
    if llm_id and "::" in llm_id:
        session_data_entry["selected_llm_provider"] = llm_id.split("::")[0]
        session_data_entry["selected_llm_model_name"] = llm_id.split("::")[1]
        await add_monitor_log_func(f"Session LLM (Executor) set to {llm_id}", "system_llm_set")
    else:
        session_data_entry["selected_llm_provider"] = settings.executor_default_provider
        session_data_entry["selected_llm_model_name"] = settings.executor_default_model_name
        await add_monitor_log_func(f"Session LLM (Executor) reset to default.", "system_llm_set")

async def process_get_available_models(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: Optional[AddMonitorLogFunc] = None,
    **kwargs
) -> None:
    # This function's logic is correct and remains unchanged.
    logger.info(f"[{session_id}] Received request for available models.")
    role_llm_defaults = {
        "intent_classifier": f"{settings.intent_classifier_provider}::{settings.intent_classifier_model_name}",
        "planner": f"{settings.planner_provider}::{settings.planner_model_name}",
        "controller": f"{settings.controller_provider}::{settings.controller_model_name}",
        "evaluator": f"{settings.evaluator_provider}::{settings.evaluator_model_name}",
    }
    await send_ws_message_func("available_models", {
        "gemini": settings.gemini_available_models,
        "ollama": settings.ollama_available_models,
        "default_executor_llm_id": f"{settings.executor_default_provider}::{settings.executor_default_model_name}",
        "role_llm_defaults": role_llm_defaults
    })
    if add_monitor_log_func:
        await add_monitor_log_func("Sent available models list to client.", "system_info")


async def process_set_session_role_llm(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    add_monitor_log_func: AddMonitorLogFunc, **kwargs
) -> None:
    """
    MODIFIED: This function now correctly validates the role against a hardcoded list
    of valid roles, fixing the "Invalid role" error spam.
    """
    role = data.get("role")
    llm_id_override = data.get("llm_id")

    # MODIFIED: Correct validation logic.
    valid_roles = ["intent_classifier", "planner", "controller", "evaluator"]
    if not role or role not in valid_roles:
        logger.warning(f"[{session_id}] Invalid or missing 'role' in set_session_role_llm: {role}")
        await add_monitor_log_func(f"Error: Invalid role for LLM override: {role}", "error_system")
        return

    # The keys in session_data_entry are like "session_planner_llm_id"
    session_key = f"session_{role}_llm_id"

    if llm_id_override and isinstance(llm_id_override, str) and "::" in llm_id_override:
        session_data_entry[session_key] = llm_id_override
        logger.info(f"[{session_id}] Session LLM override for role '{role}' set to: {llm_id_override}")
        await add_monitor_log_func(f"Session LLM for role '{role}' overridden to {llm_id_override}.", "system_llm_set")
    else:
        # This handles both empty strings (from "Use System Default") and invalid data.
        session_data_entry[session_key] = None
        logger.info(f"[{session_id}] Cleared session LLM override for role '{role}'. It will use the system default.")
        await add_monitor_log_func(f"Session LLM for role '{role}' reset to system default.", "system_llm_set")
