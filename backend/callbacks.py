# backend/callbacks.py
import logging
import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Coroutine
from uuid import UUID
import json
import re

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult, GenerationChunk, ChatGenerationChunk
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish

logger = logging.getLogger(__name__)

class AgentCancelledException(Exception):
    """Custom exception to signal agent cancellation via callbacks."""
    pass

SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]

class WebSocketCallbackHandler(AsyncCallbackHandler):
    """Async Callback handler for LangChain agent events, simplified for ReAct agent."""
    always_verbose: bool = True
    
    def __init__(self, session_id: str, send_ws_message_func: SendWSMessageFunc, db_add_message_func: DBAddMessageFunc, session_data_ref: Dict[str, Any]):
        super().__init__()
        self.session_id = session_id
        self.send_ws_message = send_ws_message_func
        self.db_add_message = db_add_message_func
        self.session_data = session_data_ref
        self.current_task_id: Optional[str] = None
        logger.info(f"[{self.session_id}] WebSocketCallbackHandler initialized (Simplified).")

    def set_task_id(self, task_id: Optional[str]):
        logger.debug(f"[{self.session_id}] Callback handler task ID set to: {task_id}")
        self.current_task_id = task_id

    def _get_log_prefix(self) -> str:
        timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
        return f"[{timestamp}][{self.session_id[:8]}]"

    async def _save_message(self, msg_type: str, content: str):
        if self.current_task_id:
            try:
                await self.db_add_message(self.current_task_id, self.session_id, msg_type, str(content))
            except Exception as e:
                logger.error(f"[{self.session_id}] Callback DB save error: {e}", exc_info=True)

    def _check_cancellation(self, step_name: str):
        """Checks if cancellation has been requested for the current session."""
        session_state = self.session_data.get(self.session_id, {})
        if session_state.get('cancellation_requested', False):
            logger.warning(f"[{self.session_id}] Cancellation detected in callback before {step_name}.")
            raise AgentCancelledException("Cancellation requested by user.")

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self._check_cancellation("LLM execution")
        log_msg = f"[LLM Start] Role: {kwargs.get('metadata', {}).get('role_hint', 'ReActAgent')}"
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_msg}", "log_source": "LLM_CORE"})

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        self._check_cancellation("Chat Model execution")
        log_msg = f"[Chat Model Start] Role: {kwargs.get('metadata', {}).get('role_hint', 'ReActAgent')}"
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_msg}", "log_source": "LLM_CORE"})

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            token_usage = response.llm_output.get('token_usage', {}) if response.llm_output else {}
            total_tokens = token_usage.get('total_tokens', 0)
            if total_tokens > 0:
                token_usage_payload = {
                    "model_name": response.llm_output.get('model_name', 'unknown'),
                    "role_hint": "ReActAgent",
                    "total_tokens": total_tokens,
                    "input_tokens": token_usage.get('prompt_tokens', 0),
                    "output_tokens": token_usage.get('completion_tokens', 0),
                }
                await self.send_ws_message("llm_token_usage", token_usage_payload)
                await self._save_message("llm_token_usage", json.dumps(token_usage_payload))
        except Exception as e:
            logger.error(f"[{self.session_id}] Error processing token usage in on_llm_end: {e}")

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._check_cancellation(f"Tool execution ('{serialized.get('name')}')")
        log_content = f"[Tool Start] Using '{serialized.get('name')}' with input: '{input_str[:200]}...'"
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_content}", "log_source": f"TOOL_START_{serialized.get('name')}"})
        await self.send_ws_message("agent_thinking_update", {"status": f"Using tool: {serialized.get('name')}..."})
        await self._save_message("tool_input", f"{serialized.get('name')}:::{input_str}")

    async def on_tool_end(self, output: str, name: str, **kwargs: Any) -> None:
        log_content = f"[Tool Output] Tool '{name}' returned (first 500 chars):\n---\n{output[:500]}...\n---"
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_content}", "log_source": f"TOOL_OUTPUT_{name}"})
        await self.send_ws_message("agent_thinking_update", {"status": f"Processed tool: {name}."})
        await self._save_message("tool_output", output)

    async def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], name: str, **kwargs: Any) -> None:
        log_content = f"[Tool Error] Tool '{name}' failed: {type(error).__name__}: {error}"
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_content}", "log_source": f"TOOL_ERROR_{name}"})
        await self.send_ws_message("status_message", {"text": f"Error in tool: {name}.", "isError": True})
        await self._save_message("error_tool", f"{name}::{error}")

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self._check_cancellation("Agent action")
        thought_match = re.search(r"Thought: (.*)", action.log, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "No thought extracted."
        log_content = f"[Agent Thought] {thought}"
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_content}", "log_source": "AGENT_THOUGHT"})
        await self._save_message("agent_thought", thought)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        final_answer = finish.return_values.get('output', 'No final answer provided.')
        log_content = f"[Agent Finish] Final answer: {final_answer[:200]}..."
        await self.send_ws_message("monitor_log", {"text": f"{self._get_log_prefix()} {log_content}", "log_source": "AGENT_FINISH"})
        await self.send_ws_message("agent_message", final_answer)
        await self._save_message("agent_message", final_answer)
