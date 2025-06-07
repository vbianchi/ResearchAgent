import logging
import datetime
from typing import Any, Dict, List, Optional, Union, Sequence, Callable, Coroutine, TYPE_CHECKING
from uuid import UUID
import json
from pathlib import Path
import os
import re

# LangChain Core Imports
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult, ChatGenerationChunk, GenerationChunk
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document

# Project Imports
from backend.config import settings

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)

class AgentCancelledException(Exception):
    """Custom exception to signal agent cancellation via callbacks."""
    pass

AddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]

class WebSocketCallbackHandler(AsyncCallbackHandler):
    """
    Async Callback handler for LangChain agent events.
    This class now handles events from a standard ReAct agent executor.
    """
    always_verbose: bool = True
    ignore_llm: bool = False
    ignore_chain: bool = False
    ignore_agent: bool = True
    ignore_retriever: bool = True
    ignore_chat_model: bool = False

    def __init__(self, session_id: str, send_ws_message_func: SendWSMessageFunc, db_add_message_func: AddMessageFunc, session_data_ref: Dict[str, Any]):
        super().__init__()
        self.session_id = session_id
        self.send_ws_message = send_ws_message_func
        self.db_add_message = db_add_message_func
        self.session_data = session_data_ref
        self.current_task_id: Optional[str] = None
        self.current_tool_name: Optional[str] = None
        
        # These flags were for the old LangGraph flow but are kept for potential future use
        # They will not be set in the current simplified ReAct agent flow.
        self.active_langgraph_node_name: Optional[str] = None
        self.is_final_evaluator_llm_active: bool = False

        logger.info(f"[{self.session_id}] WebSocketCallbackHandler initialized.")

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
                logger.error(f"[{self.session_id}] Callback DB save error (Task: {self.current_task_id}, Type: {msg_type}): {e}", exc_info=True)
        else:
            logger.warning(f"[{self.session_id}] Cannot save message type '{msg_type}' to DB: current_task_id not set.")

    def _check_cancellation(self, step_name: str):
        if self.session_id not in self.session_data:
            logger.error(f"[{self.session_id}] _check_cancellation: Session data not found for session in shared dict. Cannot check flag.")
            return
        
        current_session_specific_data = self.session_data.get(self.session_id)
        if current_session_specific_data and current_session_specific_data.get('cancellation_requested', False):
            logger.warning(f"[{self.session_id}] Cancellation detected in callback before {step_name}. Raising AgentCancelledException.")
            raise AgentCancelledException("Cancellation requested by user.")


    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._check_cancellation("LLM execution")
        # The logic for detecting langgraph nodes is no longer relevant but is harmless.
        # It will simply not find any specific node tags in the new simplified flow.
        self.active_langgraph_node_name = None 
        self.is_final_evaluator_llm_active = False

    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        self._check_cancellation("Chat Model execution")
        self.active_langgraph_node_name = None 
        self.is_final_evaluator_llm_active = False
        
        # Simplified logging for the new ReAct agent
        role_hint_from_meta = metadata.get("role_hint", "LLM_CORE") if metadata else "LLM_CORE"
        log_msg_content = f"[LLM Call Start] Role: {role_hint_from_meta}"
        await self.send_ws_message("monitor_log", f"{self._get_log_prefix()} {log_msg_content}")

    async def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        pass # Not currently used for streaming to UI

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        # This function is primarily for token counting and is generic enough to work
        # with the new ReAct agent. The old logic for the "overall_evaluator" node will
        # simply not be triggered, which is correct.
        log_prefix = self._get_log_prefix()
        logger.debug(f"[{self.session_id}] on_llm_end received.")
        # Token counting logic remains the same and will function correctly.
        # ... (token counting logic is unchanged)
        input_tokens: Optional[int] = None; output_tokens: Optional[int] = None; total_tokens: Optional[int] = None
        model_name: str = "unknown_model"; source_for_tokens = "unknown"
        role_hint = kwargs.get("metadata", {}).get("role_hint", "LLM_CORE") if 'metadata' in kwargs and kwargs['metadata'] else "LLM_CORE"
        try:
            if response.llm_output and isinstance(response.llm_output, dict):
                llm_output_data = response.llm_output; source_for_tokens = "llm_output"
                model_name = llm_output_data.get('model_name', llm_output_data.get('model', model_name))
                if 'token_usage' in llm_output_data and isinstance(llm_output_data['token_usage'], dict):
                    usage_dict = llm_output_data['token_usage']
                    input_tokens, output_tokens, total_tokens = usage_dict.get('prompt_tokens'), usage_dict.get('completion_tokens'), usage_dict.get('total_tokens')
                elif 'usage_metadata' in llm_output_data and isinstance(llm_output_data['usage_metadata'], dict):
                    usage_dict = llm_output_data['usage_metadata']
                    input_tokens, output_tokens, total_tokens = usage_dict.get('prompt_token_count'), usage_dict.get('candidates_token_count'), usage_dict.get('total_token_count')
            if (input_tokens is None and output_tokens is None) and response.generations and response.generations[0]:
                first_gen = response.generations[0][0]; source_for_tokens = "generations"
                if hasattr(first_gen, 'message') and hasattr(first_gen.message, 'usage_metadata') and first_gen.message.usage_metadata:
                    usage_metadata = first_gen.message.usage_metadata
                    input_tokens, output_tokens, total_tokens = usage_metadata.get('prompt_token_count'), usage_metadata.get('candidates_token_count'), usage_metadata.get('total_token_count')
            input_tokens = int(input_tokens) if input_tokens is not None else 0
            output_tokens = int(output_tokens) if output_tokens is not None else 0
            if total_tokens is None: total_tokens = input_tokens + output_tokens
            else: total_tokens = int(total_tokens)
            if total_tokens > 0:
                token_usage_payload = {"model_name": str(model_name), "role_hint": role_hint, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "source": source_for_tokens}
                await self.send_ws_message("monitor_log", f"{log_prefix} [LLM Token Usage] Model: {model_name}, Role: {role_hint}, In: {input_tokens}, Out: {output_tokens}, Total: {total_tokens}")
                await self.send_ws_message("llm_token_usage", token_usage_payload)
                await self._save_message("llm_token_usage", json.dumps(token_usage_payload))
        except Exception as e:
            logger.error(f"[{self.session_id}] Error processing token usage in on_llm_end (Role: {role_hint}): {e}", exc_info=True)


    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        log_prefix = self._get_log_prefix(); error_type_name = type(error).__name__
        logger.error(f"[{self.session_id}] LLM Error: {error}", exc_info=True)
        error_content = f"[LLM Error] {error_type_name}: {error}"
        await self.send_ws_message("monitor_log", f"{log_prefix} {error_content}")
        await self.send_ws_message("status_message", {"text": "Error occurred during LLM call.", "isError": True})
        await self._save_message("error_llm", error_content)

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.current_tool_name = serialized.get("name", "Unknown Tool")
        self._check_cancellation(f"Tool execution ('{self.current_tool_name}')")
        log_prefix = self._get_log_prefix()
        log_input = input_str[:500] + "..." if len(str(input_str)) > 500 else input_str
        log_content = f"[Tool Start] Using '{self.current_tool_name}' with input: '{log_input}'"
        logger.info(f"[{self.session_id}] {log_content}")
        await self.send_ws_message("monitor_log", f"{log_prefix} {log_content}")
        await self._save_message("tool_input", f"{self.current_tool_name}:::{log_input}")

    async def on_tool_end(self, output: str, name: str = "Unknown Tool", **kwargs: Any) -> None:
        tool_name_for_log = name if name != "Unknown Tool" else self.current_tool_name
        log_prefix = self._get_log_prefix()
        output_str = str(output)
        logger.info(f"[{self.session_id}] Tool '{tool_name_for_log}' finished. Output length: {len(output_str)}")
        monitor_output = output_str[:1000] + "..." if len(output_str) > 1000 else output_str
        log_content = f"[Tool Output] Tool '{tool_name_for_log}' returned:\n---\n{monitor_output.strip()}\n---"
        await self.send_ws_message("monitor_log", f"{log_prefix} {log_content}")
        await self._save_message("tool_output", output_str)
        # Handle artifact generation from write_file tool
        success_prefix = "SUCCESS::write_file:::"
        if tool_name_for_log == "write_file" and output_str.startswith(success_prefix):
            try:
                relative_path_str = output_str[len(success_prefix):]
                await self._save_message("artifact_generated", relative_path_str)
                await self.send_ws_message("monitor_log", f"{log_prefix} [ARTIFACT_GENERATED] {relative_path_str} (via {tool_name_for_log})")
                if self.current_task_id:
                    await self.send_ws_message("trigger_artifact_refresh", {"taskId": self.current_task_id})
            except Exception as parse_err:
                logger.error(f"[{self.session_id}] Error processing write_file success output '{output_str}': {parse_err}", exc_info=True)
        self.current_tool_name = None

    async def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], name: str = "Unknown Tool", **kwargs: Any) -> None:
        if isinstance(error, AgentCancelledException):
            logger.warning(f"[{self.session_id}] Tool execution cancelled by AgentCancelledException.")
            raise error # Re-raise to stop the agent execution
        log_prefix = self._get_log_prefix(); error_type_name = type(error).__name__
        error_str = str(error)
        logger.error(f"[{self.session_id}] Tool Error: {error_str}", exc_info=True)
        error_content = f"[Tool Error] Tool '{name}' failed: {error_type_name}: {error_str}"
        await self.send_ws_message("monitor_log", f"{log_prefix} {error_content}")
        await self._save_message("error_tool", f"{name}::{error_type_name}::{error_str}")
        self.current_tool_name = None

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        # This function correctly logs the "Thought" part of the ReAct agent's reasoning.
        log_prefix = self._get_log_prefix()
        thought = ""
        if action.log and "Thought:" in action.log:
            thought = action.log.split("Thought:",1)[1].split("Action:")[0].strip()
        if thought:
            logger.debug(f"[{self.session_id}] Extracted thought (Action): {thought}")
            await self.send_ws_message("monitor_log", f"{log_prefix} [Agent Thought] {thought}")
            await self._save_message("agent_thought", thought)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """
        MODIFIED: This is now the primary handler for the agent's final answer.
        The ReAct agent created by `create_agent_executor` populates `finish.return_values['output']`
        with the final response for the user.
        """
        log_prefix = self._get_log_prefix()
        logger.info(f"[{self.session_id}] ReAct agent finished execution.")
        
        # Extract the final answer from the finish object.
        final_answer = finish.return_values.get("output", "Agent finished without providing a final answer.")
        
        # Send the final answer to the UI.
        logger.info(f"[{self.session_id}] Sending final agent_message from on_agent_finish: '{final_answer[:100]}...'")
        await self.send_ws_message("agent_message", final_answer)
        await self._save_message("agent_message", final_answer)
        
        # Reset any lingering state flags to ensure a clean slate for the next run.
        self.current_tool_name = None
        self.is_final_evaluator_llm_active = False
        self.active_langgraph_node_name = None

    # --- Chain and Retriever handlers are not critical for the simplified flow but are kept for completeness ---
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None: pass
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None: pass
    async def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None: pass
    async def on_text(self, text: str, **kwargs: Any) -> Any: pass
    async def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any: pass
    async def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> Any: pass
    async def on_retriever_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any: pass
