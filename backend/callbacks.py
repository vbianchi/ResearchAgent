# backend/callbacks.py
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
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage # Ensure AIMessage is imported
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document

# Project Imports
from backend.config import settings
# from backend.tools import TEXT_EXTENSIONS, get_task_workspace_path # Not directly used in this version of callbacks

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run # For type hinting run_manager if used

logger = logging.getLogger(__name__)

# --- Define Custom Exception for Cancellation ---
class AgentCancelledException(Exception):
    """Custom exception to signal agent cancellation via callbacks."""
    pass
# ---------------------------------------------

# Define type hints
AddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]


# --- ADDED: For more structured logging, especially with callback info ---
# These constants can be used in the component_hint field or similar
LOG_SOURCE_SYSTEM = "SYSTEM"
LOG_SOURCE_INTENT_CLASSIFIER = "INTENT_CLASSIFIER"
LOG_SOURCE_PLANNER = "PLANNER"
LOG_SOURCE_CONTROLLER = "CONTROLLER"
LOG_SOURCE_EXECUTOR = "EXECUTOR"
LOG_SOURCE_EVALUATOR_STEP = "EVALUATOR_STEP"
LOG_SOURCE_EVALUATOR_OVERALL = "EVALUATOR_OVERALL"
LOG_SOURCE_TOOL_INTERNAL = "TOOL_INTERNAL_LLM" # e.g., for LLM calls within a tool
LOG_SOURCE_LLM_CORE = "LLM_CORE" # Generic LLM calls not tied to a specific agent component
LOG_SOURCE_CALLBACK_HANDLER = "CALLBACK_HANDLER" # For logs originating from this handler itself

# --- MODIFIED: Callback log message type constants ---
# Standard agent output message
AGENT_MESSAGE_TYPE_FINAL_ANSWER = "agent_message"
# Agent "thinking" or intermediate status updates
AGENT_MESSAGE_TYPE_THINKING_UPDATE = "agent_thinking_update"
# More structured status/progress update from agent components (sub-statuses)
SUB_TYPE_SUB_STATUS = "sub_status"
# Agent's internal "thought" process (can be multi-line, Markdown)
SUB_TYPE_THOUGHT = "thought"
# Used when a tool's output needs to be displayed directly in chat (e.g., file content)
SUB_TYPE_TOOL_RESULT_FOR_CHAT = "tool_result_for_chat"
# Plan step announcements
AGENT_MESSAGE_TYPE_MAJOR_STEP_ANNOUNCEMENT = "agent_major_step_announcement"

# Database message types (can mirror some of the above or be more specific)
DB_MSG_TYPE_USER_INPUT = "user_input"
DB_MSG_TYPE_AGENT_FINAL_ANSWER = AGENT_MESSAGE_TYPE_FINAL_ANSWER # Align with UI
DB_MSG_TYPE_SYSTEM_EVENT = "system_event" # Generic system messages
DB_MSG_TYPE_ERROR = "error_log" # General errors
DB_MSG_TYPE_LLM_TOKEN_USAGE = "llm_token_usage"
DB_MSG_TYPE_ARTIFACT_GENERATED = "artifact_generated"
DB_MSG_TYPE_TOOL_INPUT = "tool_input"
DB_MSG_TYPE_TOOL_OUTPUT = "tool_output"
DB_MSG_TYPE_AGENT_THOUGHT_ACTION = "agent_thought_action" # Legacy, from ReAct agent
DB_MSG_TYPE_MONITOR_LOG_PREFIX = "monitor_" # Generic prefix for monitor logs
DB_MSG_TYPE_SUB_STATUS = "db_sub_status" # For storing structured sub_status updates
DB_MSG_TYPE_THOUGHT = "db_thought" # For storing structured thought updates
DB_MSG_TYPE_TOOL_RESULT_FOR_CHAT = "db_tool_result_for_chat" # For storing tool results sent to chat
DB_MSG_TYPE_CONFIRMED_PLAN_LOG = "confirmed_plan_log" # To store the confirmed plan
DB_MSG_TYPE_MAJOR_STEP_ANNOUNCEMENT = "db_major_step_announcement"


class WebSocketCallbackHandler(AsyncCallbackHandler):
    """
    Async Callback handler for LangChain agent events.
    - Sends thinking status updates to the chat UI.
    - Sends final agent messages to the chat UI (adapted for LangGraph).
    - Sends monitor logs for all steps.
    - Extracts and sends LLM token usage.
    - Handles cancellation requests.
    - Logs artifact creation from write_file and triggers UI refresh (if tool callbacks are re-enabled).
    - Saves relevant events to the database.
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
        # Critical debug log for initial check
        logger.critical(f"CRITICAL_DEBUG_CANCEL_CHECK: [{self.session_id}] _check_cancellation for '{step_name}' (Initial Check): cancellation_requested flag is currently {self.session_data.get(self.session_id, {}).get('cancellation_requested', False)}")

        if self.session_data and self.session_id in self.session_data:
            current_session_specific_data = self.session_data[self.session_id]
            if current_session_specific_data.get('cancellation_requested', False):
                logger.warning(f"[{self.session_id}] Cancellation detected in callback before {step_name}. Raising AgentCancelledException.")
                raise AgentCancelledException("Cancellation requested by user.")
        else:
            logger.error(f"[{self.session_id}] Cannot check cancellation flag: Session data not found for session in shared dict.")
        
        # Critical debug log for post-yield check (simulating a brief pause for context switch)
        # In a real async scenario, this check might happen after an `await` point.
        # Since this is a synchronous check method, we log its state if it were to yield.
        logger.critical(f"CRITICAL_DEBUG_CANCEL_CHECK: [{self.session_id}] _check_cancellation for '{step_name}' (Post-Yield Check): cancellation_requested flag is currently {self.session_data.get(self.session_id, {}).get('cancellation_requested', False)}")


    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        try:
            self._check_cancellation("LLM execution")
            self.active_langgraph_node_name = None 
            if tags:
                for tag in tags:
                    if tag.startswith("langgraph:node:"):
                        self.active_langgraph_node_name = tag.split("langgraph:node:", 1)[1]
                        logger.debug(f"[{self.session_id}] on_llm_start: Active LangGraph node identified from tags: {self.active_langgraph_node_name}")
                        break
            if not self.active_langgraph_node_name and metadata and metadata.get("langgraph_node"): 
                 self.active_langgraph_node_name = metadata.get("langgraph_node")
                 logger.debug(f"[{self.session_id}] on_llm_start: Active LangGraph node identified from metadata: {self.active_langgraph_node_name}")

            if self.active_langgraph_node_name == "overall_evaluator": 
                self.is_final_evaluator_llm_active = True
                logger.info(f"[{self.session_id}] LLM call started for OverallEvaluatorNode.")
            else:
                self.is_final_evaluator_llm_active = False
            
            logger.info(
                f"[{self.session_id}] on_llm_start: Post-determination. "
                f"Node: '{self.active_langgraph_node_name}', "
                f"is_final_evaluator_llm_active: {self.is_final_evaluator_llm_active}"
            )

        except AgentCancelledException:
            raise 

    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        try:
            # --- START: MODIFIED SECTION (Added logging) ---
            self._check_cancellation("Chat Model execution")
            
            self.active_langgraph_node_name = None 
            current_node_name_for_log = "UnknownNode" 
            if tags:
                for tag in tags:
                    if tag.startswith("langgraph:node:"):
                        self.active_langgraph_node_name = tag.split("langgraph:node:", 1)[1]
                        current_node_name_for_log = self.active_langgraph_node_name
                        logger.debug(f"[{self.session_id}] on_chat_model_start: Active LangGraph node identified from tags: {self.active_langgraph_node_name}")
                        break
            if not self.active_langgraph_node_name and metadata and metadata.get("langgraph_node"):
                 self.active_langgraph_node_name = metadata.get("langgraph_node")
                 current_node_name_for_log = self.active_langgraph_node_name
                 logger.debug(f"[{self.session_id}] on_chat_model_start: Active LangGraph node identified from metadata: {self.active_langgraph_node_name}")

            if self.active_langgraph_node_name == "overall_evaluator":
                self.is_final_evaluator_llm_active = True
                logger.info(f"[{self.session_id}] Chat model call started for OverallEvaluatorNode.")
            else:
                self.is_final_evaluator_llm_active = False
            
            logger.info(
                f"[{self.session_id}] on_chat_model_start: Post-determination. "
                f"Node: '{self.active_langgraph_node_name}', "
                f"is_final_evaluator_llm_active: {self.is_final_evaluator_llm_active}"
            )
            # --- END: MODIFIED SECTION (Added logging) ---
            
            # Critical debug log to see metadata
            role_hint_from_meta = metadata.get("role_hint", "LLM_CORE") if metadata else "LLM_CORE"
            logger.critical(f"CRITICAL_DEBUG: [{self.session_id}] on_chat_model_start: ENTERED for role_hint: {role_hint_from_meta}. Metadata: {metadata}")

            log_msg_content = f"[Chat Model Start] Role: {role_hint_from_meta}, Node: {self.active_langgraph_node_name or 'N/A'}."
            await self.send_ws_message("monitor_log", f"{self._get_log_prefix()} {log_msg_content}")

        except AgentCancelledException:
            raise 

    async def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        pass

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        role_hint_from_meta = kwargs.get("metadata", {}).get("role_hint", "LLM_CORE") # Get role hint from kwargs metadata
        # --- START: MODIFIED SECTION (Added logging) ---
        logger.info(
            f"[{self.session_id}] on_llm_end: Entered. "
            f"Current active_langgraph_node_name (from self): '{self.active_langgraph_node_name}', "
            f"Current is_final_evaluator_llm_active (from self): {self.is_final_evaluator_llm_active}"
        )
        # Critical debug log for role hint
        logger.critical(f"CRITICAL_DEBUG: [{self.session_id}] on_llm_end: ENTERED for role_hint: {role_hint_from_meta}")
        # --- END: MODIFIED SECTION (Added logging) ---

        log_prefix = self._get_log_prefix()
        logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): Full response object type: {type(response)}") 
        
        input_tokens: Optional[int] = None; output_tokens: Optional[int] = None; total_tokens: Optional[int] = None
        model_name: str = "unknown_model"; source_for_tokens = "unknown"

        try:
            # ... (existing token parsing logic - unchanged) ...
            if response.llm_output and isinstance(response.llm_output, dict):
                logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): response.llm_output IS DICT: {response.llm_output}")
                llm_output_data = response.llm_output
                source_for_tokens = "llm_output"
                model_name = llm_output_data.get('model_name', llm_output_data.get('model', model_name))
                if 'token_usage' in llm_output_data and isinstance(llm_output_data['token_usage'], dict):
                    usage_dict = llm_output_data['token_usage']
                    input_tokens = usage_dict.get('prompt_tokens', usage_dict.get('input_tokens'))
                    output_tokens = usage_dict.get('completion_tokens', usage_dict.get('output_tokens'))
                    total_tokens = usage_dict.get('total_tokens')
                elif 'usage_metadata' in llm_output_data and isinstance(llm_output_data['usage_metadata'], dict): 
                    usage_dict = llm_output_data['usage_metadata']
                    input_tokens = usage_dict.get('prompt_token_count')
                    output_tokens = usage_dict.get('candidates_token_count')
                    total_tokens = usage_dict.get('total_token_count')
                elif 'eval_count' in llm_output_data: 
                    output_tokens = llm_output_data.get('eval_count')
                    input_tokens = llm_output_data.get('prompt_eval_count')
            else:
                 logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): response.llm_output IS NONE or NOT DICT.")

            if (input_tokens is None and output_tokens is None) and response.generations:
                logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): response.generations IS PRESENT and NOT EMPTY (length {len(response.generations)}).")
                for gen_list_idx, gen_list in enumerate(response.generations):
                    if not gen_list: logger.debug(f"DEBUG: Generation list {gen_list_idx} is empty for role {role_hint_from_meta}."); continue
                    logger.debug(f"DEBUG: Generation list {gen_list_idx} (length {len(gen_list)}) for role {role_hint_from_meta}:")
                    first_gen = gen_list[0]
                    source_for_tokens = "generations"
                    logger.debug(f"DEBUG:   Item 0 type: {type(first_gen)}")
                    if hasattr(first_gen, 'text'): logger.debug(f"DEBUG:     Item 0 text: {first_gen.text[:50]}...")
                    if hasattr(first_gen, 'message'): 
                        logger.debug(f"DEBUG:     Item 0 message type: {type(first_gen.message)}")
                        if hasattr(first_gen.message, 'content'): logger.debug(f"DEBUG:       Item 0 message content: {first_gen.message.content[:50]}...")
                        if hasattr(first_gen.message, 'additional_kwargs'): logger.debug(f"DEBUG:       Item 0 message additional_kwargs: {first_gen.message.additional_kwargs}")
                        if hasattr(first_gen.message, 'usage_metadata'): logger.debug(f"DEBUG:       Item 0 message usage_metadata: {first_gen.message.usage_metadata}")
                        if hasattr(first_gen.message, 'response_metadata'): logger.debug(f"DEBUG:       Item 0 message response_metadata: {first_gen.message.response_metadata}")

                    if hasattr(first_gen, 'generation_info'): logger.debug(f"DEBUG:     Item 0 generation_info: {first_gen.generation_info}")

                    if hasattr(first_gen, 'message') and hasattr(first_gen.message, 'usage_metadata') and first_gen.message.usage_metadata:
                        usage_metadata = first_gen.message.usage_metadata
                        if isinstance(usage_metadata, dict):
                            input_tokens = usage_metadata.get('prompt_token_count', usage_metadata.get('input_tokens'))
                            output_tokens = usage_metadata.get('candidates_token_count', usage_metadata.get('output_tokens'))
                            total_tokens = usage_metadata.get('total_token_count')
                            if hasattr(first_gen.message, 'response_metadata') and isinstance(first_gen.message.response_metadata, dict):
                                model_name_candidate = first_gen.message.response_metadata.get('model_name', model_name) 
                                if model_name_candidate and model_name_candidate != "unknown_model": model_name = model_name_candidate
                            if (not model_name or model_name == "unknown_model") and first_gen.generation_info and isinstance(first_gen.generation_info, dict): # Fallback to generation_info for model_name
                                model_name_candidate = first_gen.generation_info.get('model_name', first_gen.generation_info.get('model', model_name))
                                if model_name_candidate and model_name_candidate != "unknown_model": model_name = model_name_candidate

                            logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): Tokens from generations.message.usage_metadata (Gemini/ChatChunk): In={input_tokens}, Out={output_tokens}, Total={total_tokens}, Model={model_name}")
                            break 
                    elif hasattr(first_gen, 'generation_info') and first_gen.generation_info:
                        gen_info = first_gen.generation_info
                        if isinstance(gen_info, dict):
                            model_name_candidate = gen_info.get('model', gen_info.get('model_name', model_name))
                            if model_name_candidate and model_name_candidate != "unknown_model": model_name = model_name_candidate

                            if 'token_usage' in gen_info and isinstance(gen_info['token_usage'], dict): 
                                usage_dict = gen_info['token_usage']
                                input_tokens = usage_dict.get('prompt_tokens', usage_dict.get('input_tokens'))
                                output_tokens = usage_dict.get('completion_tokens', usage_dict.get('output_tokens'))
                                total_tokens = usage_dict.get('total_tokens')
                                logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): Tokens from generations.generation_info.token_usage (OpenAI): In={input_tokens}, Out={output_tokens}, Total={total_tokens}, Model={model_name}")
                            elif 'eval_count' in gen_info: 
                                output_tokens = gen_info.get('eval_count')
                                input_tokens = gen_info.get('prompt_eval_count')
                                logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): Tokens from generations.generation_info (Ollama): In={input_tokens}, Out={output_tokens}, Model={model_name}")
                            break
            else:
                 logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): Tokens not found or incomplete in llm_output, AND response.generations is missing or empty.")


            input_tokens = int(input_tokens) if input_tokens is not None else 0
            output_tokens = int(output_tokens) if output_tokens is not None else 0
            if total_tokens is None: total_tokens = input_tokens + output_tokens
            else: total_tokens = int(total_tokens)

            if input_tokens > 0 or output_tokens > 0 or total_tokens > 0:
                token_usage_payload = {
                    "model_name": str(model_name), "role_hint": role_hint_from_meta,
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "total_tokens": total_tokens, "source": source_for_tokens
                }
                logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): Final parsed tokens before sending: In={input_tokens}, Out={output_tokens}, Total={total_tokens}, Model={model_name}, Source={source_for_tokens}")
                usage_str = f"Model: {model_name}, Role: {role_hint_from_meta}, In: {input_tokens}, Out: {output_tokens}, Total: {total_tokens} (Src: {source_for_tokens})"
                log_content_tokens = f"[LLM Token Usage] {usage_str}"
                await self.send_ws_message("monitor_log", f"{log_prefix} {log_content_tokens}")
                await self.send_ws_message("llm_token_usage", token_usage_payload)
                await self._save_message(DB_MSG_TYPE_LLM_TOKEN_USAGE, json.dumps(token_usage_payload))
                logger.info(f"[{self.session_id}] Sending 'llm_token_usage' message with payload: {token_usage_payload}")
            else:
                 logger.debug(f"[{self.session_id}] on_llm_end (Role: {role_hint_from_meta}): No valid token usage data found to send (all zero or None).")
        except Exception as e:
            logger.error(f"[{self.session_id}] Error processing token usage in on_llm_end (Role: {role_hint_from_meta}): {e}", exc_info=True)

        # --- Send final agent message if this LLM call was from OverallEvaluatorNode ---
        if self.is_final_evaluator_llm_active:
            logger.info(f"[{self.session_id}] on_llm_end: Condition `if self.is_final_evaluator_llm_active` is TRUE. Proceeding to send agent_message.") # ADDED
            final_answer_content = "No final output from OverallEvaluator's LLM."
            if response.generations and response.generations[0]:
                first_generation = response.generations[0][0]
                if hasattr(first_generation, 'message') and isinstance(first_generation.message, AIMessage):
                    final_answer_content = first_generation.message.content
                elif hasattr(first_generation, 'text'): 
                    final_answer_content = first_generation.text
                else:
                    logger.warning(f"[{self.session_id}] Could not extract final answer content from OverallEvaluator's LLM response structure.")

            logger.info(f"[{self.session_id}] Sending final agent_message from OverallEvaluatorNode: '{final_answer_content[:100]}...'")
            await self.send_ws_message(AGENT_MESSAGE_TYPE_FINAL_ANSWER, final_answer_content)
            await self._save_message(DB_MSG_TYPE_AGENT_FINAL_ANSWER, final_answer_content) 
            
            self.is_final_evaluator_llm_active = False 
            self.active_langgraph_node_name = None 
        else:
            logger.info(
                f"[{self.session_id}] on_llm_end: Condition `if self.is_final_evaluator_llm_active` is FALSE. "
                f"Not sending agent_message. "
                f"active_node (from self): '{self.active_langgraph_node_name}', "
                f"is_final_evaluator_active_flag (from self): {self.is_final_evaluator_llm_active}"
            )

    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        log_prefix = self._get_log_prefix(); error_type_name = type(error).__name__
        logger.error(f"[{self.session_id}] LLM Error: {error}", exc_info=True)
        error_content = f"[LLM Error] {error_type_name}: {error}"
        await self.send_ws_message("monitor_log", f"{log_prefix} {error_content}")
        await self.send_ws_message("status_message", "Error occurred during LLM call.")
        await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": "Error during LLM call."})
        await self._save_message(DB_MSG_TYPE_ERROR, error_content)
        self.is_final_evaluator_llm_active = False 
        self.active_langgraph_node_name = None

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        chain_name = serialized.get("name", "UnknownChain") 
        node_name_from_tags = None
        if tags:
            for tag in tags:
                if tag.startswith("langgraph:node:"):
                    node_name_from_tags = tag.split("langgraph:node:", 1)[1]
                    break
        
        effective_node_name = node_name_from_tags or chain_name 
        # CRITICAL: Only update self.active_langgraph_node_name if it's a langgraph node run
        if node_name_from_tags or "langgraph" in (metadata.get("langgraph_path", []) if metadata else []):
             self.active_langgraph_node_name = effective_node_name 
             logger.info(f"[{self.session_id}] on_chain_start: LANGGRAPH NODE START DETECTED. Node Name: '{self.active_langgraph_node_name}'. Self.is_final_eval_llm_active BEFORE check: {self.is_final_evaluator_llm_active}")
             if self.active_langgraph_node_name == "overall_evaluator":
                logger.info(f"[{self.session_id}] on_chain_start: OverallEvaluatorNode chain starting. Flag is_final_evaluator_llm_active should be set to TRUE by its LLM's on_chat_model_start shortly.")
             else:
                # Reset the flag if another node starts, to prevent stale state from a previous overall_evaluator run in the same session (if that's possible)
                if self.is_final_evaluator_llm_active and self.active_langgraph_node_name != "overall_evaluator":
                    logger.warning(f"[{self.session_id}] on_chain_start: Starting node '{self.active_langgraph_node_name}' but is_final_evaluator_llm_active was TRUE. Resetting to FALSE.")
                    self.is_final_evaluator_llm_active = False

        logger.debug(f"[{self.session_id}] on_chain_start: Chain/Node='{effective_node_name}', Inputs='{str(inputs)[:100]}...', Tags='{tags}', Metadata='{metadata}'")

    async def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        logger.debug(f"[{self.session_id}] on_chain_end: For Node='{self.active_langgraph_node_name or 'UnknownChain'}', Outputs='{str(outputs)[:100]}...'")
        # Don't reset active_langgraph_node_name here, as on_llm_end relies on it.
        # It will be reset in on_llm_end if it was the final evaluator's LLM, or by the next on_chain_start/on_llm_start.
        pass

    async def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None: 
        log_prefix = self._get_log_prefix(); error_type_name = type(error).__name__
        active_node = self.active_langgraph_node_name or "UnknownChain"
        logger.error(f"[{self.session_id}] Chain Error (Node: {active_node}): {error}", exc_info=True)
        error_content = f"[Chain Error] Node '{active_node}' failed: {error_type_name}: {error}"
        await self.send_ws_message("monitor_log", f"{log_prefix} {error_content}")
        await self.send_ws_message("status_message", f"Error occurred in agent processing node: {active_node}.")
        await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": f"Error in node: {active_node}."})
        await self._save_message(DB_MSG_TYPE_ERROR, error_content)
        # If a chain errors out, it's unlikely it was the final evaluator LLM, but reset defensively
        self.is_final_evaluator_llm_active = False 
        # Don't reset active_langgraph_node_name here to ensure errors are logged with context.
        # It will be overwritten by the next node start or cleared if the graph ends.


    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.current_tool_name = serialized.get("name", "Unknown Tool")
        try:
            self._check_cancellation(f"Tool execution ('{self.current_tool_name}')")
        except AgentCancelledException:
            self.current_tool_name = None 
            raise
        log_prefix = self._get_log_prefix()
        log_input = input_str[:500] + "..." if len(str(input_str)) > 500 else input_str
        log_content = f"[Tool Start] Using '{self.current_tool_name}' with input: '{log_input}'"
        logger.info(f"[{self.session_id}] {log_content}")
        await self.send_ws_message("monitor_log", f"{log_prefix} {log_content}")
        await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": f"Using tool: {self.current_tool_name}..."})
        await self._save_message(DB_MSG_TYPE_TOOL_INPUT, f"{self.current_tool_name}:::{log_input}")

    async def on_tool_end(self, output: str, name: str = "Unknown Tool", **kwargs: Any) -> None:
        tool_name_for_log = name
        if name == "Unknown Tool" and self.current_tool_name:
            tool_name_for_log = self.current_tool_name
        
        log_prefix = self._get_log_prefix()
        output_str = str(output)
        logger.info(f"[{self.session_id}] Tool '{tool_name_for_log}' finished. Output length: {len(output_str)}")

        monitor_output = output_str
        success_prefix = "SUCCESS::write_file:::"
        if tool_name_for_log == "write_file" and output_str.startswith(success_prefix):
            try:
                if len(output_str) > len(success_prefix):
                    relative_path_str = output_str[len(success_prefix):]
                    logger.info(f"[{self.session_id}] Detected successful write_file: '{relative_path_str}'")
                    await self._save_message(DB_MSG_TYPE_ARTIFACT_GENERATED, relative_path_str)
                    await self.send_ws_message("monitor_log", f"{log_prefix} [ARTIFACT_GENERATED] {relative_path_str} (via {tool_name_for_log})")
                    monitor_output = f"Successfully wrote file: '{relative_path_str}'"
                    if self.current_task_id:
                        logger.info(f"[{self.session_id}] Triggering artifact refresh for task {self.current_task_id} after {tool_name_for_log}.")
                        await self.send_ws_message("trigger_artifact_refresh", {"taskId": self.current_task_id})
                else:
                    logger.warning(f"[{self.session_id}] write_file output matched prefix but had no filename: '{output_str}'")
            except Exception as parse_err:
                logger.error(f"[{self.session_id}] Error processing write_file success output '{output_str}': {parse_err}", exc_info=True)
        else:
            monitor_output = output_str[:1000] + "..." if len(output_str) > 1000 else output_str

        formatted_output = f"\n---\n{monitor_output.strip()}\n---"
        log_content = f"[Tool Output] Tool '{tool_name_for_log}' returned:{formatted_output}"
        await self.send_ws_message("monitor_log", f"{log_prefix} {log_content}")
        await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": f"Processed tool: {tool_name_for_log}."})
        await self._save_message(DB_MSG_TYPE_TOOL_OUTPUT, output_str)
        self.current_tool_name = None

    async def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], name: str = "Unknown Tool", **kwargs: Any) -> None:
        actual_tool_name = name
        if name == "Unknown Tool" and self.current_tool_name:
            actual_tool_name = self.current_tool_name
        
        if isinstance(error, AgentCancelledException):
            logger.warning(f"[{self.session_id}] Tool '{actual_tool_name}' execution cancelled by AgentCancelledException.")
            await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": f"Tool cancelled: {actual_tool_name}."})
            self.current_tool_name = None 
            raise error

        log_prefix = self._get_log_prefix(); error_type_name = type(error).__name__
        error_str = str(error)
        
        logger.error(f"[{self.session_id}] Tool '{actual_tool_name}' Error: {error_str}", exc_info=True)
        error_content = f"[Tool Error] Tool '{actual_tool_name}' failed: {error_type_name}: {error_str}"
        await self.send_ws_message("monitor_log", f"{log_prefix} {error_content}")
        await self.send_ws_message("status_message", f"Error occurred during tool execution: {actual_tool_name}.")
        await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": f"Error with tool: {actual_tool_name}."})
        await self._save_message(f"{DB_MSG_TYPE_ERROR}_tool", f"{actual_tool_name}::{error_type_name}::{error_str}")
        self.current_tool_name = None

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        log_prefix = self._get_log_prefix()
        thought = ""
        if action.log and "Thought:" in action.log:
            thought = action.log.split("Thought:",1)[1].split("Action:")[0].strip()

        if thought:
            logger.debug(f"[{self.session_id}] Extracted thought (Action): {thought}")
            await self.send_ws_message("monitor_log", f"{log_prefix} [Agent Thought (Action)] {thought}")
            await self._save_message(DB_MSG_TYPE_AGENT_THOUGHT_ACTION, thought)
        await self.send_ws_message(AGENT_MESSAGE_TYPE_THINKING_UPDATE, {"status": "Processing action..."})


    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        log_prefix = self._get_log_prefix()
        logger.info(f"[{self.session_id}] on_agent_finish (ReAct style). Log: {finish.log}")
        self.current_tool_name = None 
        # Defensively reset these flags. If this on_agent_finish is from a sub-agent within a LangGraph node,
        # the main graph's on_llm_end/on_chain_start will manage the flags for the graph's context.
        self.is_final_evaluator_llm_active = False 
        self.active_langgraph_node_name = None 


    async def on_text(self, text: str, **kwargs: Any) -> Any: pass
    async def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any: pass
    async def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> Any: pass
    async def on_retriever_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any: pass

