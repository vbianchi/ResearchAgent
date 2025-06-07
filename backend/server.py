# backend/server.py
import asyncio
import websockets
import json
import datetime
import logging
import uuid
from typing import Optional, List, Dict, Any, Set, Tuple, Callable, Coroutine
from pathlib import Path
import os
import signal
import functools # For functools.partial
import warnings
import aiofiles # For async file operations
import unicodedata # For sanitize_filename
import re
import urllib.parse # For artifact URL encoding

# --- Web Server Imports ---
from aiohttp import web
from aiohttp.web import FileResponse
import aiohttp_cors
# -------------------------

# LangChain Imports
# No longer importing RunnableConfig or specific messages here as they are handled in sub-modules
from langchain_core.language_models.base import BaseLanguageModel
from langchain.memory import ConversationBufferWindowMemory # For session memory

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools, get_task_workspace_path, BASE_WORKSPACE_ROOT, TEXT_EXTENSIONS
from backend.callbacks import WebSocketCallbackHandler, AgentCancelledException
from backend.db_utils import (
    init_db, add_task, add_message, get_messages_for_task,
    delete_task_and_messages, rename_task_in_db
)
# Import all handlers from the message_processing package
from backend.message_processing import * # noqa F403 (imports all from __all__)

# --- REMOVED LANGGRAPH IMPORT ---
# from backend.langgraph_agent import research_agent_graph as compiled_research_agent_graph
# --------------------------------

# Configure logging
log_level = settings.log_level
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger
logger.info(f"Logging level set to {log_level}")

# Startup LLM check (this is still useful)
try:
    default_llm_instance_for_startup_checks: BaseLanguageModel = get_llm(
        settings, 
        provider=settings.default_provider, 
        model_name=settings.default_model_name
    )
    logger.info(f"Default Base LLM for startup checks initialized successfully: {settings.default_provider}::{settings.default_model_name}")
except Exception as llm_e:
    logger.critical(f"FATAL: Failed during startup LLM initialization: {llm_e}", exc_info=True)
    exit(1)

# Global state dictionaries
connected_clients: Dict[str, Dict[str, Any]] = {}
session_data: Dict[str, Dict[str, Any]] = {}

# File Server Configuration
FILE_SERVER_LISTEN_HOST = "0.0.0.0"  
FILE_SERVER_CLIENT_HOST = settings.file_server_hostname 
FILE_SERVER_PORT = 8766
logger.info(f"File server will listen on {FILE_SERVER_LISTEN_HOST}:{FILE_SERVER_PORT}")
logger.info(f"File server URLs constructed for client will use: http://{FILE_SERVER_CLIENT_HOST}:{FILE_SERVER_PORT}")

# Type Aliases for callback functions passed to handlers
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
DBAddTaskFunc = Callable[[str, str, str], Coroutine[Any, Any, None]]
DBGetMessagesFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, Any]]]]
DBDeleteTaskFunc = Callable[[str], Coroutine[Any, Any, bool]]
DBRenameTaskFunc = Callable[[str, str], Coroutine[Any, Any, bool]]
GetArtifactsFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, str]]]] 
ExecuteShellCommandFunc = Callable[[str, str, SendWSMessageFunc, DBAddMessageFunc, Optional[str]], Coroutine[Any, Any, bool]]


async def read_stream_for_shell(stream, stream_name, session_id, send_ws_message_func, db_add_message_func, current_task_id):
    """Helper function to read from a process stream (stdout/stderr) and send to client."""
    log_prefix_base = f"[{session_id[:8]}]" 
    while True:
        try:
            line_bytes = await stream.readline()
            if not line_bytes: break 
            line_content = line_bytes.decode(errors='replace').rstrip()
            log_content_for_monitor = f"[{stream_name}] {line_content}"
            timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
            await send_ws_message_func("monitor_log", {"text": f"[{timestamp}]{log_prefix_base} {log_content_for_monitor}", "log_source": f"SHELL_{stream_name.upper()}"})
            if current_task_id: 
                try:
                    await db_add_message_func(current_task_id, session_id, f"monitor_shell_{stream_name}", line_content)
                except Exception as db_err:
                    logger.error(f"[{session_id}] Failed to save shell {stream_name} log to DB: {db_err}")
        except asyncio.CancelledError:
            logger.info(f"[{session_id}] read_stream_for_shell ({stream_name}) cancelled.")
            break
        except Exception as e:
            logger.error(f"[{session_id}] Error reading shell stream {stream_name}: {e}")
            break
    logger.debug(f"[{session_id}] Shell {stream_name} stream finished.")


async def execute_shell_command(command: str, session_id: str, send_ws_message_func: SendWSMessageFunc, db_add_message_func: DBAddMessageFunc, current_task_id: Optional[str]) -> bool:
    """Executes a shell command and streams its output. (Used by process_run_command)"""
    # This function's logic remains unchanged.
    log_prefix_base = f"[{session_id[:8]}]"; timestamp_start = datetime.datetime.now().isoformat(timespec='milliseconds')
    start_log_content = f"[Direct Command] Executing: {command}"; logger.info(f"[{session_id}] {start_log_content}")
    await send_ws_message_func("monitor_log", {"text": f"[{timestamp_start}]{log_prefix_base} {start_log_content}", "log_source": "SHELL_COMMAND_START"})
    if current_task_id: await db_add_message_func(current_task_id, session_id, "monitor_direct_cmd_start", command)
    process = None; success = False; status_msg = "failed"; return_code = -1
    if not current_task_id: logger.error(f"[{session_id}] Cannot execute shell command: current_task_id is not set."); await send_ws_message_func("status_message", {"text": "Error: No active task for shell command.", "isError": True}); return False
    try: task_workspace = get_task_workspace_path(current_task_id, create_if_not_exists=True); cwd = str(task_workspace.resolve()); logger.info(f"[{session_id}] Shell command CWD: {cwd}")
    except (ValueError, OSError) as e: logger.error(f"[{session_id}] Error setting CWD for shell command (task: {current_task_id}): {e}"); await send_ws_message_func("status_message", {"text": f"Error setting workspace for shell command: {e}", "isError": True}); return False
    try:
        process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd)
        stdout_task = asyncio.create_task(read_stream_for_shell(process.stdout, "stdout", session_id, send_ws_message_func, db_add_message_func, current_task_id))
        stderr_task = asyncio.create_task(read_stream_for_shell(process.stderr, "stderr", session_id, send_ws_message_func, db_add_message_func, current_task_id))
        TIMEOUT_SECONDS = settings.direct_command_timeout; proc_wait_task = asyncio.create_task(process.wait())
        done, pending = await asyncio.wait([stdout_task, stderr_task, proc_wait_task], timeout=TIMEOUT_SECONDS, return_when=asyncio.ALL_COMPLETED)
        if proc_wait_task not in done: status_msg = f"failed (Timeout after {TIMEOUT_SECONDS}s)"; success = False; [t.cancel() for t in pending]; await asyncio.gather(*pending, return_exceptions=True)
        else: return_code = proc_wait_task.result(); await asyncio.gather(stdout_task, stderr_task, return_exceptions=True); success = return_code == 0; status_msg = "succeeded" if success else f"failed (Code: {return_code})"
    except FileNotFoundError: cmd_part = command.split()[0] if command else "Unknown"; status_msg = f"failed (Command Not Found: {cmd_part})"; success = False
    except Exception as e: status_msg = f"failed ({type(e).__name__})"; logger.error(f"[{session_id}] Error running direct command '{command}': {e}", exc_info=True); success = False
    finally:
        if process and process.returncode is None:
            try: process.terminate(); await process.wait()
            except (ProcessLookupError, Exception): pass
    timestamp_end = datetime.datetime.now().isoformat(timespec='milliseconds'); finish_log_content = f"[Direct Command] Finished '{command[:60]}...', {status_msg}."
    await send_ws_message_func("monitor_log", {"text": f"[{timestamp_end}]{log_prefix_base} {finish_log_content}", "log_source": "SHELL_COMMAND_END"})
    if current_task_id: await db_add_message_func(current_task_id, session_id, "monitor_direct_cmd_end", f"Command: {command} | Status: {status_msg}")
    if not success and status_msg.startswith("failed"): await send_ws_message_func("status_message", {"text": f"Error: Direct command {status_msg}", "isError": True})
    return success

async def handle_workspace_file(request: web.Request) -> web.Response:
    # This function's logic remains unchanged.
    task_id = request.match_info.get('task_id'); filename = request.match_info.get('filename'); session_id = request.headers.get("X-Session-ID", "unknown_file_request_session")
    if not task_id or not filename: raise web.HTTPBadRequest(text="Task ID and filename required")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", task_id): raise web.HTTPForbidden(text="Invalid task ID format.")
    if ".." in filename or filename.startswith(("/", "\\")): raise web.HTTPForbidden(text="Invalid filename path components.")
    try:
        task_workspace = get_task_workspace_path(task_id, create_if_not_exists=False); safe_filename = Path(filename).name; file_path = (task_workspace / safe_filename).resolve()
        if not file_path.is_relative_to(BASE_WORKSPACE_ROOT.resolve()): raise web.HTTPForbidden(text="Access denied - outside base workspace.")
    except Exception as e: raise web.HTTPInternalServerError(text="Error validating file path")
    if not file_path.is_file(): raise web.HTTPNotFound(text=f"File not found: {filename}")
    return FileResponse(path=file_path)

def sanitize_filename(filename: str) -> str:
    # This function's logic remains unchanged.
    if not filename: return f"uploaded_file_{uuid.uuid4().hex[:8]}"
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii'); filename = re.sub(r'[^\w\s.-]', '', filename).strip()
    filename = re.sub(r'\s+', '_', filename); filename = filename.strip('._-'); return Path(filename).name or f"uploaded_file_{uuid.uuid4().hex[:8]}"

async def handle_file_upload(request: web.Request) -> web.Response:
    # This function's logic remains unchanged.
    task_id = request.match_info.get('task_id'); session_id_from_header = request.headers.get("X-Session-ID", "unknown_upload_session")
    if not task_id: return web.json_response({'status': 'error', 'message': 'Task ID required'}, status=400)
    if not re.match(r"^[a-zA-Z0-9_.-]+$", task_id): return web.json_response({'status': 'error', 'message': 'Invalid task ID format'}, status=400)
    try: task_workspace = get_task_workspace_path(task_id, create_if_not_exists=True)
    except Exception as e: return web.json_response({'status': 'error', 'message': 'Error accessing/creating task workspace'}, status=500)
    reader = await request.multipart(); saved_files = []; errors = []
    if not reader: return web.json_response({'status': 'error', 'message': 'No multipart data received'}, status=400)
    while True:
        part = await reader.next();
        if part is None: break
        if part.name == 'file' and part.filename:
            original_filename = part.filename; safe_filename = sanitize_filename(original_filename); save_path = (task_workspace / safe_filename).resolve()
            if not save_path.is_relative_to(task_workspace.resolve()): errors.append({'filename': original_filename, 'message': 'Invalid file path.'}); continue
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(save_path, 'wb') as f:
                    while True: chunk = await part.read_chunk(); await f.write(chunk) if chunk else f.close(); break
                saved_files.append({'filename': safe_filename})
                target_session_id = next((sid for sid, sdata in session_data.items() if sdata.get("current_task_id") == task_id), None)
                if target_session_id:
                    await add_message(task_id, target_session_id, "artifact_generated", safe_filename)
                    client_info_for_ws = connected_clients.get(target_session_id); send_func = client_info_for_ws.get("send_ws_message") if client_info_for_ws else None
                    if send_func: await send_func("trigger_artifact_refresh", {"taskId": task_id})
            except Exception as e: errors.append({'filename': original_filename, 'message': f'Server error saving file: {type(e).__name__}'})
    if errors: return web.json_response({'status': 'error', 'errors': errors, 'saved': saved_files}, status=207 if saved_files else 400)
    return web.json_response({'status': 'success', 'saved': saved_files}, status=200) if saved_files else web.json_response({'status': 'error', 'message': 'No valid files uploaded.'}, status=400)

async def get_artifacts(task_id: str) -> List[Dict[str, str]]:
    # This function's logic remains unchanged.
    artifacts = []; task_workspace_path = get_task_workspace_path(task_id, create_if_not_exists=False)
    if not task_workspace_path.exists(): return []
    artifact_patterns = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.pdf'] + [f'*{ext}' for ext in TEXT_EXTENSIONS]
    all_potential_artifacts = sorted([p for pattern in artifact_patterns for p in task_workspace_path.glob(pattern) if p.is_file()], key=os.path.getmtime, reverse=True)
    for file_path in all_potential_artifacts:
        relative_filename = str(file_path.relative_to(task_workspace_path)); artifact_type = 'unknown'; file_suffix = file_path.suffix.lower()
        if file_suffix in ['.png', '.jpg', '.jpeg', '.gif', '.svg']: artifact_type = 'image'
        elif file_suffix in TEXT_EXTENSIONS: artifact_type = 'text'
        elif file_suffix == '.pdf': artifact_type = 'pdf'
        if artifact_type != 'unknown': encoded_filename = urllib.parse.quote(relative_filename); artifact_url = f"http://{FILE_SERVER_CLIENT_HOST}:{FILE_SERVER_PORT}/workspace_files/{task_id}/{encoded_filename}"; artifacts.append({"type": artifact_type, "url": artifact_url, "filename": relative_filename})
    return artifacts

async def setup_file_server():
    # This function's logic remains unchanged.
    app = web.Application(); app['client_max_size'] = 100 * 1024**2
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods=["GET", "POST", "OPTIONS"])})
    get_resource = app.router.add_resource('/workspace_files/{task_id}/{filename:.+}'); cors.add(get_resource.add_route('GET', handle_workspace_file))
    post_resource = app.router.add_resource('/upload/{task_id}'); cors.add(post_resource.add_route('POST', handle_file_upload))
    runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, FILE_SERVER_LISTEN_HOST, FILE_SERVER_PORT); logger.info(f"Starting file server listening on http://{FILE_SERVER_LISTEN_HOST}:{FILE_SERVER_PORT}")
    return site, runner

async def handler(websocket: Any): 
    """Handles incoming WebSocket connections and messages."""
    session_id = str(uuid.uuid4())
    logger.info(f"[{session_id}] Connection attempt from {websocket.remote_address}...")

    async def send_ws_message_for_session(msg_type: str, content: Any):
        # This function's logic remains unchanged.
        client_info = connected_clients.get(session_id); ws = client_info.get("websocket") if client_info else None
        if ws: await ws.send(json.dumps({"type": msg_type, "content": content}))

    connected_clients[session_id] = {"websocket": websocket, "agent_task": None, "send_ws_message": send_ws_message_for_session}

    async def add_monitor_log_and_save(text: str, log_source: str = "SYSTEM_EVENT"): 
        # This function's logic remains unchanged.
        timestamp = datetime.datetime.now().isoformat(timespec='milliseconds'); log_prefix_ui = f"[{timestamp}][{session_id[:8]}]"
        type_indicator_ui = f"[{log_source.upper().replace('_', '-')}]"; full_content_for_ui = f"{log_prefix_ui} {type_indicator_ui} {text}"
        await send_ws_message_for_session("monitor_log", {"text": full_content_for_ui, "log_source": log_source}) 
        active_task_id = session_data.get(session_id, {}).get("current_task_id")
        if active_task_id: await add_message(active_task_id, session_id, log_source, text)

    try:
        ws_callback_handler = WebSocketCallbackHandler(session_id, send_ws_message_for_session, functools.partial(add_message), session_data)
        session_data[session_id] = {
            "memory": ConversationBufferWindowMemory(k=settings.agent_memory_window_k, memory_key="chat_history", input_key="input", output_key="output", return_messages=True),
            "callback_handler": ws_callback_handler, "current_task_id": None, "selected_llm_provider": settings.executor_default_provider, 
            "selected_llm_model_name": settings.executor_default_model_name, "cancellation_requested": False,
            "session_intent_classifier_llm_id": None, "session_planner_llm_id": None, "session_controller_llm_id": None, "session_evaluator_llm_id": None,
        }
    except Exception as e: logger.error(f"[{session_id}] CRITICAL ERROR during session setup: {e}", exc_info=True); return

    message_handler_map: Dict[str, MessageHandler] = {
        "context_switch": process_context_switch, "user_message": process_user_message, "new_task": process_new_task,
        "delete_task": process_delete_task, "rename_task": process_rename_task, "set_llm": process_set_llm, 
        "get_available_models": process_get_available_models, "cancel_agent": process_cancel_agent,
        "get_artifacts_for_task": process_get_artifacts_for_task, "trigger_artifact_refresh": process_get_artifacts_for_task, 
        "run_command": process_run_command, "action_command": process_action_command, "set_session_role_llm": process_set_session_role_llm, 
    }

    try:
        await send_ws_message_for_session("status_message", {"text": f"Connected (Session: {session_id[:8]}). Agent Ready.", "component_hint": "SYSTEM"})
        # ... (rest of try block logic for message loop)
        async for message_str in websocket:
            try:
                parsed_data = json.loads(message_str) 
                message_type = parsed_data.get("type")
                handler_func = message_handler_map.get(message_type)

                if handler_func:
                    current_session_data_entry = session_data.get(session_id)
                    current_connected_clients_entry = connected_clients.get(session_id)

                    if not current_session_data_entry or not current_connected_clients_entry:
                        logger.error(f"[{session_id}] Critical: session_data or connected_clients entry missing for active session. Type: {message_type}")
                        await send_ws_message_for_session("status_message", {"text":"Error: Session integrity issue. Please refresh.", "isError": True})
                        continue 

                    handler_args: Dict[str, Any] = { 
                        "session_id": session_id, "data": parsed_data,
                        "session_data_entry": current_session_data_entry,
                        "connected_clients_entry": current_connected_clients_entry,
                        "send_ws_message_func": send_ws_message_for_session,
                        "add_monitor_log_func": add_monitor_log_and_save,
                    }
                    
                    # --- Argument Population logic is simplified as we no longer pass the graph ---
                    if message_type in ["user_message", "run_command"]:
                        handler_args["db_add_message_func"] = add_message
                    if message_type == "context_switch":
                        handler_args.update({"db_add_task_func": add_task, "db_get_messages_func": get_messages_for_task, "get_artifacts_func": get_artifacts})
                    elif message_type == "new_task":
                        handler_args["db_add_task_func"] = add_task
                    elif message_type == "delete_task":
                        handler_args.update({"db_delete_task_func": delete_task_and_messages, "get_artifacts_func": get_artifacts})
                    elif message_type == "rename_task":
                        handler_args["db_rename_task_func"] = rename_task_in_db
                    elif message_type in ["get_artifacts_for_task", "trigger_artifact_refresh"]:
                        handler_args["get_artifacts_func"] = get_artifacts
                    elif message_type == "run_command":
                        handler_args["execute_shell_command_func"] = execute_shell_command
                    
                    # No longer need to pass the LangGraph object
                    # if message_type in ["user_message", "execute_confirmed_plan"]:
                    #     handler_args["research_agent_lg_graph"] = compiled_research_agent_graph

                    await handler_func(**handler_args) 
                else: 
                    logger.warning(f"[{session_id}] Unknown message type received: {message_type}")

            except json.JSONDecodeError:
                logger.error(f"[{session_id}] Received non-JSON message: {message_str[:200]}")
            except Exception as e:
                mt = message_type if 'message_type' in locals() else 'UnknownType'
                logger.error(f"[{session_id}] Error processing message (type: {mt}): {e}", exc_info=True)
                await send_ws_message_for_session("status_message", {"text": f"Error processing message: {type(e).__name__}", "isError": True})
    
    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as ws_close_exc:
        logger.info(f"Client disconnected: {websocket.remote_address} (Session: {session_id})")
    except Exception as e: 
        logger.error(f"Unhandled error in WebSocket handler: {websocket.remote_address} (Session: {session_id}): {e}", exc_info=True)
    finally:
        # Cleanup logic remains the same
        logger.info(f"Cleaning up resources for session {session_id}")
        agent_task = connected_clients.get(session_id, {}).get("agent_task")
        if agent_task and not agent_task.done():
            agent_task.cancel()
            try: await agent_task 
            except (asyncio.CancelledError, AgentCancelledException): pass
        if session_id in connected_clients: del connected_clients[session_id]
        if session_id in session_data: del session_data[session_id]
        logger.info(f"Cleaned up session data for {session_id}. Active clients: {len(connected_clients)}")

async def main():
    # This function's logic remains unchanged.
    await init_db(); file_server_site, file_server_runner = await setup_file_server(); await file_server_site.start()
    ws_host = "0.0.0.0"; ws_port = 8765; shutdown_event = asyncio.Event()
    websocket_server = await websockets.serve(handler, ws_host, ws_port, max_size=settings.websocket_max_size_bytes, ping_interval=settings.websocket_ping_interval, ping_timeout=settings.websocket_ping_timeout)
    logger.info(f"Servers running (WS: ws://{ws_host}:{ws_port}, File: http://{FILE_SERVER_LISTEN_HOST}:{FILE_SERVER_PORT}). Press Ctrl+C to stop.")
    loop = asyncio.get_running_loop(); original_sigint = signal.getsignal(signal.SIGINT); original_sigterm = signal.getsignal(signal.SIGTERM)
    def sig_handler(sig, frame): shutdown_event.set(); signal.signal(signal.SIGINT, original_sigint); signal.signal(signal.SIGTERM, original_sigterm)
    try: loop.add_signal_handler(signal.SIGINT, sig_handler, signal.SIGINT, None); loop.add_signal_handler(signal.SIGTERM, sig_handler, signal.SIGTERM, None)
    except NotImplementedError: logger.warning("Signal handlers for graceful shutdown not fully supported.")
    await shutdown_event.wait(); logger.info("Shutdown signal received. Stopping servers...")
    websocket_server.close(); await websocket_server.wait_closed(); logger.info("WebSocket server stopped.")
    await file_server_runner.cleanup(); logger.info("File server stopped.")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]; [t.cancel() for t in tasks]
    await asyncio.gather(*tasks, return_exceptions=True); logger.info("Shutdown complete.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message=".*LangSmith API key.*")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually (KeyboardInterrupt).")
    except Exception as e:
        logging.critical(f"Server failed to start or crashed: {e}", exc_info=True)
        exit(1)
