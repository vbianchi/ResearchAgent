# backend/server.py
import asyncio
import websockets
import json
import datetime
import logging
import shlex
import uuid
from typing import Optional, List, Dict, Any, Set, Tuple, Callable, Coroutine
from pathlib import Path
import os
import signal
import re
import functools
import warnings
import aiofiles
import unicodedata
import urllib.parse

# --- Web Server Imports ---
from aiohttp import web
from aiohttp.web import FileResponse
import aiohttp_cors
# -------------------------

# LangChain Imports
# Removed: from langchain.agents import AgentExecutor # No longer directly used here
# Removed: from langchain.memory import ConversationBufferWindowMemory # Handled in session_data
from langchain_core.messages import AIMessage, HumanMessage # Still used for session_data['memory'] type hints if any
# Removed: from langchain_core.agents import AgentAction, AgentFinish # Handled by callbacks/graph
from langchain_core.runnables import RunnableConfig # Potentially used by handlers
from langchain_core.language_models.base import BaseLanguageModel # For LLM startup check
from langchain_core.language_models.chat_models import BaseChatModel # For LLM startup check

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools, get_task_workspace_path, BASE_WORKSPACE_ROOT, TEXT_EXTENSIONS
# Removed: from backend.agent import create_agent_executor # LangGraph replaces this
from backend.callbacks import WebSocketCallbackHandler, AgentCancelledException
from backend.db_utils import (
    init_db, add_task, add_message, get_messages_for_task,
    delete_task_and_messages, rename_task_in_db
)
# Removed: from backend.planner import generate_plan, PlanStep # Handled by agent_flow_handlers
# Removed: from backend.controller import validate_and_prepare_step_action # Handled by agent_flow_handlers
from backend.message_handlers import ( # These now import from message_processing
    process_context_switch, process_user_message,
    process_execute_confirmed_plan, process_new_task,
    process_delete_task, process_rename_task,
    process_set_llm, process_get_available_models,
    process_cancel_agent, process_get_artifacts_for_task,
    process_run_command, process_action_command,
    process_set_session_role_llm
)
# ADDED: Import the compiled LangGraph application
from backend.langgraph_agent import research_agent_graph as compiled_research_agent_graph

# ----------------------

# Define Type Aliases for callback functions used in MessageHandler hint
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
DBAddTaskFunc = Callable[[str, str, str], Coroutine[Any, Any, None]]
DBGetMessagesFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, Any]]]]
DBDeleteTaskFunc = Callable[[str], Coroutine[Any, Any, bool]]
DBRenameTaskFunc = Callable[[str, str], Coroutine[Any, Any, bool]]
GetArtifactsFunc = Callable[[str], Coroutine[Any, Any, List[Dict[str, str]]]]
ExecuteShellCommandFunc = Callable[[str, str, SendWSMessageFunc, DBAddMessageFunc, Optional[str]], Coroutine[Any, Any, bool]]


log_level = settings.log_level
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
)
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to {log_level}")

try:
    default_llm_instance_for_startup_checks: BaseLanguageModel = get_llm(settings, provider=settings.default_provider, model_name=settings.default_model_name)
    logger.info(f"Default Base LLM for startup checks initialized successfully: {settings.default_provider}::{settings.default_model_name}")
except Exception as llm_e:
    logging.critical(f"FATAL: Failed during startup LLM initialization: {llm_e}", exc_info=True)
    exit(1)

# Global dictionary to store connected clients and their associated data
# Key: session_id (str), Value: Dict containing websocket object, agent_task, send_ws_message func
connected_clients: Dict[str, Dict[str, Any]] = {}

# Global dictionary to store session-specific data (memory, callback_handler, task_id, etc.)
# Key: session_id (str), Value: Dict containing session data
session_data: Dict[str, Dict[str, Any]] = {}

# File Server Configuration
FILE_SERVER_LISTEN_HOST = "0.0.0.0"  # Listen on all interfaces
FILE_SERVER_CLIENT_HOST = settings.file_server_hostname # Hostname client uses to connect
FILE_SERVER_PORT = 8766
logger.info(f"File server will listen on {FILE_SERVER_LISTEN_HOST}:{FILE_SERVER_PORT}")
logger.info(f"File server URLs constructed for client will use: http://{FILE_SERVER_CLIENT_HOST}:{FILE_SERVER_PORT}")


async def read_stream(stream, stream_name, session_id, send_ws_message_func, db_add_message_func, current_task_id):
    """Helper function to read from a process stream (stdout/stderr) and send to client."""
    log_prefix_base = f"[{session_id[:8]}]" # Short session ID for brevity
    while True:
        try:
            line = await stream.readline()
        except Exception as e:
            logger.error(f"[{session_id}] Error reading stream {stream_name}: {e}")
            break
        if not line:
            break # End of stream
        line_content = line.decode(errors='replace').rstrip()
        log_content = f"[{stream_name}] {line_content}"
        timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
        await send_ws_message_func("monitor_log", f"[{timestamp}]{log_prefix_base} {log_content}")
        if current_task_id: # Only save to DB if a task is active
            try:
                await db_add_message_func(current_task_id, session_id, f"monitor_{stream_name}", line_content)
            except Exception as db_err:
                logger.error(f"[{session_id}] Failed to save {stream_name} log to DB: {db_err}")
    logger.debug(f"[{session_id}] {stream_name} stream finished.")


async def execute_shell_command(command: str, session_id: str, send_ws_message_func: SendWSMessageFunc, db_add_message_func: DBAddMessageFunc, current_task_id: Optional[str]) -> bool:
    """Executes a shell command and streams its output."""
    log_prefix_base = f"[{session_id[:8]}]"
    timestamp_start = datetime.datetime.now().isoformat(timespec='milliseconds')
    start_log_content = f"[Direct Command] Executing: {command}"
    logger.info(f"[{session_id}] {start_log_content}")
    await send_ws_message_func("monitor_log", f"[{timestamp_start}]{log_prefix_base} {start_log_content}")
    if current_task_id:
        try:
            await db_add_message_func(current_task_id, session_id, "monitor_direct_cmd_start", command)
        except Exception as db_err:
            logger.error(f"[{session_id}] Failed to save direct cmd start to DB: {db_err}")

    process = None
    success = False
    status_msg = "failed"
    return_code = -1
    cwd = str(BASE_WORKSPACE_ROOT.resolve()) # Always run direct commands from the root workspace for now
    logger.info(f"[{session_id}] Direct command CWD: {cwd}")

    try:
        # Using asyncio.create_subprocess_shell for direct command execution
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd # Set current working directory
        )

        # Asynchronously read stdout and stderr
        stdout_task = asyncio.create_task(read_stream(process.stdout, "stdout", session_id, send_ws_message_func, db_add_message_func, current_task_id))
        stderr_task = asyncio.create_task(read_stream(process.stderr, "stderr", session_id, send_ws_message_func, db_add_message_func, current_task_id))

        # Wait for the process to complete with a timeout
        TIMEOUT_SECONDS = settings.direct_command_timeout
        proc_wait_task = asyncio.create_task(process.wait())
        done, pending = await asyncio.wait(
            [stdout_task, stderr_task, proc_wait_task],
            timeout=TIMEOUT_SECONDS,
            return_when=asyncio.ALL_COMPLETED
        )

        if proc_wait_task not in done:
            logger.error(f"[{session_id}] Timeout executing direct command: {command}")
            status_msg = f"failed (Timeout after {TIMEOUT_SECONDS}s)"
            success = False
            if process and process.returncode is None: # Process still running
                try: process.terminate()
                except ProcessLookupError: pass # Process already exited
                await process.wait() # Ensure termination
            for task_to_cancel in pending: task_to_cancel.cancel() # Cancel pending stream readers
            await asyncio.gather(*pending, return_exceptions=True) # Wait for cancellations
        else: # Process completed within timeout
            return_code = proc_wait_task.result()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True) # Ensure stream readers finish
            success = return_code == 0
            status_msg = "succeeded" if success else f"failed (Code: {return_code})"

    except FileNotFoundError:
        cmd_part = command.split()[0] if command else "Unknown"
        status_msg = f"failed (Command Not Found: {cmd_part})"
        logger.warning(f"[{session_id}] Direct command not found: {cmd_part}")
        success = False
    except Exception as e:
        status_msg = f"failed ({type(e).__name__})"
        logger.error(f"[{session_id}] Error running direct command '{command}': {e}", exc_info=True)
        success = False
    finally:
        # Ensure process is terminated if it's still running (e.g., due to an unhandled exception before timeout logic)
        if process and process.returncode is None:
            logger.warning(f"[{session_id}] Direct command process '{command}' still running in finally block, attempting termination.")
            try:
                process.terminate()
                await process.wait()
            except ProcessLookupError: pass # Process already exited
            except Exception as term_e:
                logger.error(f"[{session_id}] Error during final termination of direct command process: {term_e}")

    timestamp_end = datetime.datetime.now().isoformat(timespec='milliseconds')
    finish_log_content = f"[Direct Command] Finished '{command[:60]}...', {status_msg}."
    await send_ws_message_func("monitor_log", f"[{timestamp_end}]{log_prefix_base} {finish_log_content}")
    if current_task_id:
        try:
            await db_add_message_func(current_task_id, session_id, "monitor_direct_cmd_end", f"Command: {command} | Status: {status_msg}")
        except Exception as db_err:
            logger.error(f"[{session_id}] Failed to save direct cmd end to DB: {db_err}")

    if not success and status_msg.startswith("failed"):
        await send_ws_message_func("status_message", f"Error: Direct command {status_msg}")
    return success


async def handle_workspace_file(request: web.Request) -> web.Response:
    """Handles GET requests for files in a task's workspace."""
    task_id = request.match_info.get('task_id')
    filename = request.match_info.get('filename') # This will be URL-encoded by browser, aiohttp decodes it
    session_id = request.headers.get("X-Session-ID", "unknown_file_request_session") # For logging, if available

    logger.info(f"[{session_id}] File server: Received request for task_id='{task_id}', filename='{filename}'")

    if not task_id or not filename:
        logger.warning(f"[{session_id}] File server request missing task_id or filename.")
        raise web.HTTPBadRequest(text="Task ID and filename required")

    # Basic validation for task_id format (alphanumeric, underscore, hyphen, dot)
    if not re.match(r"^[a-zA-Z0-9_.-]+$", task_id):
        logger.error(f"[{session_id}] Invalid task_id format rejected: {task_id}")
        raise web.HTTPForbidden(text="Invalid task ID format.")

    # Security: Prevent path traversal. Filename should not contain '..' or start with '/' or '\'
    # Path.name will extract the final component, but we also check the raw input.
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        logger.error(f"[{session_id}] Invalid filename path components detected: {filename}")
        raise web.HTTPForbidden(text="Invalid filename path components.")

    try:
        # Use get_task_workspace_path to ensure consistent path resolution and creation logic (though here we don't create)
        task_workspace = get_task_workspace_path(task_id, create_if_not_exists=False)
        # Ensure the filename is just the name, not a path itself, to prevent deeper traversal if task_workspace is miscalculated
        safe_filename = Path(filename).name # Extracts the final component, mitigating some path issues

        # Construct the full path and resolve symlinks, etc.
        file_path = (task_workspace / safe_filename).resolve()
        logger.debug(f"[{session_id}] File server: Resolved file path: {file_path}")

        # Crucial security check: Ensure the resolved path is still within the BASE_WORKSPACE_ROOT
        if not file_path.is_relative_to(BASE_WORKSPACE_ROOT.resolve()):
            logger.error(f"[{session_id}] Security Error: Access attempt outside base workspace! Req: {file_path}, Base: {BASE_WORKSPACE_ROOT.resolve()}")
            raise web.HTTPForbidden(text="Access denied - outside base workspace.")

    except ValueError as ve: # From get_task_workspace_path if task_id is invalid
        logger.error(f"[{session_id}] File server: Invalid task_id for file access: {ve}")
        raise web.HTTPBadRequest(text=f"Invalid task ID: {ve}")
    except OSError as e: # From get_task_workspace_path if there's an OS error
        logger.error(f"[{session_id}] File server: Error resolving task workspace for file access: {e}")
        raise web.HTTPInternalServerError(text="Error accessing task workspace.")
    except Exception as e: # Catch-all for other path validation issues
        logger.error(f"[{session_id}] File server: Unexpected error validating file path: {e}. Req: {filename}", exc_info=True)
        raise web.HTTPInternalServerError(text="Error validating file path")


    if not file_path.is_file():
        logger.warning(f"[{session_id}] File server: File not found request: {file_path}")
        raise web.HTTPNotFound(text=f"File not found: {filename}")

    logger.info(f"[{session_id}] File server: Serving file: {file_path}")
    return FileResponse(path=file_path)


def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename to prevent path traversal and invalid characters."""
    if not filename: # Handle empty original filename
        return f"uploaded_file_{uuid.uuid4().hex[:8]}"

    # Normalize Unicode characters to their closest ASCII representation
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Remove characters that are not alphanumeric, underscores, hyphens, periods, or spaces
    filename = re.sub(r'[^\w\s.-]', '', filename).strip()
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    # If filename becomes empty after sanitization, generate a unique name
    if not filename:
        filename = f"uploaded_file_{uuid.uuid4().hex[:8]}"
    # Ensure it's not just dots or dashes
    filename = filename.strip('._-')
    if not filename: # Check again if stripping made it empty
        filename = f"uploaded_file_{uuid.uuid4().hex[:8]}"
    # Return only the filename part, no path components
    return Path(filename).name


async def handle_file_upload(request: web.Request) -> web.Response:
    """Handles file uploads to a specific task's workspace."""
    task_id = request.match_info.get('task_id')
    session_id = request.headers.get("X-Session-ID", "unknown_upload_session") # For logging
    logger.info(f"[{session_id}] File Upload: Received request for task_id: '{task_id}'")

    if not task_id:
        logger.error(f"[{session_id}] File Upload: Missing task_id.")
        return web.json_response({'status': 'error', 'message': 'Task ID required'}, status=400)

    if not re.match(r"^[a-zA-Z0-9_.-]+$", task_id): # Basic validation
        logger.error(f"[{session_id}] File Upload: Invalid task_id format: '{task_id}'")
        return web.json_response({'status': 'error', 'message': 'Invalid task ID format'}, status=400)

    task_workspace: Path
    try:
        task_workspace = get_task_workspace_path(task_id, create_if_not_exists=True)
        logger.info(f"[{session_id}] File Upload: Ensured task workspace exists at: {task_workspace}")
    except ValueError as ve:
        logger.error(f"[{session_id}] File Upload: Invalid task_id for workspace creation: {ve}")
        return web.json_response({'status': 'error', 'message': f'Invalid task ID for workspace: {ve}'}, status=400)
    except OSError as e:
        logger.error(f"[{session_id}] File Upload: Error getting/creating workspace for task {task_id}: {e}", exc_info=True)
        return web.json_response({'status': 'error', 'message': 'Error accessing/creating task workspace'}, status=500)

    reader = None
    saved_files = []
    errors = []

    try:
        reader = await request.multipart()
    except Exception as e: # Catch errors during multipart reading setup
        logger.error(f"[{session_id}] File Upload: Error reading multipart form data for task {task_id}: {e}", exc_info=True)
        return web.json_response({'status': 'error', 'message': f'Failed to read upload data: {e}'}, status=400)

    if not reader: # Should not happen if await request.multipart() succeeded, but good check
        return web.json_response({'status': 'error', 'message': 'No multipart data received'}, status=400)

    # Iterate over parts of the multipart form
    while True:
        part = await reader.next() # Read the next part
        if part is None: # No more parts
            logger.debug(f"[{session_id}] File Upload: Finished processing multipart parts for task {task_id}.")
            break

        if part.name == 'file' and part.filename: # Check if it's a file part and has a filename
            original_filename = part.filename
            safe_filename = sanitize_filename(original_filename) # Sanitize the filename

            # Construct the full save path and resolve it
            save_path = (task_workspace / safe_filename).resolve()
            logger.info(f"[{session_id}] File Upload: Processing uploaded file: '{original_filename}' -> '{safe_filename}' for task {task_id}. Target path: {save_path}")

            # Security check: ensure the save path is within the designated task workspace
            if not save_path.is_relative_to(task_workspace.resolve()):
                logger.error(f"[{session_id}] File Upload: Security Error - Upload path resolves outside task workspace! Task: {task_id}, Orig: '{original_filename}', Safe: '{safe_filename}', Resolved: {save_path}, Workspace: {task_workspace.resolve()}")
                errors.append({'filename': original_filename, 'message': 'Invalid file path detected (path traversal attempt).'})
                continue # Skip this file

            try:
                # Ensure parent directory exists (though task_workspace should exist)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                logger.debug(f"[{session_id}] File Upload: Attempting to open {save_path} for writing.")
                async with aiofiles.open(save_path, 'wb') as f:
                    while True:
                        chunk = await part.read_chunk() # Read chunk of the file
                        if not chunk:
                            break
                        await f.write(chunk)
                logger.info(f"[{session_id}] File Upload: Successfully saved uploaded file to: {save_path}")
                saved_files.append({'filename': safe_filename}) # Use the sanitized name

                # --- Notify relevant client session about the new artifact ---
                target_session_id_for_ws_notify = None
                # Find the session_id that has this task_id active
                for sid_iter, sdata_val_iter in session_data.items(): # Iterate through all active sessions
                    if sdata_val_iter.get("current_task_id") == task_id:
                        target_session_id_for_ws_notify = sid_iter
                        logger.debug(f"[{session_id}] File Upload: Found active session {target_session_id_for_ws_notify} for task {task_id} for WS notification.")
                        break

                if target_session_id_for_ws_notify:
                    try:
                        # Log artifact generation to DB for the target session
                        await add_message(task_id, target_session_id_for_ws_notify, "artifact_generated", safe_filename)
                        logger.info(f"[{session_id}] File Upload: Saved 'artifact_generated' message to DB for {safe_filename} (session: {target_session_id_for_ws_notify}).")

                        # Send WebSocket message to trigger artifact refresh for the target session
                        client_info_for_ws = connected_clients.get(target_session_id_for_ws_notify)
                        if client_info_for_ws:
                            send_func_for_ws = client_info_for_ws.get("send_ws_message")
                            if send_func_for_ws:
                                logger.info(f"[{target_session_id_for_ws_notify}] File Upload: Sending trigger_artifact_refresh for task {task_id}")
                                await send_func_for_ws("trigger_artifact_refresh", {"taskId": task_id})
                            else:
                                logger.warning(f"[{session_id}] File Upload: Send function not found for target session {target_session_id_for_ws_notify} to send refresh trigger.")
                        else:
                            logger.warning(f"[{session_id}] File Upload: Target session {target_session_id_for_ws_notify} not found in connected_clients for WS notification.")
                    except Exception as db_log_err:
                        logger.error(f"[{session_id}] File Upload: Error during DB logging or WS notification after file upload for {safe_filename}: {db_log_err}", exc_info=True)
                else:
                    logger.warning(f"[{session_id}] File Upload: Could not find an active session for task {task_id} to notify about upload of {safe_filename} via WebSocket.")
                # --- End notification ---

            except Exception as e:
                logger.error(f"[{session_id}] File Upload: Error saving uploaded file '{safe_filename}' for task {task_id}: {e}", exc_info=True)
                errors.append({'filename': original_filename, 'message': f'Server error saving file: {type(e).__name__}'})
        else:
            logger.warning(f"[{session_id}] File Upload: Received non-file part or part without filename in upload: Name='{part.name if hasattr(part, 'name') else 'N/A'}', Filename='{part.filename if hasattr(part, 'filename') else 'N/A'}'")

    logger.debug(f"[{session_id}] File Upload: Finished processing all parts. Errors: {len(errors)}, Saved: {len(saved_files)}")

    # Construct and send the final response
    try:
        if errors:
            response_data = {'status': 'error', 'message': 'Some files failed to upload.', 'errors': errors, 'saved': saved_files}
            status_code = 400 if not saved_files else 207 # 207 Multi-Status if some succeeded
            logger.info(f"[{session_id}] File Upload: Returning error/partial success response: Status={status_code}, Data={response_data}")
            return web.json_response(response_data, status=status_code)
        elif not saved_files: # No errors, but no files saved (e.g., no file parts sent)
            response_data = {'status': 'error', 'message': 'No valid files were uploaded.'}
            status_code = 400
            logger.info(f"[{session_id}] File Upload: Returning no valid files error response: Status={status_code}, Data={response_data}")
            return web.json_response(response_data, status=status_code)
        else: # All files uploaded successfully
            response_data = {'status': 'success', 'message': f'Successfully uploaded {len(saved_files)} file(s).', 'saved': saved_files}
            status_code = 200
            logger.info(f"[{session_id}] File Upload: Returning success response: Status={status_code}, Data={response_data}")
            return web.json_response(response_data, status=status_code)
    except Exception as return_err: # Should not happen, but good to catch
        logger.error(f"[{session_id}] File Upload: CRITICAL ERROR constructing final JSON response for upload: {return_err}", exc_info=True)
        return web.Response(status=500, text="Internal server error creating upload response.")


async def get_artifacts(task_id: str) -> List[Dict[str, str]]:
    """Scans the task's workspace for artifacts and returns a list."""
    logger.debug(f"Scanning workspace for artifacts for task: {task_id}")
    artifacts = []
    try:
        task_workspace_path = get_task_workspace_path(task_id, create_if_not_exists=False) # Don't create if just scanning
        if not task_workspace_path.exists():
            logger.warning(f"Artifact scan: Workspace for task {task_id} does not exist at {task_workspace_path}. Returning empty list.")
            return []

        # Define patterns for common artifact types
        artifact_patterns = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.pdf'] + [f'*{ext}' for ext in TEXT_EXTENSIONS]

        all_potential_artifacts = []
        for pattern in artifact_patterns:
            for file_path in task_workspace_path.glob(pattern):
                if file_path.is_file(): # Ensure it's a file, not a directory
                    try:
                        mtime = file_path.stat().st_mtime # Get modification time for sorting
                        all_potential_artifacts.append((file_path, mtime))
                    except FileNotFoundError: # File might have been deleted between glob and stat
                        logger.warning(f"File disappeared during artifact scan: {file_path}")
                        continue

        # Sort files by modification time, newest first
        sorted_files = sorted(all_potential_artifacts, key=lambda x: x[1], reverse=True)

        for file_path, _ in sorted_files:
            relative_filename = str(file_path.relative_to(task_workspace_path))
            artifact_type = 'unknown'
            file_suffix = file_path.suffix.lower()

            if file_suffix in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                artifact_type = 'image'
            elif file_suffix in TEXT_EXTENSIONS:
                artifact_type = 'text'
            elif file_suffix == '.pdf':
                artifact_type = 'pdf'

            if artifact_type != 'unknown':
                # URL-encode the filename part of the URL
                encoded_filename = urllib.parse.quote(relative_filename)
                artifact_url = f"http://{FILE_SERVER_CLIENT_HOST}:{FILE_SERVER_PORT}/workspace_files/{task_id}/{encoded_filename}"
                artifacts.append({"type": artifact_type, "url": artifact_url, "filename": relative_filename})
        logger.info(f"Found {len(artifacts)} artifacts for task {task_id}.")
    except ValueError as ve: # From get_task_workspace_path if task_id is invalid
        logger.error(f"Error scanning artifacts for task {task_id} due to invalid task ID: {ve}")
    except Exception as e:
        logger.error(f"Error scanning artifacts for task {task_id}: {e}", exc_info=True)
    return artifacts


async def setup_file_server():
    """Sets up and returns the aiohttp web application for the file server."""
    app = web.Application()
    app['client_max_size'] = 100 * 1024**2  # 100MB max upload size

    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*", # Allow all headers for simplicity, can be restricted
            allow_methods=["GET", "POST", "OPTIONS"] # Allow necessary methods
        )
    })

    # Route for serving files from workspace
    # The `.+` in filename allows filenames with dots.
    get_resource = app.router.add_resource('/workspace_files/{task_id}/{filename:.+}')
    get_route = get_resource.add_route('GET', handle_workspace_file)
    cors.add(get_route) # Apply CORS to this route

    # Route for uploading files
    post_resource = app.router.add_resource('/upload/{task_id}')
    post_route = post_resource.add_route('POST', handle_file_upload)
    cors.add(post_route) # Apply CORS to this route

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, FILE_SERVER_LISTEN_HOST, FILE_SERVER_PORT)
    logger.info(f"Starting file server listening on http://{FILE_SERVER_LISTEN_HOST}:{FILE_SERVER_PORT}")
    return site, runner # Return both site and runner for cleanup


async def handler(websocket: Any): # websocket: websockets.WebSocketServerProtocol
    """Handles incoming WebSocket connections and messages."""
    session_id = str(uuid.uuid4())
    logger.info(f"[{session_id}] Connection attempt from {websocket.remote_address}...")

    # Define a send_ws_message function specific to this session's websocket
    async def send_ws_message_for_session(msg_type: str, content: Any):
        logger.debug(f"[{session_id}] Attempting to send WS message: Type='{msg_type}', Content='{str(content)[:100]}...'")
        client_info = connected_clients.get(session_id)
        if client_info:
            ws = client_info.get("websocket")
            if ws: # Ensure websocket object exists
                try:
                    await ws.send(json.dumps({"type": msg_type, "content": content}))
                    logger.debug(f"[{session_id}] Successfully sent WS message type '{msg_type}'.")
                except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as close_exc:
                    logger.warning(f"[{session_id}] WS already closed when trying to send type '{msg_type}'. Error: {close_exc}")
                except Exception as e: # Catch other potential errors during send
                    logger.error(f"[{session_id}] Error sending WS message type '{msg_type}': {e}", exc_info=True)
            else:
                logger.warning(f"[{session_id}] Websocket object not found for session when trying to send type '{msg_type}'.")
        else:
            logger.warning(f"[{session_id}] Session not found in connected_clients when trying to send type '{msg_type}'.")


    # Store client information
    connected_clients[session_id] = {"websocket": websocket, "agent_task": None, "send_ws_message": send_ws_message_for_session}
    logger.info(f"[{session_id}] Client added to connected_clients dict with send function.")


    # Define a helper to add to monitor log and save to DB
    async def add_monitor_log_and_save(text: str, log_type: str = "monitor_log"): # Default log_type
        timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
        log_prefix = f"[{timestamp}][{session_id[:8]}]" # Short session ID for UI brevity

        # Determine a more structured type indicator for the log entry
        type_indicator_for_log_entry = f"[{log_type.upper().replace('MONITOR_', '').replace('ERROR_', 'ERR_').replace('SYSTEM_', 'SYS_')}]"
        if log_type == "monitor_log" and not text.startswith("["): # Default info logs
            type_indicator_for_log_entry = "[INFO]"


        full_content_for_ui = f"{log_prefix} {type_indicator_for_log_entry} {text}"
        await send_ws_message_for_session("monitor_log", full_content_for_ui)

        active_task_id = session_data.get(session_id, {}).get("current_task_id")
        if active_task_id:
            try:
                await add_message(active_task_id, session_id, log_type, text) # Save original text
            except Exception as db_err:
                logger.error(f"[{session_id}] Failed to save monitor log '{log_type}' to DB: {db_err}")


    ws_callback_handler: Optional[WebSocketCallbackHandler] = None
    # memory: Optional[ConversationBufferWindowMemory] = None # Removed direct memory instantiation here
    session_setup_ok = False
    try:
        logger.info(f"[{session_id}] Starting session setup...")
        # Memory is now part of session_data, initialized below.
        # logger.debug(f"[{session_id}] Creating ConversationBufferWindowMemory (K={settings.agent_memory_window_k})...")
        # memory = ConversationBufferWindowMemory(
        #     k=settings.agent_memory_window_k, memory_key="chat_history", input_key="input", output_key="output", return_messages=True
        # )
        # logger.debug(f"[{session_id}] Memory object created.")

        logger.debug(f"[{session_id}] Creating WebSocketCallbackHandler...")
        # Partial for db_add_message, as its signature is fixed
        db_add_func = functools.partial(add_message)
        ws_callback_handler = WebSocketCallbackHandler(session_id, send_ws_message_for_session, db_add_func, session_data)
        logger.debug(f"[{session_id}] Callback handler created.")

        logger.debug(f"[{session_id}] Storing session data...")
        # Initialize session_data for this session_id
        from langchain.memory import ConversationBufferWindowMemory # Import here for lazy init
        session_data[session_id] = {
            "memory": ConversationBufferWindowMemory( # Initialize memory here
                k=settings.agent_memory_window_k, memory_key="chat_history", input_key="input", output_key="output", return_messages=True
            ),
            "callback_handler": ws_callback_handler,
            "current_task_id": None,
            "selected_llm_provider": settings.executor_default_provider, # Default for Executor/DirectQA
            "selected_llm_model_name": settings.executor_default_model_name, # Default for Executor/DirectQA
            "cancellation_requested": False,
            "current_plan_structured": None, # For plans awaiting confirmation or being executed
            "current_plan_human_summary": None,
            "current_plan_step_index": -1, # Current step in execution
            "plan_execution_active": False, # True if a plan is currently being executed
            "original_user_query": None, # The user query that initiated the current plan/QA
            "active_plan_filename": None, # Filename of the saved plan artifact
            # Session-specific LLM overrides for different agent roles
            "session_intent_classifier_llm_id": None, # e.g., "gemini::gemini-pro" or None to use default
            "session_planner_llm_id": None,
            "session_controller_llm_id": None,
            "session_evaluator_llm_id": None,
        }
        logger.info(f"[{session_id}] Session setup complete.")
        session_setup_ok = True
    except Exception as e:
        logger.error(f"[{session_id}] CRITICAL ERROR during session setup: {e}", exc_info=True)
        if websocket: # Check if websocket object is valid
            try:
                await websocket.close(code=1011, reason="Session setup failed")
            except Exception as close_e:
                logger.error(f"[{session_id}] Error closing websocket during setup failure: {close_e}")
        # Clean up potentially partially added entries
        if session_id in connected_clients: del connected_clients[session_id]
        if session_id in session_data: del session_data[session_id]
        return # Do not proceed if session setup failed

    if not session_setup_ok: # Should be redundant due to return above, but defensive
        logger.error(f"[{session_id}] Halting handler because session setup failed.")
        return

    # Define a mapping from message types to their handler functions
    MessageHandler = Callable[..., Coroutine[Any, Any, None]] # Type hint for handler functions

    message_handler_map: Dict[str, MessageHandler] = {
        "context_switch": process_context_switch,
        "user_message": process_user_message, # Will now need research_agent_lg_graph
        "execute_confirmed_plan": process_execute_confirmed_plan, # Will now need research_agent_lg_graph
        "new_task": process_new_task,
        "delete_task": process_delete_task,
        "rename_task": process_rename_task,
        "set_llm": process_set_llm, # Handles Executor/DirectQA LLM selection
        "get_available_models": process_get_available_models,
        "cancel_agent": process_cancel_agent,
        "get_artifacts_for_task": process_get_artifacts_for_task,
        "run_command": process_run_command, # For direct shell commands
        "action_command": process_action_command, # For UITL actions
        "set_session_role_llm": process_set_session_role_llm, # Handles role-specific LLM overrides
    }

    try:
        # Send initial status message and available models
        status_llm_info = f"Executor LLM: {settings.executor_default_provider} ({settings.executor_default_model_name})"
        logger.info(f"[{session_id}] Sending initial status message...");
        await send_ws_message_for_session("status_message", f"Connected (Session: {session_id[:8]}...). Agent Ready. {status_llm_info}.")

        role_llm_defaults = {
            "intent_classifier": f"{settings.intent_classifier_provider}::{settings.intent_classifier_model_name}",
            "planner": f"{settings.planner_provider}::{settings.planner_model_name}",
            "controller": f"{settings.controller_provider}::{settings.controller_model_name}",
            "evaluator": f"{settings.evaluator_provider}::{settings.evaluator_model_name}",
        }
        await send_ws_message_for_session("available_models", {
           "gemini": settings.gemini_available_models,
           "ollama": settings.ollama_available_models,
           "default_executor_llm_id": f"{settings.executor_default_provider}::{settings.executor_default_model_name}", # For Executor/DirectQA default
           "role_llm_defaults": role_llm_defaults # Defaults for role-specific selectors
        })
        logger.info(f"[{session_id}] Sent available_models (with role defaults) to client.")

        await add_monitor_log_and_save(f"New client connection: {websocket.remote_address}", "system_connect")
        logger.info(f"[{session_id}] Added system_connect log.")


        logger.info(f"[{session_id}] Entering message processing loop...")
        async for message_str in websocket:
            logger.debug(f"[{session_id}] Received raw message: {message_str[:200]}{'...' if len(message_str)>200 else ''}")
            try:
                parsed_data = json.loads(message_str) # Assuming message is JSON string
                message_type = parsed_data.get("type")

                handler_func = message_handler_map.get(message_type)
                if handler_func:
                    # Ensure session_data_entry and connected_clients_entry are fresh for each message
                    current_session_data_entry = session_data.get(session_id)
                    current_connected_clients_entry = connected_clients.get(session_id)

                    if not current_session_data_entry or not current_connected_clients_entry:
                        logger.error(f"[{session_id}] Critical: session_data or connected_clients entry missing for active session. Type: {message_type}")
                        await send_ws_message_for_session("status_message", "Error: Session integrity issue. Please refresh.")
                        continue # Skip processing this message

                    # Prepare arguments for the handler function
                    # Common arguments for most handlers
                    handler_args: Dict[str, Any] = { # Use a dictionary for clarity
                        "session_id": session_id, "data": parsed_data,
                        "session_data_entry": current_session_data_entry,
                        "connected_clients_entry": current_connected_clients_entry,
                        "send_ws_message_func": send_ws_message_for_session,
                        "add_monitor_log_func": add_monitor_log_and_save,
                    }

                    # Add DB functions as needed by specific handlers
                    if message_type in ["context_switch", "user_message", "run_command", "execute_confirmed_plan"]:
                        handler_args["db_add_message_func"] = add_message
                    if message_type == "context_switch": # context_switch needs several DB and artifact functions
                        handler_args["db_add_task_func"] = add_task
                        handler_args["db_get_messages_func"] = get_messages_for_task
                        handler_args["get_artifacts_func"] = get_artifacts # For loading artifacts on switch
                    elif message_type == "new_task" or message_type == "delete_task" or message_type == "get_artifacts_for_task":
                        # new_task might not need get_artifacts directly but context_switch after it will
                        # delete_task might clear UI artifacts or trigger UI to request them for new active task
                        handler_args["get_artifacts_func"] = get_artifacts
                    if message_type == "delete_task":
                        handler_args["db_delete_task_func"] = delete_task_and_messages
                    elif message_type == "rename_task":
                        handler_args["db_rename_task_func"] = rename_task_in_db
                    elif message_type == "run_command":
                        handler_args["execute_shell_command_func"] = execute_shell_command
                    # MODIFIED: Pass compiled_research_agent_graph to relevant handlers
                    if message_type == "user_message" or message_type == "execute_confirmed_plan":
                        handler_args["research_agent_lg_graph"] = compiled_research_agent_graph


                    await handler_func(**handler_args) # Call the appropriate handler

                else: # Unknown message type
                    logger.warning(f"[{session_id}] Unknown message type received: {message_type}")
                    await add_monitor_log_and_save(f"Received unknown message type: {message_type}", "error_unknown_msg")

            except json.JSONDecodeError:
                logger.error(f"[{session_id}] Received non-JSON message: {message_str[:200]}{'...' if len(message_str)>200 else ''}")
                await add_monitor_log_and_save("Error: Received invalid message format (not JSON).", "error_json")
            except asyncio.CancelledError:
                logger.info(f"[{session_id}] Message processing loop cancelled.")
                raise # Re-raise to be caught by outer handler for cleanup
            except Exception as e:
                logger.error(f"[{session_id}] Error processing message: {e}", exc_info=True)
                try:
                    await add_monitor_log_and_save(f"Error processing message: {e}", "error_processing")
                    await send_ws_message_for_session("status_message", f"Error processing message: {type(e).__name__}")
                except Exception as inner_e: # Error during error reporting
                    logger.error(f"[{session_id}] Further error during error reporting: {inner_e}")

    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as ws_close_exc:
        if isinstance(ws_close_exc, websockets.exceptions.ConnectionClosedOK):
            logger.info(f"Client disconnected normally: {websocket.remote_address} (Session: {session_id}) - Code: {ws_close_exc.code}, Reason: {ws_close_exc.reason}")
        else: # ConnectionClosedError or other abnormal close
            logger.warning(f"Connection closed abnormally: {websocket.remote_address} (Session: {session_id}) - Code: {ws_close_exc.code}, Reason: {ws_close_exc.reason}")
    except asyncio.CancelledError:
        logger.info(f"WebSocket handler for session {session_id} cancelled.")
        if websocket: await websocket.close(code=1012, reason="Server shutting down") # Server initiated shutdown
    except Exception as e: # Catch-all for unexpected errors in the handler's main loop
        logger.error(f"Unhandled error in WebSocket handler: {websocket.remote_address} (Session: {session_id}): {e}", exc_info=True)
        try:
            if websocket: await websocket.close(code=1011, reason="Internal server error")
        except Exception as close_e:
            logger.error(f"[{session_id}] Error closing websocket after unhandled handler error: {close_e}")
    finally:
        logger.info(f"Cleaning up resources for session {session_id}")
        # Cancel any active agent task for this session
        agent_task = connected_clients.get(session_id, {}).get("agent_task")
        if agent_task and not agent_task.done():
            logger.warning(f"[{session_id}] Cancelling active task during cleanup.")
            agent_task.cancel()
            try:
                await agent_task # Wait for the task to actually cancel
            except asyncio.CancelledError:
                pass # Expected
            except AgentCancelledException: # Custom exception from callback
                pass
            except Exception as cancel_e:
                logger.error(f"[{session_id}] Error waiting for task cancellation during cleanup: {cancel_e}")

        # Remove client from connected_clients and session_data
        if session_id in connected_clients:
            del connected_clients[session_id]
        if session_id in session_data:
            del session_data[session_id]
        logger.info(f"Cleaned up session data for {session_id}. Client removed: {websocket.remote_address}. Active clients: {len(connected_clients)}")


async def main():
    """Main function to initialize DB, start servers, and handle shutdown."""
    await init_db() # Initialize the database

    # Start the file server
    file_server_site, file_server_runner = await setup_file_server()
    await file_server_site.start()
    logger.info("File server started.")

    # Start the WebSocket server
    ws_host = "0.0.0.0" # Listen on all interfaces
    ws_port = 8765
    logger.info(f"Starting WebSocket server on ws://{ws_host}:{ws_port}")

    shutdown_event = asyncio.Event()

    # Create and start the WebSocket server
    websocket_server = await websockets.serve(
        handler,
        ws_host,
        ws_port,
        max_size=settings.websocket_max_size_bytes,
        ping_interval=settings.websocket_ping_interval,
        ping_timeout=settings.websocket_ping_timeout
    )
    logger.info("WebSocket server started.")

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}. Initiating shutdown...")
        shutdown_event.set()
        # Restore original handlers to prevent multiple triggers if shutdown is slow
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)

    try:
        # Add signal handlers only if not on Windows or if using ProactorEventLoop
        loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT, None)
        loop.add_signal_handler(signal.SIGTERM, signal_handler, signal.SIGTERM, None)
    except NotImplementedError:
        logger.warning("Signal handlers not available on this platform (e.g., Windows without ProactorEventLoop). Use Ctrl+C if available, or send SIGTERM.")

    logger.info("Application servers running. Press Ctrl+C to stop (or send SIGTERM).")

    # Wait for the shutdown signal
    await shutdown_event.wait()

    logger.info("Shutdown signal received. Stopping servers...")

    # Stop WebSocket server
    logger.info("Stopping WebSocket server...")
    websocket_server.close()
    await websocket_server.wait_closed()
    logger.info("WebSocket server stopped.")

    # Stop file server
    logger.info("Stopping file server...")
    await file_server_runner.cleanup() # Correctly use the runner for cleanup
    logger.info("File server stopped.")

    # Cancel any other outstanding asyncio tasks
    tasks_to_cancel = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks_to_cancel:
        logger.info(f"Cancelling {len(tasks_to_cancel)} outstanding tasks...")
        for task_to_cancel_item in tasks_to_cancel:
            task_to_cancel_item.cancel()
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True) # Wait for cancellations
        logger.info("Outstanding tasks cancelled.")


if __name__ == "__main__":
    # Suppress LangSmith warnings if not configured
    warnings.filterwarnings("ignore", category=UserWarning, message=".*LangSmith API key.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*LangSmithMissingAPIKeyWarning.*")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually (KeyboardInterrupt).")
    except Exception as e:
        logging.critical(f"Server failed to start or crashed: {e}", exc_info=True)
        exit(1)

