# -----------------------------------------------------------------------------
# ResearchAgent Backend Server (Phase 9: Definitive Fix)
#
# This version fixes the core streaming and state management logic.
#
# Key Changes:
# 1. Switched from `astream` to `astream_events`. This is the correct method
#    for handling event-driven updates and provides robust access to the
#    final state and outputs of each node, resolving the "no final answer" bug.
# 2. Refactored the `stream_agent_events` and handler logic to properly
#    parse the `astream_events` output. It now correctly identifies the final
#    answer from the Editor node.
# 3. The HITL pause/resume logic is now correctly integrated with the event
#    streaming, which resolves the `NoneType` crash.
# -----------------------------------------------------------------------------

import asyncio
import logging
import os
import json
import threading
import cgi
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import websockets
from langchain_core.messages import HumanMessage

# --- Local Imports ---
from .langgraph_agent import agent_graph
from .tools.file_system import _resolve_path

# --- Configuration ---
load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PAUSED_STATE_CACHE = {}

def format_model_name(model_id):
    try:
        provider, name = model_id.split("::")
        name_parts = name.replace('-', ' ').split()
        return f"{provider.capitalize()} {' '.join(part.capitalize() for part in name_parts)}"
    except:
        return model_id

def _safe_delete_workspace(task_id: str):
    try:
        workspace_path = _resolve_path("/app/workspace", task_id)
        if not os.path.abspath(workspace_path).startswith(os.path.abspath("/app/workspace")):
            raise PermissionError("Security check failed: Attempt to delete directory outside of workspace.")
        if os.path.isdir(workspace_path):
            shutil.rmtree(workspace_path)
            logger.info(f"Task '{task_id}': Successfully deleted workspace: {workspace_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Task '{task_id}': Error deleting workspace: {e}", exc_info=True)
        return False

class WorkspaceHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/models': self._handle_get_models()
        elif parsed_path.path == '/files': self._handle_list_files(parsed_path)
        elif parsed_path.path == '/file-content': self._handle_get_file_content(parsed_path)
        else: self._send_json_response(404, {'error': 'Not Found'})
    def do_POST(self):
        if urlparse(self.path).path == '/upload': self._handle_file_upload()
        else: self._send_json_response(404, {'error': 'Not Found'})
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()
    def _send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    def _handle_get_models(self):
        available_models = []
        model_ids = set()
        def parse_and_add_models(env_var, provider_prefix):
            models_str = os.getenv(env_var)
            if models_str:
                for model_name in models_str.split(','):
                    full_id = f"{provider_prefix}::{model_name.strip()}"
                    if full_id not in model_ids:
                        available_models.append({"id": full_id, "name": format_model_name(full_id)})
                        model_ids.add(full_id)
        parse_and_add_models("GEMINI_AVAILABLE_MODELS", "gemini")
        parse_and_add_models("OLLAMA_AVAILABLE_MODELS", "ollama")
        safe_fallback = "gemini::gemini-1.5-flash-latest"
        if not available_models: available_models.append({"id": safe_fallback, "name": format_model_name(safe_fallback)})
        global_default = os.getenv("DEFAULT_LLM_ID", safe_fallback)
        default_models = {
            "ROUTER_LLM_ID": os.getenv("ROUTER_LLM_ID", global_default),
            "HANDYMAN_LLM_ID": os.getenv("HANDYMAN_LLM_ID", global_default),
            "CHIEF_ARCHITECT_LLM_ID": os.getenv("CHIEF_ARCHITECT_LLM_ID", global_default),
            "SITE_FOREMAN_LLM_ID": os.getenv("SITE_FOREMAN_LLM_ID", global_default),
            "PROJECT_SUPERVISOR_LLM_ID": os.getenv("PROJECT_SUPERVISOR_LLM_ID", global_default),
            "EDITOR_LLM_ID": os.getenv("EDITOR_LLM_ID", "gemini::gemini-1.5-pro-latest")
        }
        self._send_json_response(200, {"available_models": available_models, "default_models": default_models})
    def _handle_list_files(self, parsed_path):
        query = parse_qs(parsed_path.query)
        subdir = query.get("path", [None])[0]
        if not subdir: return self._send_json_response(400, {"error": "Missing 'path' parameter."})
        try:
            full_path = _resolve_path("/app/workspace", subdir)
            if os.path.isdir(full_path): self._send_json_response(200, {"files": os.listdir(full_path)})
            else: self._send_json_response(404, {"error": f"Directory '{subdir}' not found."})
        except Exception as e: self._send_json_response(500, {"error": str(e)})
    def _handle_get_file_content(self, parsed_path):
        query = parse_qs(parsed_path.query)
        workspace_id, filename = query.get("path", [None])[0], query.get("filename", [None])[0]
        if not workspace_id or not filename: return self._send_json_response(400, {"error": "Missing 'path' or 'filename'."})
        try:
            full_path = _resolve_path(f"/app/workspace/{workspace_id}", filename)
            with open(full_path, 'r', encoding='utf-8') as f: content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e: self._send_json_response(500, {"error": f"Error reading file: {e}"})
    def _handle_file_upload(self):
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
            workspace_id, file_item = form.getvalue('workspace_id'), form['file']
            if not workspace_id or not file_item or not file_item.filename: return self._send_json_response(400, {'error': 'Missing workspace_id or file.'})
            filename = os.path.basename(file_item.filename)
            full_path = _resolve_path(f"/app/workspace/{workspace_id}", filename)
            with open(full_path, 'wb') as f: f.write(file_item.file.read())
            self._send_json_response(200, {'message': f"File '{filename}' uploaded."})
        except Exception as e: self._send_json_response(500, {'error': f'Upload error: {e}'})

def run_http_server():
    host, port = os.getenv("BACKEND_HOST", "0.0.0.0"), int(os.getenv("FILE_SERVER_PORT", 8766))
    httpd = HTTPServer((host, port), WorkspaceHTTPHandler)
    logger.info(f"Starting HTTP file server at http://{host}:{port}")
    httpd.serve_forever()

# --- WebSocket Handlers ---
async def stream_agent_events(websocket, agent_state, task_id):
    """Helper to stream agent events and handle pausing/errors."""
    config = {"recursion_limit": 150}
    last_state = None

    try:
        async for event in agent_graph.astream_events(agent_state, config=config, version="v1"):
            event_name = event["event"]
            
            # Send all intermediate steps for real-time tracking
            response = {"type": "agent_event", "event": event_name, "name": event.get("name"), "data": event['data'], "task_id": task_id}
            await websocket.send(json.dumps(response, default=str))

            if event_name == "on_chain_end":
                node_name = event["name"]
                if node_name == "WaitFor_User_Feedback":
                    logger.info(f"Task '{task_id}': Pausing for user feedback.")
                    PAUSED_STATE_CACHE[task_id] = event['data']['output'] # Save the full state
                    plan = event['data']['output'].get("plan", [])
                    await websocket.send(json.dumps({"type": "plan_approval_request", "plan": plan, "task_id": task_id}))
                    return # Exit stream, execution paused.
                
                if node_name == "Editor":
                    final_answer = event['data']['output'].get('answer')
                    await websocket.send(json.dumps({"type": "final_answer", "data": final_answer, "task_id": task_id}))
                    return # Task is fully complete

    except Exception as e:
        logger.error(f"Task '{task_id}': Exception during agent execution: {e}", exc_info=True)
        await websocket.send(json.dumps({"type": "error", "data": str(e), "task_id": task_id}))

async def run_agent_handler(websocket, data):
    """Starts a new agent task."""
    task_id, prompt, llm_config = data.get("task_id"), data.get("prompt"), data.get("llm_config", {})
    if not task_id or not prompt: return
    logger.info(f"Task '{task_id}': Starting new agent run with prompt: {prompt[:100]}...")
    initial_state = {"messages": [HumanMessage(content=prompt)], "llm_config": llm_config, "task_id": task_id}
    await stream_agent_events(websocket, initial_state, task_id)

async def resume_agent_handler(websocket, data):
    """Resumes a paused agent task with user feedback."""
    task_id, feedback = data.get("task_id"), data.get("feedback")
    if not task_id or not feedback: return

    paused_state = PAUSED_STATE_CACHE.pop(task_id, None)
    if not paused_state:
        logger.error(f"Task '{task_id}': No paused state found to resume.")
        return
        
    logger.info(f"Task '{task_id}': Resuming agent run with user feedback: '{feedback}'")
    paused_state["user_plan_feedback"] = feedback
    await stream_agent_events(websocket, paused_state, task_id)

async def handle_task_create(websocket, data):
    if task_id := data.get("task_id"):
        os.makedirs(f"/app/workspace/{task_id}", exist_ok=True)
        logger.info(f"Task '{task_id}': Workspace created.")

async def handle_task_delete(websocket, data):
    if task_id := data.get("task_id"):
        _safe_delete_workspace(task_id)

async def message_router(websocket):
    """Routes incoming WebSocket messages."""
    logger.info(f"Client connected from {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")
                handlers = {
                    "run_agent": run_agent_handler, "user_plan_feedback": resume_agent_handler,
                    "task_create": handle_task_create, "task_delete": handle_task_delete,
                }
                if handler := handlers.get(message_type):
                    await handler(websocket, data)
                else:
                    logger.warning(f"Unknown message type: '{message_type}'")
            except json.JSONDecodeError: logger.error(f"Failed to decode JSON from: {message}")
            except Exception as e: logger.error(f"Error processing message: {e}", exc_info=True)
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client disconnected: {e}")
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)

async def main():
    threading.Thread(target=run_http_server, daemon=True).start()
    host, port = os.getenv("BACKEND_HOST", "0.0.0.0"), int(os.getenv("BACKEND_PORT", 8765))
    logger.info(f"Starting ResearchAgent WebSocket server at ws://{host}:{port}")
    async with websockets.serve(message_router, host, port, max_size=None):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shut down gracefully.")
