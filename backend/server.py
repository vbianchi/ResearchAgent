# -----------------------------------------------------------------------------
# ResearchAgent Backend Server (Phase 9: UI Hang Bugfix)
#
# This version adds a fallback mechanism to the agent handler to prevent the
# UI from getting stuck in a "thinking" state.
#
# 1. Robust Final Message Handling: In the `run_agent_handler`, after the
#    agent stream finishes, if a specific `final_answer` or `direct_answer`
#    is not found, a generic "Task Completed" message is now sent to the UI.
#    This ensures the client always receives a terminal message, allowing it
#    to exit the "thinking" state gracefully.
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

# --- Helper to format model names ---
def format_model_name(model_id):
    """Creates a user-friendly name from a model ID."""
    try:
        provider, name = model_id.split("::")
        name_parts = name.replace('-', ' ').split()
        formatted_name = ' '.join(part.capitalize() for part in name_parts)
        return f"{provider.capitalize()} {formatted_name}"
    except:
        return model_id

# --- Workspace Deletion Helper ---
def _safe_delete_workspace(task_id: str):
    """Safely and recursively deletes a task's workspace directory."""
    try:
        workspace_path = _resolve_path("/app/workspace", task_id)
        if not os.path.abspath(workspace_path).startswith(os.path.abspath("/app/workspace")):
            raise PermissionError("Security check failed: Attempt to delete directory outside of workspace.")
            
        if os.path.isdir(workspace_path):
            shutil.rmtree(workspace_path)
            logger.info(f"Task '{task_id}': Successfully deleted workspace directory: {workspace_path}")
            return True
        else:
            logger.warning(f"Task '{task_id}': Workspace directory not found for deletion: {workspace_path}")
            return False
            
    except Exception as e:
        logger.error(f"Task '{task_id}': Error deleting workspace: {e}", exc_info=True)
        return False


# --- HTTP File Server for Workspace ---
class WorkspaceHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/models': self._handle_get_models()
        elif parsed_path.path == '/files': self._handle_list_files(parsed_path)
        elif parsed_path.path == '/file-content': self._handle_get_file_content(parsed_path)
        else: self._send_json_response(404, {'error': 'Not Found'})
    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/upload': self._handle_file_upload()
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
        logger.info("Parsing .env to serve available and default models.")
        available_models = []
        model_ids = set()
        def parse_and_add_models(env_var, provider_prefix):
            models_str = os.getenv(env_var)
            if models_str:
                for model_name in models_str.split(','):
                    model_name = model_name.strip()
                    if model_name:
                        full_id = f"{provider_prefix}::{model_name}"
                        if full_id not in model_ids:
                            available_models.append({"id": full_id, "name": format_model_name(full_id)})
                            model_ids.add(full_id)
        parse_and_add_models("GEMINI_AVAILABLE_MODELS", "gemini")
        parse_and_add_models("OLLAMA_AVAILABLE_MODELS", "ollama")
        safe_fallback_model = "gemini::gemini-1.5-flash-latest"
        if not available_models:
            available_models.append({"id": safe_fallback_model, "name": format_model_name(safe_fallback_model)})
        global_default_llm = os.getenv("DEFAULT_LLM_ID", safe_fallback_model)
        default_models = {
            "ROUTER_LLM_ID": os.getenv("ROUTER_LLM_ID", global_default_llm),
            "LIBRARIAN_LLM_ID": os.getenv("LIBRARIAN_LLM_ID", global_default_llm),
            "CHIEF_ARCHITECT_LLM_ID": os.getenv("CHIEF_ARCHITECT_LLM_ID", global_default_llm),
            "SITE_FOREMAN_LLM_ID": os.getenv("SITE_FOREMAN_LLM_ID", global_default_llm),
            "PROJECT_SUPERVISOR_LLM_ID": os.getenv("PROJECT_SUPERVISOR_LLM_ID", global_default_llm),
            "EDITOR_LLM_ID": os.getenv("EDITOR_LLM_ID", "gemini::gemini-1.5-pro-latest")
        }
        response_data = { "available_models": available_models, "default_models": default_models }
        self._send_json_response(200, response_data)
    def _handle_list_files(self, parsed_path):
        query_components = parse_qs(parsed_path.query)
        subdir = query_components.get("path", [None])[0]
        if not subdir: return self._send_json_response(400, {"error": "Missing 'path' query parameter."})
        base_workspace = "/app/workspace"
        try:
            full_path = _resolve_path(base_workspace, subdir)
            if os.path.isdir(full_path): self._send_json_response(200, {"files": os.listdir(full_path)})
            else: self._send_json_response(404, {"error": f"Directory '{subdir}' not found."})
        except Exception as e: self._send_json_response(500, {"error": str(e)})
    def _handle_get_file_content(self, parsed_path):
        query_components = parse_qs(parsed_path.query)
        workspace_id = query_components.get("path", [None])[0]
        filename = query_components.get("filename", [None])[0]
        if not workspace_id or not filename: return self._send_json_response(400, {"error": "Missing 'path' or 'filename' parameter."})
        try:
            workspace_dir = f"/app/workspace/{workspace_id}"
            full_path = _resolve_path(workspace_dir, filename)
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
            workspace_id = form.getvalue('workspace_id')
            file_item = form['file']
            if not workspace_id or not hasattr(file_item, 'filename') or not file_item.filename: return self._send_json_response(400, {'error': 'Missing workspace_id or file.'})
            filename = os.path.basename(file_item.filename)
            workspace_dir = f"/app/workspace/{workspace_id}"
            full_path = _resolve_path(workspace_dir, filename)
            with open(full_path, 'wb') as f: f.write(file_item.file.read())
            self._send_json_response(200, {'message': f"File '{filename}' uploaded successfully."})
        except Exception as e:
            self._send_json_response(500, {'error': f'Server error during file upload: {e}'})

def run_http_server():
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("FILE_SERVER_PORT", 8766))
    server_address = (host, port)
    httpd = HTTPServer(server_address, WorkspaceHTTPHandler)
    logger.info(f"Starting HTTP file server at http://{host}:{port}")
    httpd.serve_forever()

# --- WebSocket Handlers ---

async def run_agent_handler(websocket, data):
    """Handles the 'run_agent' message type."""
    prompt = data.get("prompt")
    llm_config = data.get("llm_config", {})
    task_id = data.get("task_id")

    if not prompt or not task_id: return

    initial_state = {"messages": [HumanMessage(content=prompt)], "llm_config": llm_config, "task_id": task_id}
    config = {"recursion_limit": 100}

    logger.info(f"Task '{task_id}': Invoking agent with prompt: {prompt[:100]}...")
    
    last_event = None
    try:
        async for event in agent_graph.astream_events(initial_state, config=config, version="v1"):
            last_event = event
            event_type = event["event"]
            if event_type in ["on_chain_start", "on_chain_end"] and event["name"] in ["Chief_Architect", "Site_Foreman", "Worker", "Project_Supervisor", "Editor", "Librarian"]:
                response = {"type": "agent_event", "event": event_type, "name": event["name"], "data": event['data'], "task_id": task_id}
                await websocket.send(json.dumps(response, default=str))
    except Exception as e:
        logger.error(f"Task '{task_id}': Exception during agent execution: {e}", exc_info=True)
        await websocket.send(json.dumps({"type": "error", "data": str(e), "task_id": task_id}))
        return

    # --- MODIFIED: Robust final message handling ---
    answer_found = False
    if last_event and last_event["event"] == "on_chain_end":
        final_output = last_event.get("data", {}).get("output")
        if isinstance(final_output, list):
            for node_output in final_output:
                if isinstance(node_output, dict):
                    if 'Librarian' in node_output:
                        answer = node_output.get('Librarian', {}).get('answer')
                        await websocket.send(json.dumps({"type": "direct_answer", "data": answer, "task_id": task_id}))
                        answer_found = True
                        break
                    elif 'Editor' in node_output:
                        answer = node_output.get('Editor', {}).get('answer')
                        await websocket.send(json.dumps({"type": "final_answer", "data": answer, "task_id": task_id}))
                        answer_found = True
                        break
    
    if not answer_found:
        logger.warning(f"Task '{task_id}': Agent finished but no specific answer was found. Sending generic completion message.")
        await websocket.send(json.dumps({"type": "final_answer", "data": "The task has completed. Review the execution log for details.", "task_id": task_id}))


async def handle_task_create(websocket, data):
    """Handles the 'task_create' message type."""
    task_id = data.get("task_id")
    if not task_id: return
    workspace_path = f"/app/workspace/{task_id}"
    os.makedirs(workspace_path, exist_ok=True)
    logger.info(f"Task '{task_id}': Workspace created at {workspace_path}")

async def handle_task_delete(websocket, data):
    """Handles the 'task_delete' message type."""
    task_id = data.get("task_id")
    if not task_id: return
    _safe_delete_workspace(task_id)

# --- Main WebSocket Router ---
async def message_router(websocket):
    """Routes incoming WebSocket messages to the appropriate handler."""
    logger.info(f"Client connected from {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")

                if message_type == "run_agent": await run_agent_handler(websocket, data)
                elif message_type == "task_create": await handle_task_create(websocket, data)
                elif message_type == "task_delete": await handle_task_delete(websocket, data)
                else: logger.warning(f"Received unknown message type: '{message_type}'")

            except json.JSONDecodeError: logger.error(f"Failed to decode JSON from message: {message}")
            except Exception as e: logger.error(f"Error processing message: {e}", exc_info=True)
    
    except websockets.exceptions.ConnectionClosed as e: logger.info(f"Client disconnected: {e}")
    except Exception as e: logger.error(f"An unexpected error occurred in the handler: {e}", exc_info=True)


# --- Main Server Function ---
async def main():
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8765))
    logger.info(f"Starting ResearchAgent WebSocket server at ws://{host}:{port}")
    async with websockets.serve(message_router, host, port, max_size=None):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shut down gracefully.")
