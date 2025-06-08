# backend/server.py
print("--- EXECUTING LATEST server.py - V11 ---")
import asyncio
import websockets
import json
import datetime
import logging
import uuid
import re
import urllib.parse
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

# --- Web Server Imports ---
from aiohttp import web
from aiohttp.web import FileResponse
import aiohttp_cors

# LangChain Imports
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Project Imports ---
from backend.config import settings
from backend.agent import create_agent_executor
from backend.tools import get_dynamic_tools, get_task_workspace_path, BASE_WORKSPACE_ROOT, TEXT_EXTENSIONS
from backend.callbacks import WebSocketCallbackHandler, AgentCancelledException
from backend.db_utils import (
    init_db, add_task, add_message, get_messages_for_task,
    delete_task_and_messages, rename_task_in_db
)
from backend.llm_setup import get_llm
from backend.intent_classifier import classify_intent
from backend.prompts import DIRECT_QA_SYSTEM_PROMPT

# --- File Server & Logging Setup ---
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(levelname)s - %(name)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state dictionaries
connected_clients: Dict[str, Dict[str, Any]] = {}
session_data: Dict[str, Dict[str, Any]] = {}

FILE_SERVER_LISTEN_HOST = "0.0.0.0"
FILE_SERVER_CLIENT_HOST = settings.file_server_hostname
FILE_SERVER_PORT = 8766

# --- Helper functions for file server and artifacts ---
async def handle_workspace_file(request: web.Request) -> web.Response:
    task_id = request.match_info.get('task_id')
    filename = request.match_info.get('filename')
    if not task_id or not filename: raise web.HTTPBadRequest(text="Task ID and filename required")
    if ".." in filename or filename.startswith('/'): raise web.HTTPForbidden(text="Invalid filename path.")
    try:
        task_workspace = get_task_workspace_path(task_id, create_if_not_exists=False)
        file_path = (task_workspace / Path(filename).name).resolve()
        if not file_path.is_relative_to(BASE_WORKSPACE_ROOT.resolve()): raise web.HTTPForbidden(text="Access denied.")
    except Exception as e:
        logger.error(f"Error accessing workspace for file request: {e}", exc_info=True)
        raise web.HTTPInternalServerError(text=f"Error accessing workspace: {e}")
    if not file_path.is_file(): raise web.HTTPNotFound(text=f"File not found: {filename}")
    return FileResponse(path=file_path)

def sanitize_filename(filename: str) -> str:
    if not filename: return f"uploaded_file_{uuid.uuid4().hex[:8]}"
    return re.sub(r'[^\w\s.-]', '', filename).strip()

async def handle_file_upload(request: web.Request) -> web.Response:
    task_id = request.match_info.get('task_id')
    if not task_id: return web.json_response({'status': 'error', 'message': 'Task ID required'}, status=400)
    try:
        task_workspace = get_task_workspace_path(task_id, create_if_not_exists=True)
        reader = await request.multipart()
        saved_files = []
        while True:
            part = await reader.next()
            if part is None: break
            if part.name == 'file' and part.filename:
                safe_filename = sanitize_filename(part.filename)
                save_path = task_workspace / safe_filename
                with open(save_path, 'wb') as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk: break
                        f.write(chunk)
                saved_files.append({'filename': safe_filename})
        return web.json_response({'status': 'success', 'saved': saved_files}, status=200)
    except Exception as e:
        logger.error(f"File upload failed for task {task_id}: {e}", exc_info=True)
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)

async def get_artifacts(task_id: str) -> List[Dict[str, str]]:
    artifacts = []
    try:
        task_workspace_path = get_task_workspace_path(task_id, create_if_not_exists=False)
        if not task_workspace_path.exists(): return []
        for file_path in sorted(task_workspace_path.iterdir(), key=os.path.getmtime, reverse=True):
            if file_path.is_file():
                artifact_type = 'unknown'
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']: artifact_type = 'image'
                elif file_path.suffix.lower() in TEXT_EXTENSIONS: artifact_type = 'text'
                elif file_path.suffix.lower() == '.pdf': artifact_type = 'pdf'
                if artifact_type != 'unknown':
                    encoded_filename = urllib.parse.quote(file_path.name)
                    artifact_url = f"http://{FILE_SERVER_CLIENT_HOST}:{FILE_SERVER_PORT}/workspace_files/{task_id}/{encoded_filename}"
                    artifacts.append({"type": artifact_type, "url": artifact_url, "filename": file_path.name})
    except Exception as e:
        logger.error(f"Error scanning artifacts for task {task_id}: {e}", exc_info=True)
    return artifacts

async def setup_file_server():
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
    get_resource = cors.add(app.router.add_resource("/workspace_files/{task_id}/{filename:.+}"))
    cors.add(get_resource.add_route("GET", handle_workspace_file))
    post_resource = cors.add(app.router.add_resource("/upload/{task_id}"))
    cors.add(post_resource.add_route("POST", handle_file_upload))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, FILE_SERVER_LISTEN_HOST, FILE_SERVER_PORT)
    return site, runner

# --- Agent Runners ---

async def run_direct_qa(user_input: str, session_id: str):
    """Handles simple, direct questions without using the ReAct agent."""
    session = session_data[session_id]
    send_ws_message = connected_clients[session_id]['send_ws_message']
    
    await send_ws_message("agent_thinking_update", {"status": "Answering directly..."})
    
    provider, model_name = session["selected_llm_id"].split("::", 1)
    llm = get_llm(settings, provider, model_name, callbacks=[session["callback_handler"]], requested_for_role="DirectQA")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIRECT_QA_SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    final_answer = await chain.ainvoke({"question": user_input})
    
    await send_ws_message("agent_message", final_answer)
    await add_message(session["current_task_id"], session_id, "agent_message", final_answer)
    logger.info(f"[{session_id}] Direct QA finished.")

async def run_react_agent(user_input: str, session_id: str):
    """Creates and runs the ReAct agent for complex tasks requiring tools."""
    session = session_data[session_id]
    send_ws_message = connected_clients[session_id]['send_ws_message']
    
    await send_ws_message("agent_thinking_update", {"status": "Initializing agent..."})
    
    llm_provider, llm_model = session["selected_llm_id"].split("::", 1)
    agent_llm = get_llm(settings, llm_provider, llm_model, callbacks=[session["callback_handler"]], requested_for_role="ReActAgent")
    tools = get_dynamic_tools(session["current_task_id"])
    
    agent_executor = create_agent_executor(
        llm=agent_llm,
        tools=tools,
        memory=session["memory"],
        max_iterations=settings.agent_max_iterations
    )

    await send_ws_message("agent_thinking_update", {"status": "Thinking..."})
    
    await agent_executor.ainvoke(
        {"input": user_input},
        config=RunnableConfig(callbacks=[session["callback_handler"]])
    )
    
    logger.info(f"[{session_id}] ReAct Agent execution finished.")

# --- Core WebSocket Handler ---

async def handler(websocket: websockets.WebSocketServerProtocol):
    session_id = str(uuid.uuid4())
    logger.info(f"[{session_id}] New client connection from {websocket.remote_address}.")

    async def send_ws_message(msg_type: str, content: Any):
        if session_id in connected_clients and connected_clients[session_id]["websocket"]:
            try:
                await connected_clients[session_id]["websocket"].send(json.dumps({"type": msg_type, "content": content}))
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[{session_id}] WebSocket closed when trying to send message '{msg_type}'.")

    connected_clients[session_id] = {"websocket": websocket, "agent_task": None, "send_ws_message": send_ws_message}
    ws_callback_handler = WebSocketCallbackHandler(session_id, send_ws_message, add_message, session_data)
    session_data[session_id] = {
        "memory": ConversationBufferWindowMemory(k=settings.agent_memory_window_k, memory_key="chat_history", return_messages=True),
        "callback_handler": ws_callback_handler,
        "current_task_id": None,
        "selected_llm_id": f"{settings.default_provider}::{settings.default_model_name}",
        "cancellation_requested": False
    }

    try:
        await send_ws_message("status_message", {"text": "Connected to agent backend."})
        await send_ws_message("available_models", {
           "gemini": settings.gemini_available_models,
           "ollama": settings.ollama_available_models,
           "default_executor_llm_id": f"{settings.default_provider}::{settings.default_model_name}",
           "role_llm_defaults": {} 
        })

        async for message_str in websocket:
            try:
                data = json.loads(message_str)
                message_type = data.get("type")
                current_session = session_data[session_id]
                
                if message_type == "user_message":
                    user_input = data.get("content")
                    if not current_session.get("current_task_id"):
                        await send_ws_message("status_message", {"text": "Please select a task first."})
                        continue
                    if connected_clients.get(session_id, {}).get("agent_task"):
                        await send_ws_message("status_message", {"text": "Agent is currently busy. Please wait."})
                        continue
                    
                    await add_message(current_session["current_task_id"], session_id, "user_input", user_input)
                    current_session['cancellation_requested'] = False

                    # <<< FIX: Robust lifecycle management >>>
                    agent_logic_task = None
                    try:
                        classification = await classify_intent(user_input)
                        intent = classification.get("intent")
                        logger.info(f"[{session_id}] Query classified with intent: {intent}")

                        if intent == "DIRECT_QA":
                            agent_logic_task = asyncio.create_task(run_direct_qa(user_input, session_id))
                        else: # Default to AGENT_ACTION
                            agent_logic_task = asyncio.create_task(run_react_agent(user_input, session_id))
                        
                        connected_clients[session_id]["agent_task"] = agent_logic_task
                        await agent_logic_task

                    except AgentCancelledException:
                        logger.warning(f"[{session_id}] Agent task caught AgentCancelledException in handler.")
                        await send_ws_message("status_message", {"text": "Operation cancelled."})
                    except asyncio.CancelledError:
                         logger.warning(f"[{session_id}] Agent task caught asyncio.CancelledError in handler.")
                         await send_ws_message("status_message", {"text": "Operation cancelled by user."})
                    except Exception as e:
                        logger.error(f"[{session_id}] Agent task failed with unhandled exception: {e}", exc_info=True)
                        await send_ws_message("agent_message", f"A critical error occurred: {str(e)}")
                    finally:
                        logger.info(f"[{session_id}] Agent task finished or was cancelled. Resetting agent_task state.")
                        await send_ws_message("agent_thinking_update", {"status": "Idle."})
                        if session_id in connected_clients:
                            connected_clients[session_id]["agent_task"] = None
                
                elif message_type == "context_switch":
                    task_id = data.get("taskId")
                    task_title = data.get("taskTitle", f"Task {task_id}")
                    current_session["current_task_id"] = task_id
                    current_session["callback_handler"].set_task_id(task_id)
                    current_session["memory"].clear()
                    await add_task(task_id, task_title, datetime.datetime.now(datetime.timezone.utc).isoformat())
                    history = await get_messages_for_task(task_id)
                    await send_ws_message("history_start", {})
                    if history:
                        for msg in history:
                            if msg['message_type'] == 'user_input':
                                await send_ws_message('user', {"content": msg['content']})
                                current_session['memory'].chat_memory.add_user_message(msg['content'])
                            elif msg['message_type'] == 'agent_message':
                                await send_ws_message('agent_message', {"content": msg['content']})
                                current_session['memory'].chat_memory.add_ai_message(msg['content'])
                    await send_ws_message("history_end", {})
                    artifacts = await get_artifacts(task_id)
                    await send_ws_message("update_artifacts", artifacts)

                elif message_type == "cancel_agent":
                    logger.warning(f"[{session_id}] Received request to cancel agent.")
                    current_session['cancellation_requested'] = True
                    agent_task = connected_clients.get(session_id, {}).get("agent_task")
                    if agent_task and not agent_task.done():
                        agent_task.cancel()
                        await send_ws_message("status_message", {"text": "Cancellation request sent."})
                
                elif message_type == "set_llm":
                     llm_id = data.get("llm_id", "")
                     if "::" in llm_id: current_session["selected_llm_id"] = llm_id
                     else: current_session["selected_llm_id"] = f"{settings.default_provider}::{settings.default_model_name}"
                     logger.info(f"[{session_id}] Session LLM set to: {current_session['selected_llm_id']}")

            except json.JSONDecodeError:
                logger.error(f"[{session_id}] Received non-JSON message: {message_str[:200]}")
            except Exception as e:
                logger.error(f"[{session_id}] Error in message loop: {e}", exc_info=True)
                await send_ws_message("status_message", {"text": f"An error occurred: {str(e)}", "isError": True})

    except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
        logger.info(f"[{session_id}] Client disconnected: {e}")
    finally:
        logger.info(f"[{session_id}] Cleaning up session.")
        if session_id in connected_clients:
            if connected_clients[session_id].get("agent_task"):
                connected_clients[session_id]["agent_task"].cancel()
            del connected_clients[session_id]
        if session_id in session_data:
            del session_data[session_id]
        logger.info(f"[{session_id}] Cleanup complete.")

async def main():
    await init_db()
    file_server_site, _ = await setup_file_server()
    await file_server_site.start()
    ws_host = "0.0.0.0"
    ws_port = 8765
    logger.info(f"Starting WebSocket server on ws://{ws_host}:{ws_port}")
    async with websockets.serve(handler, ws_host, ws_port, max_size=settings.websocket_max_size_bytes):
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")
