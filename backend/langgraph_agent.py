# -----------------------------------------------------------------------------
# ResearchAgent Core Agent (Phase 9: Definitive Fix v2)
#
# This is the final, robust version of the agent graph. The key change is a
# simplification and unification of the execution tracks.
#
# 1. Unified Execution Path: Both the `Handyman` and `Chief_Architect` now
#    produce a plan that is passed to the `Site_Foreman`. This eliminates the
#    need for a complex `after_worker_router` and makes the flow of control
#    unambiguous.
# 2. Correct HITL Routing: The `plan_feedback_router` is now the single,
#    authoritative path out of the HITL checkpoint, ensuring that user
#    feedback is always handled correctly.
# -----------------------------------------------------------------------------

import os
import logging
import json
import re
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

from .tools import get_available_tools
from .prompts import (
    router_prompt_template, handyman_prompt_template, structured_planner_prompt_template,
    site_foreman_prompt_template, evaluator_prompt_template, editor_prompt_template,
    USER_PLAN_FEEDBACK_TEMPLATE, FAILURE_FEEDBACK_TEMPLATE, REPLAN_FEEDBACK_TEMPLATE
)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    input: str
    task_id: str
    plan: List[dict]
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    llm_config: Dict[str, str]
    current_step_index: int
    current_tool_call: Optional[dict]
    tool_output: Optional[str]
    history: Annotated[List[str], lambda x, y: x + y]
    workspace_path: str
    step_outputs: Annotated[Dict[int, str], lambda x, y: {**x, **y}]
    step_evaluation: Optional[dict]
    answer: str
    max_step_retries: int
    step_retries: int
    max_plan_retries: int
    plan_retries: int
    failure_feedback: Optional[str]
    replan_feedback: Optional[str]
    user_plan_feedback: Optional[str]

LLM_CACHE = {}
def get_llm(state: GraphState, role_env_var: str, default_llm_id: str):
    run_config = state.get("llm_config", {})
    llm_id = run_config.get(role_env_var) or os.getenv(role_env_var, default_llm_id)
    if llm_id in LLM_CACHE: return LLM_CACHE[llm_id]
    provider, model_name = llm_id.split("::")
    logger.info(f"Task '{state.get('task_id')}': Initializing LLM for '{role_env_var}': {llm_id}")
    if provider == "gemini": llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
    elif provider == "ollama": llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL"))
    else: raise ValueError(f"Unsupported LLM provider: {provider}")
    LLM_CACHE[llm_id] = llm
    return llm

AVAILABLE_TOOLS = get_available_tools()
TOOL_MAP = {tool.name: tool for tool in AVAILABLE_TOOLS}
SANDBOXED_TOOLS = {"write_file", "read_file", "list_files", "workspace_shell"}
def format_tools_for_prompt():
    return "\n".join([f"  - {tool.name}: {tool.description}" for tool in AVAILABLE_TOOLS])

def task_setup_node(state: GraphState):
    return {
        "input": state['messages'][-1].content, "history": [], "current_step_index": 0, "step_outputs": {},
        "workspace_path": f"/app/workspace/{state['task_id']}", "llm_config": state.get("llm_config", {}),
        "max_step_retries": 5, "step_retries": 0, "max_plan_retries": 2, "plan_retries": 0,
        "failure_feedback": None, "replan_feedback": None, "user_plan_feedback": None
    }

def router_node(state: GraphState):
    llm = get_llm(state, "ROUTER_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = router_prompt_template.format(input=state["input"])
    response = llm.invoke(prompt)
    route = response.content.strip().replace("`", "")
    logger.info(f"Task '{state['task_id']}': Router classified as '{route}'")
    return {"history": [f"Initial classification: '{route}'"]}

def handyman_node(state: GraphState):
    llm = get_llm(state, "HANDYMAN_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = handyman_prompt_template.format(input=state["input"], tools=format_tools_for_prompt())
    response = llm.invoke(prompt)
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response.content, re.DOTALL)
        plan = json.loads(match.group(1).strip() if match else response.content.strip()).get("plan", [])
        return {"plan": plan, "history": [f"Handyman created single-step plan: {json.dumps(plan)}"]}
    except Exception as e:
        return {"plan": [{"error": f"Handyman failed: {e}"}]}

def chief_architect_node(state: GraphState):
    feedback, replan = state.get("user_plan_feedback"), state.get("replan_feedback")
    feedback_section = ""
    if feedback: feedback_section = USER_PLAN_FEEDBACK_TEMPLATE.format(feedback=feedback)
    elif replan: feedback_section = REPLAN_FEEDBACK_TEMPLATE.format(history="\n".join(state['history']))
    
    llm = get_llm(state, "CHIEF_ARCHITECT_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = structured_planner_prompt_template.format(input=state["input"], tools=format_tools_for_prompt(), user_plan_feedback=feedback_section)
    response = llm.invoke(prompt)
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response.content, re.DOTALL)
        plan = json.loads(match.group(1).strip() if match else response.content.strip()).get("plan", [])
        return {"plan": plan, "step_retries": 0, "current_step_index": 0, "failure_feedback": None, "replan_feedback": None, "user_plan_feedback": None}
    except Exception as e:
        return {"plan": [{"error": f"Architect failed: {e}"}]}

def wait_for_user_feedback_node(state: GraphState):
    logger.info(f"Task '{state['task_id']}': Reached HITL checkpoint.")
    return state

def site_foreman_node(state: GraphState):
    step_index, plan = state["current_step_index"], state["plan"]
    if not plan or step_index >= len(plan): return {"current_tool_call": {"error": "Invalid plan or step."}}
    current_step, failure_reason = plan[step_index], state.get("failure_feedback")
    failure_section = FAILURE_FEEDBACK_TEMPLATE.format(failure_reason=failure_reason) if failure_reason else ""
    llm = get_llm(state, "SITE_FOREMAN_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = site_foreman_prompt_template.format(
        tools=format_tools_for_prompt(), plan=json.dumps(plan, indent=2), history="\n".join(state["history"]),
        current_step=current_step.get("instruction", ""), failure_feedback_section=failure_section)
    response = llm.invoke(prompt)
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response.content, re.DOTALL)
        tool_call = json.loads(match.group(1).strip() if match else response.content.strip())
        return {"current_tool_call": tool_call}
    except Exception as e:
        return {"current_tool_call": {"error": f"Foreman failed: {e}"}}

async def worker_node(state: GraphState):
    tool_call = state.get("current_tool_call", {})
    if not tool_call or not tool_call.get("tool_name") or "error" in tool_call:
        output = f"Error: {tool_call.get('error', 'No tool.')}"
        return {"tool_output": output, "history": [f"Worker Error: {output}"]}

    tool_name, tool_input = tool_call["tool_name"], tool_call.get("tool_input", {})
    tool = TOOL_MAP.get(tool_name)
    if not tool: return {"tool_output": f"Error: Tool '{tool_name}' not found."}
    
    final_args = {}
    if isinstance(tool_input, dict): final_args.update(tool_input)
    else: final_args[next(iter(tool.args))] = tool_input
    if tool_name in SANDBOXED_TOOLS: final_args["workspace_path"] = state["workspace_path"]
    
    try:
        output = await tool.ainvoke(final_args)
        str_output = str(output)
        return {"tool_output": str_output, "history": [f"Worker executed `{tool_name}`. Output: {str_output[:300]}"]}
    except Exception as e:
        return {"tool_output": f"Error: {e}", "history": [f"Worker tool error: {e}"]}

def project_supervisor_node(state: GraphState):
    current_step = state["plan"][state["current_step_index"]]
    tool_call, tool_output = state.get("current_tool_call", {}), state.get("tool_output", "No output.")
    llm = get_llm(state, "PROJECT_SUPERVISOR_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = evaluator_prompt_template.format(current_step=current_step.get('instruction', ''), tool_call=json.dumps(tool_call), tool_output=tool_output)
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", llm.invoke(prompt).content, re.DOTALL)
        evaluation = json.loads(match.group(1).strip() if match else llm.invoke(prompt).content)
    except Exception:
        evaluation = {"status": "failure", "reasoning": "Supervisor output parse error."}
    
    history_record = f"Supervisor evaluation: {evaluation.get('status')} - {evaluation.get('reasoning')}"
    updates = {"step_evaluation": evaluation, "history": [history_record]}
    if evaluation.get("status") == "success":
        updates["step_retries"] = 0
        updates["failure_feedback"] = None if state.get("failure_feedback") else None
        if step_id := current_step.get("step_id"):
            updates.setdefault("step_outputs", {})[step_id] = tool_output
    else:
        updates["step_retries"] = state["step_retries"] + 1
        updates["failure_feedback"] = evaluation.get("reasoning", "Unknown error.")
    return updates

def advance_to_next_step_node(state: GraphState):
    return {"current_step_index": state["current_step_index"] + 1}

def escalate_to_architect_node(state: GraphState):
    return {"plan_retries": state["plan_retries"] + 1, "replan_feedback": "Previous plan failed.", "step_retries": 0, "failure_feedback": None}

def editor_node(state: GraphState):
    llm = get_llm(state, "EDITOR_LLM_ID", "gemini::gemini-1.5-pro-latest")
    prompt = editor_prompt_template.format(input=state["input"], history="\n".join(state["history"]))
    return {"answer": llm.invoke(prompt).content}

def main_router(state: GraphState):
    route = state["history"][-1]
    if "DIRECT_QA" in route: return "Editor"
    if "SIMPLE_TOOL_USE" in route: return "Handyman"
    if "COMPLEX_PROJECT" in route: return "Chief_Architect"
    return END

def plan_feedback_router(state: GraphState):
    feedback = (state.get("user_plan_feedback") or "approve").lower()
    if "approve" in feedback: return "Site_Foreman"
    if "abort" in feedback: return "Editor"
    return "Chief_Architect"

def after_plan_step_router(state: GraphState):
    if state["step_evaluation"].get("status") == "success":
        if state["current_step_index"] + 1 >= len(state.get("plan", [])): return "Editor"
        else: return "Advance_To_Next_Step"
    else: # Failure
        if state["step_retries"] >= state["max_step_retries"]:
            if state["plan_retries"] >= state["max_plan_retries"]: return "Editor"
            else: return "escalate_to_architect"
        else: return "Site_Foreman"

def create_agent_graph():
    workflow = StateGraph(GraphState)
    nodes = {
        "Task_Setup": task_setup_node, "Router": router_node, "Handyman": handyman_node,
        "Chief_Architect": chief_architect_node, "WaitFor_User_Feedback": wait_for_user_feedback_node,
        "Site_Foreman": site_foreman_node, "Worker": worker_node, "Project_Supervisor": project_supervisor_node,
        "Advance_To_Next_Step": advance_to_next_step_node, "escalate_to_architect": escalate_to_architect_node,
        "Editor": editor_node
    }
    for name, node in nodes.items(): workflow.add_node(name, node)

    workflow.set_entry_point("Task_Setup")
    workflow.add_edge("Task_Setup", "Router")
    workflow.add_conditional_edges("Router", main_router, {"Editor": "Editor", "Handyman": "Handyman", "Chief_Architect": "Chief_Architect"})
    
    # Unified execution start
    workflow.add_edge("Handyman", "Site_Foreman")
    workflow.add_edge("Chief_Architect", "WaitFor_User_Feedback")
    workflow.add_conditional_edges("WaitFor_User_Feedback", plan_feedback_router, {"Site_Foreman": "Site_Foreman", "Chief_Architect": "Chief_Architect", "Editor": "Editor"})
    
    workflow.add_edge("Site_Foreman", "Worker")
    
    # After worker, route based on plan length
    workflow.add_conditional_edges("Worker", lambda s: "Editor" if len(s.get("plan",[])) == 1 else "Project_Supervisor", {"Editor": "Editor", "Project_Supervisor": "Project_Supervisor"})

    workflow.add_conditional_edges("Project_Supervisor", after_plan_step_router, {"Editor": "Editor", "Advance_To_Next_Step": "Advance_To_Next_Step", "Site_Foreman": "Site_Foreman", "escalate_to_architect": "escalate_to_architect"})
    workflow.add_edge("Advance_To_Next_Step", "Site_Foreman")
    workflow.add_edge("escalate_to_architect", "Chief_Architect")
    workflow.add_edge("Editor", END)

    return workflow.compile()

agent_graph = create_agent_graph()
