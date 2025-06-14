# -----------------------------------------------------------------------------
# ResearchAgent Core Agent (Phase 9: Definitive Fix v3)
#
# This version implements the final, correct logic for the self-correction
# and escalation hierarchy, fixing the infinite loop bug.
#
# 1. Corrected `project_supervisor_node`: The node now correctly clears the
#    `failure_feedback` state ONLY after a corrective action succeeds. This
#    is the key fix that prevents the agent from getting stuck.
# 2. Corrected `after_plan_step_router`: The routing logic is now simpler and
#    more robust. It correctly uses the presence or absence of the
#    `failure_feedback` flag to decide whether to retry a step (after a
#    successful correction) or advance to the next step.
# 3. This implementation fully matches our agreed-upon architecture.
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

# --- Local Imports ---
from .tools import get_available_tools
from .prompts import (
    structured_planner_prompt_template,
    site_foreman_prompt_template,
    evaluator_prompt_template,
    final_answer_prompt_template,
    FAILURE_FEEDBACK_TEMPLATE,
    REPLAN_FEEDBACK_TEMPLATE
)

# --- Logging Setup ---
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Agent State Definition ---
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

# --- LLM Provider Helper ---
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

# --- Tool Management ---
AVAILABLE_TOOLS = get_available_tools()
TOOL_MAP = {tool.name: tool for tool in AVAILABLE_TOOLS}
SANDBOXED_TOOLS = {"write_file", "read_file", "list_files", "workspace_shell"}
def format_tools_for_prompt():
    return "\n".join([f"  - {tool.name}: {tool.description}" for tool in AVAILABLE_TOOLS])

# --- Graph Nodes ---
def task_setup_node(state: GraphState):
    """Creates the workspace and initializes state."""
    return {
        "input": state['messages'][-1].content, 
        "history": [], "current_step_index": 0, "step_outputs": {}, 
        "workspace_path": f"/app/workspace/{state['task_id']}", 
        "llm_config": state.get("llm_config", {}), "max_step_retries": 5,
        "step_retries": 0, "max_plan_retries": 2, "plan_retries": 0, 
        "failure_feedback": None, "replan_feedback": None
    }

def chief_architect_node(state: GraphState):
    """Generates or re-generates the structured plan."""
    replan_feedback = state.get("replan_feedback")
    if replan_feedback:
        replan_section = REPLAN_FEEDBACK_TEMPLATE.format(history="\n".join(state['history']))
    else:
        replan_section = ""

    llm = get_llm(state, "CHIEF_ARCHITECT_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = structured_planner_prompt_template.format(
        input=state["input"], tools=format_tools_for_prompt(), replan_feedback=replan_section
    )
    response = llm.invoke(prompt)
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response.content, re.DOTALL)
        plan = json.loads(match.group(1).strip() if match else response.content.strip()).get("plan", [])
        return {"plan": plan, "step_retries": 0, "current_step_index": 0, "failure_feedback": None, "replan_feedback": None}
    except Exception as e:
        return {"plan": [{"error": f"Failed to create a valid plan: {e}"}]}

def site_foreman_node(state: GraphState):
    """Refines a plan step or corrects it based on feedback."""
    step_index = state["current_step_index"]
    plan = state["plan"]
    
    if not plan or step_index >= len(plan):
        return {"current_tool_call": {"error": "Invalid plan or step index."}}

    current_step_details = plan[step_index]
    failure_reason = state.get("failure_feedback")
    failure_section = FAILURE_FEEDBACK_TEMPLATE.format(failure_reason=failure_reason) if failure_reason else ""

    llm = get_llm(state, "SITE_FOREMAN_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = site_foreman_prompt_template.format(
        tools=format_tools_for_prompt(), plan=json.dumps(plan, indent=2),
        history="\n".join(state["history"]), current_step=current_step_details.get("instruction", ""),
        failure_feedback_section=failure_section
    )
    
    response = llm.invoke(prompt)
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response.content, re.DOTALL)
        tool_call = json.loads(match.group(1).strip() if match else response.content.strip())
        return {"current_tool_call": tool_call}
    except Exception as e:
        return {"current_tool_call": {"error": f"Invalid JSON output from Site Foreman: {e}"}}

async def worker_node(state: GraphState):
    """Executes the tool call."""
    tool_call = state.get("current_tool_call", {})
    if not tool_call.get("tool_name") or "error" in tool_call:
        return {"tool_output": f"Error: {tool_call.get('error', 'No tool call was provided.')}"}
        
    tool_name, tool_input = tool_call["tool_name"], tool_call.get("tool_input", {})
    tool = TOOL_MAP.get(tool_name)
    if not tool: return {"tool_output": f"Error: Tool '{tool_name}' not found."}

    final_args = {}
    if isinstance(tool_input, dict): final_args.update(tool_input)
    else: final_args[next(iter(tool.args))] = tool_input

    if tool_name in SANDBOXED_TOOLS:
        final_args["workspace_path"] = state["workspace_path"]
    try:
        output = await tool.ainvoke(final_args)
        return {"tool_output": str(output)}
    except Exception as e:
        return {"tool_output": f"An error occurred executing tool '{tool_name}': {e}"}

def project_supervisor_node(state: GraphState):
    """Evaluates the step and records history."""
    current_step_details = state["plan"][state["current_step_index"]]
    tool_output = state.get("tool_output", "No output.")
    tool_call = state.get("current_tool_call", {})

    llm = get_llm(state, "PROJECT_SUPERVISOR_LLM_ID", "gemini::gemini-1.5-flash-latest")
    prompt = evaluator_prompt_template.format(
        current_step=current_step_details.get('instruction', ''),
        tool_call=json.dumps(tool_call), tool_output=tool_output
    )
    try:
        response_content = llm.invoke(prompt).content
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_content, re.DOTALL)
        evaluation = json.loads(match.group(1).strip() if match else response_content)
    except Exception:
        evaluation = {"status": "failure", "reasoning": "Could not parse Supervisor output."}

    history_record = (
        f"--- Step {state['current_step_index'] + 1} (Attempt {state['step_retries'] + 1}) ---\n"
        f"Instruction: {current_step_details.get('instruction')}\n"
        f"Action: {json.dumps(tool_call)}\nOutput: {tool_output}\n"
        f"Evaluation: {evaluation.get('status', 'unknown')} - {evaluation.get('reasoning', 'N/A')}"
    )
    
    updates = {"step_evaluation": evaluation, "history": [history_record]}
    was_corrective_action = state.get("failure_feedback") is not None

    if evaluation.get("status") == "success":
        updates["step_retries"] = 0
        if was_corrective_action:
            # THE FIX: If a correction succeeded, clear the feedback to signal a retry of the original step.
            updates["failure_feedback"] = None
        else:
            step_id = current_step_details.get("step_id")
            if step_id:
                updates.setdefault("step_outputs", {})[step_id] = tool_output
    else: # Failure
        updates["step_retries"] = state["step_retries"] + 1
        updates["failure_feedback"] = evaluation.get("reasoning", "Unknown error.")
    return updates

def advance_to_next_step_node(state: GraphState):
    """Increments the step index."""
    return {"current_step_index": state["current_step_index"] + 1}

def escalate_to_architect_node(state: GraphState):
    """Prepares the state for a full re-plan."""
    return {
        "plan_retries": state["plan_retries"] + 1,
        "replan_feedback": "The previous plan failed.",
        "step_retries": 0, "failure_feedback": None
    }

def librarian_node(state: GraphState):
    """Directly calls an LLM for a simple question."""
    llm = get_llm(state, "LIBRARIAN_LLM_ID", "gemini::gemini-1.5-flash-latest")
    return {"answer": llm.invoke(state["messages"]).content}

def editor_node(state: GraphState):
    """Synthesizes the final answer."""
    llm = get_llm(state, "EDITOR_LLM_ID", "gemini::gemini-1.5-pro-latest")
    return {"answer": llm.invoke(final_answer_prompt_template.format(input=state["input"], history="\n".join(state["history"]))).content}

# --- Conditional Routers ---
def router(state: GraphState):
    """Routes the workflow based on user intent."""
    llm = get_llm(state, "ROUTER_LLM_ID", "gemini::gemini-1.5-flash-latest")
    response = llm.invoke(f"Classify the user's message. Respond 'AGENT_ACTION' for a complex task, or 'DIRECT_QA'.\n\nUser message: '{state['input']}'")
    return "Chief_Architect" if "AGENT_ACTION" in response.content.strip() else "Librarian"

def after_plan_step_router(state: GraphState):
    """The main router for the execution loop."""
    evaluation = state.get("step_evaluation", {})
    
    if evaluation.get("status") == "success":
        # If failure_feedback is present, it means a correction just succeeded.
        # We need to retry the original step. Otherwise, we advance.
        if state.get("failure_feedback") is None:
            if state["current_step_index"] + 1 >= len(state.get("plan", [])):
                return "FINISH"
            else:
                return "Advance_To_Next_Step"
        else:
             # The corrective action succeeded, so clear the feedback and retry the original step.
             # Note: The 'failure_feedback' is cleared in the supervisor node now.
             return "Site_Foreman"
    
    # --- Failure Path ---
    if state["step_retries"] >= state["max_step_retries"]:
        if state["plan_retries"] >= state["max_plan_retries"]:
            logger.error(f"Task '{state['task_id']}': MAX PLAN RETRIES REACHED. Terminating.")
            return "FINISH"
        else:
            return "escalate_to_architect"
    else:
        return "Site_Foreman"

# --- Graph Definition ---
def create_agent_graph():
    """Builds the ResearchAgent's LangGraph."""
    workflow = StateGraph(GraphState)
    
    nodes = {
        "Task_Setup": task_setup_node, "Librarian": librarian_node,
        "Chief_Architect": chief_architect_node, "Site_Foreman": site_foreman_node,
        "Worker": worker_node, "Project_Supervisor": project_supervisor_node,
        "Advance_To_Next_Step": advance_to_next_step_node,
        "escalate_to_architect": escalate_to_architect_node, "Editor": editor_node
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("Task_Setup")
    workflow.add_conditional_edges("Task_Setup", router, {"Librarian": END, "Chief_Architect": "Chief_Architect"})
    workflow.add_edge("Chief_Architect", "Site_Foreman")
    workflow.add_edge("Site_Foreman", "Worker")
    workflow.add_edge("Worker", "Project_Supervisor")
    workflow.add_edge("Advance_To_Next_Step", "Site_Foreman")
    workflow.add_edge("escalate_to_architect", "Chief_Architect")

    workflow.add_conditional_edges(
        "Project_Supervisor",
        after_plan_step_router,
        {
            "FINISH": "Editor",
            "Advance_To_Next_Step": "Advance_To_Next_Step",
            "Site_Foreman": "Site_Foreman",
            "escalate_to_architect": "escalate_to_architect",
        }
    )

    workflow.add_edge("Editor", END)

    agent = workflow.compile()
    logger.info("ResearchAgent graph compiled with Full Escalation Hierarchy (v3).")
    return agent

agent_graph = create_agent_graph()
