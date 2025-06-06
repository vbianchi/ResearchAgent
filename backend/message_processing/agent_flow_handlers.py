# backend/message_processing/agent_flow_handlers.py
import logging
import json
import datetime
from typing import Dict, Any, Callable, Coroutine, Optional, List
import asyncio
from pathlib import Path
import aiofiles
import re
import uuid

# LangChain Imports
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

# Project Imports
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools, get_task_workspace_path
from backend.planner import generate_plan, PlanStep
from backend.callbacks import AgentCancelledException
from backend.intent_classifier import classify_intent, IntentClassificationOutput # Import the Pydantic model
from backend.langgraph_agent import research_agent_graph, ResearchAgentState

logger = logging.getLogger(__name__)

# Type Hints for Passed-in Functions
SendWSMessageFunc = Callable[[str, Any], Coroutine[Any, Any, None]]
AddMonitorLogFunc = Callable[[str, str], Coroutine[Any, Any, None]]
DBAddMessageFunc = Callable[[str, str, str, str], Coroutine[Any, Any, None]]


async def _update_plan_file_step_status(
    task_workspace_path: Path,
    plan_filename: str,
    step_number: int,
    status_char: str
) -> None:
    # ... (function content remains the same) ...
    if not plan_filename:
        logger.warning("Cannot update plan file: no active plan filename.")
        return

    plan_file_path = task_workspace_path / plan_filename
    if not await asyncio.to_thread(plan_file_path.exists):
        logger.warning(f"Plan file {plan_file_path} not found for updating step {step_number}.")
        return

    try:
        async with aiofiles.open(plan_file_path, 'r', encoding='utf-8') as f_read:
            lines = await f_read.readlines()

        updated_lines = []
        found_step = False
        step_pattern = re.compile(rf"^\s*-\s*\[\s*[ x!-]?\s*\]\s*{re.escape(str(step_number))}\.\s+.*", re.IGNORECASE)
        checkbox_pattern = re.compile(r"(\s*-\s*\[)\s*[ x!-]?\s*(\])")


        for line_no, line_content in enumerate(lines):
            if not found_step and step_pattern.match(line_content):
                updated_line = checkbox_pattern.sub(rf"\g<1>{status_char}\g<2>", line_content, count=1)
                updated_lines.append(updated_line)
                found_step = True
                logger.info(f"Updated plan file for step {step_number} to status '[{status_char}]'. Line: {updated_line.strip()}")
            else:
                updated_lines.append(line_content)

        if found_step:
            async with aiofiles.open(plan_file_path, 'w', encoding='utf-8') as f_write:
                await f_write.writelines(updated_lines)
        else:
            logger.warning(f"Step {step_number} pattern not found in plan file {plan_file_path} for status update. Regex was: {step_pattern.pattern}")
            if len(lines) > 0:
                logger.debug("First few lines of plan file for debugging update failure:")
                for i, l in enumerate(lines[:5]):
                    logger.debug(f"  Line {i+1}: {l.strip()}")
    except Exception as e:
        logger.error(f"Error updating plan file {plan_file_path} for step {step_number}: {e}", exc_info=True)


async def process_user_message(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc, db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any
) -> None:
    user_input_content = ""
    content_payload = data.get("content")
    if isinstance(content_payload, str):
        user_input_content = content_payload
    elif isinstance(content_payload, dict) and 'content' in content_payload and isinstance(content_payload['content'], str):
        user_input_content = content_payload['content']
    else:
        logger.warning(f"[{session_id}] Received non-string or unexpected content for user_message: {type(content_payload)}. Ignoring.")
        return

    active_task_id = session_data_entry.get("current_task_id")
    if not active_task_id:
        logger.warning(f"[{session_id}] User message received but no task active.")
        await send_ws_message_func("status_message", "Please select or create a task first.")
        return

    if (connected_clients_entry.get("agent_task") or
        session_data_entry.get("plan_execution_active")):
        logger.warning(f"[{session_id}] User message received while agent/plan is already running for task {active_task_id}.")
        await send_ws_message_func("status_message", "Agent is busy. Please wait or stop the current process.")
        return

    await db_add_message_func(active_task_id, session_id, "user_input", user_input_content)
    await add_monitor_log_func(f"User Input: {user_input_content}", "monitor_user_input")

    session_data_entry['original_user_query'] = user_input_content
    session_data_entry['cancellation_requested'] = False
    session_data_entry['active_plan_filename'] = None

    await send_ws_message_func("agent_thinking_update", {"status": "Classifying intent..."})

    dynamic_tools = get_dynamic_tools(active_task_id)
    tool_names_for_prompt = ", ".join([f"'{tool.name}'" for tool in dynamic_tools]) if dynamic_tools else "None"
    tools_summary_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in dynamic_tools]) if dynamic_tools else "No tools available."


    # Call the refactored classify_intent
    # classified_intent_value_dict is now an IntentClassificationOutput object or "PLAN" string
    classified_intent_result = await classify_intent(
        user_query=user_input_content,
        session_data_entry=session_data_entry,
        tool_names_for_prompt=tool_names_for_prompt, # Pass tool names for prompt
        tools_summary_for_prompt=tools_summary_for_prompt # Pass tool summaries for prompt
    )

    # <<< START MODIFICATION: Handle return from classify_intent >>>
    classified_intent_value: str
    intent_reasoning: Optional[str] = "No reasoning provided."
    # DIRECT_TOOL_REQUEST related fields are now on the Pydantic model
    intent_tool_name: Optional[str] = None
    intent_tool_input: Optional[Union[str, Dict[str, Any]]] = None

    if isinstance(classified_intent_result, IntentClassificationOutput):
        classified_intent_value = classified_intent_result.intent
        intent_reasoning = classified_intent_result.reasoning or "No reasoning provided."
        intent_tool_name = classified_intent_result.tool_name
        intent_tool_input = classified_intent_result.tool_input
        logger.info(f"IntentClassifier returned Pydantic model. Intent: {classified_intent_value}, Tool: {intent_tool_name}")
    elif isinstance(classified_intent_result, str) and classified_intent_result == "PLAN":
        classified_intent_value = "PLAN" # Error fallback from classify_intent
        intent_reasoning = "Intent classification failed, defaulting to PLAN."
        logger.warning("classify_intent returned string 'PLAN', indicating an error during classification.")
    else: # Should not happen if classify_intent adheres to its new return type
        logger.error(f"classify_intent returned unexpected type: {type(classified_intent_result)}. Defaulting to PLAN.")
        classified_intent_value = "PLAN"
        intent_reasoning = "Unexpected error during intent classification."
    # <<< END MODIFICATION >>>

    log_msg_intent = f"Intent classified as: {classified_intent_value}."
    if intent_tool_name: log_msg_intent += f" Tool: {intent_tool_name}."
    if isinstance(intent_tool_input, dict) or (isinstance(intent_tool_input, str) and len(intent_tool_input) < 50) :
        if intent_tool_input is not None: log_msg_intent += f" Input: {str(intent_tool_input)[:50]}..."
    elif isinstance(intent_tool_input, str): # Only log length for longer strings
        if intent_tool_input is not None: log_msg_intent += f" Input (len): {len(intent_tool_input)}."
    log_msg_intent += f" Reason: {intent_reasoning}"
    await add_monitor_log_func(log_msg_intent, "system_intent_classified")


    initial_llm_ids_for_state = {
        "intent_classifier_llm_id": session_data_entry.get("session_intent_classifier_llm_id"),
        "planner_llm_id": session_data_entry.get("session_planner_llm_id"),
        "controller_llm_id": session_data_entry.get("session_controller_llm_id"),
        "executor_llm_id": f"{session_data_entry.get('selected_llm_provider', settings.executor_default_provider)}::{session_data_entry.get('selected_llm_model_name', settings.executor_default_model_name)}",
        "evaluator_llm_id": session_data_entry.get("session_evaluator_llm_id"),
    }
    filtered_initial_llm_ids = {k: v for k,v in initial_llm_ids_for_state.items() if v is not None}


    if classified_intent_value == "PLAN":
        await send_ws_message_func("agent_thinking_update", {"status": "Generating plan..."})
        human_plan_summary, structured_plan_steps = await generate_plan(
            user_query=user_input_content,
            available_tools_summary=tools_summary_for_prompt, # Use the more detailed summary
            session_data_entry=session_data_entry
        )
        if human_plan_summary and structured_plan_steps:
            session_data_entry["current_plan_human_summary"] = human_plan_summary
            session_data_entry["current_plan_structured"] = structured_plan_steps
            
            plan_timestamp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            session_data_entry['current_plan_proposal_id_backend'] = plan_timestamp_id

            await send_ws_message_func("propose_plan_for_confirmation", {
                "human_summary": human_plan_summary,
                "structured_plan": structured_plan_steps,
                "plan_id": plan_timestamp_id
            })
            await add_monitor_log_func(f"Plan generated. Summary: {human_plan_summary}. Steps: {len(structured_plan_steps)}. Plan ID: {plan_timestamp_id}. Awaiting user confirmation.", "system_plan_generated")
            await send_ws_message_func("status_message", "Plan generated. Please review and confirm.")
            await send_ws_message_func("agent_thinking_update", {"status": "Awaiting plan confirmation..."})
        else:
            logger.error(f"[{session_id}] Failed to generate a plan for query: {user_input_content}")
            await add_monitor_log_func(f"Error: Failed to generate a plan.", "error_system")
            await send_ws_message_func("status_message", "Error: Could not generate a plan for your request.")
            await send_ws_message_func("agent_message", "I'm sorry, I couldn't create a plan for that request. Please try rephrasing or breaking it down.")
            await send_ws_message_func("agent_thinking_update", {"status": "Planning failed."})
    # <<< START MODIFICATION: Handle DIRECT_TOOL_REQUEST >>>
    elif classified_intent_value == "DIRECT_TOOL_REQUEST":
        await send_ws_message_func("agent_thinking_update", {"status": f"Preparing direct tool request: {intent_tool_name}..."})
        await add_monitor_log_func(f"Handling as DIRECT_TOOL_REQUEST with LangGraph. Tool: {intent_tool_name}, Input: {str(intent_tool_input)[:100]}...", "system_direct_tool_request")

        initial_graph_input_dict_tool = ResearchAgentState(
            user_query=user_input_content, # Still useful for context
            classified_intent="DIRECT_TOOL_REQUEST",
            # Pass the identified tool and input to the graph state
            controller_output_tool_name=intent_tool_name, # LangGraph will use this in direct_tool_executor_node
            controller_output_tool_input=intent_tool_input, # LangGraph will use this
            current_task_id=active_task_id,
            chat_history=session_data_entry["memory"].chat_memory.messages,
            # Other fields might not be strictly necessary for direct tool request but good to init
            plan_steps=[],
            current_step_index=0,
            retry_count_for_current_step=0,
            accumulated_plan_summary=f"Direct tool request: {intent_tool_name}",
            **filtered_initial_llm_ids
        )
        session_data_entry["callback_handler"].set_task_id(active_task_id)
        configurable_fields_for_run_tool = {
            "task_id": active_task_id, "session_id": session_id, **filtered_initial_llm_ids
        }
        runnable_config_tool = RunnableConfig(
            callbacks=[session_data_entry["callback_handler"]],
            configurable=configurable_fields_for_run_tool
        )
        logger.info(f"[{session_id}] Invoking research_agent_graph.astream_events for DIRECT_TOOL_REQUEST. Tool: '{intent_tool_name}'. Config: {runnable_config_tool.get('configurable')}")

        async def graph_streaming_task_direct_tool():
            try:
                async for event in research_agent_lg_graph.astream_events(
                    initial_graph_input_dict_tool, config=runnable_config_tool, version="v1"
                ):
                    event_node_name = event.get("name", "graph")
                    logger.debug(f"[{session_id}] Graph Event (TOOL): {event['event']} for Node: {event_node_name}, Tags: {event.get('tags')}")
                    if event["event"] == "on_chain_end" and event_node_name == "ResearchAgentGraph":
                        logger.info(f"[{session_id}] LangGraph stream finished for DIRECT_TOOL_REQUEST.")
            except AgentCancelledException:
                logger.warning(f"[{session_id}] DIRECT_TOOL_REQUEST execution cancelled by user (LangGraph).")
                await send_ws_message_func("status_message", "Tool request cancelled.")
                await add_monitor_log_func("Direct tool request cancelled by user (LangGraph).", "system_cancel")
            except Exception as e:
                logger.error(f"[{session_id}] Error during DIRECT_TOOL_REQUEST execution (LangGraph): {e}", exc_info=True)
                await add_monitor_log_func(f"Error during Direct Tool Request (LangGraph): {e}", "error_direct_tool")
                await send_ws_message_func("agent_message", f"Sorry, I encountered an error trying to use the tool '{intent_tool_name}': {e}")
                await send_ws_message_func("status_message", "Error during tool processing.")
            finally:
                connected_clients_entry["agent_task"] = None
                await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

        current_graph_task_tool = asyncio.create_task(graph_streaming_task_direct_tool())
        connected_clients_entry["agent_task"] = current_graph_task_tool
    # <<< END MODIFICATION >>>
    elif classified_intent_value == "DIRECT_QA":
        await send_ws_message_func("agent_thinking_update", {"status": "Processing directly (LangGraph)..."})
        await add_monitor_log_func(f"Handling as DIRECT_QA with LangGraph.", "system_direct_qa")

        initial_graph_input_dict_qa = ResearchAgentState(
            user_query=user_input_content,
            classified_intent="DIRECT_QA",
            current_task_id=active_task_id,
            chat_history=session_data_entry["memory"].chat_memory.messages,
            plan_steps=[],
            current_step_index=0,
            retry_count_for_current_step=0,
            accumulated_plan_summary="",
            **filtered_initial_llm_ids
        )
        session_data_entry["callback_handler"].set_task_id(active_task_id)
        configurable_fields_for_run_qa = {
            "task_id": active_task_id, "session_id": session_id, **filtered_initial_llm_ids
        }
        runnable_config_qa = RunnableConfig(
            callbacks=[session_data_entry["callback_handler"]],
            configurable=configurable_fields_for_run_qa
        )
        logger.info(f"[{session_id}] Invoking research_agent_graph.astream_events for DIRECT_QA. Input State (partial): user_query='{user_input_content}', classified_intent='DIRECT_QA'. Config: {runnable_config_qa.get('configurable')}")

        async def graph_streaming_task_direct_qa():
            try:
                async for event in research_agent_lg_graph.astream_events(
                    initial_graph_input_dict_qa, config=runnable_config_qa, version="v1"
                ):
                    event_node_name = event.get("name", "graph")
                    logger.debug(f"[{session_id}] Graph Event (QA): {event['event']} for Node: {event_node_name}, Tags: {event.get('tags')}")
                    if event["event"] == "on_chain_end" and event_node_name == "ResearchAgentGraph":
                        logger.info(f"[{session_id}] LangGraph stream finished for DIRECT_QA.")
            except AgentCancelledException:
                logger.warning(f"[{session_id}] DIRECT_QA execution cancelled by user (LangGraph).")
                await send_ws_message_func("status_message", "Direct QA cancelled.")
                await add_monitor_log_func("Direct QA cancelled by user (LangGraph).", "system_cancel")
            except Exception as e:
                logger.error(f"[{session_id}] Error during DIRECT_QA execution (LangGraph): {e}", exc_info=True)
                await add_monitor_log_func(f"Error during Direct QA (LangGraph): {e}", "error_direct_qa")
                await send_ws_message_func("agent_message", f"Sorry, I encountered an error trying to answer directly: {e}")
                await send_ws_message_func("status_message", "Error during direct processing.")
            finally:
                connected_clients_entry["agent_task"] = None
                await send_ws_message_func("agent_thinking_update", {"status": "Idle."})

        current_graph_task_qa = asyncio.create_task(graph_streaming_task_direct_qa())
        connected_clients_entry["agent_task"] = current_graph_task_qa
    else: # Fallback
        logger.error(f"[{session_id}] Fallback: classify_intent returned '{classified_intent_value}', which is not 'PLAN', 'DIRECT_QA', or 'DIRECT_TOOL_REQUEST'. Defaulting to planning.")
        await add_monitor_log_func(f"Error: classify_intent returned unknown value '{classified_intent_value}'. Defaulting to PLAN.", "error_system")
        await send_ws_message_func("agent_thinking_update", {"status": "Generating plan (fallback)..."})
        human_plan_summary, structured_plan_steps = await generate_plan(
            user_query=user_input_content, 
            session_data_entry=session_data_entry,
            available_tools_summary=tools_summary_for_prompt
        )
        if human_plan_summary and structured_plan_steps:
            session_data_entry["current_plan_human_summary"] = human_plan_summary
            session_data_entry["current_plan_structured"] = structured_plan_steps
            session_data_entry["current_plan_step_index"] = 0
            session_data_entry["plan_execution_active"] = False
            plan_timestamp_id_fallback = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            session_data_entry['current_plan_proposal_id_backend'] = plan_timestamp_id_fallback
            await send_ws_message_func("propose_plan_for_confirmation", {
                "human_summary": human_plan_summary,
                "structured_plan": structured_plan_steps,
                "plan_id": plan_timestamp_id_fallback
            })
            await add_monitor_log_func(f"Plan generated (fallback). Summary: {human_plan_summary}. Steps: {len(structured_plan_steps)}. Plan ID: {plan_timestamp_id_fallback}. Awaiting user confirmation.", "system_plan_generated")
            await send_ws_message_func("status_message", "Plan generated. Please review and confirm.")
            await send_ws_message_func("agent_thinking_update", {"status": "Awaiting plan confirmation..."})
        else:
            logger.error(f"[{session_id}] Failed to generate a plan (fallback) for query: {user_input_content}")
            await add_monitor_log_func(f"Error: Failed to generate a plan (fallback).", "error_system")
            await send_ws_message_func("status_message", "Error: Could not generate a plan for your request (fallback).")
            await send_ws_message_func("agent_message", "I'm sorry, I couldn't create a plan for that request. Please try rephrasing or breaking it down.")
            await send_ws_message_func("agent_thinking_update", {"status": "Planning failed."})


async def process_execute_confirmed_plan(
    session_id: str,
    data: Dict[str, Any],
    session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any],
    send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc,
    db_add_message_func: DBAddMessageFunc,
    research_agent_lg_graph: Any
) -> None:
    # ... (function content remains the same as in prompt.txt, with the LangGraph PLAN execution path) ...
    logger.info(f"[{session_id}] Received 'execute_confirmed_plan'.")
    active_task_id = session_data_entry.get("current_task_id")
    if not active_task_id:
        logger.warning(f"[{session_id}] 'execute_confirmed_plan' received but no active task.")
        await send_ws_message_func("status_message", "Error: No active task to execute plan for.")
        return

    confirmed_plan_steps_dicts = data.get("confirmed_plan")
    plan_id_from_frontend = data.get("plan_id")
    
    if plan_id_from_frontend != session_data_entry.get('current_plan_proposal_id_backend'):
        logger.error(f"[{session_id}] Plan ID mismatch! Frontend sent '{plan_id_from_frontend}', backend expected '{session_data_entry.get('current_plan_proposal_id_backend')}'. Aborting execution.")
        await send_ws_message_func("status_message", "Error: Plan confirmation ID mismatch. Please try again.")
        session_data_entry["current_plan_structured"] = None
        session_data_entry["current_plan_human_summary"] = None
        session_data_entry['current_plan_proposal_id_backend'] = None
        session_data_entry["plan_execution_active"] = False
        return

    if not confirmed_plan_steps_dicts or not isinstance(confirmed_plan_steps_dicts, list):
        logger.error(f"[{session_id}] Invalid or missing plan in 'execute_confirmed_plan' message. Data received: {data}")
        await send_ws_message_func("status_message", "Error: Invalid plan received for execution.")
        return

    session_data_entry["current_plan_structured"] = confirmed_plan_steps_dicts
    session_data_entry["current_plan_step_index"] = 0
    session_data_entry["plan_execution_active"] = True
    session_data_entry['cancellation_requested'] = False

    await add_monitor_log_func(f"User confirmed plan (ID: {plan_id_from_frontend}). Starting execution of {len(confirmed_plan_steps_dicts)} steps.", "system_plan_confirmed")
    await send_ws_message_func("status_message", "Plan confirmed. Executing steps...")

    plan_filename = f"_plan_{plan_id_from_frontend}.md"
    session_data_entry['active_plan_filename'] = plan_filename
    plan_markdown_content = [f"# Agent Plan for Task: {active_task_id}\n", f"## Plan ID: {plan_id_from_frontend}\n"]
    original_query_for_plan_file = session_data_entry.get('original_user_query', 'N/A')
    plan_markdown_content.append(f"## Original User Query:\n{original_query_for_plan_file}\n")
    plan_markdown_content.append(f"## Plan Summary (from Planner):\n{session_data_entry.get('current_plan_human_summary', 'N/A')}\n")
    plan_markdown_content.append("## Steps:\n")
    for i, step_data_dict in enumerate(confirmed_plan_steps_dicts):
        desc = step_data_dict.get('description', 'N/A') if isinstance(step_data_dict, dict) else 'N/A (Invalid Step Format)'
        tool_sugg = step_data_dict.get('tool_to_use', 'None') if isinstance(step_data_dict, dict) else 'N/A'
        input_instr = step_data_dict.get('tool_input_instructions', 'None') if isinstance(step_data_dict, dict) else 'N/A'
        expected_out = step_data_dict.get('expected_outcome', 'N/A') if isinstance(step_data_dict, dict) else 'N/A'
        plan_markdown_content.append(f"- [ ] {i+1}. **{desc}**")
        plan_markdown_content.append(f"    - Tool Suggestion (Planner): `{tool_sugg}`")
        plan_markdown_content.append(f"    - Input Instructions (Planner): `{input_instr}`")
        plan_markdown_content.append(f"    - Expected Outcome (Planner): `{expected_out}`\n")
    
    task_workspace_path = get_task_workspace_path(active_task_id)
    try:
        plan_file_path = task_workspace_path / plan_filename
        async with aiofiles.open(plan_file_path, 'w', encoding='utf-8') as f:
            await f.write("\n".join(plan_markdown_content))
        logger.info(f"[{session_id}] Saved confirmed plan to {plan_file_path}")
        await add_monitor_log_func(f"Confirmed plan saved to artifact: {plan_filename}", "system_info")
        await send_ws_message_func("trigger_artifact_refresh", {"taskId": active_task_id})
    except Exception as e:
        logger.error(f"[{session_id}] Failed to save plan to file '{plan_filename}': {e}", exc_info=True)
        await add_monitor_log_func(f"Error saving plan to file '{plan_filename}': {e}", "error_system")

    await send_ws_message_func("agent_thinking_update", {"status": "Executing plan (LangGraph)..."})
    await add_monitor_log_func(f"Executing PLAN intent with LangGraph for {len(confirmed_plan_steps_dicts)} steps.", "system_plan_execution_start")

    initial_llm_ids_for_state = {
        "intent_classifier_llm_id": session_data_entry.get("session_intent_classifier_llm_id"),
        "planner_llm_id": session_data_entry.get("session_planner_llm_id"),
        "controller_llm_id": session_data_entry.get("session_controller_llm_id"),
        "executor_llm_id": f"{session_data_entry.get('selected_llm_provider', settings.executor_default_provider)}::{session_data_entry.get('selected_llm_model_name', settings.executor_default_model_name)}",
        "evaluator_llm_id": session_data_entry.get("session_evaluator_llm_id"),
    }
    filtered_initial_llm_ids = {k: v for k, v in initial_llm_ids_for_state.items() if v is not None}

    initial_graph_input_plan = ResearchAgentState(
        user_query=session_data_entry.get('original_user_query', "No original query found"),
        classified_intent="PLAN",
        plan_steps=confirmed_plan_steps_dicts,
        plan_summary=session_data_entry.get("current_plan_human_summary", "No summary."),
        current_task_id=active_task_id,
        chat_history=session_data_entry["memory"].chat_memory.messages,
        current_step_index=0,
        retry_count_for_current_step=0,
        accumulated_plan_summary=f"Original Query: {session_data_entry.get('original_user_query', 'N/A')}\nPlan Summary: {session_data_entry.get('current_plan_human_summary', 'N/A')}\n--- Confirmed Plan Steps ---\n" + "".join([f"{i+1}. {s.get('description','N/A')}\n" for i, s in enumerate(confirmed_plan_steps_dicts)]),
        **filtered_initial_llm_ids
    )
    session_data_entry["callback_handler"].set_task_id(active_task_id)
    
    configurable_fields_for_run_plan = {
        "task_id": active_task_id, "session_id": session_id, **filtered_initial_llm_ids
    }
    runnable_config_plan = RunnableConfig(
        callbacks=[session_data_entry["callback_handler"]],
        configurable=configurable_fields_for_run_plan
    )
    logger.info(f"[{session_id}] Invoking research_agent_graph.astream_events for PLAN execution. Input State (partial): user_query='{initial_graph_input_plan.get('user_query')}', plan_steps_count={len(confirmed_plan_steps_dicts)}. Config: {runnable_config_plan.get('configurable')}")

    async def graph_streaming_task_plan():
        try:
            async for event in research_agent_lg_graph.astream_events(
                initial_graph_input_plan, config=runnable_config_plan, version="v1"
            ):
                event_node_name = event.get("name", "graph")
                logger.debug(f"[{session_id}] Graph Event (PLAN): {event['event']} for Node: {event_node_name}, Tags: {event.get('tags')}")
                if event["event"] == "on_chain_end" and event_node_name == "ResearchAgentGraph":
                    logger.info(f"[{session_id}] LangGraph stream finished for PLAN execution.")
        except AgentCancelledException:
            logger.warning(f"[{session_id}] PLAN execution cancelled by user (LangGraph).")
            await send_ws_message_func("status_message", "Plan execution cancelled.")
            await add_monitor_log_func("Plan execution cancelled by user (LangGraph).", "system_cancel")
            current_step_idx_on_cancel = session_data_entry.get("current_step_index", -1)
            if current_step_idx_on_cancel >= 0 and session_data_entry.get('active_plan_filename'):
                 await _update_plan_file_step_status(task_workspace_path, session_data_entry['active_plan_filename'], current_step_idx_on_cancel + 1, '!')
        except Exception as e:
            logger.error(f"[{session_id}] Error during PLAN execution (LangGraph): {e}", exc_info=True)
            await add_monitor_log_func(f"Error during Plan Execution (LangGraph): {e}", "error_plan_execution")
            await send_ws_message_func("agent_message", f"Sorry, I encountered an error during plan execution: {e}")
            await send_ws_message_func("status_message", "Error during plan execution.")
        finally:
            connected_clients_entry["agent_task"] = None
            session_data_entry["plan_execution_active"] = False
            await send_ws_message_func("agent_thinking_update", {"status": "Idle."})
            session_data_entry['current_plan_proposal_id_backend'] = None

    current_graph_task_plan = asyncio.create_task(graph_streaming_task_plan())
    connected_clients_entry["agent_task"] = current_graph_task_plan


async def process_cancel_plan_proposal(
    session_id: str, data: Dict[str, Any], session_data_entry: Dict[str, Any],
    connected_clients_entry: Dict[str, Any], send_ws_message_func: SendWSMessageFunc,
    add_monitor_log_func: AddMonitorLogFunc
) -> None:
    # ... (function content remains the same) ...
    logger.info(f"[{session_id}] Received 'cancel_plan_proposal'.")
    plan_id_to_cancel = data.get("plan_id") 
    backend_stored_plan_id = session_data_entry.get('current_plan_proposal_id_backend')

    if plan_id_to_cancel != backend_stored_plan_id:
        logger.warning(f"[{session_id}] Plan cancellation ID mismatch. Frontend sent '{plan_id_to_cancel}', backend had '{backend_stored_plan_id}'. Ignoring.")
        await add_monitor_log_func(f"Plan cancellation ID mismatch. Frontend: {plan_id_to_cancel}, Backend: {backend_stored_plan_id}. No action taken.", "warning_system")
        return

    session_data_entry["current_plan_human_summary"] = None
    session_data_entry["current_plan_structured"] = None
    session_data_entry["current_plan_step_index"] = -1
    session_data_entry["plan_execution_active"] = False
    session_data_entry['current_plan_proposal_id_backend'] = None 

    await add_monitor_log_func(f"User cancelled plan proposal (ID: {plan_id_to_cancel}).", "system_plan_cancelled")
    await send_ws_message_func("status_message", "Plan proposal cancelled by user.")
    await send_ws_message_func("agent_message", "Okay, the proposed plan has been cancelled.")
    await send_ws_message_func("agent_thinking_update", {"status": "Idle."})