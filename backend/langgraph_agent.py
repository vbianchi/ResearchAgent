# backend/langgraph_agent.py
import logging
import asyncio 

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END 
from typing import TypedDict, Optional, List, Dict as TypingDict, Union

from backend.config import settings 
from backend.tools import get_dynamic_tools 
from backend.llm_setup import get_llm 
from langchain_core.language_models.chat_models import BaseChatModel 

from .graph_state import ResearchAgentState 
from .intent_classifier import classify_intent, IntentClassificationOutput 
from .planner import generate_plan 
from .controller import validate_and_prepare_step_action, ControllerOutput 
from backend.callbacks import WebSocketCallbackHandler, LOG_SOURCE_EXECUTOR

logger = logging.getLogger(__name__)

# --- Node Definitions (intent_classifier_node, planner_node, controller_node, executor_node) ---
# These remain the same as the previous version (Canvas 10)

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Intent Classifier Node ---")
    user_query = state.get("user_query")
    existing_messages = state.get("messages", []) 
    if not isinstance(existing_messages, list):
        logger.warning("intent_classifier_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    if not user_query:
        logger.error("Intent Classifier Node: User query is missing from state.")
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content="System Error: User query was not found for intent classification.")]
        return {**state, "error_message": "User query is missing for intent classification.", "messages": new_messages}
    callback_handler_from_config = config.get("callbacks")
    try:
        classification_output: IntentClassificationOutput = await classify_intent(
            user_query=user_query, available_tools_summary=None, callback_handler=callback_handler_from_config)
        logger.info(f"Intent classification result: {classification_output.intent}, Reasoning: {classification_output.reasoning}")
        ai_message_content = f"Intent Classified as: {classification_output.intent}\nReasoning: {classification_output.reasoning or 'N/A'}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=ai_message_content)]
        return {**state, "classified_intent": classification_output.intent, "intent_classifier_reasoning": classification_output.reasoning, "messages": new_messages, "error_message": None }
    except Exception as e:
        logger.error(f"Intent Classifier Node: Error during intent classification: {e}", exc_info=True)
        error_msg_content = f"System Error during intent classification: {str(e)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "error_message": f"Error in intent classification: {str(e)}", "classified_intent": "PLAN", "intent_classifier_reasoning": f"Error during intent classification: {str(e)}", "messages": new_messages}

async def planner_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Planner Node ---")
    user_query = state.get("user_query")
    current_task_id_for_tools = state.get("task_id")
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("planner_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    logger.info(f"Planner Node: current_task_id_for_tools from state: {current_task_id_for_tools}")
    if not user_query:
        logger.error("Planner Node: User query is missing from state.")
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content="System Error: User query was not found for planning.")]
        return {**state, "plan_generation_error": "User query is missing for planning.", "messages": new_messages}
    callback_handler_from_config = config.get("callbacks")
    try:
        tools = get_dynamic_tools(current_task_id_for_tools) 
        available_tools_summary = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        if not available_tools_summary: available_tools_summary = "No specific tools loaded."
    except Exception as e:
        logger.error(f"Planner Node: Failed to load tools for summary: {e}", exc_info=True)
        available_tools_summary = "Error loading tool information."
    try:
        plan_result_dict = await generate_plan(
            user_query=user_query, available_tools_summary=available_tools_summary, callback_handler=callback_handler_from_config)
        if plan_result_dict.get("plan_generation_error"):
            error_content = f"Planning Error: {plan_result_dict['plan_generation_error']}"
            logger.error(f"Planner Node: Plan generation failed: {error_content}")
            safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
            new_messages = safe_existing_messages + [AIMessage(content=error_content)]
            return {**state, "plan_generation_error": plan_result_dict["plan_generation_error"], "messages": new_messages}
        plan_summary = plan_result_dict.get("plan_summary")
        plan_steps = plan_result_dict.get("plan_steps", [])
        logger.info(f"Planner Node: Plan generated. Summary: {plan_summary}")
        ai_message_content = f"Plan Generated:\nSummary: {plan_summary}\nNumber of steps: {len(plan_steps)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=ai_message_content)]
        return {**state, "plan_summary": plan_summary, "plan_steps": plan_steps, "plan_generation_error": None, "messages": new_messages, "error_message": None }
    except Exception as e:
        logger.error(f"Planner Node: Unexpected error during planning: {e}", exc_info=True)
        error_msg_content = f"System Error during planning: {str(e)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "plan_generation_error": f"Unexpected error in planner: {str(e)}", "messages": new_messages}

async def controller_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Controller Node ---")
    original_user_query = state.get("user_query")
    plan_steps = state.get("plan_steps")
    current_step_idx = 0 
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("controller_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    if not original_user_query:
        logger.error("Controller Node: Original user query missing from state.")
        new_messages = existing_messages + [AIMessage(content="System Error: Original query missing for Controller.")]
        return {**state, "controller_error": "Original user query missing.", "messages": new_messages}
    if not plan_steps or not isinstance(plan_steps, list) or current_step_idx >= len(plan_steps):
        logger.error(f"Controller Node: Invalid plan_steps or current_step_idx ({current_step_idx}).")
        new_messages = existing_messages + [AIMessage(content="System Error: Invalid plan or step index for Controller.")]
        return {**state, "controller_error": "Invalid plan or step index.", "messages": new_messages}
    current_plan_step_dict = plan_steps[current_step_idx]
    logger.info(f"Controller Node: Processing Step {current_step_idx + 1}: {current_plan_step_dict.get('description')}")
    previous_step_output = state.get("previous_step_executor_output")
    current_task_id_for_tools = state.get("task_id")
    try:
        tools = get_dynamic_tools(current_task_id_for_tools)
    except Exception as e:
        logger.error(f"Controller Node: Failed to load tools: {e}", exc_info=True)
        new_messages = existing_messages + [AIMessage(content=f"System Error: Failed to load tools for Controller: {e}")]
        return {**state, "controller_error": f"Failed to load tools: {e}", "messages": new_messages}
    callback_handler_from_config = config.get("callbacks")
    try:
        controller_result: Dict[str, Any] = await validate_and_prepare_step_action(
            original_user_query=original_user_query, current_plan_step=current_plan_step_dict,
            available_tools=tools, previous_step_executor_output=previous_step_output,
            controller_llm_id_override=None, callback_handler=callback_handler_from_config)
        if controller_result.get("controller_error"):
            error_msg = f"Controller Error for Step {current_step_idx + 1}: {controller_result['controller_error']}"
            logger.error(error_msg)
            new_messages = existing_messages + [AIMessage(content=error_msg)]
            return {**state, **controller_result, "messages": new_messages}
        ai_message_content = (f"Controller for Step {current_step_idx + 1}:\n"
            f"  Tool: {controller_result.get('controller_tool_name', 'None')}\n"
            f"  Input (summary): {str(controller_result.get('controller_tool_input', 'N/A'))[:100]}...\n"
            f"  Reasoning: {controller_result.get('controller_reasoning', 'N/A')}")
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        return {**state, **controller_result, "current_step_index": current_step_idx, "messages": new_messages, "error_message": None}
    except Exception as e:
        logger.error(f"Controller Node: Unexpected error for step {current_step_idx + 1}: {e}", exc_info=True)
        error_msg_content = f"System Error in Controller Node for step {current_step_idx + 1}: {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {**state, "controller_error": f"Unexpected error in Controller: {str(e)}", "messages": new_messages}

async def executor_node(state: ResearchAgentState, config: RunnableConfig):
    logger.info("--- Entering Executor Node ---")
    tool_name = state.get("controller_tool_name")
    tool_input_str = state.get("controller_tool_input")
    task_id = state.get("task_id")
    current_step_idx = state.get("current_step_index", 0) 
    plan_steps = state.get("plan_steps", [])
    step_description = "N/A"
    if plan_steps and current_step_idx < len(plan_steps):
        step_description = plan_steps[current_step_idx].get("description", "N/A")
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("executor_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []
    logger.info(f"Executor Node: Task ID '{task_id}', Step {current_step_idx + 1} ('{step_description}'). Tool: '{tool_name}', Input: '{str(tool_input_str)[:100]}...'")
    output = ""
    error_msg = None
    callback_handler_from_config = config.get("callbacks")
    if tool_name and tool_name.lower() != "none":
        try:
            available_tools = get_dynamic_tools(task_id)
            selected_tool = next((t for t in available_tools if t.name == tool_name), None)
            if selected_tool:
                logger.info(f"Executor Node: Executing tool '{selected_tool.name}'")
                output = await selected_tool.arun(tool_input=tool_input_str, callbacks=callback_handler_from_config) 
                logger.info(f"Executor Node: Tool '{selected_tool.name}' executed. Output length: {len(str(output))}")
            else:
                error_msg = f"Tool '{tool_name}' not found in available tools."
                logger.error(f"Executor Node: {error_msg}")
        except Exception as e:
            logger.error(f"Executor Node: Error executing tool '{tool_name}': {e}", exc_info=True)
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            output = f"Tool Execution Error: {str(e)}" 
    elif tool_input_str: 
        logger.info(f"Executor Node: 'None' tool specified. Executing direct LLM call with input: {tool_input_str[:100]}...")
        try:
            executor_llm: BaseChatModel = get_llm(
                settings, provider=settings.executor_default_provider, 
                model_name=settings.executor_default_model_name, 
                requested_for_role=LOG_SOURCE_EXECUTOR + "_DirectLLM",
                callbacks=callback_handler_from_config)
            llm_response = await executor_llm.ainvoke(
                [HumanMessage(content=tool_input_str)], 
                config=RunnableConfig(callbacks=callback_handler_from_config, metadata={"component_name": LOG_SOURCE_EXECUTOR + "_DirectLLM"}))
            output = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            logger.info(f"Executor Node: Direct LLM call completed. Output length: {len(output)}")
        except Exception as e:
            logger.error(f"Executor Node: Error during direct LLM call: {e}", exc_info=True)
            error_msg = f"Error during direct LLM call: {str(e)}"
            output = f"LLM Execution Error: {str(e)}"
    else:
        logger.warning("Executor Node: No tool name and no tool input provided.")
        output = "No action taken by executor (no tool or input)."
    ai_message_content = f"Executor (Step {current_step_idx + 1}):\nTool: {tool_name or 'None'}\nOutput: {str(output)[:500]}{'...' if len(str(output)) > 500 else ''}"
    if error_msg: ai_message_content += f"\nError: {error_msg}"
    new_messages = existing_messages + [AIMessage(content=ai_message_content)]
    return {**state, "current_executor_output": str(output), "executor_error_message": error_msg, "messages": new_messages, "error_message": None}

# --- Graph Definition ---
workflow_builder = StateGraph(ResearchAgentState)
workflow_builder.add_node("intent_classifier", intent_classifier_node)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("controller", controller_node)
workflow_builder.add_node("executor", executor_node) 

workflow_builder.set_entry_point("intent_classifier")

def route_after_intent_classification(state: ResearchAgentState):
    intent = state.get("classified_intent")
    if intent == "PLAN": return "planner"
    else: return END

def route_after_planner(state: ResearchAgentState):
    if state.get("plan_generation_error") or not state.get("plan_steps"): return END
    return "controller"

def route_after_controller(state: ResearchAgentState):
    if state.get("controller_error"):
        logger.error(f"Routing: Controller error detected: {state['controller_error']}. Ending.")
        return END
    logger.info("Routing: Controller finished, proceeding to executor.")
    return "executor"

workflow_builder.add_conditional_edges("intent_classifier", route_after_intent_classification, {"planner": "planner", END: END})
workflow_builder.add_conditional_edges("planner", route_after_planner, {"controller": "controller", END: END})
workflow_builder.add_conditional_edges("controller", route_after_controller, {"executor": "executor", END: END}) 
workflow_builder.add_edge("executor", END) 

try:
    research_agent_graph = workflow_builder.compile()
    logger.info("ResearchAgent LangGraph compiled successfully with Executor node.")
except Exception as e:
    logger.critical(f"Failed to compile ResearchAgent LangGraph: {e}", exc_info=True)
    research_agent_graph = None

# --- Test Runner ---
async def run_graph_example(user_input: str, ws_callback_handler: Optional[WebSocketCallbackHandler] = None):
    if not research_agent_graph: logger.error("Graph not compiled."); return None
    initial_state: ResearchAgentState = {
        "user_query": user_input, "messages": [HumanMessage(content=user_input)], 
        "task_id": "test_task_for_executor_node", "classified_intent": None, 
        "intent_classifier_reasoning": None, "plan_summary": None, "plan_steps": None,
        "plan_generation_error": None, "current_step_index": None, 
        "previous_step_executor_output": None, "controller_tool_name": None,
        "controller_tool_input": None, "controller_reasoning": None, 
        "controller_confidence": None, "controller_error": None,
        "current_executor_output": None, "executor_error_message": None,
        "error_message": None 
    }
    callbacks_to_use = [ws_callback_handler] if ws_callback_handler else []
    config_for_run = RunnableConfig(callbacks=callbacks_to_use)
    logger.info(f"Streaming LangGraph execution for query: '{user_input}'")
    accumulated_state: Optional[ResearchAgentState] = None 
    async for chunk in research_agent_graph.astream(initial_state, config=config_for_run):
        logger.info(f"--- Graph Stream Chunk ---")
        for node_name, state_after_run in chunk.items():
            logger.info(f"State after Node '{node_name}':")
            accumulated_state = state_after_run
            keys_to_log = ["classified_intent", "plan_summary", "controller_tool_name", "current_executor_output", "executor_error_message", "error_message", "plan_generation_error", "controller_error"]
            for key in keys_to_log:
                if key in accumulated_state and accumulated_state[key] is not None:
                    logger.info(f"    {key.replace('_', ' ').title()}: {str(accumulated_state[key])[:100]}...")
            if "messages" in accumulated_state and accumulated_state["messages"]:
                 logger.info(f"    Last Message: {str(accumulated_state['messages'][-1].content)[:70]}...")
    logger.info("--- End of Graph Stream ---")
    if accumulated_state:
        logger.info("--- Final Accumulated Graph State ---")
        for key, value in accumulated_state.items():
            if key == "messages": logger.info(f"Final Messages ({len(value)}): {[f'({type(m).__name__}) {m.content[:70]}...' for m in value]}")
            elif key == "plan_steps" and value: logger.info(f"Final Plan Steps ({len(value)}): {[s.get('description') for s in value]}")
            else: logger.info(f"{key.replace('_', ' ').title()}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
        return accumulated_state
    else: logger.warning("Graph stream did not yield final state."); return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    logger_langgraph = logging.getLogger("langgraph")
    logger_langgraph.setLevel(logging.INFO) 
    async def run_main_test():
        try:
            from backend.config import settings as app_settings 
            if not app_settings.google_api_key: print("WARNING: API keys might not be loaded.")
        except ImportError: print("ERROR: Could not import app_settings."); return

        # MODIFIED: Reverted to the solar power query for PLAN path testing
        test_query_plan_tool = "Research the benefits of solar power and write a short summary file called 'solar_benefits.txt'."
        print(f"\n--- Running LangGraph Test (PLAN, Controller, Executor - Tool) for: '{test_query_plan_tool}' ---")
        final_state_tool = await run_graph_example(test_query_plan_tool) 
        if final_state_tool:
            print(f"\n--- Test Run Complete (PLAN, Controller, Executor - Tool) ---")
            assert final_state_tool.get('classified_intent') == "PLAN", f"Intent should be PLAN! Got: {final_state_tool.get('classified_intent')}"
            # Check if controller selected a tool (it should for the first step of this plan)
            assert final_state_tool.get('controller_tool_name') is not None, "Controller should have selected a tool for the first step."
            print(f"Controller Tool: {final_state_tool.get('controller_tool_name')}")
            print(f"Executor Output: {str(final_state_tool.get('current_executor_output'))[:200]}...")
            print(f"Executor Error: {final_state_tool.get('executor_error_message')}")
            # Expect Human, Intent AI, Plan AI, Controller AI, Executor AI messages
            assert len(final_state_tool.get('messages', [])) >= 5, f"Expected at least 5 messages, got {len(final_state_tool.get('messages', []))}"
        else: print("\n--- Test Run Failed (PLAN, Controller, Executor - Tool) ---")

        test_query_qa = "What is the chemical symbol for water?"
        print(f"\n--- Running LangGraph Test (DIRECT_QA intent) for: '{test_query_qa}' ---")
        final_state_qa = await run_graph_example(test_query_qa)
        if final_state_qa:
            print(f"\n--- Test Run Complete (DIRECT_QA intent) ---")
            assert final_state_qa.get('classified_intent') == "DIRECT_QA", f"Intent should be DIRECT_QA! Got: {final_state_qa.get('classified_intent')}"
            assert final_state_qa.get('controller_tool_name') is None 
            assert final_state_qa.get('current_executor_output') is None 
        else: print("\n--- Test Run Failed (DIRECT_QA intent) ---")
            
    asyncio.run(run_main_test())
