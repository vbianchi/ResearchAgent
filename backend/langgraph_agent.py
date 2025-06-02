# backend/langgraph_agent.py
import logging
import asyncio 

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END 
from typing import TypedDict, Optional, List, Dict as TypingDict, Union

from backend.config import settings 
from backend.tools import get_dynamic_tools 

from .graph_state import ResearchAgentState 
from .intent_classifier import classify_intent, IntentClassificationOutput 
from .planner import generate_plan 
from .controller import validate_and_prepare_step_action, ControllerOutput # Import controller function and model
from backend.callbacks import WebSocketCallbackHandler 

logger = logging.getLogger(__name__)

# --- Node Definitions ---
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
            user_query=user_query,
            available_tools_summary=None, 
            callback_handler=callback_handler_from_config
        )
        logger.info(f"Intent classification result: {classification_output.intent}, Reasoning: {classification_output.reasoning}")
        ai_message_content = f"Intent Classified as: {classification_output.intent}\nReasoning: {classification_output.reasoning or 'N/A'}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=ai_message_content)]
        return {
            **state, 
            "classified_intent": classification_output.intent,
            "intent_classifier_reasoning": classification_output.reasoning,
            "messages": new_messages, 
            "error_message": None 
        }
    except Exception as e:
        logger.error(f"Intent Classifier Node: Error during intent classification: {e}", exc_info=True)
        error_msg_content = f"System Error during intent classification: {str(e)}"
        safe_existing_messages = existing_messages if isinstance(existing_messages, list) else []
        new_messages = safe_existing_messages + [AIMessage(content=error_msg_content)]
        return {
            **state,
            "error_message": f"Error in intent classification: {str(e)}",
            "classified_intent": "PLAN", 
            "intent_classifier_reasoning": f"Error during intent classification: {str(e)}",
            "messages": new_messages
        }

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
        if not available_tools_summary:
            available_tools_summary = "No specific tools loaded. Agent will rely on general capabilities."
    except Exception as e:
        logger.error(f"Planner Node: Failed to load tools for summary: {e}", exc_info=True)
        available_tools_summary = "Error loading tool information."
    try:
        plan_result_dict = await generate_plan(
            user_query=user_query,
            available_tools_summary=available_tools_summary,
            callback_handler=callback_handler_from_config
        )
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
        return {
            **state,
            "plan_summary": plan_summary,
            "plan_steps": plan_steps, 
            "plan_generation_error": None,
            "messages": new_messages,
            "error_message": None 
        }
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
    # For this iteration, we'll process step 0.
    # In a full loop, current_step_index would be managed.
    current_step_idx = 0 # Hardcoded for first step processing
    
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        logger.warning("controller_node: 'messages' in state was not a list, resetting.")
        existing_messages = [msg for msg in existing_messages if hasattr(msg, 'content')] if existing_messages else []

    if not original_user_query:
        logger.error("Controller Node: Original user query missing from state.")
        new_messages = existing_messages + [AIMessage(content="System Error: Original query missing for Controller.")]
        return {**state, "controller_error": "Original user query missing.", "messages": new_messages}

    if not plan_steps or not isinstance(plan_steps, list) or current_step_idx >= len(plan_steps):
        logger.error(f"Controller Node: Invalid plan_steps or current_step_idx ({current_step_idx}). Plan steps: {plan_steps}")
        new_messages = existing_messages + [AIMessage(content="System Error: Invalid plan or step index for Controller.")]
        return {**state, "controller_error": "Invalid plan or step index.", "messages": new_messages}

    current_plan_step_dict = plan_steps[current_step_idx]
    logger.info(f"Controller Node: Processing Step {current_step_idx + 1}: {current_plan_step_dict.get('description')}")

    previous_step_output = state.get("previous_step_executor_output") # Will be None for the first step
    
    # Get available tools (similar to planner)
    current_task_id_for_tools = state.get("task_id")
    try:
        tools = get_dynamic_tools(current_task_id_for_tools)
        # available_tools_summary = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) # Not directly needed by controller function
    except Exception as e:
        logger.error(f"Controller Node: Failed to load tools: {e}", exc_info=True)
        new_messages = existing_messages + [AIMessage(content=f"System Error: Failed to load tools for Controller: {e}")]
        return {**state, "controller_error": f"Failed to load tools: {e}", "messages": new_messages}

    callback_handler_from_config = config.get("callbacks")
    # TODO: Get controller_llm_id_override from state if we add it to ResearchAgentState for session overrides
    # For now, validate_and_prepare_step_action will use its internal logic for LLM selection.
    
    try:
        controller_result: Dict[str, Any] = await validate_and_prepare_step_action(
            original_user_query=original_user_query,
            current_plan_step=current_plan_step_dict, # Pass the dict for the current step
            available_tools=tools,
            previous_step_executor_output=previous_step_output,
            controller_llm_id_override=None, # Pass override if available from state
            callback_handler=callback_handler_from_config
        )

        if controller_result.get("controller_error"):
            error_msg = f"Controller Error for Step {current_step_idx + 1}: {controller_result['controller_error']}"
            logger.error(error_msg)
            new_messages = existing_messages + [AIMessage(content=error_msg)]
            return {**state, **controller_result, "messages": new_messages} # Merge controller_result which includes the error

        ai_message_content = (
            f"Controller for Step {current_step_idx + 1}:\n"
            f"  Tool: {controller_result.get('controller_tool_name', 'None')}\n"
            f"  Input (summary): {str(controller_result.get('controller_tool_input', 'N/A'))[:100]}...\n"
            f"  Reasoning: {controller_result.get('controller_reasoning', 'N/A')}"
        )
        new_messages = existing_messages + [AIMessage(content=ai_message_content)]
        
        # Update state with all fields from controller_result and the new messages
        # Also ensure current_step_index is explicitly part of the returned state for clarity
        return {
            **state, 
            **controller_result, 
            "current_step_index": current_step_idx, # Though it was hardcoded, good to reflect
            "messages": new_messages,
            "error_message": None # Clear general error if controller itself succeeded
        }

    except Exception as e:
        logger.error(f"Controller Node: Unexpected error for step {current_step_idx + 1}: {e}", exc_info=True)
        error_msg_content = f"System Error in Controller Node for step {current_step_idx + 1}: {str(e)}"
        new_messages = existing_messages + [AIMessage(content=error_msg_content)]
        return {
            **state,
            "controller_error": f"Unexpected error in Controller: {str(e)}",
            "messages": new_messages
        }


# --- Graph Definition ---
workflow_builder = StateGraph(ResearchAgentState)
workflow_builder.add_node("intent_classifier", intent_classifier_node)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("controller", controller_node) # Add new controller node

workflow_builder.set_entry_point("intent_classifier")

def route_after_intent_classification(state: ResearchAgentState):
    intent = state.get("classified_intent")
    if intent == "PLAN":
        logger.info("Routing: Intent is PLAN, proceeding to planner.")
        return "planner"
    else: 
        logger.info(f"Routing: Intent is '{intent}', proceeding to END.")
        return END

def route_after_planner(state: ResearchAgentState):
    if state.get("plan_generation_error"):
        logger.error(f"Routing: Plan generation error detected: {state['plan_generation_error']}. Proceeding to END.")
        return END
    if not state.get("plan_steps"): # No steps generated, even if no explicit error
        logger.warning("Routing: No plan steps generated by planner. Proceeding to END.")
        return END
    logger.info("Routing: Plan generated successfully, proceeding to controller (for first step).")
    # Initialize current_step_index here or ensure controller_node handles first call
    # For this iteration, controller_node will assume index 0 if not set.
    # To be more explicit, we could add a small node or update state here.
    # For now, let's rely on controller_node to pick plan_steps[0].
    return "controller"


workflow_builder.add_conditional_edges(
    "intent_classifier",
    route_after_intent_classification,
    {"planner": "planner", END: END}
)
workflow_builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {"controller": "controller", END: END} # Route from planner
)

# For now, controller always goes to END. Later, it will loop or go to executor.
workflow_builder.add_edge("controller", END)


try:
    research_agent_graph = workflow_builder.compile()
    logger.info("ResearchAgent LangGraph compiled successfully with Controller node.")
except Exception as e:
    logger.critical(f"Failed to compile ResearchAgent LangGraph: {e}", exc_info=True)
    research_agent_graph = None

# --- Test Runner ---
async def run_graph_example(user_input: str, ws_callback_handler: Optional[WebSocketCallbackHandler] = None):
    if not research_agent_graph:
        logger.error("Graph is not compiled. Cannot run example.")
        return None
    initial_state: ResearchAgentState = {
        "user_query": user_input,
        "messages": [HumanMessage(content=user_input)], 
        "task_id": "test_task_for_controller_node", 
        "classified_intent": None, 
        "intent_classifier_reasoning": None,
        "plan_summary": None,
        "plan_steps": None,
        "plan_generation_error": None,
        "current_step_index": None, # Controller will pick 0 if this is None and plan_steps exist
        "previous_step_executor_output": None,
        "controller_tool_name": None,
        "controller_tool_input": None,
        "controller_reasoning": None,
        "controller_confidence": None,
        "controller_error": None,
        "error_message": None 
    }
    callbacks_to_use = []
    if ws_callback_handler: callbacks_to_use.append(ws_callback_handler)
    config_for_run = RunnableConfig(callbacks=callbacks_to_use)
    logger.info(f"Streaming LangGraph execution for query: '{user_input}' using astream()")
    accumulated_state: Optional[ResearchAgentState] = None 
    async for chunk in research_agent_graph.astream(initial_state, config=config_for_run):
        logger.info(f"--- Graph Stream Chunk ---")
        for node_name_that_ran, state_after_node_run in chunk.items():
            logger.info(f"State after Node '{node_name_that_ran}':")
            accumulated_state = state_after_node_run 
            # Log key fields for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Full state after {node_name_that_ran}: {accumulated_state}")
            else: 
                if "messages" in accumulated_state and accumulated_state["messages"]:
                    messages_list = accumulated_state["messages"]
                    if isinstance(messages_list, list) and messages_list:
                        last_message = messages_list[-1]
                        logger.info(f"    Last message: ({type(last_message).__name__}) {str(last_message.content)[:100]}...")
                # Log other key fields as needed for INFO level
                for key in ["classified_intent", "plan_summary", "controller_tool_name", "controller_reasoning", "plan_generation_error", "controller_error", "error_message"]:
                    if key in accumulated_state and accumulated_state[key] is not None:
                        logger.info(f"    {key.replace('_', ' ').title()}: {str(accumulated_state[key])[:100]}...")
                if "plan_steps" in accumulated_state and accumulated_state["plan_steps"] is not None:
                     logger.info(f"    Plan Steps count: {len(accumulated_state['plan_steps'])}")
    logger.info("--- End of Graph Stream ---")
    if accumulated_state:
        logger.info("--- Final Accumulated Graph State (from run_graph_example) ---")
        for key, value in accumulated_state.items():
            if key == "messages":
                messages_content = [f"({type(msg).__name__}) {msg.content}" for msg in value if hasattr(msg, 'content')]
                logger.info(f"Final Messages ({len(messages_content)}): {messages_content}")
            elif key == "plan_steps" and value:
                logger.info(f"Final Plan Steps ({len(value)}):")
                for i, step in enumerate(value): logger.info(f"  Step {i+1}: {step.get('description')} (Tool: {step.get('tool_to_use')})")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")
        return accumulated_state
    else:
        logger.warning("Graph stream (astream) did not yield any state chunks or final state could not be determined.")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    logger_langgraph = logging.getLogger("langgraph")
    logger_langgraph.setLevel(logging.INFO) 
    # SimpleTestCallbackHandler definition (can be copied from previous version if needed for deep logging)

    async def run_main_test():
        try:
            from backend.config import settings as app_settings 
            if not app_settings.google_api_key: print("WARNING: API keys might not be loaded.")
        except ImportError: print("ERROR: Could not import app_settings."); return

        test_query_plan = "Research the benefits of solar power and write a short summary file called 'solar_benefits.txt'."
        print(f"\n--- Running LangGraph Agent Test (PLAN intent, Controller) for: '{test_query_plan}' ---")
        final_state_plan = await run_graph_example(test_query_plan) 
        if final_state_plan:
            print(f"\n--- Test Run Complete (PLAN intent, Controller) ---")
            print(f"Final Intent: {final_state_plan.get('classified_intent')}")
            assert final_state_plan.get('classified_intent') == "PLAN", f"Intent should be PLAN! Got: {final_state_plan.get('classified_intent')}"
            print(f"Plan Summary: {final_state_plan.get('plan_summary')}")
            if final_state_plan.get('plan_steps'): 
                print(f"Number of plan steps: {len(final_state_plan.get('plan_steps'))}")
            else: 
                print("No plan steps generated.")
            print(f"Controller Tool Name (for first step): {final_state_plan.get('controller_tool_name')}")
            print(f"Controller Tool Input (summary for first step): {str(final_state_plan.get('controller_tool_input'))[:100]}...")
            print(f"Controller Reasoning (for first step): {final_state_plan.get('controller_reasoning')}")
            print(f"Controller Error: {final_state_plan.get('controller_error')}")
            print(f"Plan Generation Error: {final_state_plan.get('plan_generation_error')}")
            final_messages = final_state_plan.get('messages', [])
            print(f"Number of final messages: {len(final_messages)}")
            # Expect Human, Intent AI, Planner AI, Controller AI messages
            assert len(final_messages) >= 4, f"Expected at least 4 messages, got {len(final_messages)}"
        else: 
            print("\n--- Test Run Failed or No Final State (PLAN intent, Controller) ---")

        test_query_qa = "What is the capital of Spain?" # Should skip planner and controller
        print(f"\n--- Running LangGraph Agent Test (DIRECT_QA intent) for: '{test_query_qa}' ---")
        final_state_qa = await run_graph_example(test_query_qa)
        if final_state_qa:
            print(f"\n--- Test Run Complete (DIRECT_QA intent) ---")
            print(f"Final Intent: {final_state_qa.get('classified_intent')}")
            assert final_state_qa.get('classified_intent') == "DIRECT_QA", f"Intent should be DIRECT_QA! Got: {final_state_qa.get('classified_intent')}"
            print(f"Controller Tool Name (should be None): {final_state_qa.get('controller_tool_name')}")
        else: 
            print("\n--- Test Run Failed or No Final State (DIRECT_QA intent) ---")
            
    asyncio.run(run_main_test())
