# backend/langgraph_agent.py
import logging
import asyncio 

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END 
from typing import TypedDict, Optional, List, Dict as TypingDict, Union # Removed Annotated

from backend.config import settings 
from backend.tools import get_dynamic_tools 

# MODIFIED: Removed add_messages from this import as it's no longer in graph_state.py
from .graph_state import ResearchAgentState 
from .intent_classifier import classify_intent, IntentClassificationOutput 
from .planner import generate_plan 
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
        # Ensure existing_messages is a list before concatenation
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

# --- Graph Definition ---
# (Graph definition remains the same as Canvas 7 / previous version)
workflow_builder = StateGraph(ResearchAgentState)
workflow_builder.add_node("intent_classifier", intent_classifier_node)
workflow_builder.add_node("planner", planner_node)
workflow_builder.set_entry_point("intent_classifier")

def route_after_intent_classification(state: ResearchAgentState):
    intent = state.get("classified_intent")
    if intent == "PLAN":
        logger.info("Routing: Intent is PLAN, proceeding to planner.")
        return "planner"
    else: 
        logger.info(f"Routing: Intent is '{intent}', proceeding to END.")
        return END
workflow_builder.add_conditional_edges("intent_classifier", route_after_intent_classification, {"planner": "planner", END: END})
workflow_builder.add_edge("planner", END)

try:
    research_agent_graph = workflow_builder.compile()
    logger.info("ResearchAgent LangGraph compiled successfully.")
except Exception as e:
    logger.critical(f"Failed to compile ResearchAgent LangGraph: {e}", exc_info=True)
    research_agent_graph = None

# --- Test Runner ---
# (run_graph_example and __main__ block remain the same as Canvas 7 / previous version)
async def run_graph_example(user_input: str, ws_callback_handler: Optional[WebSocketCallbackHandler] = None):
    if not research_agent_graph:
        logger.error("Graph is not compiled. Cannot run example.")
        return None
    initial_state: ResearchAgentState = {
        "user_query": user_input,
        "messages": [HumanMessage(content=user_input)], 
        "task_id": "test_task_for_planner_node", 
        "classified_intent": None, 
        "intent_classifier_reasoning": None,
        "plan_summary": None,
        "plan_steps": None,
        "plan_generation_error": None,
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Full state after {node_name_that_ran}: {accumulated_state}")
            else: 
                if accumulated_state.get("messages"):
                    messages_list = accumulated_state["messages"]
                    if isinstance(messages_list, list) and messages_list:
                        last_message = messages_list[-1]
                        logger.info(f"    Last message: ({type(last_message).__name__}) {str(last_message.content)[:100]}...")
                if "classified_intent" in accumulated_state and accumulated_state["classified_intent"] is not None:
                    logger.info(f"    Intent: {accumulated_state['classified_intent']}")
                if "plan_summary" in accumulated_state and accumulated_state["plan_summary"] is not None:
                    logger.info(f"    Plan Summary: {str(accumulated_state['plan_summary'])[:100]}...")
                if "plan_generation_error" in accumulated_state and accumulated_state["plan_generation_error"] is not None:
                    logger.error(f"    Plan Gen Error: {accumulated_state['plan_generation_error']}")
                if "error_message" in accumulated_state and accumulated_state["error_message"] is not None:
                    logger.error(f"    General Error: {accumulated_state['error_message']}")
    logger.info("--- End of Graph Stream ---")
    if accumulated_state:
        logger.info("--- Final Accumulated Graph State (from run_graph_example) ---")
        logger.info(f"User Query: {accumulated_state.get('user_query')}")
        logger.info(f"Task ID: {accumulated_state.get('task_id')}")
        logger.info(f"Classified Intent: {accumulated_state.get('classified_intent')}")
        logger.info(f"Intent Reasoning: {accumulated_state.get('intent_classifier_reasoning')}")
        logger.info(f"Plan Summary: {accumulated_state.get('plan_summary')}")
        plan_steps_final = accumulated_state.get('plan_steps')
        if plan_steps_final:
            logger.info(f"Plan Steps ({len(plan_steps_final)}):")
            for i, step in enumerate(plan_steps_final):
                logger.info(f"  Step {i+1}: {step.get('description')} (Tool: {step.get('tool_to_use')})")
        logger.info(f"Plan Generation Error: {accumulated_state.get('plan_generation_error')}")
        messages_content = [f"({type(msg).__name__}) {msg.content}" for msg in accumulated_state.get('messages', []) if hasattr(msg, 'content')]
        logger.info(f"Final Messages: {messages_content}")
        logger.info(f"General Error (if any): {accumulated_state.get('error_message')}")
        return accumulated_state
    else:
        logger.warning("Graph stream (astream) did not yield any state chunks or final state could not be determined.")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    logger_langgraph = logging.getLogger("langgraph")
    logger_langgraph.setLevel(logging.INFO) 
    class SimpleTestCallbackHandler(WebSocketCallbackHandler):
        def __init__(self): super().__init__(session_id="test_session", send_ws_message_func=self._dummy_send, db_add_message_func=self._dummy_db_add, session_data_ref={})
        async def _dummy_send(self,t,c): print(f"MockWS: {t} - {str(c)[:50]}")
        async def _dummy_db_add(self,tid,sid,t,cs): print(f"MockDB: {tid},{t} - {cs[:50]}")
        async def on_llm_start(self,s:dict,p:List[str],**k): print(f"LLM Start: {p[0][:70]}...")
        async def on_llm_end(self,r,**k): 
            try: print(f"LLM End: {str(r.generations[0][0].text)[:70]}...")
            except Exception: print(f"LLM End: (Could not parse response text) Full: {str(r)[:100]}")
    async def run_main_test():
        try:
            from backend.config import settings as app_settings 
            if not app_settings.google_api_key: print("WARNING: API keys might not be loaded.")
        except ImportError: print("ERROR: Could not import app_settings."); return
        test_query_plan = "Research the benefits of solar power and write a short summary file called 'solar_benefits.txt'."
        print(f"\n--- Running LangGraph Agent Test (PLAN intent) for: '{test_query_plan}' ---")
        final_state_plan = await run_graph_example(test_query_plan) 
        if final_state_plan:
            print(f"\n--- Test Run Complete (PLAN intent) ---")
            print(f"Final Intent: {final_state_plan.get('classified_intent')}")
            assert final_state_plan.get('classified_intent') == "PLAN", f"Intent should be PLAN! Got: {final_state_plan.get('classified_intent')}"
            print(f"Plan Summary: {final_state_plan.get('plan_summary')}")
            if final_state_plan.get('plan_steps'): print(f"Number of plan steps: {len(final_state_plan.get('plan_steps'))}")
            else: print("No plan steps generated.")
            print(f"Plan Generation Error: {final_state_plan.get('plan_generation_error')}")
            final_messages = final_state_plan.get('messages', [])
            print(f"Number of final messages: {len(final_messages)}")
            assert len(final_messages) >= 3, f"Expected at least 3 messages (Human, Intent AI, Planner AI), got {len(final_messages)}"
            if len(final_messages) >= 1: assert isinstance(final_messages[0], HumanMessage), "First message should be HumanMessage"
            if len(final_messages) >= 2: assert "Intent Classified as: PLAN" in final_messages[1].content, "Second message should be Intent classification"
            if len(final_messages) >= 3: assert "Plan Generated:" in final_messages[2].content, "Third message should be Plan generation"
        else: print("\n--- Test Run Failed or No Final State (PLAN intent) ---")
        test_query_qa = "What is the capital of Spain?"
        print(f"\n--- Running LangGraph Agent Test (DIRECT_QA intent) for: '{test_query_qa}' ---")
        final_state_qa = await run_graph_example(test_query_qa)
        if final_state_qa:
            print(f"\n--- Test Run Complete (DIRECT_QA intent) ---")
            print(f"Final Intent: {final_state_qa.get('classified_intent')}")
            assert final_state_qa.get('classified_intent') == "DIRECT_QA", f"Intent should be DIRECT_QA! Got: {final_state_qa.get('classified_intent')}"
            print(f"Plan Summary (should be None): {final_state_qa.get('plan_summary')}")
            final_messages_qa = final_state_qa.get('messages', [])
            print(f"Number of final messages (QA): {len(final_messages_qa)}")
            assert len(final_messages_qa) >= 2, f"Expected at least 2 messages (Human, Intent AI), got {len(final_messages_qa)}"
        else: print("\n--- Test Run Failed or No Final State (DIRECT_QA intent) ---")
    asyncio.run(run_main_test())
