# backend/langgraph_agent.py
import logging
import asyncio 

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END 
from typing import TypedDict, Annotated, Optional, List, Dict as TypingDict

from .graph_state import ResearchAgentState, add_messages 
from .intent_classifier import classify_intent, IntentClassificationOutput 
from backend.callbacks import WebSocketCallbackHandler 

logger = logging.getLogger(__name__)

# --- Node Definitions ---
async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig):
    """
    Classifies the user's intent.
    Updates the state with 'classified_intent' and 'intent_classifier_reasoning'.
    Appends an AIMessage to the 'messages' list in the state.
    """
    logger.info("--- Entering Intent Classifier Node ---")
    user_query = state.get("user_query")
    if not user_query:
        logger.error("Intent Classifier Node: User query is missing from state.")
        return {
            "error_message": "User query is missing for intent classification.",
            "messages": [AIMessage(content="System Error: User query was not found in the state for intent classification.")]
        }

    callback_handler_from_config = config.get("callbacks")
    if callback_handler_from_config:
        logger.debug(f"Intent Classifier Node: Received callbacks in config: {type(callback_handler_from_config)}")
    else:
        logger.debug("Intent Classifier Node: No callbacks received in config.")

    try:
        classification_output: IntentClassificationOutput = await classify_intent(
            user_query=user_query,
            available_tools_summary=None, 
            callback_handler=callback_handler_from_config
        )
        
        logger.info(f"Intent classification result: {classification_output.intent}, Reasoning: {classification_output.reasoning}")
        
        ai_message_content = f"Intent Classified as: {classification_output.intent}\nReasoning: {classification_output.reasoning or 'N/A'}"
        
        return {
            "classified_intent": classification_output.intent,
            "intent_classifier_reasoning": classification_output.reasoning,
            "messages": [AIMessage(content=ai_message_content)]
        }
    except Exception as e:
        logger.error(f"Intent Classifier Node: Error during intent classification: {e}", exc_info=True)
        return {
            "error_message": f"Error in intent classification: {str(e)}",
            "messages": [AIMessage(content=f"System Error during intent classification: {str(e)}")]
        }

# --- Graph Definition ---
workflow_builder = StateGraph(ResearchAgentState)

workflow_builder.add_node("intent_classifier", intent_classifier_node)
workflow_builder.set_entry_point("intent_classifier")
workflow_builder.add_edge("intent_classifier", END)

try:
    research_agent_graph = workflow_builder.compile()
    logger.info("ResearchAgent LangGraph compiled successfully with IntentClassifier node and direct END.")
except Exception as e:
    logger.critical(f"Failed to compile ResearchAgent LangGraph: {e}", exc_info=True)
    research_agent_graph = None

# --- Test Runner ---
async def run_graph_example(user_input: str, ws_callback_handler: Optional[WebSocketCallbackHandler] = None):
    """
    Example function to run the compiled LangGraph with a given user input.
    Uses astream() to get state snapshots after each node.
    """
    if not research_agent_graph:
        logger.error("Graph is not compiled. Cannot run example.")
        return None

    initial_state: ResearchAgentState = {
        "user_query": user_input,
        "messages": [HumanMessage(content=user_input)],
        "classified_intent": None, 
        "intent_classifier_reasoning": None,
        "error_message": None
    }

    callbacks_to_use = []
    if ws_callback_handler:
        callbacks_to_use.append(ws_callback_handler)

    config_for_run = RunnableConfig(
        callbacks=callbacks_to_use,
    )

    logger.info(f"Streaming LangGraph execution for query: '{user_input}' using astream()")
    final_state = None # Initialize final_state

    async for output_state_chunk in research_agent_graph.astream(initial_state, config=config_for_run):
        # Each `output_state_chunk` from astream() is the full state snapshot 
        # after a node (or set of parallel nodes) has completed.
        # The keys in the chunk indicate which node(s) just ran.
        logger.info("--- Graph Stream Chunk (State after Node Completion) ---")
        for node_name_that_ran, state_after_node in output_state_chunk.items():
            logger.info(f"State after Node '{node_name_that_ran}':")
            # For debugging, you can log parts of the state:
            if state_after_node.get("messages"):
                # Ensure messages is a list before trying to access its last element
                messages_list = state_after_node["messages"]
                if isinstance(messages_list, list) and messages_list:
                    last_message = messages_list[-1]
                    logger.debug(f"  Last message from state: ({type(last_message).__name__}) {str(last_message.content)[:100]}...")
                elif messages_list: # If not a list but exists
                    logger.debug(f"  Messages field (not list or empty): {str(messages_list)[:100]}...")

            if "classified_intent" in state_after_node and state_after_node["classified_intent"] is not None:
                logger.debug(f"  Intent in state: {state_after_node['classified_intent']}")
            if "error_message" in state_after_node and state_after_node["error_message"] is not None:
                logger.error(f"  Error in state: {state_after_node['error_message']}")
        
        # The last state snapshot yielded by astream is the final state of the graph.
        # We take the state from the first (and in this simple graph, only) node that ran in the chunk.
        if output_state_chunk:
            final_state = list(output_state_chunk.values())[0]


    logger.info("--- End of Graph Stream ---")

    if final_state:
        logger.info("--- Final Graph State (from run_graph_example using astream) ---")
        logger.info(f"User Query: {final_state.get('user_query')}")
        logger.info(f"Classified Intent: {final_state.get('classified_intent')}")
        logger.info(f"Intent Reasoning: {final_state.get('intent_classifier_reasoning')}")
        
        messages_content = []
        final_messages_list = final_state.get('messages', [])
        if isinstance(final_messages_list, list):
            for msg in final_messages_list:
                messages_content.append(f"({type(msg).__name__}) {msg.content}")
        else: # Handle if messages is not a list for some reason
             messages_content.append(f"(Messages field not a list: {str(final_messages_list)[:100]})")

        logger.info(f"Final Messages: {messages_content}")
        logger.info(f"Error (if any): {final_state.get('error_message')}")
        return final_state
    else:
        logger.warning("Graph stream (astream) did not yield any state chunks.")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    logger_langgraph = logging.getLogger("langgraph")
    logger_langgraph.setLevel(logging.INFO) # Can set to DEBUG for more LangGraph internal logs

    # SimpleTestCallbackHandler from previous version can be used if desired for deeper inspection
    class SimpleTestCallbackHandler(WebSocketCallbackHandler):
        def __init__(self):
            super().__init__(session_id="test_session", send_ws_message_func=self._dummy_send, db_add_message_func=self._dummy_db_add, session_data_ref={})
            self.log = []

        async def _dummy_send(self, msg_type, content):
            self.log.append(f"WS_SEND: {msg_type} - {str(content)[:100]}")
            # print(f"Mock WS Send: {msg_type} - {str(content)[:100]}") # Optional: print to console

        async def _dummy_db_add(self, task_id, session_id, msg_type, content_str):
            self.log.append(f"DB_ADD: {task_id}, {msg_type} - {content_str[:100]}")
            # print(f"Mock DB Add: {task_id}, {msg_type} - {content_str[:100]}") # Optional

        async def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs) -> None:
            msg = f"LLM Start: {prompts[0][:100]}..."
            self.log.append(msg)
            print(msg) # Make LLM calls visible for this test

        async def on_llm_end(self, response, **kwargs) -> None:
            # Assuming response.generations[0][0].text exists for this LLM type
            try:
                msg_content = response.generations[0][0].text
                msg = f"LLM End: {str(msg_content)[:100]}..."
                self.log.append(msg)
                print(msg)
            except (AttributeError, IndexError) as e:
                print(f"LLM End: (Could not parse response text: {e}) Full response: {str(response)[:200]}")


    async def run_main_test():
        try:
            from backend.config import settings as app_settings 
            if not app_settings.google_api_key:
                 print("WARNING: API keys might not be loaded for this standalone test. LLM calls might fail.")
        except ImportError:
            print("ERROR: Could not import app_settings from backend.config. LLM calls will likely fail.")
            return

        test_query = "What is the capital of France and its main attractions?"
        
        print(f"\n--- Running LangGraph Agent Test for: '{test_query}' ---")
        
        # To test with callbacks and see LLM start/end:
        # test_cb_handler = SimpleTestCallbackHandler()
        # final_state = await run_graph_example(test_query, test_cb_handler)
        
        # To run without the mock WS/DB callbacks (LLM provider's internal callbacks might still log):
        final_state = await run_graph_example(test_query) 

        if final_state:
            print(f"\n--- Test Run Complete (langgraph_agent.py) ---")
            print(f"Final Intent: {final_state.get('classified_intent')}")
            print(f"Final Reasoning: {final_state.get('intent_classifier_reasoning')}")
            print("-------------------------------------------------")
        else:
            print("\n--- Test Run Failed or No Final State (langgraph_agent.py) ---")

    asyncio.run(run_main_test())
