# backend/langgraph_agent.py
import logging
from typing import TypedDict, Optional, List, Annotated, Dict, Any

import asyncio # For the __main__ test block

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.planner import generate_plan # <<< Import generate_plan

logger = logging.getLogger(__name__)

# --- Define State ---
class ResearchAgentState(TypedDict, total=False): # [cite: 535]
    user_query: str
    classified_intent: Optional[str]
    plan_steps: Optional[List[Dict[str, Any]]] # [cite: 498]
    current_step_index: int # [cite: 535]
    current_task_id: Optional[str] # [cite: 498]
    chat_history: Optional[List[BaseMessage]]
    
    controller_output_tool_name: Optional[str]
    controller_output_tool_input: Optional[str]
    executor_output: Optional[str]
    previous_step_executor_output: Optional[str] # [cite: 498]
    
    step_evaluator_output: Optional[Dict[str, Any]]
    overall_evaluator_output: Optional[Dict[str, Any]]
    
    retry_count_for_current_step: int # [cite: 500]
    accumulated_plan_summary: str # [cite: 535]
    
    # LLM configuration to be used by nodes, passed from initial state or config
    intent_classifier_llm_id: Optional[str] # [cite: 536]
    planner_llm_id: Optional[str] # [cite: 536]
    controller_llm_id: Optional[str] # [cite: 536]
    executor_llm_id: Optional[str] # [cite: 536]
    evaluator_llm_id: Optional[str] # [cite: 536]

    is_direct_qa_flow: bool # [cite: 535]

    # Fields for PlannerNode output
    plan_summary: Optional[str] # [cite: 498]
    plan_generation_error: Optional[str] # For PlannerNode errors

# --- Node Implementations ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Intent Classifier") # [cite: 537]
    configurable_settings = config.get("configurable", {}) # [cite: 541]

    # Extract session-specific LLM overrides passed in initial graph state or config
    session_intent_llm_id = state.get("intent_classifier_llm_id") or configurable_settings.get("intent_classifier_llm_id")
    session_planner_llm_id = state.get("planner_llm_id") or configurable_settings.get("planner_llm_id")
    session_controller_llm_id = state.get("controller_llm_id") or configurable_settings.get("controller_llm_id")
    session_executor_llm_id = state.get("executor_llm_id") or configurable_settings.get("executor_llm_id")
    session_evaluator_llm_id = state.get("evaluator_llm_id") or configurable_settings.get("evaluator_llm_id")
    
    # The actual classification is assumed to have happened before this node (e.g., in agent_flow_handlers)
    # and `classified_intent` is passed in the initial state.
    # This node primarily ensures LLM IDs are set for subsequent nodes.
    return {
        "classified_intent": state.get("classified_intent", "DIRECT_QA"),
        "current_step_index": 0,
        "retry_count_for_current_step": 0,
        "is_direct_qa_flow": state.get("classified_intent") == "DIRECT_QA",
        "intent_classifier_llm_id": session_intent_llm_id, # [cite: 542]
        "planner_llm_id": session_planner_llm_id, # [cite: 542]
        "controller_llm_id": session_controller_llm_id, # [cite: 542]
        "executor_llm_id": session_executor_llm_id, # [cite: 542]
        "evaluator_llm_id": session_evaluator_llm_id, # [cite: 542]
    }

async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Direct QA")
    user_query = state.get("user_query", "No query provided.")
    chat_history = state.get("chat_history", [])
    
    llm_id_str = state.get("executor_llm_id") # [cite: 543]
    if not llm_id_str:
        logger.warning("DirectQANode: executor_llm_id not found in state. Using system default executor LLM.") # [cite: 544]
        provider = settings.executor_default_provider
        model_name = settings.executor_default_model_name
    else:
        try:
            provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"DirectQANode: Invalid LLM ID format '{llm_id_str}'. Using system default executor LLM.")
            provider = settings.executor_default_provider # [cite: 545]
            model_name = settings.executor_default_model_name
    
    logger.info(f"DirectQANode: Using LLM {provider}::{model_name}")
    
    try:
        # Pass callbacks from config to get_llm
        llm = get_llm(settings, provider=provider, model_name=model_name, 
                      callbacks=config.get("callbacks"), 
                      requested_for_role="DirectQA_Node")
    except Exception as e:
        logger.error(f"DirectQANode: Failed to initialize LLM: {e}")
        return {"executor_output": f"Error: Could not initialize LLM for Direct QA. {e}", "is_direct_qa_flow": True} # [cite: 546]

    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's query directly and concisely. Consider the chat history for context if relevant."),
        ("human", "Chat History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:")
    ])
    
    chain = prompt_template | llm
    
    logger.info(f"DirectQANode: Invoking LLM for query: '{user_query}'")
    try:
        response = await chain.ainvoke({"user_query": user_query, "chat_history": history_str}, config=config) # [cite: 548]
        answer_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"DirectQANode: LLM produced answer (raw): {answer_text[:200]}...")
        return {"executor_output": answer_text, "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"executor_output": f"Error processing Direct QA: {e}", "is_direct_qa_flow": True}

# <<< START NEW PlannerNode IMPLEMENTATION >>>
async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Planner")
    user_query = state.get("user_query")
    current_task_id = state.get("current_task_id")
    planner_llm_id_str = state.get("planner_llm_id") # Get LLM ID from state

    if not user_query:
        logger.error("PlannerNode: User query is missing from state.")
        return {"plan_steps": None, "plan_summary": None, "plan_generation_error": "User query is missing."}

    # Determine Planner LLM
    provider = settings.planner_provider
    model_name = settings.planner_model_name
    if planner_llm_id_str:
        try:
            parsed_provider, parsed_model_name = planner_llm_id_str.split("::", 1)
            provider = parsed_provider
            model_name = parsed_model_name
            logger.info(f"PlannerNode: Using LLM ID from state: {planner_llm_id_str}")
        except ValueError:
            logger.warning(f"PlannerNode: Invalid planner_llm_id format '{planner_llm_id_str}'. Falling back to system default for Planner.")
    else:
        logger.info(f"PlannerNode: planner_llm_id not found in state. Using system default for Planner: {provider}::{model_name}")

    try:
        # Pass callbacks from config to get_llm
        planner_llm_instance = get_llm(
            settings,
            provider=provider,
            model_name=model_name,
            callbacks=config.get("callbacks"), # Pass callbacks from graph's config
            requested_for_role="PlannerNode_LLM"
        )
    except Exception as e:
        logger.error(f"PlannerNode: Failed to initialize Planner LLM ({provider}::{model_name}): {e}", exc_info=True)
        return {
            "plan_steps": None, "plan_summary": None, 
            "plan_generation_error": f"Failed to initialize Planner LLM: {e}",
            "current_step_index": 0, "retry_count_for_current_step": 0, 
            "accumulated_plan_summary": "Error: Planning LLM failed to initialize.",
            "is_direct_qa_flow": False
        }

    # Get available tools summary
    available_tools = get_dynamic_tools(current_task_id=current_task_id)
    available_tools_summary = "\n".join(
        [f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools]
    )
    if not available_tools_summary:
        available_tools_summary = "No tools are currently available."
        logger.warning("PlannerNode: No tools available for planning.")
    
    # Get session_data_entry for generate_plan (though it's not directly used in the modified generate_plan for LLM selection)
    # However, generate_plan might evolve to use other parts of session_data_entry.
    # For now, we pass an empty dict if not available in state, or we assume it's part of `config.configurable`
    session_data_for_planner = {"session_planner_llm_id": planner_llm_id_str} # Minimal info it might use for logging its decision

    human_summary, structured_steps = await generate_plan(
        user_query=user_query,
        available_tools_summary=available_tools_summary,
        session_data_entry=session_data_for_planner, # Pass session data (might be used if generate_plan evolves)
        llm_instance=planner_llm_instance      # Pass the specific LLM instance
    )

    if human_summary and structured_steps:
        logger.info(f"PlannerNode: Plan generated successfully. Summary: {human_summary[:100]}...")
        initial_accumulated_summary = (
            f"Original Query: {user_query}\n"
            f"Overall Plan Summary: {human_summary}\n"
            f"--- Plan Steps ---\n"
        )
        for i, step in enumerate(structured_steps):
            initial_accumulated_summary += f"{i+1}. {step.get('description', 'N/A')}\n"
        
        return {
            "plan_steps": structured_steps,
            "plan_summary": human_summary,
            "plan_generation_error": None,
            "current_step_index": 0, # Start at the first step
            "retry_count_for_current_step": 0,
            "accumulated_plan_summary": initial_accumulated_summary,
            "is_direct_qa_flow": False
        }
    else:
        logger.error(f"PlannerNode: Failed to generate plan for query: {user_query}")
        error_msg = "Failed to generate a plan."
        # Check if planner_llm_instance itself was the issue from a previous log, though usually an exception
        return {
            "plan_steps": None,
            "plan_summary": None,
            "plan_generation_error": error_msg,
            "current_step_index": 0, # Or handle error state appropriately
            "retry_count_for_current_step": 0,
            "accumulated_plan_summary": f"Error: Planning failed. Query: {user_query}",
            "is_direct_qa_flow": False # Still part of a plan flow, even if it failed at planning
        }
# <<< END NEW PlannerNode IMPLEMENTATION >>>

async def placeholder_controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Controller (Placeholder)") # [cite: 550]
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    step_info = plan_steps[current_idx] if plan_steps and 0 <= current_idx < len(plan_steps) else {}
    logger.info(f"Controller for step {current_idx + 1}: {step_info.get('description')}")
    return {"controller_output_tool_name": step_info.get("tool_to_use", "None"), 
            "controller_output_tool_input": "placeholder_input_for_tool"}

async def placeholder_executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Executor (Placeholder)") # [cite: 551]
    tool_name = state.get("controller_output_tool_name", "None")
    tool_input = state.get("controller_output_tool_input", "")
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
    output = f"Placeholder output from {tool_name}."
    return {"executor_output": output}

async def placeholder_step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Step Evaluator (Placeholder)") # [cite: 552]
    current_idx = state.get("current_step_index", 0)
    executor_out = state.get("executor_output", "")
    eval_output = {"step_achieved_goal": True, "assessment_of_step": "Placeholder: Step looks good.", "is_recoverable_via_retry": False}
    logger.info(f"Step {current_idx + 1} evaluation: Achieved={eval_output['step_achieved_goal']}")
    new_accumulated_summary = state.get("accumulated_plan_summary", "") + f"Step {current_idx + 1} Output: {executor_out[:100]}\n" # [cite: 552]
    if eval_output["step_achieved_goal"]:
        return {"step_evaluator_output": eval_output, "previous_step_executor_output": executor_out, "retry_count_for_current_step": 0, "accumulated_plan_summary": new_accumulated_summary}
    else: 
        return {"step_evaluator_output": eval_output, "previous_step_executor_output": None, "accumulated_plan_summary": new_accumulated_summary + f"Step {current_idx + 1} Failed Assessment: {eval_output['assessment_of_step']}\n"}

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Overall Evaluator") # [cite: 553]
    
    final_assessment_text = "Evaluation: Processing complete."
    is_direct_qa = state.get("is_direct_qa_flow", False) # [cite: 553]
    
    if is_direct_qa:
        raw_answer = state.get("executor_output", "No direct answer was generated.") # [cite: 553]
        logger.info(f"OverallEvaluatorNode: Processing direct QA response: {raw_answer[:200]}...")
        final_assessment_text = raw_answer # [cite: 554]
    else: # Plan execution flow
        logger.info(f"OverallEvaluatorNode: Processing plan execution outcome.") # [cite: 555]
        logger.info(f"Accumulated Plan Summary for Eval:\n{state.get('accumulated_plan_summary')}")
        final_assessment_text = state.get("executor_output", "Plan execution completed. See logs for details.") # [cite: 556]

    llm_id_str = state.get("evaluator_llm_id") # [cite: 557]
    if not llm_id_str:
        logger.warning("OverallEvaluatorNode: evaluator_llm_id not found in state. Using system default.")
        provider = settings.evaluator_provider
        model_name = settings.evaluator_model_name
    else:
        try: provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"OverallEvaluatorNode: Invalid LLM ID format '{llm_id_str}'. Using system default.") # [cite: 558]
            provider = settings.evaluator_provider
            model_name = settings.evaluator_model_name
            
    logger.info(f"OverallEvaluatorNode: Using LLM {provider}::{model_name} to finalize output.")
    try:
        # Pass callbacks from config to get_llm
        finalizing_llm = get_llm(
            settings, provider=provider, model_name=model_name, 
            callbacks=config.get("callbacks"), # Pass graph's callbacks
            requested_for_role="OverallEvaluator_Finalize"
        )
        
        prompt = ChatPromptTemplate.from_template("Present the following information as the agent's final response: {assessment_text}") # [cite: 560]
        chain = prompt | finalizing_llm # [cite: 561]
        
        logger.info(f"OverallEvaluatorNode: Invoking LLM to finalize assessment: '{final_assessment_text[:100]}...'")
        final_response_message_obj = await chain.ainvoke({"assessment_text": final_assessment_text}, config=config) # Pass config to chain
        
        final_content_for_state = final_response_message_obj.content if hasattr(final_response_message_obj, 'content') else str(final_response_message_obj) # [cite: 563]
        logger.info(f"OverallEvaluatorNode: Finalized response for state: {final_content_for_state[:200]}...")
        return {"overall_evaluator_output": {"assessment": final_content_for_state, "overall_success": True}}

    except Exception as e:
        logger.error(f"OverallEvaluatorNode: Error during final LLM invocation: {e}", exc_info=True) # [cite: 564]
        return {"overall_evaluator_output": {"assessment": f"Error finalizing response: {final_assessment_text}", "overall_success": False}} # [cite: 567]


# --- Conditional Edge Logic ---
def should_proceed_to_plan_or_qa(state: ResearchAgentState) -> str: # Removed config as it's not used here
    logger.info("--- DECISION: Plan or Direct QA? ---")
    intent = state.get("classified_intent")
    if intent == "PLAN":
        logger.info(f"Intent is '{intent}', proceeding to Planner.")
        return "planner"
    elif intent == "DIRECT_QA":
        logger.info(f"Intent is '{intent}', proceeding to Direct QA.")
        return "direct_qa"
    # <<< START MODIFIED Fallback for failed plan generation >>>
    elif state.get("plan_generation_error"): # If planning failed in planner_node
        logger.warning(f"Planning failed with error: {state['plan_generation_error']}. Routing to overall_evaluator for error reporting.")
        # Ensure executor_output is set to the error for overall_evaluator to present
        state["executor_output"] = f"Planning failed: {state['plan_generation_error']}"
        state["is_direct_qa_flow"] = True # Treat as direct flow for error presentation
        return "overall_evaluator"
    # <<< END MODIFIED Fallback for failed plan generation >>>
    else:
        logger.warning(f"Unknown intent '{intent}', defaulting to Direct QA.") # [cite: 568]
        # Ensure executor_output is set to indicate an issue if intent is truly unknown
        state["executor_output"] = f"System Error: Intent classification resulted in an unknown state ('{intent}'). Cannot proceed."
        state["is_direct_qa_flow"] = True # Treat as direct flow for error presentation
        return "overall_evaluator" # Route to overall_evaluator to present the error


def should_retry_step_or_proceed(state: ResearchAgentState) -> str: # Removed config
    logger.info("--- DECISION: Step Outcome - Retry, Next Step, or Evaluate Overall? ---")
    step_eval = state.get("step_evaluator_output", {})
    current_retries = state.get("retry_count_for_current_step", 0)
    max_retries = settings.agent_max_step_retries

    if not step_eval.get("step_achieved_goal", False):
        if step_eval.get("is_recoverable_via_retry", False) and current_retries < max_retries:
            logger.info(f"Step failed but is recoverable. Will attempt retry {current_retries + 1} of {max_retries}.") # [cite: 569]
            return "retry_step"
        else:
            logger.info("Step failed and is not recoverable or retries exhausted. Proceeding to overall plan evaluation.")
            # Populate executor_output with error for overall_evaluator
            state["executor_output"] = state.get("accumulated_plan_summary", "") + \
                                     f"\nStep {state.get('current_step_index', 0) + 1} ultimately failed. " + \
                                     (step_eval.get('assessment_of_step') or "No specific assessment provided.")
            return "evaluate_overall_plan"
    else:
        logger.info("Step succeeded. Checking for more steps in the plan.")
        plan_steps = state.get("plan_steps", [])
        current_idx = state.get("current_step_index", -1) # [cite: 570]
        if current_idx + 1 < len(plan_steps):
            logger.info(f"More steps exist. Will proceed to step {current_idx + 2} (index {current_idx + 1}).")
            return "next_step"
        else:
            logger.info("No more steps in plan. Proceeding to overall plan evaluation.")
            # The final executor_output from the last successful step is already in state.executor_output
            return "evaluate_overall_plan" # [cite: 571]

# --- Utility nodes for state updates ---
def increment_retry_count_node(state: ResearchAgentState) -> Dict[str, Any]: # Removed config
    logger.info(">>> UTILITY NODE: Incrementing Retry Count")
    current_retries = state.get("retry_count_for_current_step", 0)
    return {"retry_count_for_current_step": current_retries + 1}

def advance_to_next_step_node(state: ResearchAgentState) -> Dict[str, Any]: # Removed config
    logger.info(">>> UTILITY NODE: Advancing to Next Step")
    current_idx = state.get("current_step_index", -1)
    return {"current_step_index": current_idx + 1, "retry_count_for_current_step": 0}

# --- Graph Definition Function ---
def create_research_agent_graph(): # [cite: 572]
    logger.info("Building Research Agent Graph...")
    workflow = StateGraph(ResearchAgentState)

    # Add Nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_qa", direct_qa_node)
    workflow.add_node("planner", planner_node) # <<< UPDATED from placeholder_planner_node
    workflow.add_node("controller", placeholder_controller_node)
    workflow.add_node("executor", placeholder_executor_node)
    workflow.add_node("step_evaluator", placeholder_step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    
    workflow.add_node("increment_retry_count", increment_retry_count_node) # [cite: 573]
    workflow.add_node("advance_to_next_step", advance_to_next_step_node) # [cite: 573]

    # --- Define Edges ---
    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        should_proceed_to_plan_or_qa,
        {
            "planner": "planner", 
            "direct_qa": "direct_qa",
            "overall_evaluator": "overall_evaluator" # Added edge for plan_generation_error
        }
    )
    
    # <<< START MODIFIED Planner to Controller or Overall Evaluator (if planning failed) >>>
    # If planner_node results in plan_generation_error, it should go to overall_evaluator.
    # Otherwise, it goes to controller. This conditional logic is now inside `should_proceed_to_plan_or_qa`
    # if we consider planning part of intent resolution.
    # Alternatively, add a dedicated conditional edge from planner if it can fail gracefully.
    # For now, `should_proceed_to_plan_or_qa` handles the error from planner if planner sets `plan_generation_error`
    # and the intent was PLAN. If planner itself is a separate step from intent classification's direct routing:
    workflow.add_conditional_edges(
        "planner",
        lambda state: "overall_evaluator" if state.get("plan_generation_error") else "controller",
        {
            "controller": "controller",
            "overall_evaluator": "overall_evaluator" # For plan generation failure
        }
    )
    # <<< END MODIFIED Planner Edge >>>


    workflow.add_edge("direct_qa", "overall_evaluator") 

    # Path for Planning (successful plan generation)
    # workflow.add_edge("planner", "controller") # This is now conditional from planner
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")

    workflow.add_conditional_edges(
        "step_evaluator", # [cite: 574]
        should_retry_step_or_proceed,
        {
            "retry_step": "increment_retry_count",
            "next_step": "advance_to_next_step",
            "evaluate_overall_plan": "overall_evaluator"
        }
    )
    
    workflow.add_edge("increment_retry_count", "controller")
    workflow.add_edge("advance_to_next_step", "controller")

    workflow.add_edge("overall_evaluator", END)

    logger.info("Compiling Research Agent Graph...")
    app = workflow.compile() # [cite: 575]
    logger.info("Research Agent Graph compiled successfully.")
    return app

# --- Create and export the compiled graph instance ---
research_agent_graph = create_research_agent_graph() # [cite: 576]
logger.info(f"Module-level variable 'research_agent_graph' (compiled graph app) created. Type: {type(research_agent_graph)}") # [cite: 576]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
    )
    logger.info("Running langgraph_agent.py directly for testing graph compilation and structure.")
    
    # Basic test for planner node invocation (conceptual)
    async def test_graph_with_plan_intent():
        print("\n--- Testing Graph with PLAN Intent ---")
        initial_state_plan = ResearchAgentState(
            user_query="Develop a strategy to market a new AI-powered coffee machine.",
            classified_intent="PLAN", # This would normally be set by an external classifier or first node
            current_task_id="test_task_001",
            chat_history=[],
            # Provide LLM IDs for the nodes
            planner_llm_id=f"{settings.planner_provider}::{settings.planner_model_name}", # Example
            # ... other LLM IDs if needed by other nodes in this specific test path
        )
        
        # Mock callbacks for testing
        class TestCallbackHandler:
            async def on_llm_start(self, serialized, prompts, **kwargs): print(f"  [Callback] LLM Start: {kwargs.get('metadata', {}).get('langgraph_node', 'UnknownNode')}, Prompts: {str(prompts)[:100]}...")
            async def on_llm_end(self, response, **kwargs): print(f"  [Callback] LLM End: {kwargs.get('metadata', {}).get('langgraph_node', 'UnknownNode')}, Response: {str(response.generations[0][0].text)[:100]}...")
            async def on_chain_start(self, serialized, inputs, **kwargs): print(f"  [Callback] Chain Start: {serialized.get('name')}, Node: {kwargs.get('metadata', {}).get('langgraph_node', 'UnknownNode')}")
            async def on_chain_end(self, outputs, **kwargs): print(f"  [Callback] Chain End: {kwargs.get('metadata', {}).get('langgraph_node', 'UnknownNode')}, Output Keys: {list(outputs.keys())}")
        
        test_callbacks = [TestCallbackHandler()]
        test_config = RunnableConfig(callbacks=test_callbacks, configurable={"task_id": "test_task_001"})

        print(f"Initial State for PLAN test: {initial_state_plan}")
        final_state = None
        try:
            async for event in research_agent_graph.astream_events(initial_state_plan, config=test_config, version="v1"):
                node_name = event.get("name", "graph_event")
                event_type = event.get("event")
                tags = event.get("tags", [])
                print(f"Event: {event_type} for Node/Event: {node_name}, Tags: {tags}")
                if event_type == "on_chain_end" and "data" in event and "output" in event["data"]:
                    print(f"  Output from {node_name}: {str(event['data']['output'])[:200]}...")
                    if node_name == "planner": # Check output of planner node
                        assert "plan_steps" in event["data"]["output"], "Planner output missing plan_steps"
                        assert "plan_summary" in event["data"]["output"], "Planner output missing plan_summary"
                        print(f"  Planner produced {len(event['data']['output'].get('plan_steps',[]))} steps.")
                if event_type == "on_graph_end":
                    final_state = event["data"]["output"]
            
            print("\n--- Final State after PLAN test ---")
            if final_state:
                print(f"  Plan Summary: {final_state.get('plan_summary')}")
                print(f"  Plan Steps: {final_state.get('plan_steps')}")
                print(f"  Plan Error: {final_state.get('plan_generation_error')}")
                print(f"  Overall Eval Assessment: {final_state.get('overall_evaluator_output', {}).get('assessment')}")

        except Exception as e:
            print(f"Error during PLAN graph execution test: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_graph_with_plan_intent())
    logger.info("langgraph_agent.py finished execution if run as __main__.")