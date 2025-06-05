# backend/langgraph_agent.py (Part 1/2)
import logging
from typing import TypedDict, Optional, List, Annotated, Dict, Any
# backend/langgraph_agent.py (Part 2/2)
# (Continued from Part 1)
import asyncio # For the __main__ test block

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate # For direct_qa_node
from langchain_core.runnables import RunnableConfig # For type hinting if needed
from langgraph.graph import StateGraph, END

from backend.config import settings
from backend.llm_setup import get_llm # To instantiate LLMs in nodes
from backend.tools import get_dynamic_tools # If DirectQANode needs tools

logger = logging.getLogger(__name__)

# --- Define State ---
class ResearchAgentState(TypedDict, total=False):
    user_query: str
    classified_intent: Optional[str]
    plan_steps: Optional[List[Dict[str, Any]]]
    current_step_index: int
    current_task_id: Optional[str]
    chat_history: Optional[List[BaseMessage]]
    
    controller_output_tool_name: Optional[str]
    controller_output_tool_input: Optional[str]
    executor_output: Optional[str] # Output from DirectQANode LLM or ExecutorNode LLM/tool
    previous_step_executor_output: Optional[str]
    
    step_evaluator_output: Optional[Dict[str, Any]]
    overall_evaluator_output: Optional[Dict[str, Any]] # Node that produces final AIMessage
    
    retry_count_for_current_step: int
    accumulated_plan_summary: str
    
    # LLM configuration to be used by nodes, passed from initial state or config
    # This allows nodes to select the correct LLM based on session settings
    intent_classifier_llm_id: Optional[str]
    planner_llm_id: Optional[str]
    controller_llm_id: Optional[str]
    executor_llm_id: Optional[str] # Used by DirectQANode and ExecutorNode
    evaluator_llm_id: Optional[str] # Used by StepEvaluatorNode and OverallEvaluatorNode

    is_direct_qa_flow: bool # Flag to signal a direct QA operation to OverallEvaluator


# --- Node Implementations ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Intent Classifier")
    # Actual intent classification logic would be here, using an LLM.
    # For now, we'll rely on the classified_intent passed in the initial state.
    # This node needs to ensure the LLM IDs for subsequent nodes are set in the state
    # if they are not already present from the initial graph input.
    
    # Retrieve LLM IDs from the config if available, falling back to state or defaults.
    # The RunnableConfig 'configurable' dict is where these should be.
    configurable_settings = config.get("configurable", {})

    return {
        "classified_intent": state.get("classified_intent", "DIRECT_QA"), # Assume this is pre-set for now
        "current_step_index": 0,
        "retry_count_for_current_step": 0,
        "is_direct_qa_flow": state.get("classified_intent") == "DIRECT_QA",
        # Ensure LLM IDs are in the state for other nodes to pick up
        "intent_classifier_llm_id": configurable_settings.get("intent_classifier_llm_id", state.get("intent_classifier_llm_id")),
        "planner_llm_id": configurable_settings.get("planner_llm_id", state.get("planner_llm_id")),
        "controller_llm_id": configurable_settings.get("controller_llm_id", state.get("controller_llm_id")),
        "executor_llm_id": configurable_settings.get("executor_llm_id", state.get("executor_llm_id")),
        "evaluator_llm_id": configurable_settings.get("evaluator_llm_id", state.get("evaluator_llm_id")),
    }

async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Direct QA")
    user_query = state.get("user_query", "No query provided.")
    chat_history = state.get("chat_history", [])
    
    # Get LLM configuration from state (which should have been populated by intent_classifier_node or initial input)
    llm_id_str = state.get("executor_llm_id") # Using executor_llm_id for Direct QA
    if not llm_id_str:
        logger.warning("DirectQANode: executor_llm_id not found in state. Using system default.")
        provider = settings.executor_default_provider
        model_name = settings.executor_default_model_name
    else:
        try:
            provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"DirectQANode: Invalid LLM ID format '{llm_id_str}'. Using system default.")
            provider = settings.executor_default_provider
            model_name = settings.executor_default_model_name

    logger.info(f"DirectQANode: Using LLM {provider}::{model_name}")
    
    try:
        llm = get_llm(settings, provider=provider, model_name=model_name, requested_for_role="DirectQA_Node")
    except Exception as e:
        logger.error(f"DirectQANode: Failed to initialize LLM: {e}")
        return {"executor_output": f"Error: Could not initialize LLM for Direct QA. {e}", "is_direct_qa_flow": True}

    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's query directly and concisely. Consider the chat history for context if relevant."),
        ("human", "Chat History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:")
    ])
    
    chain = prompt_template | llm
    
    # IMPORTANT: The graph's `RunnableConfig` (containing callbacks) is automatically
    # propagated to LLM calls made by nodes run within the graph.
    # So, `WebSocketCallbackHandler`'s `on_llm_start`/`on_llm_end` will be triggered.
    # However, on_llm_end in the callback by default sends a "monitor_log", not an "agent_message".
    # We need the OverallEvaluatorNode to produce the final AIMessage.
    
    logger.info(f"DirectQANode: Invoking LLM for query: '{user_query}'")
    try:
        # This LLM call will trigger callbacks, but its direct output is for the next node.
        response = await chain.ainvoke({"user_query": user_query, "chat_history": history_str}, config=config)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"DirectQANode: LLM produced answer (raw): {answer_text[:200]}...")
        return {"executor_output": answer_text, "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"executor_output": f"Error processing Direct QA: {e}", "is_direct_qa_flow": True}


async def placeholder_planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Planner (Placeholder)")
    plan = [{"step_id": 1, "description": "Placeholder: Do step 1", "tool_to_use": "some_tool", "expected_outcome": "Step 1 done"}]
    logger.info(f"Plan generated: {plan}")
    return {"plan_steps": plan, "current_step_index": 0, "retry_count_for_current_step": 0, "accumulated_plan_summary": "Plan:\n1. Do step 1\n", "is_direct_qa_flow": False}

async def placeholder_controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Controller (Placeholder)")
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    step_info = plan_steps[current_idx] if plan_steps and 0 <= current_idx < len(plan_steps) else {}
    logger.info(f"Controller for step {current_idx + 1}: {step_info.get('description')}")
    return {"controller_output_tool_name": step_info.get("tool_to_use", "None"), 
            "controller_output_tool_input": "placeholder_input_for_tool"}

async def placeholder_executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Executor (Placeholder)")
    tool_name = state.get("controller_output_tool_name", "None")
    tool_input = state.get("controller_output_tool_input", "")
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
    output = f"Placeholder output from {tool_name}."
    return {"executor_output": output}

async def placeholder_step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Step Evaluator (Placeholder)")
    current_idx = state.get("current_step_index", 0)
    executor_out = state.get("executor_output", "")
    eval_output = {"step_achieved_goal": True, "assessment_of_step": "Placeholder: Step looks good.", "is_recoverable_via_retry": False}
    logger.info(f"Step {current_idx + 1} evaluation: Achieved={eval_output['step_achieved_goal']}")
    new_accumulated_summary = state.get("accumulated_plan_summary", "") + f"Step {current_idx + 1} Output: {executor_out[:100]}\n"
    if eval_output["step_achieved_goal"]:
        return {"step_evaluator_output": eval_output, "previous_step_executor_output": executor_out, "retry_count_for_current_step": 0, "accumulated_plan_summary": new_accumulated_summary}
    else: 
        return {"step_evaluator_output": eval_output, "previous_step_executor_output": None, "accumulated_plan_summary": new_accumulated_summary + f"Step {current_idx + 1} Failed Assessment: {eval_output['assessment_of_step']}\n"}

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> NODE: Overall Evaluator")
    
    final_assessment_text = "Evaluation: Processing complete."
    is_direct_qa = state.get("is_direct_qa_flow", False)
    
    if is_direct_qa:
        raw_answer = state.get("executor_output", "No direct answer was generated.")
        logger.info(f"OverallEvaluatorNode: Processing direct QA response: {raw_answer[:200]}...")
        # This is where the final AIMessage needs to be generated and sent via callback
        # We'll use an LLM to "present" this answer, ensuring callbacks are triggered.
        final_assessment_text = raw_answer # For now, directly use it.
                                         # In a real scenario, you might format it or summarize if too long.
    else: # Plan execution flow
        logger.info(f"OverallEvaluatorNode: Processing plan execution outcome.")
        logger.info(f"Accumulated Plan Summary for Eval:\n{state.get('accumulated_plan_summary')}")
        # For plans, this node would typically use an LLM to assess the overall plan success
        # based on accumulated_plan_summary and the final executor_output from the last step.
        # For placeholder, use a default message.
        final_assessment_text = state.get("executor_output", "Plan execution completed. See logs for details.")

    # Use an LLM to "finalize" the assessment_text. This LLM call will trigger the WebSocketCallbackHandler.
    llm_id_str = state.get("evaluator_llm_id") # Use evaluator's LLM for this
    if not llm_id_str:
        logger.warning("OverallEvaluatorNode: evaluator_llm_id not found in state. Using system default.")
        provider = settings.evaluator_provider
        model_name = settings.evaluator_model_name
    else:
        try: provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"OverallEvaluatorNode: Invalid LLM ID format '{llm_id_str}'. Using system default.")
            provider = settings.evaluator_provider
            model_name = settings.evaluator_model_name
            
    logger.info(f"OverallEvaluatorNode: Using LLM {provider}::{model_name} to finalize output.")
    try:
        finalizing_llm = get_llm(settings, provider=provider, model_name=model_name, requested_for_role="OverallEvaluator_Finalize")
        
        # The prompt asks the LLM to essentially echo the input, ensuring an LLM call happens.
        # The WebSocketCallbackHandler's on_llm_end will pick up this invocation's result.
        prompt = ChatPromptTemplate.from_template("Present the following information as the agent's final response: {assessment_text}")
        chain = prompt | finalizing_llm
        
        # The result of this chain.ainvoke will be an AIMessage (or similar, depending on LLM)
        # and the callback handler will send its content as "agent_message"
        logger.info(f"OverallEvaluatorNode: Invoking LLM to finalize assessment: '{final_assessment_text[:100]}...'")
        final_response_message_obj = await chain.ainvoke({"assessment_text": final_assessment_text}, config=config)
        
        # The actual sending to UI is done by WebSocketCallbackHandler.
        # We just update the state here for completeness or if other parts of the graph need it.
        final_content_for_state = final_response_message_obj.content if hasattr(final_response_message_obj, 'content') else str(final_response_message_obj)
        logger.info(f"OverallEvaluatorNode: Finalized response for state: {final_content_for_state[:200]}...")
        return {"overall_evaluator_output": {"assessment": final_content_for_state, "overall_success": True}}

    except Exception as e:
        logger.error(f"OverallEvaluatorNode: Error during final LLM invocation: {e}", exc_info=True)
        # Fallback: Send the un-finalized text if LLM fails, though callback won't be ideal.
        # This case is problematic as the UI might not get the message correctly.
        # For robustness, the callback handler should also have a way to send raw text if needed.
        # For now, we rely on the LLM call succeeding.
        return {"overall_evaluator_output": {"assessment": f"Error finalizing response: {final_assessment_text}", "overall_success": False}}


# --- Conditional Edge Logic ---
def should_proceed_to_plan_or_qa(state: ResearchAgentState, config: RunnableConfig) -> str:
    logger.info("--- DECISION: Plan or Direct QA? ---")
    intent = state.get("classified_intent")
    if intent == "PLAN":
        logger.info(f"Intent is '{intent}', proceeding to Planner.")
        return "planner"
    elif intent == "DIRECT_QA":
        logger.info(f"Intent is '{intent}', proceeding to Direct QA.")
        return "direct_qa"
    else:
        logger.warning(f"Unknown intent '{intent}', defaulting to Direct QA.")
        return "direct_qa"

def should_retry_step_or_proceed(state: ResearchAgentState, config: RunnableConfig) -> str:
    logger.info("--- DECISION: Step Outcome - Retry, Next Step, or Evaluate Overall? ---")
    step_eval = state.get("step_evaluator_output", {})
    current_retries = state.get("retry_count_for_current_step", 0)
    max_retries = settings.agent_max_step_retries

    if not step_eval.get("step_achieved_goal", False):
        if step_eval.get("is_recoverable_via_retry", False) and current_retries < max_retries:
            logger.info(f"Step failed but is recoverable. Will attempt retry {current_retries + 1} of {max_retries}.")
            return "retry_step"
        else:
            logger.info("Step failed and is not recoverable or retries exhausted. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"
    else:
        logger.info("Step succeeded. Checking for more steps in the plan.")
        plan = state.get("plan_steps", [])
        current_idx = state.get("current_step_index", -1)
        if current_idx + 1 < len(plan):
            logger.info(f"More steps exist. Will proceed to step {current_idx + 2} (index {current_idx + 1}).")
            return "next_step"
        else:
            logger.info("No more steps in plan. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"

# --- Utility nodes for state updates ---
def increment_retry_count_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Incrementing Retry Count")
    current_retries = state.get("retry_count_for_current_step", 0)
    return {"retry_count_for_current_step": current_retries + 1}

def advance_to_next_step_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info(">>> UTILITY NODE: Advancing to Next Step")
    current_idx = state.get("current_step_index", -1)
    return {"current_step_index": current_idx + 1, "retry_count_for_current_step": 0}

# --- Graph Definition Function ---
def create_research_agent_graph():
    """
    Builds and compiles the LangGraph research agent.
    """
    logger.info("Building Research Agent Graph...")
    workflow = StateGraph(ResearchAgentState)

    # Add Nodes
    workflow.add_node("intent_classifier", intent_classifier_node) # Using the more functional version
    workflow.add_node("direct_qa", direct_qa_node)                 # Using the new functional version
    workflow.add_node("planner", placeholder_planner_node)
    workflow.add_node("controller", placeholder_controller_node)
    workflow.add_node("executor", placeholder_executor_node)
    workflow.add_node("step_evaluator", placeholder_step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node) # Using the adapted version
    
    workflow.add_node("increment_retry_count", increment_retry_count_node)
    workflow.add_node("advance_to_next_step", advance_to_next_step_node)

    # --- Define Edges ---
    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        should_proceed_to_plan_or_qa,
        {"planner": "planner", "direct_qa": "direct_qa"}
    )

    # Path for Direct QA: direct_qa_node's output (raw text) goes to overall_evaluator
    workflow.add_edge("direct_qa", "overall_evaluator") 

    # Path for Planning
    workflow.add_edge("planner", "controller")
    workflow.add_edge("controller", "executor")
    workflow.add_edge("executor", "step_evaluator")

    workflow.add_conditional_edges(
        "step_evaluator",
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
    app = workflow.compile()
    logger.info("Research Agent Graph compiled successfully.")
    return app

# --- Create and export the compiled graph instance ---
research_agent_graph = create_research_agent_graph()
logger.info(f"Module-level variable 'research_agent_graph' (compiled graph app) created. Type: {type(research_agent_graph)}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
    )
    logger.info("Running langgraph_agent.py directly for testing graph compilation and structure.")
    # ... (optional __main__ test block) ...
    logger.info("langgraph_agent.py (Part 2) finished execution if run as __main__.")
