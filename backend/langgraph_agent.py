# backend/langgraph_agent.py
import logging
from typing import TypedDict, Optional, List, Annotated, Dict, Any

import asyncio 

# --- Pydantic and LangChain Core Imports ---
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# --- Project Imports ---
from backend.config import settings
from backend.llm_setup import get_llm
from backend.tools import get_dynamic_tools
from backend.planner import generate_plan, PlanStep
from backend.graph_state import ResearchAgentState
from backend.controller import validate_and_prepare_step_action


logger = logging.getLogger(__name__)


# --- Node Implementations ---

async def intent_classifier_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Initial node that classifies the user's intent and initializes the graph state.
    It populates the state with any LLM overrides and sets up initial counters.
    """
    logger.info(">>> NODE: Intent Classifier")
    configurable_settings = config.get("configurable", {})

    # Extract session-specific LLM overrides from the state or config
    session_intent_llm_id = state.get("intent_classifier_llm_id") or configurable_settings.get("intent_classifier_llm_id")
    session_planner_llm_id = state.get("planner_llm_id") or configurable_settings.get("planner_llm_id")
    session_controller_llm_id = state.get("controller_llm_id") or configurable_settings.get("controller_llm_id")
    session_executor_llm_id = state.get("executor_llm_id") or configurable_settings.get("executor_llm_id")
    session_evaluator_llm_id = state.get("evaluator_llm_id") or configurable_settings.get("evaluator_llm_id")
    
    # Populate the classified intent into the state from the config if available
    classified_intent = configurable_settings.get("classified_intent", state.get("classified_intent", "DIRECT_QA"))
    
    # Initialize state fields required for all paths
    return {
        "classified_intent": classified_intent,
        "is_direct_qa_flow": classified_intent == "DIRECT_QA",
        # Pass LLM overrides into the state for subsequent nodes to use
        "intent_classifier_llm_id": session_intent_llm_id,
        "planner_llm_id": session_planner_llm_id,
        "controller_llm_id": session_controller_llm_id,
        "executor_llm_id": session_executor_llm_id,
        "evaluator_llm_id": session_evaluator_llm_id,
        # Initialize loop counters
        "current_step_index": 0,
        "retry_count_for_current_step": 0,
    }

async def direct_qa_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handles simple questions that don't require tools by directly calling an LLM.
    """
    logger.info(">>> NODE: Direct QA")
    user_query = state.get("user_query", "No query provided.")
    chat_history = state.get("chat_history", [])
    
    llm_id_str = state.get("executor_llm_id")
    if not llm_id_str:
        logger.warning("DirectQANode: executor_llm_id not found in state. Using system default executor LLM.")
        provider = settings.executor_default_provider
        model_name = settings.executor_default_model_name
    else:
        try:
            provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"DirectQANode: Invalid LLM ID format '{llm_id_str}'. Using system default executor LLM.")
            provider = settings.executor_default_provider
            model_name = settings.executor_default_model_name
    
    logger.info(f"DirectQANode: Using LLM {provider}::{model_name}")
    
    try:
        llm = get_llm(settings, provider=provider, model_name=model_name, 
                      callbacks=config.get("callbacks"), 
                      requested_for_role="DirectQA_Node")
    except Exception as e:
        logger.error(f"DirectQANode: Failed to initialize LLM: {e}")
        return {"overall_evaluation_error": f"Error: Could not initialize LLM for Direct QA. {e}"}

    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's query directly and concisely. Consider the chat history for context if relevant."),
        ("human", "Chat History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:")
    ])
    
    chain = prompt_template | llm
    
    logger.info(f"DirectQANode: Invoking LLM for query: '{user_query}'")
    try:
        response = await chain.ainvoke({"user_query": user_query, "chat_history": history_str}, config=config)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"DirectQANode: LLM produced answer (raw): {answer_text[:200]}...")
        return {"overall_evaluation_final_answer_content": answer_text, "is_direct_qa_flow": True}
    except Exception as e:
        logger.error(f"DirectQANode: Error during LLM invocation: {e}", exc_info=True)
        return {"overall_evaluation_error": f"Error processing Direct QA: {e}"}

async def planner_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a multi-step plan for complex user queries.
    """
    logger.info(">>> NODE: Planner")
    user_query = state.get("user_query")
    current_task_id = state.get("current_task_id")
    planner_llm_id_str = state.get("planner_llm_id")

    if not user_query:
        logger.error("PlannerNode: User query is missing from state.")
        return {"plan_generation_error": "User query is missing."}

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
        planner_llm_instance = get_llm(
            settings,
            provider=provider,
            model_name=model_name,
            callbacks=config.get("callbacks"),
            requested_for_role="PlannerNode_LLM"
        )
    except Exception as e:
        logger.error(f"PlannerNode: Failed to initialize Planner LLM ({provider}::{model_name}): {e}", exc_info=True)
        return { "plan_generation_error": f"Failed to initialize Planner LLM: {e}" }

    available_tools = get_dynamic_tools(current_task_id=current_task_id)
    available_tools_summary = "\n".join(
        [f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools]
    )
    if not available_tools_summary:
        available_tools_summary = "No tools are currently available."
        logger.warning("PlannerNode: No tools available for planning.")
    
    session_data_for_planner = {"session_planner_llm_id": planner_llm_id_str}

    human_summary, structured_steps = await generate_plan(
        user_query=user_query,
        available_tools_summary=available_tools_summary,
        session_data_entry=session_data_for_planner,
        llm_instance=planner_llm_instance
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
            "accumulated_plan_summary": initial_accumulated_summary,
            "is_direct_qa_flow": False
        }
    else:
        logger.error(f"PlannerNode: Failed to generate plan for query: {user_query}")
        error_msg = "Failed to generate a plan."
        return {
            "plan_generation_error": error_msg,
        }

async def controller_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Takes a single step from the plan and determines the precise tool and input for the Executor.
    """
    logger.info(">>> NODE: Controller")
    
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    user_query = state.get("user_query")
    previous_step_output = state.get("previous_step_executor_output")
    current_task_id = state.get("current_task_id")
    controller_llm_id_str = state.get("controller_llm_id")

    if not plan_steps or not (0 <= current_idx < len(plan_steps)):
        logger.error(f"ControllerNode: Invalid plan or step index. Index: {current_idx}, Plan Steps: {len(plan_steps)}")
        return {"controller_error": "Controller error: Invalid plan or step index."}
    
    step_info_dict = plan_steps[current_idx]
    try:
        current_plan_step = PlanStep(**step_info_dict)
    except Exception as e:
        logger.error(f"ControllerNode: Could not parse step dictionary into PlanStep model: {e}")
        return {"controller_error": f"Could not parse plan step {current_idx + 1}: {e}"}

    logger.info(f"Controller for step {current_idx + 1}: {current_plan_step.description}")

    available_tools = get_dynamic_tools(current_task_id=current_task_id)
    session_data_for_controller = {"session_controller_llm_id": controller_llm_id_str}

    controller_output = await validate_and_prepare_step_action(
        original_user_query=user_query,
        plan_step=current_plan_step,
        available_tools=available_tools,
        session_data_entry=session_data_for_controller,
        previous_step_output=previous_step_output
    )

    if controller_output:
        logger.info(f"ControllerNode: Action prepared: Tool='{controller_output.tool_name}', Input='{str(controller_output.tool_input)[:100]}...'")
        return {
            "controller_tool_name": controller_output.tool_name,
            "controller_tool_input": controller_output.tool_input,
            "controller_reasoning": controller_output.reasoning,
            "controller_confidence": controller_output.confidence_score,
            "controller_error": None
        }
    else:
        logger.error(f"ControllerNode: validate_and_prepare_step_action returned None. Controller failed.")
        return {
            "controller_error": "Controller failed to produce a valid action for the current step.",
            "controller_tool_name": None,
            "controller_tool_input": None
        }

# --- START: New Executor Node Implementation ---
async def executor_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes the action determined by the Controller, either a tool call or a direct LLM call.
    This replaces the placeholder.
    """
    logger.info(">>> NODE: Executor")
    tool_name = state.get("controller_tool_name")
    tool_input = state.get("controller_tool_input")
    current_task_id = state.get("current_task_id")

    try:
        if tool_name and tool_name.lower() != "none":
            # --- Tool Execution Path ---
            logger.info(f"ExecutorNode: Attempting to execute tool '{tool_name}' with input: '{str(tool_input)[:100]}...'")
            available_tools = get_dynamic_tools(current_task_id=current_task_id)
            tool_to_execute = next((t for t in available_tools if t.name == tool_name), None)

            if not tool_to_execute:
                error_msg = f"Executor error: Tool '{tool_name}' not found in the available tools."
                logger.error(error_msg)
                return {"executor_error_message": error_msg}

            # Execute the tool's async `arun` method
            # The tool itself will handle callbacks if run_manager is passed,
            # but top-level tool start/end is already handled by the graph's main callback handler.
            output = await tool_to_execute.arun(tool_input, callbacks=config.get("callbacks"))
            logger.info(f"ExecutorNode: Tool '{tool_name}' executed. Output length: {len(str(output))}")
            return {"executor_output": output}
        else:
            # --- "No Tool" / Direct LLM Execution Path ---
            logger.info(f"ExecutorNode: No tool specified. Executing direct LLM call with directive: '{str(tool_input)[:100]}...'")
            llm_id_str = state.get("executor_llm_id")
            if not llm_id_str:
                logger.warning("ExecutorNode (LLM): executor_llm_id not found in state. Using system default.")
                provider, model_name = settings.executor_default_provider, settings.executor_default_model_name
            else:
                try:
                    provider, model_name = llm_id_str.split("::", 1)
                except ValueError:
                    logger.warning(f"ExecutorNode (LLM): Invalid LLM ID '{llm_id_str}'. Using system default.")
                    provider, model_name = settings.executor_default_provider, settings.executor_default_model_name
            
            executor_llm = get_llm(settings, provider=provider, model_name=model_name, 
                                   callbacks=config.get("callbacks"), requested_for_role="ExecutorNode_LLM")
            
            # Prepare a prompt for the LLM
            previous_step_output_context = state.get("previous_step_executor_output", "No output from previous step is available.")
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Execute the following instruction. Use the provided 'Previous Step Output' as your primary data source if the instruction refers to it."),
                ("human", "Previous Step Output:\n---\n{previous_step_output}\n---\n\nInstruction for this step:\n{instruction}")
            ])
            chain = prompt_template | executor_llm
            
            response = await chain.ainvoke({
                "previous_step_output": previous_step_output_context,
                "instruction": tool_input
            }, config=config)

            llm_output = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"ExecutorNode (LLM): Direct call completed. Output length: {len(llm_output)}")
            return {"executor_output": llm_output}

    except Exception as e:
        logger.error(f"ExecutorNode: An error occurred during execution. Tool: '{tool_name}'. Error: {e}", exc_info=True)
        return {"executor_error_message": f"Error during execution: {e}"}
# --- END: New Executor Node Implementation ---


async def placeholder_step_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    (Placeholder) Evaluates the outcome of the Executor's action.
    """
    logger.info(">>> NODE: Step Evaluator (Placeholder)")
    current_idx = state.get("current_step_index", 0)
    executor_out = state.get("executor_output", "")
    eval_output = {"step_achieved_goal": True, "assessment_of_step": "Placeholder: Step looks good.", "is_recoverable_via_retry": False}
    logger.info(f"Step {current_idx + 1} evaluation: Achieved={eval_output['step_achieved_goal']}")
    
    # Update the accumulated summary with the output of the current step
    accumulated_summary = state.get("accumulated_plan_summary", "")
    current_step_info = state.get("plan_steps", [])[current_idx]
    accumulated_summary += f"\n--- Step {current_idx + 1}: {current_step_info.get('description')} ---\n"
    accumulated_summary += f"Action: Tool='{state.get('controller_tool_name', 'N/A')}', Input='{str(state.get('controller_tool_input', 'N/A'))[:150]}...'\n"
    accumulated_summary += f"Output: {executor_out[:200]}...\n"

    if eval_output["step_achieved_goal"]:
        return {
            "step_evaluator_output": eval_output, 
            "previous_step_executor_output": executor_out, 
            "retry_count_for_current_step": 0, 
            "accumulated_plan_summary": accumulated_summary
        }
    else: 
        return {
            "step_evaluator_output": eval_output, 
            "previous_step_executor_output": None, # Don't pass failed output to next step
            "accumulated_plan_summary": accumulated_summary + f"Assessment: Step FAILED. {eval_output.get('assessment_of_step', '')}\n"
        }

async def overall_evaluator_node(state: ResearchAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Final node that synthesizes all information to provide a coherent answer to the user.
    """
    logger.info(">>> NODE: Overall Evaluator")
    
    final_assessment_text = "Evaluation: Processing complete."
    is_direct_qa = state.get("is_direct_qa_flow", False)
    plan_generation_error = state.get("plan_generation_error")
    direct_qa_answer = state.get("overall_evaluation_final_answer_content")
    
    if plan_generation_error:
        logger.warning(f"OverallEvaluatorNode: Processing a plan generation failure: {plan_generation_error}")
        final_assessment_text = f"I'm sorry, I was unable to create a plan for your request. Error: {plan_generation_error}"
    elif is_direct_qa and direct_qa_answer:
        logger.info(f"OverallEvaluatorNode: Processing direct QA response: {direct_qa_answer[:200]}...")
        final_assessment_text = direct_qa_answer
    else:
        logger.info(f"OverallEvaluatorNode: Processing plan execution outcome.")
        logger.info(f"Accumulated Plan Summary for Eval:\n{state.get('accumulated_plan_summary')}")
        # The final output of the last successful step is now in 'executor_output'
        final_assessment_text = state.get("executor_output", "Plan execution completed. See logs for details.")

    llm_id_str = state.get("evaluator_llm_id")
    if not llm_id_str:
        logger.warning("OverallEvaluatorNode: evaluator_llm_id not found in state. Using system default.")
        provider, model_name = settings.evaluator_provider, settings.evaluator_model_name
    else:
        try: provider, model_name = llm_id_str.split("::", 1)
        except ValueError:
            logger.warning(f"OverallEvaluatorNode: Invalid LLM ID format '{llm_id_str}'. Using system default.")
            provider, model_name = settings.evaluator_provider, settings.evaluator_model_name
            
    logger.info(f"OverallEvaluatorNode: Using LLM {provider}::{model_name} to finalize output.")
    try:
        finalizing_llm = get_llm(
            settings, provider=provider, model_name=model_name, 
            callbacks=config.get("callbacks"),
            requested_for_role="OverallEvaluatorNode"
        )
        
        prompt = ChatPromptTemplate.from_template("Present the following information as the agent's final response, without adding any conversational phrases like 'Here is the answer'. Just provide the information directly:\n\n{assessment_text}")
        chain = prompt | finalizing_llm
        
        logger.info(f"OverallEvaluatorNode: Invoking LLM to finalize assessment: '{final_assessment_text[:100]}...'")
        await chain.ainvoke({"assessment_text": final_assessment_text}, config=config)
        return {}

    except Exception as e:
        logger.error(f"OverallEvaluatorNode: Error during final LLM invocation: {e}", exc_info=True)
        return {"overall_evaluation_error": f"Error finalizing response: {e}"}


# --- Conditional Edge Logic ---

def should_proceed_to_plan_or_qa(state: ResearchAgentState) -> str:
    """Decision point after intent classification."""
    logger.info("--- DECISION: Plan or Direct QA? ---")
    intent = state.get("classified_intent", "DIRECT_QA")
    if intent == "PLAN":
        logger.info(f"Intent is '{intent}', proceeding to Planner.")
        return "planner"
    else:
        logger.info(f"Intent is '{intent}', proceeding to Direct QA / Tool execution path.")
        return "direct_qa"
        
def did_planning_succeed(state: ResearchAgentState) -> str:
    """Decision point after the planner runs."""
    logger.info("--- DECISION: Planning Success or Failure? ---")
    if state.get("plan_generation_error"):
        logger.warning(f"Planning failed: {state['plan_generation_error']}. Routing to overall_evaluator.")
        return "failure"
    else:
        logger.info("Planning succeeded. Routing to controller.")
        return "success"

def should_retry_step_or_proceed(state: ResearchAgentState) -> str:
    """Decision point after a step is evaluated."""
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
            state["executor_output"] = state.get("accumulated_plan_summary", "") + \
                                     f"\nStep {state.get('current_step_index', 0) + 1} ultimately failed. " + \
                                     (step_eval.get('assessment_of_step') or "No specific assessment provided.")
            return "evaluate_overall_plan"
    else:
        logger.info("Step succeeded. Checking for more steps in the plan.")
        plan_steps = state.get("plan_steps", [])
        current_idx = state.get("current_step_index", -1)
        if current_idx + 1 < len(plan_steps):
            logger.info(f"More steps exist. Will proceed to step {current_idx + 2} (index {current_idx + 1}).")
            return "next_step"
        else:
            logger.info("No more steps in plan. Proceeding to overall plan evaluation.")
            return "evaluate_overall_plan"

# --- Utility nodes for state updates ---
def increment_retry_count_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Increments the retry counter for the current step."""
    logger.info(">>> UTILITY NODE: Incrementing Retry Count")
    current_retries = state.get("retry_count_for_current_step", 0)
    return {"retry_count_for_current_step": current_retries + 1}

def advance_to_next_step_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Advances the step index and resets the retry counter."""
    logger.info(">>> UTILITY NODE: Advancing to Next Step")
    current_idx = state.get("current_step_index", -1)
    return {"current_step_index": current_idx + 1, "retry_count_for_current_step": 0}

# --- Graph Definition Function ---
def create_research_agent_graph():
    """Builds and compiles the LangGraph StateGraph."""
    logger.info("Building Research Agent Graph...")
    workflow = StateGraph(ResearchAgentState)

    # Add Nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_qa", direct_qa_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("controller", controller_node)
    workflow.add_node("executor", executor_node) # MODIFIED: Using new executor_node
    workflow.add_node("step_evaluator", placeholder_step_evaluator_node)
    workflow.add_node("overall_evaluator", overall_evaluator_node)
    workflow.add_node("increment_retry_count", increment_retry_count_node)
    workflow.add_node("advance_to_next_step", advance_to_next_step_node)

    # --- Define Edges ---
    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        should_proceed_to_plan_or_qa,
        {"planner": "planner", "direct_qa": "direct_qa"}
    )
    
    workflow.add_conditional_edges(
        "planner",
        did_planning_succeed,
        {"success": "controller", "failure": "overall_evaluator"}
    )

    workflow.add_edge("direct_qa", "overall_evaluator") 

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
    # This block allows for testing the graph definition directly.
    logging.basicConfig(level=logging.INFO)
    logger.info("Running langgraph_agent.py directly for structural check...")
    
    print("\n--- Graph Definition (ASCII) ---")
    try:
        # The get_graph().print_ascii() method is useful for a quick console view
        research_agent_graph.get_graph().print_ascii()
    except Exception as e:
        print(f"Could not print ASCII graph: {e}")

    print("\n--- State Schema (JSON) ---")
    try:
        # Pydantic v1 way to get schema
        if hasattr(ResearchAgentState, 'schema_json'):
            print(ResearchAgentState.schema_json(indent=2))
        # If using Pydantic v2 models directly in TypedDict, this might change
        # For now, this is the expected way based on langchain_core's Pydantic usage.
    except Exception as e:
        print(f"Could not print state schema: {e}")
    
    print("\nScript finished.")
