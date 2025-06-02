# backend/controller.py
import logging
from typing import List, Dict, Any, Tuple, Optional, Union # Added Union
import json
import re
import traceback

from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field, ValidationError # Pydantic v2
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager # Added BaseCallbackManager

from backend.config import settings
from backend.llm_setup import get_llm
# PlanStep might not be strictly needed if we pass current_plan_step as dict, but good for reference
from backend.planner import PlanStep 
from backend.callbacks import LOG_SOURCE_CONTROLLER

logger = logging.getLogger(__name__)

class ControllerOutput(BaseModel): 
    tool_name: Optional[str] = Field(default=None, description="The exact name of the tool to use, or 'None' if no tool is directly needed for this step. Must be one of the available tools.")
    tool_input: Optional[str] = Field(default=None, description="The precise, complete input string for the chosen tool, or a concise summary/directive for the LLM if tool_name is 'None'. Can be explicitly 'None' or null if the step description and expected outcome are self-sufficient for a 'None' tool LLM action.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating the controller's confidence in this tool/input choice for the step. 1.0 is high confidence.")
    reasoning: str = Field(description="Brief explanation of why this tool/input was chosen or why no tool is needed.")

CONTROLLER_SYSTEM_PROMPT_TEMPLATE = """You are an expert "Controller" for a research agent.
Your role is to analyze a single step from a pre-defined plan and decide the BEST action for the "Executor" (a ReAct agent) to take for that step.
**Current Task Context:**
- Original User Query: {original_user_query}
- Current Plan Step Description: {current_step_description}
- Expected Outcome for this Step: {current_step_expected_outcome}
- Tool suggested by Planner (can be overridden): {planner_tool_suggestion}
- Planner's input instructions (guidance, not literal input): {planner_tool_input_instructions}

**Available Tools for the Executor:**
{available_tools_summary}

**Output from the PREVIOUS successful plan step (if available and relevant for the current step):**
{previous_step_output_context}

**Your Task:**
Based on ALL the above information, determine the most appropriate `tool_name` and formulate the precise `tool_input`.
**Key Considerations:**
1.  **Tool Selection:**
    * If the Planner's `tool_suggestion` is appropriate and aligns with the step description and available tools, prioritize it.
    * If the Planner's suggestion is 'None' or unsuitable, you MUST select an appropriate tool from the `Available Tools` list if one is clearly needed to achieve the `expected_outcome`.
    * If the step is purely analytical, requires summarization of previous context/memory, or involves creative generation that the LLM can do directly (and no tool is a better fit), set `tool_name` to "None".
2.  **Tool Input Formulation:**
    * If a tool is chosen, `tool_input` MUST be the exact, complete, and correctly formatted string the tool expects.
    * **CRUCIAL: If a tool's description (from `Available Tools` above) explicitly states its input MUST be a JSON string (e.g., it mentions "Input MUST be a JSON string matching...schema..."), then your `tool_input` field MUST be that exact, complete, and valid JSON string. Do not provide just the raw content for one of its keys; provide the full JSON structure as a string (e.g., "{{\\"query\\": \\"actual research query\\", \\"num_sources_to_deep_dive\\": 3}}").**
    * **CRUCIAL: If `previous_step_output_context` is provided AND the `current_step_description` or `planner_tool_input_instructions` clearly indicate that the current step should use the output of the previous step (e.g., "write the generated poem", "summarize the search results", "use the extracted data"), you MUST use the content from `previous_step_output_context` to form the `tool_input` (e.g., for `write_file`, the content part of the input) or as the direct basis for a "None" tool generation.
    * Do NOT re-generate information or create new example content if it's already present in `previous_step_output_context` and is meant to be used by the current step.
    * If `tool_name` is "None", `tool_input` should be a concise summary of what the Executor LLM should generate or reason about to achieve the `expected_outcome`.
    * It can be explicitly `null` or a string "None" if the description and expected outcome are self-sufficient for the LLM, especially if `previous_step_output_context` contains the necessary data for the LLM to work on directly.
3.  **Confidence Score:** Provide a `confidence_score` (0.0 to 1.0) for your decision.
4.  **Reasoning:** Briefly explain your choices, including how you used (or why you didn't use) the `previous_step_output_context`.
Output ONLY a JSON object adhering to this schema:
{format_instructions}

Do not include any preamble or explanation outside the JSON object.
If you determine an error or impossibility in achieving the step as described, set tool_name to "None", tool_input to a description of the problem, confidence_score to 0.0, and explain in reasoning.
"""

def escape_template_curly_braces(text: Optional[str]) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return text.replace("{", "{{").replace("}", "}}")

async def validate_and_prepare_step_action(
    original_user_query: str,
    current_plan_step: Dict[str, Any], # Expecting a dict representation of PlanStep
    available_tools: List[BaseTool],
    previous_step_executor_output: Optional[str] = None,
    controller_llm_id_override: Optional[str] = None, # For Controller's own LLM
    callback_handler: Union[List[BaseCallbackHandler], BaseCallbackManager, None] = None
) -> Dict[str, Any]: # Returns a dict to update ResearchAgentState
    """
    Uses an LLM to validate the current plan step and determine the precise tool and input.
    """
    step_desc_for_log = current_plan_step.get('description', 'N/A')[:100]
    planner_tool_sugg_for_log = current_plan_step.get('tool_to_use', 'None')
    logger.info(f"Controller: Validating plan step: '{step_desc_for_log}...' (Planner suggestion: {planner_tool_sugg_for_log})")
    if previous_step_executor_output:
        logger.info(f"Controller: Received previous_step_executor_output (first 100 chars): {previous_step_executor_output[:100]}...")

    llm_provider_to_use = settings.controller_provider
    llm_model_to_use = settings.controller_model_name

    if controller_llm_id_override:
        try:
            provider_override, model_override = controller_llm_id_override.split("::", 1)
            if provider_override in ["gemini", "ollama"] and model_override:
                llm_provider_to_use, llm_model_to_use = provider_override, model_override
                logger.info(f"Controller: Using session override LLM for Controller: {controller_llm_id_override}")
            else:
                logger.warning(f"Controller: Invalid structure or unknown provider in session LLM ID override '{controller_llm_id_override}'. Using system default for Controller.")
        except ValueError:
            logger.warning(f"Controller: Invalid session LLM ID override format '{controller_llm_id_override}'. Using system default for Controller.")

    try:
        controller_llm: BaseChatModel = get_llm(
            settings,
            provider=llm_provider_to_use,
            model_name=llm_model_to_use,
            requested_for_role=LOG_SOURCE_CONTROLLER,
            callbacks=callback_handler # Pass the manager or list of handlers
        )
        logger.info(f"Controller: Using LLM {llm_provider_to_use}::{llm_model_to_use}")
    except Exception as e:
        logger.error(f"Controller: Failed to initialize LLM: {e}", exc_info=True)
        return {
            "controller_tool_name": None,
            "controller_tool_input": None,
            "controller_reasoning": f"LLM initialization failed: {e}",
            "controller_confidence": 0.0,
            "controller_error": f"LLM initialization failed: {e}"
        }

    parser = JsonOutputParser(pydantic_object=ControllerOutput)
    format_instructions = parser.get_format_instructions()

    tools_summary_list = [f"- {tool.name}: {tool.description}" for tool in available_tools]
    tools_summary_for_controller = "\n".join(tools_summary_list)

    previous_step_output_context_str = "Not applicable (this is the first step or previous step had no direct output, or its output was not relevant to pass)."
    if previous_step_executor_output is not None:
        previous_step_output_context_str = f"The direct output from the PREVIOUS successfully completed step was:\n---\n{previous_step_executor_output}\n---"

    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTROLLER_SYSTEM_PROMPT_TEMPLATE),
        ("human", "Analyze the current plan step and provide your output in the specified JSON format.")
    ])
    
    # Prepare inputs for the prompt from current_plan_step dictionary
    step_description = current_plan_step.get('description', 'No description provided.')
    step_expected_outcome = current_plan_step.get('expected_outcome', 'No expected outcome specified.')
    planner_suggestion = current_plan_step.get('tool_to_use', 'None')
    planner_instructions = current_plan_step.get('tool_input_instructions', 'None')

    invoke_payload = {
        "original_user_query": escape_template_curly_braces(original_user_query),
        "current_step_description": escape_template_curly_braces(step_description),
        "current_step_expected_outcome": escape_template_curly_braces(step_expected_outcome),
        "planner_tool_suggestion": escape_template_curly_braces(planner_suggestion),
        "planner_tool_input_instructions": escape_template_curly_braces(planner_instructions),
        "available_tools_summary": escape_template_curly_braces(tools_summary_for_controller),
        "previous_step_output_context": escape_template_curly_braces(previous_step_output_context_str),
        "format_instructions": format_instructions
    }
    run_config = RunnableConfig(
        callbacks=callback_handler if isinstance(callback_handler, BaseCallbackManager) else ([callback_handler] if callback_handler else None),
        metadata={"component_name": LOG_SOURCE_CONTROLLER}
    )
    
    chain = prompt | controller_llm | parser
    raw_llm_output_str = "" # For error logging

    try:
        # The parser should return the Pydantic ControllerOutput object
        controller_output_model: ControllerOutput = await chain.ainvoke(invoke_payload, config=run_config)
        
        # Handle if parser returns a dict instead of model instance (as seen with other parsers)
        if isinstance(controller_output_model, dict):
            logger.debug("Controller: LLM output parser returned a dict, validating with Pydantic model ControllerOutput.")
            try:
                controller_output_model = ControllerOutput(**controller_output_model)
            except ValidationError as ve_controller:
                logger.error(f"Controller: Pydantic validation failed for ControllerOutput from LLM dict: {ve_controller}. Raw dict: {controller_output_model}", exc_info=True)
                raise

        tool_name_final = controller_output_model.tool_name if controller_output_model.tool_name and controller_output_model.tool_name.lower() != "none" else None
        
        logger.info(f"Controller LLM decided: Tool='{tool_name_final}', Input (summary)='{str(controller_output_model.tool_input)[:100]}...', Confidence={controller_output_model.confidence_score:.2f}")
        return {
            "controller_tool_name": tool_name_final,
            "controller_tool_input": controller_output_model.tool_input,
            "controller_reasoning": controller_output_model.reasoning,
            "controller_confidence": controller_output_model.confidence_score,
            "controller_error": None
        }

    except ValidationError as ve: # If Pydantic validation fails directly from parser
        logger.error(f"Controller: Pydantic validation error for ControllerOutput: {ve}", exc_info=True)
        error_message = f"Controller output validation failed: {ve}"
    except Exception as e: 
        logger.error(f"Controller: Error during step validation or LLM call: {e}", exc_info=True)
        error_message = f"Error in Controller: {type(e).__name__} - {str(e)}"

    # Attempt to get raw output on error
    try:
        raw_output_chain = prompt | controller_llm | StrOutputParser()
        raw_llm_output_str = await raw_output_chain.ainvoke(invoke_payload, config=run_config) # Use same config
        logger.error(f"Controller: Raw LLM output on error: {raw_llm_output_str}")
        error_message += f". Raw LLM output: {raw_llm_output_str[:200]}..."
    except Exception as raw_e:
        logger.error(f"Controller: Failed to get raw LLM output on error: {raw_e}")
        error_message += ". Failed to retrieve raw LLM output."
        
    return {
        "controller_tool_name": None,
        "controller_tool_input": None,
        "controller_reasoning": "Controller processing failed.",
        "controller_confidence": 0.0,
        "controller_error": error_message
    }

# __main__ block for testing (would need significant mocking for standalone run)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     # This test would require mocking available_tools, settings, get_llm, etc.
#     # For now, testing will be primarily through langgraph_agent.py
#     async def test_controller():
#         print("Controller standalone test would require extensive mocking.")
#         # Example mock data:
#         mock_original_query = "Analyze data.csv and generate a report."
#         mock_current_step = {
#             "step_id": 1,
#             "description": "Read the file 'data.csv'.",
#             "tool_to_use": "read_file",
#             "tool_input_instructions": "The file path is 'data.csv'.",
#             "expected_outcome": "The content of 'data.csv' is available."
#         }
#         # Mock BaseTool for available_tools
#         class MockTool(BaseTool):
#             name: str
#             description: str
#             def _run(self, *args, **kwargs): raise NotImplementedError
#             async def _arun(self, *args, **kwargs): raise NotImplementedError
        
#         mock_available_tools = [
#             MockTool(name="read_file", description="Reads a file from the workspace."),
#             MockTool(name="python_repl", description="Executes Python code.")
#         ]
#         mock_prev_output = None
#         # mock_controller_llm_override = "gemini::gemini-1.5-flash" # Example
        
#         # Need to mock get_llm or ensure settings are loaded for a real call.
#         # For a simple structural test:
#         # result = await validate_and_prepare_step_action(
#         #     mock_original_query,
#         #     mock_current_step,
#         #     mock_available_tools,
#         #     mock_prev_output,
#         #     None, # No override for this simple test structure
#         #     None  # No callback handler for this simple test structure
#         # )
#         # print(f"Controller Test Result: {result}")

#     # asyncio.run(test_controller())
