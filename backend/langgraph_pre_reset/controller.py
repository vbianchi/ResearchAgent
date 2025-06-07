# backend/controller.py
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from backend.config import settings
from backend.llm_setup import get_llm
# --- MODIFIED: Corrected Pydantic model imports ---
from backend.pydantic_models import PlanStep, ControllerOutput
from backend.prompts import CONTROLLER_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

async def validate_and_prepare_step_action(
    original_user_query: str,
    plan_step: PlanStep,
    available_tools: List[BaseTool],
    session_data_entry: Dict[str, Any],
    previous_step_output: Optional[str] = None,
    config: Optional[RunnableConfig] = None
) -> Optional[ControllerOutput]:
    """
    Uses an LLM to validate the current plan step and determine the precise tool and input.
    """
    logger.info(f"Controller: Validating plan step ID {plan_step.step_id}: '{plan_step.description[:100]}...'")

    controller_llm_id_override = session_data_entry.get("session_controller_llm_id")
    provider, model_name = (controller_llm_id_override.split("::", 1) if "::" in (controller_llm_id_override or "")
                           else (settings.controller_provider, settings.controller_model_name))
    try:
        controller_llm = get_llm(settings, provider, model_name, config.get("callbacks"), "Controller")
    except Exception as e:
        logger.error(f"Controller: Failed to initialize LLM: {e}", exc_info=True)
        return None

    parser = JsonOutputParser(pydantic_object=ControllerOutput)
    format_instructions = parser.get_format_instructions()
    tools_summary = "\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools])
    previous_context = previous_step_output or "Not applicable (this is the first step)."

    system_prompt_text = CONTROLLER_SYSTEM_PROMPT_TEMPLATE.format(
        original_user_query=original_user_query,
        current_step_description=plan_step.description,
        current_step_expected_outcome=plan_step.expected_outcome,
        planner_tool_suggestion=plan_step.tool_to_use or "None",
        planner_tool_input_instructions=plan_step.tool_input_instructions or "None",
        available_tools_summary=tools_summary,
        previous_step_output_context=previous_context,
        format_instructions=format_instructions
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "Analyze the current plan step and provide your output in the specified JSON format.")
    ])
    chain = prompt | controller_llm | parser

    try:
        controller_result = await chain.ainvoke({}, config=config)
        return ControllerOutput(**controller_result) if isinstance(controller_result, dict) else controller_result
    except Exception as e:
        logger.error(f"Controller: Error during step validation: {e}", exc_info=True)
        return None
