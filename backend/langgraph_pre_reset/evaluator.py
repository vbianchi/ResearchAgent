# backend/evaluator.py
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from backend.config import settings
from backend.llm_setup import get_llm
# --- MODIFIED: Corrected Pydantic model imports ---
from backend.pydantic_models import PlanStep, StepCorrectionOutcome, EvaluationResult
from backend.prompts import STEP_EVALUATOR_SYSTEM_PROMPT_TEMPLATE, OVERALL_EVALUATOR_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

async def evaluate_step_outcome_and_suggest_correction(
    original_user_query: str,
    plan_step_being_evaluated: PlanStep,
    controller_tool_used: Optional[str],
    controller_tool_input: Optional[str],
    step_executor_output: str,
    available_tools: List[BaseTool],
    session_data_entry: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Optional[StepCorrectionOutcome]:
    logger.info(f"Evaluator (Step): Evaluating step '{plan_step_being_evaluated.step_id}'")

    evaluator_llm_id_override = session_data_entry.get("session_evaluator_llm_id")
    provider, model_name = (evaluator_llm_id_override.split("::", 1) if "::" in (evaluator_llm_id_override or "")
                           else (settings.evaluator_provider, settings.evaluator_model_name))
    try:
        evaluator_llm = get_llm(settings, provider, model_name, config.get("callbacks"), "EVALUATOR_STEP")
    except Exception as e:
        logger.error(f"Evaluator (Step): Failed to initialize LLM: {e}", exc_info=True)
        return None

    parser = JsonOutputParser(pydantic_object=StepCorrectionOutcome)
    format_instructions = parser.get_format_instructions()
    tools_summary = "\n".join([f"- {tool.name}: {tool.description.split('.')[0]}" for tool in available_tools])

    system_prompt_text = STEP_EVALUATOR_SYSTEM_PROMPT_TEMPLATE.format(
        original_user_query=original_user_query,
        current_step_description=plan_step_being_evaluated.description,
        current_step_expected_outcome=plan_step_being_evaluated.expected_outcome,
        controller_tool_used=controller_tool_used or "None",
        controller_tool_input=controller_tool_input or "None",
        step_executor_output=step_executor_output,
        available_tools_summary=tools_summary,
        format_instructions=format_instructions
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "Evaluate the step execution and provide your assessment in the specified JSON format.")
    ])
    chain = prompt | evaluator_llm | parser

    try:
        eval_result = await chain.ainvoke({}, config=config)
        return StepCorrectionOutcome(**eval_result) if isinstance(eval_result, dict) else eval_result
    except Exception as e:
        logger.error(f"Evaluator (Step): Error during evaluation: {e}", exc_info=True)
        return None


async def evaluate_plan_outcome(
    original_user_query: str,
    executed_plan_summary: str,
    final_agent_answer: str,
    session_data_entry: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Optional[EvaluationResult]:
    logger.info(f"Evaluator (Overall): Evaluating outcome for query: {original_user_query[:100]}...")
    
    evaluator_llm_id_override = session_data_entry.get("session_evaluator_llm_id")
    provider, model_name = (evaluator_llm_id_override.split("::", 1) if "::" in (evaluator_llm_id_override or "")
                           else (settings.evaluator_provider, settings.evaluator_model_name))
    try:
        evaluator_llm = get_llm(settings, provider, model_name, config.get("callbacks"), "EVALUATOR_OVERALL")
    except Exception as e:
        logger.error(f"Evaluator (Overall): Failed to initialize LLM: {e}", exc_info=True)
        return None

    parser = JsonOutputParser(pydantic_object=EvaluationResult)
    format_instructions = parser.get_format_instructions()

    system_prompt_text = OVERALL_EVALUATOR_SYSTEM_PROMPT_TEMPLATE.format(
        original_user_query=original_user_query,
        executed_plan_summary=executed_plan_summary,
        final_agent_answer=final_agent_answer,
        format_instructions=format_instructions
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "Evaluate the overall plan execution and provide your assessment in the specified JSON format.")
    ])
    chain = prompt | evaluator_llm | parser

    try:
        eval_result = await chain.ainvoke({}, config=config)
        return EvaluationResult(**eval_result) if isinstance(eval_result, dict) else eval_result
    except Exception as e:
        logger.error(f"Evaluator (Overall): Error during evaluation: {e}", exc_info=True)
        return None
