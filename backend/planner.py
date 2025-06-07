# backend/planner.py
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from backend.pydantic_models import AgentPlan
from backend.prompts import PLANNER_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

async def generate_plan(
    user_query: str,
    available_tools_summary: str,
    llm_instance: BaseChatModel,
    config: Optional[RunnableConfig] = None
) -> Optional[AgentPlan]:
    """
    Generates a multi-step plan based on the user query using an LLM.
    """
    logger.info(f"Planner: Generating plan for user query: {user_query[:100]}...")

    parser = JsonOutputParser(pydantic_object=AgentPlan)
    format_instructions = parser.get_format_instructions()

    # --- START OF FIX ---
    # Pre-format the system prompt to avoid the KeyError.
    # The JSON schema within format_instructions contains curly braces that
    # were being misinterpreted as template variables. This bakes them into the string.
    system_prompt_text = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        available_tools_summary=available_tools_summary,
        format_instructions=format_instructions,
        user_query=user_query  # The synthesis instruction within the prompt needs the query.
    )
    # --- END OF FIX ---

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "User Query: {user_query}") # The human message still uses a variable.
    ])

    chain = prompt | llm_instance | parser

    try:
        # The invoke payload now only needs to satisfy the variables in the human message.
        invoke_payload = {"user_query": user_query}
        
        planned_result = await chain.ainvoke(invoke_payload, config=config)
        
        # The parser should return an AgentPlan object directly, but we handle the dict case just in case.
        if isinstance(planned_result, dict):
            agent_plan = AgentPlan(**planned_result)
        else:
            agent_plan = planned_result

        logger.info(f"Planner: Plan generated successfully. Summary: {agent_plan.human_readable_summary}")
        return agent_plan

    except Exception as e:
        logger.error(f"Planner: Error during plan generation: {e}", exc_info=True)
        return None
