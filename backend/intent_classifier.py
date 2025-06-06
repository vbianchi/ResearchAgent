import logging
from typing import Dict, Any, Optional, Union

from backend.config import settings
from backend.llm_setup import get_llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)

class IntentClassificationOutput(BaseModel):
    """
    Defines the structured output for the Intent Classifier LLM.
    """
    intent: str = Field(description="The classified intent. Must be one of ['PLAN', 'DIRECT_QA', 'DIRECT_TOOL_REQUEST'].")
    reasoning: Optional[str] = Field(description="Brief reasoning for the classification.", default=None)
    tool_name: Optional[str] = Field(default=None, description="If intent is DIRECT_TOOL_REQUEST, the name of the tool to use.")
    tool_input: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="If intent is DIRECT_TOOL_REQUEST, the input for the tool.")


INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant responsible for classifying user intent.
Your goal is to determine if a user's query requires a multi-step plan, a direct answer, or a direct specific tool request.

Available intents:
-   "PLAN": Use this if the query implies a multi-step process, requires breaking down into sub-tasks, involves creating or manipulating multiple pieces of data, or clearly needs a sequence of tool uses. Examples:
    - "Research the latest treatments for X, summarize them, and write a report."
    - "Find three recent news articles about Y, extract key points from each, and compare them."
-   "DIRECT_QA": Use this if the query is a straightforward question, a request for a simple definition or explanation, a request for brainstorming, a simple calculation, or a conversational remark that doesn't require a complex plan or specific tool. The agent can likely answer this using its internal knowledge. Examples:
    - "What is the capital of France?"
    - "Explain the concept of X in simple terms."
    - "Tell me a fun fact."
-   "DIRECT_TOOL_REQUEST": Use this if the query explicitly asks to use a specific known tool or implies a single, direct action that maps clearly to one of the available tools. You MUST also identify the `tool_name` and formulate the `tool_input`.
    - The `tool_name` MUST be one of these: {tool_names_for_prompt}
    - The `tool_input` should be the appropriate input string or JSON string for that specific tool, based on its description:
        {tools_summary_for_prompt}
    - Examples:
        - "Use tavily_search_api to find out about X." -> intent: DIRECT_TOOL_REQUEST, tool_name: tavily_search_api, tool_input: {{"query": "X"}}
        - "What's the weather like today?" (Assume 'weather_tool' is available) -> intent: DIRECT_TOOL_REQUEST, tool_name: weather_tool, tool_input: {{"location": "current"}}
        - "Read the content of the file 'report.txt'." -> intent: DIRECT_TOOL_REQUEST, tool_name: read_file, tool_input: "report.txt"

Consider the complexity and the likely number of distinct operations or tool uses implied by the query.
If the query asks to use a specific tool, it is almost always a "DIRECT_TOOL_REQUEST".

Respond with a single JSON object matching the following schema:
{format_instructions}

Do not include any preamble or explanation outside of the JSON object.
"""


async def classify_intent(
    user_query: str,
    session_data_entry: Dict[str, Any],
    tool_names_for_prompt: str, # <<< ENSURE THIS PARAMETER EXISTS
    tools_summary_for_prompt: str # <<< ENSURE THIS PARAMETER EXISTS
) -> Union[IntentClassificationOutput, str]:
    """
    Classifies the user's intent.
    Returns an IntentClassificationOutput Pydantic model instance on success,
    or a string "PLAN" (defaulting intent) if classification fails critically.
    """
    logger.info(f"IntentClassifier: Classifying intent for query: {user_query[:100]}...")

    intent_llm_id_override = session_data_entry.get("session_intent_classifier_llm_id")
    provider = settings.intent_classifier_provider
    model_name = settings.intent_classifier_model_name

    if intent_llm_id_override:
        try:
            override_provider, override_model_name = intent_llm_id_override.split("::", 1)
            if override_provider and override_model_name:
                provider = override_provider
                model_name = override_model_name
                logger.info(f"IntentClassifier: Using session override LLM: {provider}::{model_name}")
            else:
                logger.warning(f"IntentClassifier: Invalid session LLM ID format '{intent_llm_id_override}'. Using system default for Intent Classifier.")
        except ValueError:
            logger.warning(f"IntentClassifier: Could not parse session LLM ID '{intent_llm_id_override}'. Using system default for Intent Classifier.")


    try:
        intent_llm: BaseChatModel = get_llm(
            settings,
            provider=provider,
            model_name=model_name,
            requested_for_role="INTENT_CLASSIFIER"
        )
        logger.info(f"IntentClassifier: Using LLM {provider}::{model_name}")
    except Exception as e:
        logger.error(f"IntentClassifier: Failed to initialize LLM for intent classification: {e}", exc_info=True)
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent (string) due to LLM initialization error.")
        return "PLAN"

    parser = JsonOutputParser(pydantic_object=IntentClassificationOutput)
    format_instructions = parser.get_format_instructions()

    system_prompt_filled = INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE.format(
        tool_names_for_prompt=tool_names_for_prompt,
        tools_summary_for_prompt=tools_summary_for_prompt,
        format_instructions=format_instructions
    )
    human_template = "User Query: \"{user_query}\"\nClassify the intent of the user query."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_filled),
        ("human", human_template)
    ])
    chain = prompt | intent_llm | parser

    try:
        invoke_params = {"user_query": user_query}

        classification_result_data = await chain.ainvoke(invoke_params)

        if isinstance(classification_result_data, IntentClassificationOutput):
            logger.info(f"IntentClassifier: Successfully classified. Intent: '{classification_result_data.intent}', Tool: '{classification_result_data.tool_name}', Reasoning: {classification_result_data.reasoning}")
            return classification_result_data
        elif isinstance(classification_result_data, dict):
            try:
                model_instance = IntentClassificationOutput(**classification_result_data)
                logger.info(f"IntentClassifier: Classified (from dict). Intent: '{model_instance.intent}', Tool: '{model_instance.tool_name}', Reasoning: {model_instance.reasoning}")
                return model_instance
            except Exception as pydantic_e:
                logger.error(f"IntentClassifier: Failed to parse dict into Pydantic model: {pydantic_e}. Dict: {classification_result_data}")
                logger.warning("IntentClassifier: Defaulting to 'PLAN' intent (string) due to Pydantic parsing error from dict.")
                return "PLAN"
        else:
            logger.error(f"IntentClassifier: LLM chain returned unexpected type: {type(classification_result_data)}. Output: {str(classification_result_data)[:200]}")
            logger.warning("IntentClassifier: Defaulting to 'PLAN' intent (string) due to unexpected output type.")
            return "PLAN"

    except Exception as e:
        logger.error(f"IntentClassifier: Error during intent classification: {e}", exc_info=True)
        try:
            error_chain = prompt | intent_llm | StrOutputParser()
            raw_output_params = {"user_query": user_query} # format_instructions is already in system_prompt_filled
            raw_output = await error_chain.ainvoke(raw_output_params)
            logger.error(f"IntentClassifier: Raw LLM output on error: {raw_output[:500]}...")
        except Exception as raw_e:
            logger.error(f"IntentClassifier: Failed to get raw LLM output during error: {raw_e}")
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent (string) due to classification error.")
        return "PLAN"