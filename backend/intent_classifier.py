import logging
from typing import Dict, Any, Optional

from backend.config import settings
from backend.llm_setup import get_llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field # Using v1 as per project context

logger = logging.getLogger(__name__)

class IntentClassificationOutput(BaseModel):
    """
    Defines the structured output for the Intent Classifier LLM.
    """
    intent: str = Field(description="The classified intent. Must be one of ['PLAN', 'DIRECT_QA'].")
    reasoning: Optional[str] = Field(description="Brief reasoning for the classification.", default=None)

INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant responsible for classifying user intent.
Your goal is to determine if a user's query requires a multi-step plan involving tools and complex reasoning, or if it's a simple question/statement that can be answered directly or via a single tool use (like a quick web search).

Available intents:
-   "PLAN": Use this if the query implies a multi-step process, requires breaking down into sub-tasks, involves creating or manipulating multiple pieces of data, or clearly needs a sequence of tool uses. Examples:
    - "Research the latest treatments for X, summarize them, and write a report."
    - "Find three recent news articles about Y, extract key points from each, and compare them."
    - "Download the data from Z, process it, and generate a plot."
-   "DIRECT_QA": Use this if the query is a straightforward question, a request for a simple definition or explanation, a request for brainstorming, a simple calculation, or a conversational remark that doesn't require a complex plan. The agent can likely answer this using its internal knowledge or a single quick tool use (like a web search for a current fact). Examples:
    - "What is the capital of France?"
    - "Explain the concept of X in simple terms."
    - "Tell me a fun fact."
    - "What's the weather like today?" (implies a single tool use)
    - "Can you help me brainstorm ideas for a project about Y?"
    - "Thanks, that was helpful!"

Consider the complexity and the likely number of distinct operations or tool uses implied by the query.

Respond with a single JSON object matching the following schema:
{format_instructions}

Do not include any preamble or explanation outside of the JSON object.
"""

async def classify_intent(
    user_query: str,
    session_data_entry: Dict[str, Any], # Added to get LLM override
    available_tools_summary: Optional[str] = None
) -> str: # Ensure return type is str
    """
    Classifies the user's intent as either requiring a plan or direct Q&A.
    Fetches its own LLM based on settings and session overrides.

    Args:
        user_query: The user's input query.
        session_data_entry: The session data containing potential LLM overrides.
        available_tools_summary: An optional summary of available tools for context.

    Returns:
        A string representing the classified intent (e.g., "PLAN", "DIRECT_QA").
        Defaults to "PLAN" if classification fails or is uncertain.
    """
    logger.info(f"IntentClassifier: Classifying intent for query: {user_query[:100]}...")

    intent_llm_id_override = session_data_entry.get("session_intent_classifier_llm_id")
    provider = settings.intent_classifier_provider
    model_name = settings.intent_classifier_model_name

    if intent_llm_id_override:
        try:
            override_provider, override_model_name = intent_llm_id_override.split("::", 1)
            if override_provider and override_model_name: # Basic check
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
            requested_for_role="INTENT_CLASSIFIER" # Added role for clarity in logs
        )
        logger.info(f"IntentClassifier: Using LLM {provider}::{model_name}")
    except Exception as e:
        logger.error(f"IntentClassifier: Failed to initialize LLM for intent classification: {e}", exc_info=True)
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent due to LLM initialization error.")
        return "PLAN"

    parser = JsonOutputParser(pydantic_object=IntentClassificationOutput)
    format_instructions = parser.get_format_instructions()

    human_template = "User Query: \"{user_query}\"\n"
    if available_tools_summary:
        human_template += "\nFor context, the agent has access to tools like: {available_tools_summary}\n"
    human_template += "\nClassify the intent of the user query."

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE),
        ("human", human_template)
    ])
    chain = prompt | intent_llm | parser

    try:
        invoke_params = {
            "user_query": user_query,
            "format_instructions": format_instructions
        }
        if available_tools_summary:
            invoke_params["available_tools_summary"] = available_tools_summary

        # The chain with JsonOutputParser should return a dictionary
        # or the Pydantic model instance directly if configured.
        # Let's assume it returns a dictionary that needs to be parsed into the Pydantic model.
        classification_result_data = await chain.ainvoke(invoke_params)

        # Ensure we have the Pydantic model instance
        if isinstance(classification_result_data, IntentClassificationOutput):
            classified_output_model = classification_result_data
        elif isinstance(classification_result_data, dict):
            classified_output_model = IntentClassificationOutput(**classification_result_data)
        else:
            logger.error(f"IntentClassifier: LLM chain returned unexpected type: {type(classification_result_data)}. Output: {str(classification_result_data)[:200]}")
            logger.warning("IntentClassifier: Defaulting to 'PLAN' due to unexpected output type.")
            return "PLAN"

        intent_str = classified_output_model.intent.upper()
        reasoning_str = classified_output_model.reasoning or "No reasoning provided."
        
        # This log is internal to intent_classifier.py
        logger.info(f"IntentClassifier: Internally classified intent as '{intent_str}'. Reasoning: {reasoning_str}")

        if intent_str in ["PLAN", "DIRECT_QA"]:
            return intent_str  # CRITICAL: Return ONLY the intent string
        else:
            logger.warning(f"IntentClassifier: LLM returned an unknown intent value '{intent_str}'. Defaulting to 'PLAN'.")
            return "PLAN"

    except Exception as e:
        logger.error(f"IntentClassifier: Error during intent classification: {e}", exc_info=True)
        # Attempt to get raw output for debugging if parsing failed
        try:
            error_chain = prompt | intent_llm | StrOutputParser()
            raw_output_params = {
                "user_query": user_query,
                "format_instructions": format_instructions # Still include for prompt consistency
            }
            if available_tools_summary:
                raw_output_params["available_tools_summary"] = available_tools_summary
            raw_output = await error_chain.ainvoke(raw_output_params)
            logger.error(f"IntentClassifier: Raw LLM output on error: {raw_output[:500]}...") # Log first 500 chars
        except Exception as raw_e:
            logger.error(f"IntentClassifier: Failed to get raw LLM output during error: {raw_e}")
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent due to classification error.")
        return "PLAN"

