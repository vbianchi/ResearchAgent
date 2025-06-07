# backend/intent_classifier.py
import logging
from typing import Dict, Any, Optional, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableConfig

from backend.config import settings
from backend.llm_setup import get_llm
from backend.pydantic_models import IntentClassificationOutput
from backend.prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

async def classify_intent(
    user_query: str,
    tool_names_for_prompt: str,
    session_data_entry: Dict[str, Any],
    config: RunnableConfig
) -> IntentClassificationOutput:
    """
    Classifies the user's intent using a pre-formatted system prompt to avoid template errors.
    """
    logger.info(f"IntentClassifier: Classifying intent for query: {user_query[:100]}...")
    callbacks_for_invoke = config.get("callbacks", [])
    
    intent_llm_id_override = session_data_entry.get("session_intent_classifier_llm_id")
    provider, model_name = (intent_llm_id_override.split("::", 1) if "::" in (intent_llm_id_override or "")
                           else (settings.intent_classifier_provider, settings.intent_classifier_model_name))
    
    logger.info(f"IntentClassifier: Using LLM {provider}::{model_name}")
    
    try:
        intent_llm = get_llm(settings, provider, model_name, callbacks_for_invoke, "INTENT_CLASSIFIER")
    except Exception as e:
        logger.error(f"IntentClassifier: LLM initialization failed: {e}", exc_info=True)
        # Return a default "PLAN" response on failure to prevent crashing
        return IntentClassificationOutput(intent="PLAN", reasoning="Defaulted to PLAN due to LLM initialization error.")

    parser = JsonOutputParser(pydantic_object=IntentClassificationOutput)
    format_instructions = parser.get_format_instructions()

    # --- START OF THE FIX ---
    # Pre-format the system prompt with the format_instructions.
    # This prevents the main ChatPromptTemplate from misinterpreting the curly braces
    # within the JSON schema as template variables.
    system_prompt_text = INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE.format(format_instructions=format_instructions)
    # --- END OF THE FIX ---

    # The human template now only needs the user_query
    human_template = "User Query: \"{user_query}\"\nClassify the intent of this query."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text), # Pass the pre-formatted text directly
        ("human", human_template)
    ])

    chain = prompt | intent_llm | parser

    try:
        # We only need to provide the variable for the human template now
        invoke_params = {"user_query": user_query}
        
        classification_result = await chain.ainvoke(invoke_params, config)
        
        if isinstance(classification_result, dict):
            classification_result = IntentClassificationOutput(**classification_result)

        logger.info(f"IntentClassifier: Successfully classified intent as '{classification_result.intent}'.")
        return classification_result

    except Exception as e:
        logger.error(f"IntentClassifier: Error during intent classification: {e}", exc_info=True)
        # Attempt to get raw output for debugging
        try:
            raw_output_chain = prompt | intent_llm | StrOutputParser()
            raw_output = await raw_output_chain.ainvoke({"user_query": user_query}, config)
            logger.error(f"IntentClassifier: Raw LLM output on error: {raw_output}")
        except Exception as raw_e:
            logger.error(f"IntentClassifier: Failed to get raw LLM output on error: {raw_e}")
        
        # Return a default "PLAN" response on failure
        return IntentClassificationOutput(intent="PLAN", reasoning=f"Defaulted to PLAN due to classification error: {e}")
