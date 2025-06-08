# backend/intent_classifier.py
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from backend.config import settings
from backend.llm_setup import get_llm
from backend.prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

async def classify_intent(user_query: str) -> Dict[str, str]:
    """
    Classifies the user's intent to decide the most efficient path.
    """
    logger.info(f"IntentClassifier: Classifying query: '{user_query[:100]}...'")
    
    # For intent classification, we can use a fast model.
    # We will use the default model for now, but this could be a specific, faster model in the future.
    provider, model_name = settings.default_provider, settings.default_model_name
    
    try:
        intent_llm = get_llm(settings, provider, model_name, callbacks=None, requested_for_role="IntentClassifier")
    except Exception as e:
        logger.error(f"IntentClassifier: LLM initialization failed: {e}", exc_info=True)
        # On failure, default to the full agent to be safe.
        return {"intent": "AGENT_ACTION"}

    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFIER_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    
    chain = prompt | intent_llm | parser

    try:
        classification_result = await chain.ainvoke({"query": user_query})
        
        intent = classification_result.get("intent")
        if intent not in ["AGENT_ACTION", "DIRECT_QA"]:
             logger.warning(f"IntentClassifier returned invalid intent: '{intent}'. Defaulting to AGENT_ACTION.")
             return {"intent": "AGENT_ACTION"}

        logger.info(f"IntentClassifier: Successfully classified intent as '{intent}'.")
        return classification_result

    except Exception as e:
        logger.error(f"IntentClassifier: Error during classification: {e}", exc_info=True)
        # Default to the full agent on any error.
        return {"intent": "AGENT_ACTION"}
