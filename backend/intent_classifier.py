# backend/intent_classifier.py
import logging
from typing import Dict, Any, Optional, List, Union # MODIFIED: Added Union

# Using Pydantic v2 directly
from pydantic import BaseModel, Field, ValidationError

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager # MODIFIED: Added BaseCallbackManager
import asyncio 

from backend.config import settings 
from backend.llm_setup import get_llm 
from backend.callbacks import LOG_SOURCE_INTENT_CLASSIFIER 

logger = logging.getLogger(__name__)

class IntentClassificationOutput(BaseModel):
    intent: str = Field(description="The classified intent. Must be one of ['PLAN', 'DIRECT_QA'].")
    reasoning: Optional[str] = Field(default=None, description="Brief reasoning for the classification.")

INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant responsible for classifying user intent.
Your goal is to determine if a user's query requires a multi-step plan involving tools and complex reasoning, or if it's a simple question/statement that can be answered directly or via a single tool use (like a quick web search).
Available intents:
-   "PLAN": Use this if the query implies a multi-step process, requires breaking down into sub-tasks, involves creating or manipulating multiple pieces of data, or clearly needs a sequence of tool uses.
    Examples:
    - "Research the latest treatments for X, summarize them, and write a report."
    - "Find three recent news articles about Y, extract key points from each, and compare them."
    - "Download the data from Z, process it, and generate a plot."
-   "DIRECT_QA": Use this if the query is a straightforward question, a request for a simple definition or explanation, a request for brainstorming, a simple calculation, or a conversational remark that doesn't require a complex plan.
    The agent can likely answer this using its internal knowledge or a single quick tool use (like a web search for a current fact).
    Examples:
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
    available_tools_summary: Optional[str] = None,
    # MODIFIED: callback_handler can now be a manager instance or list of handlers, or None
    callback_handler: Union[List[BaseCallbackHandler], BaseCallbackManager, None] = None
) -> IntentClassificationOutput:
    logger.info(f"IntentClassifier: Classifying intent for query: {user_query[:100]}...")

    # The callback_handler received here is what's passed directly to get_llm
    if callback_handler:
        logger.debug(f"IntentClassifier: Received callback_handler of type: {type(callback_handler)}")
    else:
        logger.debug("IntentClassifier: No callback_handler provided.")

    default_intent_output = IntentClassificationOutput(intent="PLAN", reasoning="Default due to error or empty/invalid LLM response.")

    try:
        logger.debug(f"IntentClassifier: Getting LLM with provider '{settings.intent_classifier_provider}' and model '{settings.intent_classifier_model_name}'.")
        intent_llm: BaseChatModel = get_llm(
            settings,
            provider=settings.intent_classifier_provider,
            model_name=settings.intent_classifier_model_name,
            requested_for_role=LOG_SOURCE_INTENT_CLASSIFIER,
            callbacks=callback_handler # MODIFIED: Pass callback_handler directly
        )
        logger.info(f"IntentClassifier: Using LLM {settings.intent_classifier_provider}::{settings.intent_classifier_model_name}")
    except Exception as e:
        logger.error(f"IntentClassifier: Failed to initialize LLM for intent classification: {e}", exc_info=True)
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent due to LLM initialization error.")
        return default_intent_output

    parser = JsonOutputParser(pydantic_object=IntentClassificationOutput)
    format_instructions = parser.get_format_instructions()

    human_template_parts = [f"User Query: \"{user_query}\"\n"]
    if available_tools_summary:
        human_template_parts.append(f"\nFor context, the agent has access to tools like: {available_tools_summary}\n")
    human_template_parts.append("\nClassify the intent of the user query.")
    human_template = "".join(human_template_parts)

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

        # The config for ainvoke should contain the callback manager directly if it's to be used by the runnable.
        # The `callback_handler` variable here *is* the manager instance from LangGraph's config.
        run_config = RunnableConfig(
            callbacks=callback_handler if isinstance(callback_handler, BaseCallbackManager) else ([callback_handler] if callback_handler else None), # Ensure it's a list if a single handler, or the manager itself
            metadata={"component_name": LOG_SOURCE_INTENT_CLASSIFIER}
        )
        # If callback_handler is already a manager, LangChain might handle it.
        # If it's a list of handlers, LangChain will create a manager.
        # The key is that ChatGoogleGenerativeAI needs to receive it in an acceptable format.
        # The `get_llm` was modified to accept the manager or list directly.

        parsed_llm_output = await chain.ainvoke(invoke_params, config=run_config)
        
        classified_output_model: IntentClassificationOutput
        if isinstance(parsed_llm_output, IntentClassificationOutput):
            classified_output_model = parsed_llm_output
        elif isinstance(parsed_llm_output, dict):
            logger.debug("IntentClassifier: LLM output parser returned a dict, attempting to validate with Pydantic model.")
            try:
                classified_output_model = IntentClassificationOutput(**parsed_llm_output)
            except ValidationError as ve:
                logger.error(f"IntentClassifier: Pydantic validation failed for LLM output dict: {ve}. Raw dict: {parsed_llm_output}", exc_info=True)
                raise 
        else:
            logger.error(f"IntentClassifier: Unexpected output type from LLM parser chain: {type(parsed_llm_output)}. Output: {str(parsed_llm_output)[:500]}")
            raise TypeError(f"Unexpected output type from LLM parser: {type(parsed_llm_output)}")

        if classified_output_model.intent.upper() not in ["PLAN", "DIRECT_QA"]:
            original_intent_from_llm = classified_output_model.intent
            logger.warning(f"IntentClassifier: LLM returned an unknown intent '{original_intent_from_llm}'. Defaulting to 'PLAN'.")
            classified_output_model.intent = "PLAN"
            current_reasoning = classified_output_model.reasoning or ""
            classified_output_model.reasoning = f"(LLM returned unknown intent '{original_intent_from_llm}', defaulted to PLAN. Original reasoning: {current_reasoning})".strip()
        
        logger.info(f"IntentClassifier: Classified intent as '{classified_output_model.intent}'. Reasoning: {classified_output_model.reasoning or 'No reasoning provided.'}")
        return classified_output_model

    except Exception as e:
        logger.error(f"IntentClassifier: Error during intent classification chain execution: {e}", exc_info=True)
        try:
            raw_output_chain = prompt | intent_llm | StrOutputParser()
            raw_output_params = {
                "user_query": user_query,
                "format_instructions": format_instructions
            }
            if available_tools_summary:
                raw_output_params["available_tools_summary"] = available_tools_summary
            
            # Construct config for error handler call as well
            error_run_config = RunnableConfig(
                 callbacks=callback_handler if isinstance(callback_handler, BaseCallbackManager) else ([callback_handler] if callback_handler else None),
                 metadata={"component_name": LOG_SOURCE_INTENT_CLASSIFIER + "_ERROR_HANDLER"}
            )
            raw_output = await raw_output_chain.ainvoke(raw_output_params, config=error_run_config)
            logger.error(f"IntentClassifier: Raw LLM output on error: {raw_output}")
        except Exception as raw_e:
            logger.error(f"IntentClassifier: Failed to get raw LLM output during error handling: {raw_e}")
        
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent due to classification error.")
        return default_intent_output

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    
    async def test_intent_classifier_main():
        queries = [
            "What is photosynthesis?", 
            "Research the impact of AI on climate change and write a report.",
        ]
        tools_summary_example = "- tavily_search_api: For web searches."
        
        print("\n--- Testing Intent Classifier ---")
        for q_idx, q_text in enumerate(queries):
            print(f"\nTest Query {q_idx + 1}: \"{q_text}\"")
            result_obj = await classify_intent(q_text, tools_summary_example, callback_handler=None)
            print(f"  -> Classified Intent: {result_obj.intent}")
            print(f"  -> Reasoning: {result_obj.reasoning or 'N/A'}")
        print("-----------------------------\n")
            
    asyncio.run(test_intent_classifier_main())
