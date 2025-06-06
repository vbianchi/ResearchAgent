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
    intent: str = Field(description="The classified intent. Must be one of ['PLAN', 'DIRECT_TOOL_REQUEST', 'DIRECT_QA'].")
    reasoning: Optional[str] = Field(description="Brief reasoning for the classification.", default=None)
    tool_name: Optional[str] = Field(default=None, description="If intent is DIRECT_TOOL_REQUEST, the name of the tool to use.")
    tool_input: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="If intent is DIRECT_TOOL_REQUEST, the input for the tool.")


# <<< START REVISED PROMPT TEMPLATE >>>
INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant responsible for classifying user intent. Your goal is to analyze the user's query and determine the most efficient path for the agent to take.

Follow this decision hierarchy:
1.  Is the query complex and requires a sequence of multiple actions or tools to be resolved? If YES, classify as **"PLAN"**.
2.  If not a plan, can the query be fully resolved by a **single call** to one of the available tools? If YES, classify as **"DIRECT_TOOL_REQUEST"**.
3.  If it's neither a plan nor a direct tool request, is it a simple question I can answer with my internal knowledge, a conversational remark, or a request for brainstorming? If YES, classify as **"DIRECT_QA"**.

Here are the detailed descriptions for each intent:

### 1. "PLAN"
-   **Use Case**: The query implies a multi-step process, requires breaking down into sub-tasks, involves creating and manipulating multiple pieces of data, or clearly needs a sequence of different tool uses.
-   **Examples**:
    - "Research the latest treatments for X, summarize them, and write a report."
    - "Find three recent news articles about Y, extract key points from each, and compare them."
    - "Download the data from Z, process it, and generate a plot."

### 2. "DIRECT_TOOL_REQUEST"
-   **Use Case**: The query can be fully and directly answered by using a **single** available tool. This is for explicit tool requests or implicit requests that map cleanly to one tool's function.
-   **Task**: You MUST identify the `tool_name` and formulate the `tool_input`.
-   The `tool_name` MUST be one of these: {tool_names_for_prompt}
-   The `tool_input` should be the appropriate input string or JSON string for that specific tool.
-   **Examples**:
    - "Use tavily_search_api to find out about recent AI developments." -> intent: DIRECT_TOOL_REQUEST, tool_name: tavily_search_api, tool_input: {{"query": "recent AI developments"}}
    - "Read the content of the file 'analysis_summary.txt'." -> intent: DIRECT_TOOL_REQUEST, tool_name: read_file, tool_input: "analysis_summary.txt"
    - "Search for news about lung cancer immunotherapy." -> intent: DIRECT_TOOL_REQUEST, tool_name: tavily_search_api, tool_input: {{"query": "news about lung cancer immunotherapy"}}
    - "What is the weather in Zurich?" -> (This implies a single call to a hypothetical 'weather' tool, so you would classify it as a DIRECT_TOOL_REQUEST if such a tool were available).

### 3. "DIRECT_QA"
-   **Use Case**: The query is a straightforward question, a request for a simple definition or explanation, a request for brainstorming, a simple calculation, or a conversational remark that **does not require any tools**. The agent can answer this using its own internal knowledge.
-   **Examples**:
    - "Explain the concept of neural networks in simple terms."
    - "Tell me a fun fact about Switzerland."
    - "Can you help me brainstorm ideas for a project about renewable energy?"
    - "Thanks, that was helpful!"
    - "What is 2 + 2?"

**Your Response Format:**
Respond with a single JSON object matching the following schema. Do not include any preamble or explanation outside of the JSON object.
{format_instructions}
"""
# <<< END REVISED PROMPT TEMPLATE >>>


async def classify_intent(
    user_query: str,
    session_data_entry: Dict[str, Any],
    tool_names_for_prompt: str,
    tools_summary_for_prompt: str
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
    ).replace('{"query": "X"}', '{{"query": "X"}}').replace('{"location": "current"}', '{{"location": "current"}}')

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
            raw_output = await error_chain.ainvoke({"user_query": user_query})
            logger.error(f"IntentClassifier: Raw LLM output on error: {raw_output[:500]}...")
        except Exception as raw_e:
            logger.error(f"IntentClassifier: Failed to get raw LLM output during error: {raw_e}")
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent (string) due to classification error.")
        return "PLAN"