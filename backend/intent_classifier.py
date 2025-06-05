# backend/intent_classifier.py
import logging
from typing import Dict, Any, Optional

from backend.config import settings
from backend.llm_setup import get_llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field # Using Pydantic v2

logger = logging.getLogger(__name__)

class IntentClassificationOutput(BaseModel):
    intent: str = Field(description="The classified intent. Must be one of ['PLAN', 'DIRECT_QA', 'DIRECT_TOOL_REQUEST'].")
    reasoning: Optional[str] = Field(description="Brief reasoning for the classification.", default=None)
    identified_tool_name: Optional[str] = Field(
        default=None,
        description="If intent is 'DIRECT_TOOL_REQUEST', the exact name of the tool identified from the 'Available Tools' list. Otherwise null."
    )
    extracted_tool_input: Optional[str] = Field(
        default=None,
        description="If intent is 'DIRECT_TOOL_REQUEST', the precise, complete input string for the identified tool. Otherwise null."
    )
    model_config = {"arbitrary_types_allowed": False}


INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant responsible for classifying user intent and extracting information for direct tool use.
Your goal is to determine if a user's query requires:
1.  A multi-step plan ("PLAN").
2.  A simple direct answer without tools ("DIRECT_QA").
3.  A direct request to use a specific tool ("DIRECT_TOOL_REQUEST").

Available intents:
-   "PLAN": Use this if the query implies a multi-step process, requires breaking down into sub-tasks, involves creating or manipulating multiple pieces of data, or clearly needs a sequence of tool uses where the tools are NOT explicitly named or the input is not straightforward.
    Examples:
    - "Research the latest treatments for X, summarize them, and write a report."
    - "Find three recent news articles about Y, extract key points from each, and compare them."
    - "Write a poem about stars and save it to 'poem.txt'." (Involves generation then a tool)
    - **Critically, if the user asks about your capabilities, what tools you have, or how you work (e.g., "What tools can you use?", "List your tools."), you MUST classify this as "PLAN". This is because answering requires retrieving and presenting structured information, not a simple direct answer.**

-   "DIRECT_QA": Use this ONLY if the query is a straightforward question, a request for a simple definition or explanation, a request for brainstorming that doesn't imply tool use, a simple calculation, or a conversational remark that doesn't require a complex plan or explicit tool. The agent can likely answer this using its internal knowledge.
    Examples:
    - "What is the capital of France?"
    - "Explain the concept of X in simple terms."
    - "Tell me a fun fact."
    - "Thanks, that was helpful!"
    - **Do NOT use for questions about your tools; use "PLAN" for that.**

-   "DIRECT_TOOL_REQUEST": Use this if the query explicitly and clearly asks to use a specific tool for a single, direct action, and the input for that tool is evident in the query.
    You MUST match the requested action to one of the "Available Tools" listed below.
    - If a match is found:
        - `intent` should be "DIRECT_TOOL_REQUEST".
        - `identified_tool_name` should be the EXACT name of the tool from the "Available Tools" list.
        - `extracted_tool_input` should be the PRECISE and COMPLETE input string that the identified tool expects, based on its description and the user's query. Ensure the input format matches the tool's requirements (e.g., for 'write_file', it's 'filepath:::content').
    - Examples:
        - User: "Use tavily_search_api to find recent news on AI in healthcare."
            - intent: "DIRECT_TOOL_REQUEST"
            - identified_tool_name: "tavily_search_api"
            - extracted_tool_input: (JSON string) "{{\"query\": \"recent news on AI in healthcare\"}}" # Ensure JSON quotes are escaped for the string value
        - User: "Read the file 'summary.txt' from my workspace."
            - intent: "DIRECT_TOOL_REQUEST"
            - identified_tool_name: "read_file"
            - extracted_tool_input: "summary.txt"

**Available Tools (for DIRECT_TOOL_REQUEST matching):**
{available_tools_summary}

**Input Formatting Notes for Specific Tools (for DIRECT_TOOL_REQUEST):**
- `tavily_search_api`: Input is a JSON string, e.g., '{{"query": "search terms", "max_results": 5}}'.
- `web_page_reader`: Input is a single URL string.
- `write_file`: Input is 'relative_file_path:::text_content'.
- `read_file`: Input is 'relative_file_path'.
- `workspace_shell`: Input is a shell command string.
- `pubmed_search`: Input is a query string, optionally with ' max_results=N'.
- `deep_research_synthesizer`: Input is a JSON string, e.g., '{{"query": "research topic"}}'.
- `Python_REPL`: Input is a Python code string.
- `python_package_installer`: Input is a package name string.

Consider the complexity. If a query mentions a tool but requires prior steps (e.g., "Generate a summary and then write it to summary.txt"), it's a "PLAN".
If the query is vague about the tool or input, or seems to imply multiple operations even if a tool is named, prefer "PLAN".

Respond with a single JSON object matching the following schema:
{format_instructions}

Do not include any preamble or explanation outside of the JSON object.
"""

async def classify_intent(
    user_query: str,
    session_data_entry: Dict[str, Any], 
    available_tools_summary: str 
) -> IntentClassificationOutput: 
    logger.info(f"IntentClassifier: Classifying intent for query: {user_query[:100]}...")
    logger.debug(f"IntentClassifier: Tools summary provided: {available_tools_summary[:200]}...")

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
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent due to LLM initialization error.")
        return IntentClassificationOutput(intent="PLAN", reasoning="LLM initialization error.")

    parser = JsonOutputParser(pydantic_object=IntentClassificationOutput)
    format_instructions = parser.get_format_instructions()

    human_template = "User Query: \"{user_query}\"\nClassify the intent of the user query based on the system instructions and provided tool summary."

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFIER_SYSTEM_PROMPT_TEMPLATE), 
        ("human", human_template)
    ])
    chain = prompt | intent_llm | parser

    try:
        invoke_params = {
            "user_query": user_query,
            "format_instructions": format_instructions,
            "available_tools_summary": available_tools_summary 
        }

        classification_result_dict = await chain.ainvoke(invoke_params)

        if isinstance(classification_result_dict, IntentClassificationOutput):
            classified_output = classification_result_dict
        elif isinstance(classification_result_dict, dict):
            classified_output = IntentClassificationOutput(**classification_result_dict)
        else:
            logger.error(f"IntentClassifier: LLM chain returned unexpected type: {type(classification_result_dict)}. Output: {str(classification_result_dict)[:200]}")
            logger.warning("IntentClassifier: Defaulting to 'PLAN' due to unexpected output type.")
            return IntentClassificationOutput(intent="PLAN", reasoning="LLM returned unexpected output type.")

        valid_intents = ["PLAN", "DIRECT_QA", "DIRECT_TOOL_REQUEST"]
        # Ensure intent is uppercase for comparison
        classified_intent_upper = classified_output.intent.upper() if classified_output.intent else ""

        if classified_intent_upper not in valid_intents:
            logger.warning(f"IntentClassifier: LLM returned an unknown intent value '{classified_output.intent}'. Defaulting to 'PLAN'.")
            classified_output.intent = "PLAN"
            classified_output.reasoning = f"Original intent '{classified_output.intent}' was invalid, defaulted to PLAN. {classified_output.reasoning or ''}".strip()
            classified_output.identified_tool_name = None 
            classified_output.extracted_tool_input = None

        if classified_intent_upper == "DIRECT_TOOL_REQUEST":
            if not classified_output.identified_tool_name:
                logger.warning(f"IntentClassifier: Intent is DIRECT_TOOL_REQUEST but identified_tool_name is missing. Query: '{user_query}'. Reasoning: '{classified_output.reasoning}'. Changing to PLAN.")
                classified_output.intent = "PLAN" # Override intent
                classified_output.reasoning = f"DIRECT_TOOL_REQUEST identified, but no tool name extracted. Changed to PLAN. {classified_output.reasoning or ''}".strip()
                classified_output.identified_tool_name = None
                classified_output.extracted_tool_input = None 
        
        # Ensure intent is stored in uppercase if it was changed.
        classified_output.intent = classified_output.intent.upper()


        logger.info(f"IntentClassifier: Final Classified intent as '{classified_output.intent}'. "
                    f"Tool: '{classified_output.identified_tool_name}', Input: '{str(classified_output.extracted_tool_input)[:50]}...'. "
                    f"Reasoning: {classified_output.reasoning}")
        
        return classified_output

    except Exception as e:
        logger.error(f"IntentClassifier: Error during intent classification: {e}", exc_info=True)
        try:
            error_chain = prompt | intent_llm | StrOutputParser()
            raw_output_params = {
                "user_query": user_query,
                "format_instructions": format_instructions,
                "available_tools_summary": available_tools_summary
            }
            raw_output = await error_chain.ainvoke(raw_output_params)
            logger.error(f"IntentClassifier: Raw LLM output on error: {raw_output[:500]}...")
        except Exception as raw_e:
            logger.error(f"IntentClassifier: Failed to get raw LLM output during error: {raw_e}")
        logger.warning("IntentClassifier: Defaulting to 'PLAN' intent due to classification error.")
        return IntentClassificationOutput(intent="PLAN", reasoning=f"Classification error: {e}")

