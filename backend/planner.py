# backend/planner.py
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field, ValidationError # Ensure Pydantic v2
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager
import asyncio

from backend.config import settings
from backend.llm_setup import get_llm
from backend.callbacks import LOG_SOURCE_PLANNER

logger = logging.getLogger(__name__)

class PlanStep(BaseModel):
    step_id: int = Field(description="A unique sequential identifier for this step, starting from 1.")
    description: str = Field(description="A concise, human-readable description of what this single sub-task aims to achieve.")
    tool_to_use: Optional[str] = Field(default=None, description="The exact name of the tool to be used for this step, if any. Must be one of the available tools or 'None'.")
    tool_input_instructions: Optional[str] = Field(default=None, description="Specific instructions or key parameters for the tool_input, if a tool is used. This is not the full tool input itself, but guidance for forming it.")
    expected_outcome: str = Field(description="What is the expected result or artifact from completing this step successfully? For generative 'No Tool' steps, this should describe the actual content to be produced (e.g., 'The text of a short poem about stars.'). For tool steps or producing data for subsequent steps, describe the state or data (e.g., 'File 'data.csv' downloaded.', 'The full text of 'report.txt' is available.').")

class AgentPlan(BaseModel):
    human_readable_summary: str = Field(description="A brief, conversational summary of the overall plan for the user.")
    steps: List[PlanStep] = Field(description="A list of detailed steps to accomplish the user's request.")

PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are an expert planning assistant for a research agent.
Your goal is to take a user's complex request and break it down into a sequence of logical, actionable sub-tasks.
The research agent has access to the following tools:
{available_tools_summary}

For each sub-task in the plan:
1.  Provide a clear `description` of what the sub-task aims to achieve.
2.  If a tool is needed, specify the exact `tool_to_use` from the list of available tools.
3.  If no tool is directly needed for a step (e.g., the LLM itself will perform the reasoning, summarization, or content generation), use "None" for `tool_to_use`.
    -   For such "No Tool" steps that involve the LLM generating specific text output (e.g., a poem, a summary, a specific format of data like JSON or Markdown):
        -   The `description` should clearly state the generation task (e.g., "Generate a short poem about stars", "Summarize the previous findings").
        -   The `expected_outcome` for these generative "No Tool" steps MUST clearly state that the *actual generated content itself* is the outcome. For example:
            - "The text of a short poem about stars." (NOT "A poem is generated and available")
            - "A concise summary of the key findings from the search results." (NOT "A summary is created")
            - "A JSON object listing the extracted entities." (NOT "Entities are extracted and available in JSON")
    -   If a "No Tool" step involves complex, multi-part generation or intricate formatting (e.g., a detailed Markdown table from unstructured data), consider breaking it into two "No Tool" steps:
        1.  An initial step to generate the core data/elements. `expected_outcome`: "Intermediate data/elements for X are generated."
        2.  A subsequent step to format this intermediate data. `expected_outcome`: "The [data from step 1] formatted as a Markdown table."
4.  Provide brief `tool_input_instructions` highlighting key parameters or data the tool might need if a tool is used. This is NOT the full tool input, but guidance for the agent when it forms the tool input later. For "None" tool steps, this can be a brief note on what information the LLM should focus on if not obvious from the description.
5.  State the `expected_outcome` of successfully completing the step.
    -   For tool steps or steps producing data primarily for *subsequent processing* within the plan (not direct user presentation yet): the `expected_outcome` should describe the state or data becoming available (e.g., "File 'data.csv' downloaded to workspace", "The full text content of the file 'report.txt' is returned and available for the next step", "The list of search results is available.").
    -   **Crucially, for "No Tool" steps that are intended to generate the final piece of information or creative content for that part of the plan, the `expected_outcome` MUST describe the actual content itself, as shown in point 3.**

**Handling Multi-Part User Queries & Final Output Synthesis:**
If the original user query explicitly or implicitly asks for multiple distinct pieces of information to be delivered, or involves the creation of files or multiple distinct outputs (e.g., "generate two poems and save them", "research X and write a report to report.txt"), ensure your plan includes all necessary steps to gather/generate each piece and perform any actions like file writing.
**Crucially, after all individual pieces of information have been gathered/generated and actions (like file writing) have been performed by preceding steps, you MUST add a final "No Tool" step to synthesize these outcomes into a comprehensive final answer for the user.**
-   The `description` for this final synthesis step should clearly state its purpose, for example: "Synthesize all findings and actions into a final comprehensive answer for the user."
-   The `tool_input_instructions` (for this "No Tool" synthesis step) should guide the LLM on what prior step outputs and actions need to be confirmed and/or synthesized. For example: "Review the outcomes of all previous steps. If files were written (e.g., 'poem1.txt', 'analysis_report.pdf'), confirm their successful creation and briefly state their purpose or content. Combine this confirmation with any other generated information (e.g., search result summaries, direct answers from previous steps) into a single, coherent, and user-friendly response that fully addresses the original query: '{user_query}'. If the query asked for content to be generated and saved, you can optionally include a very brief snippet of the generated content in your final answer if it's concise and relevant."
-   The `expected_outcome` for this synthesis step MUST be: "A single, consolidated final answer that confirms all actions taken (like file creation) and presents all requested information, thereby fully addressing the user's original request, is generated."

Additionally, provide a `human_readable_summary` of the entire plan that can be shown to the user for confirmation.
Ensure the output is a JSON object that strictly adheres to the following JSON schema:
{format_instructions}

Do not include any preamble or explanation outside of the JSON object."""

async def generate_plan(
    user_query: str,
    available_tools_summary: str,
    callback_handler: Union[List[BaseCallbackHandler], BaseCallbackManager, None] = None
) -> Dict[str, Any]:
    logger.info(f"Planner: Generating plan for user query: {user_query[:100]}...")
    if callback_handler:
        logger.debug(f"Planner: Received callback_handler of type: {type(callback_handler)}")

    try:
        planner_llm: BaseChatModel = get_llm(
            settings,
            provider=settings.planner_provider,
            model_name=settings.planner_model_name,
            requested_for_role=LOG_SOURCE_PLANNER,
            callbacks=callback_handler
        )
        logger.info(f"Planner: Using LLM {settings.planner_provider}::{settings.planner_model_name}")
    except Exception as e:
        logger.error(f"Planner: Failed to initialize LLM: {e}", exc_info=True)
        return {
            "plan_summary": None,
            "plan_steps": None,
            "plan_generation_error": f"LLM initialization failed: {e}"
        }

    parser = JsonOutputParser(pydantic_object=AgentPlan)
    format_instructions = parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT_TEMPLATE),
        ("human", "{user_query}")
    ])

    chain = prompt_template | planner_llm | parser
    
    invoke_payload = { # Defined here so it's available for error handling's raw output call
        "user_query": user_query,
        "available_tools_summary": available_tools_summary,
        "format_instructions": format_instructions
    }
    run_config = RunnableConfig( # Defined here for the same reason
        callbacks=callback_handler if isinstance(callback_handler, BaseCallbackManager) else ([callback_handler] if callback_handler else None),
        metadata={"component_name": LOG_SOURCE_PLANNER}
    )

    try:
        # The parser is *supposed* to return the Pydantic object directly.
        # Handle if it returns a dict instead.
        parsed_llm_output = await chain.ainvoke(invoke_payload, config=run_config)
        
        agent_plan_model: AgentPlan
        if isinstance(parsed_llm_output, AgentPlan):
            agent_plan_model = parsed_llm_output
        elif isinstance(parsed_llm_output, dict):
            logger.debug("Planner: LLM output parser returned a dict, attempting to validate with Pydantic model AgentPlan.")
            try:
                agent_plan_model = AgentPlan(**parsed_llm_output)
            except ValidationError as ve_plan:
                logger.error(f"Planner: Pydantic validation failed for AgentPlan from LLM output dict: {ve_plan}. Raw dict: {parsed_llm_output}", exc_info=True)
                raise # Re-raise to be caught by the outer try-except
        else:
            logger.error(f"Planner: Unexpected output type from LLM parser chain: {type(parsed_llm_output)}. Output: {str(parsed_llm_output)[:500]}")
            raise TypeError(f"Unexpected output type from LLM parser for AgentPlan: {type(parsed_llm_output)}")
        
        human_summary = agent_plan_model.human_readable_summary
        structured_steps_dicts = [step.model_dump() for step in agent_plan_model.steps]
        
        for i, step_dict in enumerate(structured_steps_dicts):
            step_dict['step_id'] = step_dict.get('step_id', i + 1)

        logger.info(f"Planner: Plan generated successfully. Summary: {human_summary}")
        logger.debug(f"Planner: Structured plan (as dicts): {structured_steps_dicts}")
        return {
            "plan_summary": human_summary,
            "plan_steps": structured_steps_dicts,
            "plan_generation_error": None
        }

    except ValidationError as ve: 
        logger.error(f"Planner: Pydantic validation error during plan generation: {ve}", exc_info=True)
        error_message = f"Plan parsing failed (validation error): {ve}"
    except Exception as e: 
        logger.error(f"Planner: Error during plan generation: {e}", exc_info=True)
        error_message = f"Unexpected error during plan generation: {e}"
    
    try:
        error_chain = prompt_template | planner_llm | StrOutputParser()
        raw_output = await error_chain.ainvoke(
            invoke_payload, 
            config=RunnableConfig(
                callbacks=callback_handler if isinstance(callback_handler, BaseCallbackManager) else ([callback_handler] if callback_handler else None),
                metadata={"component_name": LOG_SOURCE_PLANNER + "_ERROR_HANDLER"}
            )
        )
        logger.error(f"Planner: Raw LLM output on error: {raw_output}")
        error_message += f". Raw LLM output: {raw_output[:200]}..."
    except Exception as raw_e:
        logger.error(f"Planner: Failed to get raw LLM output on error: {raw_e}")
        error_message += ". Failed to retrieve raw LLM output."

    return {
        "plan_summary": None,
        "plan_steps": None,
        "plan_generation_error": error_message
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    async def test_planner_main():
        query_example = "Research the benefits of solar power and write a short summary file called 'solar_benefits.txt'."
        tools_summary_example = (
            "- tavily_search_api: For web searches.\n"
            "- web_page_reader: To read content from a specific URL.\n"
            "- write_file: To write text content to a file in the workspace."
        )
        
        print(f"\n--- Testing Planner for query: \"{query_example}\" ---")
        plan_result_dict = await generate_plan(query_example, tools_summary_example, callback_handler=None)
        
        if plan_result_dict.get("plan_generation_error"):
            print(f"ERROR generating plan: {plan_result_dict['plan_generation_error']}")
        elif plan_result_dict.get("plan_summary") and plan_result_dict.get("plan_steps"):
            print("\n---- Human Readable Summary ----")
            print(plan_result_dict["plan_summary"])
            print("\n---- Structured Plan (List of Dictionaries) ----")
            for i, step_data_dict in enumerate(plan_result_dict["plan_steps"]):
                print(f"Step {step_data_dict.get('step_id', i+1)}:")
                print(f"  Description: {step_data_dict.get('description')}")
                print(f"  Tool: {step_data_dict.get('tool_to_use')}")
                print(f"  Input Instructions: {step_data_dict.get('tool_input_instructions')}")
                print(f"  Expected Outcome: {step_data_dict.get('expected_outcome')}")
        else:
            print("Failed to generate a plan for the query. No summary or steps returned.")
        print("--------------------------------------------------\n")

    asyncio.run(test_planner_main())
