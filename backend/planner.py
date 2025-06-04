import logging
from typing import List, Dict, Any, Tuple, Optional

from backend.config import settings
from backend.llm_setup import get_llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field # Ensure Pydantic v1 for Langchain compatibility
import asyncio

logger = logging.getLogger(__name__)

# Define the structure for a single step in the plan
class PlanStep(BaseModel):
    step_id: int = Field(description="A unique sequential identifier for this step, starting from 1.")
    description: str = Field(description="A concise, human-readable description of what this single sub-task aims to achieve.")
    tool_to_use: Optional[str] = Field(description="The exact name of the tool to be used for this step, if any. Must be one of the available tools or 'None'.")
    tool_input_instructions: Optional[str] = Field(description="Specific instructions or key parameters for the tool_input, if a tool is used. This is not the full tool input itself, but guidance for forming it.")
    expected_outcome: str = Field(description="What is the expected result or artifact from completing this step successfully? For generative 'No Tool' steps, this should describe the actual content to be produced (e.g., 'The text of a short poem about stars.'). For tool steps or steps producing data for subsequent steps, describe the state or data (e.g., 'File 'data.csv' downloaded.', 'The full text of 'report.txt' is available.').")

# Define the structure for the overall plan
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

**Handling Multi-Part User Queries for Final Output:**
If the original user query explicitly or implicitly asks for multiple distinct pieces of information to be delivered in the final response (e.g., "tell me X, and also provide Y"), ensure your plan includes all necessary steps to gather or generate each piece of information.
**Crucially, after all individual pieces of information have been gathered by preceding steps and need to be presented to the user as a combined answer, you MUST add a final "No Tool" step.**
-   The `description` for this final synthesis step should clearly state that it's combining the necessary prior outputs to fully address the user's original request.
-   The `tool_input_instructions` (for this "No Tool" synthesis step) should guide the LLM on which prior step outputs need to be synthesized, for example: "Combine the result from Step X [e.g., the extracted fact] and the result from Step Y [e.g., the summary] into a complete answer to the user's original query: '{user_query}'".
-   The `expected_outcome` for this synthesis step should be: "A single, consolidated final answer addressing all parts of the user's original request is generated."

Additionally, provide a `human_readable_summary` of the entire plan that can be shown to the user for confirmation.
Ensure the output is a JSON object that strictly adheres to the following JSON schema:
{format_instructions}

Do not include any preamble or explanation outside of the JSON object."""

async def generate_plan(
    user_query: str,
    available_tools_summary: str
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Generates a multi-step plan based on the user query using an LLM.
    Fetches its own LLM based on settings.
    """
    logger.info(f"Planner: Generating plan for user query: {user_query[:100]}...")

    try:
        planner_llm: BaseChatModel = get_llm(
            settings,
            provider=settings.planner_provider,
            model_name=settings.planner_model_name,
            requested_for_role="Planner"
        ) 
        logger.info(f"Planner: Using LLM {settings.planner_provider}::{settings.planner_model_name}")
    except Exception as e:
        logger.error(f"Planner: Failed to initialize LLM: {e}", exc_info=True)
        return None, None

    parser = JsonOutputParser(pydantic_object=AgentPlan)
    format_instructions = parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT_TEMPLATE),
        ("human", "{user_query}") 
    ])

    chain = prompt_template | planner_llm | parser

    try:
        logger.debug(f"Planner prompt input variables: {prompt_template.input_variables}")
        planned_result_dict = await chain.ainvoke({
            "user_query": user_query, 
            "available_tools_summary": available_tools_summary,
            "format_instructions": format_instructions
        })

        agent_plan: AgentPlan
        if isinstance(planned_result_dict, AgentPlan):
            agent_plan = planned_result_dict
        elif isinstance(planned_result_dict, dict):
            agent_plan = AgentPlan(**planned_result_dict)
        else:
            logger.error(f"Planner LLM call returned an unexpected type: {type(planned_result_dict)}. Content: {planned_result_dict}")
            try:
                raw_output_chain = prompt_template | planner_llm | StrOutputParser()
                raw_output = await raw_output_chain.ainvoke({
                    "user_query": user_query,
                    "available_tools_summary": available_tools_summary,
                    "format_instructions": format_instructions
                })
                logger.error(f"Planner: Raw LLM output on parsing error: {raw_output}")
            except Exception as raw_e:
                logger.error(f"Planner: Failed to get raw LLM output on parsing error: {raw_e}")
            return None, None

        human_summary = agent_plan.human_readable_summary
        structured_steps = []
        for i, step_model in enumerate(agent_plan.steps):
            step_dict = step_model.dict()
            step_dict['step_id'] = i + 1 
            structured_steps.append(step_dict)


        logger.info(f"Planner: Plan generated successfully. Summary: {human_summary}")
        logger.debug(f"Planner: Structured plan: {structured_steps}")
        
        # --- ADDED DEBUG LOGGING ---
        logger.info(f"Planner (DEBUG): Returning human_summary (type: {type(human_summary)}): '{str(human_summary)[:200]}...'")
        logger.info(f"Planner (DEBUG): Returning structured_steps (type: {type(structured_steps)}, length: {len(structured_steps) if isinstance(structured_steps, list) else 'N/A'}): '{str(structured_steps)[:200]}...'")
        # --- END ADDED DEBUG LOGGING ---
        
        return human_summary, structured_steps

    except Exception as e:
        logger.error(f"Planner: Error during plan generation: {e}", exc_info=True)
        try:
            error_chain = prompt_template | planner_llm | StrOutputParser() 
            raw_output = await error_chain.ainvoke({
                "user_query": user_query,
                "available_tools_summary": available_tools_summary,
                "format_instructions": format_instructions
            })
            logger.error(f"Planner: Raw LLM output on error: {raw_output}")
        except Exception as raw_e:
            logger.error(f"Planner: Failed to get raw LLM output on error: {raw_e}")
        return None, None

if __name__ == '__main__':
    async def test_planner_poem_generation():
        query_poem = "Create a file called poem.txt and write in it a small poem about stars."
        tools_summary = "- write_file: To write files to workspace.\n- read_file: To read files from workspace."
        
        print(f"\n--- Testing Planner with Poem Generation Query ---")
        print(f"Query: {query_poem}")
        summary, plan = await generate_plan(query_poem, tools_summary)

        if summary and plan:
            print("---- Human Readable Summary ----")
            print(summary)
            print("\n---- Structured Plan ----")
            for i, step_data in enumerate(plan):
                if isinstance(step_data, dict):
                    print(f"Step {step_data.get('step_id', i+1)}:")
                    print(f"  Description: {step_data.get('description')}")
                    print(f"  Tool: {step_data.get('tool_to_use')}")
                    print(f"  Input Instructions: {step_data.get('tool_input_instructions')}")
                    print(f"  Expected Outcome: {step_data.get('expected_outcome')}") 
                else:
                    print(f"Step {i+1}: Invalid step data format: {step_data}")
            
            if len(plan) > 0 and "poem about stars" in plan[0].get("description", "").lower():
                print(f"\nDEBUG: Expected outcome for poem generation step (Step 1): '{plan[0].get('expected_outcome')}'")
                if "actual text" in plan[0].get('expected_outcome', '').lower() or \
                   "generated poem" in plan[0].get('expected_outcome', '').lower() or \
                   "the poem itself" in plan[0].get('expected_outcome', '').lower():
                    print("DEBUG: Step 1 expected outcome seems correctly formulated for direct content generation.")
                else:
                    print("DEBUG WARNING: Step 1 expected outcome might still be too indirect for direct content generation.")
        else:
            print("Failed to generate a plan for poem generation query.")

    asyncio.run(test_planner_poem_generation())
