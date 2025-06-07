# backend/pydantic_models.py
"""
This file centralizes all Pydantic data models used across the ResearchAgent application.
This helps in avoiding circular import errors and provides a single source of truth for data structures.
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# --- Intent Classifier Model ---
class IntentClassificationOutput(BaseModel):
    """
    Defines the structured output for the Intent Classifier LLM.
    """
    intent: str = Field(description="The classified intent. Must be one of ['PLAN', 'DIRECT_QA'].")
    reasoning: Optional[str] = Field(description="Brief reasoning for the classification.", default=None)

# --- Planner Models ---
class PlanStep(BaseModel):
    """
    Defines the structure for a single step in the agent's plan.
    """
    step_id: int = Field(description="A unique sequential identifier for this step, starting from 1.")
    description: str = Field(description="A concise, human-readable description of what this single sub-task aims to achieve.")
    tool_to_use: Optional[str] = Field(description="The exact name of the tool to be used for this step, if any. Must be one of the available tools or 'None'.")
    tool_input_instructions: Optional[str] = Field(description="Specific instructions or key parameters for the tool_input, if a tool is used. This is not the full tool input itself, but guidance for forming it.")
    expected_outcome: str = Field(description="What is the expected result or artifact from completing this step successfully? For generative 'No Tool' steps, this should describe the actual content to be produced (e.g., 'The text of a short poem about stars.'). For tool steps or steps producing data for subsequent steps, describe the state or data (e.g., 'File 'data.csv' downloaded.', 'The full text of 'report.txt' is available.').")

class AgentPlan(BaseModel):
    """
    Defines the overall structure for the agent's plan.
    """
    human_readable_summary: str = Field(description="A brief, conversational summary of the overall plan for the user.")
    steps: List[PlanStep] = Field(description="A list of detailed steps to accomplish the user's request.")

# --- Controller Model ---
class ControllerOutput(BaseModel):
    """
    Defines the structured output for the Controller LLM.
    """
    tool_name: Optional[str] = Field(description="The exact name of the tool to use, or 'None' if no tool is directly needed for this step. Must be one of the available tools.")
    tool_input: Optional[str] = Field(default=None, description="The precise, complete input string for the chosen tool, or a concise summary/directive for the LLM if tool_name is 'None'. Can be explicitly 'None' or null if the step description and expected outcome are self-sufficient for a 'None' tool LLM action.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating the controller's confidence in this tool/input choice for the step. 1.0 is high confidence.")
    reasoning: str = Field(description="Brief explanation of why this tool/input was chosen or why no tool is needed.")

# --- Step Evaluator Model ---
class StepCorrectionOutcome(BaseModel):
    """
    Defines the structured output for the Step Evaluator LLM.
    """
    step_achieved_goal: bool = Field(description="Boolean indicating if the step's primary goal was achieved based on the executor's output.")
    assessment_of_step: str = Field(description="Detailed assessment of the step's outcome, explaining why it succeeded or failed. If failed, explain the discrepancy between expected and actual outcome.")
    is_recoverable_via_retry: Optional[bool] = Field(description="If step_achieved_goal is false, boolean indicating if the step might be recoverable with a retry using a modified approach. Null if goal achieved.", default=None)
    suggested_new_tool_for_retry: Optional[str] = Field(description="If recoverable, the suggested tool name for the retry attempt (can be 'None'). Null if not recoverable or goal achieved.", default=None)
    suggested_new_input_instructions_for_retry: Optional[str] = Field(
        description="If recoverable, new or revised input instructions for the Controller to formulate the tool_input for the retry. Null if not recoverable or goal achieved. **Crucially, if the failure was because the Executor described its action or output instead of providing the direct content, these instructions MUST explicitly guide the Controller to instruct the Executor to output ONLY the raw content itself in its 'Final Answer'.**"
    )
    confidence_in_correction: Optional[float] = Field(description="If recoverable, confidence (0.0-1.0) in the suggested correction. Null if not recoverable or goal achieved.", default=None)

# --- Overall Evaluator Model ---
class EvaluationResult(BaseModel):
    """
    Defines the structured output for the Overall Plan Evaluator LLM.
    """
    overall_success: bool = Field(description="Boolean indicating if the overall user query was successfully addressed by the executed plan.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating confidence in the success/failure assessment.")
    assessment: str = Field(description="A concise, user-facing explanation of why the plan succeeded or failed overall, OR a brief confirmation if successful and final_answer_content is provided.")
    final_answer_content: Optional[str] = Field(
        default=None,
        description="If overall_success is True and the plan generated a direct user-facing answer (typically from the last step), this field MUST contain that exact, complete answer. Otherwise, it should be null or omitted."
    )
    suggestions_for_replan: Optional[List[str]] = Field(description="If the plan failed, a list of high-level suggestions for how a new plan might better achieve the goal. Null if successful.", default=None)
