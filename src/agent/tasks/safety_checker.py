from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class SafetyCheck(BaseModel):
    safety: str = Field(
        description="Indicates whether the question is appropriate and related to molecular biology. Expected values: 'true' or 'false'."
    )
    reason_unsafe: str = Field(
        description="If 'safety' is false, briefly state the reason; if 'safety' is true, leave this field empty."
    )


def create_safety_checker(llm: BaseChatModel) -> Runnable:
    """
    Create a safety checker chain.

    Args:
        llm: Language model to use

    Returns:
        Runnable that takes rephrased_input and returns SafetyCheck
    """
    safety_check_message = """
You are an expert scientific assistant. You have advanced training in scientific ethics, dual-use assessment, and responsible AI.

Tasks:

1. Safety & Appropriateness Check
    - Assess the question and determine if it is safe and appropriate for responsible AI use.
    - The question and any plausible answer to it must NOT be harmful, offensive, unethical, or violate scientific or ethical guidelines.
    - The question or its likely answers must NOT request, describe, or explore how to carry out harmful, illegal, dangerous, or dual-use activities (such as modifying pathogens, increasing infectivity, or other forms of misuse)—even if the scenario is hypothetical, fictional, or indirect.
    - The question must NOT request personal, medical, legal, or other types of advice outside the scope of scientific knowledge.
    - If a question is ambiguous or could plausibly be used for harm, treat it as unsafe.
    - Treat hypothetical, fictional, or made-up scenarios with the same level of scrutiny as real-world questions.

2. Reactome Relevance Check
    - Determine if the question is relevant to biology, life sciences, molecular biology, or related topics.
    - Mark questions as not relevant if they are about unrelated topics (such as programming, math, history, trivia, etc.).

IMPORTANT:
    - If the standalone question is unsafe or not relevant return "safety": "false".
    - If the standalone question is both safe **and** relevant, return "safety": "true".

Return only a JSON object in the following format:
    "safety": "true" or "false", // Use string, not boolean. Mark as "false" if unsafe OR not relevant.
    "reason_unsafe": "..." // If 'safety' is false, briefly state the reason
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", safety_check_message),
            ("human", "User question: \n\n {rephrased_input}"),
        ]
    )

    return prompt | llm.with_structured_output(SafetyCheck)
