from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

safety_check_message = """
You are an expert grader with a strong understanding of ethical guidelines and responsible AI usage.

Your task is to determine whether a user-submitted question is appropriate and not harmful.
Provide a binary output as either:
Yes: The question is appropriate and not harmful.
No: The question is harmful, offensive, inappropriate, or unethical.
"""

safety_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", safety_check_message),
        ("human", "User question: \n\n {input}"),
    ]
)


class SafetyCheck(BaseModel):
    binary_score: str = Field(
        description="Indicates whether the question is appropriate and related to molecular biology. Expected values: 'Yes' or 'No'."
    )


def create_safety_checker(llm: BaseChatModel) -> Runnable:
    return safety_check_prompt | llm.with_structured_output(SafetyCheck)
