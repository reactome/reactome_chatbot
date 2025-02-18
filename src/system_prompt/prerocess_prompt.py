from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

from src.external_search.state import GraphState

# System message defining appropriateness criteria
safety_check_message = """
You are an expert grader with a strong understanding of ethical guidelines and responsible AI usage.

Your task is to determine whether a user-submitted question is appropriate and not harmful.
Provide a binary output as either:
Yes: The question is appropriate and not harmful.
No: The question is harmful, offensive, inappropriate, or unethical.
"""
# Prompt template for the appropriateness check
safety_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", safety_check_message),
        ("human", "User question: \n\n {user_input}"),
    ]
)


class SafetyCheck(BaseModel):
    """Determines if a user-submitted question is appropriate and non-harmful."""

    binary_score: str = Field(
        description="Indicates whether the question is appropriate and related to molecular biology. Expected values: 'Yes' or 'No'."
    )

class SafetyChecker:
    def __init__(self, llm: BaseChatModel):
        structured_safety_checker: Runnable = llm.with_structured_output(
            SafetyCheck
        )
        self.runnable: Runnable = safety_check_prompt | structured_safety_checker

    async def ainvoke(
        self, state: GraphState, config: RunnableConfig
    ) -> dict[str, str]:
        result: SafetyCheck = await self.runnable.ainvoke(
            {
                "user_input": state["user_input"],
            },
            config,
        )
        return {"safety_check": result.binary_score}



language_detection_message = """
You are an expert linguist capable of identifying languages from text input. 
Your task is to determine the language of the user's question and return it as a single-word response. 

- Return only the language name in English (e.g., "French", "Farsi", "Simplified Chinese").
- Do not return phrases, sentences, or explanations.
- If the language is unknown or ambiguous, return "English".
"""

language_detection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", language_detection_message),
        ("human", "User question: \n\n {user_input}"),
    ]
)

# Contextualize question prompt
contextualize_q_system_prompt = """
You are an expert in question formulation with deep expertise in molecular biology and experience as a Reactome curator. Your task is to analyze the conversation history and the user’s latest query to fully understand their intent and what they seek to learn.
If the user's question is not in English, reformulate the question and translate it to English, ensuring the meaning and intent are preserved.
Reformulate the user’s question into a standalone version that retains its full meaning without requiring prior context. The reformulated question should be:
    - Clear, concise, and precise
    - Optimized for both vector search (semantic meaning) and case-sensitive keyword search
    - Faithful to the user’s intent and scientific accuracy

the returned question should always be in English.
If the user’s question is already in English, self-contained and well-formed, return it as is.
Do NOT answer the question or provide explanations.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)