from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from util.answer_text import AnswerTextParser

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


def create_language_detector(llm: BaseChatModel) -> Runnable:
    return (
        language_detection_prompt
        | llm
        | AnswerTextParser().with_config(run_name="detect_language")
    )
