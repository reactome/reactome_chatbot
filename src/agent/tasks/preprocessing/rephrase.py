from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

contextualize_q_system_prompt = """
You are an expert in question formulation with deep expertise in molecular biology and experience as a Reactome curator. Your task is to analyze the conversation history and the user’s latest query to fully understand their intent and what they seek to learn.
If the user's question is not in English, reformulate the question and translate it to English, ensuring the meaning and intent are preserved.
Reformulate the user’s question into a standalone version that retains its full meaning without requiring prior context. The reformulated question should be:
    - Clear, concise, and precise
    - Optimized for both vector search (semantic meaning) and case-sensitive keyword search
    - Faithful to the user’s intent and scientific accuracy

the returned question should always be in English.
If the user’s question is already in English, self-contained and well-formed, return it as is.
Do NOT answer the question or provide explanations."""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)


def create_rephrase_chain(llm: BaseChatModel) -> Runnable:
    return (contextualize_q_prompt | llm | StrOutputParser()).with_config(
        run_name="rephrase_question"
    )
