from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

SourceName = Literal["reactome", "userguide"]

intent_classifier_message = """
You route user questions for the React-to-Me assistant to the correct knowledge source.

Choose exactly one source:

- **reactome**: Questions about biology, molecular mechanisms, pathways, reactions, proteins, genes,
  diseases, and other scientific content in the Reactome Knowledgebase.
  Examples: "What is apoptosis?", "Which pathways involve TP53?", "What does CDK5 do?"

- **userguide**: Questions about how to use the Reactome **website**, tools, or interface.
  Examples: "How do I use the pathway browser?", "How do I search Reactome?",
  "How do I run gene list analysis?", "What is the Details Panel?"

Rules:
- If the user asks how to perform a task in Reactome or about UI features, choose **userguide**.
- If the user asks about biological facts or pathway content, choose **reactome**.
- When unsure, prefer **reactome** for science content and **userguide** for clear how-to or UI questions.
"""

intent_classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", intent_classifier_message),
        ("human", "User question:\n\n{rephrased_input}"),
    ]
)


class QueryIntent(BaseModel):
    source: SourceName = Field(
        description="The knowledge source that should answer this question: 'reactome' or 'userguide'."
    )


_FALLBACK_ORDER: tuple[SourceName, ...] = ("reactome", "userguide")


def resolve_active_sources(
    source: SourceName,
    available_sources: frozenset[SourceName],
) -> list[SourceName]:
    if not available_sources:
        raise ValueError("available_sources must not be empty")
    if source in available_sources:
        return [source]
    for fallback in _FALLBACK_ORDER:
        if fallback in available_sources:
            return [fallback]
    return [next(iter(available_sources))]


def create_intent_classifier(llm: BaseChatModel) -> Runnable:
    return intent_classifier_prompt | llm.with_structured_output(QueryIntent)
