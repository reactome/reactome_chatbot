from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

SourceName = Literal["reactome", "userguide", "live"]

_REACTOME_SOURCE = """- **reactome**: Questions about biology, molecular mechanisms, pathways, reactions, proteins, genes,
  diseases, and other scientific content in the Reactome Knowledgebase.
  Examples: "What is apoptosis?", "Which pathways involve TP53?", "What does CDK5 do?"
"""

_USERGUIDE_SOURCE = """- **userguide**: Questions about how to use the Reactome **website**, tools, or interface.
  Examples: "How do I use the pathway browser?", "How do I search Reactome?",
  "How do I run gene list analysis?", "What is the Details Panel?"
"""

# Only offered when the live services are reachable. The distinction is not
# "biology vs not" -- it is a question about the DATABASE rather than about its
# contents, and the vector store cannot answer those at any retrieval quality.
# Asked which species Reactome includes, retrieval answered "primarily Homo
# sapiens ... no indications of other species". Reactome has 96. The documents
# it retrieved were all human, and it reported what it saw; a sample of the
# content cannot describe the scope of the database.
_LIVE_SOURCE = """- **live**: Questions about the Reactome database *itself* rather than about biology --
  what it covers, how big it is, which release this is, or whether some specific thing
  exists in it at all. These need a live lookup; they cannot be answered from stored
  documents.
  Examples: "What species does Reactome cover?", "Which release is this?",
  "How many pathways are there?", "Is there a pathway for ferroptosis?",
  "Does Reactome have anything on SARS-CoV-2?"
"""

_RULES = """Rules:
- If the user asks how to perform a task in Reactome or about UI features, choose **userguide**.
- If the user asks about biological facts or pathway content, choose **reactome**.
- When unsure, prefer **reactome** for science content and **userguide** for clear how-to or UI questions."""

_LIVE_RULE = """- If the user asks what the database *contains* or *covers*, rather than asking about the
  biology in it, choose **live**. "What does CDK5 do?" is **reactome**; "does Reactome have
  CDK5?" is **live**.
- Naming or listing curated entities is **reactome**, not **live**. "Which ABCA1 variants
  are there?" and "which diseases involve PTEN variants?" are answered from stored
  documents, which hold the variants themselves. The line is scope versus content:
  **live** answers how many, which species, which release, and whether a thing exists at
  all; **reactome** answers what is curated about a given gene, disease or pathway --
  including listing it. A question naming a specific gene or disease is almost always
  **reactome**."""

_SOURCE_BLOCKS: dict[SourceName, str] = {
    "reactome": _REACTOME_SOURCE,
    "userguide": _USERGUIDE_SOURCE,
    "live": _LIVE_SOURCE,
}


def build_classifier_message(sources: frozenset[SourceName]) -> str:
    """The prompt, describing only the destinations that actually exist.

    Offering a destination that is not wired up is worse than not having it:
    the model routes there, the source is missing, and the fallback answers as
    though it had been asked instead. Listing only what is available also means
    a deployment without the MCP gets the prompt it had before this existed,
    token for token.
    """
    blocks = [
        _SOURCE_BLOCKS[s] for s in ("reactome", "userguide", "live") if s in sources
    ]
    rules = _RULES + (f"\n{_LIVE_RULE}" if "live" in sources else "")
    return f"""
You route user questions for the React-to-Me assistant to the correct knowledge source.

Choose exactly one source:

{chr(10).join(blocks)}

{rules}
"""


intent_classifier_message = build_classifier_message(
    frozenset({"reactome", "userguide"})
)

intent_classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", intent_classifier_message),
        ("human", "User question:\n\n{rephrased_input}"),
    ]
)


class QueryIntent(BaseModel):
    source: SourceName = Field(
        description="The knowledge source that should answer this question."
    )


_FALLBACK_ORDER: tuple[SourceName, ...] = ("reactome", "userguide", "live")


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


# json_schema, not the default function_calling: the gpt-5.6 family refuses
# function tools on /v1/chat/completions ("Function tools with reasoning_effort
# are not supported ... use /v1/responses or set reasoning_effort to 'none'"),
# and langchain-openai 0.2.14 has no Responses API support. json_schema uses
# response_format instead, which every model here accepts -- verified against
# gpt-4o-mini and gpt-5.6-luna for all three graders.
def create_intent_classifier(
    llm: BaseChatModel, sources: frozenset[SourceName] | None = None
) -> Runnable:
    """Build the classifier for the destinations this deployment actually has.

    `sources=None` keeps the two-source prompt this had before `live` existed,
    so a deployment without the MCP classifies exactly as it did.
    """
    if sources is None or sources == frozenset({"reactome", "userguide"}):
        prompt = intent_classifier_prompt
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", build_classifier_message(sources)),
                ("human", "User question:\n\n{rephrased_input}"),
            ]
        )
    return prompt | llm.with_structured_output(QueryIntent, method="json_schema")
