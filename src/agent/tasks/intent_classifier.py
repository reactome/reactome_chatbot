from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from retrievers.reactome.metadata_info import reactome_descriptions_info

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


def _collections_block() -> str:
    """The selectable collections, described by the bundle's own metadata.

    Sourced from `reactome_descriptions_info` rather than a literal list so a
    collection added to the bundle cannot be missing from the prompt that is
    supposed to offer it -- the failure mode would be silent, since an
    unmentioned collection is simply never chosen.
    """
    lines = [
        f"  - **{name}**: {description.strip()}"
        for name, description in reactome_descriptions_info.items()
    ]
    return chr(10).join(lines)


# Narrowing is measured, not assumed. On 2026-09-19, forcing every tracked
# question to `reactions` + `summations` took the sweep from 13/13 to 10/13,
# and the questions that failed were the ones whose answers live only in the
# collections that were dropped: the UniProt accession needs `ewas`, PTEN's
# variants need `disease_variants`. A wrong narrow does not degrade an answer,
# it removes it -- so the instruction below leans hard on leaving the list
# empty, because widening costs latency and narrowing wrongly costs the answer.
_COLLECTIONS_RULE = """
Collections (only when source is **reactome**):

The Reactome content is split into collections holding different kinds of record:

{collections}

Set `collections` ONLY when the question plainly needs one or two specific kinds
of record. Leave it EMPTY otherwise, and an empty list searches all of them.
An empty list is the right answer for most questions.

- Naming or listing variants of a gene, or which disease a variant causes: `disease_variants`.
- A UniProt accession, a gene synonym, or which protein a gene maps to: `ewas`.
- What a pathway or reaction's curated description says: `summations`.
- What a complex is made of: `complexes`.
- The inputs, outputs or catalyst of a reaction: `reactions`.

Two things to respect:
- Include `summations` alongside any other choice unless the question is purely
  about identifiers. The curated prose supports most biological answers.
- If the question is broad, mechanistic, or you are at all unsure, leave
  `collections` empty. Searching everything is slower; searching the wrong
  subset means the answer is not there at all.
"""


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
    collections = (
        _COLLECTIONS_RULE.format(collections=_collections_block())
        if "reactome" in sources
        else ""
    )
    return f"""
You route user questions for the React-to-Me assistant to the correct knowledge source.

Choose exactly one source:

{chr(10).join(blocks)}

{rules}
{collections}"""


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
    collections: list[str] = Field(
        default_factory=list,
        description=(
            "Which Reactome collections to search, when the question plainly "
            "needs only some of them. Empty means search all of them, which is "
            "the right answer for most questions. Ignored unless source is "
            "'reactome'."
        ),
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
