import json
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda


def QueryExpansionParser(output: str) -> List[str]:
    """Parse JSON array output from LLM."""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("LLM output was not valid JSON. Output:\n" + output)


def create_query_expander(llm: BaseChatModel) -> Runnable:
    """
    Create a query expansion chain that generates 4 alternative queries.

    Args:
        llm: Language model to use

    Returns:
        Runnable that takes standalone_query and returns List[str]
    """
    system_prompt = """
You are a biomedical question expansion engine for information retrieval over the Reactome biological pathway database.

Given a single user question, generate **exactly 4** alternate standalone questions. These should be:

- Semantically related to the original question.
- Lexically diverse to improve retrieval via vector search and RAG-fusion.
- Biologically enriched with inferred or associated details.

Your goal is to improve recall of relevant documents by expanding the original query using:
- Synonymous gene/protein names (e.g., EGFR, ErbB1, HER1)
- Pathway or process-level context (e.g., signal transduction, apoptosis)
- Known diseases, phenotypes, or biological functions
- Cellular localization (e.g., nucleus, cytoplasm, membrane)
- Upstream/downstream molecular interactions

Rules:
- Each question must be **fully standalone** (no "this"/"it").
- Do not change the core intent—preserve the user's informational goal.
- Use appropriate biological terminology and Reactome-relevant concepts.
- Vary the **phrasing**, **focus**, or **biological angle** of each question.
- If the input is ambiguous, infer a biologically meaningful interpretation.

Output:
Return only a valid JSON array of 4 strings (no explanations, no metadata).
Do not include any explanations or metadata.
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "Original Question: {rephrased_input}")]
    )

    return prompt | llm | StrOutputParser() | RunnableLambda(QueryExpansionParser)