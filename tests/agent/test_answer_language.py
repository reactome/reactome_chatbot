"""The language reaches the answer prompt and nothing else.

The whole design rests on one property: `create_retrieval_chain` passes only
`input` to the retriever --

    retrieval_docs = (lambda x: x["input"]) | retriever

-- so a language instruction placed anywhere else cannot affect what is retrieved.
#140 placed it inside `input`, and measured through the whole retriever that changed
about half the fused documents. These tests pin the property that makes this
approach different.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langchain", reason="retrieval stack not installed")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from agent.tasks.language_instruction import LANGUAGE_INSTRUCTION  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent


class _RecordingRetriever(BaseRetriever):
    """Captures exactly what the chain asks it to retrieve."""

    seen: list[str]

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        self.seen.append(query)
        return [Document(page_content="TP53 induces apoptosis via BAX.")]


def _retrieval_query(**extra: Any) -> str:
    """Build the real chain over a recording retriever and return the query it saw."""
    from langchain_classic.chains.combine_documents import (
        create_stuff_documents_chain,
    )
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from retrievers.reactome.prompt import reactome_qa_prompt

    retriever = _RecordingRetriever(seen=[])
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=create_stuff_documents_chain(
            llm=FakeListChatModel(responses=["an answer"]), prompt=reactome_qa_prompt
        ),
    )
    chain.invoke({"input": "What role does TP53 play in apoptosis?", **extra})
    return retriever.seen[0]


def test_the_language_never_reaches_the_retriever() -> None:
    """FR-003. The property the whole approach depends on."""
    with_language = _retrieval_query(detected_language="French", chat_history=[])

    assert with_language == "What role does TP53 play in apoptosis?"
    assert "French" not in with_language
    assert "CRITICAL" not in with_language, "no instruction prose in the query"


def test_the_retrieval_query_is_identical_whatever_the_language() -> None:
    """FR-007: byte-identical, not merely similar.

    Stronger than a retrieval baseline and free: if the query cannot differ, the
    documents cannot differ, so there is nothing to measure statistically.
    """
    queries = {
        _retrieval_query(detected_language=lang, chat_history=[])
        for lang in ("English", "French", "Japanese", "German")
    }
    assert len(queries) == 1, f"the retrieval query varied by language: {queries}"


def test_the_instruction_lives_in_the_prompt_not_the_input() -> None:
    """FR-008. The difference between this and #140, asserted structurally."""
    for name in ("react_to_me", "plantreactome"):
        source = (REPO_ROOT / "src" / "agent" / "profiles" / f"{name}.py").read_text()
        assert '"detected_language": state["detected_language"],' in source, name
        assert (
            'state["rephrased_input"],' in source
        ), f"{name} must pass the rephrasing alone as input"


def test_the_instruction_protects_scientific_nomenclature() -> None:
    """#140's contribution, and the reason a naive translation is wrong.

    SET, MAX and CAT are gene symbols and ordinary English words; R-HSA-9612973
    means nothing translated.
    """
    text = LANGUAGE_INSTRUCTION.lower()
    for term in ("gene symbols", "protein names", "pathway names", "r-hsa", "url"):
        assert term in text, f"the instruction must protect {term}"
    assert "do not translate" in text


def test_both_profiles_share_one_instruction() -> None:
    """Two deployments, one wording -- they cannot drift apart.

    Matched on the instruction's distinctive phrase rather than on "R-HSA", which
    appears legitimately in both prompts as a citation example. A cruder check
    passed on nothing and failed on that.
    """
    for name in ("reactome", "plantreactome"):
        source = (REPO_ROOT / "src" / "retrievers" / name / "prompt.py").read_text()
        assert "LANGUAGE_INSTRUCTION" in source, name
        assert (
            "Do not translate them" not in source
        ), f"{name} inlines its own copy of the instruction instead of sharing it"
