"""The async retrieval path must return what the sync one does.

`HybridRetriever` implements retrieval twice: `retrieve_documents` and
`aretrieve_documents`, the second gathering coroutines so the collections are
queried concurrently. They are separate implementations of the same fusion, and
nothing exercised the async one at all.

That asymmetry matters beyond the usual duplication argument: the application
serves through the async path, while `bin/retrieval_baseline` -- the tool the
constitution names for measuring retrieval changes -- drives the sync one. If
they drift, every measurement is of a path no user takes, and Principle II's
before-and-after comparison measures the wrong thing.

Uses DeterministicFakeEmbedding rather than FakeEmbeddings: the latter returns a
fresh random vector per call, so the same query embeds differently on the second
run and the two paths appear to disagree when they do not. That false result is
what prompted this test.

What this does NOT cover, measured rather than assumed: the per-collection cap.
Deterministic fake vectors carry no relation to the text, so every query retrieves
much the same set, fusion lands on exactly `max_documents_per_collection` however
many queries are given, and changing the cap on one path alone stays invisible.
Tried at 12 and 60 documents per collection and with one, three and seven
queries; fusion produced ten per collection every time. Covering the cap needs
real embeddings and therefore an installed bundle, which is what
`bin/retrieval_baseline` is for.
"""

import asyncio
import csv
from pathlib import Path

import pytest

pytest.importorskip("langchain_chroma")

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.callbacks import (  # noqa: E402
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeListChatModel,
)

from retrievers.csv_chroma import (  # noqa: E402
    HybridRetriever,
    chroma_settings,
    selected_collections,
)


# Large enough that the per-collection cap actually binds. At twelve it did
# not: fusion produced fewer documents than the cap, so a test that changed
# the cap on one path only still passed.
def _require_bm25_tokenizer() -> None:
    """BM25 tokenises with nltk's word_tokenize, which needs punkt_tab.

    CI installs it, matching the Dockerfile. A developer who has not downloaded
    it should get a skip saying so rather than a LookupError from inside nltk.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        pytest.skip(
            "nltk punkt_tab not downloaded: python -m nltk.downloader punkt_tab"
        )


COLLECTIONS = {"alpha": 60, "beta": 60}

# Varied on purpose. With near-identical text BM25 and vector search return the
# same ten documents, fusion never exceeds the per-collection cap, and a test
# that changes the cap on one path only still passes -- which this one did.
WORDS = [
    "kinase phosphorylation cascade",
    "cholesterol transport vesicle",
    "ubiquitin ligase complex",
    "mitochondrial respiratory chain",
    "DNA mismatch repair",
    "interferon signalling",
    "collagen assembly",
]


def _bundle(tmp_path: Path, embedding: DeterministicFakeEmbedding) -> Path:
    csv_dir = tmp_path / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for collection, count in COLLECTIONS.items():
        with open(csv_dir / f"{collection}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["st_id", "display_name", "text"]
            )
            writer.writeheader()
            for i in range(count):
                writer.writerow(
                    {
                        "st_id": f"{collection}-{i}",
                        "display_name": f"{collection} item {i}",
                        "text": WORDS[i % len(WORDS)] + f" {collection} {i}",
                    }
                )
        Chroma.from_documents(
            documents=[
                Document(
                    page_content=(
                        f"st_id: {collection}-{i}\n"
                        f"text: {WORDS[i % len(WORDS)]} {collection} {i}"
                    ),
                    metadata={"st_id": f"{collection}-{i}"},
                )
                for i in range(count)
            ],
            embedding=embedding,
            persist_directory=str(tmp_path / collection),
            client_settings=chroma_settings(),
        )
    return tmp_path


@pytest.mark.requires_retrieval_stack
@pytest.mark.parametrize(
    "queries",
    [
        ["kinase phosphorylation"],
        ["nothing matches this at all"],
        # Enough distinct queries that fusion overflows the per-collection cap,
        # so a change to the cap on one path only is visible. With one query the
        # two retrievers return largely the same documents, fusion stays under
        # the cap, and such a change passes unnoticed -- as it did here.
        [
            "kinase phosphorylation cascade",
            "cholesterol transport vesicle",
            "ubiquitin ligase complex",
            "mitochondrial respiratory chain",
            "DNA mismatch repair",
            "interferon signalling",
            "collagen assembly",
        ],
    ],
)
def test_async_returns_exactly_what_sync_returns(
    tmp_path: Path, queries: list[str]
) -> None:
    _require_bm25_tokenizer()
    embedding = DeterministicFakeEmbedding(size=16)
    retriever = HybridRetriever.from_subdirectory(
        # Never called: these tests drive retrieve_documents directly, below the
        # query-expansion step. It is here because the constructor requires one.
        llm=FakeListChatModel(responses=[""]),
        embedding=embedding,
        embeddings_directory=_bundle(tmp_path, embedding),
    )

    sync = retriever.retrieve_documents(
        queries, CallbackManagerForRetrieverRun.get_noop_manager()
    )
    asynchronous = asyncio.run(
        retriever.aretrieve_documents(
            queries, AsyncCallbackManagerForRetrieverRun.get_noop_manager()
        )
    )

    # Order, not just membership: RRF resolves ties by first appearance, so a
    # reordering is a ranking change and the top documents are the ones that
    # reach the model.
    assert [d.page_content for d in asynchronous] == [d.page_content for d in sync]


def test_both_paths_honour_the_same_collection_selection(tmp_path: Path) -> None:
    """T017: a filter applied to one path only is the bug this pins.

    The application serves through the async path and `bin/retrieval_baseline`
    drives the sync one, so a selection honoured by only one means the measured
    path and the served path search different collections -- and neither the
    measurement nor the answer would look wrong.
    """
    _require_bm25_tokenizer()
    embedding = DeterministicFakeEmbedding(size=16)
    retriever = HybridRetriever.from_subdirectory(
        llm=FakeListChatModel(responses=[""]),
        embedding=embedding,
        embeddings_directory=_bundle(tmp_path, embedding),
    )
    queries = ["apoptosis signalling"]

    token = selected_collections.set(["alpha"])
    try:
        sync = retriever.retrieve_documents(
            queries, CallbackManagerForRetrieverRun.get_noop_manager()
        )
        asynchronous = asyncio.run(
            retriever.aretrieve_documents(
                queries, AsyncCallbackManagerForRetrieverRun.get_noop_manager()
            )
        )
    finally:
        selected_collections.reset(token)

    assert [d.page_content for d in asynchronous] == [d.page_content for d in sync]
    # And the selection actually bit: `beta` documents must be absent from both.
    for documents in (sync, asynchronous):
        assert documents, "the selection removed everything, so this proves nothing"
        assert all("beta" not in d.page_content for d in documents)


REAL_NAMES = ("complexes", "disease_variants", "ewas", "reactions", "summations")


def _named_bundle(
    tmp_path: Path, embedding: DeterministicFakeEmbedding, names: tuple[str, ...]
) -> Path:
    global COLLECTIONS
    previous = COLLECTIONS
    COLLECTIONS = {name: 20 for name in names}
    try:
        return _bundle(tmp_path, embedding)
    finally:
        COLLECTIONS = previous


@pytest.mark.requires_retrieval_stack
@pytest.mark.parametrize("wanted", REAL_NAMES)
def test_every_collection_can_be_reached_and_only_it(
    tmp_path: Path, wanted: str
) -> None:
    """T005a and T007: assert at retrieval level that a collection was searched.

    Neither `complexes` nor `reactions` can be guarded by asking a question.
    Measured 2026-09-19 (specs/009-collection-routing/research.md): their
    content is duplicated in `summations` prose and in the input/output names
    carried by `reactions`, so every candidate answered just as well with the
    collection removed. Two failed hunts and a structural explanation.

    So the guard lives here instead. A collection that routing can never reach
    -- a name mismatch, a lookup that silently yields nothing -- is invisible
    to `answer-sweep` by construction, and this is what would catch it.
    """
    _require_bm25_tokenizer()
    embedding = DeterministicFakeEmbedding(size=16)
    retriever = HybridRetriever.from_subdirectory(
        llm=FakeListChatModel(responses=[""]),
        embedding=embedding,
        embeddings_directory=_named_bundle(tmp_path, embedding, REAL_NAMES),
    )

    token = selected_collections.set([wanted])
    try:
        documents = retriever.retrieve_documents(
            ["kinase phosphorylation"],
            CallbackManagerForRetrieverRun.get_noop_manager(),
        )
    finally:
        selected_collections.reset(token)

    assert documents, f"{wanted} is selectable but returned nothing"
    for other in REAL_NAMES:
        if other == wanted:
            continue
        assert not any(
            other in d.page_content for d in documents
        ), f"selecting {wanted} also searched {other}"
