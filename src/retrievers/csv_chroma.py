import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypedDict

import chromadb.config
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from nltk.tokenize import word_tokenize
from pydantic import ConfigDict

chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a biomedical question expansion engine for information retrieval over the Reactome biological pathway database.

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
Return only the 4 alternative questions separated by newlines.
Do not include any explanations or metadata.

Original Question: {question}""",
)


RESULTS_PER_RETRIEVER = 10
# The vector store is asked for more than we intend to keep, because one
# Reactome entity can occupy several rows -- a reaction appears once per
# pathway/input/output/catalyst combination -- and those rows have distinct
# page_content, so nothing upstream collapses them. Without over-fetching, a
# request for 10 returns about 5 distinct reactions. See issue #169.
#
# 4x, not 3x: measured over the 20 questions in tests/golden/questions.txt, 3x
# still came up short on `complexes` for one of them (8 distinct of 10). 4x
# clears all 20 on every collection and 6x gains nothing further.
VECTOR_OVERFETCH = 4

# How many fused documents each collection contributes to the answer prompt.
#
# weighted_reciprocal_rank returns *every* unique document across the lists it is
# given, not a top-N, so without this the retriever ranked ~222 documents by
# relevance and then sent all of them -- roughly 32k tokens, a quarter of
# gpt-4o-mini's window, on every message -- which made the ranking decorative.
#
# The cap is per collection rather than global on purpose: reactions, summations,
# complexes and ewas hold different kinds of information, and one global top-N
# would let a single collection crowd the others out. Per collection guarantees
# each one contributes.
#
# This value is a starting point, not a tuned one. It matches what a single
# retriever returns. Changing it trades recall against the model's difficulty
# attending to the middle of a long context; the right number should come from an
# answer-quality evaluation rather than from taste.
MAX_DOCUMENTS_PER_COLLECTION = RESULTS_PER_RETRIEVER


def dedupe_by_entity(docs: list[Document], limit: int) -> list[Document]:
    """Keep the highest-ranked row per Reactome stable ID, up to `limit`.

    Falls back to page_content for documents with no st_id, so a collection
    without that metadata degrades to the previous behaviour rather than raising.
    """
    seen: set[str] = set()
    kept: list[Document] = []
    for doc in docs:
        key = str(doc.metadata.get("st_id") or doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        kept.append(doc)
        if len(kept) == limit:
            break
    return kept


RRF_K = 60
"""Reciprocal Rank Fusion constant, from Cormack et al. (SIGIR 2009).

Copied from the value LangChain's EnsembleRetriever uses, so that vendoring the
maths here changes nothing. It damps the difference between adjacent ranks: at
k=60 a first-place hit scores 1/61 and an eleventh-place hit 1/71.
"""


def reciprocal_rank_fusion(
    doc_lists: list[list[Document]], weights: list[float] | None = None
) -> list[Document]:
    """Fuse ranked lists by Reciprocal Rank Fusion.

    Vendored rather than borrowed. Reaching LangChain's implementation required
    constructing `EnsembleRetriever(retrievers=[])` -- an ensemble with no
    retrievers -- purely to call one method, on every fusion. Ranking is the
    product's core retrieval quality, so a library upgrade should not be able to
    reorder results without anyone noticing.

    Behaviour is identical to `EnsembleRetriever.weighted_reciprocal_rank`, which
    the tests in tests/retrievers/ pin:

    - a document's score is the sum over lists of `weight / (rank + RRF_K)`,
      with rank counted from 1
    - documents are identified by `page_content`, so the same text found by two
      retrievers accumulates both scores
    - ties keep the order of first appearance across the concatenated lists,
      because the sort is stable
    """
    if weights is None:
        weights = [1 / len(doc_lists)] * len(doc_lists)
    if len(doc_lists) != len(weights):
        raise ValueError(
            f"Got {len(doc_lists)} document lists and {len(weights)} weights; "
            "they must correspond one to one."
        )

    scores: dict[str, float] = {}
    for docs, weight in zip(doc_lists, weights, strict=True):
        for rank, doc in enumerate(docs, start=1):
            scores[doc.page_content] = scores.get(doc.page_content, 0.0) + weight / (
                rank + RRF_K
            )

    seen: set[str] = set()
    unique: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
    return sorted(unique, key=lambda d: scores[d.page_content], reverse=True)


def unique_documents(documents: list[Document]) -> list[Document]:
    """Drop repeats, keeping first appearance.

    Matches MultiQueryRetriever's final `unique_union`, which compares whole
    Document objects rather than just their text.
    """
    seen: list[Document] = []
    for doc in documents:
        if doc not in seen:
            seen.append(doc)
    return seen


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Split an LLM response into non-empty lines.

    Same behaviour as LangChain's parser of the same name, which the query
    expansion prompt was written against.
    """

    def parse(self, text: str) -> list[str]:
        return [line for line in text.strip().split("\n") if line]


def list_chroma_subdirectories(directory: Path) -> list[str]:
    return [
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    ]


def create_bm25_chroma_ensemble_retriever(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    descriptions_info: dict[str, str],
    field_info: dict[str, list[AttributeInfo]],
) -> "HybridRetriever":
    return HybridRetriever.from_subdirectory(
        llm,
        embedding,
        embeddings_directory,
        descriptions_info=descriptions_info,
        field_info=field_info,
        include_original=True,
    )


class RetrieverDict(TypedDict):
    bm25: BM25Retriever
    vector: SelfQueryRetriever


class HybridRetriever(BaseRetriever):
    """BM25 and vector search over each Chroma collection, fused by RRF.

    A plain BaseRetriever. It previously subclassed MultiQueryRetriever and
    reached into LangChain internals in five places -- overriding a required
    field to None through SkipJsonSchema, building a throwaway
    MultiQueryRetriever to steal its llm_chain, assigning a private attribute
    outside pydantic, instantiating an empty EnsembleRetriever to borrow one
    method, and overriding two internal methods. None of those are public API,
    which is what blocked the LangChain upgrade.

    Query expansion is now done here rather than inherited: one LLM call turns
    the question into alternates, and `include_original` appends the original
    LAST, matching the order MultiQueryRetriever used. Order matters, because
    RRF resolves ties by first appearance.
    """

    query_expander: Runnable[dict[str, str], list[str]]
    include_original: bool = False
    collection_retrievers: dict[str, RetrieverDict]

    # BM25Retriever and SelfQueryRetriever are not pydantic models.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_subdirectory(
        cls,
        llm: BaseChatModel,
        embedding: Embeddings,
        embeddings_directory: Path,
        *,
        descriptions_info: dict[str, str],
        field_info: dict[str, list[AttributeInfo]],
        include_original: bool = False,
    ) -> "HybridRetriever":
        _retrievers: dict[str, RetrieverDict] = {}
        for subdirectory in list_chroma_subdirectories(embeddings_directory):
            # set up BM25 retriever
            csv_file_name = subdirectory + ".csv"
            reactome_csvs_dir: Path = embeddings_directory / "csv_files"
            loader = CSVLoader(file_path=reactome_csvs_dir / csv_file_name)
            data = loader.load()
            bm25_retriever = BM25Retriever.from_documents(
                data,
                preprocess_func=lambda text: word_tokenize(
                    text.casefold(), language="english"
                ),
            )
            bm25_retriever.k = RESULTS_PER_RETRIEVER

            # set up vectorstore SelfQuery retriever
            vectordb = Chroma(
                persist_directory=str(embeddings_directory / subdirectory),
                embedding_function=embedding,
                client_settings=chroma_settings,
            )

            selfq_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vectordb,
                document_contents=descriptions_info[subdirectory],
                metadata_field_info=field_info[subdirectory],
                search_kwargs={"k": RESULTS_PER_RETRIEVER * VECTOR_OVERFETCH},
            )

            _retrievers[subdirectory] = {
                "bm25": bm25_retriever,
                "vector": selfq_retriever,
            }
        # The expansion chain, built directly. This used to be extracted from a
        # throwaway MultiQueryRetriever constructed only to reach its .llm_chain.
        return cls(
            query_expander=multi_query_prompt | llm | LineListOutputParser(),
            include_original=include_original,
            collection_retrievers=_retrievers,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Expand the question, retrieve for every variant, fuse, de-duplicate.

        Reproduces what MultiQueryRetriever._get_relevant_documents did, so this
        stage changes structure without changing results. Note the original query
        is appended AFTER the generated ones: RRF breaks ties by first
        appearance, so reordering here would silently change the ranking.
        """
        queries = self.query_expander.invoke(
            {"question": query}, config={"callbacks": run_manager.get_child()}
        )
        if self.include_original:
            queries.append(query)
        return unique_documents(self.retrieve_documents(queries, run_manager))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Async twin of the above; must agree with it document for document."""
        queries = await self.query_expander.ainvoke(
            {"question": query}, config={"callbacks": run_manager.get_child()}
        )
        if self.include_original:
            queries.append(query)
        return unique_documents(await self.aretrieve_documents(queries, run_manager))

    def weighted_reciprocal_rank(
        self, doc_lists: list[list[Document]]
    ) -> list[Document]:
        """Kept as a method so the existing tests address it unchanged."""
        return reciprocal_rank_fusion(doc_lists)

    def retrieve_documents(
        self, queries: list[str], run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        subdirectory_docs: list[Document] = []
        for subdirectory, retrievers in self.collection_retrievers.items():
            bm25_retriever = retrievers["bm25"]
            vector_retriever = retrievers["vector"]
            doc_lists: list[list[Document]] = []
            for i, query in enumerate(queries):
                bm25_docs = bm25_retriever.invoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-bm25-{i}"
                        )
                    },
                )
                vector_docs = vector_retriever.invoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-vector-{i}"
                        )
                    },
                )
                # Separate lists, not `bm25_docs + vector_docs`. RRF scores by
                # position, so concatenating put every vector result at rank 11+
                # and scored the best of them 1/71 against BM25's 1/61 -- and it
                # meant the two retrievers were never fused against each other,
                # only across query variants. See issue #170.
                doc_lists.append(dedupe_by_entity(bm25_docs, RESULTS_PER_RETRIEVER))
                doc_lists.append(dedupe_by_entity(vector_docs, RESULTS_PER_RETRIEVER))
            subdirectory_docs.extend(
                self.weighted_reciprocal_rank(doc_lists)[:MAX_DOCUMENTS_PER_COLLECTION]
            )
        return subdirectory_docs

    async def aretrieve_documents(
        self,
        queries: list[str],
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        subdirectory_results: dict[str, list[Coroutine[Any, Any, list[Document]]]] = {}
        for subdirectory, retrievers in self.collection_retrievers.items():
            bm25_retriever = retrievers["bm25"]
            vector_retriever = retrievers["vector"]
            subdirectory_results[subdirectory] = []
            for i, query in enumerate(queries):
                bm25_results = asyncio.to_thread(
                    bm25_retriever.invoke,
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-bm25-{i}"
                        )
                    },
                )
                vector_results = asyncio.to_thread(
                    vector_retriever.invoke,
                    query,
                    config={
                        "callbacks": run_manager.get_child(
                            tag=f"{subdirectory}-vector-{i}"
                        )
                    },
                )
                subdirectory_results[subdirectory].extend(
                    (bm25_results, vector_results)
                )
        subdirectory_docs: list[Document] = []
        for subdir_results in subdirectory_results.values():
            # Separate lists, de-duplicated per entity, matching the synchronous
            # path above. See issues #169 and #170.
            doc_lists: list[list[Document]] = [
                dedupe_by_entity(docs, RESULTS_PER_RETRIEVER)
                for docs in await asyncio.gather(*subdir_results)
            ]
            subdirectory_docs.extend(
                self.weighted_reciprocal_rank(doc_lists)[:MAX_DOCUMENTS_PER_COLLECTION]
            )
        return subdirectory_docs
