import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb.config
import pandas as pd
from langchain_chroma.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from retrievers.retrieval_utils import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

CHROMA_SETTINGS = chromadb.config.Settings(anonymized_telemetry=False)
DEFAULT_RETRIEVAL_K = 20
RRF_FINAL_K = 10
RRF_LAMBDA_MULTIPLIER = 60.0
EXCLUDED_CONTENT_COLUMNS = {"st_id"}


def create_documents_from_csv(csv_path: Path) -> List[Document]:
    """Create Document objects from CSV file with proper metadata extraction."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")

    documents = []

    for index, row in df.iterrows():
        content_parts = []
        for column in df.columns:
            if column not in EXCLUDED_CONTENT_COLUMNS:
                value = str(row[column]) if pd.notna(row[column]) else ""
                if value and value != "nan":
                    content_parts.append(f"{column}: {value}")

        page_content = "\n".join(content_parts)

        metadata = {
            str(column): str(value)
            for column in df.columns
            for value in [row[column]]
            if pd.notna(value) and str(value) != "nan"
        }
        metadata.update({"source": str(csv_path), "row_index": index})

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def list_chroma_subdirectories(directory: Path) -> List[str]:
    """Discover all subdirectories containing ChromaDB files."""
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    subdirectories = [
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    ]

    if not subdirectories:
        logger.warning(f"No ChromaDB subdirectories found in {directory}")

    return subdirectories


class HybridRetriever:
    """Advanced hybrid retriever supporting RRF, parallel processing, and multi-source search."""

    def __init__(self, embedding: Embeddings, embeddings_directory: Path):

        self.embedding = embedding
        self.embeddings_directory = embeddings_directory
        self._retrievers: Dict[
            str, Dict[str, Optional[Union[BM25Retriever, object]]]
        ] = {}

        try:
            self._initialize_retrievers()
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            raise RuntimeError(f"Hybrid retriever initialization failed: {e}") from e

    def _initialize_retrievers(self) -> None:
        """Initialize BM25 and vector retrievers for all discovered subdirectories."""
        subdirectories = list_chroma_subdirectories(self.embeddings_directory)

        if not subdirectories:
            raise ValueError(f"No subdirectories found in {self.embeddings_directory}")

        for subdirectory in subdirectories:
            bm25_retriever = self._create_bm25_retriever(subdirectory)
            vector_retriever = self._create_vector_retriever(subdirectory)

            self._retrievers[subdirectory] = {
                "bm25": bm25_retriever,
                "vector": vector_retriever,
            }

        logger.info(f"Initialized retrievers for {len(subdirectories)} subdirectories")

    def _create_bm25_retriever(self, subdirectory: str) -> Optional[BM25Retriever]:
        """Create BM25 retriever for a specific subdirectory."""
        csv_path = self.embeddings_directory / "csv_files" / f"{subdirectory}.csv"

        if not csv_path.exists():
            logger.warning(f"CSV file not found for {subdirectory}: {csv_path}")
            return None

        try:
            documents = create_documents_from_csv(csv_path)
            retriever = BM25Retriever.from_documents(documents)
            retriever.k = DEFAULT_RETRIEVAL_K
            logger.debug(
                f"Created BM25 retriever for {subdirectory} with {len(documents)} documents"
            )
            return retriever
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever for {subdirectory}: {e}")
            return None

    def _create_vector_retriever(self, subdirectory: str) -> Optional[object]:
        """Create vector retriever for a specific subdirectory."""
        vector_directory = self.embeddings_directory / subdirectory

        if not vector_directory.exists():
            logger.warning(
                f"Vector directory not found for {subdirectory}: {vector_directory}"
            )
            return None

        try:
            vector_store = Chroma(
                persist_directory=str(vector_directory),
                embedding_function=self.embedding,
                client_settings=CHROMA_SETTINGS,
            )
            retriever = vector_store.as_retriever(
                search_kwargs={"k": DEFAULT_RETRIEVAL_K}
            )
            logger.debug(f"Created vector retriever for {subdirectory}")
            return retriever
        except Exception as e:
            logger.error(f"Failed to create vector retriever for {subdirectory}: {e}")
            return None

    async def _search_with_bm25(
        self, query: str, retriever: BM25Retriever
    ) -> List[Document]:
        """Search using BM25 retriever asynchronously."""
        return await asyncio.to_thread(retriever.get_relevant_documents, query)

    async def _search_with_vector(self, query: str, retriever: Any) -> List[Document]:
        """Search using vector retriever asynchronously."""
        return await asyncio.to_thread(retriever.get_relevant_documents, query)

    async def _execute_hybrid_search(
        self, query: str, subdirectory: str
    ) -> List[Document]:
        """Execute hybrid search (BM25 + vector) for a single query on a subdirectory."""
        retriever_info = self._retrievers.get(subdirectory)
        if not retriever_info:
            logger.warning(f"No retrievers found for subdirectory: {subdirectory}")
            return []

        search_tasks = []

        if retriever_info["bm25"] and isinstance(retriever_info["bm25"], BM25Retriever):
            search_tasks.append(self._search_with_bm25(query, retriever_info["bm25"]))

        if retriever_info["vector"]:
            search_tasks.append(
                self._search_with_vector(query, retriever_info["vector"])
            )

        if not search_tasks:
            logger.warning(f"No active retrievers for subdirectory: {subdirectory}")
            return []

        try:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            combined_documents = []
            for result in search_results:
                if isinstance(result, list):
                    combined_documents.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Search error in {subdirectory}: {result}")

            return combined_documents
        except Exception as e:
            logger.error(f"Failed to execute hybrid search for {subdirectory}: {e}")
            return []

    def _generate_document_identifier(self, document: Document) -> str:
        """Generate unique identifier for a document."""
        for field in ["url", "id", "st_id"]:
            if document.metadata.get(field):
                return document.metadata[field]

        return hashlib.md5(document.page_content.encode()).hexdigest()

    async def _apply_reciprocal_rank_fusion(
        self, queries: List[str], subdirectory: str
    ) -> List[Document]:
        """Apply Reciprocal Rank Fusion to results from multiple queries on a subdirectory."""
        logger.info(
            f"Executing hybrid search for {len(queries)} queries in {subdirectory}"
        )

        search_tasks = [
            self._execute_hybrid_search(query, subdirectory) for query in queries
        ]
        all_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        valid_result_sets = []
        for i, result in enumerate(all_search_results):
            if isinstance(result, list):
                valid_result_sets.append(result)
                logger.debug(f"Query {i+1}: {len(result)} results")
            elif isinstance(result, Exception):
                logger.error(f"Query {i+1} failed: {result}")

        if not valid_result_sets:
            logger.warning(f"No valid results for {subdirectory}")
            return []

        logger.info(
            f"Applying RRF to {len(valid_result_sets)} result sets in {subdirectory}"
        )

        top_documents, _, rrf_scores = reciprocal_rank_fusion(
            ranked_lists=valid_result_sets,
            final_k=RRF_FINAL_K,
            lambda_mult=RRF_LAMBDA_MULTIPLIER,
            rrf_k=None,
            id_getter=self._generate_document_identifier,
        )

        logger.info(f"RRF completed for {subdirectory}: {len(top_documents)} documents")
        if rrf_scores:
            top_scores = dict(list(rrf_scores.items())[:3])
            logger.debug(f"Top RRF scores: {top_scores}")

        return top_documents

    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """Main retrieval method supporting RRF and parallel processing."""
        original_query = inputs.get("input", "").strip()
        if not original_query:
            raise ValueError("Input query cannot be empty")

        expanded_queries = inputs.get("expanded_queries", [])
        all_queries = [original_query] + (expanded_queries or [])

        logger.info(
            f"Processing {len(all_queries)} queries across {len(self._retrievers)} subdirectories"
        )
        for i, query in enumerate(all_queries, 1):
            logger.debug(f"Query {i}: {query}")

        rrf_tasks = [
            self._apply_reciprocal_rank_fusion(all_queries, subdirectory)
            for subdirectory in self._retrievers.keys()
        ]

        subdirectory_results = await asyncio.gather(*rrf_tasks, return_exceptions=True)

        context_parts = []

        for i, subdirectory in enumerate(self._retrievers.keys()):
            result = subdirectory_results[i]

            if isinstance(result, Exception):
                logger.error(f"Subdirectory {subdirectory} failed: {result}")
                continue

            if isinstance(result, list) and result:
                for document in result:
                    context_parts.append(document.page_content)
                    context_parts.append("")

        final_context = "\n".join(context_parts)

        total_documents = sum(
            len(result) if isinstance(result, list) else 0
            for result in subdirectory_results
        )
        logger.info(f"Retrieved {total_documents} documents total")

        for i, subdirectory in enumerate(self._retrievers.keys()):
            result = subdirectory_results[i]
            if isinstance(result, list):
                logger.info(f"{subdirectory}: {len(result)} documents")
            else:
                logger.warning(f"{subdirectory}: Failed")

        logger.debug(f"Final context length: {len(final_context)} characters")

        return final_context


def create_hybrid_retriever(
    embedding: Embeddings, embeddings_directory: Path
) -> HybridRetriever:
    """Create a hybrid retriever with RRF and parallel processing support."""
    try:
        return HybridRetriever(
            embedding=embedding, embeddings_directory=embeddings_directory
        )
    except Exception as e:
        logger.error(f"Failed to create hybrid retriever: {e}")
        raise RuntimeError(f"Hybrid retriever creation failed: {e}") from e
