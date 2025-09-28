from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb.config
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrievers.csv_chroma import list_chroma_subdirectories
from src.util.embedding_environment import EmbeddingEnvironment

from .retrieval_utils import UniProtRetrievalConfig, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

CHROMA_SETTINGS = chromadb.config.Settings(anonymized_telemetry=False)


class UniProtRetriever:
    """
    UniProt vector retriever that supports RRF and similarity search.

    This retriever provides the same configuration options as the Graph RAG
    retriever but operates only on vector embeddings without graph traversal.
    Returns page content strings (protein information only, no metadata).

    Features:
    - Single vectorstore using the most recent subdirectory
    - Single BM25 retriever using the most recent CSV file
    - Reciprocal Rank Fusion (RRF) support
    - Hybrid vector + BM25 search
    - Clean page content output for LLM consumption
    """

    DEFAULT_BM25_K = 10

    def __init__(
        self,
        embedding: Embeddings,
        embeddings_directory: Optional[Path] = None,
    ) -> None:
        """
        Initialize the UniProt vector retriever.

        Args:
            embedding: Embedding model for vector operations
            embeddings_directory: Path to UniProt embeddings directory.
                                 Defaults to EmbeddingEnvironment.get_dir("uniprot")

        Raises:
            ValueError: If embeddings_directory doesn't exist
            RuntimeError: If initialization fails
        """
        self.embedding = embedding
        self.embeddings_directory = (
            embeddings_directory or EmbeddingEnvironment.get_dir("uniprot")
        )

        if not self.embeddings_directory.exists():
            raise ValueError(
                f"Embeddings directory does not exist: {self.embeddings_directory}"
            )

        self._vectorstore: Optional[Chroma] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._subdirectory: Optional[str] = None

        try:
            self._initialize_retrievers()
        except Exception as e:
            logger.error(f"Failed to initialize UniProt retriever: {e}")
            raise RuntimeError(f"UniProt retriever initialization failed: {e}") from e

    def _initialize_retrievers(self) -> None:
        """Initialize both vectorstore and BM25 retriever."""
        subdirectories = list_chroma_subdirectories(self.embeddings_directory)

        if not subdirectories:
            raise RuntimeError("No UniProt subdirectories found")

        # Use the most recently created subdirectory
        self._subdirectory = self._get_latest_subdirectory(subdirectories)

        self._initialize_vectorstore()
        self._initialize_bm25_retriever()

        logger.info(
            f"UniProt retriever initialized successfully using subdirectory: {self._subdirectory}"
        )

    def _initialize_vectorstore(self) -> None:
        """Initialize Chroma vectorstore for UniProt."""
        try:
            self._vectorstore = Chroma(
                persist_directory=str(self.embeddings_directory / self._subdirectory),
                embedding_function=self.embedding,
                client_settings=CHROMA_SETTINGS,
            )
            logger.info(f"Initialized UniProt vectorstore: {self._subdirectory}")
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}")
            raise

    def _initialize_bm25_retriever(self) -> None:
        """Initialize BM25 retriever for UniProt."""
        try:
            csv_file_name = f"{self._subdirectory}.csv"
            csvs_dir = self.embeddings_directory / "csv_files"
            csv_path = csvs_dir / csv_file_name

            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                return

            loader = CSVLoader(file_path=str(csv_path))
            data = loader.load()

            if not data:
                logger.warning(f"No data loaded from CSV: {csv_path}")
                return

            self._bm25_retriever = BM25Retriever.from_documents(data)
            self._bm25_retriever.k = self.DEFAULT_BM25_K

            logger.info(f"Initialized UniProt BM25 retriever: {self._subdirectory}")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 retriever: {e}")

    def _get_latest_subdirectory(self, subdirectories: List[str]) -> str:
        """
        Get the most recently created subdirectory.

        Args:
            subdirectories: List of subdirectory names

        Returns:
            Name of the most recently created subdirectory
        """
        subdir_times = []

        for subdir in subdirectories:
            subdir_path = self.embeddings_directory / subdir
            if subdir_path.exists():
                mtime = os.path.getmtime(subdir_path)
                subdir_times.append((subdir, mtime))

        if not subdir_times:
            logger.warning("No valid subdirectories found, using first available")
            return subdirectories[0]

        # Sort by modification time (most recent first) and return the latest
        subdir_times.sort(key=lambda x: x[1], reverse=True)
        latest_subdir = subdir_times[0][0]

        logger.info(f"Using most recent UniProt subdirectory: {latest_subdir}")
        return latest_subdir

    async def ainvoke(
        self,
        query: str,
        cfg: UniProtRetrievalConfig,
        expanded_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Invoke the UniProt retrieval pipeline.

        Args:
            query: Search query
            cfg: Retrieval configuration
            expanded_queries: Optional list of expanded queries for RRF

        Returns:
            List of page content strings (protein information only, no metadata)
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            logger.info(
                f"UniProt retrieve called with query='{query}', expanded_queries={expanded_queries}, use_rrf={cfg.vector_config.use_rrf}"
            )

            if (
                cfg.vector_config.use_rrf
                and expanded_queries
                and len(expanded_queries) > 1
            ):
                logger.info(f"Using RRF with {len(expanded_queries)} expanded queries")
                return await self._search_with_rrf(query, cfg, expanded_queries)
            elif expanded_queries and len(expanded_queries) == 1:
                logger.info(f"Using single expanded query: '{expanded_queries[0]}'")
                return await self._search_simple(expanded_queries[0], cfg)
            else:
                logger.info(f"Using simple search with main query: '{query}'")
                return await self._search_simple(query, cfg)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    async def _search_with_rrf(
        self,
        query: str,
        cfg: UniProtRetrievalConfig,
        expanded_queries: List[str],
    ) -> List[str]:
        """Search documents using Reciprocal Rank Fusion with parallel query processing."""
        tasks = []

        for expanded_query in expanded_queries:
            tasks.append(
                self._search_vectorstore(
                    expanded_query,
                    k=cfg.vector_config.rrf_per_query_k,
                    alpha=cfg.vector_config.rrf_alpha,
                )
            )

            if self._bm25_retriever:
                tasks.append(
                    self._search_bm25(
                        expanded_query, k=cfg.vector_config.rrf_per_query_k
                    )
                )

        logger.info(f"Executing {len(tasks)} search tasks in parallel for RRF")
        ranked_lists = await asyncio.gather(*tasks)

        for i, ranked_list in enumerate(ranked_lists):
            logger.info(f"Search {i+1} returned {len(ranked_list)} results")
            if ranked_list:
                first_doc = ranked_list[0]
                doc_id = first_doc.metadata.get(
                    "url", first_doc.metadata.get("id", hash(first_doc.page_content))
                )
                logger.info(f"  First result ID: {doc_id}")
                logger.info(
                    f"  First result content: {first_doc.page_content[:100]}..."
                )

        # Apply RRF to combine all ranked lists
        logger.info(
            f"Applying RRF with final_k={cfg.vector_config.rrf_final_k}, lambda={cfg.vector_config.rrf_lambda}"
        )
        top_docs, _, _ = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            final_k=cfg.vector_config.rrf_final_k,
            lambda_mult=cfg.vector_config.rrf_lambda,
            rrf_k=cfg.vector_config.rrf_cutoff_k,
            id_getter=lambda doc: doc.metadata.get(
                "url", doc.metadata.get("id", hash(doc.page_content))
            ),
        )

        logger.info(f"RRF returned {len(top_docs)} final results")

        return [doc.page_content for doc in top_docs]

    async def _search_simple(
        self, query: str, cfg: UniProtRetrievalConfig
    ) -> List[str]:
        """Search documents using simple similarity search."""
        top_docs = await self._search_vectorstore(
            query=query,
            k=cfg.vector_config.rrf_final_k,
            alpha=cfg.vector_config.alpha,
        )

        return [doc.page_content for doc in top_docs]

    async def _search_vectorstore(
        self,
        query: str,
        k: int,
        alpha: Optional[float] = None,
    ) -> List[Document]:
        """Search vectorstore using asyncio.to_thread."""
        return await asyncio.to_thread(self._search_vectorstore_sync, query, k, alpha)

    async def _search_bm25(
        self,
        query: str,
        k: int,
    ) -> List[Document]:
        """Search BM25 retriever using asyncio.to_thread."""
        return await asyncio.to_thread(self._search_bm25_sync, query, k)

    def _search_vectorstore_sync(
        self,
        query: str,
        k: int,
        alpha: Optional[float] = None,
    ) -> List[Document]:
        """Search the UniProt vectorstore."""
        if not self._vectorstore:
            logger.error("Vectorstore not initialized")
            return []

        try:
            if alpha is not None:
                docs_with_scores = self._vectorstore.similarity_search_with_score(
                    query, k=k
                )
                # Filter by score threshold (alpha) - higher scores are better
                docs = [doc for doc, score in docs_with_scores if score >= alpha]
            else:
                docs = self._vectorstore.similarity_search(query, k=k)

            for doc in docs:
                doc.metadata["search_type"] = "vector"

            return docs[:k]
        except Exception as e:
            logger.error(f"Error searching vectorstore: {e}")
            return []

    def _search_bm25_sync(
        self,
        query: str,
        k: int,
    ) -> List[Document]:
        """Search the UniProt BM25 retriever."""
        if not self._bm25_retriever:
            logger.debug("BM25 retriever not available")
            return []

        try:
            self._bm25_retriever.k = k
            docs = self._bm25_retriever.get_relevant_documents(query)

            for doc in docs:
                doc.metadata["search_type"] = "bm25"

            return docs[:k]
        except Exception as e:
            logger.error(f"Error searching BM25 retriever: {e}")
            return []

    def get_subdirectory(self) -> Optional[str]:
        """Get the current subdirectory being used."""
        return self._subdirectory

    def is_initialized(self) -> bool:
        """Check if the retriever is properly initialized."""
        return self._vectorstore is not None and (
            self._bm25_retriever is not None or self._vectorstore is not None
        )

    def close(self) -> None:
        """Close all connections and clear caches."""
        self._vectorstore = None
        self._bm25_retriever = None
        self._subdirectory = None
        logger.info("UniProt retriever closed")

    def __enter__(self) -> "UniProtRetriever":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation of the retriever."""
        status = "initialized" if self.is_initialized() else "not initialized"
        subdir = self._subdirectory or "unknown"
        return f"UniProtRetriever(subdirectory='{subdir}', status='{status}')"
