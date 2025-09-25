import logging
from typing import List, Optional, Dict, Any

import weaviate
from neo4j import GraphDatabase, Record
from neo4j.exceptions import Neo4jError
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from .types import IVectorClient, IGraphClient

logger = logging.getLogger(__name__)

class WeaviateVectorClient(IVectorClient):
    """
    Unified Weaviate vector client with configurable functionality.
    
    This class provides vector search capabilities that can be configured
    for different use cases (Reactome, UniProt, etc.) through constructor parameters.
    """
    
    DEFAULT_MMR_FETCH_K_MULTIPLIER = 4
    MIN_MMR_FETCH_K = 20
    
    def __init__(
        self,
        host: str,
        port: int,
        grpc_port: int,
        index_name: str,
        openai_api_key: str,
        embedding_model: str,
        text_key: str = "text_content",
        attributes: List[str] = None,
        validate_stable_ids: bool = False,
    ):
        """
        Initialize the Weaviate vector client.
        
        Args:
            host: Weaviate host
            port: Weaviate HTTP port
            grpc_port: Weaviate gRPC port
            index_name: Weaviate index/class name
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model name
            text_key: Key for text content in documents
            attributes: List of metadata attributes to store
            validate_stable_ids: Whether to validate stable_id presence in documents
        """
        self.validate_stable_ids = validate_stable_ids
        self._client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port,
            headers={"X-OpenAI-Api-Key": openai_api_key},
        )
        self._store = WeaviateVectorStore(
            client=self._client,
            index_name=index_name,
            text_key=text_key,
            attributes=attributes or ["name", "labels"],
            embedding=OpenAIEmbeddings(model=embedding_model),
        )

    def search_mmr(
        self, query: str, k: int, lambda_mult: float, fetch_k: Optional[int] = None
    ) -> List[Document]:
        """Perform MMR (Maximal Marginal Relevance) search."""
        try:
            fk = fetch_k if fetch_k is not None else max(self.MIN_MMR_FETCH_K, self.DEFAULT_MMR_FETCH_K_MULTIPLIER * k)
            docs = self._store.max_marginal_relevance_search(
                query=query, k=k, fetch_k=fk, lambda_mult=lambda_mult
            )
            return self._process_docs(docs)
        except Exception:
            logger.exception("MMR search failed for query=%r", query)
            raise

    def search_similar(
        self, query: str, k: int, alpha: Optional[float] = None
    ) -> List[Document]:
        """Perform similarity search with optional alpha threshold."""
        try:
            if alpha is None:
                docs = self._store.similarity_search(query, k=k)
            else:
                docs = self._store.similarity_search(query, k=k, alpha=alpha)
            return self._process_docs(docs)
        except Exception:
            logger.exception("Similarity search failed for query=%r", query)
            raise

    def extract_ids(self, docs: List[Document]) -> List[str]:
        """Extract stable IDs from documents."""
        stable_ids = []
        for doc in docs:
            stable_id = doc.metadata.get("stId") or doc.metadata.get("stable_id")
            if stable_id:
                stable_ids.append(stable_id)
            else:
                logger.warning("Document missing stable ID: %s", doc.metadata.get("name", "unknown"))
        return stable_ids

    def _process_docs(self, docs: List[Document]) -> List[Document]:
        """
        Process documents after retrieval.
        
        Optionally validates stable_id presence based on configuration.
        """
        if self.validate_stable_ids:
            for doc in docs:
                if "stId" not in doc.metadata and "stable_id" not in doc.metadata:
                    logger.warning("Document missing stable ID: %s", doc.metadata.get("name", "unknown"))
        return docs

    def close(self) -> None:
        """Close the Weaviate client."""
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception as e:
            logger.warning("Error closing Weaviate client: %s", e)

class BaseNeo4jGraphClient(IGraphClient):
    """
    Base Neo4j graph client with common functionality.
    
    This class provides the core graph database operations that are shared
    across different implementations.
    """
    
    def __init__(
        self, 
        uri: str, 
        user: str, 
        password: str, 
        default_db: str = "neo4j"
    ):
        """
        Initialize the base Neo4j graph client.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            default_db: Default database name
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._default_db = default_db

    def invoke(
        self, query: str, params: Dict[str, Any], database: Optional[str] = None
    ) -> List[Record]:
        """Execute a Cypher query with parameters."""
        db = database or self._default_db
        with self._driver.session(database=db) as session:
            try:
                result = session.run(query, **params)
                return list(result)
            except Neo4jError as e:
                logger.exception("Cypher execution failed: %s", e)
                raise

    def close(self) -> None:
        """Close the Neo4j driver."""
        try:
            if hasattr(self, "_driver"):
                self._driver.close()
        except Exception as e:
            logger.warning("Error closing Neo4j driver: %s", e)
