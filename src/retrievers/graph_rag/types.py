from typing import List, Optional, Protocol, Dict, Any
from dataclasses import dataclass
from neo4j import Record
from langchain.schema import Document


@dataclass
class NodeInfo:
    node_id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class NeighborInfo:
    node_id: str
    labels: List[str]
    properties: Dict[str, Any]
    edge_type: str
    edge_direction: str
    edge_props: Dict[str, Any]


@dataclass
class GraphContext:
    seeds: Dict[str, NodeInfo]
    nodes: Dict[str, NodeInfo]
    edges: List[NeighborInfo]
    summary: Dict[str, Any]


class IVectorClient(Protocol):
    def search_mmr(
        self,
        query: str,
        k: int,
        lambda_mult: float,
        fetch_k: Optional[int] = None,
    ) -> List[Document]:
        ...

    def search_similar(
        self,
        query: str,
        k: int,
        alpha: Optional[float] = None,
    ) -> List[Document]:
        ...

    def close(self) -> None:
        ...


class IGraphClient(Protocol):
    def invoke(
        self,
        query: str,
        params: Dict[str, Any],
        database: Optional[str] = None,
    ) -> List[Record]:
        ...

    def close(self) -> None:
        ...


