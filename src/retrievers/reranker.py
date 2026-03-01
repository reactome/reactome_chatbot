import asyncio

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


def rerank(
    docs: list[Document],
    query: str,
    top_n: int = 5,
) -> list[Document]:
    passages = [
        {"id": i, "text": doc.page_content}
        for i, doc in enumerate(docs)
    ]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    return [docs[result["id"]] for result in results[:top_n]]


async def arerank(
    docs: list[Document],
    query: str,
    top_n: int = 5,
) -> list[Document]:
    passages = [
        {"id": i, "text": doc.page_content}
        for i, doc in enumerate(docs)
    ]
    request = RerankRequest(query=query, passages=passages)
    # ranker.rerank() is a blocking CPU operation (neural network inference)
    # calling it directly inside async would freeze the entire event loop
    # asyncio.to_thread runs it in a background thread keeping the event loop free
    results = await asyncio.to_thread(ranker.rerank, request)
    return [docs[result["id"]] for result in results[:top_n]]