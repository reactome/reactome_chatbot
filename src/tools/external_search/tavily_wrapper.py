import time
from threading import Lock
from typing import Any, Literal

from tavily import AsyncTavilyClient, MissingAPIKeyError

from tools.external_search.state import GraphState, WebSearchResult
from util.logging import logging


class TavilyWrapper:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_depth: Literal["basic", "advanced"] = "advanced",
        max_results: int = 5,
        rate_limit: int = 100,  # requests per minute
    ):
        self.tavily_client: AsyncTavilyClient | None = None
        self.search_depth = search_depth
        self.max_results = max_results

        self.rate_interval: float = 60 / rate_limit  # seconds between requests
        self.last_request_time: float = time.monotonic()
        self.lock = Lock()

        try:
            self.tavily_client = AsyncTavilyClient(api_key)
        except MissingAPIKeyError:
            logging.warning(
                "No Tavily API key was provided (TAVILY_API_KEY) - "
                "external search feature is disabled."
            )

    async def search(self, query: str) -> list[WebSearchResult]:
        if self.tavily_client is None:
            return []

        with self.lock:
            now: float = time.monotonic()
            if now - self.last_request_time < self.rate_interval:
                return []
            self.last_request_time = now

        try:
            response: dict[str, Any] = await self.tavily_client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
            )
        except Exception:
            logging.warning("Tavily Search raised an Exception:", exc_info=True)
            return []

        results: list[dict[str, Any]] = response.get("results", [])
        return [
            WebSearchResult(
                title=result["title"],
                url=result["url"],
                content=result.get("content", ""),
            )
            for result in results
            if all(key in result for key in ["title", "url"])
        ]

    async def ainvoke(self, state: GraphState) -> dict[str, list[WebSearchResult]]:
        query: str = state["question"]
        search_results: list[WebSearchResult] = await self.search(query)
        return {"search_results": search_results}

    @staticmethod
    def format_results(web_search_results: list[WebSearchResult]) -> str:
        if len(web_search_results) == 0:
            return ""
        formatted = "Here are some external resources you may find helpful:"
        for result in web_search_results:
            formatted += f"\n- [{result['title']}]({result['url']})"
        return formatted
