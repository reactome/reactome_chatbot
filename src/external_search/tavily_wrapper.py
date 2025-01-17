from typing import Any, Iterable, Literal, TypedDict

from tavily import AsyncTavilyClient, MissingAPIKeyError

from external_search.state import GraphState
from util.logging import logging

class WebSearchResult(TypedDict):
    title: str
    url: str
    content: str

class TavilyWrapper:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_depth: Literal["basic", "advanced"] = "advanced",
        max_results: int = 5,
    ):
        self.tavily_client: AsyncTavilyClient | None = None
        self.search_depth = search_depth
        self.max_results = max_results

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

        try:
            response: dict[str, Any] = await self.tavily_client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
            )
        except Exception as e:
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

    async def ainvoke(
        self,
        state: GraphState
    ) -> dict[str, str]:
        query: str = state["question"]
        web_search_results: list[WebSearchResult] = await self.search(query)
        formatted_results: str = self.format_results(web_search_results)
        return { "search_results": formatted_results }

    @staticmethod
    def format_results(web_search_results: Iterable[WebSearchResult]) -> str:
        formatted = f"Here are some external resources you may find helpful:"
        for result in web_search_results:
            formatted += f"\n- [{result['title']}]({result['content']})"
        return formatted
