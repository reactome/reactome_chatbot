from tavily import TavilyClient

class TavilyWrapper:
    """
    A wrapper for the Tavily API to perform searches and extract relevant results.
    """
    def __init__(self, api_key=None, search_depth="advanced", max_results=5):
        """
        Initialize the TavilyWrapper with optional API configuration.

        :param api_key: Optional API key for Tavily. If None, it will use the default environment variable.
        :param search_depth: The depth of the search, default is "advanced".
        :param max_results: Maximum number of results to fetch, default is 5.
        """
        self.tavily_client = TavilyClient(api_key=api_key)
        self.search_depth = search_depth
        self.max_results = max_results

    def invoke(self, query):
        """
        Perform a search using the Tavily API.

        :param query: The query string to search for.
        :return: A list of dictionaries containing titles and URLs of the results.
        """
        try:
            response = self.tavily_client.search(
                query, 
                search_depth=self.search_depth, 
                max_results=self.max_results
            )
            return self._extract_results(response)
        except Exception as e:
            raise RuntimeError(f"Error during Tavily search: {e}")

    def _extract_results(self, response):
        """
        Extract titles and URLs from the API response.

        :param response: The raw response from the Tavily API.
        :return: A list of dictionaries with 'title' and 'url' keys.
        """
        results = response.get("results", [])
        return [{"title": result["title"], "url": result["url"]} for result in results]
