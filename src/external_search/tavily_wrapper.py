from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

class TavilyWrapper:
    """
    A wrapper for the Tavily API to perform searches, extract relevant results, and handle truncated titles.
    """
    def __init__(self, api_key=None, search_depth="advanced", max_results=5, include_domain=None):
        """
        Initialize the TavilyWrapper with optional API configuration.
        """
        self.tavily_client = TavilyClient(api_key=api_key)
        self.search_depth = search_depth
        self.max_results = max_results
        self.include_domain = include_domain

    def get_website_title(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptigitons.RequestException as e:
            print(f"Error retrieving the URL: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Check the <title> tag
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)

        # Check Open Graph meta tag
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content']

        # Check other common meta tags for titles
        meta_title = soup.find('meta', attrs={'name': 'title'})
        if meta_title and meta_title.get('content'):
            return meta_title['content']

        # Check common JSON-LD structured data used by search engines
        json_ld = soup.find('script', {'type': 'application/ld+json'})
        if json_ld:
            import json
            try:
                data = json.loads(json_ld.string)
                if 'headline' in data:
                    return data['headline']
            except json.JSONDecodeError:
                pass

        # Fallback to using headings if no title or meta information is adequate
        for heading in ['h1', 'h2']:
            h_tag = soup.find(heading)
            if h_tag:
                return h_tag.get_text(strip=True)

        return None

    def invoke(self, query):
        """
        Perform a search using the Tavily API.
        """
        try:
            response = self.tavily_client.search(
                query, 
                search_depth=self.search_depth, 
                max_results=self.max_results,
                include_domain="pubmed.ncbi.nlm.nih.gov"
            )
            return self._extract_results(response)
        except Exception as e:
            raise RuntimeError(f"Error during Tavily search: {e}")

    def _extract_results(self, response):
        """
        Extract and possibly augment titles and URLs from the API response.
        """
        results = response.get("results", [])
        full_results = []
        for result in results:
            title = result["title"]
            url = result["url"]
            # Check if title is truncated
            if '...' in title or 'PDF' in title:
                full_title = self.get_website_title(url)
                # Only add to results if full title does not contain '...'
                if full_title and '...' not in full_title:
                    title = full_title
                    full_results.append({"title": title, "link": url})
            else:
                full_results.append({"title": title, "link": url})
        return full_results
