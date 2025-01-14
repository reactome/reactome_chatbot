import requests
from typing import List, Dict



class PMCBestMatchAPIWrapper:
    """
    Wrapper for accessing PubMed Central (PMC) API with "best match" capabilities.
    Combines Entrez E-utilities for ranked queries and OA Web Service for full-text retrieval.
    """
    
    def __init__(self, email: str):
        """
        Initialize the wrapper with the required email for API requests.
        :param email: Your email address for API compliance.
        """
        self.base_url_entrez = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.base_url_oa = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.email = email

    def search_best_match(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search for PMCIDs based on a query, ranked by relevance (best match).
        :param query: Search term(s).
        :param max_results: Maximum number of results to return.
        :return: List of PMCIDs.
        """
        params = {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
            "email": self.email
        }
        response = requests.get(self.base_url_entrez, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_titles_and_links(self, pmc_ids: List[str]) -> List[Dict[str, str]]:
        """
        Retrieve titles and full-text links for the given PMCIDs.
        :param pmc_ids: List of PMCIDs to retrieve.
        :return: List of dictionaries with PMCIDs, titles, and full-text links.
        """
        results = []

        # Fetch titles using esummary
        params_summary = {
            "db": "pmc",
            "id": ",".join(pmc_ids),
            "retmode": "json",
            "email": self.email
        }
        response_summary = requests.get(self.base_url_summary, params=params_summary)
        response_summary.raise_for_status()
        summaries = response_summary.json().get("result", {})

        for pmc_id in pmc_ids:
            record = {
                "title": summaries.get(pmc_id, {}).get("title", "No title available"),
                "link": []
            }

            # Add the online version link
            online_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            record["link"] = online_link

            results.append(record)

        return results

    def invoke(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform a complete query: search for best-match articles and fetch titles and full-text links.
        :param query: Search term(s).
        :param max_results: Maximum number of results to return.
        :return: List of dictionaries containing PMCIDs, titles, and their full-text links.
        """
        pmc_ids = self.search_best_match(query, max_results)
        return self.fetch_titles_and_links(pmc_ids)