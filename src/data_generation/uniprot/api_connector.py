import re

import requests
from requests.adapters import HTTPAdapter, Retry


class UniProtAPIConnector:
    BASE_URL = "https://rest.uniprot.org/uniprotkb/stream"

    @staticmethod
    def get_download_url():
        """
        Returns the UniProt API URL for downloading human-reviewed protein data.
        """
        query_params = (
            "?fields=accession%2Cgene_names%2Cid%2Cprotein_name%2Cprotein_families%2Cmass"
            "%2Cft_domain%2Ccc_domain%2Cft_motif%2Ccc_subunit%2Ccc_pathway%2Ccc_induction"
            "%2Ccc_activity_regulation%2Ccc_subcellular_location%2Ccc_tissue_specificity"
            "%2Ccc_disease%2Ccc_function%2Ccc_miscellaneous&format=xlsx"
            "&query=%28reviewed%3Atrue%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29"
        )
        return UniProtAPIConnector.BASE_URL + query_params

    def __init__(self):
        self.session = self._initialize_session()

    def _initialize_session(self):
        """Creates a session with retry logic for robust downloading."""
        retries = Retry(
            total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504]
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def get_next_link(self, headers):
        """Parses the 'Link' header to find the URL for the next batch of data."""
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)
        return None

    def get_batch(self, batch_url):
        """Generator to download data in batches."""
        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()  # Ensure we stop on HTTP errors
            total = response.headers.get("x-total-results", 0)
            yield response, total
            batch_url = self.get_next_link(response.headers)
