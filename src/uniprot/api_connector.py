import pandas as pd
import re
import requests
from requests.adapters import HTTPAdapter, Retry

class UniProtDataCleaner:
    """
    A class to download and clean data from the UniProt database.
    
    Attributes:
    download_url (str): URL to download the raw data.
    file_path (str): Local path to save the downloaded file.
    df (pd.DataFrame): DataFrame holding the data during processing.
    
    Methods:
    download_data(): Downloads the data from the UniProt database.
    clean_data(): Processes the downloaded data and saves it in CSV format.
    """
    def __init__(self, download_url, file_path):
        """
        Initializes the UniProtDataCleaner with the download URL and file path.
        
        Args:
        download_url (str): URL to download the raw data.
        file_path (str): Local path to save the downloaded Excel file.
        """
        self.download_url = download_url
        self.file_path = file_path
        self.df = None
        self.session = self._initialize_session()


    def _initialize_session(self):
        """
        Creates a session with retry logic for robust downloading.
        """
        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def get_next_link(self, headers):
        """
        Parses the 'Link' header to find the URL for the next batch of data.

        Args:
        headers (dict): HTTP headers from the response.

        Returns:
        str: URL of the next batch or None if there is no next batch.
        """
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)
        return None

    def get_batch(self, batch_url):
        """
        Generator to download data in batches.

        Args:
        batch_url (str): URL to start downloading batches from.

        Yields:
        tuple: The response object and the total number of results.
        """
        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()  # Ensure we stop on HTTP errors
            total = response.headers.get("x-total-results", 0)
            yield response, total
            batch_url = self.get_next_link(response.headers)

    def download_data(self):
        """
        Downloads data batch by batch, updating each batch to an Excel file until complete.
        """
        progress = 0
        with open(self.file_path, 'wb') as f:
            for response, total in self.get_batch(self.download_url):
                f.write(response.content)
                progress += 1  # Increment progress per batch (adjust if you know the batch size)
                print(f"Downloaded {progress} batches; Total: {total}")

        print("Download completed successfully.")

    def load_data(self):
        """Loads data from Excel file into a DataFrame."""
        print("Loading data from file...")
        self.df = pd.read_excel(self.file_path)

    def clean_entry_names(self):
        """Remove specific substrings from the 'Entry Name' column."""
        self.df['Entry Name'] = self.df['Entry Name'].str.replace('_HUMAN', '', regex=False)

    def remove_prefixes(self):
        """Remove prefixes from specified columns."""
        prefix_map = {
            'Pathway': 'PATHWAY: ',
            'Subunit structure': 'SUBUNIT: ',
            'Subcellular location [CC]': 'SUBCELLULAR LOCATION: ',
            'Domain [CC]': 'DOMAIN: ',
            'Tissue specificity': 'TISSUE SPECIFICITY: ',
            'Involvement in disease': 'DISEASE: ',
            'Function [CC]': 'FUNCTION: ',
            'Miscellaneous [CC]': 'MISCELLANEOUS: ',
            'Induction': 'INDUCTION: ',
        }
        for column, prefix in prefix_map.items():
            if column in self.df.columns:
                self.df[column] = self.df[column].str.replace(prefix, '', regex=False).str.strip()
    
    def format_mass(self):
        """Format the 'Mass' column by appending ' Da' to each mass value."""
        if 'Mass' in self.df.columns:
            self.df['Mass'] = self.df['Mass'].apply(lambda x: f"{x} Da")
    
    def clean_citations_and_evidence_codes(self):
        """Remove citations and evidence codes from textual columns."""
        patterns = [
            r'\{ECO:[^\}]*\}', 
            r'\(PubMed:[^\)]*\)', 
            r'\[MIM:[^\]]*\]', 
            r'  +'
        ]
        for column in self.df.columns:
            for pattern in patterns:
                self.df[column] = self.df[column].str.replace(pattern, '', regex=True).str.strip()

    def clean_motif_column(self):
        """Reformat entries in the 'Motif' column."""
        def reformat_motif(entry):
            if pd.isna(entry):
                return entry
            pattern = r"MOTIF (\d+\.\.\d+); /note=\"([^\"]*)\"; /evidence=\"[^\"]*\""
            matches = re.findall(pattern, entry)
            return '; '.join([f"Has a {note} at position {pos.replace('..', '-')}" for pos, note in matches])

        self.df['Motif'] = self.df['Motif'].apply(reformat_motif)

    def replace_entry_with_url(self):
        """Replace 'Entry' column with URLs constructed from entry IDs."""
        base_url = "https://www.uniprot.org/uniprotkb/"
        self.df['Entry'] = base_url + self.df['Entry'].astype(str) + "/entry"

    def format_gene_and_protein_names(self):
        """Format gene synonyms and protein names with semicolons and proper punctuation."""
        self.df['Gene Names (synonym)'] = self.df['Gene Names (synonym)'].str.replace(' ', '; ', regex=False).str.strip('; ')
        self.df['Protein names'] = self.df['Protein names'].apply(
            lambda x: '; '.join([item.strip() for item in re.split(r'\) \(', x.replace('(', '; ').replace(')', ''))]))

    def rename_columns(self):
        """Rename columns as specified."""
        new_column_names = {
            'Entry': 'url',
            'Entry Name': 'short_protein_name',
            'Gene Names (primary)': 'primary_gene_name',
            'Gene Names (synonym)': 'synonym_gene_names',
            'Protein names': 'full_protein_name',
            'Mass': 'protein_mass',
            'Protein families': 'protein_family',
            'Pathway': 'pathways_involved',
            'Subunit structure': 'subunit_structure',
            'Subcellular location [CC]': 'subcellular_localization',
            'Domain [CC]': 'protein_domain',
            'Motif': 'protein_motif',
            'Induction': 'induced_by',
            'Tissue specificity': 'tissue_specificity',
            'Involvement in disease': 'disease_involvement',
            'Function [CC]': 'protein_function',
            'Miscellaneous [CC]': 'additional_info'
        }
        self.df.rename(columns=new_column_names, inplace=True)

    def save_cleaned_data(self):
        """Save the cleaned data to a CSV file."""
        output_path = self.file_path.replace('.xlsx', '_cleaned.csv')
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

    def clean_data(self):
        """Execute all cleaning steps in sequence."""
        self.load_data()
        self.clean_entry_names()
        self.remove_prefixes()
        self.format_mass()  # Format the Mass column before cleaning citations and codes
        self.clean_citations_and_evidence_codes()
        self.clean_motif_column()
        self.replace_entry_with_url()
        self.format_gene_and_protein_names()
        self.rename_columns()
        self.save_cleaned_data()

# Example usage:
if __name__ == "__main__":
    url = "https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Cid%2Cgene_primary%2Cgene_synonym%2Cprotein_name%2Cmass%2Cprotein_families%2Ccc_pathway%2Ccc_subunit%2Ccc_subcellular_location%2Ccc_domain%2Cft_motif%2Ccc_induction%2Ccc_tissue_specificity%2Ccc_disease%2Ccc_function%2Ccc_miscellaneous&format=xlsx&query=%28reviewed%3Atrue%29+AND+%28reviewed%3Atrue%29+AND+%28model_organism%3A9606%29"
    file_path = '../embeddings/uniprot_data.xlsx'
    cleaner = UniProtDataCleaner(url, file_path)
    cleaner.download_data()
    cleaner.clean_data()
