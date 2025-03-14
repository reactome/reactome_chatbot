import re
from pathlib import Path

import pandas as pd

from data_generation.uniprot.api_connector import UniProtAPIConnector


class UniProtDataCleaner:
    def __init__(self, csv_dir: Path):
        self.download_url = UniProtAPIConnector.get_download_url()
        self.xlsx_path = csv_dir / "uniprot_data.xlsx"
        self.csv_path = self.xlsx_path.with_suffix(".csv")
        self.df = None
        self.api = UniProtAPIConnector()

    def download_data(self):
        """Downloads data batch by batch using UniProt API connector."""
        progress = 0
        with open(self.xlsx_path, "wb") as f:
            for response, total in self.api.get_batch(self.download_url):
                f.write(response.content)
                progress += 1
                print(f"Downloaded {progress} batches; Total: {total}")
        print(f"✅ UniProt data downloaded successfully to {self.xlsx_path}")

    def load_data(self):
        """Loads data from Excel file into a DataFrame."""
        print(f"Loading data from {self.xlsx_path}")
        self.df = pd.read_excel(self.xlsx_path)

    def clean_data(self):
        """Cleans the UniProt data using predefined processing steps."""
        self.load_data()
        self.remove_prefixes()
        self.format_mass()
        self.clean_evidence_codes()
        self.clean_columns()
        self.add_url()
        self.format_names()
        self.rename_columns()
        self.df.to_csv(self.csv_path, index=False)
        print(f"Cleaned data saved to {self.csv_path}")

    def remove_prefixes(self):
        """Remove prefixes from specified columns."""
        prefix_map = {
            "Entry Name": "_HUMAN",
            "Pathway": "PATHWAY: ",
            "Subunit structure": "SUBUNIT: ",
            "Subcellular location [CC]": "SUBCELLULAR LOCATION: ",
            "Domain [CC]": "DOMAIN: ",
            "Tissue specificity": "TISSUE SPECIFICITY: ",
            "Involvement in disease": "DISEASE: ",
            "Function [CC]": "FUNCTION: ",
            "Miscellaneous [CC]": "MISCELLANEOUS: ",
            "Induction": "INDUCTION: ",
            "Activity regulation": "ACTIVITY REGULATION:",
        }
        for column, prefix in prefix_map.items():
            if column in self.df.columns:
                self.df[column] = (
                    self.df[column].str.replace(prefix, "", regex=False).str.strip()
                )

    def add_url(self):
        """Replace 'Entry' column with URLs constructed from entry IDs."""
        base_url = "https://www.uniprot.org/uniprotkb/"
        self.df["Entry"] = base_url + self.df["Entry"].astype(str) + "/entry"

    def format_names(self):
        """Format gene synonyms and protein names with semicolons and proper punctuation."""
        self.df["Gene Names"] = (
            self.df["Gene Names"].str.replace(" ", "; ", regex=False).str.strip("; ")
        )
        self.df["Protein names"] = self.df["Protein names"].apply(
            lambda x: "; ".join(
                [
                    item.strip()
                    for item in re.split(
                        r"\) \(", x.replace("(", "; ").replace(")", "")
                    )
                ]
            )
        )

    def format_mass(self):
        """Format the 'Mass' column by appending ' Da' to each mass value."""
        if "Mass" in self.df.columns:
            self.df["Mass"] = self.df["Mass"].apply(lambda x: f"{x} Da")

    def clean_evidence_codes(self):
        """Remove citations and evidence codes from textual columns."""
        patterns = [r"\{ECO:[^\}]*\}", r"\(PubMed:[^\)]*\)", r"\[MIM:[^\]]*\]", r"  +"]
        for column in self.df.columns:
            for pattern in patterns:
                self.df[column] = (
                    self.df[column].str.replace(pattern, "", regex=True).str.strip()
                )

    def clean_columns(self):
        """Reformat entries in the 'Motif' column."""

        def reformat_motif(entry):
            if pd.isna(entry):
                return entry
            pattern = r"MOTIF (\d+\.\.\d+); /note=\"([^\"]*)\"; /evidence=\"[^\"]*\""
            matches = re.findall(pattern, entry)
            return "; ".join(
                [
                    f"Has a {note}  at position {pos.replace('..', '-')}"
                    for pos, note in matches
                ]
            )

        def reformat_domain(entry):
            if pd.isna(entry):
                return entry
            pattern = r"DOMAIN (\d+\.\.\d+); /note=\"([^\"]*)\"; /evidence=\"[^\"]*\""
            matches = re.findall(pattern, entry)
            return "; ".join(
                [
                    f"Has a {note} domain at position {pos.replace('..', '-')}"
                    for pos, note in matches
                ]
            )

        self.df["Motif"] = self.df["Motif"].apply(reformat_motif)
        self.df["Domain [FT]"] = self.df["Domain [FT]"].apply(reformat_domain)

    def rename_columns(self):
        """Rename columns as specified."""
        new_column_names = {
            "Entry": "url",
            "Gene Names": "gene_names",
            "Entry Name": "short_protein_name",
            "Protein names": "full_protein_name",
            "Protein families": "protein_family",
            "Mass": "molecular_weight",
            "Domain [FT]": "protein_domains",
            "Domain [CC]": "domain_annotations",
            "Motif": "protein_motif",
            "Subunit structure": "subunit_structure",
            "Pathway": "biological_pathways",
            "Induction": "expression_induction",
            "Activity regulation": "activity_regulation",
            "Subcellular location [CC]": "subcellular_localization",
            "Tissue specificity": "tissue_expression",
            "Involvement in disease": "disease_associations",
            "Function [CC]": "protein_function",
            "Miscellaneous [CC]": "additional_notes",
        }
        self.df.rename(columns=new_column_names, inplace=True)


def generate_uniprot_csv(parent_dir: Path) -> Path:
    csv_dir = Path(parent_dir) / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)

    cleaner = UniProtDataCleaner(csv_dir)
    cleaner.download_data()
    cleaner.clean_data()
    cleaner.xlsx_path.unlink(missing_ok=True)
    return cleaner.csv_path
