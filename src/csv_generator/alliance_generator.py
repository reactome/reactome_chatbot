import gzip
import os
import shutil

import requests
from typings import Optional


def download_file(url: str, dest: str, force: bool) -> Optional[str]:
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Check if the file exists and force is not set
    if os.path.exists(dest) and not force:
        print(f"File {dest} already exists. Skipping download.")
        return dest

    # Send the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        with open(dest, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {dest}.")

        # Check if the file is gzipped and decompress if necessary
        if dest.endswith(".gz"):
            unzipped_dest = dest[:-3]  # Remove '.gz' from the filename
            with gzip.open(dest, "rb") as f_in:
                with open(unzipped_dest, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"File unzipped successfully and saved to {unzipped_dest}.")
            os.remove(dest)  # Remove the gzipped file after extraction
            return unzipped_dest
        return dest
    else:
        print(
            f"Failed to download the file from {url}. Status code: {response.status_code}"
        )
        return None


def get_genes(version: str, force: bool) -> str:
    # Define the file path
    directory = f"csv_files/alliance/{version}"
    os.makedirs(directory, exist_ok=True)
    gene_csv = f"{directory}/genes.tsv"

    # Check if the file already exists and if regeneration is forced
    if not force and os.path.exists(gene_csv):
        print(f"File {gene_csv} already exists. Skipping download.")
        return gene_csv

    # Define the URL for the POST request
    url = "https://caltech-curation.textpressolab.com/pub/cgi-bin/forms/agr_simplemine.cgi"

    # Define the form data
    form_data = {
        "species": "MIX",
        "caseSensitiveToggle": "caseSensitive",
        "outputFormat": "download",
        "duplicatesToggle": "merge",
        "Gene ID": "Gene ID",
        "Gene Symbol": "Gene Symbol",
        "Gene Name": "Gene Name",
        "Description": "Description",
        "Species": "Species",
        "NCBI ID": "NCBI ID",
        "ENSEMBL ID": "ENSEMBL ID",
        "UniProtKB ID": "UniProtKB ID",
        "PANTHER ID": "PANTHER ID",
        "RefSeq ID": "RefSeq ID",
        "Synonym": "Synonym",
        "Disease Association": "Disease Association",
        "Expression Location": "Expression Location",
        "Expression Stage": "Expression Stage",
        "Variants": "Variants",
        "Genetic Interaction": "Genetic Interaction",
        "Molecular/Physical Interaction": "Molecular/Physical Interaction",
        "Homo sapiens Ortholog": "Homo sapiens Ortholog",
        "Mus musculus Ortholog": "Mus musculus Ortholog",
        "Rattus norvegicus Ortholog": "Rattus norvegicus Ortholog",
        "Danio rerio Ortholog": "Danio rerio Ortholog",
        "Drosophila melanogaster Ortholog": "Drosophila melanogaster Ortholog",
        "Caenorhabditis elegans Ortholog": "Caenorhabditis elegans Ortholog",
        "Saccharomyces cerevisiae Ortholog": "Saccharomyces cerevisiae Ortholog",
        "Xenopus laevis Ortholog": "Xenopus laevis Ortholog",
        "Xenopus tropicalis Ortholog": "Xenopus tropicalis Ortholog",
        "headers": "Gene ID\tGene Symbol\tGene Name\tDescription\tSpecies\tNCBI ID\tENSEMBL ID\tUniProtKB ID\tPANTHER ID\tRefSeq ID\tSynonym\tDisease Association\tExpression Location\tExpression Stage\tVariants\tGenetic Interaction\tMolecular/Physical Interaction\tHomo sapiens Ortholog\tMus musculus Ortholog\tRattus norvegicus Ortholog\tDanio rerio Ortholog\tDrosophila melanogaster Ortholog\tCaenorhabditis elegans Ortholog\tSaccharomyces cerevisiae Ortholog\tXenopus laevis Ortholog\tXenopus tropicalis Ortholog",
        "select all": "select all",
        "action": "all genes in this species",
        "geneInput": "",
        "geneNamesFile": "",
    }

    # Send the POST request
    response = requests.post(url, data=form_data)

    if response.status_code == 200:
        # Save the response content to a file
        with open(gene_csv, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file. Status code:", response.status_code)

    return gene_csv


def generate_all_csvs(version: str, force: bool) -> tuple:
    files = []

    # Download gene file
    gene_csv = get_genes(version, force)
    files.append(gene_csv)

    # Define other files to download
    other_files = {
        "disease": "https://fms.alliancegenome.org/download/DISEASE-ALLIANCE_COMBINED.tsv.gz",
        "expression": "https://fms.alliancegenome.org/download/EXPRESSION-ALLIANCE_COMBINED.tsv.gz",
        "molecular_interaction": "https://fms.alliancegenome.org/download/INTERACTION-MOL_COMBINED.tsv.gz",
        "genetic_interaction": "https://fms.alliancegenome.org/download/INTERACTION-GEN_COMBINED.tsv.gz",
        "orthology": "https://fms.alliancegenome.org/download/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz",
        "variants_c_elegans": "https://fms.alliancegenome.org/download/VARIANT-ALLELE_NCBITaxon6239.tsv.gz",
        "variants_zebrafish": "https://fms.alliancegenome.org/download/VARIANT-ALLELE_NCBITaxon7955.tsv.gz",
        "variants_fly": "https://fms.alliancegenome.org/download/VARIANT-ALLELE_NCBITaxon7227.tsv.gz",
        "variants_mouse": "https://fms.alliancegenome.org/download/VARIANT-ALLELE_NCBITaxon10090.tsv.gz",
        "variants_rat": "https://fms.alliancegenome.org/download/VARIANT-ALLELE_NCBITaxon10116.tsv.gz",
        "variants_yeast": "https://fms.alliancegenome.org/download/VARIANT-ALLELE_NCBITaxon559292.tsv.gz",
    }

    for name, url in other_files.items():
        gz_dest = f"csv_files/alliance/{version}/{name}.tsv.gz"
        csv_dest = f"csv_files/alliance/{version}/{name}.tsv"

        if not os.path.exists(csv_dest) or force:
            unzipped_dest = download_file(url, gz_dest, force)
            if unzipped_dest:
                files.append(unzipped_dest)
        else:
            print(f"File {csv_dest} already exists. Skipping download.")
            files.append(csv_dest)

    return tuple(files)
