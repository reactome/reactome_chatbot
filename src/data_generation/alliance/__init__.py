"""Building embeddings from Alliance of Genome Resources data.

PARKED, NOT ABANDONED. Deliberately not deployed: every profile default is
React-to-Me, in `config_default.yml` and in `chat-chainlit.py`'s fallback, so this
runs only if a `config.yml` names it explicitly.

Do not delete it. Adam Wright did the work to make Alliance results possible, and it is kept so the capability can be resurrected
rather than rebuilt. Equally, do not invest in it while it is parked -- it does not
need new features, and a change that merely keeps it importable is enough.

If it is ever un-parked, note that it has not been exercised since 2026-09-10, so
its behaviour is unverified even where the code still type-checks.
"""

import logging
import shutil
from pathlib import Path

import requests
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_community.vectorstores import Chroma

from data_generation.alliance.csv_generator import generate_all_csvs
from data_generation.embeddings import build_embeddings
from data_generation.metadata_csv_loader import MetaDataCSVLoader


def get_release_version() -> str:
    url: str = "https://www.alliancegenome.org/api/releaseInfo"
    response = requests.get(url, timeout=60)
    if response.status_code == 200:
        response_json = response.json()
        release_version: str | None = response_json.get("releaseVersion")
        if release_version:
            return release_version
        raise ValueError("Release version not found in the response.")
    raise ConnectionError(
        f"Failed to get the response. Status code: {response.status_code}"
    )


# The Alliance column schemas, at module scope because they are data: 203 lines
# of them inside the function made a 40-line routine read as a 246-line one.
# tests/data_generation/test_alliance_columns.py parses these and guards them
# against the missing-comma bug that once collapsed seven molecular_interaction
# columns into three.
COLUMN_SCHEMAS: dict[str, list[str]] = {
    "genes": [
        "Your Input",
        "Gene ID",
        "Gene Symbol",
        "Gene Name",
        "Description",
        "Species",
        "NCBI ID",
        "ENSEMBL ID",
        "UniProtKB ID",
        "PANTHER ID",
        "RefSeq ID",
        "Synonym",
        "Disease Association",
        "Expression Location",
        "Expression Stage",
        "Variants",
        "Genetic Interaction",
        "Molecular/Physical Interaction",
        "Homo sapiens Ortholog",
        "Mus musculus Ortholog",
        "Rattus norvegicus Ortholog",
        "Danio rerio Ortholog",
        "Drosophila melanogaster Ortholog",
        "Caenorhabditis elegans Ortholog",
        "Saccharomyces cerevisiae Ortholog",
        "Xenopus laevis Ortholog",
        "Xenopus tropicalis Ortholog",
    ],
    "disease": [
        "Taxon",
        "SpeciesName",
        "DBobjectType",
        "DBObjectID",
        "DBObjectSymbol",
        "AssociationType",
        "DOID",
        "DOtermName",
        "WithOrtholog",
        "InferredFromID",
        "InferredFromSymbol",
        "ExperimentalCondition",
        "Modifier",
        "EvidenceCode",
        "EvidenceCodeName",
        "Reference",
        "Date",
        "Source",
    ],
    "expression": [
        "Species",
        "SpeciesID",
        "GeneID",
        "GeneSymbol",
        "Location",
        "StageTerm",
        "AssayID",
        "AssayTermName",
        "CellularComponentID",
        "CellularComponentTerm",
        "CellularComponentQualifierIDs",
        "CellularComponentQualifierTermNames",
        "SubStructureID",
        "SubStructureName",
        "SubStructureQualifierIDs",
        "SubStructureQualifierTermNames",
        "AnatomyTermID",
        "AnatomyTermName",
        "AnatomyTermQualifierIDs",
        "AnatomyTermQualifierTermNames",
        "SourceURL",
        "Source,Reference",
    ],
    "molecular_interaction": [
        "ID(s) interactor A",
        "ID(s) interactor B",
        "Alt. ID(s) interactor A",
        "Alt. ID(s) interactor B",
        "Alias(es) interactor A",
        "Alias(es) interactor B",
        "Interaction detection method(s)",
        "Publication 1st author(s)",
        "Publication Identifier(s)",
        "Taxid interactor A",
        "Taxid interactor B",
        "Interaction type(s)",
        "Source database(s)",
        "Interaction identifier(s)",
        "Confidence value(s)",
        "Expansion method(s)",
        "Biological role(s) interactor A",
        "Biological role(s) interactor B",
        "Experimental role(s) interactor A",
        "Experimental role(s) interactor B",
        "Type(s) interactor A",
        "Type(s) interactor B",
        "Xref(s) interactor A",
        "Xref(s) interactor B",
        "Interaction Xref(s)",
        "Annotation(s) interactor A",
        "Annotation(s) interactor B",
        "Interaction annotation(s)",
        "Host organism(s)",
        "Interaction parameter(s)",
        "Creation date",
        "Update date",
        "Checksum(s) interactor A",
        "Checksum(s) interactor B",
        "Interaction Checksum(s) Negative",
        "Feature(s) interactor A",
        "Feature(s) interactor B",
        "Stoichiometry(s) interactor A",
        "Stoichiometry(s) interactor B",
        "Identification method participant A",
        "Identification method participant B",
    ],
    "genetic_interaction": [
        "ID(s) interactor A",
        "ID(s) interactor B",
        "Alt. ID(s) interactor A",
        "Alt. ID(s) interactor B",
        "Alias(es) interactor A",
        "Alias(es) interactor B",
        "Interaction detection method(s)",
        "Publication 1st author(s)",
        "Publication Identifier(s)",
        "Taxid interactor A",
        "Taxid interactor B",
        "Interaction type(s)",
        "Source database(s)",
        "Interaction identifier(s)",
        "Confidence value(s)",
        "Expansion method(s)",
        "Biological role(s) interactor A",
        "Biological role(s) interactor B",
        "Experimental role(s) interactor A",
        "Experimental role(s) interactor B",
        "Type(s) interactor A",
        "Type(s) interactor B",
        "Xref(s) interactor A",
        "Xref(s) interactor B",
        "Interaction Xref(s)",
        "Annotation(s) interactor A",
        "Annotation(s) interactor B",
        "Interaction annotation(s)",
        "Host organism(s)",
        "Interaction parameter(s)",
        "Creation date",
        "Update date",
        "Checksum(s) interactor A",
        "Checksum(s) interactor B",
        "Interaction Checksum(s) Negative",
        "Feature(s) interactor A",
        "Feature(s) interactor B",
        "Stoichiometry(s) interactor A",
        "Stoichiometry(s) interactor B",
        "Identification method participant A",
        "Identification method participant B",
    ],
    "orthology": [
        "Gene1ID Gene1Symbol",
        "Gene1SpeciesTaxonID",
        "Gene1SpeciesName",
        "Gene2ID Gene2Symbol",
        "Gene2SpeciesTaxonID",
        "Gene2SpeciesName",
        "Algorithms",
        "AlgorithmsMatch",
        "OutOfAlgorithms",
        "IsBestScore",
        "IsBestRevScore",
    ],
    "variants": [
        "Taxon",
        "SpeciesName",
        "AlleleId",
        "AlleleSymbol",
        "AlleleSynonyms",
        "VariantId",
        "VariantSymbol",
        "VariantSynonyms",
        "VariantCrossReferences",
        "AlleleAssociatedGeneId",
        "AlleleAssociatedGeneSymbol",
        "VariantAffectedGeneId",
        "VariantAffectedGeneSymbol",
        "Category",
        "VariantsTypeId",
        "VariantsTypeName",
        "VariantsHgvsNames",
        "Assembly",
        "Chromosome",
        "StartPosition",
        "EndPosition",
        "SequenceOfReference",
        "SequenceOfVariant",
        "MostSevereConsequenceName",
        "VariantInformationReference",
        "HasDiseaseAnnotations",
        "HasPhenotypeAnnotations",
    ],
}


def upload_to_chromadb(
    embeddings_dir: str,
    version: str,
    force: bool,  # Changed from str to bool
    hf_model: str | None = None,
    device: str | None = None,
) -> Chroma | None:
    csv_dir = "./csv_files/alliance/" + version + "/"

    # Only `genes` is embedded. The other six lists above are curated MITAB and
    # Alliance schemas kept for when they are, and `tests/data_generation/
    # test_alliance_columns.py` guards them against the missing-comma bug that
    # once collapsed seven molecular_interaction columns into three. They are
    # data waiting on code, not dead code.
    #
    # They are not reachable as written: csv_generator downloads variants as
    # variants_c_elegans.tsv, variants_zebrafish.tsv and so on, so the
    # "variants" key below could never match a file. Anything that starts
    # embedding them must reconcile the two lists rather than assume they agree.
    embedded = ("genes",)

    db = None
    for filetype, column_names in COLUMN_SCHEMAS.items():
        if filetype not in embedded:
            continue
        file = Path(csv_dir) / f"{filetype}.tsv"
        if not file.is_file():
            # Returned rather than raised from inside Chroma: `force` governs
            # whether the CSVs were downloaded at all, and a caller that skipped
            # that should get a clear message, not a FileNotFoundError.
            logging.warning("No Alliance %s file at %s; skipping.", filetype, file)
            continue

        loader = MetaDataCSVLoader(
            file_path=str(file),
            metadata_columns=column_names,
            encoding="utf-8",
            csv_args={"delimiter": "\t"},
        )
        docs = loader.load()
        logging.info("Embedding %d Alliance %s documents.", len(docs), filetype)

        persist = Path(embeddings_dir) / filetype
        if persist.exists():
            # Chroma.from_documents appends; a second run would double the
            # collection silently, as it did to the reactome bundle.
            shutil.rmtree(persist)
            SharedSystemClient.clear_system_cache()

        db = Chroma.from_documents(
            documents=docs,
            embedding=build_embeddings(hf_model, device),
            persist_directory=str(persist),
        )

    return db


def generate_alliance_embeddings(
    embeddings_dir: str,
    force: bool = False,
    hf_model: str | None = None,
    device: str | None = None,
    **kwargs: object,
) -> None:
    release_version = get_release_version()
    print(f"Release Version: {release_version}")
    if not embeddings_dir.endswith(release_version):
        print(
            "The embeddings dir you gave is:",
            embeddings_dir,
            " where the live version of Alliance is ",
            release_version,
        )
        exit()

    generate_all_csvs(release_version, force)
    upload_to_chromadb(embeddings_dir, release_version, force, hf_model, device)
