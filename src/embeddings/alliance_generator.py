import os
import requests

import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import (HuggingFaceEmbeddings,
                                   HuggingFaceEndpointEmbeddings)
from langchain_openai import OpenAIEmbeddings

from csv_generator.alliance_generator import generate_all_csvs
from metadata_csv_loader import MetaDataCSVLoader


def get_release_version() -> str:
    url: str = "https://www.alliancegenome.org/api/releaseInfo"
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        release_version = response_json.get("releaseVersion")
        if release_version:
            return release_version
        else:
            raise ValueError("Release version not found in the response.")
    else:
        raise ConnectionError(
            f"Failed to get the response. Status code: {response.status_code}"
        )


def upload_to_chromadb(
    embeddings_dir: str, hf_model: str, device: str, version: str, force: str
) -> None:
    metadata_columns: Dict[str, list] = {
        "genes": ["Your Input",
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
                  "Xenopus tropicalis Ortholog"],
        "disease": ["Taxon",
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
                    "Source"],
        "expression":["Species",
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
                      "Source,"
                      "Reference"],
        "molecular_interaction":["ID(s) interactor A",
                                 "ID(s) interactor B",
                                 "Alt. ID(s) interactor A",
                                 "Alt. ID(s) interactor B",
                                 "Alias(es) interactor A",
                                 "Alias(es) interactor B",
                                 "Interaction detection method(s) Publication 1st author(s)",
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
                                 "Host organism(s)"
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
                                 "Identification method participant B"],
        "genetic_interaction":["ID(s) interactor A",
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
                               "Annotation(s) interactor A"
                               "Annotation(s) interactor B"
                               "Interaction annotation(s)"
                               "Host organism(s)"
                               "Interaction parameter(s)"
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
                               "Identification method participant B"],
        "orthology":["Gene1ID Gene1Symbol",
                     "Gene1SpeciesTaxonID",
                     "Gene1SpeciesName",
                     "Gene2ID Gene2Symbol",
                     "Gene2SpeciesTaxonID",
                     "Gene2SpeciesName",
                     "Algorithms",
                     "AlgorithmsMatch",
                     "OutOfAlgorithms",
                     "IsBestScore",
                     "IsBestRevScore"],
        "variants":["Taxon",
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
                    ]}

    csv_dir = "./csv_files/alliance/" + version + "/"

    for filename in os.listdir(csv_dir):
        file_path = os.path.join(csv_dir, filename)

        if os.path.isfile(file_path):
            # Get the basename (filename without path)
            base_name = os.path.basename(file_path)
            print(base_name)

    for filetype, column_names in metadata_columns.items():
        file = csv_dir + filetype + ".tsv"
        if filetype == "genes":
            print(column_names)
            loader = MetaDataCSVLoader(
                    file_path=file,
                    metadata_columns=column_names,
                    encoding="utf-8",
                    csv_args={"delimiter": "\t"}
                )
            docs = loader.load()

            if hf_model is None:  # Use OpenAI
                embeddings = OpenAIEmbeddings()
            elif hf_model.startswith("openai/text-embedding-"):
                embeddings = OpenAIEmbeddings(model=hf_model[len("openai/") :])
            elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
                embeddings = HuggingFaceEndpointEmbeddings(
                    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                    model=hf_model,
                )
            else:
                if device == "cuda":
                    torch.cuda.empty_cache()
                embeddings = HuggingFaceEmbeddings(
                    model_name=hf_model,
                    model_kwargs={"device": device, "trust_remote_code": True},
                    encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
                )

            db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=os.path.join(embeddings_dir, filetype),
            )

        print("filetype")
        print(filetype)


def generate_alliance_embeddings(
    embeddings_dir: str,
    force: bool = False,
    hf_model: str = None,
    device: str = None,
    **kwargs,
) -> None:
    release_version = get_release_version()
    print(f"Release Version: {release_version}")
    if not embeddings_dir.endswith(release_version):
        print("the embeddings dir you gave is:", embeddings_dir, " where the live version of Alliance is ", release_version)
        exit()

    generate_all_csvs(release_version, force)
    upload_to_chromadb(embeddings_dir, hf_model, device, release_version, force)
