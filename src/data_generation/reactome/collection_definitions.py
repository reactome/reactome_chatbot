"""Where every column of every Reactome source goes, in one place.

Three collections replace the five the bundle used to have. Each is fed by
one or more of the queries in `neo4j_connector.SOURCE_QUERIES`, and each
column a query returns goes to one or more of three places:

- **embedded**: the document's text. Keyword and vector search match on it,
  and it is what the model reads. Names and prose: what people type and what
  the model reasons over.
- **metadata**: read by code only -- de-duplication, citations, balancing.
  Every identifier and every link goes here. An identifier in the text is
  noise in every vector and costs tokens in every document.
- **shown to the model**: metadata the answer chain adds to each retrieved
  document without it being searched. The `url`, so the model can write a
  link, and the UniProt accession, so it can quote one.

`display_name` and `record_type` are both embedded and metadata: the model
reads them, and citations and balancing use them.

Embedded columns are listed in the order they appear in the text: identity
first, then structure and disease context, then the summation last, so the
model reads what a record is before the long prose about it.

Generation stops if a query returns a column that is not placed here, or if a
column placed here is returned by no query, so neither can drift silently.
"""

from dataclasses import dataclass

#: The column added to every row as the CSVs are written, saying what kind of
#: record it is. Embedded, so the model can tell a defective reaction from a
#: normal one.
RECORD_TYPE = "record_type"


@dataclass(frozen=True)
class Source:
    """One query, and the label every row it returns is given."""

    name: str
    record_type: str


@dataclass(frozen=True)
class Collection:
    """One vector store: the sources that feed it and where each column goes."""

    name: str
    sources: tuple[Source, ...]
    embedded: tuple[str, ...]
    metadata: tuple[str, ...]
    shown_to_model: tuple[str, ...]
    #: What the router is told the collection holds and when to choose it.
    #: Written into the manifest, so the router describes the bundle that
    #: is installed rather than the one the code was last written for.
    description: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column, in CSV order: embedded ones first, in text order.

        The loader renders a document's text in the CSV's column order, so the
        embedded columns must lead in the order they should be read.
        """
        rest = tuple(c for c in self.metadata if c not in self.embedded)
        return self.embedded + rest


PATHWAYS = Collection(
    name="pathways",
    sources=(
        Source("normal_pathways", "normal pathway"),
        Source("disease_pathways", "disease pathway"),
    ),
    embedded=(
        "display_name",
        RECORD_TYPE,
        "disease",
        "disrupted_in_diseases",
        "parent_pathway_names",
        "child_event_names",
        "preceding_event_names",
        "normal_pathway_names",
        "disease_pathway_names",
        "summation",
    ),
    metadata=(
        "st_id",
        "display_name",
        RECORD_TYPE,
        "url",
        "parent_pathway_ids",
        "child_event_ids",
        "preceding_event_ids",
        "normal_pathway_ids",
        "disease_pathway_ids",
    ),
    shown_to_model=("url",),
    description=(
        "Reactome pathways, normal and disease, one record each: the "
        "pathway's curated summary, its parent pathways and first few "
        "sub-events, the events before it, and its disease or normal "
        "counterpart with the diseases involved. Choose it for what a pathway "
        "is or does, what it contains, or where it sits in the hierarchy."
    ),
)

REACTIONS = Collection(
    name="reactions",
    sources=(
        Source("normal_reactions", "normal reaction"),
        Source("disease_reactions", "disease reaction"),
    ),
    embedded=(
        "display_name",
        RECORD_TYPE,
        "cellular_compartment",
        "input_names",
        "output_names",
        "catalyst_names",
        "regulated_by",
        "disease",
        "failure_reason",
        "normal_reaction_names",
        "disease_reaction_names",
        "disrupted_in_diseases",
        "parent_pathway_names",
        "preceding_event_names",
        "following_event_names",
        "summation",
    ),
    metadata=(
        "st_id",
        "display_name",
        RECORD_TYPE,
        "url",
        "input_ids",
        "output_ids",
        "catalyst_ids",
        "normal_reaction_ids",
        "disease_reaction_ids",
        "parent_pathway_ids",
        "preceding_event_ids",
        "following_event_ids",
    ),
    shown_to_model=("url",),
    description=(
        "Reactome reactions, normal and disease, one record each: inputs, "
        "outputs, catalysts and regulators, compartment, the pathways it "
        "belongs to, the events before and after it, its curated summary, "
        "and for a disease reaction the disease, how it fails and its normal "
        "counterpart. Choose it for what a reaction consumes, produces, needs "
        "or is controlled by, or what happens before or after it."
    ),
)

ENTITIES = Collection(
    name="entities",
    sources=(
        # Generic until the complex queries say which a row is: both return
        # complexes and entity sets, and nothing in a row tells them apart.
        Source("normal_complexes", "normal complex or set"),
        Source("disease_complexes", "disease complex or set"),
        Source("ewas", "protein"),
        Source("diseases", "disease term"),
    ),
    embedded=(
        "display_name",
        RECORD_TYPE,
        "gene_names",
        "synonyms",
        "disease",
        "failure_reason",
        "disrupted_in_diseases",
        "component_names",
        "member_names",
        "candidate_names",
        "normal_entity_names",
        "disease_entity_names",
        "summation",
    ),
    metadata=(
        "st_id",
        "protein_id",
        "disease_id",
        "display_name",
        RECORD_TYPE,
        "url",
        "component_ids",
        "member_ids",
        "candidate_ids",
        "normal_entity_ids",
        "disease_entity_ids",
    ),
    shown_to_model=("url", "protein_id"),
    description=(
        "Molecules and diseases, one record each. Complexes and sets with "
        "their components, members and candidates. Human proteins with their "
        "gene names, UniProt accession and function text, and the disease "
        "variants Reactome curates for them. Disease terms with their "
        "synonyms and Disease Ontology definition. Choose it for a protein's "
        "accession, gene names or function; a gene's variants and the "
        "diseases they cause; what a complex or set is made of; or what a "
        "disease is."
    ),
)

COLLECTIONS: tuple[Collection, ...] = (PATHWAYS, REACTIONS, ENTITIES)
