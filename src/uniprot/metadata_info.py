from langchain.chains.query_constructor.base import AttributeInfo

uniprot_descriptions_info = {
    "uniprot_data": "Contains detailed protein information about gene names, protein names, subcellular localizations, family classifications, biological pathway associations, domains, motifs, disease associations, and functional descriptions. ",
}
uniprot_field_info: dict[str, list[AttributeInfo]] = {
    "uniprot_data": [
        AttributeInfo(
            name="gene_names",
            description="The official gene name(s) associated with the protein. Gene names may include primary and alternative names \
                used in different research contexts or species-specific databases.",
            type="string",
        ),
        AttributeInfo(
            name="short_protein_name",
            description="The short, standardized name for the protein entry, often derived from its gene name or commonly used abbreviation. \
                This provides a concise reference to the protein.",
            type="string",
        ),
        AttributeInfo(
            name="full_protein_name",
            description="The complete and descriptive name of the protein, detailing its function, structure, or significant features. \
                This name is derived from biological literature and protein function annotations.",
            type="string",
        ),
        AttributeInfo(
            name="protein_family",
            description="The family or group of related proteins to which this protein belongs, based on sequence similarity, \
                structural features, or shared functional characteristics.",
            type="string",
        ),
        AttributeInfo(
            name="biological_pathways",
            description="The biological pathways in which the protein is involved, as curated from databases like Reactome or KEGG. \
                This provides insights into the protein's role in metabolic, signaling, or regulatory networks.",
            type="string",
        ),
    ]
}
