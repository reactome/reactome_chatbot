from langchain.chains.query_constructor.base import AttributeInfo

uniprot_descriptions_info = {
    "binding_sites": "Contains detailed information on protein-ligand binding interactions, crucial for understanding biochemical pathways. Each record in the file provides a comprehensive view of the binding site characteristics of a protein with its associated ligand.",
    "domain": "Contains detailed infromation about protein domains, detailing the specific structural or functional segments within proteins that are critical to their roles in cellular processes. Each entry in the dataset identifies a domain within a particular protein",
    "motif": "Contains specific motifs found within proteins, which are . These motifs, which are short, conserved regions within a protein sequence, play key roles in determining the structure, function, and interactions of the protein. Each entry provides detailed information about a particular motif.",
    "uniprot_data": "Contains detailed protein information about gene names, protein names, subcellular localizations, family classifications, active sites, domains, post-translational modifications, and functional descriptions. ",
}

uniprot_field_info: dict[str, list[AttributeInfo]] = {
    "binding_sites": [
        AttributeInfo(
            name="citation",
            description="Contains a URL link to the corresponding UniProt entry for each protein. This link serves as a direct reference to a comprehensive database entry, providing additional details and validated information about the protein, facilitating further research and verification.",
            type="string",
        ),
        AttributeInfo(
            name="organism",
            description="Specifies the organism from which the protein is derived, indicating its biological source. This field is essential for comparative biological studies and understanding species-specific protein functions and adaptations.",
            type="string",
        ),
        AttributeInfo(
            name="canonical_gene_name",
            description="Lists the official gene symbol that is recommended for use and standardized across scientific databases.",
            type="string",
        ),
        AttributeInfo(
            name="short_protein_name",
            description="A brief, commonly used abbreviation or symbol for the protein. This name often facilitates quick identification within the scientific community.",
            type="string",
        ),
        AttributeInfo(
            name="full_protein_name",
            description="The complete descriptive name of the protein. This full name provides deeper insights into the protein's biological role and functional characteristics.",
            type="string",
        ),
    ],
    "domain": [
        AttributeInfo(
            name="citation",
            description="Contains a URL link to the corresponding UniProt entry for each protein. This link serves as a direct reference to a comprehensive database entry, providing additional details and validated information about the protein, facilitating further research and verification.",
            type="string",
        ),
        AttributeInfo(
            name="organism",
            description="Specifies the organism from which the protein is derived, indicating its biological source. This field is essential for comparative biological studies and understanding species-specific protein functions and adaptations.",
            type="string",
        ),
        AttributeInfo(
            name="canonical_gene_name",
            description="Lists the official gene symbol that is recommended for use and standardized across scientific databases.",
            type="string",
        ),
        AttributeInfo(
            name="short_protein_name",
            description="A brief, commonly used abbreviation or symbol for the protein. This name often facilitates quick identification within the scientific community.",
            type="string",
    ),
        AttributeInfo(
            name="full_protein_name",
            description="The complete descriptive name of the protein. This full name provides deeper insights into the protein's biological role and functional characteristics.",
            type="string",
    ),
        AttributeInfo(
            name="domain_name",
            description="Identifies the specific name of a protein domain, highlighting distinct structural or functional regions crucial for the protein's activity and interactions",
            type="string",
        ),
    ],
    "motif": [
        AttributeInfo(
            name="citation",
            description="Contains a URL link to the corresponding UniProt entry for each protein. This link serves as a direct reference to a comprehensive database entry, providing additional details and validated information about the protein, facilitating further research and verification.",
            type="string",
        ),
        AttributeInfo(
            name="organism",
            description="Specifies the organism from which the protein is derived, indicating its biological source. This field is essential for comparative biological studies and understanding species-specific protein functions and adaptations.",
            type="string",
        ),
        AttributeInfo(
            name="canonical_gene_name",
            description="Lists the official gene symbol that is recommended for use and standardized across scientific databases.",
            type="string",
        ),
        AttributeInfo(
            name="short_protein_name",
            description="A brief, commonly used abbreviation or symbol for the protein. This name often facilitates quick identification within the scientific community.",
            type="string",
        ),
        AttributeInfo(
            name="full_protein_name",
            description="The complete descriptive name of the protein. This full name provides deeper insights into the protein's biological role and functional characteristics.",
            type="string",
        ),
        AttributeInfo(
            name="motif_name",
            description="Identifies the specific motif within the protein, highlighting its unique structural or functional role. Motifs are critical for understanding protein interactions, regulatory mechanisms, and evolutionary conservation.",
            type="string",
        ),
    ],
    "uniprot_data": [
        AttributeInfo(
            name="citation",
            description="Contains a URL link to the corresponding UniProt entry for each protein. This link serves as a direct reference to a comprehensive database entry, providing additional details and validated information about the protein, facilitating further research and verification.",
            type="string",
        ),
        AttributeInfo(
            name="organism",
            description="Specifies the organism from which the protein is derived, indicating its biological source. This field is essential for comparative biological studies and understanding species-specific protein functions and adaptations.",
            type="string",
        ),
        AttributeInfo(
            name="canonical_gene_name",
            description="Lists the official gene symbol that is recommended for use and standardized across scientific databases.",
            type="string",
        ),
        AttributeInfo(
            name="synonym_gene_names",
            description="Includes all alternative gene names used historically or in various scientific contexts, excluding the official gene symbol.",
            type="string",
        ),
        AttributeInfo(
            name="short_protein_name",
            description="A brief, commonly used abbreviation or symbol for the protein. This name often facilitates quick identification within the scientific community.",
            type="string",
        ),
        AttributeInfo(
            name="full_protein_name",
            description="The complete descriptive name of the protein. This full name provides deeper insights into the protein's biological role and functional characteristics.",
            type="string",
        ),
        AttributeInfo(
            name="protein_family",
            description="Categorizes the protein into a family, grouping it with proteins that share similar structural features or functions.",
            type="string",
        ),
    ],
    
}