from typing import Dict, List

from langchain.chains.query_constructor.base import AttributeInfo

pathway_id_description = (
    "A Reactome Identifier unique to each pathway. A pathway name may appear multiple times in the dataset\
    This ID allows for the specific identification and exploration of each pathway's details within the Reactome Database.",
)
pathway_name_description = "The name of the biological pathway, indicating a specific series of interactions or processes within a cell.\
                A pathway name may appear multiple times in the dataset, reflecting the fact that several reactions (identified by 'reaction_name') contribute to a single pathway.\
                The relationship between 'reaction_name' and 'pathway_name' is foundational, with each reaction serving as a step or component within the overarching pathway, contributing to its completion and functional outcome.\
                This relationship is critical to understanding the biological processes and mechanisms within the Reactome Database."

descriptions_info: dict[str, str] = {
    "ewas": "Contains data on proteins and nucleic acids with known sequences. Includes entity names, IDs, canonical and synonymous gene names, and functions.",
    "complexes": "Catalogs biological complexes, listing complex names and IDs along with the names and IDs of their components. ",
    "reactions": "Documents biological pathways and their constituent reactions, detailing pathway and reaction names and IDs. It includes information on the inputs, outputs, and catalysts for each reaction, emphasizing the interconnected nature of cellular processes. Inputs and outputs, critical to the initiation and conclusion of reactions, along with catalysts that facilitate these processes, are cataloged to highlight their roles across various reactions and pathways",
    "summations": "Enumerates biological reactions, accompanied by concise summaries ('summations') of each reaction. These summations encapsulate the essence and biochemical significance of the reactions, offering insights into their roles within cellular processes and pathways.",
}


field_info: Dict[str : List[AttributeInfo]] = {
    "summations": [
        AttributeInfo(
            name="st_id",
            description=pathway_id_description,
            type="string",
        ),
        AttributeInfo(
            name="display_name",
            description=pathway_name_description,
            type="string",
        ),
        AttributeInfo(name="summation", description="The descriptions of the pathway"),
    ],
    "reactions": [
        AttributeInfo(
            name="st_id",
            description="The Reactome Identifier (ID) for each biological reaction, serving as a unique key.\
                This ID allows for the specific identification and exploration of each reaction's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="display_name",
            description="The name of the biological reaction, encapsulating the interaction between proteins or molecules.\
                Each reaction name is a unique entry, reflecting a specific biological process.\
                These names provide insight into the dynamic processes within cellular functions, highlighting the roles of various proteins and molecules in biological mechanisms",
            type="string",
        ),
        AttributeInfo(
            name="pathway_id",
            description=pathway_id_description,
            type="string",
        ),
        AttributeInfo(
            name="pathway_name",
            description=pathway_name_description,
            type="string",
        ),
        AttributeInfo(
            name="input_id",
            description="The Reactome Identifier (ID) for each input.\
                Given that a single input can be involved in various reactionss, this ID may repeat across multiple rows, each associated with a different reaction.\
                This ID allows for the specific identification and exploration of each input's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="input_name",
            description="Identifies the inputs of a biological reaction ('reaction_name'), which can be either entities or part of complexes.\
                Inputs are crucial for initiating reactions, acting as the reactants that drive the biochemical processes. \
                Given their fundamental role, inputs may repeat across multiple reactions, reflecting their involvement in various parts of the cellular machinery.",
            type="string",
        ),
        AttributeInfo(
            name="output_id",
            description=" A Reactome Identifier unique to each output of a reaction.\
                Given that a single input can be involved in various reactionss, this ID may repeat across multiple rows, each associated with a different reaction.\
                This ID allows for the specific identification and exploration of each output's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="output_name",
            description="Represents the outputs of a biological reaction ('reaction_name'), denoting the products generated as a result of the biochemical interactions. \
                Outputs can be entities or complexes and may appear in multiple reactions, highlighting their multifunctional role in cellular pathways. \
                This repetition underscores the interconnected nature of biological processes, where one reaction's output can serve as another's input.",
            type="string",
        ),
        AttributeInfo(
            name="catalyst_id",
            description="The Reactome Identifier (ID) for each biological catalyst, serving as a unique key.\
                Given that a single catalyst can be involved in various reactions, this ID may repeat across multiple rows, each associated with a different reaction.\
                This ID allows for the specific identification and exploration of each catalyst's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="catalyst_name",
            description="Specifies the catalysts that facilitate a biological reaction, potentially speeding up the process without being consumed.\
                Catalysts are crucial for modulating reaction rates and guiding the direction of the reaction, ensuring the efficient progression of biological pathways.\
                Catalysts can be proteins, enzymes, or molecular compounds, underscoring their vital role in cellular operations.",
            type="string",
        ),
    ],
    "complexes": [
        AttributeInfo(
            name="st_id",
            description="The Reactome Identifier (ID) for each biological complex, serving as a unique key.\
                Given that a single complex can consist of various components, this ID may repeat across multiple rows, each associated with a different component of the same complex.",
            type="string",
        ),
        AttributeInfo(
            name="display_name",
            description="The name of the biological complex.\
                  This field provides a reference to the complex itself, which may be listed across several rows to account for its multiple components.",
            type="string",
        ),
        AttributeInfo(
            name="component_id",
            description=" A Reactome Identifier unique to each component within a complex.\
                  This ID allows for the specific identification and exploration of each component's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="component_name",
            description="The name of the individual component associated with the complex in that row.\
                  This reveals the specific protein or molecule constituting part of the complex, emphasizing the diversity of components within a single biological entity.",
            type="string",
        ),
    ],
    "ewas": [
        AttributeInfo(
            name="st_id",
            description="The Reactome Identifier (ID) for each biological complex, serving as a unique key.\
                Given that a single complex can consist of various components, this ID may repeat across multiple rows, each associated with a different component of the same complex.",
            type="string",
        ),
        AttributeInfo(
            name="display_name",
            description="The name of the biological complex.\
                  This field provides a reference to the complex itself, which may be listed across several rows to account for its multiple components.",
            type="string",
        ),
        AttributeInfo(
            name="canonical_geneName",
            description=" A Reactome Identifier unique to each component within a complex.\
                  This ID allows for the specific identification and exploration of each component's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="synonyms_geneName",
            description="The name of the individual component associated with the complex in that row.\
                  This reveals the specific protein or molecule constituting part of the complex, emphasizing the diversity of components within a single biological entity.",
            type="string",
        ),
    ],
}
