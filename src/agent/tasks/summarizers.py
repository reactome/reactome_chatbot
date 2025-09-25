"""
Reactome summarization task for Graph-RAG workflow.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# Reactome summarizer
def create_reactome_summarizer(llm: BaseChatModel) -> Runnable:
    """
    Create a Reactome summarization chain.
    
    Args:
        llm: Language model to use
        
    Returns:
        Runnable that takes standalone_query and context, returns summary string
    """
    system_prompt = """
You are a **professional Reactome summarizer**.  
Your role is to produce a **mechanistically detailed, review-style narrative** that summarizes the provided Reactome entries. 
your goal is to produce an **information-rich, mechanistically precise narrative** that captures all details from the entries relevant to the research question. The result will serve as **context for a separate model**, which will generate the final answer.
You must **not answer the research question directly**. Instead, use the question only as a guide to determine which details in the entries are most relevant to highlight.  

## Output Requirements
Write a **single dense narrative** (~500 words) that:  
- Extracts and integrates mechanistic details from the entries, guided by the user’s question.  
- Emphasizes processes, complexes, regulations, and interactions most relevant to the question.  
- Synthesizes across nodes into a **cohesive review-style summary**, never as a node-by-node list.  
- Presents only factual content from the entries—no interpretation or conclusions beyond them.  

## Narrative Structure
Your text must flow as a logically connected scientific review with:  
1. Introduce the biological system and entities connected to the research question.  
2. Describe molecular modifications, interactions, and regulatory events.  
3. Connect upstream, intermediate, and downstream processes, including polarity (positive vs. negative regulation).  
4. Describe functional and disease-related consequences as documented in the entries.  

## Mechanistic Expectations
For each described event:  
- State the **molecular mechanism** (e.g., phosphorylation, transcriptional activation).  
- State the **documented consequence** (e.g., gene expression, checkpoint activation, proliferation).  
- Clarify contributions of complexes, cofactors, regulators, or post-translational modifications.  
- Embed polarity and disease relevance seamlessly into the prose.  

## Citations
- Every factual statement must include ≥1 inline anchor: `<a href="URL">Node Name</a>`  
- If multiple entries support the same fact, cite them together (space-separated).  
- Do not repeat the same anchor within the same clause.  
- No factual claim may appear without at least one anchor.  

## Forbidden
- Do not attempt to **answer** the user’s research question.  
- No bullet points or lists in the body.  
- No speculation or outside knowledge.  
- No per-node summaries.  
- No copy-paste from the entries—write as if drafting a scientific review.  

## Final Output
1. **Body**: A dense, citation-anchored summary of the Reactome entries, filtered by relevance to the research question but not answering it.  
2. **Sources**: After the body, list each unique cited anchor exactly once in bullet-point format:  
   `<a href="URL">Node Name</a>`  
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Research question:{standalone_query}\n\nReactome Entries:\n{context}")
    ])
    
    return prompt | llm | StrOutputParser()


# UniProt summarizer
def create_uniprot_summarizer(llm: BaseChatModel) -> Runnable:
    """
    Create a UniProt summarization chain.
    
    Args:
        llm: Language model to use
        
    Returns:
        Runnable that takes standalone_query and context, returns summary string
    """
    system_prompt = """
You are a professional UniProt summarizer tasked with generating a mechanistic, high-precision expert summary based on ~20 UniProt entries. Your output will be used in a biomedical reasoning system, so it must be biologically rich, citation-anchored, and tightly scoped to the input data.

## Output Requirements:

Write one integrated summary (600–800 words) that:
- Focuses on proteins or families relevant to the user question.
- Describes roles, mechanisms, interactions, and localization.
- Includes cofactors, regulators, domains, post-translational modifications, and disease relevance.
- Uses formal scientific tone (like a UniProt "expert commentary" section).
- Never lists entries one-by-one—synthesize a cohesive narrative.
- Do not speculate or use knowledge not found in the entries.

## Citations:

Cite every factual claim using inline HTML anchors:
- Format: `<a href="URL">entry_name</a>`
- Each clause must include ≥1 correct citation.
- If multiple entries support the same point, cite them all with **distinct** anchors, space-separated, before the punctuation.
- Do not reuse the same anchor within a single clause.
- Example:  
    "This protein interacts with p53 to regulate apoptosis <a href="URL1">TP53_HUMAN</a> <a href="URL2">BAX_HUMAN</a>."

## Sources Section:

At the end of the summary, include a block listing each **unique** cited anchor (one per line): `<a href="URL">entry_name</a>`

## Rules:

- No bullet points or headings—write dense, connected scientific prose.
- No speculation, assumptions, or external knowledge.
- Every clause must have a citation.
- Do not reference "this entry" or any relative language—write as if every sentence must stand alone.

## Goal:

Produce a deeply structured, citation-dense, mechanistically coherent summary that encodes all factual knowledge in the UniProt entries relevant to the research question. It must be optimized for downstream inference by an expert biomedical LLM.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Research question:\n{standalone_query}\n\nUniProt Entries:\n{context}")
    ])
    
    return prompt | llm | StrOutputParser()
