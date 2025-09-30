"""
Reactome summarization task for Graph-RAG workflow.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

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
You must **not answer the research question directly**. Instead, use the question only as a guide to determine which details in the entries are most relevant to highlight.  

## Output Requirements
Write a **single dense narrative** (~500 words) that:  
- Extracts and integrates mechanistic details from the entries, guided by the user’s question.  
- Emphasizes processes, complexes, regulations, and interactions most relevant to the question.  
- Synthesizes across nodes into a **cohesive review-style summary**, never as a node-by-node list.  
- Presents only factual content from the entries—no interpretation or conclusions beyond them.  

## Narrative and Mechanistic Expectations
Your summary must read as a cohesive scientific review that integrates mechanistic depth into a clear narrative flow. Structure the text so that it:  
1. **Context/Setup**: Introduces the biological system and entities relevant to the research question.  
2. **Mechanistic Detail**: Describes molecular events such as modifications, interactions, regulations, and assembly of complexes, always specifying:  
   - the **mechanism** (e.g., phosphorylation, transcriptional activation),  
   - the **documented consequence** (e.g., expression changes, checkpoint activation, proliferation),  
   - contributions of cofactors, regulators, or post-translational modifications.  
3. **Pathway Integration**: Connects upstream, intermediate, and downstream processes into a continuous chain of events, embedding regulatory polarity (positive vs. negative) and disease associations naturally into the prose.  
4. **Outcome**: Concludes with functional or disease-relevant consequences as documented in the entries.  

## Citations
- Every factual statement must include ≥1 inline anchor: `<a href="URL">Node Name</a>`  
- If multiple entries support the same fact, cite them together (space-separated).  
- Do not repeat the same anchor within the same clause.  
- No factual claim may appear without at least one anchor.  

## Forbidden
- Answering the research question.  
- Bullet points or lists in the body.  
- Copy-paste from entries.  
- Casual tone, speculation, or outside knowledge.  

## Final Output
1. **Body**: A dense, citation-anchored summary filtered by relevance to the question but not answering it.  
2. **Sources**: At the end, list each unique cited anchor once, as a bullet point:  
   `<a href="URL">Node Name</a>`
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Research question:{standalone_query}\n\nReactome Entries:\n{context}")
    ])
    
    return prompt | llm | StrOutputParser()