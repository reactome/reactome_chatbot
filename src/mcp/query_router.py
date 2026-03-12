from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

ROUTE_RAG = "rag"
ROUTE_MCP_SEARCH = "mcp_search"
ROUTE_MCP_ANALYSIS = "mcp_analysis"

ROUTER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a query router for a Reactome biological pathway chatbot.

Classify the question into exactly one of these categories:

rag          - General knowledge question about biological pathways, proteins,
               or genes answerable from a static knowledge base.
               Example: "What does TP53 do?", "Explain MAPK signalling"

mcp_search   - Requires live Reactome database lookup by specific identifier,
               name, species list, or database metadata.
               Example: "Show pathway R-HSA-109582", "What species does Reactome cover?"

mcp_analysis - Involves a specific list of gene or protein identifiers
               needing pathway enrichment analysis.
               Example: "Analyze these genes: TP53, BRCA1, EGFR"

Rules:
- mcp_analysis requires both: (1) explicit analysis/enrichment intent AND 
  (2) a list of identifiers to analyze. A list alone is not enough.
- mcp_search is for retrieving specific known entities by ID or name,
  or querying database metadata like species or version.
- If a question combines multiple intents, identify the primary action 
  the user wants performed and route based on that.
- If genuinely unclear, return rag - it is always the safest fallback.

Return only the category name. Nothing else.

Question: {question}""",
)


def create_query_router(llm: BaseChatModel):
    """
    Returns an async routing function that classifies a user question into
    one of three routes: rag, mcp_search, or mcp_analysis.
    Intended to be used with a lightweight model like gpt-4o-mini.
    """
    llm_chain = ROUTER_PROMPT | llm | StrOutputParser()

    async def route(question: str) -> str:
        result = await llm_chain.ainvoke({"question": question})
        result = result.strip().lower()
        # fall back to rag if LLM returns unexpected output
        if result not in (ROUTE_RAG, ROUTE_MCP_SEARCH, ROUTE_MCP_ANALYSIS):
            return ROUTE_RAG
        return result

    return route