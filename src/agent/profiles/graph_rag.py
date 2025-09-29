from typing import Any, Literal, Annotated

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.final_answer_generators import (
    create_expert_answer_generator,
    create_lay_answer_generator,
    create_unsafe_answer_generator
)
from agent.tasks.completeness_grader import (
    CompletenessGrade,
    create_completeness_grader,
)
from agent.tasks.query_expansion import create_query_expander
from agent.tasks.summarizers import create_reactome_summarizer, create_uniprot_summarizer
from agent.tasks.safety_checker import SafetyCheck, create_safety_checker
from agent.tasks.tone_language import (
    ToneLanguageParser,
    create_tone_language_detector,
)
from retrievers.graph_rag.config import (
    create_reactome_retriever_custom, 
    create_uniprot_retriever_custom,
)
from retrievers.graph_rag.retrieval_utils import (
    ReactomeRetrievalConfig, 
    UniProtRetrievalConfig, 
    VectorSearchConfig, 
    GraphTraversalConfig
)

def merge_contexts(left: str, right: str) -> str:
    """Reducer function to merge context strings from parallel searches."""
    if not left and not right:
        return ""
    elif not left:
        return right
    elif not right:
        return left
    else:
        # Both contexts exist, return the non-empty one (they should be the same)
        return left if left else right

class GraphRAGState(BaseState):
    safety: str                     # LLM-assessed safety level of the user input
    reason_unsafe: str              # Reason for unsafe question
    query_language: str             # language of the user input
    user_tone: str                  # Detected user expertise level (lay/expert)
    complexity: str                 # Query complexity assessment (simple/complex)
    expanded_queries: list[str]     # Expanded queries for complex searches
    reactome_context: Annotated[str, merge_contexts]  
    uniprot_context: Annotated[str, merge_contexts]   
    final_complete: str             # LLM-assessed completeness of the final answer
    did_fallback: str

class GraphRAGGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        super().__init__(llm, embedding)
 
        self.safety_checker = create_safety_checker(llm)
        self.tone_language_detector = create_tone_language_detector(llm)
        self.query_expander = create_query_expander(llm)
        self.completeness_checker = create_completeness_grader(llm)
        self.reactome_summarizer = create_reactome_summarizer(llm)
        self.uniprot_summarizer = create_uniprot_summarizer(llm)
        
        # Simple mode: vector-only search
        simple_vector_config = VectorSearchConfig(
            use_rrf=True,
            rrf_final_k=5,
            rrf_per_query_k=20,
            rrf_lambda=60.0,
            rrf_alpha=0.8
        )
        self.reactome_simple_config = ReactomeRetrievalConfig(
            vector_config=simple_vector_config,
            graph_config=None
        )
        
        # Complex mode: vector + graph traversal
        complex_vector_config = VectorSearchConfig(
            use_rrf=True,
            rrf_final_k=7,
            rrf_per_query_k=20,
            rrf_lambda=60.0,
            rrf_alpha=0.8
        )
        complex_graph_config = GraphTraversalConfig(
            strategy_sequence=["steiner_tree", "one_hop"],
            max_neighbors_per_type=2,
            max_total=5,
            source_id=None,
            gds_graph_name=None
        )
        self.reactome_complex_config = ReactomeRetrievalConfig(
            vector_config=complex_vector_config,
            graph_config=complex_graph_config
        )
        
        # UniProt configuration
        self.uniprot_config = UniProtRetrievalConfig(
            vector_config=VectorSearchConfig(
                use_rrf=True,
                rrf_final_k=10,
                rrf_per_query_k=20,
                rrf_lambda=60.0,
                rrf_alpha=0.8
            )
        )
        
        self.reactome_retriever, _ = create_reactome_retriever_custom()
        self.uniprot_retriever, _ = create_uniprot_retriever_custom(embedding, self.uniprot_config)
        
        streaming_llm = llm.model_copy(update={"streaming": True})
        self.lay_answer_generator = create_lay_answer_generator(streaming_llm)
        self.expert_answer_generator = create_expert_answer_generator(streaming_llm)
        self.unsafe_answer_generator = create_unsafe_answer_generator(streaming_llm)

        state_graph = StateGraph(GraphRAGState)
        state_graph.add_node("check_question_safety", self.check_question_safety)
        state_graph.add_node("preprocess_question", self.preprocess)
        state_graph.add_node("identify_query_language", self.identify_query_language)
        state_graph.add_node("conduct_research", self.conduct_research)
        state_graph.add_node("query_expansion", self.query_expansion)
        state_graph.add_node("reactome_search", self.reactome_search)
        state_graph.add_node("uniprot_search", self.uniprot_search)
        state_graph.add_node("assess_completeness", self.assess_completeness)
        state_graph.add_node("generate_final_response", self.generate_final_response)
        state_graph.add_node("postprocess", self.postprocess)
        
        # Set up edges
        state_graph.set_entry_point("preprocess_question")
        state_graph.add_edge("preprocess_question", "check_question_safety")
        state_graph.add_edge("preprocess_question", "identify_query_language")
        state_graph.add_edge("preprocess_question", "query_expansion")
        state_graph.add_conditional_edges(
            "check_question_safety",
            self.proceed_with_research,
            {"Continue": "conduct_research", "Finish": "generate_final_response"},
        )

        # Start both searches in parallel
        state_graph.add_edge("conduct_research", "reactome_search")
        state_graph.add_edge("conduct_research", "uniprot_search")
        
        # Both searches go directly to generate_final_response with state merging
        state_graph.add_edge("reactome_search", "generate_final_response")
        state_graph.add_edge("uniprot_search", "generate_final_response")

        state_graph.add_edge("conduct_research", "assess_completeness")

        # Run generate_final_response and assess_completeness in parallel
        state_graph.add_edge("generate_final_response", "postprocess")
        state_graph.add_edge("assess_completeness", "postprocess")
        
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

    async def preprocess(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        """Override preprocess to add logging for rephrased input."""
        result = await super().preprocess(state, config)
        
        return result

    async def check_question_safety(
        self, state: GraphRAGState, config: RunnableConfig
    ) -> GraphRAGState:
        result: SafetyCheck = await self.safety_checker.ainvoke(
            {"rephrased_input": state["rephrased_input"]},
            config,
        )
        print(f"🛡️ OG graph_rag SAFETY CHECK: {result.safety}")
        
        if result.safety == "false":
            return GraphRAGState(
                safety=result.safety,
                reason_unsafe=result.reason_unsafe,
                reactome_context="",
                uniprot_context="",
            )
        else:
            return GraphRAGState(safety=result.safety, reason_unsafe=result.reason_unsafe)
        
    async def proceed_with_research(
        self, state: GraphRAGState
    ) -> Literal["Continue", "Finish"]:
        if state["safety"] == "true":
            return "Continue"
        else:
            return "Finish"
    
    async def conduct_research(
        self, state: GraphRAGState, config: RunnableConfig
    ) -> GraphRAGState:
        return GraphRAGState()

    async def identify_query_language(
        self, state: GraphRAGState, config: RunnableConfig
    ) -> GraphRAGState:
        result: ToneLanguageParser = await self.tone_language_detector.ainvoke(
            {"user_input": state["user_input"]}, config
        )
        
        return GraphRAGState(
            query_language=result.language,
            user_tone=result.user_tone,
            complexity=result.complexity,
        )

    async def query_expansion(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        expanded_queries: list[str] = await self.query_expander.ainvoke(
            {"rephrased_input": state["rephrased_input"]}, config
        )
        print(f"🔍 OG graph_rag QUERY EXPANSION: {expanded_queries}")
        
        return GraphRAGState(expanded_queries=expanded_queries)
    
    async def reactome_search(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        """Perform Reactome search using simple or complex mode based on query complexity."""
        complexity = state.get("complexity", "simple")
        expanded_queries = state.get("expanded_queries", [])
        try:
            # Complex search
            if complexity == "complex":
                retrieval_config = self.reactome_complex_config
                context_str: str = await self.reactome_retriever.ainvoke(
                    query=state["rephrased_input"], 
                    cfg=retrieval_config, 
                    expanded_queries=expanded_queries, 
                    output_format="llm_text"
                )            
                # Summarize the context
                reactome_context: str = await self.reactome_summarizer.ainvoke(
                    {"standalone_query": state["rephrased_input"], "context": context_str}, config
                )
                return GraphRAGState(reactome_context=reactome_context)
            else:
                # Use vector-only search for simple queries
                context_str: str = await self.reactome_retriever.ainvoke(
                    query=state["rephrased_input"], 
                    cfg=self.reactome_simple_config, 
                    expanded_queries=expanded_queries, 
                    output_format="llm_text"
                )
                return GraphRAGState(reactome_context=context_str)
            
        except Exception as e:
            return GraphRAGState(reactome_context="")

    async def uniprot_search(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        """Perform UniProt search using vector retrieval."""
        complexity = state.get("complexity", "simple")
        expanded_queries = state.get("expanded_queries", [])
        
        try:
            uniprot_context_list = await self.uniprot_retriever.ainvoke(
                query=state["rephrased_input"],
                cfg=self.uniprot_config,
                expanded_queries=expanded_queries,
            )
            
            uniprot_context_str = "\n\n".join(uniprot_context_list) if uniprot_context_list else ""
            
            # Summarize for complex queries
            if complexity == "complex":
                uniprot_context = await self.uniprot_summarizer.ainvoke(
                    {"standalone_query": state["rephrased_input"], "context": uniprot_context_str}, 
                    config
                )
            else:
                uniprot_context = uniprot_context_str
            
            return GraphRAGState(uniprot_context=uniprot_context)
        except Exception as e:
            return GraphRAGState(uniprot_context="")

    async def generate_final_response(
        self, state: GraphRAGState, config: RunnableConfig
    ) -> GraphRAGState:
        """Generate final response based on safety, user tone, and available data."""
        reactome_context = state.get("reactome_context", "")
        uniprot_context = state.get("uniprot_context", "")
        reason_unsafe = state.get("reason_unsafe", "")
        safety = state.get("safety", "true")

        if safety == "false" and reason_unsafe:
            final_answer: str = await self.unsafe_answer_generator.ainvoke({
                "language": state.get("query_language", "English"),
                "user_input": state["rephrased_input"],
                "reason_unsafe": reason_unsafe,
            }, config)
        else:
            user_tone = state.get("user_tone", "lay")
            answer_generator = (self.expert_answer_generator if user_tone == "expert" 
                              else self.lay_answer_generator)
            
            final_answer: str = await answer_generator.ainvoke({
                "language": state.get("query_language", "English"),
                "user_input": state["rephrased_input"],
                "reactome_context": reactome_context,
                "uniprot_context": uniprot_context,
            }, config)
        
        return GraphRAGState(
            chat_history=[
                HumanMessage(state["rephrased_input"]),
                AIMessage(final_answer),
            ],
            answer=final_answer,
        )

    async def assess_completeness(
        self, state: GraphRAGState, config: RunnableConfig
    ) -> GraphRAGState:
        query = state.get("rephrased_input")
        generation = state.get("answer", "") or ""
        
        if not generation:
            return GraphRAGState(final_complete="Yes")
        
        completeness_async = self.completeness_checker.ainvoke(
            {"input": query, "generation": generation},
            config,
        )
        completeness: CompletenessGrade = await completeness_async
        
        return GraphRAGState(final_complete=completeness.binary_score)

    async def postprocess(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        """Override postprocess to handle external search for GraphRAG profile."""
        from tools.external_search.state import SearchState, WebSearchResult
        from tools.external_search.workflow import create_search_workflow
        
        search_results: list[WebSearchResult] = []
        final_complete = state.get("final_complete", "Yes")
        
        if config["configurable"]["enable_postprocess"] and final_complete == "No":
            search_workflow = self.search_workflow
            result: SearchState = await search_workflow.ainvoke(
                SearchState(
                    input=state["rephrased_input"],
                    generation=state["answer"],
                ),
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]
            return GraphRAGState(
                additional_content={"search_results": search_results},
                did_fallback="Yes"
            )
        else:
            return GraphRAGState(
                additional_content={"search_results": search_results},
                did_fallback="No"
            )

def create_graph_rag_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return GraphRAGGraphBuilder(llm, embedding).uncompiled_graph
