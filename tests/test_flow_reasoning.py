import sys
from unittest.mock import MagicMock

# Mock missing modules that are breaking the cross_database import in this environment
if 'langchain.retrievers' not in sys.modules:
    mock_retrievers = MagicMock()
    sys.modules['langchain.retrievers'] = mock_retrievers
    sys.modules['langchain.retrievers.contextual_compression'] = mock_retrievers
    sys.modules['langchain.retrievers.document_compressors'] = mock_retrievers
    sys.modules['langchain.retrievers.merger_retriever'] = mock_retrievers
    sys.modules['langchain.retrievers.self_query'] = mock_retrievers
    sys.modules['langchain.retrievers.self_query.base'] = mock_retrievers

# langchain.chains moved to langchain_core or langchain-community — mock it for the test environment
if 'langchain.chains' not in sys.modules:
    mock_chains = MagicMock()
    sys.modules['langchain.chains'] = mock_chains
    sys.modules['langchain.chains.base'] = mock_chains
    sys.modules['langchain.chains.combine_documents'] = mock_chains
    sys.modules['langchain.chains.retrieval'] = mock_chains

if 'chromadb' not in sys.modules:
    sys.modules['chromadb'] = MagicMock()
    sys.modules['chromadb.config'] = MagicMock()

if 'langchain_chroma' not in sys.modules:
    sys.modules['langchain_chroma'] = MagicMock()
    sys.modules['langchain_chroma.vectorstores'] = MagicMock()

if 'nltk' not in sys.modules:
    sys.modules['nltk'] = MagicMock()
    sys.modules['nltk.tokenize'] = MagicMock()

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from agent.profiles.cross_database import CrossDatabaseGraphBuilder
from agent.profiles.base import BaseState

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def mock_embedding():
    return MagicMock()

@pytest.fixture
def builder(mock_llm, mock_embedding):
    # Patch all chain/tool creation to avoid heavy initialization and TypeErrors
    with patch('agent.profiles.cross_database.create_reactome_rag'), \
         patch('agent.profiles.cross_database.create_uniprot_rag'), \
         patch('agent.profiles.cross_database.create_completeness_grader'), \
         patch('agent.profiles.cross_database.create_reactome_rewriter_w_uniprot'), \
         patch('agent.profiles.cross_database.create_uniprot_rewriter_w_reactome'), \
         patch('agent.profiles.cross_database.create_reactome_uniprot_summarizer'), \
         patch('agent.profiles.cross_database.create_flow_reasoner'), \
         patch('agent.profiles.base.create_rephrase_chain'), \
         patch('agent.profiles.base.create_safety_checker'), \
         patch('agent.profiles.base.create_language_detector'), \
         patch('agent.profiles.base.create_search_workflow'):
        return CrossDatabaseGraphBuilder(mock_llm, mock_embedding)

@pytest.mark.anyio
async def test_identify_flow(builder):
    state = {
        "reactome_answer": "The reaction R-HSA-123456 and R-HSA-789012 are involved."
    }
    
    mock_topology = MagicMock()
    mock_topology.get_flow_context.side_effect = lambda x: f"Context for {x}"
    builder.topology_tool = mock_topology
    
    result = await builder.identify_flow(state, {})
    
    assert "Context for R-HSA-123456" in result["flow_context"]
    assert "Context for R-HSA-789012" in result["flow_context"]
    assert mock_topology.get_flow_context.call_count == 2

@pytest.mark.anyio
async def test_verify_mechanism(builder):
    state = {
        "rephrased_input": "How does it work?",
        "reactome_answer": "Initial answer.",
        "flow_context": "Topological data."
    }
    
    builder.flow_reasoner = AsyncMock()
    builder.flow_reasoner.ainvoke.return_value = "Verified answer."
    
    result = await builder.verify_mechanism(state, {})
    
    assert result["reactome_answer"] == "Verified answer."
    builder.flow_reasoner.ainvoke.assert_called_once()

@pytest.mark.anyio
async def test_decide_next_steps_mechanistic(builder):
    # Case 1: Mechanistic intent
    state = {
        "rephrased_input": "What happens after TLR4 activation?",
        "reactome_answer": "Some answer.",
        "reactome_completeness": "Yes",
        "uniprot_completeness": "Yes"
    }
    decision = await builder.decide_next_steps(state)
    assert decision == "identify_flow"

    # Case 2: Normal intent
    state = {
        "rephrased_input": "What is TLR4?",
        "reactome_answer": "Some answer.",
        "reactome_completeness": "Yes",
        "uniprot_completeness": "Yes"
    }
    decision = await builder.decide_next_steps(state)
    assert decision == "generate_final_response"


def test_flow_context_in_state():
    """
    Regression test: flow_context must be declared in CrossDatabaseState.

    Without this field, LangGraph silently drops the topology data returned
    by identify_flow() before verify_mechanism() can read it — making the
    entire topological reasoning feature a no-op.
    """
    from agent.profiles.cross_database import CrossDatabaseState
    import typing

    hints = typing.get_type_hints(CrossDatabaseState)
    assert "flow_context" in hints, (
        "CrossDatabaseState is missing the 'flow_context' field. "
        "LangGraph will silently drop topology data between nodes without it."
    )
