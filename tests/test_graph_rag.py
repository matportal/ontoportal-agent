import importlib
import pytest
from unittest.mock import patch, MagicMock

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.rag_client import RagClient, RagChunk, RagResult
from ontoportal_agent.config import get_settings, AgentSettings
from ontoportal_agent.agent.graph import build_agent_graph
from ontoportal_agent.agent.options import AgentRuntimeOptions


@pytest.fixture(autouse=True)
def setup_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()


# ==========================================
# 1. Test RagClient.graph_query
# ==========================================

@patch("ontoportal_agent.rag_client.requests.post")
def test_rag_client_graph_query(mock_post):
    # Mock API response for graph-query
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "answer": "Graph response",
        "sources": [
            {
                "ontology_id": "CHMO",
                "version": "2",
                "content": "Graph facts about class",
                "metadata": {"type": "owl_class"}
            }
        ]
    }
    mock_post.return_value = mock_response

    client = RagClient(base_url="http://mock-rag:8000")
    result = client.graph_query("What is class X?", top_k=5, ontology_id="CHMO")

    # Assert POST request details
    mock_post.assert_called_once_with(
        "http://mock-rag:8000/api/v1/graph-query",
        json={"query": "What is class X?", "top_k": 5, "ontology_id": "CHMO"},
        timeout=60
    )

    # Assert parsed response
    assert result.answer == "Graph response"
    assert len(result.sources) == 1
    assert result.sources[0].ontology_id == "CHMO"
    assert result.sources[0].version == "2"
    assert result.sources[0].content == "Graph facts about class"
    assert result.sources[0].metadata == {"type": "owl_class"}


# ==========================================
# 2. Test Retrieve Node Routing
# ==========================================

@patch("ontoportal_agent.agent.graph.ChatOpenAI")
@patch("ontoportal_agent.agent.graph.McpClient")
@patch("ontoportal_agent.agent.graph.RagClient")
def test_retrieve_answer_node_calls_graph_query_when_enabled(
    mock_rag_client_class, mock_mcp_client_class, mock_chat_openai_class
):
    # Set up mocks
    mock_rag_client = MagicMock()
    mock_rag_client_class.return_value = mock_rag_client

    # Define mock response for graph_query
    mock_rag_client.graph_query.return_value = RagResult(
        answer="GraphRAG answer",
        sources=[RagChunk(ontology_id="CHMO", version="2", content="source content", metadata={})]
    )

    # Build agent graph with runtime options specifying RAG endpoints
    options = AgentRuntimeOptions(
        openai_api_key="test-key",
        rag_base_url="http://mock-rag:8000",
        rag_query_path="/api/v1/query",
        rag_top_k=3,
    )

    # Clear cached settings and force graph_rag_enabled to True
    settings = get_settings()
    original_enabled = settings.graph_rag_enabled
    settings.graph_rag_enabled = True

    try:
        # Build the graph
        graph = build_agent_graph(repository=MagicMock(), runtime_options=options)
        
        # Invoke retrieve node directly using func
        state = {"user_input": "Test question"}
        retrieved_state = graph.nodes["retrieve"].runnable.func(state)

        # Verify graph_query was called and query was NOT called
        mock_rag_client.graph_query.assert_called_once_with("Test question", top_k=3)
        mock_rag_client.query.assert_not_called()

        # Verify state values
        assert retrieved_state["rag_result"] == "GraphRAG answer"
        assert retrieved_state["citations"] == ["CHMO v2"]
        assert retrieved_state["retrieval_backend"] == "graphrag-http"

    finally:
        settings.graph_rag_enabled = original_enabled


@patch("ontoportal_agent.agent.graph.ChatOpenAI")
@patch("ontoportal_agent.agent.graph.McpClient")
@patch("ontoportal_agent.agent.graph.RagClient")
def test_retrieve_answer_node_falls_back_on_graph_query_failure(
    mock_rag_client_class, mock_mcp_client_class, mock_chat_openai_class
):
    # Set up mocks
    mock_rag_client = MagicMock()
    mock_rag_client_class.return_value = mock_rag_client

    # Make graph_query fail, but standard query succeed
    mock_rag_client.graph_query.side_effect = Exception("GraphRAG service down")
    mock_rag_client.query.return_value = RagResult(
        answer="Legacy RAG answer",
        sources=[RagChunk(ontology_id="CHMO", version="1", content="legacy content", metadata={})]
    )

    # Build agent graph with runtime options specifying RAG endpoints
    options = AgentRuntimeOptions(
        openai_api_key="test-key",
        rag_base_url="http://mock-rag:8000",
        rag_query_path="/api/v1/query",
        rag_top_k=2,
    )

    # Clear cached settings and force graph_rag_enabled to True
    settings = get_settings()
    original_enabled = settings.graph_rag_enabled
    settings.graph_rag_enabled = True

    try:
        # Build the graph
        graph = build_agent_graph(repository=MagicMock(), runtime_options=options)
        
        # Invoke retrieve node directly using func
        state = {"user_input": "Test fallback question"}
        retrieved_state = graph.nodes["retrieve"].runnable.func(state)

        # Verify both graph_query and query were called
        mock_rag_client.graph_query.assert_called_once_with("Test fallback question", top_k=2)
        mock_rag_client.query.assert_called_once_with("Test fallback question", top_k=2)

        # Verify state values
        assert retrieved_state["rag_result"] == "Legacy RAG answer"
        assert retrieved_state["citations"] == ["CHMO v1"]
        assert retrieved_state["retrieval_backend"] == "rag-http"

    finally:
        settings.graph_rag_enabled = original_enabled
