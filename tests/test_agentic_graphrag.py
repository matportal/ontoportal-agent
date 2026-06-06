import pytest
from unittest.mock import MagicMock, call
from langchain_core.messages import AIMessage
from ontoportal_agent.rag_client import RagClient, RagChunk, RagResult
from ontoportal_agent.agentic_graphrag import run_agentic_graphrag, _extract_json

def test_extract_json():
    # Test basic extraction
    assert _extract_json('{"key": "value"}') == {"key": "value"}
    # Test markdown code block extraction
    assert _extract_json('```json\n{"key": "value"}\n```') == {"key": "value"}
    # Test extraction with leading/trailing noise
    assert _extract_json('Some text {"key": "value"} other text') == {"key": "value"}
    # Test invalid json
    assert _extract_json('invalid') == {}

def test_run_agentic_graphrag_success():
    # Mock LLM
    mock_llm = MagicMock()
    # 1. Decomposition response
    msg_decomp = AIMessage(content='{"needs": ["Need 1", "Need 2"]}')
    # 2. Evaluation responses
    # For Need 1: satisfied on first try
    msg_eval_1 = AIMessage(content='{"status": "satisfied", "assessment": "Need 1 satisfied", "missing_info": "", "suggested_rewrite": null}')
    # For Need 2: unsatisfied first, then satisfied
    msg_eval_2_try1 = AIMessage(content='{"status": "unsatisfied", "assessment": "Need 2 needs synonym", "missing_info": "Synonyms", "suggested_rewrite": "Need 2 synonym"}')
    msg_eval_2_try2 = AIMessage(content='{"status": "satisfied", "assessment": "Need 2 satisfied now", "missing_info": "", "suggested_rewrite": null}')
    
    mock_llm.invoke.side_effect = [
        msg_decomp,
        msg_eval_1,
        msg_eval_2_try1,
        msg_eval_2_try2,
    ]

    # Mock RagClient
    mock_rag_client = MagicMock(spec=RagClient)
    # We return mock RagResult objects
    mock_rag_client.graph_query.side_effect = [
        # Need 1 try 1
        RagResult(
            answer="Answer 1",
            sources=[RagChunk(ontology_id="ONT1", version="1", content="Content 1", metadata={"citation_text": "cit1"})]
        ),
        # Need 2 try 1
        RagResult(
            answer="Answer 2 empty",
            sources=[]
        ),
        # Need 2 try 2 (synonym rewrite)
        RagResult(
            answer="Answer 2 synonym solved",
            sources=[RagChunk(ontology_id="ONT2", version="1", content="Content 2", metadata={"citation_text": "cit2"})]
        ),
    ]

    # Run loop
    result = run_agentic_graphrag(
        question="How to reuse Y in Z?",
        rag_client=mock_rag_client,
        llm=mock_llm,
        ontology_id="ONT",
        strict_scope=True,
        allow_scope_expansion=False,
        max_iterations=3
    )

    # Assertions on calls
    assert mock_llm.invoke.call_count == 4
    
    # Check graph_query calls
    expected_calls = [
        call("Need 1", ontology_id="ONT", strict_scope=True, allow_scope_expansion=False, top_k=None),
        call("Need 2", ontology_id="ONT", strict_scope=True, allow_scope_expansion=False, top_k=None),
        call("Need 2 synonym", ontology_id="ONT", strict_scope=True, allow_scope_expansion=False, top_k=None),
    ]
    mock_rag_client.graph_query.assert_has_calls(expected_calls)

    # Check overall result structure
    assert result["sufficient_context"] is True
    assert len(result["gaps"]) == 0
    
    expected_final_context = (
        "Agentic GraphRAG evidence for: How to reuse Y in Z?\n\n"
        "Need: Need 1\n"
        "Status: satisfied\n"
        "Evidence-backed answer: Answer 1 [cit1]\n\n"
        "Need: Need 2\n"
        "Status: satisfied\n"
        "Evidence-backed answer: Answer 2 synonym solved [cit2]\n\n"
        "Sources:\n"
        "- cit1: ONT1 v1\n"
        "- cit2: ONT2 v1"
    )
    assert result["final_context"] == expected_final_context
    assert len(result["sources"]) == 2
    assert result["sources"][0]["ontology_id"] == "ONT1"
    assert result["sources"][1]["ontology_id"] == "ONT2"
    
    # Check coverage list
    assert len(result["coverage"]) == 2
    assert result["coverage"][0]["need"] == "Need 1"
    assert result["coverage"][0]["status"] == "satisfied"
    assert result["coverage"][0]["attempts_count"] == 1
    
    assert result["coverage"][1]["need"] == "Need 2"
    assert result["coverage"][1]["status"] == "satisfied"
    assert result["coverage"][1]["attempts_count"] == 2


def test_run_agentic_graphrag_max_iterations_limit():
    # Mock LLM that keeps suggesting rewrites
    mock_llm = MagicMock()
    msg_decomp = AIMessage(content='{"needs": ["Need 1"]}')
    msg_eval = AIMessage(content='{"status": "unsatisfied", "assessment": "still missing", "missing_info": "everything", "suggested_rewrite": "Need 1 rewrite"}')
    msg_synth = AIMessage(content='{"sufficient_context": false, "final_context": "Incomplete synthesis"}')
    
    mock_llm.invoke.side_effect = [
        msg_decomp,
        msg_eval,
        msg_eval,
        msg_eval,
        msg_synth
    ]

    # Mock RagClient
    mock_rag_client = MagicMock(spec=RagClient)
    mock_rag_client.graph_query.return_value = RagResult(answer="No luck", sources=[])

    # Run loop with max_iterations=3
    result = run_agentic_graphrag(
        question="Find X",
        rag_client=mock_rag_client,
        llm=mock_llm,
        ontology_id="ONT",
        max_iterations=3
    )

    # Verify we hit the max_iterations limit (3 graph queries for Need 1)
    assert mock_rag_client.graph_query.call_count == 3
    assert result["sufficient_context"] is False
    assert len(result["gaps"]) == 1
    assert result["gaps"][0] == "Need 1"
    assert result["coverage"][0]["attempts_count"] == 3
    assert result["coverage"][0]["status"] == "unsatisfied"


def test_run_agentic_graphrag_unsatisfied_no_rewrite():
    # Mock LLM that returns partially_satisfied but no suggested rewrite
    mock_llm = MagicMock()
    msg_decomp = AIMessage(content='{"needs": ["Need 1"]}')
    # LLM says partially_satisfied, no rewrite
    msg_eval = AIMessage(content='{"status": "partially_satisfied", "assessment": "some evidence but incomplete", "missing_info": "specific mapping", "suggested_rewrite": null}')
    
    mock_llm.invoke.side_effect = [
        msg_decomp,
        msg_eval
    ]

    # Mock RagClient returning some sources and answer
    mock_rag_client = MagicMock(spec=RagClient)
    mock_rag_client.graph_query.return_value = RagResult(
        answer="Partially correct answer",
        sources=[RagChunk(ontology_id="ONT1", version="1", content="Some content", metadata={"citation_text": "cit1"})]
    )

    # Run loop with max_iterations=1 so we only do 1 try and stop
    result = run_agentic_graphrag(
        question="Find X",
        rag_client=mock_rag_client,
        llm=mock_llm,
        ontology_id="ONT",
        max_iterations=1
    )

    assert mock_rag_client.graph_query.call_count == 1
    # Check that sufficient_context is false because we have a gap
    assert result["sufficient_context"] is False
    assert len(result["gaps"]) == 1
    assert result["gaps"][0] == "Need 1"
    assert result["coverage"][0]["status"] == "partially_satisfied"
    assert result["coverage"][0]["missing_info"] == "specific mapping"


def test_run_agentic_graphrag_satisfied_without_sources_is_coerced():
    # Mock LLM that returns satisfied but there are actually no sources
    mock_llm = MagicMock()
    msg_decomp = AIMessage(content='{"needs": ["Need 1"]}')
    # LLM says satisfied, but RAG returned no sources
    msg_eval = AIMessage(content='{"status": "satisfied", "assessment": "looked good to me", "missing_info": "none", "suggested_rewrite": null}')
    
    mock_llm.invoke.side_effect = [
        msg_decomp,
        msg_eval
    ]

    # Mock RagClient returning empty sources
    mock_rag_client = MagicMock(spec=RagClient)
    mock_rag_client.graph_query.return_value = RagResult(
        answer="I have no sources for this answer",
        sources=[]
    )

    result = run_agentic_graphrag(
        question="Find X",
        rag_client=mock_rag_client,
        llm=mock_llm,
        ontology_id="ONT",
        max_iterations=1
    )

    # Status should be coerced to unsatisfied because there are no sources
    assert result["sufficient_context"] is False
    assert len(result["gaps"]) == 1
    assert result["coverage"][0]["status"] == "unsatisfied"


