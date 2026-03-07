import importlib

import pytest
from langchain_core.messages import AIMessage

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.intent import INTENT_EDIT, INTENT_RETRIEVE, classify_user_intent


class _AlwaysEditLlm:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content="EDIT")


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "Write a markdown answer with a heading, then ten bullet points about why ontology portals matter, then a fenced json block, and keep streaming as you write.",
            INTENT_RETRIEVE,
        ),
        (
            "Compare ONTOMAT, MATONTO, and PMDCO in markdown with a heading, a short table, and one bullet list of differences.",
            INTENT_RETRIEVE,
        ),
        (
            "What is MatPortal? Answer in markdown with a heading, twelve bullet points, and a short json code block.",
            INTENT_RETRIEVE,
        ),
        (
            "Generate a tensile test ontology for polymers, validate it, and submit privately to MatPortal.",
            INTENT_EDIT,
        ),
        (
            "Add a new class to BWMD, write the Turtle file, and publish the ontology submission.",
            INTENT_EDIT,
        ),
    ],
)
def test_classify_user_intent_handles_retrieve_and_edit_boundaries(prompt, expected):
    llm = _AlwaysEditLlm()

    intent = classify_user_intent(prompt, llm=llm)

    assert intent == expected
    if expected == INTENT_RETRIEVE:
        assert llm.calls == 0


def test_classify_user_intent_uses_llm_for_ambiguous_ontology_change_requests():
    class _FallbackLlm:
        def invoke(self, messages):
            content = " ".join(
                str(getattr(message, "content", message))
                for message in messages
            )
            assert "Return EDIT only" in content
            return AIMessage(content="EDIT")

    intent = classify_user_intent(
        "Can you help me revise the ontology artifact for the next submission?",
        llm=_FallbackLlm(),
    )

    assert intent == INTENT_EDIT
