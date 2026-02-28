from __future__ import annotations

from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    rag_result: Optional[str]
    citations: List[str]
    workspace: Optional[str]
    plan_actions: List[dict]
    sandbox_output: Optional[str]
    change_notes: List[str]
    approval_required: bool
    auto_publish: bool
    publish_payload: Optional[dict]
    final_response: Optional[str]
    retrieval_backend: Optional[str]
    retrieval_error: Optional[str]
    generation_backend: Optional[str]
    generation_error: Optional[str]
