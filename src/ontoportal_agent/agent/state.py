from __future__ import annotations

from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    rag_result: Optional[str]
    citations: List[str]
    sandbox_output: Optional[str]
    change_notes: List[str]
    approval_required: bool
    publish_payload: Optional[dict]
    final_response: Optional[str]
