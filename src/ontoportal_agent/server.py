from __future__ import annotations

import json
import logging
import re
from threading import Lock
from typing import Iterator, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent.runtime import OntoPortalAgent
from .config import get_settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OntoPortal Agent API",
    description="Streaming bridge for the MatPortal assistant UI.",
    version="1.0.0",
)

_agent_lock = Lock()
_agent_instance: Optional[OntoPortalAgent] = None


class ChatStreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _iter_text_chunks(text: str, max_chars: int = 220) -> Iterator[str]:
    clean = text.strip()
    if not clean:
        return

    # Prefer sentence-sized chunks; fall back to hard splits for long lines.
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        start = 0
        while start < len(sentence):
            end = min(len(sentence), start + max_chars)
            if end < len(sentence):
                split = sentence.rfind(" ", start, end)
                if split > start:
                    end = split
            chunk = sentence[start:end].strip()
            if chunk:
                yield chunk + (" " if end < len(sentence) else "")
            start = end


def _get_agent() -> OntoPortalAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = OntoPortalAgent()
    return _agent_instance


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/chat/stream")
def chat_stream(
    request: ChatStreamRequest,
    x_internal_token: Optional[str] = Header(default=None),
) -> StreamingResponse:
    settings = get_settings()
    expected_token = settings.internal_api_token
    if expected_token and x_internal_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )

    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be blank",
        )

    def event_stream() -> Iterator[str]:
        graph_config = None
        if request.thread_id:
            graph_config = {"configurable": {"thread_id": request.thread_id}}

        yield _sse({"type": "status", "message": "Assistant request received."})
        yield _sse({"type": "status", "message": "Executing ontology assistant pipeline..."})
        try:
            final_state = {}
            with _agent_lock:
                if graph_config:
                    updates = _get_agent().graph.stream(
                        {"user_input": prompt},
                        config=graph_config,
                        stream_mode="updates",
                    )
                else:
                    updates = _get_agent().graph.stream(
                        {"user_input": prompt},
                        stream_mode="updates",
                    )
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    for node_name, node_state in update.items():
                        if isinstance(node_state, dict):
                            final_state.update(node_state)

            model_reasoning = final_state.get("generation_reasoning")
            if model_reasoning:
                yield _sse({"type": "model_reasoning", "content": str(model_reasoning)})

            generation_usage = final_state.get("generation_usage")
            if isinstance(generation_usage, dict):
                yield _sse({"type": "usage", "content": generation_usage})

            sandbox_output = final_state.get("sandbox_output")
            if sandbox_output:
                for line in str(sandbox_output).splitlines():
                    clean = line.strip()
                    if clean:
                        yield _sse({"type": "status", "message": clean})

            final_response = (
                final_state.get("final_response")
                or final_state.get("rag_result")
                or "No response generated."
            )
            for chunk in _iter_text_chunks(str(final_response)):
                yield _sse({"type": "delta", "content": chunk})
        except Exception as exc:  # pragma: no cover - smoke tests cover happy path.
            logger.exception("Assistant stream failed: %s", exc)
            yield _sse({"type": "delta", "content": "Assistant backend failed while handling the request."})
        yield _sse_done()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    uvicorn.run("ontoportal_agent.server:app", host="0.0.0.0", port=8090, reload=False)


if __name__ == "__main__":
    main()
