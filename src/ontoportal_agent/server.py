from __future__ import annotations

import json
import logging
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
            with _agent_lock:
                if graph_config:
                    final_state = _get_agent().graph.invoke({"user_input": prompt}, config=graph_config)
                else:
                    final_state = _get_agent().graph.invoke({"user_input": prompt})

            retrieval_backend = final_state.get("retrieval_backend")
            if retrieval_backend:
                yield _sse({"type": "terminal_log", "content": f"retrieval_backend={retrieval_backend}"})

            retrieval_error = final_state.get("retrieval_error")
            if retrieval_error:
                yield _sse({"type": "terminal_log", "content": f"retrieval_error={retrieval_error}"})

            generation_backend = final_state.get("generation_backend")
            if generation_backend:
                yield _sse({"type": "terminal_log", "content": f"generation_backend={generation_backend}"})

            generation_error = final_state.get("generation_error")
            if generation_error:
                yield _sse({"type": "terminal_log", "content": f"generation_error={generation_error}"})

            generation_usage = final_state.get("generation_usage")
            if isinstance(generation_usage, dict):
                for key in ("model", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
                    value = generation_usage.get(key)
                    if value is not None:
                        yield _sse({"type": "terminal_log", "content": f"generation_{key}={value}"})

            sandbox_output = final_state.get("sandbox_output")
            if sandbox_output:
                for line in str(sandbox_output).splitlines():
                    clean = line.strip()
                    if clean:
                        yield _sse({"type": "terminal_log", "content": clean})

            final_response = (
                final_state.get("final_response")
                or final_state.get("rag_result")
                or "No response generated."
            )
            yield _sse({"content": final_response})
        except Exception as exc:  # pragma: no cover - smoke tests cover happy path.
            logger.exception("Assistant stream failed: %s", exc)
            yield _sse(
                {
                    "type": "status",
                    "message": "Assistant backend failed while handling the request.",
                }
            )
            yield _sse({"type": "terminal_log", "content": str(exc)})
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
