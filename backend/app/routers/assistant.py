"""Ask AirShield - the retrieval-grounded assistant endpoints.

Two routes:

* ``POST /api/assistant/chat`` - ask a question.
* ``GET  /api/assistant/status`` - report whether the assistant is ready, which
  embedder built the index, and whether a language model is configured.

The assistant explains; it does not predict. Every response says whether the
answer was grounded in retrieved knowledge, which mode produced it, and which
AirShield values were supplied to the model.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_assistant_service
from app.models import AssistantChatRequest, AssistantChatResponse, AssistantStatusResponse
from app.services.assistant_service import AssistantService, AssistantUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.get("/status", response_model=AssistantStatusResponse)
def assistant_status(
    service: AssistantService = Depends(get_assistant_service),
) -> AssistantStatusResponse:
    """Report the assistant's real readiness.

    A missing index or an unconfigured language model is stated plainly here so
    the UI can show an honest degraded state instead of a chat box that fails
    once the user types.
    """
    return AssistantStatusResponse(**service.status())


@router.post("/chat", response_model=AssistantChatResponse)
def assistant_chat(
    request: AssistantChatRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> AssistantChatResponse:
    """Answer a question from retrieved knowledge and AirShield's own outputs.

    The response never contains a number the application did not compute. When
    the knowledge base cannot support an answer, ``insufficient_knowledge`` is
    true and the answer says so.
    """
    try:
        payload = service.chat(
            message=request.message,
            location_slug=request.location_slug,
            activity=request.activity,
            duration_minutes=request.duration_minutes,
            supplied_context=(
                request.context.model_dump() if request.context is not None else None
            ),
        )
    except AssistantUnavailable as exc:
        # The assistant is a feature, not the core forecast path: 503 with the
        # real reason so the UI can explain why the panel is unavailable.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # unexpected failures must be visible, never faked
        logger.exception("assistant chat failed")
        raise HTTPException(status_code=500, detail=f"assistant failed: {exc}") from exc

    return AssistantChatResponse(**payload)
