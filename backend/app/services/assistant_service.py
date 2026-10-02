"""The Ask AirShield assistant service.

Wires the RAG layer to the rest of the application. Two things matter here:

* **Context comes from the existing services.** When a user asks "why this
  recommendation?", the numbers handed to the language model are fetched by
  calling :class:`ForecastService` and :class:`PlanningService` — the same code
  that serves ``/api/forecast`` and ``/api/plan``. There is no parallel
  computation and therefore no way for the assistant to explain different
  numbers than the dashboard shows.

* **Failure is reported, not papered over.** A missing index, an unreachable
  forecast or a broken language model each produce an explicit state. None of
  them causes the assistant to fall back on invented content.
"""

from __future__ import annotations

import logging

from airshield_core.config import Settings
from airshield_core.rag.answering import Answer, GroundedAnswerer
from airshield_core.rag.context import ForecastContext
from airshield_core.rag.providers import build_provider
from airshield_core.rag.retrieval import RetrievalError, Retriever
from airshield_core.rag.vectorstore import VectorStoreError
from airshield_core.sources.registry import Location, LocationNotFoundError, get_location

logger = logging.getLogger(__name__)

#: Contextual prompts shown as buttons in the UI. Each maps to a question the
#: corpus and the engines can genuinely support.
SUGGESTED_QUESTIONS: list[dict] = [
    {
        "id": "why_recommendation",
        "label": "Why this recommendation?",
        "question": "Why are you recommending this time window?",
        "needs_context": True,
    },
    {
        "id": "what_is_pm25",
        "label": "What is PM2.5?",
        "question": "What is PM2.5 and why does it matter?",
        "needs_context": False,
    },
    {
        "id": "explain_exposure",
        "label": "Explain my exposure",
        "question": "How does the exposure score for my activity work?",
        "needs_context": True,
    },
    {
        "id": "pollution_rising",
        "label": "Why is pollution rising?",
        "question": "Why is pollution predicted to rise over the next few hours?",
        "needs_context": True,
    },
    {
        "id": "what_is_aqi",
        "label": "What does AQI mean?",
        "question": "What does the Air Quality Index mean?",
        "needs_context": False,
    },
    {
        "id": "running_vs_walking",
        "label": "Running vs walking",
        "question": "How does running differ from walking in the exposure model?",
        "needs_context": False,
    },
]


class AssistantUnavailable(RuntimeError):
    """Raised when the assistant cannot run at all (e.g. no vector index)."""


class AssistantService:
    """Retrieval-grounded question answering over AirShield's own outputs."""

    def __init__(self, settings: Settings, forecast_service, planning_service=None):
        self.settings = settings
        self.forecasts = forecast_service
        self.planning = planning_service
        self._retriever: Retriever | None = None
        self._answerer: GroundedAnswerer | None = None
        self._load_error: str | None = None

    # ------------------------------------------------------------------ setup
    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            if self._load_error:
                raise AssistantUnavailable(self._load_error)
            try:
                self._retriever = Retriever.from_index(
                    self.settings.rag_index_path,
                    top_k=self.settings.rag_top_k,
                    min_score=self.settings.rag_min_score,
                )
            except (VectorStoreError, RetrievalError, OSError) as exc:
                self._load_error = str(exc)
                raise AssistantUnavailable(self._load_error) from exc
        return self._retriever

    @property
    def answerer(self) -> GroundedAnswerer:
        if self._answerer is None:
            self._answerer = GroundedAnswerer(
                self.retriever, build_provider(self.settings)
            )
        return self._answerer

    def reset(self) -> None:
        """Drop cached state - used by tests that rebuild the index."""
        self._retriever = None
        self._answerer = None
        self._load_error = None

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        """Honest readiness report for the assistant."""
        provider = build_provider(self.settings)
        info: dict = {
            "enabled": self.settings.assistant_enabled,
            "llm_available": provider.is_available,
            "llm_model": provider.model or None,
            "suggested_questions": SUGGESTED_QUESTIONS,
        }
        try:
            stats = self.retriever.stats()
            info.update(
                {
                    "ready": True,
                    "index": stats,
                    "embedder_semantic": stats.get("semantic", False),
                }
            )
            if not stats.get("semantic", False):
                info["notice"] = (
                    "The retrieval index was built with the lexical fallback "
                    "embedder, so matching is keyword-based rather than semantic. "
                    "Rebuild with `python knowledge/build_index.py` after "
                    "installing `model2vec` for full quality."
                )
        except AssistantUnavailable as exc:
            info.update({"ready": False, "index": None, "detail": str(exc)})

        if not provider.is_available:
            info["llm_detail"] = (
                "No language model configured. The assistant returns retrieved "
                "source passages instead of generated explanations. Set "
                "AIRSHIELD_LLM_BASE_URL and AIRSHIELD_LLM_MODEL to enable it."
            )
        return info

    # ---------------------------------------------------------------- context
    def build_context(
        self,
        *,
        location_slug: str | None,
        activity: str | None,
        duration_minutes: int | None,
    ) -> tuple[ForecastContext | None, str | None]:
        """Assemble AirShield's own values for the prompt.

        Returns ``(context, error)``. The error is surfaced to the user rather
        than silently dropping the context, so they know the answer is not
        informed by a live forecast.
        """
        if not location_slug:
            return None, None

        try:
            location: Location = get_location(location_slug)
        except LocationNotFoundError as exc:
            return None, str(exc)

        forecast_payload = None
        plan_payload = None
        errors: list[str] = []

        try:
            forecast_payload = self.forecasts.build_forecast(location)
        except Exception as exc:
            errors.append(f"forecast unavailable ({exc})")

        if activity and self.planning is not None:
            try:
                plan_payload = self.planning.plan(
                    location,
                    activity=activity,
                    duration_minutes=duration_minutes or 45,
                )
            except Exception as exc:
                errors.append(f"exposure plan unavailable ({exc})")

        context = ForecastContext.from_payloads(
            forecast=forecast_payload,
            plan=plan_payload,
            location=location.label,
            location_slug=location.slug,
        )
        if activity and not plan_payload:
            context.activity = activity
            context.duration_minutes = duration_minutes

        return context, ("; ".join(errors) if errors else None)

    # ------------------------------------------------------------------- chat
    def chat(
        self,
        *,
        message: str,
        location_slug: str | None = None,
        activity: str | None = None,
        duration_minutes: int | None = None,
        supplied_context: dict | None = None,
    ) -> dict:
        """Answer a question, grounded in retrieved knowledge and real outputs."""
        if not self.settings.assistant_enabled:
            raise AssistantUnavailable("the assistant is disabled by configuration")

        context_error: str | None = None
        context: ForecastContext | None = None

        if supplied_context:
            # The client sent its own view of the numbers; trust it as-is rather
            # than recomputing, so the answer matches what the user is looking at.
            context = ForecastContext.from_payloads(
                forecast=supplied_context.get("forecast"),
                plan=supplied_context.get("exposure") or supplied_context.get("recommendation"),
                location=supplied_context.get("location") or None,
            )
        else:
            context, context_error = self.build_context(
                location_slug=location_slug,
                activity=activity,
                duration_minutes=duration_minutes,
            )

        answer: Answer = self.answerer.answer(message, context=context)
        payload = answer.to_dict()
        payload["retrieved_chunks"] = answer.retrieved_chunks

        if context_error:
            # Do not hide a failed context fetch behind a confident answer.
            payload["context_error"] = context_error
            note = (
                f"AirShield could not load live context for this location "
                f"({context_error}), so this answer is based on the knowledge base "
                f"only."
            )
            payload["notice"] = f"{payload['notice']} {note}".strip() if payload.get("notice") else note

        return payload
