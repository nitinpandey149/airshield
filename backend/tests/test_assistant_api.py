"""API tests for the Ask AirShield assistant.

The assistant is a thin layer over the forecast/planning services and the
retrieval index, so these tests exercise the whole path through the API: the
endpoint, the service, the context bridge and the answerer. The critical
assertion throughout is that the assistant explains the *real* API values and
invents nothing.
"""

from __future__ import annotations

import pytest

from airshield_core.rag.answering import INSUFFICIENT_KNOWLEDGE


@pytest.fixture
def assistant_client(client, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A client whose assistant index is built from the real corpus.

    The index is built with the same semantic embedder the application uses by
    default. Retrieval quality is the thing under test here, and the lexical
    fallback is deliberately weaker (its scores overlap between relevant and
    irrelevant questions), so it would not be a meaningful test of the
    relevance floor.
    """
    from airshield_core.rag.chunking import chunk_document
    from airshield_core.rag.documents import KnowledgeBaseError, load_knowledge_base
    from airshield_core.rag.embeddings import build_embedder
    from airshield_core.rag.vectorstore import VectorStore

    try:
        kb = load_knowledge_base()
    except KnowledgeBaseError as exc:
        pytest.skip(f"knowledge base unavailable: {exc}")

    embedder = build_embedder("semantic")
    if not embedder.is_semantic:  # pragma: no cover - depends on model download
        pytest.skip("semantic embedder unavailable (model2vec not installed)")

    chunks = [c for doc in kb.documents for c in chunk_document(doc)]
    vectors = embedder.encode([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vectors)
    store.embedder_name = embedder.name
    store.embedder_is_semantic = True
    store.built_at = "2026-01-01T00:00:00+00:00"
    index_dir = tmp_path / "rag-index"
    store.save(index_dir)

    monkeypatch.setenv("AIRSHIELD_RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("AIRSHIELD_RAG_MIN_SCORE", "0.22")
    # Ensure no ambient language model leaks into the test.
    monkeypatch.setenv("AIRSHIELD_LLM_BASE_URL", "")
    monkeypatch.setenv("AIRSHIELD_LLM_MODEL", "")

    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()
    yield client
    deps.reset_service()
    reset_settings_cache()


# ------------------------------------------------------------------- status
def test_status_reports_index_and_llm_honestly(assistant_client) -> None:
    response = assistant_client.get("/api/assistant/status")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["ready"] is True
    assert body["index"]["chunks"] > 0
    assert body["index"]["documents"] > 0
    # No language model is configured in the test environment, and the API says so.
    assert body["llm_available"] is False
    assert body["llm_detail"]
    assert body["suggested_questions"]


def test_status_reports_missing_index_rather_than_pretending(
    client, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRSHIELD_RAG_INDEX_DIR", str(tmp_path / "absent"))
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()
    try:
        response = client.get("/api/assistant/status")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert body["detail"]
        assert "build_index" in body["detail"]
    finally:
        deps.reset_service()
        reset_settings_cache()


# --------------------------------------------------------------------- chat
def test_chat_answers_a_general_knowledge_question(assistant_client) -> None:
    response = assistant_client.post(
        "/api/assistant/chat", json={"message": "What is PM2.5?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["retrieved_chunks"] > 0
    assert body["sources"], "a grounded answer must cite its sources"
    assert body["insufficient_knowledge"] is False
    for source in body["sources"]:
        assert source["url"].startswith("https://")
        assert source["source"]


def test_chat_refuses_an_unanswerable_question(assistant_client) -> None:
    response = assistant_client.post(
        "/api/assistant/chat",
        json={"message": "How do I bake sourdough bread at home?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_knowledge"] is True
    assert body["answer"] == INSUFFICIENT_KNOWLEDGE
    assert body["sources"] == []
    assert body["retrieved_chunks"] == 0


def test_chat_rejects_an_empty_message(assistant_client) -> None:
    response = assistant_client.post("/api/assistant/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_handles_an_unknown_location(assistant_client) -> None:
    """A bad location is reported, not silently ignored."""
    response = assistant_client.post(
        "/api/assistant/chat",
        json={"message": "What is PM2.5?", "location_slug": "atlantis"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["context_error"]
    assert "atlantis" in body["context_error"]


# --------------------------------------------------------- ML integration
def test_context_values_match_the_forecast_and_plan_endpoints(assistant_client) -> None:
    """The assistant must explain the same numbers the dashboard shows."""
    forecast = assistant_client.get("/api/forecast/berlin").json()
    plan = assistant_client.get(
        "/api/plan/berlin?activity=running&duration_minutes=45"
    ).json()

    response = assistant_client.post(
        "/api/assistant/chat",
        json={
            "message": "Why are you recommending this time window?",
            "location_slug": "berlin",
            "activity": "running",
            "duration_minutes": 45,
        },
    )
    assert response.status_code == 200
    used = response.json()["context_used"]

    # Observed and predicted PM2.5 come from the forecast endpoint verbatim.
    assert used["observed_pm25"] == pytest.approx(forecast["current"]["pm2_5"])
    assert used["predicted_pm25"] == pytest.approx(forecast["forecast"]["predicted_pm25"])
    assert used["observed_aqi"] == forecast["current"]["aqi"]
    assert used["forecast_horizon_hours"] == forecast["forecast"]["horizon_hours"]

    # The recommendation comes from the plan endpoint verbatim.
    assert used["recommended_window_exposure"] == pytest.approx(
        plan["best_window"]["exposure"]["score"]
    )
    assert used["relative_reduction_percent"] == pytest.approx(
        plan["relative_reduction_percent"]
    )
    assert used["activity"] == plan["activity"]
    assert used["duration_minutes"] == plan["duration_minutes"]


def test_no_prediction_value_is_generated_by_the_assistant(assistant_client) -> None:
    """Every number in context_used must be traceable to an engine."""
    response = assistant_client.post(
        "/api/assistant/chat",
        json={
            "message": "Explain my exposure",
            "location_slug": "berlin",
            "activity": "running",
            "duration_minutes": 45,
        },
    )
    used = response.json()["context_used"]
    forecast = assistant_client.get("/api/forecast/berlin").json()

    # The predicted value is exactly the model's output, not a rounded or
    # re-derived copy.
    assert used["predicted_pm25"] == pytest.approx(forecast["forecast"]["predicted_pm25"])
    # Provenance travels with it, so the answer can name the model.
    assert used["data_mode"] in ("live", "demo")


def test_client_supplied_context_is_used_verbatim(assistant_client) -> None:
    """When the client sends its own payloads, they are not recomputed."""
    response = assistant_client.post(
        "/api/assistant/chat",
        json={
            "message": "Why is my exposure score higher now?",
            "context": {
                "location": "Delhi, India",
                "forecast": {
                    "current": {"pm2_5": 52.0, "aqi": 143, "category": "Unhealthy"},
                    "forecast": {"predicted_pm25": 78.0, "horizon_hours": 2},
                    "source": {"mode": "live"},
                },
                "exposure": {
                    "activity": "running",
                    "activity_label": "Running",
                    "duration_minutes": 45,
                    "relative_reduction_percent": 42.0,
                    "best_window": {
                        "start": "2026-10-02T10:00:00Z",
                        "end": "2026-10-02T10:45:00Z",
                        "exposure": {"score": 41.0, "level": "moderate"},
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    used = response.json()["context_used"]
    assert used["observed_pm25"] == 52.0
    assert used["predicted_pm25"] == 78.0
    assert used["recommended_window"] == "10:00-10:45"
    assert used["relative_reduction_percent"] == 42.0


# ------------------------------------------------------------- degradation
def test_missing_index_yields_a_clear_503(
    client, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRSHIELD_RAG_INDEX_DIR", str(tmp_path / "nothing"))
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()
    try:
        response = client.post("/api/assistant/chat", json={"message": "What is PM2.5?"})
        assert response.status_code == 503
        assert "build_index" in response.json()["detail"]
    finally:
        deps.reset_service()
        reset_settings_cache()


def test_disabled_assistant_reports_unavailable(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRSHIELD_ASSISTANT_ENABLED", "false")
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()
    try:
        response = client.post("/api/assistant/chat", json={"message": "What is PM2.5?"})
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]
    finally:
        deps.reset_service()
        reset_settings_cache()


def test_llm_failure_does_not_break_the_endpoint(
    assistant_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-broken model must degrade, not fabricate or 500."""
    from app import deps
    from app.services import assistant_service as module
    from airshield_core.rag.providers import LLMError

    class BrokenProvider:
        name = "broken"
        model = "broken-model"
        is_available = True

        def complete(self, system: str, user: str, *, temperature: float = 0.0):
            raise LLMError("simulated model outage")

    monkeypatch.setattr(module, "build_provider", lambda settings: BrokenProvider())
    deps.reset_service()

    response = assistant_client.post(
        "/api/assistant/chat", json={"message": "What is PM2.5?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "llm_error"
    assert body["llm_available"] is False
    assert body["sources"], "retrieved sources should still be returned"

    deps.reset_service()


# ---------------------------------------------------------------- banner
def test_service_banner_advertises_the_assistant(client) -> None:
    body = client.get("/api/info").json()
    assert "assistant_chat" in body["endpoints"]
    assert "assistant_status" in body["endpoints"]
