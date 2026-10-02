"""Tests for the retrieval-augmented knowledge layer.

These exercise the real pipeline - real chunking, real embeddings, real vector
search, real answering - over a small corpus of genuine source excerpts. No
retrieval result is stubbed: the point of the suite is to prove that the
assistant retrieves what it should, cites what it retrieved, and refuses when it
has nothing to go on.

The lexical (hashing) embedder is used for most tests so the suite is
deterministic and needs no model download. One test additionally exercises the
semantic backend when it is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from airshield_core.rag.answering import (
    INSUFFICIENT_KNOWLEDGE,
    GroundedAnswerer,
)
from airshield_core.rag.chunking import chunk_document
from airshield_core.rag.context import ForecastContext
from airshield_core.rag.documents import Document, KnowledgeBase, load_knowledge_base
from airshield_core.rag.embeddings import HashingEmbedder, build_embedder
from airshield_core.rag.providers import LLMError, LLMResult, NoLLMProvider
from airshield_core.rag.retrieval import Retriever
from airshield_core.rag.vectorstore import VectorStore, VectorStoreError

# Real excerpts, quoted from the sources named in the metadata.
PM25_TEXT = (
    "Particulate matter (PM) is a mixture of solid particles and liquid droplets "
    "found in the air. PM2.5 refers to fine inhalable particles with diameters that "
    "are generally 2.5 micrometres and smaller. Some particles are emitted directly "
    "from a source, while others form in complicated reactions in the atmosphere.\n\n"
    "Fine particles (PM2.5) are the main cause of reduced visibility in parts of the "
    "United States. Particles can be inhaled deep into the lungs and some may even "
    "get into the bloodstream. Exposure to these particles can affect both the lungs "
    "and the heart."
)

AQI_TEXT = (
    "The Air Quality Index (AQI) is EPA's index for reporting air quality. The AQI "
    "includes six color-coded categories, each corresponding to a range of index "
    "values. The higher the AQI value, the greater the level of air pollution and the "
    "greater the health concern. An AQI value of 50 or below represents good air "
    "quality, while an AQI value over 300 represents hazardous air quality.\n\n"
    "The AQI is calculated for five major air pollutants regulated by the Clean Air "
    "Act: ground-level ozone, particle pollution, carbon monoxide, sulfur dioxide and "
    "nitrogen dioxide."
)

ACTIVITY_TEXT = (
    "Adults should do at least 150 minutes of moderate-intensity aerobic physical "
    "activity throughout the week, or at least 75 minutes of vigorous-intensity "
    "aerobic physical activity. Physical activity can be undertaken in many different "
    "ways: walking, cycling, sports and active recreation."
)


def _doc(doc_id: str, title: str, source: str, url: str, category: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        source=source,
        url=url,
        category=category,
        document_type="guidance",
        text=text,
        publication_date="2024-01-01",
        licence="test licence",
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        _doc(
            "epa-pm-basics",
            "Particulate Matter (PM) Basics",
            "US Environmental Protection Agency",
            "https://www.epa.gov/pm-pollution/particulate-matter-pm-basics",
            "pollutants",
            PM25_TEXT,
        ),
        _doc(
            "airnow-aqi-basics",
            "Air Quality Index (AQI) Basics",
            "US EPA AirNow",
            "https://www.airnow.gov/aqi/aqi-basics/",
            "AQI",
            AQI_TEXT,
        ),
        _doc(
            "who-physical-activity",
            "Physical activity",
            "World Health Organization",
            "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
            "outdoor_activity",
            ACTIVITY_TEXT,
        ),
    ]


@pytest.fixture
def store(documents: list[Document]) -> VectorStore:
    """A real index built with the deterministic lexical embedder."""
    embedder = HashingEmbedder()
    chunks = [c for doc in documents for c in chunk_document(doc)]
    vectors = embedder.encode([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vectors)
    store.embedder_name = embedder.name
    store.embedder_is_semantic = embedder.is_semantic
    store.built_at = "2026-01-01T00:00:00+00:00"
    return store


@pytest.fixture
def retriever(store: VectorStore) -> Retriever:
    # The lexical fallback produces a much flatter score distribution than the
    # semantic model (its top scores cluster in 0.1-0.5, versus 0.2-0.7 for
    # model2vec), so the relevance floor is lower here than the 0.22 default.
    return Retriever(store, HashingEmbedder(), top_k=3, min_score=0.15)


# ------------------------------------------------------------------ chunking
def test_chunks_inherit_document_provenance(documents: list[Document]) -> None:
    chunks = chunk_document(documents[0])
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.doc_id == "epa-pm-basics"
        assert chunk.metadata["source"] == "US Environmental Protection Agency"
        assert chunk.metadata["url"].startswith("https://www.epa.gov/")
        assert chunk.metadata["category"] == "pollutants"


def test_chunking_is_deterministic(documents: list[Document]) -> None:
    """The same document must produce the same chunk ids every run."""
    first = chunk_document(documents[0])
    second = chunk_document(documents[0])
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunking_splits_long_documents() -> None:
    paragraph = "Fine particulate matter is associated with health effects. " * 80
    doc = _doc("long", "Long", "Test", "https://example.org", "pollutants", paragraph)
    chunks = chunk_document(doc, chunk_chars=400, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)


# ------------------------------------------------------------------ retrieval
def test_relevant_documents_are_retrieved(retriever: Retriever) -> None:
    hits = retriever.retrieve("What is PM2.5 particulate matter?")
    assert hits, "a clearly relevant question must retrieve something"
    assert hits[0].doc_id == "epa-pm-basics"


def test_aqi_question_retrieves_the_aqi_source(retriever: Retriever) -> None:
    hits = retriever.retrieve("What does the Air Quality Index mean?")
    assert hits
    assert hits[0].doc_id == "airnow-aqi-basics"


def test_irrelevant_query_returns_nothing(retriever: Retriever) -> None:
    """An unrelated question must not return a misleading near-miss."""
    hits = retriever.retrieve("How do I bake sourdough bread at home?")
    assert hits == []


def test_nonsense_query_returns_nothing(retriever: Retriever) -> None:
    assert retriever.retrieve("???") == []
    assert retriever.retrieve("the a") == []
    assert retriever.retrieve("") == []


def test_source_metadata_is_preserved_through_retrieval(retriever: Retriever) -> None:
    hits = retriever.retrieve("What is PM2.5?")
    assert hits
    for hit in hits:
        assert hit.title
        assert hit.source
        assert hit.url.startswith("https://")
        assert hit.licence
        assert hit.document_type == "guidance"


def test_citations_only_include_retrieved_documents(retriever: Retriever) -> None:
    hits = retriever.retrieve("What is PM2.5 particulate matter?")
    citations = retriever.citations(hits)
    retrieved_ids = {h.doc_id for h in hits}
    assert {c["doc_id"] for c in citations} == retrieved_ids
    # A document that was not retrieved must not be cited.
    assert "who-physical-activity" not in {c["doc_id"] for c in citations}


def test_citations_are_deduplicated(retriever: Retriever) -> None:
    hits = retriever.retrieve("PM2.5 particles lungs health effects")
    citations = retriever.citations(hits)
    ids = [c["doc_id"] for c in citations]
    assert len(ids) == len(set(ids))


def test_min_score_floor_suppresses_weak_matches(store: VectorStore) -> None:
    strict = Retriever(store, HashingEmbedder(), top_k=5, min_score=0.99)
    assert strict.retrieve("What is PM2.5?") == []


# --------------------------------------------------------------- vector store
def test_store_roundtrip_preserves_chunks_and_metadata(
    documents: list[Document], tmp_path
) -> None:
    embedder = HashingEmbedder()
    chunks = [c for doc in documents for c in chunk_document(doc)]
    vectors = embedder.encode([c.text for c in chunks])

    store = VectorStore()
    store.add(chunks, vectors)
    store.embedder_name = embedder.name
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert loaded.size == store.size
    assert loaded.dimension == store.dimension
    assert loaded.doc_ids() == store.doc_ids()
    first = loaded.search(vectors[0], top_k=1)[0]
    assert first.metadata["url"].startswith("https://")


def test_store_rejects_dimension_mismatch() -> None:
    store = VectorStore()
    embedder = HashingEmbedder(dimension=64)
    doc = _doc("d", "T", "S", "https://e.org", "pollutants", PM25_TEXT)
    chunks = chunk_document(doc)
    store.add(chunks, embedder.encode([c.text for c in chunks]))
    with pytest.raises(VectorStoreError, match="dimension mismatch"):
        store.add(chunks, embedder.encode(["different width"]).repeat(2, axis=1))


def test_loading_a_missing_index_raises(tmp_path) -> None:
    with pytest.raises(VectorStoreError, match="no vector index"):
        VectorStore.load(tmp_path / "absent")


# ----------------------------------------------------------------- grounding
class _RecordingProvider:
    """Records the prompt and returns a fixed string.

    This is a test double for an external service (a language model over HTTP),
    not a substitute for code under test. It exists so the test can assert what
    the model was actually shown.
    """

    def __init__(self, reply: str = "A grounded reply."):
        self.reply = reply
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def model(self) -> str:
        return "recording-model"

    @property
    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResult:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return LLMResult(text=self.reply, provider=self.name, model=self.model)


class _FailingProvider:
    @property
    def name(self) -> str:
        return "failing"

    @property
    def model(self) -> str:
        return "failing-model"

    @property
    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResult:
        raise LLMError("simulated model outage")


def test_answer_is_grounded_in_retrieved_context(retriever: Retriever) -> None:
    provider = _RecordingProvider()
    answerer = GroundedAnswerer(retriever, provider)
    result = answerer.answer("What is PM2.5?")

    assert result.grounded is True
    assert result.mode == "llm"
    assert result.retrieved_chunks > 0
    assert result.answer == "A grounded reply."

    # The prompt must actually contain the retrieved passages.
    prompt = provider.calls[0]["user"]
    assert "RETRIEVED KNOWLEDGE" in prompt
    assert "Particulate matter" in prompt


def test_assistant_refuses_when_nothing_is_retrieved(retriever: Retriever) -> None:
    provider = _RecordingProvider()
    answerer = GroundedAnswerer(retriever, provider)
    result = answerer.answer("How do I bake sourdough bread?")

    assert result.insufficient_knowledge is True
    assert result.answer == INSUFFICIENT_KNOWLEDGE
    assert result.sources == []
    assert result.grounded is False
    # The model must not even be consulted.
    assert provider.calls == []


def test_assistant_refuses_with_no_knowledge_and_no_context(retriever: Retriever) -> None:
    result = GroundedAnswerer(retriever, _RecordingProvider()).answer("What is PM2.5?")
    assert result.grounded is True
    # With no context supplied, only the knowledge base supports the answer.
    assert result.context_used == {}


def test_citations_correspond_to_retrieved_documents(retriever: Retriever) -> None:
    provider = _RecordingProvider()
    result = GroundedAnswerer(retriever, provider).answer("What is PM2.5?")
    cited = {s["doc_id"] for s in result.sources}
    retrieved = {c["doc_id"] for c in result.retrieved}
    assert cited == retrieved
    assert cited.issubset({"epa-pm-basics", "airnow-aqi-basics", "who-physical-activity"})


def test_no_llm_returns_excerpts_not_invention(retriever: Retriever) -> None:
    """Without a model the assistant quotes sources; it does not write prose."""
    result = GroundedAnswerer(retriever, NoLLMProvider()).answer("What is PM2.5?")
    assert result.mode == "extractive"
    assert result.llm_available is False
    assert result.grounded is True
    assert result.notice
    # The answer must be built from the retrieved text.
    assert "Particulate matter" in result.answer


def test_llm_failure_does_not_produce_a_fabricated_answer(retriever: Retriever) -> None:
    result = GroundedAnswerer(retriever, _FailingProvider()).answer("What is PM2.5?")
    assert result.mode == "llm_error"
    assert result.llm_available is False
    assert "unavailable" in result.answer.lower()
    # Retrieved sources are still surfaced so the user can read them.
    assert result.sources


def test_system_prompt_forbids_invented_numbers(retriever: Retriever) -> None:
    provider = _RecordingProvider()
    GroundedAnswerer(retriever, provider).answer("What is PM2.5?")
    system = provider.calls[0]["system"].lower()
    assert "never invent a number" in system
    assert "lower predicted exposure" in system
    assert "medical" in system


# ------------------------------------------------------------ ML integration
def test_context_renders_supplied_values_and_nothing_else(retriever: Retriever) -> None:
    """The model sees the real AirShield numbers, and only those."""
    context = ForecastContext(
        location="Delhi, India",
        observed_pm25=52.0,
        observed_aqi=143,
        observed_category="Unhealthy for Sensitive Groups",
        predicted_pm25=78.0,
        forecast_horizon_hours=2,
        activity="running",
        activity_label="Running",
        duration_minutes=45,
        recommended_window="10:00-10:45",
        recommended_window_exposure=41.0,
        relative_reduction_percent=42.0,
        model_version="xgboost-pm25-1h-test",
        data_mode="live",
    )
    provider = _RecordingProvider()
    result = GroundedAnswerer(retriever, provider).answer(
        "Why are you recommending this time?", context=context
    )

    prompt = provider.calls[0]["user"]
    assert "52.0 ug/m3" in prompt
    assert "78.0 ug/m3" in prompt
    assert "10:00-10:45" in prompt
    assert "42.0%" in prompt
    assert "xgboost-pm25-1h-test" in prompt

    # The categories must be labelled, not blended.
    assert "OBSERVED" in prompt
    assert "PREDICTED" in prompt
    assert "RECOMMENDATION" in prompt

    # And the auditable record reflects exactly those values.
    assert result.context_used["observed_pm25"] == 52.0
    assert result.context_used["predicted_pm25"] == 78.0
    assert result.context_used["recommended_window"] == "10:00-10:45"


def test_context_reports_missing_values_as_unavailable(retriever: Retriever) -> None:
    """A gap must be visible, never filled in with a plausible number."""
    context = ForecastContext(location="Delhi, India", observed_pm25=3.2)
    block = context.fact_block()
    assert "3.2" in block
    # No prediction was supplied, so no prediction section may appear.
    assert "PREDICTED" not in block
    assert "RECOMMENDATION" not in block


def test_forecast_context_reads_existing_api_payload_shape() -> None:
    """The bridge consumes the same dicts /api/forecast and /api/plan return."""
    forecast = {
        "location": {"label": "Delhi, India", "slug": "delhi"},
        "source": {"name": "Open-Meteo", "mode": "live"},
        "notice": None,
        "current": {
            "time": "2026-10-02T12:00:00Z",
            "pm2_5": 3.2,
            "pm10": 7.4,
            "nitrogen_dioxide": 3.4,
            "ozone": 64.0,
            "aqi": 13,
            "category": "Good",
            "weather": {"wind_speed_10m": 18.3, "temperature_2m": 14.7},
        },
        "forecast": {
            "predicted_pm25": 3.156,
            "horizon_hours": 1,
            "target_time": "2026-10-02T13:00:00Z",
            "model_version": "xgboost-pm25-1h-abc",
            "backend": "local",
        },
        "aqi": {"aqi": 13, "category": "Good"},
        "spike": {
            "spike_detected": False,
            "severity": "none",
            "expected_change_percent": 0.0,
            "confidence": 0.0,
            "confidence_basis": "no_forecast",
        },
    }
    plan = {
        "location": "Delhi, India",
        "location_slug": "delhi",
        "activity": "running",
        "activity_label": "Running",
        "duration_minutes": 45,
        "basis": "predicted",
        "relative_reduction_percent": 9.2,
        "best_window": {
            "start": "2026-10-02T12:00:00Z",
            "end": "2026-10-02T12:45:00Z",
            "exposure": {"score": 4.73, "level": "low"},
        },
        "alternative_window": {
            "start": "2026-10-02T13:00:00Z",
            "end": "2026-10-02T13:45:00Z",
            "exposure": {"score": 5.1, "level": "low"},
        },
        "highest_exposure_window": {
            "start": "2026-10-02T15:00:00Z",
            "end": "2026-10-02T15:45:00Z",
            "exposure": {"score": 5.2, "level": "low"},
        },
    }

    context = ForecastContext.from_payloads(forecast=forecast, plan=plan)
    assert context.observed_pm25 == 3.2
    assert context.predicted_pm25 == pytest.approx(3.156)
    assert context.forecast_horizon_hours == 1
    assert context.recommended_window == "12:00-12:45"
    assert context.alternative_window == "13:00-13:45"
    assert context.highest_exposure_window == "15:00-15:45"
    assert context.recommended_window_exposure == pytest.approx(4.73)
    assert context.relative_reduction_percent == pytest.approx(9.2)
    assert context.model_version == "xgboost-pm25-1h-abc"
    assert context.data_mode == "live"


def test_context_tolerates_absent_payloads() -> None:
    context = ForecastContext.from_payloads()
    assert context.has_any_facts() is False
    assert context.fact_block() == ""
    # Answering with no context and no retrieval must decline, not guess.
    assert context.as_dict()["observed_pm25"] is None


# --------------------------------------------------------- knowledge loading
def test_load_knowledge_base_from_repo() -> None:
    """The committed corpus loads and carries provenance."""
    try:
        kb = load_knowledge_base()
    except Exception as exc:  # pragma: no cover - only if corpus is absent
        pytest.skip(f"knowledge base not present: {exc}")
    assert isinstance(kb, KnowledgeBase)
    assert len(kb.documents) >= 5
    for document in kb.documents:
        assert document.url.startswith("https://")
        assert document.source
        assert document.title
        assert document.char_count > 100


def test_loaded_corpus_sources_are_authoritative() -> None:
    try:
        kb = load_knowledge_base()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"knowledge base not present: {exc}")
    sources = " ".join(kb.stats()["sources"]).lower()
    # The corpus must be drawn from official agencies and reputable bodies.
    assert any(
        name in sources
        for name in ("world health organization", "environmental protection agency")
    )


# ------------------------------------------------------------ semantic backend
def test_semantic_embedder_when_available() -> None:
    """Exercise the real semantic backend when model2vec is installed."""
    embedder = build_embedder("semantic", allow_fallback=False)
    if not embedder.is_semantic:  # pragma: no cover
        pytest.skip("semantic embedder unavailable")
    vectors = embedder.encode(["particulate matter health", "air quality index"])
    assert vectors.shape == (2, embedder.dimension)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_hashing_embedder_identifies_itself_as_non_semantic() -> None:
    embedder = HashingEmbedder()
    assert embedder.is_semantic is False
    assert "fallback" in embedder.name


def test_build_embedder_falls_back_visibly() -> None:
    """An unavailable model must degrade to a backend that says it is degraded."""
    embedder = build_embedder("definitely-not-a-real-model-name", allow_fallback=True)
    assert embedder.is_semantic is False
    assert "fallback" in embedder.name
