"""Grounded answering.

The assistant has exactly two jobs: retrieve relevant passages, and phrase an
answer that is faithful to them plus the numbers AirShield computed. It has no
license to add facts.

Three behaviours are enforced here rather than left to prompt wording:

1. **No retrieval, no knowledge answer.** If nothing clears the relevance floor,
   the assistant says it does not have verified information. It does not answer
   general questions from model memory.
2. **No LLM, no invention.** With no language model configured the assistant
   returns the retrieved source excerpts verbatim, labelled as excerpts, instead
   of fabricating prose.
3. **Numbers come from AirShield.** The prompt supplies the current values as
   fixed facts and forbids introducing others; the response records which values
   were supplied so a test can prove the model never produced them.

Every response carries the citations of the passages actually retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from airshield_core.rag.context import ForecastContext
from airshield_core.rag.providers import LLMError, LLMProvider, NoLLMProvider
from airshield_core.rag.retrieval import RetrievedChunk, Retriever

#: The wording required by the project when the corpus cannot support an answer.
INSUFFICIENT_KNOWLEDGE = (
    "I don't have enough verified information in my knowledge base to answer "
    "that confidently."
)

#: Shown when retrieval worked but no language model is configured.
NO_LLM_NOTICE = (
    "No language model is configured, so this is a summary of the retrieved "
    "source text rather than a generated explanation."
)

SYSTEM_PROMPT = """You are the AirShield Pulse assistant. You explain air-quality \
concepts and AirShield's own outputs. You are not a doctor and you do not give \
medical advice.

Absolute rules:
1. Answer ONLY from the RETRIEVED KNOWLEDGE and the AIRSHIELD CONTEXT below. \
Do not use any other knowledge, and do not fill gaps from memory.
2. NEVER invent a number. Do not state a PM2.5 value, AQI, exposure score, \
percentage, confidence or timestamp that is not present in the AIRSHIELD \
CONTEXT. If a number is absent, say it is not available.
3. Keep the categories distinct and label them in your answer when you use them:
   - "Observed" for measured data,
   - "Predicted" for model output,
   - "AirShield recommends" for the exposure engine's output,
   - "Guidance suggests" for anything from the retrieved knowledge.
4. Never claim or imply that a time window or a route is safe. AirShield compares \
relative predicted exposure. Say "lower predicted exposure", never "safe".
5. Never present a retrieved signal as a proven cause. If the context lists \
contributing signals, describe them as possible associations.
6. Never say you retrieved something you did not. If RETRIEVED KNOWLEDGE is \
empty, answer only with the AirShield context, or say you lack verified \
information.
7. If the question needs a fact the sources do not cover, say exactly: \
"I don't have enough verified information in my knowledge base to answer that \
confidently."
8. Do not diagnose, do not assess an individual's medical risk, and do not tell \
the user they are safe or unsafe.

Be concise and plain. Two short paragraphs at most unless asked for detail."""


@dataclass
class Answer:
    """A grounded answer plus the evidence behind it."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved_chunks: int = 0
    grounded: bool = False
    #: True when the assistant declined for lack of verified information.
    insufficient_knowledge: bool = False
    #: Which engine produced the prose: "llm" or "extractive".
    mode: str = "extractive"
    llm_available: bool = False
    llm_model: str = ""
    #: The AirShield values supplied to the model, for auditability.
    context_used: dict = field(default_factory=dict)
    notice: str | None = None
    retrieved: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_chunks": self.retrieved_chunks,
            "grounded": self.grounded,
            "insufficient_knowledge": self.insufficient_knowledge,
            "mode": self.mode,
            "llm_available": self.llm_available,
            "llm_model": self.llm_model,
            "context_used": self.context_used,
            "notice": self.notice,
            "retrieved": self.retrieved,
        }


class GroundedAnswerer:
    """Ties retrieval, AirShield context and the language model together."""

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider | None = None,
        *,
        min_context_chars: int = 0,
    ):
        self.retriever = retriever
        self.provider = provider or NoLLMProvider()
        self.min_context_chars = min_context_chars

    # ---------------------------------------------------------------- prompts
    def _build_user_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        context: ForecastContext | None,
    ) -> str:
        if chunks:
            knowledge = "\n\n".join(
                f"[{index + 1}] {chunk.title} - {chunk.source}\n{chunk.text}"
                for index, chunk in enumerate(chunks)
            )
        else:
            knowledge = "(none retrieved)"

        facts = context.fact_block() if context else ""
        if not facts:
            facts = "(no AirShield context supplied for this question)"

        return (
            f"AIRSHIELD CONTEXT (values computed by AirShield; the only numbers "
            f"you may use):\n{facts}\n\n"
            f"RETRIEVED KNOWLEDGE (the only external knowledge you may use):\n"
            f"{knowledge}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Answer now, following the rules. Do not list the sources; the "
            "application attaches them."
        )

    # ---------------------------------------------------------------- extract
    @staticmethod
    def _extractive_answer(
        question: str, chunks: list[RetrievedChunk], context: ForecastContext | None
    ) -> str:
        """Compose an answer from retrieved text when no LLM is available.

        This is deliberately not a generated answer. It quotes the retrieved
        passages, so the user still gets grounded information, and the caller
        labels the mode as ``extractive``.
        """
        parts: list[str] = []
        if context and context.has_any_facts():
            facts = context.fact_block()
            if facts:
                parts.append(f"From AirShield's current output:\n{facts}")

        if chunks:
            excerpts = "\n\n".join(
                f"[{index + 1}] {chunk.title} ({chunk.source}):\n{chunk.text}"
                for index, chunk in enumerate(chunks)
            )
            parts.append(f"Relevant passages from the knowledge base:\n{excerpts}")
        elif not parts:
            return INSUFFICIENT_KNOWLEDGE
        else:
            parts.append(
                "No matching passages were found in the knowledge base for this "
                "question, so only AirShield's own values are shown."
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ entry
    def answer(
        self,
        question: str,
        *,
        context: ForecastContext | None = None,
        top_k: int | None = None,
        category: str | None = None,
    ) -> Answer:
        """Retrieve, then answer. Never answers from model memory."""
        chunks = self.retriever.retrieve(question, top_k=top_k, category=category)
        citations = self.retriever.citations(chunks)
        context_used = context.as_dict() if context else {}
        has_facts = bool(context and context.has_any_facts())

        # Rule 1: nothing retrieved and nothing about the current situation to
        # explain -> decline rather than answer from memory.
        if not chunks and not has_facts:
            return Answer(
                answer=INSUFFICIENT_KNOWLEDGE,
                sources=[],
                retrieved_chunks=0,
                grounded=False,
                insufficient_knowledge=True,
                mode="refusal",
                llm_available=self.provider.is_available,
                llm_model=self.provider.model,
                context_used=context_used,
            )

        # Rule 2: no language model -> return grounded excerpts, clearly labelled.
        if not self.provider.is_available:
            return Answer(
                answer=self._extractive_answer(question, chunks, context),
                sources=citations,
                retrieved_chunks=len(chunks),
                grounded=bool(chunks),
                insufficient_knowledge=False,
                mode="extractive",
                llm_available=False,
                llm_model="",
                context_used=context_used,
                notice=NO_LLM_NOTICE,
                retrieved=[_chunk_summary(c) for c in chunks],
            )

        prompt = self._build_user_prompt(question, chunks, context)
        try:
            result = self.provider.complete(SYSTEM_PROMPT, prompt)
        except LLMError as exc:
            # A configured-but-broken model must not turn into a made-up answer.
            return Answer(
                answer=(
                    f"The language model is unavailable, so I cannot generate an "
                    f"explanation right now ({exc})."
                ),
                sources=citations,
                retrieved_chunks=len(chunks),
                grounded=bool(chunks),
                insufficient_knowledge=False,
                mode="llm_error",
                llm_available=False,
                llm_model=self.provider.model,
                context_used=context_used,
                notice=str(exc),
                retrieved=[_chunk_summary(c) for c in chunks],
            )

        return Answer(
            answer=result.text,
            sources=citations,
            retrieved_chunks=len(chunks),
            grounded=bool(chunks),
            insufficient_knowledge=False,
            mode="llm",
            llm_available=True,
            llm_model=result.model,
            context_used=context_used,
            retrieved=[_chunk_summary(c) for c in chunks],
        )


def _chunk_summary(chunk: RetrievedChunk) -> dict:
    """Compact, auditable record of a retrieved passage."""
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "source": chunk.source,
        "url": chunk.url,
        "score": round(chunk.score, 4),
        "ordinal": chunk.ordinal,
    }
