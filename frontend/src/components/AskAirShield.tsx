import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../lib/api'
import type { AssistantChatResponse, AssistantStatusResponse, SuggestedQuestion } from '../lib/types'

interface Turn {
  id: number
  question: string
  response: AssistantChatResponse | null
  error: string | null
  loading: boolean
}

interface Props {
  /** Current location, so the assistant can load live context. */
  slug: string
  /** Activity selected in the planner, used for exposure questions. */
  activity?: string
  durationMinutes?: number
}

const MODE_LABEL: Record<string, string> = {
  llm: 'generated from retrieved sources',
  extractive: 'retrieved source text',
  refusal: 'no verified information found',
  llm_error: 'language model unavailable',
}

/**
 * Ask AirShield.
 *
 * The panel explains AirShield's own outputs and retrieves environmental
 * knowledge. It never computes a forecast, AQI or exposure score — those arrive
 * in the response's `context_used` block, which came from the same services the
 * dashboard uses. Every knowledge answer carries the sources that were actually
 * retrieved, and the assistant says so when it has nothing verified to offer.
 */
export default function AskAirShield({ slug, activity, durationMinutes }: Props) {
  const [status, setStatus] = useState<AssistantStatusResponse | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const nextId = useRef(0)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .assistantStatus()
      .then((value) => {
        if (!cancelled) setStatus(value)
      })
      .catch(() => {
        if (!cancelled) setStatus(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'nearest' })
  }, [turns])

  const ask = useCallback(
    async (question: string, needsContext: boolean) => {
      const trimmed = question.trim()
      if (!trimmed || busy) return

      const id = nextId.current++
      setTurns((prev) => [
        ...prev,
        { id, question: trimmed, response: null, error: null, loading: true },
      ])
      setInput('')
      setBusy(true)

      try {
        const response = await api.assistantChat({
          message: trimmed,
          // Only send location context when the question is about the user's
          // own situation; general knowledge questions stay context-free.
          ...(needsContext && slug ? { location_slug: slug } : {}),
          ...(needsContext && activity ? { activity } : {}),
          ...(needsContext && durationMinutes ? { duration_minutes: durationMinutes } : {}),
        })
        setTurns((prev) =>
          prev.map((turn) => (turn.id === id ? { ...turn, response, loading: false } : turn)),
        )
      } catch (cause) {
        const message = cause instanceof ApiError ? cause.message : String(cause)
        setTurns((prev) =>
          prev.map((turn) => (turn.id === id ? { ...turn, error: message, loading: false } : turn)),
        )
      } finally {
        setBusy(false)
      }
    },
    [busy, slug, activity, durationMinutes],
  )

  const suggestions: SuggestedQuestion[] = status?.suggested_questions ?? []
  const unavailable = status !== null && !status.ready

  return (
    <section className="card card-pad animate-fade-up" aria-label="Ask AirShield">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="stat-label">Ask AirShield</p>
          <h3 className="mt-1 text-lg font-semibold text-white">
            Questions about the air and this recommendation
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {status && (
            <span
              className={`chip ${
                unavailable ? 'bg-amber-400/20 text-amber-200' : 'bg-sky-500/20 text-sky-200'
              }`}
              title={status.detail ?? undefined}
            >
              {unavailable ? 'knowledge base unavailable' : 'knowledge base ready'}
            </span>
          )}
          {status && (
            <span
              className={`chip ${
                status.llm_available
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-white/10 text-slate-300'
              }`}
              title={status.llm_detail ?? undefined}
            >
              {status.llm_available ? `LLM: ${status.llm_model}` : 'no LLM configured'}
            </span>
          )}
        </div>
      </div>

      {/* State the grounding honestly, before the user asks anything. */}
      {unavailable && (
        <p role="alert" className="mt-3 rounded-xl border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-100">
          {status?.detail ??
            'The knowledge index is not available, so the assistant cannot retrieve verified information.'}
        </p>
      )}
      {status && status.ready && !status.llm_available && (
        <p className="mt-3 rounded-xl border border-white/10 bg-ink-900/50 p-3 text-xs text-slate-300">
          {status.llm_detail ??
            'No language model is configured. Answers are shown as retrieved source passages.'}
        </p>
      )}
      {status?.notice && (
        <p className="mt-3 rounded-xl border border-amber-400/30 bg-amber-400/10 p-3 text-xs text-amber-100">
          {status.notice}
        </p>
      )}

      {/* Contextual buttons, driven by the backend's own list. */}
      {suggestions.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {suggestions.map((item) => (
            <button
              key={item.id}
              type="button"
              disabled={busy || unavailable}
              onClick={() => ask(item.question, item.needs_context)}
              className="rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {item.label}
            </button>
          ))}
        </div>
      )}

      <form
        className="mt-4 flex flex-col gap-2 sm:flex-row"
        onSubmit={(event) => {
          event.preventDefault()
          ask(input, true)
        }}
      >
        <label className="sr-only" htmlFor="ask-airshield-input">
          Ask a question about air quality or this recommendation
        </label>
        <input
          id="ask-airshield-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask about PM2.5, AQI, or why this window was recommended…"
          disabled={busy || unavailable}
          className="flex-1 rounded-xl border border-white/15 bg-ink-900/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-400/60 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={busy || unavailable || input.trim().length === 0}
          className="rounded-xl bg-sky-500/90 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </form>

      {/* Conversation */}
      <div className="mt-4 space-y-4">
        {turns.length === 0 && !unavailable && (
          <p className="text-xs text-slate-500">
            AirShield answers from a curated knowledge base of WHO, US EPA, AirNow and
            peer-reviewed material, plus its own forecast and exposure outputs. It will
            tell you when it does not have verified information.
          </p>
        )}

        {turns.map((turn) => (
          <div key={turn.id} className="space-y-2">
            <p className="text-sm font-medium text-slate-200">
              <span className="text-slate-500">You: </span>
              {turn.question}
            </p>

            {turn.loading && (
              <p className="text-sm text-slate-400" aria-live="polite">
                Retrieving from the knowledge base…
              </p>
            )}

            {turn.error && (
              <div role="alert" className="rounded-xl border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-100">
                <p className="font-semibold text-red-200">The assistant could not answer</p>
                <p className="mt-1">{turn.error}</p>
              </div>
            )}

            {turn.response && <AnswerBlock response={turn.response} />}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <p className="mt-4 border-t border-white/10 pt-3 text-xs text-slate-500">
        The assistant explains AirShield's outputs; it does not produce them. Forecasts,
        AQI and exposure scores are computed by the XGBoost model and the exposure engine.
        This is not medical advice.
      </p>
    </section>
  )
}

function AnswerBlock({ response }: { response: AssistantChatResponse }) {
  const refused = response.insufficient_knowledge
  const failed = response.mode === 'llm_error'

  return (
    <div
      className={`rounded-xl border p-3 ${
        refused || failed
          ? 'border-amber-400/40 bg-amber-400/10'
          : 'border-white/10 bg-ink-900/50'
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="chip bg-white/10 text-slate-300">
          {MODE_LABEL[response.mode] ?? response.mode}
        </span>
        {response.grounded && (
          <span className="chip bg-emerald-500/20 text-emerald-300">
            {response.retrieved_chunks} passage{response.retrieved_chunks === 1 ? '' : 's'} retrieved
          </span>
        )}
      </div>

      <div className="mt-2 whitespace-pre-wrap text-sm text-slate-100">{response.answer}</div>

      {response.context_error && (
        <p className="mt-2 text-xs text-amber-200/90">
          AirShield could not load live context: {response.context_error}
        </p>
      )}
      {response.notice && !response.context_error && (
        <p className="mt-2 text-xs text-slate-400">{response.notice}</p>
      )}

      {/* Only sources that were actually retrieved are ever listed. */}
      {response.sources.length > 0 && (
        <div className="mt-3 border-t border-white/10 pt-3">
          <p className="stat-label">Sources</p>
          <ul className="mt-2 space-y-1.5">
            {response.sources.map((source) => (
              <li key={source.doc_id} className="text-xs">
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="font-medium text-sky-300 underline decoration-sky-400/40 hover:text-sky-200"
                >
                  {source.title}
                </a>
                <span className="text-slate-400"> — {source.source}</span>
                {source.publication_date && (
                  <span className="text-slate-500"> ({source.publication_date})</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
