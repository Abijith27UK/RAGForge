"use client";

/**
 * Grounded chat — a RAG investigation interface, not a chat clone.
 *
 * Layout is deliberate (left corpus / centre conversation / right evidence):
 * every answer must be inspectable next to the evidence that produced it, so
 * the user can verify a claim instead of trusting a fluent paragraph.
 *
 * Honesty rules honoured by this UI (see docs/grounding-and-citations.md):
 *  - grounding state is shown as a categorical state, never a fake percentage;
 *  - a missing page/slide/section renders as "not recorded", never a guess;
 *  - a mock or fallback generator is labelled, because a deterministic mock is
 *    not the same thing as a language model;
 *  - "Why this answer?" reveals the real gate signals, including the ones that
 *    were NOT measured and why.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import {
  BookOpen, ChevronDown, CircleSlash, FileText, MessagesSquare, Scale,
  Send, ShieldCheck, SquareStack, TriangleAlert,
} from "lucide-react";
import {
  api, Answer, AnswerRun, ChatMessage, ChatResponse, Citation, ConversationSummary,
  Evidence, Grounding, GroundingState, KBOverview, QueryTrace,
} from "@/lib/api";
import { Badge, Button, EmptyState, Panel, PanelHeader, ScoreBar } from "@/components/ui";
import { cn, fmtScore } from "@/lib/utils";

const STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_reranked"] as const;

/** The five grounding states, with the tone the UI renders them in. */
const STATE_TONE: Record<GroundingState, "ok" | "warn" | "bad" | "accent" | "neutral"> = {
  ANSWERED: "ok",
  PARTIALLY_SUPPORTED: "warn",
  CONFLICTING_EVIDENCE: "warn",
  INSUFFICIENT_EVIDENCE: "bad",
  NO_RELEVANT_EVIDENCE: "bad",
};

const STATE_HINT: Record<GroundingState, string> = {
  ANSWERED: "The retrieved evidence supports the question.",
  PARTIALLY_SUPPORTED: "Only part of the question is covered by the evidence.",
  CONFLICTING_EVIDENCE: "Sources disagree. The disagreement is reported, not resolved.",
  INSUFFICIENT_EVIDENCE: "This knowledge base did not contain enough evidence to answer.",
  NO_RELEVANT_EVIDENCE: "Nothing relevant to this question was found in the corpus.",
};

/* ----------------------------------------------------------------- pieces */

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1 text-2xs">
      <span className="shrink-0 text-ink-faint">{label}</span>
      <span className="min-w-0 truncate text-right text-ink-muted" title={typeof value === "string" ? value : undefined}>
        {value}
      </span>
    </div>
  );
}

/** "not recorded" — never an invented page number. */
function Provenance({ citation: c }: { citation: Citation }) {
  const page = c.page_number ?? c.page;
  const slide = c.slide_number ?? c.slide;
  return (
    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-2xs text-ink-faint">
      <span className="text-ink-muted">{c.source_title || c.document_id}</span>
      <span>{page != null ? `p. ${page}` : "page not recorded"}</span>
      <span>{slide != null ? `slide ${slide}` : "no slide"}</span>
      {c.section_path ? <span>§ {c.section_path}</span> : null}
      {c.source_type ? <span>{c.source_type}</span> : null}
    </div>
  );
}

function GroundingBadge({ grounding }: { grounding: Grounding }) {
  const tone = STATE_TONE[grounding.state];
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={tone}>{grounding.state.replace(/_/g, " ")}</Badge>
        <span className="text-2xs text-ink-faint">
          {grounding.evidence_count} chunk{grounding.evidence_count === 1 ? "" : "s"} ·{" "}
          {grounding.document_count} document{grounding.document_count === 1 ? "" : "s"}
        </span>
      </div>
      <p className="text-xs leading-relaxed text-ink-muted">{STATE_HINT[grounding.state]}</p>
    </div>
  );
}

/** "Why this answer?" — the real, measured gate signals. */
function WhyPanel({
  grounding, queryTrace, run,
}: {
  grounding: Grounding;
  queryTrace: QueryTrace;
  run: AnswerRun | null;
}) {
  return (
    <div className="space-y-3 text-2xs">
      <section>
        <div className="section-label mb-1">Grounding reason</div>
        <div className="rounded border border-line bg-surface-1 px-2.5 py-2 text-ink-muted">
          <code className="text-accent-soft">{grounding.reason_code}</code>
          <p className="mt-1 leading-relaxed">{grounding.reason}</p>
        </div>
      </section>

      {grounding.unsupported_aspects.length > 0 ? (
        <section>
          <div className="section-label mb-1">Not covered by the evidence</div>
          <ul className="space-y-1 text-ink-muted">
            {grounding.unsupported_aspects.map((a) => (
              <li key={a} className="flex gap-1.5">
                <CircleSlash className="mt-0.5 size-3 shrink-0 text-warn" />
                <span>{a}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section>
        <div className="section-label mb-1">Gate signals</div>
        {grounding.signals.length === 0 ? (
          <p className="text-ink-faint">
            No evidence signals were computed — this turn was not treated as a knowledge
            question.
          </p>
        ) : (
          <div className="space-y-1.5">
            {grounding.signals.map((s) => (
              <div key={s.name} className="rounded border border-line bg-surface-1 px-2.5 py-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ink-muted">{s.name.replace(/_/g, " ")}</span>
                  {s.measured ? (
                    <span className="flex items-center gap-1.5">
                      {s.value != null ? (
                        <span className="text-ink">{fmtScore(s.value)}</span>
                      ) : null}
                      <span
                        className={cn(
                          "text-2xs",
                          s.passed === true ? "text-ok" : s.passed === false ? "text-warn" : "text-ink-faint"
                        )}
                      >
                        {s.passed === null ? "—" : s.passed ? "pass" : "fail"}
                      </span>
                    </span>
                  ) : (
                    <span className="text-ink-faint" title={s.not_performed_reason}>
                      not measured
                    </span>
                  )}
                </div>
                <p className="mt-0.5 leading-relaxed text-ink-faint">
                  {s.measured ? s.interpretation : s.not_performed_reason}
                </p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <div className="section-label mb-1">Query processing</div>
        <div className="space-y-0.5 rounded border border-line bg-surface-1 px-2.5 py-2">
          <Row label="Processor" value={`${queryTrace.processor} ${queryTrace.processor_version}`} />
          <Row label="Turn type" value={queryTrace.nature.replace(/_/g, " ")} />
          <Row label="Query type" value={queryTrace.query_type} />
          <Row label="Rewritten" value={queryTrace.rewritten_query ?? "no — passed through unchanged"} />
          <Row label="Subqueries" value={queryTrace.subqueries.length ? queryTrace.subqueries.join(" | ") : "none"} />
          <Row label="Transformations" value={queryTrace.transformations.length ? queryTrace.transformations.join(", ") : "none"} />
          <Row label="Processing time" value={queryTrace.timing_ms != null ? `${queryTrace.timing_ms} ms` : "not measured"} />
          <Row label="Retrieval strategy" value={run?.strategy || "knowledge base default"} />
        </div>
      </section>

      {run ? (
        <section>
          <div className="section-label mb-1">Answer run (measured)</div>
          <div className="space-y-0.5 rounded border border-line bg-surface-1 px-2.5 py-2">
            <Row label="Answer run" value={run.id} />
            <Row label="Retrieval run" value={run.retrieval_run_id ?? "none"} />
            <Row label="Retrieval" value={run.retrieval_ms != null ? `${run.retrieval_ms} ms` : "not measured"} />
            <Row label="Evidence selection" value={run.evidence_selection_ms != null ? `${run.evidence_selection_ms} ms` : "not measured"} />
            <Row label="Grounding gate" value={run.grounding_ms != null ? `${run.grounding_ms} ms` : "not measured"} />
            <Row label="Generation" value={run.generation_ms != null ? `${run.generation_ms} ms` : "skipped / not measured"} />
            <Row label="Citation validation" value={run.citation_validation_ms != null ? `${run.citation_validation_ms} ms` : "not measured"} />
            <Row label="Total" value={run.total_ms != null ? `${run.total_ms} ms` : "not measured"} />
            <Row label="Prompt version" value={run.prompt_version} />
            <Row label="Answerer version" value={run.answerer_version} />
            <Row label="Evidence selector" value={run.evidence_selector} />
          </div>
        </section>
      ) : null}
    </div>
  );
}

function Disclaimer({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2 rounded border border-warn/30 bg-warn/5 px-2.5 py-2 text-2xs leading-relaxed text-ink-muted">
      <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warn" />
      <span>{text}</span>
    </div>
  );
}

/* --------------------------------------------------------------- chat turn */

interface Turn {
  id: string;
  question: string;
  response: ChatResponse;
  run: AnswerRun | null;
}

function TurnCard({
  turn, onInspect, inspected,
}: {
  turn: Turn;
  onInspect: (evidenceId: string | null) => void;
  inspected: string | null;
}) {
  const [showWhy, setShowWhy] = useState(false);
  const { response: r } = turn;
  const tone = STATE_TONE[r.grounding.state];
  const isGrounded = r.grounding.sufficient;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-2.5"
    >
      {/* user question */}
      <div className="ml-8 rounded-lg border border-line bg-surface-2 px-3 py-2">
        <p className="text-xs leading-relaxed text-ink">{turn.question}</p>
      </div>

      {/* grounding header */}
      <div className="rounded-lg border border-line bg-surface-1 px-3 py-2.5">
        <GroundingBadge grounding={r.grounding} />
      </div>

      {/* answer text — or the abstention message */}
      {isGrounded ? (
        <div className="rounded-lg border border-line bg-surface-2 px-3.5 py-3">
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{r.answer}</p>
        </div>
      ) : (
        <div
          className={cn(
            "rounded-lg border px-3.5 py-3",
            tone === "bad" ? "border-warn/30 bg-warn/5" : "border-line bg-surface-2"
          )}
        >
          <p className="flex items-start gap-2 text-sm leading-relaxed text-ink">
            <ShieldCheck className="mt-0.5 size-4 shrink-0 text-warn" />
            <span>
              I couldn&apos;t find enough information in this knowledge base to answer that
              reliably.
            </span>
          </p>
          <p className="mt-1.5 pl-6 text-xs leading-relaxed text-ink-muted">{r.grounding.reason}</p>
        </div>
      )}

      {/* claim support */}
      {r.claims.length > 0 ? (
        <div className="space-y-1">
          {r.claims.map((c) => (
            <div key={c.claim_id} className="flex items-start gap-2 px-1 text-2xs">
              <Badge tone={c.support_status === "supported" ? "ok" : "warn"}>
                {c.support_status.replace(/_/g, " ")}
              </Badge>
              <span className="min-w-0 flex-1 leading-relaxed text-ink-muted">{c.text}</span>
            </div>
          ))}
          <p className="px-1 text-2xs text-ink-faint">
            Support is checked by lexical overlap — a deterministic heuristic. Semantic
            entailment is not performed.
          </p>
        </div>
      ) : null}

      {/* citations */}
      {r.citations.length > 0 ? (
        <div className="space-y-1">
          {r.citations.map((c, i) => (
            <button
              key={c.citation_id}
              onClick={() => onInspect(c.evidence_id)}
              className={cn(
                "flex w-full items-start gap-2 rounded border px-2.5 py-1.5 text-left transition-colors",
                inspected === c.evidence_id
                  ? "border-accent/40 bg-accent/5"
                  : "border-line bg-surface-1 hover:border-line-strong"
              )}
            >
              <span className="shrink-0 font-mono text-2xs text-accent-soft">[{i + 1}]</span>
              <span className="min-w-0 flex-1">
                <Provenance citation={c} />
              </span>
            </button>
          ))}
        </div>
      ) : null}

      {/* generator disclosure */}
      {r.generation.degraded ? (
        <Disclaimer
          text={`The configured generator was unavailable. This answer was produced by "${r.generation.generated_by}"${r.generation.is_mock ? " — a deterministic mock, not a language model" : ""}.`}
        />
      ) : null}
      {r.generation.is_mock && !r.generation.degraded ? (
        <p className="px-1 text-2xs text-ink-faint">
          Generated by the deterministic mock provider ({r.generation.model}). No language
          model was used.
        </p>
      ) : null}
      {r.warnings.map((w) => (
        <Disclaimer key={w} text={w} />
      ))}

      {/* why this answer */}
      <button
        onClick={() => setShowWhy((v) => !v)}
        className="flex items-center gap-1 px-1 text-2xs text-ink-faint transition-colors hover:text-ink-muted"
      >
        <ChevronDown className={cn("size-3 transition-transform", showWhy && "rotate-180")} />
        Why this answer?
      </button>
      {showWhy ? (
        <div className="rounded border border-line bg-surface-2 px-3 py-2.5">
          <WhyPanel
            grounding={r.grounding}
            queryTrace={r.query_trace}
            run={turn.run}
          />
        </div>
      ) : null}
    </motion.div>
  );
}

/* ------------------------------------------------------------- history */

/**
 * Rebuild displayable turns from a stored conversation.
 *
 * Stored messages hold the answer/trace links but NOT the full evidence
 * payload, so each assistant message is re-hydrated from its stored answer and
 * answer run. Anything that genuinely cannot be recovered (the original query
 * trace, when the message predates it) is marked as unavailable rather than
 * reconstructed from memory — an old turn must never show a fabricated trace.
 */
async function rehydrateTurns(
  kbId: string,
  messages: ChatMessage[]
): Promise<Turn[]> {
  const turns: Turn[] = [];
  for (let i = 0; i < messages.length; i += 1) {
    const message = messages[i];
    if (message.role !== "user") continue;
    const reply = messages[i + 1];
    if (!reply || reply.role !== "assistant" || !reply.answer_id) {
      turns.push({
        id: message.id,
        question: message.content,
        response: unavailableResponse(message, reply),
        run: null,
      });
      continue;
    }
    try {
      const [answer, run] = await Promise.all([
        api.getAnswer(kbId, reply.answer_id),
        reply.answer_run_id
          ? api.getAnswerRun(kbId, reply.answer_run_id).catch(() => null)
          : Promise.resolve(null),
      ]);
      turns.push({
        id: reply.answer_run_id ?? message.id,
        question: message.content,
        response: answerToChatResponse(answer, reply, run),
        run,
      });
    } catch {
      turns.push({
        id: message.id,
        question: message.content,
        response: unavailableResponse(message, reply),
        run: null,
      });
    }
  }
  return turns;
}

/** Minimal, HONEST response for a turn whose stored answer could not be read. */
function unavailableResponse(user: ChatMessage, reply?: ChatMessage): ChatResponse {
  return {
    answer: reply?.content ?? "",
    answer_id: reply?.answer_id ?? "",
    status: reply?.grounding_state ? "grounded" : "abstained",
    citations: [],
    claims: [],
    grounding: {
      state: reply?.grounding_state ?? "INSUFFICIENT_EVIDENCE",
      decision: "ABSTAIN",
      sufficient: false,
      confidence: "none",
      reason_code: "HISTORY_UNAVAILABLE",
      reason:
        "This stored turn could not be reloaded with its full provenance, so its evidence and citations are not shown here. Open the answer run for the recorded decision.",
      evidence_count: 0,
      document_count: 0,
      supporting_evidence_ids: [],
      unsupported_aspects: [],
      missing_information: [],
      recommended_action: "ask_clarification",
      signals: [],
      notes: [],
    },
    evidence: [],
    retrieval_run_id: reply?.retrieval_run_id ?? null,
    answer_run_id: reply?.answer_run_id ?? "",
    answer_trace_id: reply?.answer_trace_id ?? "",
    query_trace: {
      original_query: user.content,
      normalized_query: user.content,
      rewritten_query: null,
      subqueries: [],
      processor: "history",
      processor_version: "",
      query_type: "unknown",
      nature: "knowledge",
      classification_method: "unavailable",
      classification_signals: [],
      transformations: [],
      expanded_terms: [],
      warnings: [],
      notes: ["query trace not stored with this historical turn"],
      timing_ms: null,
      enabled: true,
    },
    conversation_id: user.conversation_id,
    user_message_id: user.id,
    assistant_message_id: reply?.id ?? "",
    generation: {
      generated_by: "unknown",
      model: "",
      is_mock: false,
      prompt_version: "",
      answerer_version: "",
      answer_mode: "",
      degraded: false,
      generator_status: "unknown",
    },
    warnings: [],
    created_at: reply?.created_at ?? user.created_at,
  };
}

/** Project a stored `Answer` back into the chat response shape. */
function answerToChatResponse(
  answer: Answer,
  reply: ChatMessage,
  run: AnswerRun | null,
): ChatResponse {
  const assessment = answer.assessment;
  return {
    answer: answer.text,
    answer_id: answer.answer_id,
    status: answer.status,
    citations: answer.citations,
    claims: answer.claims,
    grounding: {
      state: assessment?.grounding_state ?? "INSUFFICIENT_EVIDENCE",
      decision: assessment?.decision ?? "ABSTAIN",
      sufficient: assessment?.sufficient ?? false,
      confidence: assessment?.confidence ?? answer.confidence,
      reason_code: assessment?.reason_code ?? "UNAVAILABLE",
      reason: assessment?.reason ?? "Grounding assessment unavailable for this stored answer.",
      evidence_count: assessment?.evidence_count ?? 0,
      document_count: assessment?.document_count ?? 0,
      supporting_evidence_ids: assessment?.supporting_evidence_ids ?? [],
      unsupported_aspects: assessment?.unsupported_aspects ?? [],
      missing_information: assessment?.missing_information ?? [],
      recommended_action: assessment?.recommended_action ?? "ask_clarification",
      signals: assessment?.signals ?? [],
      notes: assessment?.notes ?? [],
    },
    // The stored answer does not carry the evidence text; the citation snippets
    // and chunk ids remain available, so provenance is not invented to fill it.
    evidence: [],
    retrieval_run_id: answer.retrieval_run_id,
    answer_run_id: reply.answer_run_id ?? "",
    answer_trace_id: answer.answer_trace_id,
    query_trace: {
      original_query: answer.question,
      normalized_query: answer.question,
      rewritten_query: null,
      subqueries: [],
      processor: run?.query_processor ?? "history",
      processor_version: run?.query_processor_version ?? "",
      query_type: "unknown",
      nature: "knowledge",
      classification_method: "unavailable",
      classification_signals: [],
      transformations: [],
      expanded_terms: [],
      warnings: [],
      notes: ["query trace not stored with this historical turn"],
      timing_ms: run?.query_processing_ms ?? null,
      enabled: true,
    },
    conversation_id: reply.conversation_id,
    user_message_id: "",
    assistant_message_id: reply.id,
    generation: {
      generated_by: answer.generated_by,
      model: answer.model,
      is_mock: answer.is_mock,
      prompt_version: answer.prompt_version,
      answerer_version: run?.answerer_version ?? "",
      answer_mode: answer.answer_mode,
      degraded: false,
      generator_status: answer.is_mock ? "mock" : "live",
    },
    warnings: answer.warnings,
    created_at: reply.created_at,
  };
}

/* ------------------------------------------------------------------- page */

export default function GroundedChatPage() {
  const params = useParams<{ id: string }>();
  const kbId = params?.id as string;

  const [overview, setOverview] = useState<KBOverview | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [strategy, setStrategy] = useState<string>("");
  const [conversationId, setConversationId] = useState<string>("");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspected, setInspected] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ turn: Turn; evidenceId: string } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!kbId) return;
    api.kbOverview(kbId).then(setOverview).catch(() => setOverview(null));
    api.listConversations(kbId).then(setConversations).catch(() => setConversations([]));
  }, [kbId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  const send = useCallback(async () => {
    const message = input.trim();
    if (!message || busy) return;
    setBusy(true);
    setError(null);
    try {
      const response = await api.chat(kbId, message, {
        ...(strategy ? { retrieval_strategy: strategy } : {}),
        ...(conversationId ? { conversation_id: conversationId } : {}),
      });
      setConversationId(response.conversation_id);
      setInput("");
      // Fetch the observability record for this run (best effort: the answer
      // itself must render even if this lookup fails).
      let run: AnswerRun | null = null;
      try {
        run = await api.getAnswerRun(kbId, response.answer_run_id);
      } catch {
        run = null;
      }
      const turn: Turn = { id: response.answer_run_id, question: message, response, run };
      setTurns((t) => [...t, turn]);
      setInspected(response.evidence[0]?.evidence_id ?? null);
      api.listConversations(kbId).then(setConversations).catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Chat request failed");
    } finally {
      setBusy(false);
    }
  }, [input, busy, kbId, strategy, conversationId]);

  const inspect = useCallback((turn: Turn, evidenceId: string | null) => {
    setInspected(evidenceId);
    if (evidenceId) setSelected({ turn, evidenceId });
  }, []);

  const latest = turns[turns.length - 1];
  const inspectedEvidence: Evidence | undefined = latest?.response.evidence.find(
    (e) => e.evidence_id === inspected
  );
  const selectedEvidence: Evidence | undefined = selected
    ? selected.turn.response.evidence.find((e) => e.evidence_id === selected.evidenceId)
    : undefined;

  return (
    <div className="flex h-[calc(100vh-var(--shell-top))] min-h-0">
      {/* ---------------------------------------------------- LEFT: corpus */}
      <aside className="hidden w-64 shrink-0 overflow-y-auto border-r border-line p-3 xl:block">
        <div className="space-y-3">
          <Panel>
            <PanelHeader title="Knowledge base" />
            <div className="space-y-2 p-3">
              <div>
                <div className="text-sm font-medium text-ink">{overview?.kb.name ?? "…"}</div>
                <div className="text-2xs text-ink-faint">{overview?.kb.domain ?? ""}</div>
              </div>
              <div className="space-y-0.5 border-t border-line pt-2">
                <Row label="Documents" value={overview?.documents ?? "—"} />
                <Row label="Chunks" value={overview?.chunks ?? "—"} />
                <Row label="Vectors" value={overview?.vectors ?? "—"} />
                <Row label="Vector store" value={overview?.vector_store_status ?? "—"} />
                <Row label="Embedding model" value={overview?.embedding_model ?? "—"} />
              </div>
            </div>
          </Panel>

          <Panel>
            <PanelHeader title="Retrieval" />
            <div className="space-y-2 p-3">
              <label className="section-label" htmlFor="strategy">
                Strategy
              </label>
              <select
                id="strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full rounded border border-line bg-surface-1 px-2 py-1.5 text-xs text-ink"
              >
                <option value="">Knowledge base default</option>
                {STRATEGIES.map((s) => (
                  <option key={s} value={s}>
                    {s.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
              <p className="text-2xs leading-relaxed text-ink-faint">
                Applied to each new question. The strategy actually used is recorded on
                every answer run.
              </p>
            </div>
          </Panel>

          <Panel>
            <PanelHeader title={<span className="flex items-center gap-1.5"><MessagesSquare className="size-3" />Conversations</span>} />
            <div className="space-y-1 p-2">
              <button
                onClick={() => {
                  setTurns([]);
                  setConversationId("");
                  setInspected(null);
                  setSelected(null);
                }}
                className="flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-left text-2xs text-ink-muted transition-colors hover:bg-surface-1"
              >
                <MessagesSquare className="size-3" /> New conversation
              </button>
              {conversations.map((c) => (
                <div
                  key={c.id}
                  className={cn(
                    "flex items-center justify-between gap-2 rounded px-2 py-1.5 text-2xs",
                    c.id === conversationId ? "bg-accent/5 text-ink" : "text-ink-muted"
                  )}
                >
                  <button
                    onClick={async () => {
                      setError(null);
                      try {
                        const detail = await api.getConversation(kbId, c.id);
                        setConversationId(c.id);
                        setSelected(null);
                        setInspected(null);
                        setTurns(await rehydrateTurns(kbId, detail.messages));
                      } catch (e) {
                        setError(
                          e instanceof Error ? e.message : "Could not load conversation"
                        );
                      }
                    }}
                    className="min-w-0 flex-1 truncate text-left"
                  >
                    {c.title || "Untitled conversation"}
                  </button>
                  <span className="shrink-0 text-ink-faint">{c.message_count}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </aside>

      {/* -------------------------------------------------- CENTER: conversation */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <div className="mx-auto max-w-2xl space-y-5">
            {turns.length === 0 ? (
              <EmptyState
                icon={<BookOpen className="size-6" />}
                title="Ask a question grounded in this corpus"
                hint="Every answer must cite retrieved evidence. If the knowledge base does not contain enough, the assistant will say so rather than answer from general knowledge."
              />
            ) : (
              turns.map((t) => (
                <TurnCard
                  key={t.id}
                  turn={t}
                  inspected={inspected}
                  onInspect={(eid) => inspect(t, eid)}
                />
              ))
            )}
            {busy ? (
              <p className="text-center text-2xs text-ink-faint">Retrieving and grounding…</p>
            ) : null}
            {error ? (
              <div className="rounded border border-warn/30 bg-warn/5 px-3 py-2 text-xs text-ink-muted">
                {error}
              </div>
            ) : null}
          </div>
        </div>

        <div className="border-t border-line p-3">
          <div className="mx-auto flex max-w-2xl items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              rows={2}
              placeholder="Ask a question about this knowledge base…  (Enter to send, Shift+Enter for a new line)"
              className="flex-1 resize-none rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-accent/40 focus:outline-none"
            />
            <Button onClick={() => void send()} loading={busy} disabled={!input.trim()}>
              <Send className="size-3.5" />
              Ask
            </Button>
          </div>
        </div>
      </main>

      {/* ------------------------------------------- RIGHT: evidence / sources */}
      <aside className="hidden w-80 shrink-0 overflow-y-auto border-l border-line p-3 lg:block">
        <div className="space-y-3">
          <Panel>
            <PanelHeader
              title={<span className="flex items-center gap-1.5"><SquareStack className="size-3" />Evidence</span>}
              right={
                latest ? (
                  <span className="text-2xs text-ink-faint">
                    {latest.response.evidence.length} selected
                  </span>
                ) : null
              }
            />
            {!latest || latest.response.evidence.length === 0 ? (
              <p className="p-3 text-2xs text-ink-faint">
                No evidence was retrieved for the last question.
              </p>
            ) : (
              <div className="space-y-1.5 p-2">
                {latest.response.evidence.map((e) => {
                  const cited = latest.response.citations.some(
                    (c) => c.evidence_id === e.evidence_id
                  );
                  return (
                    <button
                      key={e.evidence_id}
                      onClick={() => inspect(latest, e.evidence_id)}
                      className={cn(
                        "w-full rounded border px-2.5 py-2 text-left transition-colors",
                        inspected === e.evidence_id
                          ? "border-accent/40 bg-accent/5"
                          : "border-line bg-surface-1 hover:border-line-strong"
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-2xs text-ink">
                          {e.title || e.document_id}
                        </span>
                        {cited ? <Badge tone="ok">cited</Badge> : null}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 text-2xs text-ink-faint">
                        <span className="font-mono">{e.evidence_id}</span>
                        {e.page != null ? <span>p. {e.page}</span> : <span>no page</span>}
                        {e.slide != null ? <span>slide {e.slide}</span> : null}
                        <span className="ml-auto">{fmtScore(e.retrieval_score)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </Panel>

          <Panel>
            <PanelHeader
              title={<span className="flex items-center gap-1.5"><FileText className="size-3" />Chunk</span>}
            />
            {!selectedEvidence ? (
              <p className="p-3 text-2xs text-ink-faint">
                Select a citation or evidence item to inspect the exact supporting chunk.
              </p>
            ) : (
              <div className="space-y-2 p-3">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone="accent">{selectedEvidence.evidence_id}</Badge>
                  <Badge tone="neutral">{selectedEvidence.retrieval_strategy}</Badge>
                </div>
                <div className="text-2xs leading-relaxed text-ink-muted">
                  {selectedEvidence.title || selectedEvidence.document_id}
                </div>
                <div className="space-y-0.5 border-y border-line py-1.5">
                  <Row label="Chunk id" value={selectedEvidence.chunk_id} />
                  <Row label="Document" value={selectedEvidence.document_id} />
                  <Row label="Page" value={selectedEvidence.page != null ? String(selectedEvidence.page) : "not recorded"} />
                  <Row label="Slide" value={selectedEvidence.slide != null ? String(selectedEvidence.slide) : "not recorded"} />
                  <Row label="Section" value={selectedEvidence.section_path ?? "not recorded"} />
                  <Row label="Source type" value={selectedEvidence.source_type ?? "not recorded"} />
                  <Row label="Content hash" value={selectedEvidence.content_hash.slice(0, 16) || "—"} />
                  <Row label="Retrieval score" value={fmtScore(selectedEvidence.retrieval_score)} />
                  <Row label="Trust score" value={selectedEvidence.trust_score != null ? fmtScore(selectedEvidence.trust_score) : "not rated"} />
                </div>
                <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded border border-line bg-surface-1 p-2.5 text-2xs leading-relaxed text-ink-muted">
                  {selectedEvidence.content}
                </pre>
                {selectedEvidence.dedup_reason ? (
                  <Disclaimer text={`Deduplication: ${selectedEvidence.dedup_reason}`} />
                ) : null}
              </div>
            )}
          </Panel>

          {latest ? (
            <Panel>
              <PanelHeader title={<span className="flex items-center gap-1.5"><Scale className="size-3" />Provenance</span>} />
              <div className="space-y-0.5 p-3">
                <Row label="Retrieval run" value={latest.response.retrieval_run_id ?? "none"} />
                <Row label="Answer run" value={latest.response.answer_run_id} />
                <Row label="Answer trace" value={latest.response.answer_trace_id} />
                <Row label="Answer id" value={latest.response.answer_id} />
                <Row label="Status" value={latest.response.status} />
                {inspectedEvidence ? (
                  <>
                    <div className="section-label pt-2">Retrieved score</div>
                    <ScoreBar value={inspectedEvidence.retrieval_score} />
                    <p className="pt-1 text-2xs leading-relaxed text-ink-faint">
                      Pool-relative score from this retrieval run. Comparable within the run
                      only — not a probability.
                    </p>
                  </>
                ) : null}
              </div>
            </Panel>
          ) : null}
        </div>
      </aside>
    </div>
  );
}