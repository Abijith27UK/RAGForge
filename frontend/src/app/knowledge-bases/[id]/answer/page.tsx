"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronDown, CircleSlash, FileText, Quote, Search, ShieldCheck, TriangleAlert,
} from "lucide-react";
import {
  api, AnswerResponse, AnswerTrace, Citation, Evidence, EvidenceAssessment,
} from "@/lib/api";
import { Badge, Button, EmptyState, Panel, PanelHeader, ScoreBar, StatusPill } from "@/components/ui";
import { fmtScore } from "@/lib/utils";

const STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_reranked"] as const;

const STATUS_TONE: Record<string, "ok" | "warn" | "bad" | "accent" | "neutral"> = {
  grounded: "ok",
  partial: "warn",
  abstained: "bad",
  clarification_required: "warn",
  generation_failed: "bad",
};

const STAGE_LABELS: Record<string, string> = {
  query_processing: "Query processing",
  retrieval: "Retrieval",
  evidence_assembly: "Evidence selection",
  evidence_gate: "Grounding gate",
  generation: "Generation",
  citation_validation: "Citation validation",
  finalization: "Finalization",
};

/* --------------------------------------------------------------- pieces */

function SignalRow({ signal }: { signal: EvidenceAssessment["signals"][number] }) {
  if (!signal.measured) {
    return (
      <div className="flex items-start gap-2 py-1 text-2xs">
        <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-ink-ghost" />
        <span className="text-ink-faint">
          {signal.name}: not performed — {signal.not_performed_reason}
        </span>
      </div>
    );
  }
  const tone =
    signal.passed === true ? "text-ok" : signal.passed === false ? "text-warn" : "text-ink-muted";
  return (
    <div className="flex items-start gap-2 py-1 text-2xs">
      <span className={`mt-1 h-1 w-1 shrink-0 rounded-full ${signal.passed ? "bg-ok" : "bg-warn"}`} />
      <span className="text-ink-faint">
        <span className="text-ink-muted">{signal.name}</span>
        {signal.value !== null && (
          <span className="data-value mx-1 text-ink">{signal.value}</span>
        )}
        <span className={tone}>{signal.interpretation}</span>
      </span>
    </div>
  );
}

function GroundingPanel({ a }: { a: EvidenceAssessment }) {
  return (
    <Panel>
      <PanelHeader
        title="Grounding assessment"
        right={
          <div className="flex items-center gap-3">
            <Badge tone={a.sufficient ? "ok" : "bad"}>
              {a.sufficient ? "sufficient" : "insufficient"}
            </Badge>
            <Badge tone="neutral">{a.reason_code}</Badge>
          </div>
        }
      />
      <div className="px-4 py-3">
        <p className="mb-3 text-xs leading-5 text-ink-muted">{a.reason}</p>
        <div className="section-label mb-1">Signals (measured inputs to the decision)</div>
        <div className="divide-y divide-line/50">
          {a.signals.map((s) => (
            <SignalRow key={s.name} signal={s} />
          ))}
        </div>
        {a.unsupported_aspects.length > 0 && (
          <div className="mt-3 rounded border border-warn/40 bg-warn/10 px-2.5 py-2 text-2xs text-warn">
            Not covered by evidence: {a.unsupported_aspects.join(" · ")}
          </div>
        )}
        <p className="mt-3 text-2xs leading-4 text-ink-faint">
          Confidence is categorical ({a.confidence}), not a percentage — there is no validated
          statistical basis for a numeric grounding confidence. Lexical checks are heuristics;
          semantic entailment was <span className="text-ink-muted">not performed</span>.
        </p>
      </div>
    </Panel>
  );
}

function CitationCard({ c, evidence }: { c: Citation; evidence?: Evidence }) {
  return (
    <Panel className="p-3">
      <div className="mb-1.5 flex items-center gap-2">
        <Quote className="h-3 w-3 shrink-0 text-accent-soft" />
        <span className="data-value text-2xs text-accent-soft">{c.citation_id}</span>
        <Badge tone={c.validation === "provenance_valid" ? "ok" : "warn"}>
          {c.validation.replace(/_/g, " ")}
        </Badge>
        <span className="ml-auto text-2xs text-ink-faint">
          {c.page != null && `p. ${c.page} `}
          {c.slide != null && `slide ${c.slide} `}
          {c.section && `· ${c.section}`}
        </span>
      </div>
      <div className="mb-1 text-xs font-medium text-ink">{c.title || c.document_id}</div>
      <p className="line-clamp-3 text-2xs leading-4 text-ink-muted">{c.snippet}</p>
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-2xs text-ink-faint">
        <span className="data-value">{c.chunk_id}</span>
        {c.publisher && <span>· {c.publisher}</span>}
        {evidence && (
          <span className="ml-auto">
            score <span className="data-value text-ink-muted">{fmtScore(evidence.retrieval_score, 3)}</span>
          </span>
        )}
      </div>
      <div className="mt-1.5 text-2xs leading-4 text-ink-faint">{c.validation_detail}</div>
    </Panel>
  );
}

function TracePanel({ trace }: { trace: AnswerTrace }) {
  const [open, setOpen] = useState(false);
  return (
    <Panel>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-surface-3"
      >
        <span className="flex items-center gap-2">
          <ShieldCheck className="h-3.5 w-3.5 text-accent-soft" />
          <span className="section-label">How was this answer produced?</span>
        </span>
        <span className="flex items-center gap-2 text-2xs text-ink-faint">
          trace <span className="data-value">{trace.id}</span>
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22 }}
            className="overflow-hidden border-t border-line"
          >
            <div className="space-y-4 px-4 py-4">
              {/* stage timeline */}
              <div className="section-label">Pipeline stages</div>
              <ol className="space-y-1.5">
                {trace.stages.map((s) => (
                  <li key={s.name} className="flex items-start gap-2.5 text-2xs">
                    <span
                      className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${
                        s.status === "ok"
                          ? "bg-ok"
                          : s.status === "error"
                          ? "bg-bad"
                          : "bg-ink-ghost"
                      }`}
                    />
                    <span className="w-40 shrink-0 text-ink-muted">
                      {STAGE_LABELS[s.name] ?? s.name}
                    </span>
                    <span className="w-16 shrink-0">
                      <span
                        className={
                          s.status === "ok"
                            ? "text-ok"
                            : s.status === "error"
                            ? "text-bad"
                            : "text-ink-faint"
                        }
                      >
                        {s.status}
                      </span>
                    </span>
                    <span className="min-w-0 flex-1 text-ink-faint">{s.detail}</span>
                    {s.ms !== null && (
                      <span className="data-value shrink-0 text-ink-ghost">{s.ms.toFixed(1)}ms</span>
                    )}
                  </li>
                ))}
              </ol>

              {/* query plan */}
              {trace.query_plan && (
                <div>
                  <div className="section-label mb-1">Query plan</div>
                  <div className="rounded border border-line bg-surface-3 px-2.5 py-2 text-2xs leading-5 text-ink-muted">
                    <div>
                      original (preserved):{" "}
                      <span className="text-ink">{trace.query_plan.original_query}</span>
                    </div>
                    <div>
                      type:{" "}
                      <span className="data-value text-ink">{trace.query_plan.query_type}</span>{" "}
                      <span className="text-ink-faint">
                        via {trace.query_plan.classification_method} (heuristic)
                      </span>
                    </div>
                    <div>
                      terms:{" "}
                      <span className="data-value text-ink">
                        {trace.query_plan.extracted_terms.join(", ") || "—"}
                      </span>
                    </div>
                    {trace.query_plan.domain_terms.length > 0 && (
                      <div>
                        domain terms:{" "}
                        <span className="data-value text-accent-soft">
                          {trace.query_plan.domain_terms.join(", ")}
                        </span>
                      </div>
                    )}
                    {trace.query_plan.notes.map((n) => (
                      <div key={n} className="text-ink-faint">
                        note: {n}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* retrieval */}
              <div>
                <div className="section-label mb-1">Retrieval</div>
                <div className="rounded border border-line bg-surface-3 px-2.5 py-2 text-2xs leading-5 text-ink-muted">
                  <div>
                    strategy: <span className="data-value text-ink">{trace.retrieval_strategy}</span>
                    {trace.retrieval_run_id && (
                      <>
                        {" · "}run:{" "}
                        <span className="data-value text-ink">{trace.retrieval_run_id}</span>
                      </>
                    )}
                  </div>
                  {trace.retrieval_params && (
                    <div className="mt-1 text-ink-faint">
                      params:{" "}
                      <span className="data-value">
                        {Object.entries(trace.retrieval_params)
                          .filter(([k]) => ["top_k", "min_score", "fusion", "normalization", "reranker", "diversity"].includes(k))
                          .map(([k, v]) => `${k}=${String(v)}`)
                          .join(" ")}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* evidence selection */}
              <div>
                <div className="section-label mb-1">
                  Evidence selection ({trace.evidence_ids.length} kept
                  {trace.evidence_dropped.length > 0 && `, ${trace.evidence_dropped.length} deduplicated`})
                </div>
                <div className="rounded border border-line bg-surface-3 px-2.5 py-2 text-2xs leading-5 text-ink-muted">
                  {trace.evidence_notes.map((n) => (
                    <div key={n}>{n}</div>
                  ))}
                  {trace.evidence_dropped.map((d) => (
                    <div key={d.evidence_id} className="text-warn">
                      dropped {d.evidence_id} ({d.reason})
                    </div>
                  ))}
                </div>
              </div>

              {/* generator */}
              <div>
                <div className="section-label mb-1">Generation</div>
                <div className="rounded border border-line bg-surface-3 px-2.5 py-2 text-2xs leading-5 text-ink-muted">
                  <div>
                    generator: <span className="data-value text-ink">{trace.generator || "—"}</span>
                    {trace.generator_model && <> · model: {trace.generator_model}</>}
                    {trace.is_mock && <Badge tone="warn">mock</Badge>}
                  </div>
                  {trace.generation_notes.map((n) => (
                    <div key={n}>{n}</div>
                  ))}
                  {trace.generation_warnings.map((n) => (
                    <div key={n} className="text-warn">
                      {n}
                    </div>
                  ))}
                  {trace.raw_generated_text && (
                    <details className="mt-1">
                      <summary className="cursor-pointer text-ink-faint">
                        raw generator output (pre-validation)
                      </summary>
                      <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-ink-ghost">
                        {trace.raw_generated_text}
                      </pre>
                    </details>
                  )}
                </div>
              </div>

              {/* citation validation */}
              <div>
                <div className="section-label mb-1">Citation validation</div>
                <div className="rounded border border-line bg-surface-3 px-2.5 py-2 text-2xs leading-5 text-ink-muted">
                  {trace.validation_actions.length === 0 ? (
                    <div>no repair actions were needed</div>
                  ) : (
                    trace.validation_actions.map((a) => <div key={a}>{a}</div>)
                  )}
                  {trace.citation_problems.map((p, i) => (
                    <div key={i} className="text-warn">
                      {JSON.stringify(p)}
                    </div>
                  ))}
                  <div className="mt-1 text-ink-faint">
                    semantic support check: NOT PERFORMED (V7 validates provenance and lexical
                    overlap only)
                  </div>
                </div>
              </div>

              {/* notes */}
              {trace.notes.length > 0 && (
                <div>
                  <div className="section-label mb-1">Trace notes</div>
                  <div className="text-2xs leading-5 text-ink-faint">
                    {trace.notes.map((n) => (
                      <div key={n}>{n}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Panel>
  );
}

/* ------------------------------------------------------------------ page */

export default function AnswerPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState<string>("dense");
  const [resp, setResp] = useState<AnswerResponse | null>(null);
  const [trace, setTrace] = useState<AnswerTrace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setTrace(null);
    try {
      const r = await api.answer(kbId, question, { strategy });
      setResp(r);
      try {
        setTrace(await api.getAnswerTrace(kbId, r.answer_trace_id));
      } catch {
        setTrace(null); // trace fetch is best-effort; the answer still renders
      }
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  const evById = new Map((resp?.evidence ?? []).map((e) => [e.evidence_id, e]));

  return (
    <div className="mx-auto max-w-4xl">
      <div className="mb-2">
        <div className="section-label mb-2">Research</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Answer</h1>
        <p className="mt-1 text-2xs leading-4 text-ink-faint">
          Ask a question against this knowledge base. The answer is grounded in retrieved evidence
          only — when the evidence is insufficient, the system says so instead of guessing.
        </p>
      </div>

      <form onSubmit={ask} className="mb-3 flex gap-2">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-ghost" />
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            required
            className="h-11 w-full rounded-lg border border-line-strong bg-surface-2 pl-9 pr-3 text-sm text-ink placeholder:text-ink-ghost focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
            placeholder="What is the effect of free surface on ship stability?"
          />
        </div>
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          className="h-11 rounded-lg border border-line-strong bg-surface-2 px-2 text-xs text-ink"
        >
          {STRATEGIES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <Button type="submit" variant="primary" loading={busy} className="h-11 px-5">
          Ask
        </Button>
      </form>

      {error && (
        <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
          {error}
        </div>
      )}

      {resp && (
        <div className="space-y-3">
          {/* ---- answer ---- */}
          <Panel>
            <PanelHeader
              title="Answer"
              right={
                <div className="flex items-center gap-3">
                  <StatusPill status={resp.status} />
                  {resp.generation_metadata.is_mock && <Badge tone="warn">mock generator</Badge>}
                </div>
              }
            />
            <div className="px-4 py-4">
              {resp.status === "abstained" && (
                <div className="mb-3 flex items-start gap-2 rounded border border-bad/40 bg-bad/10 px-3 py-2.5">
                  <CircleSlash className="mt-0.5 h-3.5 w-3.5 shrink-0 text-bad" />
                  <span className="text-2xs leading-4 text-bad">
                    The knowledge base does not contain sufficient evidence for this question.
                    Nothing was filled in from outside the corpus.
                  </span>
                </div>
              )}
              <p className="whitespace-pre-wrap text-sm leading-6 text-ink">{resp.answer.text}</p>

              {/* inline citations */}
              {resp.claims.length > 0 && (
                <div className="mt-4 space-y-2">
                  <div className="section-label">Claims</div>
                  {resp.claims.map((cl) => (
                    <div
                      key={cl.claim_id}
                      className="rounded border border-line bg-surface-3 px-2.5 py-2"
                    >
                      <div className="flex items-start gap-2">
                        <span className="min-w-0 flex-1 text-xs leading-5 text-ink">{cl.text}</span>
                        <Badge
                          tone={
                            cl.support_status === "supported"
                              ? "ok"
                              : cl.support_status === "partially_supported"
                              ? "warn"
                              : "bad"
                          }
                        >
                          {cl.support_status.replace(/_/g, " ")}
                        </Badge>
                      </div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        <Badge tone="neutral">{cl.claim_type}</Badge>
                        {cl.citation_ids.map((cid) => (
                          <Badge key={cid} tone="accent">
                            <Quote className="h-2.5 w-2.5" /> {cid}
                          </Badge>
                        ))}
                        <span className="text-2xs text-ink-faint">{cl.support_note}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {(resp.warnings.length > 0 || resp.answer.warnings.length > 0) && (
                <div className="mt-3 space-y-1">
                  {[...resp.warnings, ...resp.answer.warnings].map((w) => (
                    <div
                      key={w}
                      className="flex items-start gap-2 rounded border border-warn/40 bg-warn/10 px-2.5 py-1.5 text-2xs leading-4 text-warn"
                    >
                      <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" />
                      {w}
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-ink-faint">
                <span>
                  confidence: <span className="text-ink-muted">{resp.answer.confidence}</span>
                </span>
                <span>
                  generated by:{" "}
                  <span className="data-value text-ink-muted">
                    {resp.generation_metadata.generated_by || "—"}
                  </span>
                </span>
                <span>
                  strategy: <span className="data-value text-ink-muted">{strategy}</span>
                </span>
                {resp.retrieval_run_id && (
                  <span>
                    retrieval run:{" "}
                    <span className="data-value text-ink-muted">{resp.retrieval_run_id}</span>
                  </span>
                )}
              </div>
              {resp.answer.confidence_basis && (
                <p className="mt-1.5 text-2xs leading-4 text-ink-faint">
                  basis: {resp.answer.confidence_basis}
                </p>
              )}
            </div>
          </Panel>

          {/* ---- grounding ---- */}
          <GroundingPanel a={resp.grounding_assessment} />

          {/* ---- citations / sources ---- */}
          {resp.citations.length > 0 && (
            <div>
              <div className="section-label mb-2">Sources</div>
              <div className="grid gap-2 sm:grid-cols-2">
                {resp.citations.map((c) => (
                  <CitationCard
                    key={c.citation_id}
                    c={c}
                    evidence={evById.get(c.evidence_id)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* ---- evidence ---- */}
          {resp.evidence.length > 0 && (
            <div>
              <div className="section-label mb-2">
                Retrieved evidence ({resp.evidence.length})
              </div>
              <div className="space-y-2">
                {resp.evidence.map((ev) => (
                  <Panel key={ev.evidence_id} className="p-3">
                    <div className="mb-1.5 flex items-center gap-2">
                      <span className="data-value text-2xs text-accent-soft">{ev.evidence_id}</span>
                      <span className="text-2xs text-ink-muted">{ev.title || ev.document_id}</span>
                      <span className="ml-auto flex items-center gap-2">
                        {ev.page != null && <Badge tone="neutral">p. {ev.page}</Badge>}
                        {ev.slide != null && <Badge tone="neutral">slide {ev.slide}</Badge>}
                        {ev.dedup_reason && <Badge tone="warn">deduplicated</Badge>}
                        <span className="data-value text-2xs text-ink-muted">
                          {fmtScore(ev.retrieval_score, 3)}
                        </span>
                      </span>
                    </div>
                    <ScoreBar value={ev.retrieval_score} tone="accent" />
                    <p className="mt-2 line-clamp-4 whitespace-pre-wrap text-2xs leading-4 text-ink-muted">
                      {ev.content}
                    </p>
                    <div className="mt-1.5 flex flex-wrap gap-1.5 text-2xs text-ink-faint">
                      <span className="data-value">{ev.chunk_id}</span>
                      <span>· rank {ev.rank} (retrieval #{ev.original_rank})</span>
                      {ev.content_hash && (
                        <span className="data-value">· hash {ev.content_hash.slice(0, 10)}</span>
                      )}
                      {ev.trust_score !== null && (
                        <span>· trust {ev.trust_score} (heuristic)</span>
                      )}
                    </div>
                  </Panel>
                ))}
              </div>
            </div>
          )}

          {/* ---- trace ---- */}
          {trace ? (
            <TracePanel trace={trace} />
          ) : (
            <Panel className="px-4 py-3 text-2xs text-ink-faint">
              Trace unavailable for this answer (fetch failed or answer not persisted).
            </Panel>
          )}
        </div>
      )}

      {!resp && !error && (
        <EmptyState
          icon={<FileText className="h-6 w-6" />}
          title="Ask a question"
          hint="Every answer shows its grounding assessment, claim-level citations, the exact evidence used, and a full trace of how it was produced."
        />
      )}
    </div>
  );
}
