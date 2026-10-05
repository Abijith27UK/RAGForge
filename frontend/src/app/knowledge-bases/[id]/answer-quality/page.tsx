"use client";

/**
 * Answer Quality — failure analysis over an answer benchmark.
 *
 * Research integrity rules mirrored from the backend:
 * - A metric that could not be measured is rendered as UNKNOWN with its reason.
 *   It is NEVER rendered as 0, because "we could not measure this" and "we
 *   measured this and it was zero" are different claims.
 * - Every run displays its producers (generator, model, mock flag, evaluator,
 *   entailment provider) so a number is never shown without knowing what made
 *   it.
 * - Runs over different question subsets cannot be compared; the API refuses,
 *   and this page does not attempt to work around it.
 * - Expected abstention is never counted as a failure.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  ChevronDown, ChevronRight, CircleAlert, CircleCheck, FlaskConical, RefreshCw, Sigma,
} from "lucide-react";
import {
  answerEvaluationApi,
  AnswerEvaluationRun,
  AnswerQualityAggregate,
  AnswerQualityResult,
  AnswerReview,
  ClaimCitationVerdict,
  Measured,
  REVIEW_LABELS,
  REVIEW_VERDICTS,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  Badge, Button, EmptyState, Metric, Panel, PanelHeader, ScoreBar, StatusDot,
} from "@/components/ui";

const BENCHMARK_PATH = "benchmarks/answer-quality-automobile-v1.json";

/** Render a Measured. UNKNOWN is visually distinct from any number. */
function MetricValue({ m, digits = 3 }: { m: Measured | undefined; digits?: number }) {
  if (!m || !m.measured || m.value === null) {
    return (
      <span className="inline-flex items-center gap-1 text-ink-faint" title={m?.reason ?? ""}>
        UNKNOWN
      </span>
    );
  }
  return (
    <span title={m.reason}>
      {m.value.toFixed(digits)}
      {m.sample_size != null && (
        <span className="ml-1 text-2xs text-ink-faint">n={m.sample_size}</span>
      )}
    </span>
  );
}

/** Higher is better for these metrics; lower is better for the rest. */
const HIGHER_IS_BETTER: Record<string, boolean> = {
  citation_precision: true,
  citation_recall: true,
  citation_completeness: true,
  evidence_support_rate: true,
  retrieval_hit_rate: true,
  abstention_accuracy: true,
  grounding_state_accuracy: true,
  pass_rate: true,
  question_answer_relevance: true,
  supported_claim_ratio: true,
  expected_information_coverage: true,
  key_point_recall: true,
  unsupported_claim_rate: false,
  unsupported_claim_ratio: false,
  fabricated_citation_rate: false,
  unsupported_citation_rate: false,
  relevance_failure_rate: false,
  contradiction_rate: false,
  false_supported_rate: false,
  false_unsupported_rate: false,
  hallucination_rate: false,
};

function toneFor(name: string, m: Measured | undefined): "ok" | "warn" | "accent" {
  if (!m || !m.measured || m.value === null) return "accent";
  const higherBetter = HIGHER_IS_BETTER[name] ?? true;
  const good = higherBetter ? m.value >= 0.8 : m.value <= 0.1;
  const bad = higherBetter ? m.value < 0.5 : m.value > 0.25;
  if (good) return "ok";
  if (bad) return "warn";
  return "accent";
}

function MetricCard({ name, m, hint }: { name: string; m: Measured | undefined; hint?: string }) {
  return (
    <div className="rounded-md border border-line bg-surface-2 px-3 py-2.5">
      <div className="section-label mb-1">{name.replace(/_/g, " ")}</div>
      <div
        className={cn(
          "data-value font-medium text-lg leading-6",
          toneFor(name, m) === "ok" && "text-ok",
          toneFor(name, m) === "warn" && "text-warn"
        )}
      >
        <MetricValue m={m} />
      </div>
      {hint && <div className="mt-0.5 text-2xs text-ink-faint">{hint}</div>}
    </div>
  );
}

function ClaimVerdictRow({ v }: { v: ClaimCitationVerdict }) {
  const bad = v.problems.length > 0 || v.entailment === "NOT_SUPPORTED";
  const stateTone =
    v.evaluated_state === "SUPPORTED"
      ? "ok"
      : v.evaluated_state === "CONTRADICTED" || v.evaluated_state === "UNSUPPORTED"
        ? "bad"
        : v.evaluated_state === "PARTIALLY_SUPPORTED"
          ? "warn"
          : "neutral";
  return (
    <div className={cn("rounded-md border px-3 py-2", bad ? "border-warn/40 bg-warn/5" : "border-line")}>
      <div className="mb-1 flex items-center gap-2">
        <span className="font-mono text-2xs text-ink-faint">{v.claim_id}</span>
        <Badge tone={stateTone}>{v.evaluated_state}</Badge>
        <Badge tone={v.entailment === "SUPPORTED" ? "ok" : v.entailment === "NOT_SUPPORTED" ? "bad" : "neutral"}>
          entailment: {v.entailment}
        </Badge>
        {v.uncited && <Badge tone="bad">uncited</Badge>}
        {v.has_invalid_citation && <Badge tone="bad">invalid citation</Badge>}
      </div>
      <p className="text-sm text-ink-muted">{v.claim_text}</p>
      <div className="mt-1.5 flex flex-wrap gap-1.5 text-2xs text-ink-faint">
        <span>chunks: {v.resolved_chunk_ids.join(", ") || "none"}</span>
        {v.missing_required_chunk_ids.length > 0 && (
          <span className="text-warn">
            missing required: {v.missing_required_chunk_ids.join(", ")}
          </span>
        )}
        {v.irrelevant_chunk_ids.length > 0 && (
          <span>not labelled required: {v.irrelevant_chunk_ids.join(", ")}</span>
        )}
      </div>
      {v.problems.map((p, i) => (
        <p key={i} className="mt-1 text-2xs text-warn">
          {p}
        </p>
      ))}
    </div>
  );
}

/**
 * Human review controls for one question (V8 STEP 15).
 *
 * Append-only: every submit stores a NEW attributed review; history renders
 * oldest-first and nothing can be edited or deleted from here. After at least
 * one review exists, the derived human-evaluation run can be produced — that
 * pass reads reviews and writes a NEW run; source run and reviews stay intact.
 */
function ReviewSection({
  kbId,
  runId,
  questionId,
  onDerived,
}: {
  kbId: string;
  runId: string;
  questionId: string;
  onDerived: () => void;
}) {
  const [history, setHistory] = useState<AnswerReview[] | null>(null);
  const [open, setOpen] = useState(false);
  const [reviewer, setReviewer] = useState("");
  const [verdict, setVerdict] = useState<string>("correct");
  const [labels, setLabels] = useState<string[]>([]);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [deriving, setDeriving] = useState(false);

  const load = useCallback(async () => {
    try {
      const list = await answerEvaluationApi.reviews(kbId, runId, questionId);
      setHistory(list);
    } catch {
      setHistory([]);
    }
  }, [kbId, runId, questionId]);

  useEffect(() => {
    void load();
  }, [load]);

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await answerEvaluationApi.createReview(kbId, runId, questionId, {
        reviewer: reviewer.trim(),
        verdict,
        labels,
        notes: notes.trim(),
      });
      setNotes("");
      setLabels([]);
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const derive = async () => {
    setDeriving(true);
    setErr(null);
    try {
      await answerEvaluationApi.humanEvaluation(kbId, runId);
      onDerived();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDeriving(false);
    }
  };

  const count = history?.length ?? 0;
  return (
    <div className="border-t border-line pt-2.5">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-2xs text-ink-faint hover:text-ink-muted"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Human review ({count} stored{count > 0 ? ", append-only" : ""})
      </button>
      {open && (
        <div className="mt-2 space-y-2.5">
          {count > 0 && (
            <div className="space-y-1.5">
              {history!.map((rv) => (
                <div key={rv.id} className="rounded border border-line bg-surface-3 px-2.5 py-1.5">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge tone="accent">{rv.verdict}</Badge>
                    <span className="font-mono text-2xs text-ink-faint">{rv.reviewer}</span>
                    <span className="text-2xs text-ink-faint">{rv.created_at}</span>
                    {rv.labels.map((l) => (
                      <Badge key={l} tone="neutral">{l}</Badge>
                    ))}
                  </div>
                  {rv.notes && <p className="mt-1 text-2xs text-ink-muted">{rv.notes}</p>}
                </div>
              ))}
            </div>
          )}
          <div className="grid gap-2 sm:grid-cols-2">
            <label className="block">
              <span className="section-label">Reviewer identity (required)</span>
              <input
                value={reviewer}
                onChange={(e) => setReviewer(e.target.value)}
                placeholder="e.g. ada@example.edu"
                className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
              />
            </label>
            <label className="block">
              <span className="section-label">Verdict</span>
              <select
                value={verdict}
                onChange={(e) => setVerdict(e.target.value)}
                className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
              >
                {REVIEW_VERDICTS.map((v) => (
                  <option key={v} value={v}>{v.replace(/_/g, " ")}</option>
                ))}
              </select>
            </label>
          </div>
          <div>
            <span className="section-label">Structured labels</span>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {REVIEW_LABELS.map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() =>
                    setLabels((cur) => (cur.includes(l) ? cur.filter((x) => x !== l) : [...cur, l]))
                  }
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-2xs transition-colors",
                    labels.includes(l)
                      ? "border-accent/60 bg-accent/10 text-ink"
                      : "border-line text-ink-faint hover:text-ink-muted"
                  )}
                >
                  {l.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>
          <label className="block">
            <span className="section-label">Notes</span>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              maxLength={4000}
              placeholder="what did you see?"
              className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
            />
          </label>
          {err && <p className="text-2xs text-bad">{err}</p>}
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={submit} loading={busy} disabled={!reviewer.trim()}>
              Store review
            </Button>
            {count > 0 && (
              <Button onClick={derive} loading={deriving} variant="ghost">
                Derive human-evaluation run
              </Button>
            )}
            <span className="text-2xs text-ink-faint">
              Reviews are append-only: storing another never overwrites a previous one.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function FailureCard({
  r,
  kbId,
  runId,
  onDerived,
}: {
  r: AnswerQualityResult;
  kbId: string;
  runId: string;
  onDerived: () => void;
}) {
  const [open, setOpen] = useState(false);
  // A retrieval miss is a different defect from a citation defect. Keeping them
  // distinct stops "the answerer is bad" from being concluded when the evidence
  // was never in the context window.
  const retrievalMiss = r.retrieval_hit_rate.measured && r.retrieval_hit_rate.value === 0;
  return (
    <div className="rounded-md border border-warn/30 bg-surface-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5 text-ink-faint" /> : <ChevronRight className="h-3.5 w-3.5 text-ink-faint" />}
        <CircleAlert className="h-3.5 w-3.5 text-warn" />
        <span className="font-mono text-2xs text-ink-faint">{r.question_id}</span>
        <span className="min-w-0 flex-1 truncate text-sm text-ink">{r.question}</span>
        {retrievalMiss ? (
          <Badge tone="warn">retrieval miss</Badge>
        ) : (
          <Badge tone="bad">citation failure</Badge>
        )}
      </button>
      {open && (
        <div className="space-y-2.5 border-t border-line px-3 py-3">
          <div>
            <div className="section-label mb-1">Answer given</div>
            {/* Extractive/mock generators can emit very long boilerplate
                dumps. Cap the default height so one bad answer does not bury
                the claim audit, which is the actionable part. */}
            <p className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded border border-line bg-surface-3 p-2 text-sm text-ink-muted">
              {r.answer_text}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 text-2xs text-ink-faint sm:grid-cols-4">
            <span>state: {r.actual_grounding_state}</span>
            <span>expected: {r.expected_grounding_state}</span>
            <span>status: {r.answer_status}</span>
            <span>required rank: {r.retrieved_rank_of_required ?? "not retrieved"}</span>
            <span title={r.question_answer_relevance.reason}>
              relevance:{" "}
              {r.question_answer_relevance.measured &&
              r.question_answer_relevance.value !== null
                ? `${r.question_answer_relevance.value.toFixed(3)} (${r.relevance_method || "?"}${r.relevance_close_call ? ", close call" : ""})`
                : "UNKNOWN (abstention or no text)"}
            </span>
            <span title={r.correctness.reason}>
              correctness:{" "}
              {r.correctness.measured && r.correctness.value !== null
                ? `${r.correctness.value.toFixed(2)} (from ${r.correctness.sample_size ?? "?"} human review${r.correctness.sample_size === 1 ? "" : "s"})`
                : "UNKNOWN (needs human labels/review)"}
            </span>
          </div>
          <div>
            <div className="section-label mb-1">Evidence</div>
            <div className="flex flex-wrap gap-1.5">
              <span className="font-mono text-2xs text-ink-faint">required:</span>
              {r.required_chunk_ids.map((c) => <Badge key={c} tone="accent">{c}</Badge>)}
            </div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              <span className="font-mono text-2xs text-ink-faint">cited:</span>
              {r.cited_chunk_ids.length === 0
                ? <span className="text-2xs text-ink-faint">none</span>
                : r.cited_chunk_ids.map((c) => (
                    <Badge key={c} tone={r.required_chunk_ids.includes(c) ? "ok" : "neutral"}>{c}</Badge>
                  ))}
            </div>
          </div>
          <div>
            <div className="section-label mb-1">
              Claim audit ({r.claim_verdicts.length} claim{r.claim_verdicts.length === 1 ? "" : "s"})
            </div>
            <div className="space-y-1.5">
              {r.claim_verdicts.length === 0
                ? <p className="text-2xs text-ink-faint">the answer produced no auditable claims</p>
                : r.claim_verdicts.map((v) => <ClaimVerdictRow key={v.claim_id} v={v} />)}
            </div>
          </div>
          {r.warnings.length > 0 && (
            <div>
              <div className="section-label mb-1">Observations</div>
              {r.warnings.map((w, i) => (
                <p key={i} className="text-2xs text-ink-faint">{w}</p>
              ))}
            </div>
          )}
          <ReviewSection kbId={kbId} runId={runId} questionId={r.question_id} onDerived={onDerived} />
        </div>
      )}
    </div>
  );
}

function AggregatePanel({ agg }: { agg: AnswerQualityAggregate }) {
  const unknown = agg.unknown_metrics.length > 0;
  return (
    <Panel>
      <PanelHeader
        title="Measured answer-quality metrics"
        right={
          agg.unknown_metrics.length > 0 ? (
            <Badge tone="warn">{agg.unknown_metrics.length} unknown</Badge>
          ) : (
            <Badge tone="ok">all measured</Badge>
          )
        }
      />
      <div className="mb-3 flex flex-wrap gap-3 rounded-md border border-warn/30 bg-warn/5 px-3 py-2">
        <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
        <div className="text-2xs text-ink-muted">
          {unknown ? (
            <>
              <strong className="text-warn">Not measured:</strong>{" "}
              {[...new Set(agg.unknown_metrics)].join(", ")}.{" "}
              These are reported as UNKNOWN, not as zero. Correctness and
              completeness need human-authored labels or human reviews; when
              they exist they are measured and these lists shrink. Citation,
              grounding, relevance and abstention behaviour are measured now.
            </>
          ) : (
            <>Every metric in this view was computed from a real executed step.</>
          )}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <MetricCard name="pass_rate" m={agg.pass_rate} hint="citation + abstention checks" />
        <MetricCard name="citation_recall" m={agg.citation_recall} hint="required evidence cited" />
        <MetricCard
          name="citation_precision"
          m={agg.citation_precision}
          hint="lower bound: extra on-topic context counts against it"
        />
        <MetricCard name="evidence_support_rate" m={agg.evidence_support_rate} hint="entailment verdicts" />
        <MetricCard name="retrieval_hit_rate" m={agg.retrieval_hit_rate} hint="was the evidence retrieved" />
        <MetricCard
          name="question_answer_relevance"
          m={agg.question_answer_relevance}
          hint="does the answer text contain the question’s subject (IDF-weighted)"
        />
        <MetricCard
          name="relevance_failure_rate"
          m={agg.relevance_failure_rate}
          hint="grounded but not about the question"
        />
        <MetricCard name="supported_claim_ratio" m={agg.supported_claim_ratio} hint="claims judged SUPPORTED / all claims" />
        <MetricCard name="partial_claim_ratio" m={agg.partial_claim_ratio} hint="claims only partially backed" />
        <MetricCard name="unsupported_claim_ratio" m={agg.unsupported_claim_ratio} hint="uncited or contradicted / all claims" />
        <MetricCard
          name="fabricated_citation_rate"
          m={agg.fabricated_citation_rate}
          hint="references unresolvable to retrieved evidence"
        />
        <MetricCard name="unsupported_citation_rate" m={agg.unsupported_citation_rate} hint="citations on NOT_SUPPORTED claims (judged only)" />
        <MetricCard name="abstention_accuracy" m={agg.abstention_accuracy} hint="expected abstention never penalised" />
        <MetricCard name="grounding_state_accuracy" m={agg.grounding_state_accuracy} />
        <MetricCard name="expected_information_coverage" m={agg.expected_information_coverage} hint="human key points covered (needs human labels)" />
        <MetricCard name="reference_answer_similarity" m={agg.reference_answer_similarity} hint="lexical similarity to a human reference — NOT correctness" />
        <MetricCard name="unsupported_claim_rate" m={agg.unsupported_claim_rate} hint="uncited claims only (legacy measure)" />
        <MetricCard name="contradiction_rate" m={agg.contradiction_rate} />
        <MetricCard name="hallucination_rate" m={agg.hallucination_rate} hint="false supported or false unsupported" />
        <MetricCard name="false_supported_rate" m={agg.false_supported_rate} />
        <MetricCard name="false_unsupported_rate" m={agg.false_unsupported_rate} />
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <div>
          <div className="section-label mb-1">Grounding states produced</div>
          {Object.keys(agg.grounding_state_distribution).length === 0 ? (
            <p className="text-2xs text-ink-faint">no answers</p>
          ) : (
            Object.entries(agg.grounding_state_distribution).map(([k, v]) => (
              <div key={k} className="mb-1 flex items-center gap-2">
                <span className="w-44 shrink-0 font-mono text-2xs text-ink-faint">{k}</span>
                <ScoreBar value={v / Math.max(1, agg.question_count)} className="flex-1" />
                <span className="w-8 text-right text-2xs text-ink-muted">{v}</span>
              </div>
            ))
          )}
        </div>
        <div>
          <div className="section-label mb-1">Expected vs actual</div>
          {Object.keys(agg.confusion_matrix).length === 0 ? (
            <p className="text-2xs text-ink-faint">no expectations recorded</p>
          ) : (
            Object.entries(agg.confusion_matrix).map(([exp, row]) => (
              <div key={exp} className="mb-1.5">
                <span className="font-mono text-2xs text-ink-faint">{exp}</span>
                <div className="mt-0.5 flex flex-wrap gap-1.5">
                  {Object.entries(row).map(([k, v]) => (
                    <Badge key={k} tone={k.includes("correct") ? "ok" : "bad"}>
                      {k} &times;{v}
                    </Badge>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
      {agg.warnings.length > 0 && (
        <div className="mt-3 space-y-1 border-t border-line pt-2">
          {agg.warnings.map((w, i) => (
            <p key={i} className="text-2xs text-ink-faint">{w}</p>
          ))}
        </div>
      )}
    </Panel>
  );
}

export default function AnswerQualityPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [run, setRun] = useState<AnswerEvaluationRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassing, setShowPassing] = useState(false);
  const [strategy, setStrategy] = useState("dense");

  // Runs are persisted server-side, so a reload must not throw the view away:
  // load the most recent stored run on mount. Without this, `persist: true`
  // would be unobservable and every reload would look like a fresh start.
  useEffect(() => {
    if (!kbId) return;
    let cancelled = false;
    (async () => {
      try {
        const runs = await answerEvaluationApi.list(kbId, 1);
        if (cancelled || !runs.length) return;
        const full = await answerEvaluationApi.get(kbId, runs[0].id);
        if (!cancelled) setRun(full);
      } catch {
        // A missing run is not an error worth blocking the page for.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [kbId]);

  const runEval = useCallback(async () => {
    if (!kbId) return;
    setBusy(true);
    setError(null);
    try {
      const r = await answerEvaluationApi.run(kbId, {
        benchmark_path: BENCHMARK_PATH,
        strategy,
        persist: true,
      });
      setRun(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [kbId, strategy]);

  // After a human-evaluation derivation the newest stored run is the derived
  // one (runs are immutable; deriving creates a new row). Reload it so the
  // page reflects the human-scored correctness without touching the source.
  const reloadLatest = useCallback(async () => {
    if (!kbId) return;
    try {
      const runs = await answerEvaluationApi.list(kbId, 1);
      if (runs.length) {
        const full = await answerEvaluationApi.get(kbId, runs[0].id);
        setRun(full);
      }
    } catch {
      // keep the current view rather than blanking it
    }
  }, [kbId]);

  const failures = useMemo(
    () => (run?.per_question ?? []).filter((r) => !r.passed),
    [run]
  );
  const passing = useMemo(
    () => (run?.per_question ?? []).filter((r) => r.passed),
    [run]
  );
  const retrievalMisses = useMemo(
    () => failures.filter((r) => r.retrieval_hit_rate.measured && r.retrieval_hit_rate.value === 0),
    [failures]
  );
  const citationFailures = failures.length - retrievalMisses.length;

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-medium text-ink">Answer Quality</h1>
          <p className="mt-0.5 max-w-3xl text-2xs text-ink-faint">
            Scores grounded answers against evidence-based ground truth: were
            they correctly cited, supported, complete, and appropriately
            abstaining? Correctness against a human-written reference answer is
            UNKNOWN until one exists.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink"
          >
            <option value="dense">dense</option>
            <option value="hybrid">hybrid</option>
            <option value="bm25">bm25</option>
          </select>
          <Button onClick={runEval} loading={busy}>
            <FlaskConical className="mr-1.5 h-3.5 w-3.5" />
            Run evaluation
          </Button>
          <Button onClick={runEval} variant="ghost" disabled={busy} aria-label="Refresh">
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-bad/40 bg-bad/5 px-3 py-2 text-2xs text-bad">
          {error}
        </div>
      )}

      {!run && !error && (
        <Panel>
          <EmptyState
            icon={<Sigma className="h-6 w-6" />}
            title="No answer-quality run yet"
            hint={`Run the answer benchmark (${BENCHMARK_PATH}) against this knowledge base to produce measured citation, grounding and abstention metrics.`}
            action={
              <Button onClick={runEval} loading={busy}>
                Run evaluation
              </Button>
            }
          />
        </Panel>
      )}

      {run && (
        <>
          <Panel>
            <PanelHeader
              title="Provenance"
              right={
                run.is_mock ? (
                  <Badge tone="warn">mock generator</Badge>
                ) : (
                  <Badge tone="ok">live generator</Badge>
                )
              }
            />
            <div className="grid grid-cols-2 gap-3 text-2xs md:grid-cols-4">
              <div>
                <div className="section-label">Benchmark</div>
                <div className="text-ink-muted">
                  {run.benchmark_name} v{run.benchmark_version}
                </div>
                <div className="font-mono text-ink-faint">{run.benchmark_fingerprint}</div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  <Badge tone={run.benchmark_lifecycle === "frozen" ? "ok" : "warn"}>
                    {run.benchmark_lifecycle || "unknown lifecycle"}
                  </Badge>
                  {run.official ? (
                    <Badge tone="ok">official</Badge>
                  ) : (
                    <Badge tone="neutral">experiment (non-official)</Badge>
                  )}
                </div>
              </div>
              <div>
                <div className="section-label">Generator</div>
                <div className="text-ink-muted">{run.generator || "not recorded"}</div>
                <div className="font-mono text-ink-faint">{run.model || "not recorded"}</div>
              </div>
              <div>
                <div className="section-label">Evaluator</div>
                <div className="text-ink-muted">
                  {run.evaluator_name || "not recorded"}{" "}
                  <span className="font-mono text-ink-faint">{run.evaluator_version}</span>
                </div>
                <div className="font-mono text-ink-faint">
                  {run.evaluator_is_model_based ? "MODEL-BASED judge" : "deterministic"}
                </div>
                {run.evaluator_detail && (
                  <div className="text-ink-faint">{run.evaluator_detail}</div>
                )}
              </div>
              <div>
                <div className="section-label">Entailment</div>
                <div className="text-ink-muted">
                  {run.entailment_provider || "not recorded"}
                </div>
                <div className="font-mono text-ink-faint">
                  {run.entailment_is_model_based ? "model-based" : "heuristic"}
                </div>
              </div>
              <div>
                <div className="section-label">Relevance method</div>
                <div className="font-mono text-ink-faint">
                  {run.relevance_method || "not recorded"}
                </div>
                <div className="text-ink-faint">
                  weights: {run.relevance_weight_source || "unknown"}
                </div>
              </div>
              <div>
                <div className="section-label">Strategy</div>
                <div className="text-ink-muted">{run.strategy || "default"}</div>
              </div>
              <div>
                <div className="section-label">Coverage</div>
                <div className="text-ink-muted">
                  {run.question_count} questions ({run.answerable_count} answerable,{" "}
                  {run.unanswerable_count} unanswerable)
                </div>
                <div className="font-mono text-ink-faint">{run.subset_note}</div>
              </div>
              <div>
                <div className="section-label">Corpus</div>
                <div className="font-mono text-ink-faint">
                  {run.corpus_fingerprint ?? "not recorded"}
                </div>
              </div>
              <div>
                <div className="section-label">Run</div>
                <div className="font-mono text-ink-faint">{run.id}</div>
                <div className="text-ink-faint">{run.created_at}</div>
              </div>
            </div>
            {run.warnings.length > 0 && (
              <div className="mt-3 space-y-1 border-t border-line pt-2">
                {run.warnings.map((w, i) => (
                  <p key={i} className="text-2xs text-warn">{w}</p>
                ))}
              </div>
            )}
          </Panel>

          <AggregatePanel agg={run.aggregate} />

          <Panel>
            <PanelHeader
              title={`Failure analysis — ${failures.length} of ${run.question_count} questions`}
              right={
                <div className="flex items-center gap-2">
                  <Badge tone="bad">{citationFailures} citation</Badge>
                  <Badge tone="warn">{retrievalMisses.length} retrieval miss</Badge>
                </div>
              }
            />
            <p className="mb-3 text-2xs text-ink-faint">
              Citation failures mean the evidence was available but not cited.
              Retrieval misses mean the evidence never entered the context
              window — a different defect, and not the answerer&apos;s to fix.
            </p>
            {failures.length === 0 ? (
              <EmptyState
                icon={<CircleCheck className="h-6 w-6 text-ok" />}
                title="Every question passed its citation and abstention checks"
                hint="This covers citation completeness, unsupported claims, contradictions, abstention behaviour and grounding state. It does NOT mean factual correctness was verified — that is UNKNOWN."
              />
            ) : (
              <div className="space-y-2">
                {failures.map((r) => (
                  <FailureCard
                    key={r.question_id}
                    r={r}
                    kbId={kbId}
                    runId={run.id}
                    onDerived={reloadLatest}
                  />
                ))}
              </div>
            )}
            {passing.length > 0 && (
              <div className="mt-3 border-t border-line pt-2">
                <button
                  type="button"
                  onClick={() => setShowPassing((s) => !s)}
                  className="flex items-center gap-1.5 text-2xs text-ink-faint hover:text-ink-muted"
                >
                  {showPassing ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  {passing.length} passing question{passing.length === 1 ? "" : "s"}
                </button>
                {showPassing && (
                <div className="mt-2 space-y-2">
                  {passing.map((r) => (
                    <FailureCard
                      key={r.question_id}
                      r={r}
                      kbId={kbId}
                      runId={run.id}
                      onDerived={reloadLatest}
                    />
                  ))}
                </div>
                )}
              </div>
            )}
          </Panel>

          <Panel>
            <PanelHeader title="Run notes" />
            {run.notes.length === 0 ? (
              <p className="text-2xs text-ink-faint">
                No notes. Every question produced an answer; none were excluded
                from the metrics.
              </p>
            ) : (
              <ul className="space-y-1">
                {run.notes.map((n, i) => (
                  <li key={i} className="text-2xs text-warn">{n}</li>
                ))}
              </ul>
            )}
          </Panel>
        </>
      )}

      {!run && (
        <Panel>
          <div className="flex items-start gap-2 text-2xs text-ink-faint">
            <StatusDot tone="accent" />
            <div>
              What this page can and cannot tell you. It measures whether an
              answer cited the evidence a human designated as necessary, whether
              each claim is supported by the evidence it cites, and whether the
              system abstained exactly when it should. It does{" "}
              <strong className="text-ink-muted">not</strong> verify that the
              answer is factually correct, because no human-authored reference
              answer exists for any question in the benchmark.
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}