"use client";

/**
 * Retrieval Diagnostics — V12.
 *
 * Answers ONE question with evidence: "for each benchmark question, did the
 * retrieval pipeline surface the evidence required to answer it, and if not,
 * where did retrieval fail?"
 *
 * Three kinds of statement are kept visually and structurally apart, because
 * conflating them is how a dashboard turns a hypothesis into a fake fact:
 *
 *   1. MEASURED — retrieval metrics, ranks, latency, availability, integrity
 *      counts. A metric with `measured: false` renders as UNKNOWN with its
 *      reason; it is NEVER shown as 0.0 and never as a pass.
 *   2. INFERRED — the `classification` on each question. It names the RULE that
 *      fired and lists the exact observations, and it carries its limitations
 *      (for example: the vector index could not be probed).
 *   3. PROPOSED — every recommendation. It cites the measurements that support
 *      it, states the validation experiment, and is never applied automatically.
 *
 * Also deliberately visible:
 *   * the benchmark's own `authorship.human_review` (currently PENDING) — a
 *     result computed from a benchmark that has not been human-reviewed is not
 *     publication-grade, and the page says so;
 *   * the integrity snapshot proving the protected KB and the benchmark file
 *     were not modified by the run.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  Activity,
  CircleAlert,
  CircleCheck,
  Gauge,
  Play,
  RefreshCw,
  ShieldQuestion,
  Target,
} from "lucide-react";
import {
  DEFAULT_RETRIEVAL_BENCHMARK_PATH,
  EvidenceAudit,
  EvidenceAvailability,
  Measured,
  QuestionRetrievalDiagnostic,
  ExperimentMetricStats,
  RetrievalBenchmarkIdentity,
  RetrievalDiagnosticReport,
  RetrievalDiagnosticsRunSummary,
  RetrievalExperiment,
  RetrievalExperimentMeta,
  RetrievalExperimentSummary,
  RetrievalFailureCode,
  RetrievalRecommendation,
  StrategyComparison,
  StrategyDiagnostics,
  retrievalDiagnosticsApi,
  retrievalExperimentsApi,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, Button, EmptyState, Metric, Panel, PanelHeader } from "@/components/ui";

const ALL_CODES: RetrievalFailureCode[] = [
  "CORPUS_OR_SOURCE_DEFICIENCY",
  "EXPECTED_EVIDENCE_UNAVAILABLE",
  "RETRIEVAL_FAILURE",
  "NO_RELEVANT_EVIDENCE",
  "LOW_RECALL",
  "RANKING_FAILURE",
  "RETRIEVAL_SUCCESS",
  "UNKNOWN",
];

const CODE_TONE: Record<string, "ok" | "warn" | "bad" | "neutral" | "accent"> = {
  RETRIEVAL_SUCCESS: "ok",
  RANKING_FAILURE: "warn",
  LOW_RECALL: "warn",
  NO_RELEVANT_EVIDENCE: "bad",
  RETRIEVAL_FAILURE: "bad",
  EXPECTED_EVIDENCE_UNAVAILABLE: "bad",
  CORPUS_OR_SOURCE_DEFICIENCY: "bad",
  UNKNOWN: "neutral",
};

/** A measured value, rendered as UNKNOWN with its reason when it was not measured. */
function MeasuredValue({
  metric,
  digits = 4,
}: {
  metric: Measured | undefined;
  digits?: number;
}) {
  if (!metric) return <span className="text-ink-faint">not reported</span>;
  if (!metric.measured || metric.value === null || metric.value === undefined) {
    return (
      <span className="text-ink-muted" title={metric.reason || "not measured"}>
        UNKNOWN
      </span>
    );
  }
  return <span className="font-mono">{Number(metric.value).toFixed(digits)}</span>;
}

function CodeBadge({ code }: { code: string }) {
  return <Badge tone={CODE_TONE[code] ?? "neutral"}>{code}</Badge>;
}

export default function RetrievalDiagnosticsPage() {
  const params = useParams<{ id: string }>();
  const kbId = params?.id ?? "";

  const [benchmark, setBenchmark] = useState<RetrievalBenchmarkIdentity | null>(null);
  const [audit, setAudit] = useState<EvidenceAudit | null>(null);
  const [report, setReport] = useState<RetrievalDiagnosticReport | null>(null);
  const [runs, setRuns] = useState<RetrievalDiagnosticsRunSummary[]>([]);
  const [strategies, setStrategies] = useState<string[]>(["dense"]);
  const [topK, setTopK] = useState(10);
  const [busy, setBusy] = useState<"idle" | "audit" | "run" | "load">("idle");
  const [err, setErr] = useState<string | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [codeFilter, setCodeFilter] = useState<string>("");
  const [openQuestion, setOpenQuestion] = useState<string | null>(null);

  const refreshRuns = useCallback(async () => {
    if (!kbId) return;
    try {
      setRuns(await retrievalDiagnosticsApi.runs(kbId));
    } catch {
      /* the run list is informational; a failure here must not hide the page */
    }
  }, [kbId]);

  useEffect(() => {
    if (!kbId) return;
    (async () => {
      try {
        setBenchmark(await retrievalDiagnosticsApi.benchmark(kbId));
      } catch (e) {
        setErr(String((e as Error)?.message ?? e));
      }
      await refreshRuns();
    })();
  }, [kbId, refreshRuns]);

  const runAudit = async () => {
    setBusy("audit");
    setErr(null);
    try {
      setAudit(await retrievalDiagnosticsApi.evidenceAudit(kbId));
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally {
      setBusy("idle");
    }
  };

  const runDiagnostics = async () => {
    setBusy("run");
    setErr(null);
    try {
      const fresh = await retrievalDiagnosticsApi.runDiagnostics(kbId, {
        strategies,
        top_k: topK,
        persist: true,
      });
      setReport(fresh);
      setSelectedStrategy(fresh.strategies[0]?.strategy ?? "");
      setCodeFilter("");
      setOpenQuestion(null);
      await refreshRuns();
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally {
      setBusy("idle");
    }
  };

  const loadRun = async (runId: string) => {
    setBusy("load");
    setErr(null);
    try {
      const stored = await retrievalDiagnosticsApi.run(kbId, runId);
      setReport(stored);
      setSelectedStrategy(stored.strategies[0]?.strategy ?? "");
      setCodeFilter("");
      setOpenQuestion(null);
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally {
      setBusy("idle");
    }
  };

  const active: StrategyDiagnostics | null = useMemo(() => {
    if (!report) return null;
    return (
      report.strategies.find((s) => s.strategy === selectedStrategy) ??
      report.strategies[0] ??
      null
    );
  }, [report, selectedStrategy]);

  const visibleQuestions: QuestionRetrievalDiagnostic[] = useMemo(() => {
    if (!active) return [];
    return codeFilter
      ? active.questions.filter((q) => q.classification.code === codeFilter)
      : active.questions;
  }, [active, codeFilter]);

  const priority = useMemo(() => {
    if (!active) return [];
    const order: Record<string, number> = {
      CORPUS_OR_SOURCE_DEFICIENCY: 0,
      EXPECTED_EVIDENCE_UNAVAILABLE: 1,
      RETRIEVAL_FAILURE: 2,
      NO_RELEVANT_EVIDENCE: 3,
      LOW_RECALL: 4,
      RANKING_FAILURE: 5,
    };
    return [...active.questions]
      .filter((q) => q.classification.is_critical)
      .sort((a, b) => {
        const byCode = (order[a.classification.code] ?? 9) - (order[b.classification.code] ?? 9);
        if (byCode !== 0) return byCode;
        const ar = a.evidence.first_relevant_rank ?? 999;
        const br = b.evidence.first_relevant_rank ?? 999;
        return br - ar;
      })
      .slice(0, 10);
  }, [active]);

  const crossStrategy = (questionId: string) =>
    (report?.strategies ?? []).map((s) => ({
      strategy: s.strategy,
      status: s.status,
      question: s.questions.find((q) => q.question_id === questionId) ?? null,
    }));

  const authorshipReview = String(
    (benchmark?.authorship as Record<string, unknown>)?.human_review ?? "",
  );

  return (
    <div className="space-y-4 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-medium text-ink">Retrieval diagnostics</h1>
          <p className="mt-1 max-w-3xl text-xs text-ink-muted">
            Per-question, evidence-backed analysis of whether retrieval surfaced the evidence the
            benchmark requires. Retrieval only: no answer correctness is measured here, and no
            configuration is changed.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={runAudit} loading={busy === "audit"}>
            <ShieldQuestion className="mr-1 h-3.5 w-3.5" />
            Evidence audit
          </Button>
          <Button onClick={runDiagnostics} loading={busy === "run"}>
            <Play className="mr-1 h-3.5 w-3.5" />
            Run diagnostics
          </Button>
        </div>
      </div>

      {err && (
        <Panel className="border-bad/40">
          <p className="text-xs text-bad">{err}</p>
        </Panel>
      )}

      {/* ------------------------------------------------ benchmark identity */}
      {benchmark && (
        <Panel>
          <PanelHeader
            title="Benchmark under test"
            right={
              <Badge tone={benchmark.matches_this_kb ? "ok" : "bad"}>
                {benchmark.matches_this_kb ? "authored for this KB" : "authored for another KB"}
              </Badge>
            }
          />
          <div className="grid gap-2 text-2xs text-ink-muted md:grid-cols-2">
            <p>
              <span className="text-ink-faint">artifact</span>{" "}
              <span className="font-mono">{benchmark.benchmark}</span> v{benchmark.version} ·{" "}
              {benchmark.question_count} questions · relevance {benchmark.relevance_model}
            </p>
            <p>
              <span className="text-ink-faint">fingerprint</span>{" "}
              <span className="font-mono">{benchmark.fingerprint}</span>
            </p>
            <p className="md:col-span-2">
              <span className="text-ink-faint">authorship</span> {String(
                (benchmark.authorship as Record<string, unknown>)?.method ?? "not recorded",
              )}
            </p>
          </div>
          {authorshipReview && !/human_reviewed/i.test(authorshipReview) && (
            <p className="mt-2 flex items-start gap-1 text-2xs text-warn">
              <CircleAlert className="mt-0.5 h-3 w-3 shrink-0" />
              <span>
                <span className="font-mono">human_review: {authorshipReview}</span> — these labels
                have NOT been human-reviewed, so a result computed from them is a measured
                direction, not a publication-grade finding.
              </span>
            </p>
          )}
        </Panel>
      )}

      {/* --------------------------------------------------------- controls */}
      <Panel>
        <PanelHeader title="Run configuration" />
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <div className="section-label">Strategies</div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {["dense", "bm25", "hybrid", "hybrid_reranked"].map((s) => {
                const on = strategies.includes(s);
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() =>
                      setStrategies(
                        on ? strategies.filter((x) => x !== s) : [...strategies, s],
                      )
                    }
                    className={cn(
                      "rounded border px-2 py-1 text-2xs",
                      on
                        ? "border-accent/60 bg-accent/10 text-ink"
                        : "border-line text-ink-muted hover:text-ink",
                    )}
                  >
                    {s}
                  </button>
                );
              })}
            </div>
          </div>
          <div>
            <div className="section-label">Evaluated cut-off k</div>
            <input
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 1)}
              className="mt-1 w-20 rounded border border-line bg-surface-2 px-2 py-1 text-xs text-ink"
            />
            <p className="mt-1 text-2xs text-ink-faint">
              candidates are probed deeper (default 20) so &quot;retrieved below k&quot; stays
              distinguishable from &quot;not retrieved&quot;
            </p>
          </div>
          <div className="text-2xs text-ink-faint">
            benchmark path{" "}
            <span className="font-mono">{DEFAULT_RETRIEVAL_BENCHMARK_PATH}</span> (read-only)
          </div>
        </div>
      </Panel>

      {/* ----------------------------------------------------- evidence audit */}
      {audit && (
        <Panel>
          <PanelHeader
            title="Evidence audit (read-only)"
            right={
              <Badge tone={/fully resolvable/.test(audit.verdict) ? "ok" : "bad"}>
                {/fully resolvable/.test(audit.verdict) ? "evidence resolvable" : "evidence gaps"}
              </Badge>
            }
          />
          <div className="grid gap-2 text-2xs text-ink-muted md:grid-cols-3">
            <p>
              expected evidence: {audit.expected_chunk_count} chunks /{" "}
              {audit.expected_document_count} documents
            </p>
            <p>
              index probe: <span className="font-mono">{audit.index_probe}</span>
              {audit.index_points !== null ? ` · ${audit.index_points} points` : ""}
            </p>
            <p className="font-mono">
              sha256 {audit.benchmark_sha256.slice(0, 16)}…
            </p>
          </div>
          <p className="mt-2 text-2xs text-ink">{audit.verdict}</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {Object.entries(audit.availability_counts).map(([k, v]) => (
              <Badge
                key={k}
                tone={k === "all_indexed" ? "ok" : k === "unverified_index" ? "warn" : "bad"}
              >
                {k}: {v}
              </Badge>
            ))}
          </div>
          {audit.availability_counts.missing_from_corpus ||
          audit.availability_counts.missing_from_index ? (
            <div className="mt-2 space-y-1">
              <div className="section-label">Questions whose expected evidence is not retrievable</div>
              {audit.questions
                .filter(
                  (q) =>
                    q.availability_rollup === "missing_from_corpus" ||
                    q.availability_rollup === "missing_from_index",
                )
                .map((q) => (
                  <p key={q.question_id} className="text-2xs text-bad">
                    <span className="font-mono">{q.question_id}</span> {q.availability_rollup} —{" "}
                    {q.availability.map((a) => `${a.chunk_id}:${a.status}`).join(", ")}
                  </p>
                ))}
            </div>
          ) : null}
          {[...audit.index_probe_notes, ...audit.probe_notes].map((n, i) => (
            <p key={i} className="mt-1 text-2xs text-warn">
              {n}
            </p>
          ))}
          <p className="mt-2 text-2xs text-ink-faint">{audit.note}</p>
        </Panel>
      )}

      {/* -------------------------------------------------------- stored runs */}
      <Panel>
        <PanelHeader
          title="Stored diagnostic runs (append-only)"
          right={
            <Button variant="ghost" onClick={refreshRuns}>
              <RefreshCw className="mr-1 h-3.5 w-3.5" />
              Refresh
            </Button>
          }
        />
        {runs.length === 0 ? (
          <p className="text-2xs text-ink-muted">
            No diagnostic run has been stored for this knowledge base yet.
          </p>
        ) : (
          <div className="space-y-1">
            {runs.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => loadRun(r.id)}
                className={cn(
                  "flex w-full flex-wrap items-center gap-2 rounded border px-2 py-1 text-left text-2xs",
                  report?.id === r.id ? "border-accent/60 bg-accent/10" : "border-line hover:border-accent/40",
                )}
              >
                <span className="font-mono">{r.id}</span>
                <span className="text-ink-faint">k={r.top_k}</span>
                <span className="text-ink-faint">{r.created_at.slice(0, 19)}</span>
                {r.strategies.map((s) => (
                  <span key={s.strategy} className="flex items-center gap-1">
                    <span className="font-mono">{s.strategy}</span>
                    <Badge tone={s.status === "completed" ? "neutral" : "bad"}>{s.status}</Badge>
                    {Object.entries(s.code_counts ?? {}).map(([code, n]) => (
                      <span key={code} className="text-ink-faint">
                        {code}:{n}
                      </span>
                    ))}
                  </span>
                ))}
                {r.protected_kb_unchanged === true && r.benchmark_unchanged === true && (
                  <Badge tone="ok">KB + benchmark untouched</Badge>
                )}
                {r.protected_kb_unchanged === null && <Badge tone="warn">integrity UNVERIFIED</Badge>}
              </button>
            ))}
          </div>
        )}
      </Panel>

      <ExperimentsPanel kbId={kbId} />

      {/* ------------------------------------------------------------ report */}
      {report && (
        <>
          <Panel>
            <PanelHeader
              title={`Run ${report.id}`}
              right={
                <div className="flex items-center gap-1.5">
                  {report.strategies.map((s) => (
                    <button
                      key={s.strategy}
                      type="button"
                      onClick={() => {
                        setSelectedStrategy(s.strategy);
                        setCodeFilter("");
                      }}
                      className={cn(
                        "rounded border px-2 py-1 text-2xs",
                        selectedStrategy === s.strategy
                          ? "border-accent/60 bg-accent/10 text-ink"
                          : "border-line text-ink-muted",
                      )}
                    >
                      {s.strategy}
                      {s.status !== "completed" && " (failed)"}
                    </button>
                  ))}
                </div>
              }
            />
            <p className="text-2xs text-ink-muted">
              k={report.top_k} · {report.question_count} questions · benchmark{" "}
              <span className="font-mono">{report.benchmark_fingerprint}</span> ·{" "}
              {report.notes.join(" · ")}
            </p>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <Badge tone={report.integrity.protected_kb_unchanged ? "ok" : "bad"}>
                protected KB{" "}
                {report.integrity.protected_kb_unchanged === null
                  ? "UNVERIFIED"
                  : report.integrity.protected_kb_unchanged
                    ? "unchanged"
                    : "CHANGED"}
              </Badge>
              <Badge tone={report.integrity.benchmark_unchanged ? "ok" : "bad"}>
                benchmark file{" "}
                {report.integrity.benchmark_unchanged === null
                  ? "UNVERIFIED"
                  : report.integrity.benchmark_unchanged
                    ? "byte-identical"
                    : "CHANGED"}
              </Badge>
              <Badge tone="neutral">
                retrieval runs written: {report.integrity.retrieval_runs_persisted}
              </Badge>
              <span className="text-2xs text-ink-faint">
                index points {report.integrity.index_points_before} →{" "}
                {report.integrity.index_points_after} · chunk rows{" "}
                {report.integrity.chunk_rows_before} → {report.integrity.chunk_rows_after}
              </span>
            </div>
          </Panel>

          {/* per-strategy summary */}
          {active && (
            <Panel>
              <PanelHeader
                title={`${active.strategy} — measured summary`}
                right={
                  <Badge tone={active.status === "completed" ? "ok" : "bad"}>
                    {active.status}
                  </Badge>
                }
              />
              {active.status !== "completed" ? (
                <p className="text-2xs text-bad">
                  {active.failure_reason || "the strategy did not run"} — no other strategy&apos;s
                  numbers were substituted for it.
                </p>
              ) : (
                <>
                  <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-4">
                    <Metric
                      label={`Recall@${report.top_k}`}
                      value={<MeasuredValue metric={active.summary.recall_at_k} />}
                      hint={active.summary.recall_at_k.reason}
                    />
                    <Metric
                      label="MRR"
                      value={<MeasuredValue metric={active.summary.mrr} />}
                      hint={active.summary.mrr.reason}
                    />
                    <Metric
                      label={`NDCG@${report.top_k}`}
                      value={<MeasuredValue metric={active.summary.ndcg_at_k} />}
                      hint={active.summary.ndcg_at_k.reason}
                    />
                    <Metric
                      label={`Precision@${report.top_k}`}
                      value={<MeasuredValue metric={active.summary.precision_at_k} />}
                      hint="lower bound: the benchmark labels NECESSARY evidence only"
                    />
                    <Metric
                      label="required evidence found at any probed depth"
                      value={<MeasuredValue metric={active.summary.required_recall_in_probe} />}
                      hint={active.summary.required_recall_in_probe.reason}
                    />
                    <Metric
                      label="mean retrieval latency"
                      value={
                        active.summary.latency_ms.measured
                          ? `${Number(active.summary.latency_ms.value).toFixed(1)} ms`
                          : "UNKNOWN"
                      }
                      hint={active.summary.latency_ms.reason}
                    />
                    <Metric
                      label="probed candidate depth"
                      value={active.candidate_depth}
                      hint="deeper than k, so depth failures are distinguishable from absence"
                    />
                    <Metric
                      label="total run time"
                      value={active.duration_ms !== null ? `${(active.duration_ms / 1000).toFixed(1)} s` : "UNKNOWN"}
                    />
                  </div>

                  <div className="mt-3 flex flex-wrap items-center gap-1.5">
                    {ALL_CODES.filter((c) => active.summary.code_counts[c]).map((c) => (
                      <button
                        key={c}
                        type="button"
                        onClick={() => setCodeFilter(codeFilter === c ? "" : c)}
                        className={cn(
                          "rounded border px-2 py-1 text-2xs",
                          codeFilter === c
                            ? "border-accent/60 bg-accent/10 text-ink"
                            : "border-line text-ink-muted",
                        )}
                      >
                        {c}: {active.summary.code_counts[c]}
                      </button>
                    ))}
                    {codeFilter && (
                      <button
                        type="button"
                        className="text-2xs text-ink-faint underline"
                        onClick={() => setCodeFilter("")}
                      >
                        clear filter
                      </button>
                    )}
                  </div>

                  <p className="mt-2 text-2xs text-ink-faint">
                    availability:{" "}
                    {Object.entries(active.summary.availability_counts)
                      .map(([k, v]) => `${k} ${v}`)
                      .join(" · ")}{" "}
                    · reranker {active.reranker_status || "not requested"} · backend{" "}
                    <span className="font-mono">{active.retrieval_backend}</span> · config{" "}
                    <span className="font-mono">{active.config_fingerprint}</span>
                  </p>
                  {active.summary.warnings.map((w, i) => (
                    <p key={i} className="mt-1 text-2xs text-warn">
                      {w}
                    </p>
                  ))}
                  {active.notes.map((n, i) => (
                    <p key={i} className="mt-1 text-2xs text-ink-faint">
                      {n}
                    </p>
                  ))}
                </>
              )}
            </Panel>
          )}

          {/* priority list */}
          {active && priority.length > 0 && (
            <Panel>
              <PanelHeader
                title="Highest-priority retrieval problems"
                right={<Badge tone="bad">{priority.length}</Badge>}
              />
              <div className="space-y-1">
                {priority.map((q) => (
                  <button
                    key={q.question_id}
                    type="button"
                    onClick={() => setOpenQuestion(q.question_id)}
                    className="flex w-full flex-wrap items-center gap-2 rounded border border-line px-2 py-1 text-left text-2xs hover:border-accent/40"
                  >
                    <span className="font-mono">{q.question_id}</span>
                    <CodeBadge code={q.classification.code} />
                    <span className="text-ink-muted">
                      {q.classification.code === "LOW_RECALL"
                        ? `beyond k=${report.top_k} at rank ${
                            Object.values(q.probe.required_beyond_cutoff)[0] ?? "?"
                          }`
                        : q.classification.code === "NO_RELEVANT_EVIDENCE"
                          ? "absent from the whole probe"
                          : `first required chunk at rank ${q.evidence.first_relevant_rank ?? "?"}`}
                    </span>
                    <span className="truncate text-ink-faint">{q.question}</span>
                  </button>
                ))}
              </div>
            </Panel>
          )}

          {/* question table */}
          {active && (
            <Panel>
              <PanelHeader
                title={`Questions (${visibleQuestions.length}${
                  codeFilter ? ` filtered to ${codeFilter}` : ""
                })`}
              />
              <div className="overflow-x-auto">
                <table className="w-full text-2xs">
                  <thead className="text-ink-faint">
                    <tr>
                      <th className="text-left">question</th>
                      <th className="text-left">classification</th>
                      <th className="text-left">rule</th>
                      <th className="text-right">first required rank</th>
                      <th className="text-right">recall@k</th>
                      <th className="text-right">probed depth</th>
                      <th className="text-left">availability</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {visibleQuestions.map((q) => (
                      <tr key={q.question_id} className="border-t border-line/60">
                        <td className="py-1 pr-2">
                          <span className="font-mono">{q.question_id}</span>{" "}
                          <span className="text-ink-muted">{q.question}</span>
                        </td>
                        <td className="py-1 pr-2">
                          <CodeBadge code={q.classification.code} />
                        </td>
                        <td className="py-1 pr-2 font-mono text-ink-faint">
                          {q.classification.rule}
                        </td>
                        <td className="py-1 pr-2 text-right font-mono">
                          {q.evidence.first_relevant_rank ?? "—"}
                        </td>
                        <td className="py-1 pr-2 text-right">
                          <MeasuredValue metric={q.evidence.recall_at_k} digits={3} />
                        </td>
                        <td className="py-1 pr-2 text-right font-mono">
                          {q.probe.candidate_depth}
                        </td>
                        <td className="py-1 pr-2">
                          <span
                            className={cn(
                              q.availability_rollup === "all_indexed"
                                ? "text-ok"
                                : q.availability_rollup === "unverified_index"
                                  ? "text-warn"
                                  : "text-bad",
                            )}
                          >
                            {q.availability_rollup}
                          </span>
                        </td>
                        <td className="py-1 text-right">
                          <button
                            type="button"
                            className="text-ink-faint underline"
                            onClick={() =>
                              setOpenQuestion(openQuestion === q.question_id ? null : q.question_id)
                            }
                          >
                            {openQuestion === q.question_id ? "hide" : "inspect"}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}

          {/* per-question detail */}
          {active && openQuestion && (
            <QuestionDetail
              question={active.questions.find((q) => q.question_id === openQuestion) ?? null}
              topK={report.top_k}
              cross={crossStrategy(openQuestion)}
            />
          )}

          {/* comparisons */}
          {report.comparisons.length > 0 && (
            <Panel>
              <PanelHeader title="Strategy comparison (matched questions only)" />
              {report.comparisons.map((c, i) => (
                <ComparisonBlock key={i} comparison={c} />
              ))}
            </Panel>
          )}

          {/* recommendations */}
          <Panel>
            <PanelHeader
              title="Proposed experiments"
              right={<Badge tone="warn">not applied automatically</Badge>}
            />
            {report.recommendations.recommendations.length === 0 ? (
              <p className="text-2xs text-ink-muted">No recommendation was produced.</p>
            ) : (
              <div className="space-y-2">
                {report.recommendations.recommendations.map((r, i) => (
                  <RecommendationCard key={i} rec={r} />
                ))}
              </div>
            )}
            {report.recommendations.data_gaps.map((g, i) => (
              <p key={i} className="mt-1 text-2xs text-warn">
                data gap: {g}
              </p>
            ))}
            {report.recommendations.notes.map((n, i) => (
              <p key={i} className="mt-1 text-2xs text-ink-faint">
                {n}
              </p>
            ))}
          </Panel>

          {/* limitations */}
          <Panel>
            <PanelHeader title="Limitations of this run" />
            {report.limitations.map((l, i) => (
              <p key={i} className="text-2xs text-ink-muted">
                • {l}
              </p>
            ))}
          </Panel>
        </>
      )}

      {!report && !audit && runs.length === 0 && (
        <EmptyState
          icon={<Gauge className="h-5 w-5" />}
          title="No retrieval diagnostics yet"
          hint="Run the evidence audit to check that the benchmark's expected evidence is still in the corpus and the index, then run diagnostics to classify every question."
        />
      )}
    </div>
  );
}

/**
 * Controlled retrieval experiments (V13).
 *
 * READ-ONLY on purpose. The page shows an experiment that was DECLARED and RUN
 * through the tested API (see `POST .../retrieval-experiments` and
 * `backend/scripts/run_retrieval_experiment.py`); it deliberately offers no
 * "run" button, because an experiment is a hypothesis declared before
 * measurement, not a knob to twiddle until a score moves.
 *
 * The declaration is rendered ABOVE the result: the hypothesis, the primary
 * metric, the guardrails and the rejection conditions were all stored before any
 * arm ran, so a reader can tell whether the bar moved after the fact.
 */
function ExperimentsPanel({ kbId }: { kbId: string }) {
  const [rows, setRows] = useState<RetrievalExperimentSummary[]>([]);
  const [detail, setDetail] = useState<RetrievalExperiment | null>(null);
  const [policy, setPolicy] = useState<RetrievalExperimentMeta | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadAll = useCallback(async () => {
    if (!kbId) return;
    try {
      const [meta, list] = await Promise.all([
        retrievalExperimentsApi.meta(kbId),
        retrievalExperimentsApi.list(kbId),
      ]);
      setPolicy(meta);
      setRows(list);
      if (list.length > 0) {
        setDetail(await retrievalExperimentsApi.get(kbId, list[0].id));
      } else {
        setDetail(null);
      }
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    }
  }, [kbId]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  const select = async (id: string) => {
    setLoading(true);
    setErr(null);
    try {
      setDetail(await retrievalExperimentsApi.get(kbId, id));
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally {
      setLoading(false);
    }
  };

  const baselineLabel = detail?.baseline.label || detail?.baseline.strategy || "";

  return (
    <Panel>
      <PanelHeader
        title="Controlled retrieval experiments (V13)"
        right={
          <Button variant="ghost" onClick={loadAll} disabled={loading}>
            <RefreshCw className="mr-1 h-3.5 w-3.5" />
            Refresh
          </Button>
        }
      />

      {err && (
        <p className="mb-2 rounded border border-bad/40 bg-bad/10 px-2 py-1 text-2xs text-bad">
          {err}
        </p>
      )}

      {policy && (
        <p className="mb-2 text-2xs text-ink-muted">
          {policy.notice} {policy.promotion_policy}
        </p>
      )}

      {rows.length === 0 ? (
        <p className="text-2xs text-ink-muted">
          No controlled experiment has been recorded for this knowledge base yet. An
          experiment is declared (hypothesis, primary metric, guardrails, rejection
          conditions) and then run through the API; it is never inferred from a score.
        </p>
      ) : (
        <div className="space-y-2">
          <div className="space-y-1">
            {rows.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => select(r.id)}
                className={cn(
                  "flex w-full flex-wrap items-center gap-2 rounded border px-2 py-1 text-left text-2xs",
                  detail?.id === r.id
                    ? "border-accent/60 bg-accent/10"
                    : "border-line hover:border-accent/40",
                )}
              >
                <span className="font-mono">{r.id}</span>
                <span>{r.name}</span>
                <Badge tone={outcomeTone(r.outcome)}>{r.outcome}</Badge>
                <span className="text-ink-faint">primary={r.primary_metric}</span>
                <span className="text-ink-faint">q={r.matched_question_count}</span>
                <span className="text-ink-faint">{r.created_at.slice(0, 19)}</span>
                {r.candidate_outcomes.map((c) => (
                  <span key={c.label} className="text-ink-faint">
                    {c.label}:{c.outcome}
                  </span>
                ))}
              </button>
            ))}
          </div>

          {detail && (
            <div className="space-y-3 rounded border border-line p-2">
              <div className="space-y-1">
                <p className="text-2xs font-medium text-ink">Declaration (stored before measurement)</p>
                <p className="text-2xs text-ink-muted">
                  <span className="text-ink-faint">Hypothesis: </span>
                  {detail.hypothesis}
                </p>
                <div className="flex flex-wrap items-center gap-2 text-2xs">
                  <span className="text-ink-faint">Primary</span>
                  <Badge tone="accent">{detail.primary_metric}</Badge>
                  <span className="text-ink-faint">Guardrails</span>
                  {detail.guardrail_metrics.length === 0 ? (
                    <span className="text-ink-faint">none declared</span>
                  ) : (
                    detail.guardrail_metrics.map((m) => (
                      <Badge key={m} tone="neutral">
                        {m}
                        {policy?.meaningful_delta[m] !== undefined
                          ? ` (bar ${policy.meaningful_delta[m]})`
                          : ""}
                      </Badge>
                    ))
                  )}
                  {detail.targeted_failure_code && (
                    <>
                      <span className="text-ink-faint">Targets</span>
                      <CodeBadge code={detail.targeted_failure_code} />
                    </>
                  )}
                </div>
                <p className="text-2xs text-ink-faint">
                  benchmark {detail.benchmark_name} · fingerprint {detail.benchmark_fingerprint} ·
                  question-set {detail.question_set_fingerprint} · {detail.matched_question_ids.length}{" "}
                  matched question(s) · comparison: {detail.comparison_method}
                </p>
                {detail.rejection_conditions.length > 0 && (
                  <details className="text-2xs text-ink-muted">
                    <summary className="cursor-pointer text-ink-faint">
                      Rejection conditions ({detail.rejection_conditions.length})
                    </summary>
                    <ul className="mt-1 list-disc pl-4">
                      {detail.rejection_conditions.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>

              <div>
                <p className="mb-1 text-2xs font-medium text-ink">Measured arms</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-2xs">
                    <thead className="text-ink-faint">
                      <tr>
                        <th className="py-1 text-left">arm</th>
                        <th className="py-1 text-left">status</th>
                        <th className="py-1 text-right">k</th>
                        <th className="py-1 text-right">probe</th>
                        <th className="py-1 text-right">recall@k</th>
                        <th className="py-1 text-right">mrr</th>
                        <th className="py-1 text-right">ndcg</th>
                        <th className="py-1 text-right">prec@k</th>
                        <th className="py-1 text-right">latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.arms.map((a) => (
                        <tr key={a.label} className="border-t border-line/60">
                          <td className="py-1 font-mono">
                            {a.label}
                            {a.label === baselineLabel && (
                              <span className="ml-1 text-ink-faint">(baseline)</span>
                            )}
                          </td>
                          <td className="py-1">
                            <Badge tone={a.status === "completed" ? "ok" : "bad"}>{a.status}</Badge>
                            {a.failure_reason && (
                              <span className="ml-1 text-ink-faint">{a.failure_reason}</span>
                            )}
                          </td>
                          <td className="py-1 text-right text-ink-faint">{a.top_k}</td>
                          <td className="py-1 text-right text-ink-faint">{a.candidate_depth}</td>
                          <td className="py-1 text-right">
                            <MeasuredValue metric={toMeasured(a.metric_stats.recall_at_k)} digits={3} />
                          </td>
                          <td className="py-1 text-right">
                            <MeasuredValue metric={toMeasured(a.metric_stats.mrr)} digits={3} />
                          </td>
                          <td className="py-1 text-right">
                            <MeasuredValue metric={toMeasured(a.metric_stats.ndcg_at_k)} digits={3} />
                          </td>
                          <td className="py-1 text-right">
                            <MeasuredValue metric={toMeasured(a.metric_stats.precision_at_k)} digits={3} />
                          </td>
                          <td className="py-1 text-right">
                            {a.latency_ms === null ? (
                              <span className="text-ink-faint">UNKNOWN</span>
                            ) : (
                              `${Math.round(a.latency_ms)} ms`
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-1 text-2xs text-ink-faint">
                  A metric with no measured value renders UNKNOWN with its reason — never as 0.
                  precision@k is a LOWER BOUND under necessity-only binary labels: it falls
                  definitionally as k grows, so it is reported, not guardrailed.
                </p>
              </div>

              <div>
                <p className="mb-1 text-2xs font-medium text-ink">Outcome (derived by the declared rules)</p>
                {detail.candidate_outcomes.map((c) => (
                  <div key={c.label} className="mb-2 rounded border border-line/60 p-2 text-2xs">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono">{c.label}</span>
                      <Badge tone={outcomeTone(c.outcome)}>{c.outcome}</Badge>
                      <Badge tone={c.acceptance === "ACCEPT" ? "ok" : c.acceptance === "NOT_VALID" ? "warn" : "neutral"}>
                        {c.acceptance}
                      </Badge>
                      <span className="text-ink-faint">pairing {c.pairing}</span>
                      <span className="text-ink-faint">
                        {c.primary_metric} delta{" "}
                        {c.primary_mean_delta === null
                          ? "UNKNOWN"
                          : `${c.primary_mean_delta >= 0 ? "+" : ""}${c.primary_mean_delta.toFixed(4)}`}{" "}
                        (bar {c.primary_required_delta}, {c.paired_question_count} paired)
                      </span>
                    </div>
                    <p className="mt-1 text-ink-muted">{c.outcome_explanation}</p>
                    {c.guardrail_assessment.length > 0 && (
                      <p className="mt-1 text-ink-faint">
                        guardrails: {c.guardrail_assessment.join("; ")}
                      </p>
                    )}
                    <p className="mt-1 text-ink-faint">rules: {c.rules_applied.join("; ")}</p>
                  </div>
                ))}
              </div>

              {detail.arms
                .filter((a) => a.per_question.length > 0)
                .map((a) => {
                  const moved = a.per_question.filter(
                    (p) => p.outcome === "IMPROVED" || p.outcome === "REGRESSED",
                  );
                  return (
                    <details key={a.label} className="text-2xs text-ink-muted">
                      <summary className="cursor-pointer text-ink-faint">
                        Per-question {detail.primary_metric} for {a.label}: {moved.length} moved of{" "}
                        {a.per_question.length}
                      </summary>
                      <div className="mt-1 max-h-64 overflow-y-auto">
                        <table className="w-full text-2xs">
                          <thead className="text-ink-faint">
                            <tr>
                              <th className="py-1 text-left">question</th>
                              <th className="py-1 text-right">baseline</th>
                              <th className="py-1 text-right">candidate</th>
                              <th className="py-1 text-right">delta</th>
                              <th className="py-1 text-left">outcome</th>
                            </tr>
                          </thead>
                          <tbody>
                            {a.per_question.map((p) => (
                              <tr key={p.question_id} className="border-t border-line/60">
                                <td className="py-1 font-mono">{p.question_id}</td>
                                <td className="py-1 text-right">
                                  {p.baseline === null ? "UNKNOWN" : p.baseline.toFixed(3)}
                                </td>
                                <td className="py-1 text-right">
                                  {p.candidate === null ? "UNKNOWN" : p.candidate.toFixed(3)}
                                </td>
                                <td className="py-1 text-right">
                                  {p.delta === null
                                    ? "UNKNOWN"
                                    : `${p.delta >= 0 ? "+" : ""}${p.delta.toFixed(3)}`}
                                </td>
                                <td className="py-1">
                                  <Badge
                                    tone={
                                      p.outcome === "IMPROVED"
                                        ? "ok"
                                        : p.outcome === "REGRESSED"
                                          ? "bad"
                                          : "neutral"
                                    }
                                  >
                                    {p.outcome}
                                  </Badge>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  );
                })}

              <p className="text-2xs text-ink-faint">
                Integrity: index unchanged={" "}
                {String((detail.integrity as Record<string, unknown>).index_unchanged)} · corpus
                unchanged={" "}
                {String((detail.integrity as Record<string, unknown>).corpus_unchanged)} · benchmark
                unchanged={" "}
                {String((detail.integrity as Record<string, unknown>).benchmark_unchanged)} · scratch
                collections created={" "}
                {String((detail.integrity as Record<string, unknown>).scratch_collections_created)}
                {" · promoted to default: "}
                {String(detail.promoted_to_default)}
              </p>

              {detail.limitations.length > 0 && (
                <details className="text-2xs text-ink-muted">
                  <summary className="cursor-pointer text-ink-faint">
                    Limitations ({detail.limitations.length})
                  </summary>
                  <ul className="mt-1 list-disc pl-4">
                    {detail.limitations.map((l) => (
                      <li key={l}>{l}</li>
                    ))}
                  </ul>
                </details>
              )}

              {detail.errors.length > 0 && (
                <p className="rounded border border-bad/40 bg-bad/10 px-2 py-1 text-2xs text-bad">
                  {detail.errors.join("; ")}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

function outcomeTone(outcome: string): "ok" | "warn" | "bad" | "neutral" | "accent" {
  switch (outcome) {
    case "IMPROVED":
      return "ok";
    case "REGRESSION":
    case "BLOCKED":
      return "bad";
    case "INCONCLUSIVE":
      return "warn";
    case "NO_MEANINGFUL_IMPROVEMENT":
      return "neutral";
    default:
      return "neutral";
  }
}

/** Adapt the experiment's metric statistics to the shared Measured renderer. */
function toMeasured(s: ExperimentMetricStats | undefined): Measured | undefined {
  if (!s) return undefined;
  return {
    value: s.mean,
    measured: s.measured,
    reason: s.reason,
    sample_size: s.sample_size,
  } as Measured;
}

function QuestionDetail({
  question,
  topK,
  cross,
}: {
  question: QuestionRetrievalDiagnostic | null;
  topK: number;
  cross: Array<{
    strategy: string;
    status: string;
    question: QuestionRetrievalDiagnostic | null;
  }>;
}) {
  if (!question) {
    return (
      <Panel>
        <p className="text-2xs text-ink-muted">This question has no measurement in this strategy.</p>
      </Panel>
    );
  }
  const rankOf = new Map(question.evidence.retrieved.map((r) => [r.chunk_id, r.rank]));
  return (
    <Panel>
      <PanelHeader
        title={`${question.question_id} — ${question.question}`}
        right={<CodeBadge code={question.classification.code} />}
      />

      <div className="grid gap-3 lg:grid-cols-2">
        {/* MEASURED */}
        <div>
          <div className="section-label">Measured — expected vs retrieved</div>
          <p className="mt-1 text-2xs text-ink-muted">
            expected chunks:{" "}
            {question.expected_chunk_ids.map((c) => {
              const rank = rankOf.get(c);
              return (
                <span key={c} className="mr-2 font-mono">
                  {c}
                  {rank ? ` → rank ${rank}` : ` → not in top-${topK}`}
                </span>
              );
            })}
          </p>
          {Object.keys(question.probe.required_beyond_cutoff).length > 0 && (
            <p className="mt-1 text-2xs text-warn">
              found beyond k={topK} when probed to depth {question.probe.candidate_depth}:{" "}
              {Object.entries(question.probe.required_beyond_cutoff)
                .map(([c, r]) => `${c} at rank ${r}`)
                .join(", ")}
            </p>
          )}
          {question.probe.absent_from_probe.length > 0 && (
            <p className="mt-1 text-2xs text-bad">
              absent from the whole {question.probe.candidate_depth}-deep probe:{" "}
              {question.probe.absent_from_probe.join(", ")}
            </p>
          )}
          <table className="mt-1 w-full text-2xs">
            <thead className="text-ink-faint">
              <tr>
                <th className="text-left">rank</th>
                <th className="text-left">chunk</th>
                <th className="text-right">score</th>
                <th className="text-left">section</th>
                <th className="text-left">required</th>
              </tr>
            </thead>
            <tbody>
              {question.evidence.retrieved.map((r) => (
                <tr key={r.chunk_id} className="border-t border-line/60">
                  <td className="py-0.5 font-mono">{r.rank}</td>
                  <td className="py-0.5 font-mono">{r.chunk_id}</td>
                  <td className="py-0.5 text-right font-mono">
                    {r.score === null ? "—" : r.score.toFixed(4)}
                  </td>
                  <td className="py-0.5 text-ink-muted">{r.section ?? "not recorded"}</td>
                  <td className="py-0.5">
                    {r.is_required ? <span className="text-ok">yes</span> : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2 grid grid-cols-2 gap-1 text-2xs text-ink-muted">
            <span>
              recall@{topK}: <MeasuredValue metric={question.evidence.recall_at_k} digits={3} />
            </span>
            <span>
              mrr: <MeasuredValue metric={question.evidence.mrr} digits={3} />
            </span>
            <span>
              ndcg@{topK}: <MeasuredValue metric={question.evidence.ndcg_at_k} digits={3} />
            </span>
            <span>
              latency:{" "}
              {question.latency_ms === null ? "UNKNOWN" : `${question.latency_ms.toFixed(1)} ms`}
            </span>
          </div>

          <div className="section-label mt-2">Availability of the expected evidence</div>
          {question.availability.map((a: EvidenceAvailability) => (
            <p
              key={a.chunk_id}
              className={cn(
                "text-2xs",
                a.status === "indexed" ? "text-ink-muted" : "text-bad",
              )}
            >
              <span className="font-mono">{a.chunk_id}</span> {a.status}
              {a.in_index === null ? " (index not probed)" : ""}
              {a.document_matches_benchmark === false
                ? ` — benchmark expects document(s) ${a.expected_document_ids.join(", ")}, chunk lives in ${a.actual_document_id}`
                : ""}
            </p>
          ))}
        </div>

        {/* INFERRED */}
        <div>
          <div className="section-label">Inferred — why this classification</div>
          <p className="mt-1 text-2xs text-ink">{question.classification.reason}</p>
          <p className="mt-1 text-2xs text-ink-faint">
            rule <span className="font-mono">{question.classification.rule}</span> · confidence{" "}
            <span className="font-mono">{question.classification.confidence}</span>
          </p>
          <div className="mt-1">
            {question.classification.evidence.map((e, i) => (
              <p key={i} className="text-2xs text-ink-muted">
                • {e}
              </p>
            ))}
          </div>
          {question.classification.limitations.map((l, i) => (
            <p key={i} className="mt-1 text-2xs text-warn">
              limitation: {l}
            </p>
          ))}

          <div className="section-label mt-3">Same question under every strategy</div>
          {cross.map((c) => (
            <p key={c.strategy} className="text-2xs text-ink-muted">
              <span className="font-mono">{c.strategy}</span>{" "}
              {c.question ? (
                <>
                  <CodeBadge code={c.question.classification.code} /> first required rank{" "}
                  {c.question.evidence.first_relevant_rank ?? "—"}
                </>
              ) : (
                <span className="text-ink-faint">
                  no measurement ({c.status}
                  {c.status !== "completed" ? ": strategy failed" : ""})
                </span>
              )}
            </p>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function ComparisonBlock({ comparison }: { comparison: StrategyComparison }) {
  return (
    <div className="mt-2 rounded border border-line p-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-ink">
          {comparison.baseline_label} → {comparison.candidate_label}
        </span>
        <Badge
          tone={
            comparison.pairing === "COMPARABLE"
              ? "ok"
              : comparison.pairing === "INCONCLUSIVE"
                ? "warn"
                : "bad"
          }
        >
          {comparison.pairing}
        </Badge>
        <span className="text-2xs text-ink-faint">
          {comparison.comparable_question_ids.length} paired questions
          {comparison.exploratory ? " · exploratory" : ""}
        </span>
      </div>
      <p className="mt-1 text-2xs text-ink-muted">{comparison.pairing_reason}</p>
      <table className="mt-1 w-full text-2xs">
        <thead className="text-ink-faint">
          <tr>
            <th className="text-left">metric</th>
            <th className="text-right">baseline</th>
            <th className="text-right">candidate</th>
            <th className="text-right">mean delta</th>
            <th className="text-right">improved</th>
            <th className="text-right">regressed</th>
            <th className="text-left">outcome</th>
          </tr>
        </thead>
        <tbody>
          {comparison.metric_comparisons.map((m) => (
            <tr key={m.metric} className="border-t border-line/60">
              <td className="py-0.5 font-mono">{m.metric}</td>
              <td className="py-0.5 text-right font-mono">
                {m.baseline_mean === null ? "—" : m.baseline_mean.toFixed(4)}
              </td>
              <td className="py-0.5 text-right font-mono">
                {m.candidate_mean === null ? "—" : m.candidate_mean.toFixed(4)}
              </td>
              <td className="py-0.5 text-right font-mono">
                {m.mean_delta === null ? "—" : m.mean_delta.toFixed(4)}
              </td>
              <td className="py-0.5 text-right font-mono">{m.improved_count}</td>
              <td className="py-0.5 text-right font-mono">{m.regressed_count}</td>
              <td
                className={cn(
                  "py-0.5",
                  m.outcome === "IMPROVED"
                    ? "text-ok"
                    : m.outcome === "REGRESSED"
                      ? "text-warn"
                      : "text-ink-faint",
                )}
              >
                {m.outcome}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {comparison.significance_tests.map((t, i) => (
        <p key={i} className="mt-1 text-2xs text-ink-faint">
          {t.name}: n={t.n} p={t.p_value === null ? "not computable" : t.p_value.toFixed(4)}
          {t.exploratory ? " (exploratory — below the inference floor)" : ""} · {t.note}
        </p>
      ))}
      {comparison.notes.map((n, i) => (
        <p key={i} className="mt-1 text-2xs text-ink-faint">
          {n}
        </p>
      ))}
    </div>
  );
}

function RecommendationCard({ rec }: { rec: RetrievalRecommendation }) {
  const experiment = String(
    (rec.thresholds_used as Record<string, unknown>)?.validation_experiment ?? "",
  );
  return (
    <div className="rounded border border-line p-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge
          tone={rec.strength === "strong" ? "ok" : rec.strength === "moderate" ? "warn" : "neutral"}
        >
          {rec.strength}
        </Badge>
        <span className="font-mono text-xs text-ink">{rec.action}</span>
        <span className="text-2xs text-ink-faint">
          {rec.affected_question_ids.length} question(s) affected
        </span>
      </div>
      <p className="mt-1 text-2xs text-ink">{rec.reason}</p>
      <details className="mt-1">
        <summary className="cursor-pointer text-2xs text-ink-faint">
          supporting measurements (what the rule read)
        </summary>
        <pre className="mt-1 max-h-56 overflow-auto rounded bg-surface-3 p-2 text-2xs text-ink-muted">
          {JSON.stringify(rec.supporting_measurements, null, 2)}
        </pre>
      </details>
      {experiment && (
        <p className="mt-1 text-2xs text-ink-muted">
          <span className="text-ink-faint">validation experiment:</span> {experiment}
        </p>
      )}
      <p className="mt-1 text-2xs text-ink-muted">
        <span className="text-ink-faint">expected effect:</span> {rec.expected_effect}
      </p>
      <p className="mt-1 text-2xs text-ink-muted">
        <span className="text-ink-faint">risk:</span> {rec.risk}
      </p>
      {rec.experiment_required && (
        <p className="mt-1 flex items-center gap-1 text-2xs text-warn">
          <Target className="h-3 w-3" /> A controlled experiment is required before this is applied.
        </p>
      )}
    </div>
  );
}
