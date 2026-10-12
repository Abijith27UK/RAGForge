"use client";

/**
 * Held-out Validation — V14.
 *
 * Answers ONE question: "do the V13 retrieval findings hold on questions the
 * configurations were never chosen on?"
 *
 * The page keeps three kinds of statement apart, because conflating them is how
 * a dashboard turns a draft into a fake fact:
 *
 *   1. MEASURED — overlap counts, availability verdicts, similarity scores,
 *      metric means and per-question movement. A metric with `measured: false`
 *      renders as UNKNOWN with its reason, never as 0.0.
 *   2. DRAFT — every candidate question. Each is labelled with its status, and
 *      the evidence it rests on is shown as the FULL chunk text, not a summary,
 *      so a reviewer reads the corpus rather than the author's claim about it.
 *   3. CLAIMED — the freeze tier and claim strength. `PROVISIONAL` renders as a
 *      warning everywhere results appear, because a result from an unreviewed
 *      instrument is exploratory and the page must not let it read otherwise.
 *
 * The one thing this page will not do is let a reviewer approve work without
 * naming themselves: the reviewer field is required and stored per question.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  CircleAlert,
  CircleCheck,
  FlaskConical,
  Lock,
  RefreshCw,
  ShieldQuestion,
  Target,
  TriangleAlert,
} from "lucide-react";
import {
  HeldOutExperimentRow,
  HeldOutGeneralizationSummary,
  HeldOutReviewQuestion,
  HeldOutReviewView,
  HeldOutStatus,
  heldOutApi,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, Button, EmptyState, Metric, Panel, PanelHeader } from "@/components/ui";

const REVIEWER_KEY = "ragforge.heldout.reviewer";

const STATUS_TONE: Record<string, "ok" | "warn" | "bad" | "neutral" | "accent"> = {
  approved: "ok",
  edited: "accent",
  rejected: "bad",
  draft: "warn",
};

const OUTCOME_TONE: Record<string, "ok" | "warn" | "bad" | "neutral" | "accent"> = {
  IMPROVED: "ok",
  REGRESSION: "bad",
  NO_MEANINGFUL_IMPROVEMENT: "neutral",
  INCONCLUSIVE: "warn",
  BLOCKED: "bad",
  PENDING: "neutral",
};

function tone(t: string): "ok" | "warn" | "bad" | "neutral" | "accent" {
  return (STATUS_TONE[t] ?? "neutral") as "ok" | "warn" | "bad" | "neutral" | "accent";
}

function metricCell(stats: Record<string, { mean: number | null; measured: boolean } | undefined>, key: string) {
  const s = stats?.[key];
  if (!s || !s.measured || s.mean === null || s.mean === undefined) return "n/a";
  return s.mean.toFixed(4);
}

export default function HeldOutValidationPage() {
  const params = useParams<{ id: string }>();
  const kbId = params.id;

  const [status, setStatus] = useState<HeldOutStatus | null>(null);
  const [reviewView, setReviewView] = useState<HeldOutReviewView | null>(null);
  const [experiments, setExperiments] = useState<HeldOutExperimentRow[]>([]);
  const [devReferences, setDevReferences] = useState<HeldOutExperimentRow[]>([]);
  const [summaries, setSummaries] = useState<HeldOutGeneralizationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const [reviewer, setReviewer] = useState("");
  const [editing, setEditing] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const stored = typeof window !== "undefined" ? window.localStorage.getItem(REVIEWER_KEY) : null;
    if (stored) setReviewer(stored);
  }, []);

  const rememberReviewer = (value: string) => {
    setReviewer(value);
    if (typeof window !== "undefined") window.localStorage.setItem(REVIEWER_KEY, value);
  };

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, rv, ex] = await Promise.all([
        heldOutApi.status(kbId),
        heldOutApi.reviewView(kbId),
        heldOutApi.experiments(kbId),
      ]);
      setStatus(s);
      setReviewView(rv);
      setExperiments(ex.experiments);
      setDevReferences(ex.development_references);
      setSummaries(ex.generalization_summaries);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load the held-out benchmark");
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => {
    void load();
  }, [load]);

  const decide = async (
    question: HeldOutReviewQuestion,
    decision: "approve" | "edit" | "reject",
  ) => {
    if (!reviewer.trim()) {
      setError("Enter your name before recording a decision — an anonymous review cannot be audited.");
      return;
    }
    setBusy(question.id);
    setError(null);
    try {
      const body = {
        question_id: question.id,
        decision,
        reviewer: reviewer.trim(),
        question: decision === "edit" ? (editing[question.id] ?? question.question) : undefined,
      };
      await heldOutApi.review(kbId, [body]);
      setFlash(`Recorded ${decision} for ${question.id}`);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to record the decision");
    } finally {
      setBusy(null);
    }
  };

  const freeze = async () => {
    if (!reviewer.trim()) {
      setError("Enter your name before freezing — the tier this seals is a claim about who reviewed what.");
      return;
    }
    setBusy("freeze");
    setError(null);
    try {
      const result = await heldOutApi.freeze(kbId, {
        frozen_by: reviewer.trim(),
        notes: [
          "Frozen from the V14 held-out validation page. The tier reflects the review " +
            "records that existed at this moment and cannot be raised afterwards.",
        ],
      });
      setFlash(
        `Frozen as ${result.lifecycle.toUpperCase()} (${result.claim_strength}) — fingerprint ${result.fingerprint}`,
      );
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to freeze the benchmark");
    } finally {
      setBusy(null);
    }
  };

  const review = status?.review;
  const validation = status?.validation;
  const frozen = status?.frozen ?? null;

  const pending = useMemo(
    () => (reviewView?.questions ?? []).filter((q) => q.needs_review),
    [reviewView],
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-base font-semibold text-ink">Held-out Validation</h1>
          <p className="text-2xs text-ink-muted">
            A separate instrument from the development benchmark, used to test whether the V13
            findings generalize to questions the configurations were never chosen on.
          </p>
        </div>
        <Button variant="ghost" onClick={() => void load()} loading={loading}>
          <RefreshCw className="mr-1 h-3.5 w-3.5" />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-2xs text-bad">
          <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      {flash && (
        <div className="flex items-start gap-2 rounded border border-ok/40 bg-ok/10 px-3 py-2 text-2xs text-ok">
          <CircleCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{flash}</span>
        </div>
      )}

      {/* Instrument identity + gate */}
      <Panel>
        <PanelHeader title="Instrument identity and gate" />
        {!status ? (
          <p className="text-2xs text-ink-muted">{loading ? "Loading…" : "Unavailable"}</p>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-2xs">
              <Badge tone={frozen ? (frozen.lifecycle === "validated" ? "ok" : "warn") : "neutral"}>
                {frozen ? frozen.lifecycle : "not frozen"}
              </Badge>
              <Badge tone={frozen?.claim_strength === "confirmatory" ? "ok" : "warn"}>
                {frozen ? `${frozen.claim_strength} results only` : "exploratory only"}
              </Badge>
              <span className="text-ink-faint">
                {status.surviving_count} surviving · {status.rejected_count} rejected ·{" "}
                {review?.candidates_reviewed ?? 0}/{review?.candidates_surviving ?? 0} reviewed
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Metric label="Question count" value={String(status.surviving_count)} />
              <Metric
                label="Candidate fingerprint"
                value={<span className="font-mono text-xs">{status.candidate_fingerprint}</span>}
              />
              <Metric
                label="Shared chunks with development set"
                value={String(
                  validation?.separation.candidates_sharing_expected_chunk_with_development ?? 0,
                )}
                tone={
                  (validation?.separation.candidates_sharing_expected_chunk_with_development ?? 0) > 0
                    ? "warn"
                    : "ok"
                }
              />
              <Metric
                label="Max lexical similarity vs dev"
                value={
                  validation
                    ? `${validation.separation.max_token_jaccard_vs_development.toFixed(2)} token / ${validation.separation.max_char_ngram_jaccard_vs_development.toFixed(2)} char`
                    : "—"
                }
                tone={
                  validation &&
                  (validation.separation.max_token_jaccard_vs_development >=
                    validation.separation.thresholds.token_jaccard ||
                    validation.separation.max_char_ngram_jaccard_vs_development >=
                      validation.separation.thresholds.char_ngram_jaccard)
                    ? "warn"
                    : "ok"
                }
              />
            </div>

            {frozen && (
              <div className="space-y-1 rounded border border-line px-2 py-1.5 text-2xs text-ink-muted">
                <div>
                  <span className="text-ink-faint">frozen</span> {frozen.frozen_at} ·{" "}
                  <span className="text-ink-faint">by</span> {frozen.frozen_by}
                </div>
                <div className="font-mono break-all">
                  {frozen.benchmark} v{frozen.version} · fingerprint {frozen.fingerprint} · sha256{" "}
                  {frozen.sha256.slice(0, 24)}…
                </div>
              </div>
            )}

            <p className="text-2xs text-ink-faint">{status.notice}</p>
          </div>
        )}
      </Panel>

      {/* Validation */}
      <Panel>
        <PanelHeader title="Validation of the held-out questions" />
        {!validation ? (
          <p className="text-2xs text-ink-muted">Unavailable</p>
        ) : (
          <div className="space-y-3">
            <div
              className={cn(
                "rounded border px-2 py-1.5 text-2xs",
                validation.errors.length > 0
                  ? "border-bad/40 bg-bad/10 text-bad"
                  : "border-ok/40 bg-ok/10 text-ok",
              )}
            >
              {validation.verdict}
            </div>

            {validation.errors.map((e) => (
              <div key={e} className="text-2xs text-bad">
                ERROR — {e}
              </div>
            ))}
            {validation.warnings.map((w) => (
              <div key={w} className="text-2xs text-warn">
                WARNING — {w}
              </div>
            ))}
            {validation.checks_not_run.length > 0 && (
              <div className="space-y-0.5">
                {validation.checks_not_run.map((c) => (
                  <div key={c.check} className="text-2xs text-warn">
                    NOT RUN — {c.check}: {c.reason}
                  </div>
                ))}
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 text-2xs uppercase tracking-wider text-ink-faint">
                  Subdomains
                </div>
                <div className="space-y-0.5 text-2xs text-ink-muted">
                  {Object.entries(validation.composition.subdomains).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span>{k}</span>
                      <span className="font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-1 text-2xs uppercase tracking-wider text-ink-faint">
                  Query types
                </div>
                <div className="space-y-0.5 text-2xs text-ink-muted">
                  {Object.entries(validation.composition.query_types).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span>{k}</span>
                      <span className="font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-1 text-2xs uppercase tracking-wider text-ink-faint">
                  Difficulty
                </div>
                <div className="space-y-0.5 text-2xs text-ink-muted">
                  {Object.entries(validation.composition.difficulty).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span>{k}</span>
                      <span className="font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <p className="text-2xs text-ink-faint">{validation.separation.note}</p>
            {validation.limitations.map((l) => (
              <p key={l} className="text-2xs text-ink-faint">
                LIMITATION — {l}
              </p>
            ))}
          </div>
        )}
      </Panel>

      {/* Review */}
      <Panel>
        <PanelHeader
          title={
            <span className="flex items-center gap-2">
              <ShieldQuestion className="h-3.5 w-3.5" />
              Human review ({pending.length} awaiting a decision)
            </span>
          }
        />
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={reviewer}
              onChange={(e) => rememberReviewer(e.target.value)}
              placeholder="Your name (required, stored with each decision)"
              className="h-8 w-72 rounded border border-line-strong bg-surface-2 px-2 text-xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
            />
            <span className="text-2xs text-ink-faint">
              Decisions are recorded per question and are the only thing that can make this
              instrument VALIDATED.
            </span>
          </div>

          {frozen && (
            <div className="flex items-start gap-2 rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-2xs text-warn">
              <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                The instrument is frozen and immutable. Reviewing after a run cannot upgrade it:
                a reviewer who has seen the results is no longer blind to them.
              </span>
            </div>
          )}

          {!reviewView ? (
            <p className="text-2xs text-ink-muted">{loading ? "Loading…" : "Unavailable"}</p>
          ) : (
            <div className="space-y-2">
              {reviewView.questions.map((q) => (
                <div key={q.id} className="rounded border border-line px-2 py-2">
                  <div className="flex flex-wrap items-center gap-2 text-2xs">
                    <span className="font-mono text-ink-faint">{q.id}</span>
                    <Badge tone={tone(q.status)}>{q.status}</Badge>
                    <span className="text-ink-faint">{q.subdomain || "(no subdomain)"}</span>
                    <span className="text-ink-faint">
                      {q.query_type} · {q.difficulty}
                    </span>
                    {q.review && (
                      <span className="text-ink-faint">
                        {q.review.decision} by {q.review.reviewer} @{" "}
                        {q.review.reviewed_at.slice(0, 19)}
                      </span>
                    )}
                  </div>

                  <p className="mt-1 text-xs text-ink">{q.question}</p>

                  <div className="mt-1 space-y-0.5 text-2xs text-ink-faint">
                    <div>
                      <span className="uppercase tracking-wider">covers</span>{" "}
                      {q.coverage_rationale}
                    </div>
                    <div className="font-mono break-all">{q.provenance}</div>
                  </div>

                  <div className="mt-1">
                    <button
                      type="button"
                      className="text-2xs text-accent-soft underline-offset-2 hover:underline"
                      onClick={() => setExpanded((s) => ({ ...s, [q.id]: !s[q.id] }))}
                    >
                      {expanded[q.id] ? "Hide" : "Show"} evidence ({q.expected_chunk_ids.length}{" "}
                      chunk{q.expected_chunk_ids.length === 1 ? "" : "s"})
                    </button>
                    {expanded[q.id] && (
                      <div className="mt-1 space-y-1">
                        {q.evidence.map((ev) => (
                          <div
                            key={ev.chunk_id}
                            className="rounded border border-line bg-surface-2 px-2 py-1.5"
                          >
                            <div className="flex flex-wrap items-center gap-2 text-2xs text-ink-faint">
                              <span className="font-mono">{ev.chunk_id}</span>
                              <span>{ev.document_title}</span>
                              <span>section {ev.section || "unknown"}</span>
                              <Badge tone={ev.resolves_in_corpus ? "ok" : "bad"}>
                                {ev.resolves_in_corpus ? "in corpus" : "missing from corpus"}
                              </Badge>
                              <Badge
                                tone={
                                  ev.present_in_index === null
                                    ? "warn"
                                    : ev.present_in_index
                                      ? "ok"
                                      : "bad"
                                }
                              >
                                {ev.present_in_index === null
                                  ? "index unverified"
                                  : ev.present_in_index
                                    ? "indexed"
                                    : "not indexed"}
                              </Badge>
                              <span>{ev.char_count} chars</span>
                            </div>
                            <pre className="mt-1 max-h-56 overflow-auto whitespace-pre-wrap break-words text-2xs text-ink-muted">
                              {ev.text}
                            </pre>
                            <div className="mt-1 text-2xs text-ok">
                              claimed answering sentence: “{q.evidence_quote}”
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {q.findings.filter((f) => f.severity !== "info").length > 0 && (
                    <div className="mt-1 space-y-0.5">
                      {q.findings
                        .filter((f) => f.severity !== "info")
                        .map((f, i) => (
                          <div
                            key={`${q.id}-${f.code}-${i}`}
                            className={cn(
                              "text-2xs",
                              f.severity === "error" ? "text-bad" : "text-warn",
                            )}
                          >
                            {f.severity.toUpperCase()} {f.code} — {f.detail} → {f.action}
                          </div>
                        ))}
                    </div>
                  )}

                  {!frozen && (
                    <div className="mt-2 space-y-1">
                      <textarea
                        value={editing[q.id] ?? q.question}
                        onChange={(e) => setEditing((s) => ({ ...s, [q.id]: e.target.value }))}
                        rows={2}
                        className="w-full rounded border border-line-strong bg-surface-2 px-2 py-1 text-2xs text-ink focus:border-line-focus focus:outline-none"
                      />
                      <div className="flex flex-wrap gap-1">
                        <Button
                          size="sm"
                          variant="primary"
                          loading={busy === q.id}
                          onClick={() => void decide(q, "approve")}
                        >
                          Approve
                        </Button>
                        <Button
                          size="sm"
                          loading={busy === q.id}
                          disabled={(editing[q.id] ?? q.question).trim() === q.question.trim()}
                          onClick={() => void decide(q, "edit")}
                        >
                          Save edit
                        </Button>
                        <Button
                          size="sm"
                          variant="danger"
                          loading={busy === q.id}
                          onClick={() => void decide(q, "reject")}
                        >
                          Reject
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {!frozen && (
            <div className="flex flex-wrap items-center gap-2 border-t border-line pt-3">
              <Button variant="primary" loading={busy === "freeze"} onClick={() => void freeze()}>
                <Lock className="mr-1 h-3.5 w-3.5" />
                Freeze the instrument
              </Button>
              <span className="text-2xs text-ink-faint">
                {review?.complete
                  ? "Every surviving question is reviewed → this will seal as VALIDATED (confirmatory results allowed)."
                  : `${review?.candidates_unreviewed ?? 0} question(s) unreviewed → this will seal as PROVISIONAL (exploratory results only). Freezing is one-way.`}
              </span>
            </div>
          )}
        </div>
      </Panel>

      {/* Results */}
      <Panel>
        <PanelHeader
          title={
            <span className="flex items-center gap-2">
              <FlaskConical className="h-3.5 w-3.5" />
              Pre-registered validation experiments
            </span>
          }
        />
        {experiments.length === 0 ? (
          <EmptyState
            icon={<Target className="h-4 w-4" />}
            title="No held-out experiment has been run yet"
            hint="Run `py -m scripts.run_heldout_validation --experiment all` after freezing. The arms are pre-declared in that script; nothing here tunes them."
          />
        ) : (
          <div className="space-y-4">
            {experiments.map((exp) => (
              <div key={exp.id} className="rounded border border-line px-2 py-2">
                <div className="flex flex-wrap items-center gap-2 text-2xs">
                  <span className="font-semibold text-ink">{exp.name}</span>
                  <Badge tone={OUTCOME_TONE[exp.outcome] ?? "neutral"}>{exp.outcome}</Badge>
                  <Badge tone={exp.claim_strength === "confirmatory" ? "ok" : "warn"}>
                    {exp.claim_strength || "exploratory"}
                  </Badge>
                  <Badge tone="neutral">tier {exp.benchmark_lifecycle || "unknown"}</Badge>
                  <span className="text-ink-faint">
                    {exp.question_count} questions · retests {exp.retests || "—"}
                  </span>
                </div>

                <p className="mt-1 text-2xs text-ink-muted">{exp.outcome_explanation}</p>

                <div className="mt-1 overflow-x-auto">
                  <table className="w-full text-2xs">
                    <thead>
                      <tr className="text-left text-ink-faint">
                        <th className="pr-3 font-medium">arm</th>
                        <th className="pr-3 font-medium">status</th>
                        <th className="pr-3 font-medium">recall@k</th>
                        <th className="pr-3 font-medium">precision@k</th>
                        <th className="pr-3 font-medium">MRR</th>
                        <th className="pr-3 font-medium">nDCG@k</th>
                        <th className="pr-3 font-medium">latency</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono text-ink-muted">
                      {exp.arms.map((arm) => (
                        <tr key={arm.label}>
                          <td className="pr-3">{arm.label}</td>
                          <td className="pr-3">{arm.status}</td>
                          <td className="pr-3">{metricCell(arm.metrics, "recall_at_k")}</td>
                          <td className="pr-3">{metricCell(arm.metrics, "precision_at_k")}</td>
                          <td className="pr-3">{metricCell(arm.metrics, "mrr")}</td>
                          <td className="pr-3">{metricCell(arm.metrics, "ndcg_at_k")}</td>
                          <td className="pr-3">
                            {arm.metrics?.latency_ms?.measured
                              ? `${arm.metrics.latency_ms.mean?.toFixed(1)} ms`
                              : "n/a"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {exp.candidate_outcomes.map((o) => (
                  <div
                    key={o.label}
                    className={cn(
                      "mt-1 rounded border px-2 py-1 text-2xs",
                      o.outcome === "IMPROVED"
                        ? "border-ok/40 bg-ok/10 text-ok"
                        : o.outcome === "REGRESSION"
                          ? "border-bad/40 bg-bad/10 text-bad"
                          : "border-line text-ink-muted",
                    )}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-semibold">{o.label}</span>
                      <Badge tone={OUTCOME_TONE[o.outcome] ?? "neutral"}>{o.outcome}</Badge>
                      <Badge tone={o.acceptance === "ACCEPT" ? "ok" : "warn"}>
                        {o.acceptance}
                      </Badge>
                      <span className="font-mono">
                        {exp.primary_metric} Δ {o.primary_mean_delta?.toFixed(4) ?? "n/a"} (bar{" "}
                        {o.primary_required_delta})
                      </span>
                      <span className="font-mono">
                        wins {o.paired_wins ?? "n/a"} / losses {o.paired_losses ?? "n/a"} of{" "}
                        {o.paired_question_count}
                      </span>
                      <span className="font-mono">
                        sign-test p {o.primary_sign_test_p?.toFixed(4) ?? "n/a"} (n{" "}
                        {o.primary_sign_test_n})
                      </span>
                    </div>
                    <div className="mt-0.5">{o.outcome_explanation}</div>
                    {o.guardrails_regressed.length > 0 && (
                      <div className="mt-0.5 text-warn">
                        guardrails regressed: {o.guardrails_regressed.join(", ")}
                      </div>
                    )}
                  </div>
                ))}

                {exp.limitations.length > 0 && (
                  <div className="mt-1 space-y-0.5">
                    {exp.limitations.map((l) => (
                      <div key={l} className="text-2xs text-ink-faint">
                        LIMITATION — {l}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Panel>

      {/* Generalization */}
      {summaries.length > 0 && (
        <Panel>
          <PanelHeader title="Development vs held-out (unpaired)" />
          <div className="space-y-3">
            {summaries.map((s) => (
              <div key={s.experiment} className="rounded border border-line px-2 py-2">
                <div className="flex flex-wrap items-center gap-2 text-2xs">
                  <span className="font-semibold text-ink">{s.experiment}</span>
                  <span className="text-ink-faint">
                    {s.dev_question_count} dev questions ({s.dev_outcome || "—"}) →{" "}
                    {s.heldout_question_count} held-out questions ({s.heldout_outcome})
                  </span>
                  <Badge tone={s.heldout_claim_strength === "confirmatory" ? "ok" : "warn"}>
                    {s.heldout_claim_strength}
                  </Badge>
                </div>
                <table className="mt-1 w-full text-2xs">
                  <thead>
                    <tr className="text-left text-ink-faint">
                      <th className="pr-3 font-medium">candidate</th>
                      <th className="pr-3 font-medium">held-out Δ{s.primary_metric}</th>
                      <th className="pr-3 font-medium">held-out verdict</th>
                      <th className="pr-3 font-medium">dev Δ{s.primary_metric}</th>
                      <th className="pr-3 font-medium">direction</th>
                    </tr>
                  </thead>
                  <tbody className="text-ink-muted">
                    {Object.entries(s.agreement).map(([label, a]) => (
                      <tr key={label}>
                        <td className="pr-3 font-mono">{label}</td>
                        <td className="pr-3 font-mono">
                          {a.comparable ? (a.heldout_delta ?? 0).toFixed(4) : "n/a"}
                        </td>
                        <td className="pr-3">{a.heldout_outcome || "—"}</td>
                        <td className="pr-3 font-mono">
                          {a.development_delta !== undefined ? a.development_delta.toFixed(4) : "—"}
                        </td>
                        <td className="pr-3">
                          {a.direction_agrees === undefined ? (
                            "—"
                          ) : a.direction_agrees ? (
                            <Badge tone="ok">agrees</Badge>
                          ) : (
                            <Badge tone="warn">differs</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-1 flex items-start gap-1 text-2xs text-ink-faint">
                  <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" />
                  {s.note}
                </p>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {devReferences.length > 0 && (
        <Panel>
          <PanelHeader title="Development reference re-measurements" />
          <div className="space-y-1 text-2xs">
            {devReferences.map((exp) => (
              <div key={exp.id} className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-ink-muted">{exp.name}</span>
                <Badge tone={OUTCOME_TONE[exp.outcome] ?? "neutral"}>{exp.outcome}</Badge>
                <span className="text-ink-faint">
                  {exp.question_count} questions · {exp.created_at.slice(0, 19)}
                </span>
              </div>
            ))}
            <p className="text-ink-faint">
              These are re-measurements of the 28 development questions under identical code, so
              both instruments can be compared from the same session. They are baselines, not
              findings: those questions are the ones the arms were chosen on.
            </p>
          </div>
        </Panel>
      )}
    </div>
  );
}
