"use client";

/**
 * Reliability Dashboard — answer quality ACROSS retrieval strategies (V8 STEP 16).
 *
 * Design rules:
 * - Dimensions stay SEPARATE. There is deliberately NO combined "RAGForge
 *   score": a system can have high retrieval recall and poor answer relevance,
 *   or strong groundedness and poor completeness, and one number would hide
 *   exactly the contrast this project exists to surface.
 * - UNKNOWN is rendered as UNKNOWN with its reason, never as 0.
 * - Comparisons show their VERDICT first (COMPARABLE / INCONCLUSIVE /
 *   NOT_COMPARABLE). An INCONCLUSIVE comparison computed over a question
 *   intersection is never presented as a full-set result.
 * - Every run shows its producers (generator/mock flag, evaluator, relevance
 *   method, benchmark lifecycle, official flag) next to its numbers.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { GitCompare, RefreshCw, Sigma } from "lucide-react";
import {
  answerEvaluationApi,
  AnswerEvaluationRunSummary,
  Measured,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, Button, EmptyState, Panel, PanelHeader } from "@/components/ui";

/** One dimension group. Groups are rendered side by side and never summed. */
const DIMENSIONS: { label: string; metrics: [string, string][] }[] = [
  {
    label: "Retrieval",
    metrics: [["retrieval_hit_rate", "retrieval_hit_rate"]],
  },
  {
    label: "Citations",
    metrics: [
      ["citation_precision", "citation_precision"],
      ["citation_recall", "citation_recall"],
    ],
  },
  {
    label: "Groundedness",
    metrics: [
      ["evidence_support_rate", "evidence_support_rate"],
      ["contradiction_rate", "contradiction_rate"],
      ["unsupported_claim_ratio", "unsupported_claim_ratio"],
    ],
  },
  {
    label: "Fabrication",
    metrics: [
      ["fabricated_citation_rate", "fabricated_citation_rate"],
      ["unsupported_citation_rate", "unsupported_citation_rate"],
    ],
  },
  {
    label: "Relevance",
    metrics: [
      ["question_answer_relevance", "question_answer_relevance"],
      ["relevance_failure_rate", "relevance_failure_rate"],
    ],
  },
  {
    label: "Completeness",
    metrics: [
      ["expected_information_coverage", "expected_information_coverage"],
      ["correctness", "correctness"],
    ],
  },
  {
    label: "Abstention",
    metrics: [
      ["abstention_accuracy", "abstention_accuracy"],
      ["grounding_state_accuracy", "grounding_state_accuracy"],
    ],
  },
  {
    label: "Overall checks",
    metrics: [["pass_rate", "pass_rate"]],
  },
];

function aggMetric(
  run: AnswerEvaluationRunSummary,
  name: string,
): Measured | undefined {
  return (run.aggregate as unknown as Record<string, unknown>)[name] as
    | Measured
    | undefined;
}

function Cell({ m }: { m: Measured | undefined }) {
  if (!m || !m.measured || m.value === null) {
    return (
      <span className="text-ink-faint" title={m?.reason ?? "not present"}>
        UNKNOWN
      </span>
    );
  }
  return (
    <span title={m.reason}>
      {m.value.toFixed(3)}
      {m.sample_size != null && (
        <span className="ml-1 text-2xs text-ink-faint">n={m.sample_size}</span>
      )}
    </span>
  );
}

export default function ReliabilityPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [runs, setRuns] = useState<AnswerEvaluationRunSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [comparison, setComparison] = useState<
    Awaited<ReturnType<typeof answerEvaluationApi.compare>> | null
  >(null);
  const [comparing, setComparing] = useState(false);

  const load = useCallback(async () => {
    if (!kbId) return;
    setLoading(true);
    setError(null);
    try {
      const list = await answerEvaluationApi.list(kbId, 20);
      setRuns(list);
      setLeftId((cur) => cur || list[1]?.id || list[0]?.id || "");
      setRightId((cur) => cur || list[0]?.id || "");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => {
    void load();
  }, [load]);

  const compare = useCallback(async () => {
    if (!kbId || !leftId || !rightId) return;
    setComparing(true);
    setError(null);
    try {
      setComparison(await answerEvaluationApi.compare(kbId, leftId, rightId));
    } catch (e) {
      setComparison(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setComparing(false);
    }
  }, [kbId, leftId, rightId]);

  const byId = useMemo(
    () => Object.fromEntries(runs.map((r) => [r.id, r])),
    [runs],
  );

  const verdictTone = (v: string | undefined) =>
    v === "COMPARABLE" ? "ok" : v === "INCONCLUSIVE" ? "warn" : "bad";

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-medium text-ink">Reliability Dashboard</h1>
          <p className="mt-0.5 max-w-3xl text-2xs text-ink-faint">
            Answer-quality dimensions across retrieval strategies — kept
            separate on purpose. There is <strong>no combined score</strong>:
            retrieval recall, citation correctness, groundedness, relevance,
            completeness and abstention can each move alone, and a single
            number would hide that. UNKNOWN means unmeasured, never zero.
          </p>
        </div>
        <Button onClick={load} loading={loading} variant="ghost" aria-label="Refresh">
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-bad/40 bg-bad/5 px-3 py-2 text-2xs text-bad">
          {error}
        </div>
      )}

      {runs.length === 0 && !loading && !error ? (
        <Panel>
          <EmptyState
            icon={<Sigma className="h-6 w-6" />}
            title="No answer-evaluation runs yet"
            hint="Run an evaluation on the Answer Quality page first; runs appear here for cross-strategy comparison."
          />
        </Panel>
      ) : (
        <Panel>
          <PanelHeader
            title={`Runs (${runs.length})`}
            right={<Badge tone="neutral">dimensions separate — no combined score</Badge>}
          />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1100px] border-collapse text-2xs">
              <thead>
                <tr className="border-b border-line text-left text-ink-faint">
                  <th className="py-1.5 pr-3 font-normal">run</th>
                  <th className="py-1.5 pr-3 font-normal">strategy</th>
                  <th className="py-1.5 pr-3 font-normal">flags</th>
                  {DIMENSIONS.map((d) => (
                    <th key={d.label} className="py-1.5 pr-3 font-normal" colSpan={d.metrics.length}>
                      {d.label}
                    </th>
                  ))}
                </tr>
                <tr className="border-b border-line text-2xs text-ink-faint">
                  <th />
                  <th />
                  <th />
                  {DIMENSIONS.flatMap((d) =>
                    d.metrics.map(([label]) => (
                      <th key={`${d.label}-${label}`} className="pb-1 pr-3 text-left font-normal">
                        {label.replace(/_/g, " ")}
                      </th>
                    )),
                  )}
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id} className="border-b border-line/60 align-top">
                    <td className="py-2 pr-3">
                      <div className="font-mono text-ink-muted">{run.strategy || "default"}</div>
                      <div className="text-ink-faint">{run.created_at.slice(0, 19)}</div>
                      <div className="font-mono text-2xs text-ink-faint">{run.id}</div>
                    </td>
                    <td className="py-2 pr-3 text-ink-muted">{run.strategy || "default"}</td>
                    <td className="py-2 pr-3">
                      <div className="flex flex-wrap gap-1">
                        {run.is_mock && <Badge tone="warn">mock generator</Badge>}
                        {run.official ? (
                          <Badge tone="ok">official</Badge>
                        ) : (
                          <Badge tone="neutral">non-official</Badge>
                        )}
                        <Badge tone={run.benchmark_lifecycle === "frozen" ? "ok" : "warn"}>
                          {run.benchmark_lifecycle || "?"}
                        </Badge>
                        {run.evaluator_is_model_based && (
                          <Badge tone="warn">LLM judge</Badge>
                        )}
                        <Badge tone="neutral">
                          {run.relevance_method || "relevance n/a"}
                        </Badge>
                      </div>
                    </td>
                    {DIMENSIONS.flatMap((d) =>
                      d.metrics.map(([label, key]) => (
                        <td key={`${run.id}-${label}`} className="py-2 pr-3 text-ink">
                          <Cell m={aggMetric(run, key)} />
                        </td>
                      )),
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {runs.length >= 2 && (
        <Panel>
          <PanelHeader
            title="Strategy comparison"
            right={
              comparison && (
                <Badge tone={verdictTone(comparison.verdict)}>
                  {comparison.verdict} ({comparison.mode})
                </Badge>
              )
            }
          />
          <div className="mb-3 flex flex-wrap items-end gap-2">
            <label className="block">
              <span className="section-label">Left run</span>
              <select
                value={leftId}
                onChange={(e) => setLeftId(e.target.value)}
                className="mt-0.5 rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink"
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.strategy || "default"} — {r.created_at.slice(0, 19)}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="section-label">Right run</span>
              <select
                value={rightId}
                onChange={(e) => setRightId(e.target.value)}
                className="mt-0.5 rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink"
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.strategy || "default"} — {r.created_at.slice(0, 19)}
                  </option>
                ))}
              </select>
            </label>
            <Button onClick={compare} loading={comparing}>
              <GitCompare className="mr-1.5 h-3.5 w-3.5" />
              Compare
            </Button>
          </div>

          {comparison && (
            <div
              className={cn(
                "mb-3 rounded-md border px-3 py-2 text-2xs",
                comparison.verdict === "COMPARABLE"
                  ? "border-ok/40 bg-ok/5 text-ok"
                  : comparison.verdict === "INCONCLUSIVE"
                    ? "border-warn/40 bg-warn/5 text-warn"
                    : "border-bad/40 bg-bad/5 text-bad",
              )}
            >
              <strong>{comparison.verdict}</strong> — {comparison.reason}
              {comparison.shared_question_ids.length > 0 && (
                <div className="mt-1 font-mono text-ink-faint">
                  shared questions: {comparison.shared_question_ids.join(", ")}
                </div>
              )}
            </div>
          )}

          {comparison && Object.keys(comparison.differences).length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-2xs">
                <thead>
                  <tr className="border-b border-line text-left text-ink-faint">
                    <th className="py-1.5 pr-3 font-normal">metric</th>
                    <th className="py-1.5 pr-3 font-normal">
                      left ({byId[comparison.left.id]?.strategy ?? "?"})
                    </th>
                    <th className="py-1.5 pr-3 font-normal">
                      right ({byId[comparison.right.id]?.strategy ?? "?"})
                    </th>
                    <th className="py-1.5 font-normal">delta (right − left)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(comparison.differences).map(([name, d]) => (
                    <tr key={name} className="border-b border-line/60">
                      <td className="py-1.5 pr-3 text-ink-muted">{name.replace(/_/g, " ")}</td>
                      <td className="py-1.5 pr-3 text-ink">
                        <Cell m={d.left} />
                      </td>
                      <td className="py-1.5 pr-3 text-ink">
                        <Cell m={d.right} />
                      </td>
                      <td className="py-1.5">
                        {d.delta === null ? (
                          <span className="text-ink-faint">n/a (unmeasured)</span>
                        ) : (
                          <span className={d.delta > 0 ? "text-ok" : d.delta < 0 ? "text-warn" : "text-ink-faint"}>
                            {d.delta > 0 ? "+" : ""}
                            {d.delta.toFixed(3)}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            comparison && (
              <p className="text-2xs text-ink-faint">
                No differences computed — the verdict above explains why.
              </p>
            )
          )}

          <p className="mt-3 border-t border-line pt-2 text-2xs text-ink-faint">
            Runs over different question subsets are never shown as identical
            comparisons: partial overlap yields an INCONCLUSIVE verdict over
            the shared questions only, and zero overlap yields no numbers at
            all.
          </p>
        </Panel>
      )}
    </div>
  );
}
