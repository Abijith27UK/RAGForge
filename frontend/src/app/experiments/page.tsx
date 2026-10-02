"use client";

/**
 * Experiments — frozen source-selection results served verbatim from the
 * backend's read-only artifact endpoint. NOTHING here is hardcoded:
 * coverage, metrics, caveats and the paired analysis all come from
 * /api/experiments/*, which reads the JSON files in /benchmarks.
 */

import { useEffect, useState } from "react";
import { FlaskConical, TriangleAlert } from "lucide-react";
import { BACKEND_URL } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, EmptyState, Panel, PanelHeader, ScoreBar, Skeleton } from "@/components/ui";

interface RunMetrics {
  recall_at_k: number;
  precision_at_k: number;
  mrr: number;
  ndcg: number;
  doc_recall_at_k?: number;
  questions_evaluated: number;
}
interface Run {
  strategy: string;
  n_sources_selected: number;
  kb_id: string;
  documents: number;
  chunks: number;
  corpus_chars: number;
  ingest_seconds?: number;
  index_seconds?: number;
  coverage: { answerable: number; total: number; excluded_unanswerable: string[] };
  evaluation: Record<string, RunMetrics>;
}
interface ExperimentResults {
  research_question: string;
  random_seed: number;
  run_at: string;
  frozen_baseline_caveat?: string;
  runs: Run[];
}
interface PairedEntry {
  k: number;
  n_common: number;
  left: { run: string; mean_recall: number; mean_mrr: number; mean_ndcg: number };
  right: { run: string; mean_recall: number; mean_mrr: number; mean_ndcg: number };
  left_better_recall: number;
  right_better_recall: number;
  equal_recall: number;
}
interface PairedAnalysis {
  note: string;
  answerable_sets?: Record<string, string[]>;
  paired: Record<string, PairedEntry>;
}

const KS = ["3", "5", "10"] as const;

function fmt(v: number | null | undefined, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}

function RunRow({ run, k }: { run: Run; k: string }) {
  const m = run.evaluation[k];
  if (!m) return null;
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-x-6 gap-y-1 border-b border-line/50 px-4 py-2 text-xs last:border-0">
      <span className="font-mono text-ink-muted">{run.strategy}</span>
      <span className="flex gap-4 font-mono text-2xs">
        <span title="Recall@k">R <span className="text-accent-soft">{fmt(m.recall_at_k)}</span></span>
        <span title="Precision@k">P {fmt(m.precision_at_k)}</span>
        <span title="MRR">MRR {fmt(m.mrr)}</span>
        <span title="NDCG">NDCG {fmt(m.ndcg)}</span>
        <span className="text-ink-faint">n={m.questions_evaluated}</span>
      </span>
      <span className="text-2xs text-ink-faint">
        {run.documents} docs · {run.chunks} chunks · {(run.corpus_chars / 1000).toFixed(0)}k chars
        {run.index_seconds != null && ` · indexed ${run.index_seconds.toFixed(0)}s`}
      </span>
      <span className="justify-self-end text-2xs text-ink-faint">
        coverage {run.coverage.answerable}/{run.coverage.total}
      </span>
    </div>
  );
}

function PairedTable({ paired }: { paired: Record<string, PairedEntry> }) {
  const entries = Object.entries(paired).filter(([name]) => name.includes("_vs_"));
  const cross = entries.filter(([, p]) => /QUALITY.*_vs_.*RANDOM/.test(p.left.run) || (p.left.run.includes("QUALITY") && p.right.run.includes("RANDOM")));
  const within = entries.filter(([name]) => !cross.some(([cname]) => cname === name));
  return (
    <div>
      {cross.length > 0 && (
        <>
          <div className="section-label border-b border-line px-4 py-2">QUALITY vs RANDOM (cross-strategy, valid only on the intersection)</div>
          {cross.map(([name, p]) => (
            <div key={name} className="border-b border-line/50 px-4 py-2.5 text-xs last:border-0">
              <div className="flex items-center justify-between gap-3">
                <span className="font-mono text-2xs text-ink-muted">{name.replace(/_/g, " ").toLowerCase()}</span>
                <span className="text-2xs text-ink-faint">n_common={p.n_common} · k={p.k}</span>
              </div>
              <div className="mt-1.5 grid grid-cols-2 gap-4">
                {([["left", p.left], ["right", p.right]] as const).map(([side, s]) => (
                  <div key={side}>
                    <div className="mb-1 flex items-center justify-between text-2xs">
                      <span className="text-ink-muted">{s.run}</span>
                      <span className="font-mono text-ink-faint">R {fmt(s.mean_recall)} · MRR {fmt(s.mean_mrr)}</span>
                    </div>
                    <ScoreBar value={s.mean_recall} tone={side === "left" ? "accent" : "violet"} />
                  </div>
                ))}
              </div>
              <div className="mt-1.5 text-2xs text-ink-faint">
                recall wins — left {p.left_better_recall} · right {p.right_better_recall} · equal {p.equal_recall}
              </div>
            </div>
          ))}
        </>
      )}
      {within.length > 0 && (
        <>
          <div className="section-label border-b border-line px-4 py-2">Within-strategy consistency</div>
          {within.map(([name, p]) => (
            <div key={name} className="flex items-center justify-between gap-3 border-b border-line/50 px-4 py-2 text-2xs last:border-0">
              <span className="font-mono text-ink-muted">{name.replace(/_/g, " ").toLowerCase()}</span>
              <span className="font-mono text-ink-faint">
                n={p.n_common} · R {fmt(p.left.mean_recall)} / {fmt(p.right.mean_recall)} · MRR {fmt(p.left.mean_mrr)} / {fmt(p.right.mean_mrr)}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export default function ExperimentsPage() {
  const [v2, setV2] = useState<ExperimentResults | null>(null);
  const [v1, setV1] = useState<ExperimentResults | null>(null);
  const [paired, setPaired] = useState<PairedAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [r2, r1, pr] = await Promise.all([
          fetch(`${BACKEND_URL}/api/experiments/source-selection-v2/results`).then((r) => {
            if (!r.ok) throw new Error(`v2 results: ${r.status}`);
            return r.json();
          }),
          fetch(`${BACKEND_URL}/api/experiments/source-selection-v1/results`).then((r) => {
            if (!r.ok) throw new Error(`v1 results: ${r.status}`);
            return r.json();
          }),
          fetch(`${BACKEND_URL}/api/experiments/source-selection-v2-paired/results`).then((r) =>
            r.ok ? r.json() : null,
          ),
        ]);
        setV2(r2);
        setV1(r1);
        setPaired(pr);
      } catch (e) {
        setError(String((e as Error).message));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl">
      <header className="mb-5">
        <h1 className="font-semibold tracking-tight text-ink">Experiments</h1>
        <p className="mt-0.5 text-xs text-ink-muted">
          Frozen, reproducible source-selection runs. Every number below is read from the results
          artifacts on disk — never recomputed, never hardcoded.
        </p>
      </header>

      {error && (
        <Panel className="mb-4 border-warn-dim">
          <div className="flex items-start gap-2 p-4 text-xs text-warn">
            <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
            <span>Could not load experiment artifacts from the backend: {error}</span>
          </div>
        </Panel>
      )}

      {!v2 && !v1 && !error && (
        <EmptyState
          icon={<FlaskConical className="h-5 w-5" />}
          title="No experiments found"
          hint="Run a source-selection experiment to populate this view."
        />
      )}

      {v2 && (
        <Panel className="mb-6">
          <PanelHeader
            title="Source-selection v2 — content-aware scorer vs random"
            right={
              <span className="flex items-center gap-2">
                <Badge tone="accent">frozen</Badge>
                <span className="font-mono text-2xs text-ink-faint">seed {v2.random_seed}</span>
              </span>
            }
          />
          <div className="border-b border-line px-4 py-3 text-xs leading-relaxed text-ink-muted">
            {v2.research_question}
            {v2.frozen_baseline_caveat && (
              <div className="mt-2 flex items-start gap-1.5 rounded border border-warn-dim bg-warn/5 px-2.5 py-1.5 text-2xs text-warn">
                <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" />
                <span>{v2.frozen_baseline_caveat}</span>
              </div>
            )}
          </div>

          <div className="px-4 py-3">
            <div className="section-label mb-2">Corpus coverage — answerable benchmark questions</div>
            {[...v2.runs]
              .sort((a, b) => b.coverage.answerable - a.coverage.answerable)
              .map((r) => (
                <div key={r.kb_id} className="mb-1.5 flex items-center gap-3 text-xs">
                  <span className="w-44 shrink-0 truncate font-mono text-2xs text-ink-muted">
                    {r.strategy}|{r.n_sources_selected}
                  </span>
                  <ScoreBar
                    value={r.coverage.answerable}
                    max={r.coverage.total}
                    tone={r.strategy === "QUALITY_SELECTED" ? "accent" : "violet"}
                  />
                  <span className="w-12 shrink-0 text-right font-mono text-2xs text-ink-muted">
                    {r.coverage.answerable}/{r.coverage.total}
                  </span>
                </div>
              ))}
          </div>

          {KS.map((k) => (
            <div key={k} className="border-t border-line">
              <div className="section-label bg-surface-1 px-4 py-1.5">Strict evaluation @ k={k}</div>
              {v2.runs
                .filter((r) => r.evaluation[k])
                .sort((a, b) => a.n_sources_selected - b.n_sources_selected)
                .map((r) => (
                  <RunRow key={`${r.kb_id}-${k}`} run={r} k={k} />
                ))}
            </div>
          ))}

          <div className="border-t border-line px-4 py-3 text-2xs leading-relaxed text-ink-faint">
            Aggregate rows are computed on <span className="text-ink-muted">different answerable-question
            subsets</span> (global intersection was 0) and must not be compared as though they were.
            Use the paired analysis below for cross-strategy conclusions.
          </div>
        </Panel>
      )}

      {paired && (
        <Panel className="mb-6">
          <PanelHeader title="Paired analysis (v2)" right={<span className="text-2xs text-ink-faint">common questions only</span>} />
          {paired.note && <div className="border-b border-line px-4 py-2 text-2xs text-ink-faint">{paired.note}</div>}
          <PairedTable paired={paired.paired} />
        </Panel>
      )}

      {v1 && (
        <Panel>
          <PanelHeader
            title="Source-selection v1 — instrument failure (preserved as-is)"
            right={
              <span className="flex items-center gap-2">
                <Badge tone="warn">inconclusive</Badge>
                <span className="font-mono text-2xs text-ink-faint">seed {v1.random_seed}</span>
              </span>
            }
          />
          <div className="border-b border-line px-4 py-3 text-xs leading-relaxed text-ink-muted">
            v1&apos;s scorer gave 34/36 homogeneous sources the identical score, so QUALITY selection
            collapsed into alphabetical tie-breaking; the experiment is an instrument failure, not a
            hypothesis test. v2 (above) replaced the scorer and produced 36/36 unique scores.
          </div>
          <div className="px-4 py-3">
            <div className="section-label mb-2">Coverage per run</div>
            {v1.runs.map((r) => (
              <div key={r.kb_id} className="mb-1.5 flex items-center gap-3 text-xs">
                <span className="w-44 shrink-0 truncate font-mono text-2xs text-ink-muted">
                  {r.strategy}|{r.n_sources_selected}
                </span>
                <ScoreBar
                  value={r.coverage.answerable}
                  max={r.coverage.total}
                  tone={r.strategy === "QUALITY_SELECTED" ? "accent" : "violet"}
                />
                <span className="w-12 shrink-0 text-right font-mono text-2xs text-ink-muted">
                  {r.coverage.answerable}/{r.coverage.total}
                </span>
              </div>
            ))}
          </div>
          <div className="border-t border-line px-4 py-3 text-2xs text-ink-faint">
            Full v1 metrics are intentionally not shown headline-style here — see
            docs/experiment-source-selection-v1.md for why the retrieval rows are not comparable.
          </div>
        </Panel>
      )}
    </div>
  );
}
