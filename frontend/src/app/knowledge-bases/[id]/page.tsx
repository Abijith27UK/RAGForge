"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { Database, FileText, FlaskConical, Search, Sigma } from "lucide-react";
import { api, DomainSpec, KBOverview } from "@/lib/api";
import {
  Badge, Button, EmptyState, Metric, Panel, PanelHeader, Skeleton, StatusPill,
} from "@/components/ui";
import PipelineGraph, { PipelineCounts } from "@/components/PipelineGraph";
import { fmtDate, fmtPct, fmtScore } from "@/lib/utils";

const SOURCE_MODE_LABEL: Record<string, string> = {
  external: "External",
  user_provided: "User provided",
  mixed: "Mixed",
};

const EVAL_LABEL: Record<string, string> = {
  not_configured: "Not configured",
  questions_only: "Questions only",
  benchmark_draft: "Benchmark (draft)",
  benchmark_frozen: "Benchmark (frozen)",
  evaluated: "Evaluated",
};

const EVAL_HINT: Record<string, string> = {
  not_configured:
    "Evaluation is optional. This knowledge base is usable and READY without ground truth.",
  questions_only: "Questions exist but no benchmark version has been snapshotted yet.",
  benchmark_draft: "A benchmark snapshot exists but is not frozen; runs are diagnostic only.",
  benchmark_frozen: "A human-reviewed, frozen benchmark is available for official runs.",
  evaluated: "At least one evaluation run has been recorded against this knowledge base.",
};

export default function KBWorkspace() {
  const { id: kbId } = useParams<{ id: string }>();
  const [overview, setOverview] = useState<KBOverview | null>(null);
  const [spec, setSpec] = useState<DomainSpec | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.kbOverview(kbId).then(setOverview).catch((e) => setError(String(e.message ?? e)));
    api.getDomainSpec(kbId).then(setSpec).catch(() => setSpec(null));
  }, [kbId]);

  if (error) {
    return (
      <EmptyState
        title="Could not load this knowledge base"
        hint={String(error)}
        action={<Link href="/"><Button variant="outline" size="sm">Back to overview</Button></Link>}
      />
    );
  }
  if (!overview) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-72" />
        <Skeleton className="h-28 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  const kb = overview.kb;
  const counts: PipelineCounts = {
    documents: overview.documents,
    chunks: overview.chunks,
    sourcesAccepted: overview.sources,
    questions: overview.evaluation_questions,
    hasSpec: !!spec,
    evals: overview.evaluation_runs,
    hasBuildRun: !!overview.last_build_at,
  };
  const latest = overview.last_evaluation;
  const vectorNote =
    overview.vectors == null
      ? `vector store ${overview.vector_store_status}`
      : `${overview.vectors} vectors`;

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="truncate text-xl font-semibold tracking-tight text-ink">{kb.name}</h1>
            <StatusPill status={kb.status} />
            <Badge tone={kb.source_mode === "external" ? "accent" : "violet"}>
              {SOURCE_MODE_LABEL[kb.source_mode ?? "external"] ?? kb.source_mode}
            </Badge>
            <Badge tone="neutral">v{overview.version}</Badge>
          </div>
          <p className="mt-1 text-xs text-ink-muted">{kb.domain} · {kb.purpose}</p>
          <p className="mt-0.5 text-2xs text-ink-faint">
            audience: {kb.target_audience} · depth: {kb.depth} · created {fmtDate(kb.created_at)}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          <Link href={`/knowledge-bases/${kbId}/documents`}>
            <Button variant="outline" size="sm"><FileText className="h-3.5 w-3.5" /> Documents</Button>
          </Link>
          <Link href={`/knowledge-bases/${kbId}/retrieval`}>
            <Button variant="outline" size="sm"><Search className="h-3.5 w-3.5" /> Retrieval Lab</Button>
          </Link>
          <Link href={`/knowledge-bases/${kbId}/evaluation`}>
            <Button variant="outline" size="sm"><Sigma className="h-3.5 w-3.5" /> Evaluate</Button>
          </Link>
          <Link href={`/knowledge-bases/${kbId}/processing`}>
            <Button variant="primary" size="sm"><FlaskConical className="h-3.5 w-3.5" /> Build</Button>
          </Link>
        </div>
      </div>

      {/* Pipeline */}
      <Panel className="mb-4">
        <PanelHeader title="Pipeline" right={<Badge tone="accent">live status</Badge>} />
        <div className="p-4">
          <PipelineGraph counts={counts} kbId={kbId} />
        </div>
      </Panel>

      {/* Knowledge base facts */}
      <div className="mb-4 grid gap-4 lg:grid-cols-2">
        <Panel>
          <PanelHeader title="Knowledge base" right={<StatusPill status={overview.build_status} />} />
          <div className="grid grid-cols-3 gap-4 p-4">
            <Metric label="Documents" value={overview.documents} />
            <Metric label="Chunks" value={overview.chunks} />
            <Metric
              label="Vectors"
              value={overview.vectors ?? "—"}
              hint={overview.vectors == null ? "unavailable" : undefined}
            />
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-line px-4 py-3 text-2xs">
            {[
              ["Source mode", SOURCE_MODE_LABEL[kb.source_mode ?? "external"] ?? kb.source_mode],
              ["Embedding", overview.embedding_model || "—"],
              ["Vector store", `${overview.vector_backend} · ${overview.vector_store_status}`],
              ["Chunking", overview.chunking_strategy || "—"],
              ["Last build", overview.last_build_at ? fmtDate(overview.last_build_at) : "never"],
              ["Version", `v${overview.version}`],
            ].map(([label, value]) => (
              <div key={label}>
                <dt className="section-label">{label}</dt>
                <dd className="truncate text-ink-muted" title={value}>{value}</dd>
              </div>
            ))}
          </dl>
          {(overview.user_provided_documents > 0 || overview.external_documents > 0) && (
            <div className="flex flex-wrap gap-2 border-t border-line px-4 py-2.5 text-2xs">
              {overview.user_provided_documents > 0 && (
                <Badge tone="violet">{overview.user_provided_documents} user provided</Badge>
              )}
              {overview.external_documents > 0 && (
                <Badge tone="neutral">{overview.external_documents} externally discovered</Badge>
              )}
              {overview.documents_failed > 0 && (
                <Badge tone="bad">{overview.documents_failed} failed</Badge>
              )}
              <span className="ml-auto text-ink-faint">{vectorNote}</span>
            </div>
          )}
        </Panel>

        <Panel>
          <PanelHeader
            title="Evaluation"
            right={
              latest ? (
                <Link href={`/knowledge-bases/${kbId}/evaluation`} className="text-2xs text-accent-soft hover:underline">
                  open →
                </Link>
              ) : undefined
            }
          />
          <div className="border-b border-line px-4 py-2.5">
            <StatusPill status={overview.evaluation_status} />
            <p className="mt-1 text-2xs leading-4 text-ink-faint">
              {EVAL_HINT[overview.evaluation_status] ?? ""}
            </p>
            {overview.evaluation_required && (
              <p className="mt-1 text-2xs text-warn">Ground truth is required — this should never happen.</p>
            )}
          </div>
          {latest ? (
            <div className="grid grid-cols-4 gap-4 p-4">
              <Metric label={`R@${latest.top_k}`} value={fmtPct(latest.recall_at_k)} tone="accent" />
              <Metric label="MRR" value={fmtScore(latest.mrr)} />
              <Metric label="NDCG" value={fmtScore(latest.ndcg)} />
              <Metric label="n" value={String(latest.questions_evaluated)} />
            </div>
          ) : (
            <div className="p-4">
              <p className="text-xs text-ink-faint">
                No metrics recorded. Add an optional evaluation dataset on the Evaluation page when you
                want to measure retrieval quality — it is never required to build or use this KB.
              </p>
            </div>
          )}
          <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-line px-4 py-2.5 text-2xs text-ink-faint">
            <span>questions: <span className="data-value text-ink-muted">{overview.evaluation_questions}</span></span>
            <span>benchmarks: <span className="data-value text-ink-muted">{overview.benchmark_versions}</span></span>
            <span>frozen: <span className="data-value text-ink-muted">{overview.frozen_benchmark_versions}</span></span>
            <span>runs: <span className="data-value text-ink-muted">{overview.evaluation_runs}</span></span>
          </div>
        </Panel>
      </div>

      {/* Domain summary */}
      {spec && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <Panel>
            <PanelHeader
              title="Domain specification"
              right={
                <span className="flex items-center gap-2">
                  {spec.is_mock && <Badge tone="warn">dev mock</Badge>}
                  <Link href={`/knowledge-bases/${kbId}/domain`} className="text-2xs text-accent-soft hover:underline">open →</Link>
                </span>
              }
            />
            <div className="p-4">
              <p className="text-sm leading-5 text-ink-muted">{spec.description}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {spec.subdomains.slice(0, 8).map((s) => (
                  <Badge key={s} tone="violet">{s}</Badge>
                ))}
              </div>
            </div>
          </Panel>
        </motion.div>
      )}

      {!spec && (
        <Panel>
          <PanelHeader
            title="Domain specification"
            right={<Database className="h-3.5 w-3.5 text-ink-ghost" />}
          />
          <div className="p-4">
            <p className="text-xs text-ink-faint">
              No domain spec yet. It is only needed for external source discovery — a knowledge base
              built from your own documents works without it.
            </p>
            <Link href={`/knowledge-bases/${kbId}/domain`} className="mt-2 inline-block text-2xs text-accent-soft hover:underline">
              run domain analysis →
            </Link>
          </div>
        </Panel>
      )}
    </div>
  );
}