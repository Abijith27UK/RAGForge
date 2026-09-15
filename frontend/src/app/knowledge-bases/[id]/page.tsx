"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { FlaskConical, Search, Sigma } from "lucide-react";
import {
  api, BuildRun, Chunk, Document, DomainSpec, EvaluationQuestion, EvaluationRun, KnowledgeBase, Source,
} from "@/lib/api";
import {
  Badge, Button, EmptyState, Metric, Panel, PanelHeader, Skeleton, StatusPill,
} from "@/components/ui";
import PipelineGraph, { PipelineCounts } from "@/components/PipelineGraph";
import { fmtDate, fmtPct, fmtScore } from "@/lib/utils";

const CHUNK_API_LIMIT = 500;

export default function KBWorkspace() {
  const { id: kbId } = useParams<{ id: string }>();
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [spec, setSpec] = useState<DomainSpec | null>(null);
  const [sources, setSources] = useState<Source[] | null>(null);
  const [docs, setDocs] = useState<Document[] | null>(null);
  const [chunks, setChunks] = useState<Chunk[] | null>(null);
  const [questions, setQuestions] = useState<EvaluationQuestion[] | null>(null);
  const [evals, setEvals] = useState<EvaluationRun[] | null>(null);
  const [buildRun, setBuildRun] = useState<BuildRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getKB(kbId),
      api.getDomainSpec(kbId).catch(() => null),
      api.listSources(kbId).catch(() => null),
      api.listDocuments(kbId).catch(() => null),
      api.listChunks(kbId, CHUNK_API_LIMIT).catch(() => null),
      api.listQuestions(kbId).catch(() => null),
      api.listEvaluationRuns(kbId).catch(() => null),
      api.buildStatus(kbId).catch(() => null),
    ])
      .then(([kb, spec, sources, docs, chunks, questions, evals, run]) => {
        setKb(kb); setSpec(spec); setSources(sources); setDocs(docs);
        setChunks(chunks); setQuestions(questions); setEvals(evals); setBuildRun(run);
      })
      .catch((e) => setError(String(e.message ?? e)));
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
  if (!kb) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-72" />
        <Skeleton className="h-28 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  const counts: PipelineCounts = {
    documents: docs ? docs.length : null,
    chunks: chunks ? chunks.length : null,
    sourcesAccepted: sources ? sources.filter((s) => s.decision === "ACCEPT").length : null,
    questions: questions ? questions.length : null,
    hasSpec: !!spec,
    evals: evals ? evals.length : null,
    hasBuildRun: !!buildRun,
  };

  const latest = evals && evals.length > 0 ? evals[0] : null;
  const corpusChars = docs ? docs.reduce((s, d) => s + (d.text_length ?? 0), 0) : null;

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2.5">
            <h1 className="truncate text-xl font-semibold tracking-tight text-ink">{kb.name}</h1>
            <StatusPill status={kb.status} />
          </div>
          <p className="mt-1 text-xs text-ink-muted">
            {kb.domain} · {kb.purpose}
          </p>
          <p className="mt-0.5 text-2xs text-ink-faint">
            audience: {kb.target_audience} · depth: {kb.depth} · created {fmtDate(kb.created_at)}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
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
      <Panel className="mb-6">
        <PanelHeader title="Pipeline" right={<Badge tone="accent">live status</Badge>} />
        <div className="p-4">
          <PipelineGraph counts={counts} kbId={kbId} />
        </div>
      </Panel>

      {/* Corpus + evaluation summary */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel>
          <PanelHeader title="Corpus" />
          <div className="grid grid-cols-3 gap-4 p-4">
            <Metric label="Documents" value={docs === null ? <Skeleton className="h-5 w-8" /> : docs.length} />
            <Metric label="Chunks" value={chunks === null ? <Skeleton className="h-5 w-10" /> : chunks.length >= CHUNK_API_LIMIT ? `${CHUNK_API_LIMIT}+` : chunks.length} />
            <Metric label="Characters" value={corpusChars === null ? <Skeleton className="h-5 w-12" /> : `${(corpusChars / 1000).toFixed(0)}k`} />
          </div>
          {buildRun && (
            <div className="border-t border-line px-4 py-2.5 text-2xs text-ink-faint">
              last build: <StatusPill status={buildRun.status} className="inline-flex align-middle" />
              <span className="ml-2">{buildRun.stages.length} stages · started {fmtDate(buildRun.started_at)}</span>
            </div>
          )}
        </Panel>

        <Panel>
          <PanelHeader
            title="Latest evaluation"
            right={latest ? <Link href={`/knowledge-bases/${kbId}/evaluation`} className="text-2xs text-accent-soft hover:underline">open →</Link> : undefined}
          />
          {evals === null ? (
            <div className="p-4"><Skeleton className="h-16 w-full" /></div>
          ) : latest ? (
            <div className="grid grid-cols-4 gap-4 p-4">
              <Metric label={`R@${latest.config?.top_k ?? "?"}`} value={fmtPct(latest.aggregate?.recall_at_k)} tone="accent" />
              <Metric label="MRR" value={fmtScore(latest.aggregate?.mrr)} />
              <Metric label="NDCG" value={fmtScore(latest.aggregate?.ndcg)} />
              <Metric label="n" value={String(latest.aggregate?.questions_evaluated ?? "—")} hint={latest.aggregate?.strict_mode ? "strict" : "diagnostic"} />
            </div>
          ) : (
            <div className="p-4">
              <p className="text-xs text-ink-faint">
                No evaluation runs yet. Author ground-truth questions, then run a strict evaluation.
              </p>
            </div>
          )}
          {latest && latest.aggregate?.notes && (
            <div className="border-t border-line px-4 py-2 text-2xs leading-4 text-warn">
              {latest.aggregate.notes}
            </div>
          )}
        </Panel>
      </div>

      {/* Domain summary */}
      {spec && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="mt-4">
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
    </div>
  );
}
