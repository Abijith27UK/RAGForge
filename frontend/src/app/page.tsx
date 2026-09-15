"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowUpRight, Database, FileText, ListTree, Plus, Search, Sigma, Trash2,
} from "lucide-react";
import { api, EvaluationRun, KnowledgeBase, SystemStatus } from "@/lib/api";
import {
  Button, EmptyState, Panel, Skeleton, Sparkline, StatusPill,
} from "@/components/ui";
import { cn, fmtDate, fmtPct, fmtScore } from "@/lib/utils";

type KBRow = {
  kb: KnowledgeBase;
  docs: number | null;
  chunks: number | null; // capped at the API limit; >=500 renders as "500+"
  chars: number | null;
  evals: EvaluationRun[] | null;
};

const CHUNK_API_LIMIT = 500;

export default function Dashboard() {
  const [kbs, setKbs] = useState<KnowledgeBase[] | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, KBRow>>({});
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listKBs(), api.systemStatus().catch(() => null)])
      .then(([kbList, sys]) => { setKbs(kbList); setStatus(sys); })
      .catch((e) => { setError(String(e.message ?? e)); setKbs([]); });
  }, []);

  // Per-KB real counts + latest evaluation, fetched lazily from the real API.
  useEffect(() => {
    if (!kbs) return;
    let cancelled = false;
    (async () => {
      await Promise.all(
        kbs.map(async (kb) => {
          const [docs, chunks, evals] = await Promise.all([
            api.listDocuments(kb.id).catch(() => null),
            api.listChunks(kb.id, CHUNK_API_LIMIT).catch(() => null),
            api.listEvaluationRuns(kb.id).catch(() => null),
          ]);
          if (cancelled) return;
          setRows((prev) => ({
            ...prev,
            [kb.id]: {
              kb,
              docs: docs ? docs.length : null,
              chunks: chunks ? chunks.length : null,
              chars: docs ? docs.reduce((s, d) => s + (d.text_length ?? 0), 0) : null,
              evals,
            },
          }));
        })
      );
    })();
    return () => { cancelled = true; };
  }, [kbs]);

  async function del(id: string) {
    await api.deleteKB(id);
    setKbs((prev) => (prev ? prev.filter((k) => k.id !== id) : prev));
    setConfirming(null);
  }

  const health = [
    { label: "Qdrant", ok: status ? status.qdrant.reachable : null, value: status ? (status.qdrant.reachable ? "Connected" : "Unreachable") : "…" },
    { label: "Embeddings", ok: true, value: status?.embedding.model ?? "…" },
    {
      label: "LLM",
      ok: status ? !(status.llm.allow_mock && status.llm.provider.startsWith("not configured")) : null,
      value: status ? (status.llm.provider.startsWith("not configured") ? "Dev mock" : status.llm.provider) : "…",
    },
    { label: "Pipeline", ok: status ? status.qdrant.reachable : null, value: status ? (status.qdrant.reachable ? "Ready" : "Degraded") : "…" },
  ];

  const ordered = useMemo(() => {
    if (!kbs) return [];
    return kbs.map((kb) => rows[kb.id] ?? { kb, docs: null, chunks: null, chars: null, evals: null });
  }, [kbs, rows]);

  return (
    <div className="mx-auto max-w-6xl">
      {/* Masthead */}
      <div className="mb-8">
        <div className="section-label mb-2">Knowledge Engineering Workspace</div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">RAGForge</h1>
        <p className="mt-1.5 max-w-2xl text-sm leading-5 text-ink-muted">
          Build domain-specific RAG knowledge bases with source intelligence, provenance-rich
          chunks, and real retrieval evaluation — every number on this page comes from executed
          pipeline runs.
        </p>
      </div>

      {error && (
        <div className="mb-6 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
          Backend unreachable: {error}. Is the API running on port 8000?
        </div>
      )}

      {/* System health strip */}
      <div className="mb-8 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-line bg-surface-1 px-4 py-2.5">
        <span className="section-label">System health</span>
        {health.map((h) => (
          <span key={h.label} className="flex items-center gap-1.5 text-xs">
            <HealthDot ok={h.ok} />
            <span className="text-ink-faint">{h.label}</span>
            <span className="data-value text-ink-muted">{h.value}</span>
          </span>
        ))}
      </div>

      {/* Knowledge bases */}
      <div className="mb-3 flex items-center justify-between">
        <span className="section-label">Knowledge bases</span>
        <Link href="/knowledge-bases/new">
          <Button variant="primary" size="sm"><Plus className="h-3.5 w-3.5" /> New knowledge base</Button>
        </Link>
      </div>

      {!kbs && (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
        </div>
      )}

      {kbs && ordered.length === 0 && (
        <EmptyState
          icon={<Database className="h-8 w-8" />}
          title="No knowledge bases yet"
          hint="Create one to start the pipeline: domain analysis → source discovery → ingestion → chunking → embeddings → evaluation."
          action={
            <Link href="/knowledge-bases/new" className="mt-2">
              <Button variant="primary" size="sm"><Plus className="h-3.5 w-3.5" /> Create knowledge base</Button>
            </Link>
          }
        />
      )}

      {kbs && ordered.length > 0 && (
        <Panel className="divide-y divide-line">
          {ordered.map((row, i) => (
            <KBRowItem
              key={row.kb.id}
              row={row}
              index={i}
              confirming={confirming === row.kb.id}
              onAskDelete={() => setConfirming(row.kb.id)}
              onCancelDelete={() => setConfirming(null)}
              onDelete={() => del(row.kb.id)}
            />
          ))}
        </Panel>
      )}
    </div>
  );
}

function HealthDot({ ok }: { ok: boolean | null }) {
  return (
    <span
      className={cn(
        "inline-block h-1.5 w-1.5 rounded-full",
        ok === null ? "bg-ink-ghost" : ok ? "bg-ok" : "bg-bad"
      )}
    />
  );
}

function KBRowItem({
  row, index, confirming, onAskDelete, onCancelDelete, onDelete,
}: {
  row: KBRow;
  index: number;
  confirming: boolean;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
}) {
  const { kb, docs, chunks, chars, evals } = row;
  const latest = evals && evals.length > 0 ? evals[0] : null;
  const ndcgSeries = (evals ?? [])
    .map((r) => r.aggregate?.ndcg)
    .filter((v): v is number => typeof v === "number")
    .slice(0, 8)
    .reverse();

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: Math.min(index * 0.04, 0.2) }}
      className="group relative flex flex-col gap-3 px-4 py-3 transition-colors hover:bg-surface-2 md:flex-row md:items-center md:gap-6"
    >
      {/* Identity */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Link href={`/knowledge-bases/${kb.id}`} className="truncate text-sm font-medium text-ink transition-colors hover:text-accent-soft">
            {kb.name}
          </Link>
          <StatusPill status={kb.status} />
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-2xs text-ink-faint">
          <span className="text-ink-muted">{kb.domain}</span>
          <span>depth: {kb.depth}</span>
          <span>created {fmtDate(kb.created_at)}</span>
        </div>
      </div>

      {/* Data facts */}
      <div className="flex shrink-0 items-center gap-5 text-xs">
        <span className="flex items-center gap-1.5 text-ink-muted" title="documents">
          <FileText className="h-3.5 w-3.5 text-ink-ghost" />
          {docs === null ? <Skeleton className="h-3 w-6" /> : docs}
        </span>
        <span className="flex items-center gap-1.5 text-ink-muted" title="chunks">
          <ListTree className="h-3.5 w-3.5 text-ink-ghost" />
          {chunks === null ? (
            <Skeleton className="h-3 w-8" />
          ) : chunks >= CHUNK_API_LIMIT ? (
            `${CHUNK_API_LIMIT}+`
          ) : (
            chunks
          )}
        </span>
        <span className="hidden data-value text-2xs text-ink-faint lg:block" title="ingested characters">
          {chars === null ? <Skeleton className="h-3 w-10" /> : `${(chars / 1000).toFixed(0)}k chars`}
        </span>
        {ndcgSeries.length >= 2 ? (
          <Sparkline values={ndcgSeries} tone="#A78BFA" />
        ) : latest ? (
          <span className="data-value text-2xs text-ink-faint">NDCG {fmtScore(latest.aggregate?.ndcg)}</span>
        ) : (
          <span className="text-2xs text-ink-ghost">not evaluated</span>
        )}
        {latest && (
          <span className="hidden data-value text-2xs text-ink-faint xl:block">
            R@{latest.config?.top_k ?? "?"} {fmtPct(latest.aggregate?.recall_at_k)}
          </span>
        )}
      </div>

      {/* Quick actions on hover */}
      <div className="absolute right-3 top-2 hidden items-center gap-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 md:flex">
        <Link href={`/knowledge-bases/${kb.id}/retrieval`} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-accent-soft" title="Retrieval Lab">
          <Search className="h-3.5 w-3.5" />
        </Link>
        <Link href={`/knowledge-bases/${kb.id}/evaluation`} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-accent-soft" title="Evaluation">
          <Sigma className="h-3.5 w-3.5" />
        </Link>
        <Link href={`/knowledge-bases/${kb.id}`} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-ink" title="Open workspace">
          <ArrowUpRight className="h-3.5 w-3.5" />
        </Link>
        {confirming ? (
          <span className="flex items-center gap-1 text-2xs">
            <button onClick={onDelete} className="rounded border border-bad/40 px-1.5 py-0.5 text-bad hover:bg-bad/10">confirm</button>
            <button onClick={onCancelDelete} className="rounded border border-line-strong px-1.5 py-0.5 text-ink-faint hover:text-ink">cancel</button>
          </span>
        ) : (
          <button onClick={onAskDelete} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-bad" title="Delete KB">
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </motion.div>
  );
}
