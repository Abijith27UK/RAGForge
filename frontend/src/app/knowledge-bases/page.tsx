"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowUpRight, Database, FileText, ListTree, Plus, Search, Sigma, Trash2 } from "lucide-react";
import { api, EvaluationRun, KnowledgeBase } from "@/lib/api";
import { Button, EmptyState, Panel, Skeleton, Sparkline, StatusPill } from "@/components/ui";
import { fmtDate, fmtPct, fmtScore } from "@/lib/utils";

const CHUNK_API_LIMIT = 500;

type KBRow = {
  kb: KnowledgeBase;
  docs: number | null;
  chunks: number | null;
  chars: number | null;
  evals: EvaluationRun[] | null;
};

export default function KnowledgeBasesIndex() {
  const [kbs, setKbs] = useState<KnowledgeBase[] | null>(null);
  const [rows, setRows] = useState<Record<string, KBRow>>({});
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listKBs().then(setKbs).catch((e) => { setError(String(e.message ?? e)); setKbs([]); });
  }, []);

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
              kb, docs: docs ? docs.length : null, chunks: chunks ? chunks.length : null,
              chars: docs ? docs.reduce((s, d) => s + (d.text_length ?? 0), 0) : null, evals,
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

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6 flex items-end justify-between">
        <div>
          <div className="section-label mb-2">Workspace</div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Knowledge Bases</h1>
        </div>
        <Link href="/knowledge-bases/new">
          <Button variant="primary" size="sm"><Plus className="h-3.5 w-3.5" /> New</Button>
        </Link>
      </div>

      {error && (
        <div className="mb-6 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">Backend unreachable: {error}</div>
      )}

      {!kbs && (
        <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16 w-full" />)}</div>
      )}

      {kbs && kbs.length === 0 && (
        <EmptyState
          icon={<Database className="h-8 w-8" />}
          title="No knowledge bases yet"
          hint="Create a knowledge base to start the build pipeline."
          action={<Link href="/knowledge-bases/new" className="mt-2"><Button variant="primary" size="sm">Create</Button></Link>}
        />
      )}

      {kbs && kbs.length > 0 && (
        <Panel className="divide-y divide-line">
          {kbs.map((kb, i) => {
            const row = rows[kb.id] ?? { kb, docs: null, chunks: null, chars: null, evals: null };
            const latest = row.evals && row.evals.length > 0 ? row.evals[0] : null;
            const ndcgSeries = (row.evals ?? []).map((r) => r.aggregate?.ndcg).filter((v): v is number => typeof v === "number").slice(0, 8).reverse();
            return (
              <motion.div
                key={kb.id}
                initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: Math.min(i * 0.04, 0.2) }}
                className="group relative flex flex-col gap-3 px-4 py-3 transition-colors hover:bg-surface-2 md:flex-row md:items-center md:gap-6"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Link href={`/knowledge-bases/${kb.id}`} className="truncate text-sm font-medium text-ink hover:text-accent-soft">{kb.name}</Link>
                    <StatusPill status={kb.status} />
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-2xs text-ink-faint">
                    <span className="text-ink-muted">{kb.domain}</span>
                    <span>created {fmtDate(kb.created_at)}</span>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-5 text-xs text-ink-muted">
                  <span className="flex items-center gap-1.5" title="documents"><FileText className="h-3.5 w-3.5 text-ink-ghost" />{row.docs ?? "…"}</span>
                  <span className="flex items-center gap-1.5" title="chunks"><ListTree className="h-3.5 w-3.5 text-ink-ghost" />{row.chunks == null ? "…" : row.chunks >= CHUNK_API_LIMIT ? `${CHUNK_API_LIMIT}+` : row.chunks}</span>
                  {ndcgSeries.length >= 2 ? (
                    <Sparkline values={ndcgSeries} tone="#A78BFA" />
                  ) : latest ? (
                    <span className="data-value text-2xs text-ink-faint">NDCG {fmtScore(latest.aggregate?.ndcg)}</span>
                  ) : (
                    <span className="text-2xs text-ink-ghost">not evaluated</span>
                  )}
                  {latest && <span className="hidden data-value text-2xs text-ink-faint xl:block">R@{latest.config?.top_k ?? "?"} {fmtPct(latest.aggregate?.recall_at_k)}</span>}
                </div>
                <div className="absolute right-3 top-2 hidden items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 md:flex">
                  <Link href={`/knowledge-bases/${kb.id}/retrieval`} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-accent-soft" title="Retrieval Lab"><Search className="h-3.5 w-3.5" /></Link>
                  <Link href={`/knowledge-bases/${kb.id}/evaluation`} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-accent-soft" title="Evaluation"><Sigma className="h-3.5 w-3.5" /></Link>
                  <Link href={`/knowledge-bases/${kb.id}`} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-ink" title="Open"><ArrowUpRight className="h-3.5 w-3.5" /></Link>
                  {confirming === kb.id ? (
                    <span className="flex items-center gap-1 text-2xs">
                      <button onClick={() => del(kb.id)} className="rounded border border-bad/40 px-1.5 py-0.5 text-bad hover:bg-bad/10">confirm</button>
                      <button onClick={() => setConfirming(null)} className="rounded border border-line-strong px-1.5 py-0.5 text-ink-faint hover:text-ink">cancel</button>
                    </span>
                  ) : (
                    <button onClick={() => setConfirming(kb.id)} className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-bad" title="Delete"><Trash2 className="h-3.5 w-3.5" /></button>
                  )}
                </div>
              </motion.div>
            );
          })}
        </Panel>
      )}
    </div>
  );
}
