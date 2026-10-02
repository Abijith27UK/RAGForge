"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { ListTree } from "lucide-react";
import { api, Chunk } from "@/lib/api";
import { Badge, CopyButton, EmptyState, Panel, PanelHeader, Skeleton } from "@/components/ui";
import { cn, fmtScore } from "@/lib/utils";

const PAGE_SIZE = 60;

export default function ChunksPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [chunks, setChunks] = useState<Chunk[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [docFilter, setDocFilter] = useState("all");
  const [sectionFilter, setSectionFilter] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [visible, setVisible] = useState(PAGE_SIZE);

  useEffect(() => {
    // Fetch generously but stay within the API cap; counts show the cap honestly.
    api.listChunks(kbId, 500)
      .then((cs) => { setChunks(cs); if (cs.length > 0) setSelectedId(cs[0].id); })
      .catch((e) => { setError(String(e.message)); setChunks([]); });
  }, [kbId]);

  const docs = useMemo(() => {
    const map = new Map<string, string>();
    (chunks ?? []).forEach((c) => map.set(c.document_id, c.document_title ?? c.document_id));
    return [...map.entries()];
  }, [chunks]);

  const sections = useMemo(() => {
    const set = new Set<string>();
    (chunks ?? []).forEach((c) => { if (c.section) set.add(c.section); });
    return [...set].sort();
  }, [chunks]);

  const filtered = useMemo(() => {
    const needle = q.toLowerCase().trim();
    return (chunks ?? []).filter((c) => {
      if (docFilter !== "all" && c.document_id !== docFilter) return false;
      if (sectionFilter !== "all" && c.section !== sectionFilter) return false;
      if (needle && !(c.text.toLowerCase().includes(needle) || (c.section ?? "").toLowerCase().includes(needle))) return false;
      return true;
    });
  }, [chunks, q, docFilter, sectionFilter]);

  const selected = filtered.find((c) => c.id === selectedId) ?? filtered[0] ?? null;

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-4">
        <div className="section-label mb-2">Research</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Chunk Inspector</h1>
        <p className="mt-1 text-xs text-ink-faint">
          Every chunk carries full provenance: source, document, section, trust score, content hash.
        </p>
      </div>

      {error && <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>}

      {!chunks && <Skeleton className="h-96 w-full" />}

      {chunks && chunks.length === 0 && (
        <EmptyState icon={<ListTree className="h-8 w-8" />} title="No chunks yet"
          hint="Run ingestion + indexing first — chunks appear here with their provenance." />
      )}

      {chunks && chunks.length > 0 && (
        <div className="grid gap-3 lg:grid-cols-[220px_minmax(0,1fr)_320px]">
          {/* Filters */}
          <div className="space-y-3">
            <Panel>
              <PanelHeader title="Filters" />
              <div className="space-y-3 p-3">
                <input
                  value={q} onChange={(e) => { setQ(e.target.value); setVisible(PAGE_SIZE); }}
                  placeholder="Search text…"
                  className="h-8 w-full rounded border border-line-strong bg-surface-3 px-2.5 text-xs text-ink placeholder:text-ink-ghost focus:border-line-focus focus:outline-none"
                />
                <div>
                  <div className="section-label mb-1">Document</div>
                  <select
                    value={docFilter} onChange={(e) => { setDocFilter(e.target.value); setVisible(PAGE_SIZE); }}
                    className="h-8 w-full rounded border border-line-strong bg-surface-3 px-2 text-xs text-ink"
                  >
                    <option value="all">All documents</option>
                    {docs.map(([id, title]) => <option key={id} value={id}>{title}</option>)}
                  </select>
                </div>
                <div>
                  <div className="section-label mb-1">Section</div>
                  <select
                    value={sectionFilter} onChange={(e) => { setSectionFilter(e.target.value); setVisible(PAGE_SIZE); }}
                    className="h-8 w-full rounded border border-line-strong bg-surface-3 px-2 text-xs text-ink"
                  >
                    <option value="all">All sections</option>
                    {sections.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div className="text-2xs text-ink-faint">
                  {filtered.length} of {chunks.length >= 500 ? "500+" : chunks.length} chunks shown
                </div>
              </div>
            </Panel>
          </div>

          {/* List */}
          <Panel className="flex max-h-[calc(100vh-190px)] flex-col">
            <PanelHeader title="Chunks" right={<span className="data-value text-2xs text-ink-faint">{filtered.length}</span>} />
            <div className="min-h-0 flex-1 divide-y divide-line/50 overflow-y-auto">
              {filtered.slice(0, visible).map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedId(c.id)}
                  className={cn(
                    "block w-full px-4 py-2.5 text-left transition-colors",
                    selected?.id === c.id ? "bg-accent/10" : "hover:bg-surface-3"
                  )}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className={cn("truncate text-xs", selected?.id === c.id ? "text-accent-soft" : "text-ink")}>
                      {c.section && <span className="text-violet">[{c.section}] </span>}
                      {c.text.slice(0, 96)}
                    </span>
                    <span className="data-value shrink-0 text-2xs text-ink-ghost">#{c.chunk_index} · {c.char_count}ch</span>
                  </div>
                  <span className="mt-0.5 block truncate text-2xs text-ink-faint">{c.document_title}</span>
                </button>
              ))}
              {filtered.length > visible && (
                <button
                  onClick={() => setVisible((v) => v + PAGE_SIZE)}
                  className="w-full px-4 py-3 text-center text-2xs text-accent-soft hover:bg-surface-3"
                >
                  Show more ({filtered.length - visible} remaining)
                </button>
              )}
              {filtered.length === 0 && (
                <p className="px-4 py-8 text-center text-xs text-ink-faint">No chunks match the filters.</p>
              )}
            </div>
          </Panel>

          {/* Inspector */}
          <Panel className="flex max-h-[calc(100vh-190px)] flex-col">
            {selected ? (
              <>
                <PanelHeader title="Inspector" right={<CopyButton text={selected.id} />} />
                <div className="min-h-0 flex-1 overflow-y-auto p-4">
                  <div className="mb-3 flex flex-wrap items-center gap-1.5">
                    {selected.section && <Badge tone="violet">{selected.section}</Badge>}
                    {selected.page != null && <Badge tone="neutral">page {selected.page}</Badge>}
                    <Badge tone="neutral">#{selected.chunk_index}</Badge>
                  </div>
                  <p className="whitespace-pre-wrap rounded border border-line bg-surface-1 p-3 text-xs leading-5 text-ink">
                    {selected.text}
                  </p>
                  <div className="mt-4 space-y-1.5 text-2xs">
                    <ProvRow k="chunk id" v={selected.id} copy />
                    <ProvRow k="document" v={selected.document_title ?? selected.document_id} />
                    <ProvRow k="section path" v={selected.section_path} />
                    <ProvRow k="source url" v={selected.source_url} />
                    <ProvRow k="source type" v={selected.source_type} />
                    <ProvRow k="publisher" v={selected.publisher} />
                    <ProvRow k="page" v={selected.page != null ? String(selected.page) : null} />
                    <ProvRow k="domain" v={selected.domain} />
                    <ProvRow k="subdomain" v={selected.subdomain} />
                    <ProvRow k="trust score" v={selected.trust_score != null ? fmtScore(selected.trust_score) : null} />
                    <ProvRow k="content hash" v={selected.content_hash} copy />
                    <ProvRow k="char count" v={String(selected.char_count)} />
                  </div>
                </div>
              </>
            ) : (
              <div className="p-4 text-xs text-ink-faint">Select a chunk to inspect its provenance.</div>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}

function ProvRow({ k, v, copy }: { k: string; v: string | null | undefined; copy?: boolean }) {
  return (
    <div className="flex items-baseline gap-2 min-w-0">
      <span className="section-label w-24 shrink-0">{k}</span>
      <span className="data-value min-w-0 flex-1 break-all text-ink-muted">{v ?? "null"}</span>
      {copy && v && <CopyButton text={v} className="h-4 w-4" />}
    </div>
  );
}
