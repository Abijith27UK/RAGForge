"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  Database, FileText, Layers, RefreshCw, Trash2, Upload, X,
} from "lucide-react";
import {
  api, Chunk, Document, DocumentLibrary, DocumentText, UploadResult,
} from "@/lib/api";
import {
  Badge, Button, EmptyState, Metric, Panel, PanelHeader, Progress, Skeleton, StatusPill,
} from "@/components/ui";
import { fmtCount, fmtDate } from "@/lib/utils";

const ACCEPTED = ".pdf,.pptx,.ppt,.docx,.txt,.md,.markdown,.html,.htm";

function fmtBytes(bytes: number): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** pages/slides/sections are never invented: show whichever is real. */
function units(doc: Document): { label: string; value: number | null } {
  if (doc.page_count != null) return { label: "pages", value: doc.page_count };
  if (doc.slide_count != null) return { label: "slides", value: doc.slide_count };
  if (doc.section_count != null) return { label: "sections", value: doc.section_count };
  return { label: "pages", value: null };
}

export default function DocumentsPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [library, setLibrary] = useState<DocumentLibrary | null>(null);
  const [docs, setDocs] = useState<Document[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastUpload, setLastUpload] = useState<UploadResult | null>(null);
  const [inspect, setInspect] = useState<{ doc: Document; text: DocumentText; chunks: Chunk[] } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Prefer the backend's honest label ("unknown" when the vector store could
  // not confirm a deletion count); fall back to the raw number for older runs.
  const staleLabel = (() => {
    const ix = lastUpload?.indexing;
    if (!ix) return "";
    if (ix.stale_vectors_removed_label) return ix.stale_vectors_removed_label;
    if (!ix.stale_vectors_removed) return "";
    return String(ix.stale_vectors_removed);
  })();

  const load = useCallback(() => {
    api.documentLibrary(kbId).then(setLibrary).catch((e) => setError(String(e.message ?? e)));
    api.listDocuments(kbId).then(setDocs).catch(() => setDocs([]));
  }, [kbId]);

  useEffect(load, [load]);

  async function upload(files: File[]) {
    if (!files.length) return;
    setBusy("upload");
    setError(null);
    setProgress(0);
    try {
      const result = await api.uploadDocuments(kbId, files, { index: true }, setProgress);
      setLastUpload(result);
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
      load();
    }
  }

  async function reindex(doc: Document) {
    setBusy(`index:${doc.id}`);
    setError(null);
    try {
      await api.indexDocument(kbId, doc.id);
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
      load();
    }
  }

  async function rebuild(doc: Document) {
    setBusy(`rebuild:${doc.id}`);
    setError(null);
    try {
      await api.rebuildDocument(kbId, doc.id, true);
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
      load();
    }
  }

  async function remove(doc: Document) {
    if (!window.confirm(`Remove "${doc.file_name ?? doc.title ?? doc.id}" and its vectors?`)) return;
    setBusy(`delete:${doc.id}`);
    setError(null);
    try {
      const outcome = await api.deleteDocument(kbId, doc.id);
      if (outcome.vector_store_error) {
        setError(
          `Removed from the library, but its vectors could not be deleted ` +
          `(${outcome.vector_store_error}). Re-index once Qdrant is reachable.`,
        );
      } else if (!outcome.vectors_removal_confirmed) {
        // Honest warning: the delete ran, but Qdrant never confirmed a count.
        setError(
          "Removed from the library and a deletion was issued, but the vector store did not " +
          "confirm how many vectors were removed. Re-index to be certain no orphans remain.",
        );
      }
      setInspect(null);
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
      load();
    }
  }

  async function openInspect(doc: Document) {
    setBusy(`inspect:${doc.id}`);
    try {
      const [text, chunks] = await Promise.all([
        api.documentText(kbId, doc.id, 6000).catch(() => null),
        api.documentChunks(kbId, doc.id).catch(() => [] as Chunk[]),
      ]);
      if (!text) {
        setError("This document has no extracted text (it failed to parse).");
        return;
      }
      setInspect({ doc, text, chunks });
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
    }
  }

  const sorted = useMemo(
    () =>
      [...(docs ?? [])].sort((a, b) =>
        String(b.ingestion_timestamp).localeCompare(String(a.ingestion_timestamp)),
      ),
    [docs],
  );

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6 flex items-end justify-between gap-4">
        <div>
          <div className="section-label mb-2">Knowledge</div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Document Library</h1>
          <p className="mt-1 text-xs text-ink-faint">
            Add, inspect, re-index or remove documents. Adding one document never re-embeds the rest.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()} loading={busy === "upload"}>
          <Upload className="h-3.5 w-3.5" /> Upload documents
        </Button>
      </div>

      {error && (
        <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>
      )}

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(Array.from(e.dataTransfer.files)); }}
        onClick={() => inputRef.current?.click()}
        className={`mb-4 cursor-pointer rounded-lg border border-dashed px-4 py-6 text-center transition-colors ${
          drag ? "border-accent bg-accent/5" : "border-line-strong hover:border-line-focus"
        }`}
      >
        <Upload className="mx-auto mb-1.5 h-5 w-5 text-ink-ghost" />
        <p className="text-xs text-ink-muted">Drop PDF / PPTX / DOCX / TXT / MD files here</p>
        <p className="mt-0.5 text-2xs text-ink-faint">
          {library ? `Up to ${fmtBytes(library.max_upload_bytes)} per file · ` : ""}
          supported: {library ? library.supported_extensions.join(" ") : "…"}
        </p>
        {busy === "upload" && <div className="mx-auto mt-3 max-w-sm"><Progress value={progress} /></div>}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => { upload(Array.from(e.target.files ?? [])); e.target.value = ""; }}
        />
      </div>

      {/* Upload report */}
      {lastUpload && (
        <Panel className="mb-4">
          <PanelHeader
            title="Last upload"
            right={<button className="text-2xs text-ink-faint hover:text-ink" onClick={() => setLastUpload(null)}>dismiss</button>}
          />
          <div className="space-y-2 p-4">
            <div className="flex flex-wrap gap-2">
              <Badge tone="ok">{lastUpload.uploaded} ingested</Badge>
              {lastUpload.duplicates > 0 && <Badge tone="warn">{lastUpload.duplicates} duplicate — existing kept</Badge>}
              {lastUpload.rejected > 0 && <Badge tone="bad">{lastUpload.rejected} rejected</Badge>}
              {lastUpload.failed > 0 && <Badge tone="bad">{lastUpload.failed} failed</Badge>}
              {lastUpload.indexing?.chunk_count != null && (
                <Badge tone="accent">
                  {lastUpload.indexing.chunk_count} chunks indexed
                  {staleLabel ? ` · ${staleLabel} stale removed` : ""}
                </Badge>
              )}
            </div>
            <ul className="space-y-1 text-2xs">
              {lastUpload.files.map((f) => (
                <li key={f.file_name} className="flex items-start gap-2">
                  <StatusPill status={f.status} />
                  <span className="min-w-0">
                    <span className="text-ink">{f.file_name}</span>
                    <span className="text-ink-faint"> · {fmtBytes(f.size_bytes)}{f.message ? ` — ${f.message}` : ""}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Panel>
      )}

      {/* Library summary */}
      <Panel className="mb-4">
        <PanelHeader title="Library" right={<span className="data-value text-2xs text-ink-faint">{docs ? `${docs.length} document(s)` : "…"}</span>} />
        {library === null ? (
          <div className="p-4"><Skeleton className="h-12 w-full" /></div>
        ) : (
          <div className="grid grid-cols-2 gap-4 p-4 sm:grid-cols-4">
            <Metric label="Documents" value={library.documents} />
            <Metric label="Chunks" value={library.chunks} />
            <Metric label="User provided" value={library.user_provided} tone="accent" />
            <Metric label="Externally discovered" value={library.external} />
          </div>
        )}
      </Panel>

      {/* Table */}
      <Panel>
        <PanelHeader title="Documents" />
        {!docs ? (
          <div className="p-4"><Skeleton className="h-40 w-full" /></div>
        ) : sorted.length === 0 ? (
          <div className="p-4">
            <EmptyState icon={<Database className="h-6 w-6" />} title="No documents yet"
              hint="Upload files here, or accept external sources on the Sources page and run ingestion." />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-line text-left">
                  {["Name", "Type", "Size", "Units", "Chunks", "Version", "Status", "Created", ""].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium text-ink-faint">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((d) => {
                  const u = units(d);
                  const working = busy === `index:${d.id}` || busy === `rebuild:${d.id}` || busy === `delete:${d.id}`;
                  return (
                    <tr key={d.id} className="border-b border-line/50 transition-colors last:border-0 hover:bg-surface-3">
                      <td className="max-w-[260px] px-3 py-2">
                        <div className="truncate text-ink">{d.file_name ?? d.title ?? d.url}</div>
                        <div className="flex items-center gap-1.5">
                          {d.user_provided ? (
                            <Badge tone="violet">user provided</Badge>
                          ) : (
                            <Badge tone="neutral">discovered</Badge>
                          )}
                          {d.parse_error && <span className="truncate text-2xs text-bad">{d.parse_error}</span>}
                        </div>
                      </td>
                      <td className="px-3 py-2"><Badge tone="neutral">{d.parser ?? d.source_type}</Badge></td>
                      <td className="data-value px-3 py-2 text-ink-muted">{fmtBytes(d.file_size ?? 0)}</td>
                      <td className="data-value px-3 py-2 text-ink-muted">
                        {u.value != null ? `${u.value} ${u.label}` : "—"}
                      </td>
                      <td className="data-value px-3 py-2 text-ink-muted">{d.chunk_count ?? 0}</td>
                      <td className="data-value px-3 py-2 text-ink-muted">v{d.document_version ?? 1}</td>
                      <td className="px-3 py-2"><StatusPill status={d.status ?? "parsed"} /></td>
                      <td className="px-3 py-2 text-2xs text-ink-faint">{fmtDate(d.ingestion_timestamp)}</td>
                      <td className="px-3 py-2">
                        <span className="flex justify-end gap-1">
                          <Button variant="ghost" size="sm" title="Inspect extracted text and chunks"
                            onClick={() => openInspect(d)} loading={busy === `inspect:${d.id}`}>
                            <FileText className="h-3 w-3" />
                          </Button>
                          <Button variant="ghost" size="sm" title="Re-index this document only"
                            disabled={working || d.status === "failed"}
                            onClick={() => reindex(d)} loading={busy === `index:${d.id}`}>
                            <Layers className="h-3 w-3" />
                          </Button>
                          <Button variant="ghost" size="sm" title="Re-parse from the stored original"
                            disabled={working} onClick={() => rebuild(d)} loading={busy === `rebuild:${d.id}`}>
                            <RefreshCw className="h-3 w-3" />
                          </Button>
                          <Button variant="ghost" size="sm" title="Remove document and its vectors"
                            disabled={working} onClick={() => remove(d)} loading={busy === `delete:${d.id}`}>
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* Inspector */}
      {inspect && (
        <div className="fixed inset-0 z-40 flex justify-end bg-black/50" onClick={() => setInspect(null)}>
          <div
            className="h-full w-full max-w-2xl overflow-y-auto border-l border-line bg-surface-2"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-surface-2 px-4 py-3">
              <div className="min-w-0">
                <div className="truncate text-sm text-ink">{inspect.doc.file_name ?? inspect.doc.title}</div>
                <div className="text-2xs text-ink-faint">
                  {inspect.doc.parser} · {fmtCount(inspect.doc.text_length)} chars · {inspect.chunks.length} chunks ·
                  document v{inspect.doc.document_version ?? 1}
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setInspect(null)}><X className="h-3.5 w-3.5" /></Button>
            </div>
            <div className="space-y-4 p-4">
              <Panel>
                <PanelHeader title={`Extracted text (${fmtCount(inspect.text.char_count)} chars)`} />
                <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap p-4 font-mono text-2xs leading-4 text-ink-muted">
                  {inspect.text.text}
                </pre>
              </Panel>
              <Panel>
                <PanelHeader title="Chunks &amp; provenance" />
                <ul className="divide-y divide-line/60">
                  {inspect.chunks.map((c) => (
                    <li key={c.id} className="px-4 py-2.5">
                      <div className="mb-1 flex flex-wrap items-center gap-1.5">
                        <span className="data-value text-2xs text-ink-ghost">#{c.chunk_index}</span>
                        {c.section_path && <Badge tone="neutral">{c.section_path}</Badge>}
                        {c.slide != null && <Badge tone="accent">slide {c.slide}{c.slide_title ? ` · ${c.slide_title}` : ""}</Badge>}
                        {c.page != null && <Badge tone="accent">page {c.page}</Badge>}
                        {c.chunking_strategy && <Badge tone="neutral">{c.chunking_strategy}</Badge>}
                      </div>
                      <p className="whitespace-pre-wrap text-2xs leading-4 text-ink-muted">{c.text}</p>
                    </li>
                  ))}
                  {inspect.chunks.length === 0 && (
                    <li className="px-4 py-6 text-center text-2xs text-ink-faint">
                      Not indexed yet. Use “re-index this document” to create chunks.
                    </li>
                  )}
                </ul>
              </Panel>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}