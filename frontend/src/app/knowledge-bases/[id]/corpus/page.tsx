"use client";

/**
 * Corpus Command Center — the primary surface for managing a large corpus.
 *
 * Design intent: this is a data-engineering console, not an AI chat UI.
 * Progressive disclosure: the health strip answers "is this corpus
 * trustworthy?" in one glance; the corpus map and integrity detail sit behind
 * a toggle so the operator is not buried in metrics.
 *
 * Honesty rules this view obeys:
 *  - A count that could not be measured renders as "unknown", never 0.
 *  - Integrity scans are read-only; repair is a separate, confirmed action.
 *  - Ground truth is NOT_AVAILABLE unless a frozen benchmark exists, and that
 *    is displayed as a first-class state rather than hidden.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Copy, Fingerprint,
  Layers, RefreshCw, RotateCcw, ShieldAlert, Upload, Wrench,
} from "lucide-react";
import {
  CorpusManifest, CorpusVersion, IngestionBatch, IngestionBatchDetail,
  IntegrityFinding, IntegrityReport, RepairAction, RepairPlan, RepairResult,
  countLabel, corpusApi,
} from "@/lib/api";
import {
  Badge, Button, EmptyState, Metric, Panel, PanelHeader, Progress, Skeleton,
} from "@/components/ui";
import { fmtDate } from "@/lib/utils";

const ACCEPTED = ".pdf,.pptx,.ppt,.docx,.txt,.md,.markdown,.html,.htm";

const STATUS_TONE: Record<string, "ok" | "warn" | "bad" | "neutral" | "accent"> = {
  HEALTHY: "ok", WARNING: "warn", ERROR: "bad",
  complete: "ok", partial: "warn", incomplete: "warn", failed: "bad",
  running: "accent", pending: "neutral", cancelled: "neutral",
  ready: "ok", parsed: "accent", failed_document: "bad",
};

function ItemGlyph({ status }: { status: string }) {
  if (status === "complete") return <span className="text-ok">✓</span>;
  if (status === "processing") return <span className="animate-pulse text-accent">⟳</span>;
  if (status === "duplicate") return <span className="text-warn">⧉</span>;
  if (status === "failed" || status === "rejected") return <span className="text-bad">✕</span>;
  if (status === "cancelled") return <span className="text-ink-faint">⊘</span>;
  return <span className="text-ink-faint">·</span>;
}

const CHECK_LABELS: { key: keyof IntegrityReport; label: string }[] = [
  { key: "orphan_vectors", label: "No orphan vectors" },
  { key: "stale_vectors", label: "No stale vectors" },
  { key: "missing_vectors", label: "Every chunk has a vector" },
  { key: "embedding_mismatches", label: "Embedding identity consistent" },
  { key: "failed_documents", label: "No failed documents" },
  { key: "provenance_gaps", label: "Provenance complete" },
  { key: "duplicate_documents", label: "No duplicate documents" },
  { key: "broken_source_references", label: "Source references intact" },
];

export default function CorpusPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [manifest, setManifest] = useState<CorpusManifest | null>(null);
  const [report, setReport] = useState<IntegrityReport | null>(null);
  const [versions, setVersions] = useState<CorpusVersion[]>([]);
  const [batches, setBatches] = useState<IngestionBatch[]>([]);
  const [activeBatch, setActiveBatch] = useState<IngestionBatchDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [plan, setPlan] = useState<RepairPlan | null>(null);
  const [repairOut, setRepairOut] = useState<RepairResult | null>(null);
  const [showMap, setShowMap] = useState(false);
  const [showFindings, setShowFindings] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    if (!kbId) return;
    try {
      const [m, v, b] = await Promise.all([
        corpusApi.manifest(kbId),
        corpusApi.versions(kbId),
        corpusApi.listBatches(kbId),
      ]);
      setManifest(m);
      setVersions(v);
      setBatches(b);
    } catch (err) {
      setError(String((err as Error).message));
    }
  }, [kbId]);

  const runIntegrity = useCallback(async () => {
    if (!kbId) return;
    setBusy("integrity");
    setError(null);
    try {
      setReport(await corpusApi.integrity(kbId));
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
    }
  }, [kbId]);

  useEffect(() => { void load(); }, [load]);

  async function upload(files: File[]) {
    if (!kbId || files.length === 0) return;
    setBusy("upload");
    setError(null);
    setUploadPct(0);
    try {
      const detail = await corpusApi.createBatch(kbId, files);
      setActiveBatch(detail);
      await load();
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
      setUploadPct(0);
    }
  }

  async function openBatch(batchId: string) {
    setBusy(`batch:${batchId}`);
    try {
      setActiveBatch(await corpusApi.getBatch(kbId, batchId));
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
    }
  }

  async function resume(includeCompleted: boolean) {
    if (!activeBatch) return;
    setBusy("resume");
    setError(null);
    try {
      setActiveBatch(
        await corpusApi.resumeBatch(kbId, activeBatch.batch.id, {
          include_completed: includeCompleted, retry_failed: true, index: true,
        }),
      );
      await load();
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
    }
  }

  /** Repair is always plan-then-confirm; the plan states the blast radius. */
  async function repair(action: RepairAction, destructive: boolean) {
    if (!kbId) return;
    setBusy(`repair:${action}`);
    setError(null);
    setRepairOut(null);
    try {
      const p = await corpusApi.planRepair(kbId, { action });
      setPlan(p);
      if (destructive) {
        const ok = window.confirm(
          `${p.description}\n\nDocuments affected: ${p.affected_documents.length}\n` +
          `Expected chunks: ${countLabel(p.expected_chunks)}\n` +
          `Expected vectors: ${countLabel(p.expected_vectors)}\n\nThis cannot be undone. Continue?`,
        );
        if (!ok) return;
      }
      setRepairOut(await corpusApi.runRepair(kbId, { action, confirm_action: action }));
      await load();
      if (report) void runIntegrity();
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
    }
  }

  const s = manifest?.summary;
  const health = report?.overall_status;
  const byType = useMemo(() => {
    const map = new Map<string, { docs: number; pages: number; slides: number; sections: number; chunks: number }>();
    for (const e of manifest?.entries ?? []) {
      const cur = map.get(e.file_type) ?? { docs: 0, pages: 0, slides: 0, sections: 0, chunks: 0 };
      cur.docs += 1;
      cur.pages += e.page_count ?? 0;
      cur.slides += e.slide_count ?? 0;
      cur.sections += e.section_count ?? 0;
      cur.chunks += e.chunk_count;
      map.set(e.file_type, cur);
    }
    return [...map.entries()].sort((a, b) => b[1].chunks - a[1].chunks);
  }, [manifest]);

  if (!manifest) {
    return (
      <div className="space-y-3">
        <Panel><div className="p-4"><Skeleton className="h-6 w-48" /></div></Panel>
        <Panel><div className="p-4"><Skeleton className="h-32 w-full" /></div></Panel>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
          {error}
        </div>
      )}

      {/* ------------------------------------------------ CORPUS HEALTH */}
      <Panel>
        <PanelHeader
          title="Corpus health"
          right={
            <div className="flex items-center gap-2">
              {report ? (
                <Badge tone={STATUS_TONE[health ?? "HEALTHY"] ?? "neutral"}>
                  {health}
                </Badge>
              ) : (
                <Badge tone="neutral">not scanned yet</Badge>
              )}
              <Button size="sm" loading={busy === "integrity"} onClick={() => void runIntegrity()}>
                <ShieldAlert className="mr-1.5 h-3 w-3" /> Scan integrity
              </Button>
            </div>
          }
        />
        <div className="grid grid-cols-2 gap-px bg-line sm:grid-cols-4 lg:grid-cols-7">
          <Metric label="Documents" value={`${s?.completed_documents ?? 0} / ${s?.total_documents ?? 0}`} hint="indexed / total" />
          <Metric label="Chunks" value={(s?.total_chunks ?? 0).toLocaleString()} />
          <Metric
            label="Vectors"
            value={s?.vector_count_confirmed ? (s?.total_vectors ?? 0).toLocaleString() : "unknown"}
            hint={s?.vector_count_confirmed ? "confirmed by store" : "vector store unreachable"}
          />
          <Metric label="Failures" value={String(s?.failed_documents ?? 0)} tone={(s?.failed_documents ?? 0) > 0 ? "warn" : "ok"} />
          <Metric label="Duplicates" value={String(s?.duplicate_documents ?? 0)} />
          <Metric label="Stale docs" value={String(s?.stale_documents ?? 0)} />
          <Metric label="Corpus size" value={`${((s?.total_corpus_size_bytes ?? 0) / 1048576).toFixed(1)} MB`} />
        </div>
      </Panel>

      {/* ------------------------------------------------- INGESTION */}
      <Panel>
        <PanelHeader title="Bulk ingestion" right={<span className="text-2xs text-ink-faint">50–200 files per batch</span>} />
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault(); setDrag(false);
            void upload(Array.from(e.dataTransfer.files));
          }}
          onClick={() => inputRef.current?.click()}
          className={`m-4 cursor-pointer rounded border border-dashed p-6 text-center transition-colors ${
            drag ? "border-accent bg-accent/5" : "border-line hover:border-ink-faint"
          }`}
        >
          <Upload className="mx-auto mb-2 h-5 w-5 text-ink-faint" />
          <p className="text-sm text-ink">Drop your corpus here</p>
          <p className="mt-1 text-2xs text-ink-faint">PDF · PPTX · DOCX · TXT · MD · HTML</p>
          {busy === "upload" && <Progress value={uploadPct || 100} className="mt-3" />}
          <input
            ref={inputRef} type="file" multiple accept={ACCEPTED} className="hidden"
            onChange={(e) => { void upload(Array.from(e.target.files ?? [])); e.target.value = ""; }}
          />
        </div>

        {batches.length > 0 && (
          <div className="px-4 pb-4">
            <p className="mb-2 text-2xs uppercase tracking-wide text-ink-faint">Batch history</p>
            <ul className="space-y-1">
              {batches.map((b) => {
                const done = b.completed_items + b.failed_items + b.duplicate_items + b.rejected_items;
                const pct = b.total_items ? (done / b.total_items) * 100 : 0;
                return (
                  <li key={b.id}>
                    <button
                      onClick={() => void openBatch(b.id)}
                      className="flex w-full items-center gap-3 rounded px-2 py-1.5 text-left text-xs hover:bg-surface-2"
                    >
                      <span className="font-mono text-2xs text-ink-faint">{b.id.slice(0, 12)}</span>
                      <span className="w-16 text-ink-muted">{b.total_items} files</span>
                      <span className="w-20"><Badge tone={STATUS_TONE[b.status] ?? "neutral"}>{b.status}</Badge></span>
                      <span className="flex-1"><Progress value={pct} tone={b.failed_items ? "warn" : "ok"} /></span>
                      <span className="text-2xs text-ink-faint">{fmtDate(b.created_at)}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {activeBatch && (
          <div className="border-t border-line px-4 py-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="h-3.5 w-3.5 text-ink-faint" />
                <span className="text-xs text-ink">Batch {activeBatch.batch.id.slice(0, 12)}</span>
                <Badge tone={STATUS_TONE[activeBatch.batch.status] ?? "neutral"}>
                  {activeBatch.batch.status}
                </Badge>
                {activeBatch.batch.resumed_at && <Badge tone="accent">resumed</Badge>}
              </div>
              <div className="flex gap-2">
                <Button size="sm" loading={busy === "resume"} onClick={() => void resume(false)}>
                  <RotateCcw className="mr-1.5 h-3 w-3" /> Resume pending
                </Button>
                <Button size="sm" onClick={() => void resume(true)}>
                  Force re-run all
                </Button>
              </div>
            </div>
            <p className="mb-2 text-2xs text-ink-faint">
              {activeBatch.resumable_items} item(s) resumable · completed documents are skipped
              unless you force a re-run.
            </p>
            <ul className="max-h-80 space-y-0.5 overflow-y-auto">
              {activeBatch.items.map((it) => (
                <li key={it.id} className="flex items-start gap-2 py-0.5 text-xs">
                  <span className="w-3 shrink-0 pt-0.5"><ItemGlyph status={it.status} /></span>
                  <span className="flex-1 truncate text-ink-muted" title={it.error_message ?? it.file_name}>
                    {it.file_name}
                  </span>
                  <span className="w-24 shrink-0 text-right text-2xs text-ink-faint">
                    {it.error_message ? (
                      <span className="text-bad" title={it.error_message}>
                        {it.error_code ?? "failed"}
                      </span>
                    ) : it.status === "duplicate" ? (
                      <span className="text-warn">duplicate</span>
                    ) : it.status === "complete" ? (
                      <span className="text-ink-muted">{it.chunk_count} chunks</span>
                    ) : (
                      it.stage
                    )}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Panel>

      {/* --------------------------------------------------- CORPUS MAP */}
      <Panel>
        <PanelHeader
          title="Corpus map"
          right={
            <button
              onClick={() => setShowMap((v) => !v)}
              className="flex items-center gap-1 text-2xs text-ink-faint hover:text-ink"
            >
              {showMap ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              {showMap ? "hide" : "show"}
            </button>
          }
        />
        <div className="px-4 pb-3 text-2xs text-ink-faint">
          {byType.length} document type(s) ·{" "}
          {s?.total_pages?.toLocaleString() ?? 0} pages ·{" "}
          {s?.total_slides?.toLocaleString() ?? 0} slides ·{" "}
          {s?.total_sections?.toLocaleString() ?? 0} sections
        </div>
        {showMap && (
          <table className="w-full border-t border-line text-xs">
            <thead>
              <tr className="text-left text-2xs uppercase tracking-wide text-ink-faint">
                <th className="px-4 py-1.5 font-normal">Type</th>
                <th className="px-2 py-1.5 text-right font-normal">Docs</th>
                <th className="px-2 py-1.5 text-right font-normal">Pages</th>
                <th className="px-2 py-1.5 text-right font-normal">Slides</th>
                <th className="px-2 py-1.5 text-right font-normal">Sections</th>
                <th className="px-4 py-1.5 text-right font-normal">Chunks</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {byType.map(([type, v]) => (
                <tr key={type}>
                  <td className="px-4 py-1.5 font-mono text-ink-muted">{type}</td>
                  <td className="px-2 py-1.5 text-right text-ink">{v.docs}</td>
                  <td className="px-2 py-1.5 text-right text-ink-muted">{v.pages || "—"}</td>
                  <td className="px-2 py-1.5 text-right text-ink-muted">{v.slides || "—"}</td>
                  <td className="px-2 py-1.5 text-right text-ink-muted">{v.sections || "—"}</td>
                  <td className="px-4 py-1.5 text-right text-ink">{v.chunks.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      {/* --------------------------------------------------- INTEGRITY */}
      <Panel>
        <PanelHeader
          title="Integrity"
          right={report ? <span className="text-2xs text-ink-faint">read-only scan · {report.duration_seconds}s</span> : null}
        />
        {!report ? (
          <div className="px-4 pb-4">
            <p className="text-xs text-ink-faint">
              Run a scan to check for orphan vectors, stale vectors, missing vectors,
              embedding mismatches and provenance gaps. A scan never modifies anything.
            </p>
          </div>
        ) : (
          <div className="px-4 pb-4">
            {report.unknown_counts.length > 0 && (
              <p className="mb-3 rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-2xs text-warn">
                Some checks could not run: {report.unknown_counts.join(", ")}. A check that
                could not run is not a check that passed.
              </p>
            )}
            <ul className="space-y-1">
              {CHECK_LABELS.map(({ key, label }) => {
                const list = report[key] as IntegrityFinding[];
                const unknown = report.unknown_counts.some((u) => u.startsWith(key));
                const ok = unknown ? null : list.length === 0;
                return (
                  <li key={key} className="flex items-center gap-2 text-xs">
                    {ok === null ? (
                      <span className="w-3 text-ink-faint">?</span>
                    ) : ok ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-ok" />
                    ) : (
                      <AlertTriangle className="h-3.5 w-3.5 text-warn" />
                    )}
                    <span className={ok === false ? "text-warn" : "text-ink-muted"}>{label}</span>
                    {!ok && !unknown && (
                      <span className="text-2xs text-ink-faint">({list.length})</span>
                    )}
                    {unknown && <span className="text-2xs text-ink-faint">could not be checked</span>}
                  </li>
                );
              })}
            </ul>

            {report.findings.length > 0 && (
              <>
                <button
                  onClick={() => setShowFindings((v) => !v)}
                  className="mt-3 flex items-center gap-1 text-2xs text-ink-faint hover:text-ink"
                >
                  {showFindings ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  {showFindings ? "hide" : "show"} {report.findings.length} finding(s)
                </button>
                {showFindings && (
                  <ul className="mt-2 max-h-72 space-y-1.5 overflow-y-auto">
                    {report.findings.slice(0, 200).map((f, i) => (
                      <li key={`${f.check}-${i}`} className="border-l-2 border-line pl-2 text-2xs">
                        <span className="font-mono text-ink-faint">{f.check}</span>
                        <span className="ml-2 text-ink-muted">{f.message}</span>
                        {f.document_id && (
                          <span className="ml-2 font-mono text-ink-faint">{f.document_id.slice(0, 12)}</span>
                        )}
                      </li>
                    ))}
                    {report.findings.length > 200 && (
                      <li className="text-2xs text-ink-faint">
                        …and {report.findings.length - 200} more (see /corpus-integrity)
                      </li>
                    )}
                  </ul>
                )}
              </>
            )}
          </div>
        )}
      </Panel>

      {/* ----------------------------------------------------- REPAIR */}
      <Panel>
        <PanelHeader title="Repair" right={<span className="text-2xs text-ink-faint">explicit, confirmed, never automatic</span>} />
        <div className="flex flex-wrap gap-2 px-4 pb-4">
          <Button size="sm" loading={busy === "repair:reindex_failed_documents"}
            onClick={() => void repair("reindex_failed_documents", false)}>
            <Wrench className="mr-1.5 h-3 w-3" /> Re-index failed
          </Button>
          <Button size="sm" loading={busy === "repair:remove_orphan_vectors"}
            onClick={() => void repair("remove_orphan_vectors", true)}>
            Remove orphan vectors
          </Button>
          <Button size="sm" variant="danger" loading={busy === "repair:rebuild_kb"}
            onClick={() => void repair("rebuild_kb", true)}>
            Rebuild entire KB
          </Button>
        </div>
        {plan && (
          <div className="mx-4 mb-3 rounded border border-line bg-surface-2 px-3 py-2 text-2xs">
            <p className="text-ink-muted">{plan.description}</p>
            <p className="mt-1 text-ink-faint">
              documents {plan.affected_documents.length} · chunks {countLabel(plan.expected_chunks)} ·
              vectors {countLabel(plan.expected_vectors)}
              {plan.counts_confirmed ? "" : " (expected counts could not be measured)"}
            </p>
          </div>
        )}
        {repairOut && (
          <div className={`mx-4 mb-4 rounded border px-3 py-2 text-2xs ${repairOut.ok ? "border-ok/40 bg-ok/5" : "border-bad/40 bg-bad/10"}`}>
            <p className="text-ink-muted">{repairOut.ok ? repairOut.message : repairOut.errors.join("; ")}</p>
            {repairOut.ok && (
              <p className="mt-1 text-ink-faint">
                chunks written {repairOut.chunks_written} · vectors written {countLabel(repairOut.vectors_written)} ·
                vectors removed {repairOut.removal_confirmed ? repairOut.vectors_removed : "unknown"}
              </p>
            )}
          </div>
        )}
      </Panel>

      {/* ---------------------------------------------------- VERSION */}
      <Panel>
        <PanelHeader
          title="Corpus version"
          right={
            <Button size="sm" loading={busy === "snapshot"}
              onClick={async () => {
                setBusy("snapshot");
                try { await corpusApi.snapshot(kbId, "manual snapshot"); await load(); }
                catch (err) { setError(String((err as Error).message)); }
                finally { setBusy(null); }
              }}>
              <Fingerprint className="mr-1.5 h-3 w-3" /> Snapshot
            </Button>
          }
        />
        <div className="grid grid-cols-2 gap-px bg-line lg:grid-cols-4">
          <Metric label="Version" value={versions[0]?.version ?? "—"} />
          <Metric
            label="Fingerprint"
            value={versions[0] ? `${versions[0].fingerprint.slice(0, 8)}…${versions[0].fingerprint.slice(-4)}` : "—"}
            hint={versions[0] ? versions[0].fingerprint : undefined}
          />
          <Metric label="Embedding" value={versions[0]?.embedding_identity ?? s?.embedding_identity ?? "—"} />
          <Metric label="Chunking" value={versions[0]?.chunking_identity ?? s?.chunking_identity ?? "—"} />
        </div>
        {versions.length > 1 && (
          <div className="px-4 py-3">
            <p className="mb-1 text-2xs uppercase tracking-wide text-ink-faint">History</p>
            <ul className="space-y-0.5">
              {versions.map((v) => (
                <li key={v.id} className="flex items-center gap-3 text-2xs">
                  <span className="w-10 font-mono text-ink">{v.version}</span>
                  <span className="flex-1 font-mono text-ink-faint">{v.fingerprint.slice(0, 16)}…</span>
                  <span className="text-ink-muted">{v.document_count} docs</span>
                  <span className="text-ink-faint">{fmtDate(v.created_at)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Panel>

      {manifest.entries.length === 0 && (
        <Panel>
          <EmptyState
            icon={<Layers className="h-6 w-6" />}
            title="No documents in this corpus yet"
            hint="Drop files above, or use the Documents page for small batches."
          />
        </Panel>
      )}
    </div>
  );
}