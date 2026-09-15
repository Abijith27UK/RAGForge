"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Database, Download, Layers } from "lucide-react";
import { api, BuildRun, Document } from "@/lib/api";
import {
  Badge, Button, EmptyState, Panel, PanelHeader, Skeleton, StatusPill,
} from "@/components/ui";
import { BuildRunStages } from "@/components/PipelineGraph";
import { fmtCount } from "@/lib/utils";

export default function ProcessingPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [docs, setDocs] = useState<Document[] | null>(null);
  const [run, setRun] = useState<BuildRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(() => {
    api.listDocuments(kbId).then(setDocs).catch(() => setDocs([]));
    api.buildStatus(kbId).then(setRun).catch(() => setRun(null));
  }, [kbId]);
  useEffect(load, [load]);

  async function doIngest() {
    setBusy("ingest"); setError(null);
    try { setRun(await api.ingest(kbId)); load(); }
    catch (e: unknown) { setError(String((e as Error).message)); }
    finally { setBusy(null); }
  }

  async function doIndex() {
    setBusy("index"); setError(null);
    try { setRun(await api.index(kbId)); load(); }
    catch (e: unknown) { setError(String((e as Error).message)); }
    finally { setBusy(null); }
  }

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6 flex items-end justify-between">
        <div>
          <div className="section-label mb-2">Research</div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Processing</h1>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" loading={busy === "ingest"} disabled={busy !== null} onClick={doIngest}>
            <Download className="h-3.5 w-3.5" /> Ingest accepted sources
          </Button>
          <Button variant="primary" size="sm" loading={busy === "index"} disabled={busy !== null} onClick={doIndex}>
            <Layers className="h-3.5 w-3.5" /> Chunk + Embed + Index
          </Button>
        </div>
      </div>

      {error && <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>}

      <Panel className="mb-4">
        <PanelHeader
          title="Build pipeline"
          right={run ? <StatusPill status={run.status} /> : <span className="text-2xs text-ink-ghost">no build run yet</span>}
        />
        {run ? (
          <BuildRunStages run={run} />
        ) : (
          <p className="px-4 py-6 text-center text-xs text-ink-faint">
            Run ingestion, then chunk + embed + index. Stage status appears here from the real build run.
          </p>
        )}
      </Panel>

      <Panel>
        <PanelHeader
          title="Documents"
          right={<span className="data-value text-2xs text-ink-faint">{docs ? `${docs.length} ingested` : "…"}</span>}
        />
        {!docs ? (
          <div className="p-4"><Skeleton className="h-40 w-full" /></div>
        ) : docs.length === 0 ? (
          <div className="p-4">
            <EmptyState icon={<Database className="h-6 w-6" />} title="No documents ingested yet"
              hint="Accept sources on the Sources page, then run ingestion." />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-line text-left">
                  {["Document", "Type", "Chars", "Pages", "Content hash"].map((h) => (
                    <th key={h} className="px-4 py-2 font-medium text-ink-faint">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id} className="border-b border-line/50 transition-colors last:border-0 hover:bg-surface-3">
                    <td className="max-w-[280px] px-4 py-2">
                      <div className="truncate text-ink">{d.title ?? d.url}</div>
                      <div className="truncate text-2xs text-ink-faint">{d.url}</div>
                    </td>
                    <td className="px-4 py-2"><Badge tone="neutral">{d.source_type}</Badge></td>
                    <td className="data-value px-4 py-2 text-ink-muted">{fmtCount(d.text_length)}</td>
                    <td className="data-value px-4 py-2 text-ink-muted">{d.page_count ?? "—"}</td>
                    <td className="px-4 py-2"><code className="data-value text-2xs text-ink-faint">{d.content_hash.slice(0, 12)}…</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
