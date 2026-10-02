"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Check, Cloud, Database, FileUp, Globe, Layers, Link2, Upload, X,
} from "lucide-react";
import { api, SourceMode, UploadResult } from "@/lib/api";
import { Badge, Button, Panel, PanelHeader, Progress } from "@/components/ui";

const INPUT =
  "h-9 w-full rounded border border-line-strong bg-surface-3 px-3 text-sm text-ink placeholder:text-ink-ghost focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30";
const AREA =
  "w-full rounded border border-line-strong bg-surface-3 px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30";
const LABEL = "section-label mb-1.5 block";

const ACCEPTED = ".pdf,.pptx,.ppt,.docx,.txt,.md,.markdown,.html,.htm";
const MAX_BYTES = 100 * 1024 * 1024;

const MODES: { id: SourceMode; label: string; hint: string; icon: typeof Globe }[] = [
  { id: "external", label: "Discover sources", hint: "RAGForge finds and scores authoritative public sources.", icon: Globe },
  { id: "user_provided", label: "My files", hint: "Upload your own lecture decks, PDFs and notes.", icon: FileUp },
  { id: "mixed", label: "Mixed", hint: "Your documents plus externally discovered sources.", icon: Layers },
];

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

type DomainForm = Record<string, string>;

/** Step 1 of the guided pipeline. */
function StepDomain({
  form, setForm, onNext, submit,
}: {
  form: DomainForm;
  setForm: React.Dispatch<React.SetStateAction<DomainForm>>;
  onNext: () => void;
  submit?: () => void;
}) {
  const valid =
    form.name.trim() && form.domain.trim() && form.purpose.trim() && form.target_audience.trim();
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value });
  void submit;

  return (
    <Panel>
      <PanelHeader title="Step 1 — Domain" right={<Badge tone="accent">required</Badge>} />
      <div className="space-y-4 p-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className={LABEL}>Name</label>
            <input className={INPUT} required value={form.name} onChange={set("name")} placeholder="Naval Architecture KB" />
          </div>
          <div>
            <label className={LABEL}>Domain</label>
            <input className={INPUT} required value={form.domain} onChange={set("domain")} placeholder="Naval Architecture" />
          </div>
        </div>
        <div>
          <label className={LABEL}>Purpose / use case</label>
          <textarea className={AREA} required rows={3} value={form.purpose} onChange={set("purpose")}
            placeholder="Build a study assistant for naval architecture students" />
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className={LABEL}>Target audience</label>
            <input className={INPUT} required value={form.target_audience} onChange={set("target_audience")} placeholder="Naval Architecture students" />
          </div>
          <div>
            <label className={LABEL}>Depth</label>
            <select className={INPUT} value={form.depth} onChange={set("depth")}>
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="technical">Technical</option>
              <option value="expert">Expert</option>
            </select>
          </div>
        </div>
        <div className="flex justify-end border-t border-line pt-4">
          <Button variant="primary" onClick={onNext} disabled={!valid}>Continue to sources</Button>
        </div>
      </div>
    </Panel>
  );
}

/** Step 2: choose a source mode and stage the initial corpus. */
function StepSources({
  mode, onMode, files, setFiles, urlsText, setUrlsText,
}: {
  mode: SourceMode;
  onMode: (m: SourceMode) => void;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  urlsText: string;
  setUrlsText: (v: string) => void;
}) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}:${f.size}`));
      const next = [...prev];
      for (const f of Array.from(incoming)) {
        if (!seen.has(`${f.name}:${f.size}`)) next.push(f);
      }
      return next;
    });
  };

  const rejected = useMemo(
    () =>
      files.filter((f) => {
        const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
        return !ACCEPTED.split(",").includes(ext) || f.size > MAX_BYTES;
      }),
    [files],
  );

  const showFiles = mode === "user_provided" || mode === "mixed";
  const showUrls = mode === "user_provided" || mode === "mixed";

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader title="Step 2 — Sources" right={<Badge tone="neutral">{mode.replace("_", " ")}</Badge>} />
        <div className="space-y-4 p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            {MODES.map((m) => {
              const Icon = m.icon;
              const active = mode === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => onMode(m.id)}
                  className={`rounded border p-3 text-left transition-colors ${
                    active
                      ? "border-accent bg-accent/10"
                      : "border-line-strong bg-surface-3 hover:border-line-focus"
                  }`}
                >
                  <span className="mb-1 flex items-center gap-2">
                    <Icon className={`h-3.5 w-3.5 ${active ? "text-accent-soft" : "text-ink-ghost"}`} />
                    <span className="text-xs font-medium text-ink">{m.label}</span>
                    {active && <Check className="ml-auto h-3 w-3 text-accent-soft" />}
                  </span>
                  <span className="block text-2xs leading-4 text-ink-faint">{m.hint}</span>
                </button>
              );
            })}
          </div>

          {showFiles && (
            <div
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files); }}
              onClick={() => inputRef.current?.click()}
              className={`cursor-pointer rounded-lg border border-dashed px-4 py-8 text-center transition-colors ${
                drag ? "border-accent bg-accent/5" : "border-line-strong hover:border-line-focus"
              }`}
            >
              <Upload className="mx-auto mb-2 h-5 w-5 text-ink-ghost" />
              <p className="text-xs text-ink-muted">Drag &amp; drop files here, or click to browse</p>
              <p className="mt-1 text-2xs text-ink-faint">
                PDF · PPTX · DOCX · TXT · MD — max {fmtBytes(MAX_BYTES)} per file
              </p>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept={ACCEPTED}
                className="hidden"
                onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }}
              />
            </div>
          )}

          {files.length > 0 && (
            <div className="rounded border border-line bg-surface-3">
              <div className="flex items-center justify-between border-b border-line px-3 py-2">
                <span className="section-label">{files.length} file(s) staged</span>
                <button type="button" className="text-2xs text-ink-faint hover:text-ink" onClick={() => setFiles([])}>
                  clear
                </button>
              </div>
              <ul className="max-h-40 divide-y divide-line/60 overflow-y-auto">
                {files.map((f) => {
                  const bad = rejected.includes(f);
                  return (
                    <li key={`${f.name}:${f.size}`} className="flex items-center gap-2 px-3 py-1.5 text-xs">
                      <span className={`truncate ${bad ? "text-bad" : "ink text-ink"}`}>{f.name}</span>
                      <span className="ml-auto shrink-0 text-2xs text-ink-faint">{fmtBytes(f.size)}</span>
                      {bad && <Badge tone="bad">check format / size</Badge>}
                      <button
                        type="button"
                        aria-label={`remove ${f.name}`}
                        onClick={() => setFiles(files.filter((x) => x !== f))}
                        className="text-ink-ghost hover:text-ink"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </li>
                  );
                })}
              </ul>
              <p className="border-t border-line px-3 py-1.5 text-2xs text-ink-faint">
                Files are parsed locally. Duplicates are detected by content hash and never overwritten.
              </p>
            </div>
          )}

          {showUrls && (
            <div>
              <label className={LABEL}>Or paste URLs (one per line)</label>
              <textarea
                className={AREA}
                rows={3}
                value={urlsText}
                onChange={(e) => setUrlsText(e.target.value)}
                placeholder={"https://example.edu/notes/ship-stability\nhttps://example.org/class-rules"}
              />
              <p className="mt-1 text-2xs text-ink-faint">
                URLs are marked user-provided and are never scored down for lacking public authority signals.
              </p>
            </div>
          )}

          {mode === "external" && (
            <div className="rounded border border-line bg-surface-3 px-3 py-3 text-xs text-ink-muted">
              <Cloud className="mr-1.5 inline h-3.5 w-3.5 text-ink-ghost" />
              After creation, the <span className="text-ink">Sources</span> page will let you run
              user-URL and arXiv discovery with quality scoring.
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}

/** Staging happens after the KB exists, so failures never lose the KB. */
function StagedUpload({ kbId, files, urls }: { kbId: string; files: File[]; urls: string[] }) {
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function run() {
    setBusy(true);
    setError(null);
    try {
      let uploaded: UploadResult | null = null;
      if (files.length > 0) {
        uploaded = await api.uploadDocuments(kbId, files, { index: true }, setProgress);
      }
      if (urls.length > 0) {
        await api.addUserURLs(kbId, urls);
      }
      setResult(uploaded);
      if (!urls.length) router.refresh();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  if (result) {
    const ok = result.uploaded > 0;
    return (
      <Panel className="mt-4">
        <PanelHeader
          title={ok ? "Sources ingested" : "Nothing was ingested"}
          right={<Link2 className="h-3.5 w-3.5 text-ink-ghost" />}
        />
        <div className="space-y-2 p-4">
          <div className="flex flex-wrap gap-2 text-2xs">
            <Badge tone="ok">{result.uploaded} ingested</Badge>
            {result.duplicates > 0 && <Badge tone="warn">{result.duplicates} duplicate (kept existing)</Badge>}
            {result.rejected > 0 && <Badge tone="bad">{result.rejected} rejected</Badge>}
            {result.failed > 0 && <Badge tone="bad">{result.failed} failed</Badge>}
          </div>
          <ul className="space-y-1 text-2xs text-ink-faint">
            {result.files.map((f) => (
              <li key={f.file_name}>
                <span className={f.status === "rejected" || f.status === "failed" ? "text-bad" : "text-ink-muted"}>
                  {f.file_name}
                </span>
                {f.message ? ` — ${f.message}` : ""}
              </li>
            ))}
          </ul>
          {result.indexing?.error && (
            <p className="rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-2xs text-warn">
              Indexing did not run: {result.indexing.error}
            </p>
          )}
          <div className="flex justify-end border-t border-line pt-3">
            <Button variant="primary" onClick={() => router.push(`/knowledge-bases/${kbId}/documents`)}>
              Open document library
            </Button>
          </div>
        </div>
      </Panel>
    );
  }

  return (
    <Panel className="mt-4">
      <PanelHeader title="Step 3 — Ingest sources" right={<Database className="h-3.5 w-3.5 text-ink-ghost" />} />
      <div className="space-y-3 p-4">
        <p className="text-xs text-ink-faint">
          {files.length} file(s) and {urls.length} URL(s) will be ingested into the new knowledge base.
          Files are parsed, chunked, embedded and indexed locally.
        </p>
        {busy && <Progress value={progress} />}
        {error && <div className="rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>}
        <div className="flex justify-end gap-2 border-t border-line pt-3">
          <Button variant="ghost" onClick={() => router.push(`/knowledge-bases/${kbId}`)} disabled={busy}>
            Skip for now
          </Button>
          <Button variant="primary" loading={busy} onClick={run}>
            Ingest sources
          </Button>
        </div>
      </div>
    </Panel>
  );
}

export default function CreateKB() {
  const router = useRouter();
  const [form, setForm] = useState<DomainForm>({
    name: "",
    domain: "",
    purpose: "",
    target_audience: "",
    depth: "intermediate",
  });
  const [mode, setMode] = useState<SourceMode>("external");
  const [step, setStep] = useState<1 | 2>(1);
  const [kb, setKb] = useState<{ id: string } | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [urlsText, setUrlsText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const urls = useMemo(
    () => urlsText.split("\n").map((u) => u.trim()).filter(Boolean),
    [urlsText],
  );

  async function create() {
    setBusy(true);
    setError(null);
    try {
      const created = await api.createKB({ ...form, source_mode: mode });
      if (mode === "external") {
        router.push(`/knowledge-bases/${created.id}/sources`);
        return;
      }
      setKb({ id: created.id });
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-6">
        <div className="section-label mb-2">Workspace</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Create Knowledge Base</h1>
        <p className="mt-1 text-xs text-ink-faint">
          A knowledge base needs no ground truth. An evaluation benchmark is optional and can be added
          later — a KB becomes READY once it is indexed.
        </p>
      </div>

      <ol className="mb-4 flex flex-wrap items-center gap-x-2 gap-y-1 text-2xs text-ink-faint">
        {["Domain", "Sources", "Processing", "Indexing", "Ready"].map((label, i) => {
          const n = i + 1;
          const active = step === 1 ? n === 1 : n === 2;
          return (
            <li key={label} className="flex items-center gap-2">
              <span
                className={`inline-flex h-4 w-4 items-center justify-center rounded-full border ${
                  active ? "border-accent text-accent-soft" : "border-line-strong"
                }`}
              >
                {n}
              </span>
              <span className={active ? "text-ink" : ""}>{label}</span>
              {i < 4 && <span className="text-line-strong">—</span>}
            </li>
          );
        })}
      </ol>

      {error && (
        <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>
      )}

      {step === 1 ? (
        <StepDomain form={form} setForm={setForm} onNext={() => setStep(2)} submit={() => create()} />
      ) : (
        <>
          <StepSources
            mode={mode}
            onMode={setMode}
            files={files}
            setFiles={setFiles}
            urlsText={urlsText}
            setUrlsText={setUrlsText}
          />
          <div className="mt-4 flex justify-end border-t border-line pt-4">
            <Button variant="ghost" onClick={() => setStep(1)} disabled={busy}>Back</Button>
            <Button variant="primary" loading={busy} onClick={() => create()}>
              Create Knowledge Base
            </Button>
          </div>
        </>
      )}

      {kb && <StagedUpload kbId={kb.id} files={files} urls={urls} />}
    </div>
  );
}