"use client";

/**
 * Evaluation — ground-truth authoring, lifecycle review, and strict metric runs.
 *
 * Research integrity rules mirrored from the backend:
 * - Headline metrics come ONLY from explicit chunk/document ground truth.
 * - Keyword mode is diagnostic-only and labelled as such.
 * - FROZEN questions are immutable; editing APPROVED creates a new DRAFT revision.
 * - Only FROZEN benchmark versions can be selected for official runs.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  ChevronDown, ChevronRight, CircleAlert, GitCommitVertical, Lock, RefreshCw, ShieldCheck,
} from "lucide-react";
import {
  api,
  BenchmarkVersion,
  Chunk,
  Document,
  EvaluationQuestion,
  EvaluationRun,
  QuestionLifecycle,
} from "@/lib/api";
import { cn, fmtCount } from "@/lib/utils";
import { Badge, Button, EmptyState, Metric, Panel, PanelHeader, ScoreBar, StatusDot } from "@/components/ui";

const fmt = (v: number | null | undefined) => (v == null ? "n/a" : v.toFixed(3));

const LIFECYCLE_TONE: Record<QuestionLifecycle, "neutral" | "violet" | "ok" | "accent"> = {
  DRAFT: "neutral",
  REVIEW: "violet",
  APPROVED: "ok",
  FROZEN: "accent",
};

type GroundTruthMode = "keywords" | "chunks" | "documents";

export default function EvaluationPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [questions, setQuestions] = useState<EvaluationQuestion[]>([]);
  const [versions, setVersions] = useState<BenchmarkVersion[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);

  // Authoring state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [qText, setQText] = useState("");
  const [mode, setMode] = useState<GroundTruthMode>("keywords");
  const [keywords, setKeywords] = useState("");
  const [selectedChunkIds, setSelectedChunkIds] = useState<string[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [chunkFilter, setChunkFilter] = useState("");
  const [gtNotes, setGtNotes] = useState("");

  // Review identity + version creation
  const [reviewer, setReviewer] = useState("");
  const [versionName, setVersionName] = useState("");
  const [versionLabel, setVersionLabel] = useState("");

  // Run state
  const [topK, setTopK] = useState(5);
  const [runLabel, setRunLabel] = useState("");
  const [runVersion, setRunVersion] = useState<string>("");
  const [diagnosticMode, setDiagnosticMode] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api.listQuestions(kbId).then(setQuestions).catch(() => null);
    api.listBenchmarkVersions(kbId).then(setVersions).catch(() => null);
    api.listEvaluationRuns(kbId).then(setRuns).catch(() => null);
    api.listChunks(kbId, 500).then(setChunks).catch(() => null);
    api.listDocuments(kbId).then(setDocuments).catch(() => null);
  }, [kbId]);
  useEffect(load, [load]);

  const frozenVersions = useMemo(() => versions.filter((v) => v.status === "FROZEN"), [versions]);
  const latestRun = runs[0] ?? null;
  const chunkById = useMemo(() => new Map(chunks.map((c) => [c.id, c])), [chunks]);
  const docTitle = useCallback(
    (docId: string | null | undefined) =>
      documents.find((d) => d.id === docId)?.title ??
      chunks.find((c) => c.document_id === docId)?.document_title ??
      docId,
    [documents, chunks],
  );

  const filteredChunks = useMemo(() => {
    const f = chunkFilter.toLowerCase().trim();
    const base = f
      ? chunks.filter(
          (c) =>
            c.text.toLowerCase().includes(f) ||
            (c.section ?? "").toLowerCase().includes(f) ||
            (c.document_title ?? "").toLowerCase().includes(f),
        )
      : chunks;
    return base.slice(0, 60);
  }, [chunks, chunkFilter]);

  const toggle = (list: string[], id: string) =>
    list.includes(id) ? list.filter((x) => x !== id) : [...list, id];

  function resetForm() {
    setEditingId(null);
    setQText("");
    setKeywords("");
    setSelectedChunkIds([]);
    setSelectedDocIds([]);
    setGtNotes("");
  }

  function startEdit(q: EvaluationQuestion) {
    setEditingId(q.id);
    setQText(q.question);
    setGtNotes(q.notes ?? "");
    setMode(
      q.expected_chunk_ids.length ? "chunks"
      : q.expected_document_ids.length ? "documents"
      : "keywords",
    );
    setSelectedChunkIds(q.expected_chunk_ids);
    setSelectedDocIds(q.expected_document_ids);
    setKeywords(q.expected_keywords.join(", "));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function submitQuestion(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (mode === "keywords" && !keywords.trim()) {
      setError("Add at least one expected keyword (or switch to chunk/document selection).");
      return;
    }
    if (mode === "chunks" && selectedChunkIds.length === 0) {
      setError("Select at least one expected chunk (or switch mode).");
      return;
    }
    if (mode === "documents" && selectedDocIds.length === 0) {
      setError("Select at least one expected document (or switch mode).");
      return;
    }
    const payload = {
      question: qText,
      expected_keywords: mode === "keywords" ? keywords.split(",").map((k) => k.trim()).filter(Boolean) : [],
      expected_chunk_ids: mode === "chunks" ? selectedChunkIds : [],
      expected_document_ids: mode === "documents" ? selectedDocIds : [],
      notes: gtNotes,
    };
    try {
      if (editingId) {
        const rev = await api.editQuestion(kbId, editingId, payload);
        // Backend may return a NEW draft revision (edited an APPROVED question).
        if (rev.id !== editingId) setExpanded(rev.id);
      } else {
        await api.addQuestion(kbId, payload);
      }
      resetForm();
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function setStatus(qId: string, status: QuestionLifecycle) {
    setError(null);
    if ((status === "APPROVED" || status === "FROZEN") && !reviewer.trim()) {
      setError("Enter a reviewer name first — approvals must be attributable.");
      return;
    }
    try {
      await api.setQuestionStatus(kbId, qId, status, reviewer.trim());
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function del(qId: string) {
    setError(null);
    try {
      await api.deleteQuestion(kbId, qId);
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function createVersion(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!versionName.trim()) {
      setError("Give the benchmark version a name, e.g. automobile-engineering-v2.");
      return;
    }
    try {
      await api.createBenchmarkVersion(kbId, {
        version: versionName.trim(),
        label: versionLabel.trim(),
        created_by: reviewer.trim() || "ui",
      });
      setVersionName("");
      setVersionLabel("");
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function freezeVersion(v: BenchmarkVersion) {
    setError(null);
    if (!reviewer.trim()) {
      setError("Enter a reviewer name first — freezing must be attributable.");
      return;
    }
    try {
      await api.freezeBenchmarkVersion(kbId, v.id, reviewer.trim());
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function deleteVersion(v: BenchmarkVersion) {
    setError(null);
    try {
      await api.deleteBenchmarkVersion(kbId, v.id);
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function runEval() {
    setBusy(true);
    setError(null);
    try {
      await api.evaluate(kbId, topK, runLabel, diagnosticMode, runVersion || null);
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  const counts = useMemo(() => {
    const c = { DRAFT: 0, REVIEW: 0, APPROVED: 0, FROZEN: 0 } as Record<QuestionLifecycle, number>;
    for (const q of questions) c[q.status] = (c[q.status] ?? 0) + 1;
    return c;
  }, [questions]);

  return (
    <div className="mx-auto max-w-7xl">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-semibold tracking-tight text-ink">Evaluation</h1>
          <p className="mt-0.5 text-xs text-ink-muted">
            Author ground truth, review it through the lifecycle, freeze a benchmark version, then run
            strict retrieval metrics against it.
          </p>
        </div>
        <div className="flex items-center gap-4 text-2xs text-ink-faint">
          {(Object.entries(counts) as [QuestionLifecycle, number][]).map(([s, n]) => (
            <span key={s} className="flex items-center gap-1.5">
              <Badge tone={LIFECYCLE_TONE[s]}>{s}</Badge>
              <span className="data-value">{n}</span>
            </span>
          ))}
        </div>
      </header>

      {error && (
        <div className="mb-4 flex items-start gap-2 rounded border border-warn-dim bg-warn/10 px-3 py-2 text-xs text-warn">
          <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        {/* ------------------------------ Left: authoring + questions ------------------------------ */}
        <div className="space-y-4">
          <Panel>
            <PanelHeader
              title={editingId ? "Revise question" : "Author evaluation question"}
              right={
                editingId ? (
                  <Button size="sm" variant="ghost" onClick={resetForm}>Cancel</Button>
                ) : (
                  <span className="text-2xs text-ink-faint">explicit ground truth only</span>
                )
              }
            />
            <form onSubmit={submitQuestion} className="space-y-3 p-4 pt-3">
              <input
                value={qText}
                onChange={(e) => setQText(e.target.value)}
                required
                className="w-full rounded border border-line bg-surface-2 px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                placeholder="What are the major functions of an EV battery management system?"
              />

              <div className="flex flex-wrap gap-1.5">
                {(
                  [
                    ["keywords", "Keywords — diagnostic"],
                    ["chunks", "Expected chunks — rigorous"],
                    ["documents", "Expected documents — doc-level"],
                  ] as [GroundTruthMode, string][]
                ).map(([m, label]) => (
                  <button
                    type="button"
                    key={m}
                    onClick={() => setMode(m)}
                    className={cn(
                      "rounded border px-2.5 py-1 text-2xs font-medium transition-colors",
                      mode === m
                        ? m === "keywords"
                          ? "border-warn-dim bg-warn/10 text-warn"
                          : "border-ok-dim bg-ok/10 text-ok"
                        : "border-line bg-surface-2 text-ink-muted hover:text-ink",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {mode === "keywords" && (
                <input
                  value={keywords}
                  onChange={(e) => setKeywords(e.target.value)}
                  className="w-full rounded border border-line bg-surface-2 px-3 py-2 text-xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                  placeholder="regenerative braking, kinetic energy (comma-separated)"
                />
              )}
              {mode === "chunks" && (
                <div className="rounded border border-line bg-surface-1">
                  <input
                    value={chunkFilter}
                    onChange={(e) => setChunkFilter(e.target.value)}
                    className="w-full border-b border-line bg-transparent px-3 py-2 text-xs text-ink placeholder:text-ink-faint focus:outline-none"
                    placeholder="Filter chunks by text / section / document…"
                  />
                  <div className="max-h-64 overflow-y-auto">
                    {filteredChunks.map((c) => (
                      <label
                        key={c.id}
                        className="flex cursor-pointer items-start gap-2 border-b border-line/50 px-3 py-1.5 text-xs hover:bg-surface-2"
                      >
                        <input
                          type="checkbox"
                          checked={selectedChunkIds.includes(c.id)}
                          onChange={() => setSelectedChunkIds((l) => toggle(l, c.id))}
                          className="mt-0.5 accent-[var(--accent)]"
                        />
                        <span className="min-w-0">
                          <span className="block truncate text-ink-muted">{c.text}</span>
                          <span className="text-2xs text-ink-faint">
                            {c.document_title ?? c.document_id} · {c.section_path ?? c.section ?? "—"}
                            {c.page != null ? ` · p.${c.page}` : ""}
                          </span>
                        </span>
                      </label>
                    ))}
                    {filteredChunks.length === 0 && (
                      <div className="px-3 py-4 text-center text-2xs text-ink-faint">No chunks match.</div>
                    )}
                  </div>
                  <div className="px-3 py-1.5 text-2xs text-ink-faint">
                    {selectedChunkIds.length} selected (showing first {filteredChunks.length} of {fmtCount(chunks.length)}
                    {chunks.length >= 500 ? "+" : ""})
                  </div>
                </div>
              )}
              {mode === "documents" && (
                <div className="max-h-48 overflow-y-auto rounded border border-line bg-surface-1">
                  {documents.map((d) => (
                    <label key={d.id} className="flex cursor-pointer items-center gap-2 border-b border-line/50 px-3 py-1.5 text-xs hover:bg-surface-2">
                      <input
                        type="checkbox"
                        checked={selectedDocIds.includes(d.id)}
                        onChange={() => setSelectedDocIds((l) => toggle(l, d.id))}
                        className="accent-[var(--accent)]"
                      />
                      <span className="truncate text-ink-muted">{d.title ?? d.url}</span>
                    </label>
                  ))}
                  {documents.length === 0 && (
                    <div className="px-3 py-4 text-center text-2xs text-ink-faint">No documents ingested yet.</div>
                  )}
                </div>
              )}

              <input
                value={gtNotes}
                onChange={(e) => setGtNotes(e.target.value)}
                className="w-full rounded border border-line bg-surface-2 px-3 py-2 text-xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                placeholder="Provenance note — where does this ground truth come from?"
              />

              <div className="flex items-center justify-between">
                <span className="text-2xs text-ink-faint">
                  {editingId
                    ? "Editing an APPROVED question creates a new DRAFT revision; FROZEN questions cannot be revised."
                    : "New questions start as DRAFT."}
                </span>
                <Button type="submit" variant="primary" size="sm">
                  {editingId ? "Save revision" : "Add question"}
                </Button>
              </div>
            </form>
          </Panel>

          <Panel>
            <PanelHeader
              title={`Questions (${questions.length})`}
              right={
                <input
                  value={reviewer}
                  onChange={(e) => setReviewer(e.target.value)}
                  className="w-40 rounded border border-line bg-surface-2 px-2 py-1 text-2xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                  placeholder="Acting as reviewer…"
                />
              }
            />
            <div>
              {questions.length === 0 ? (
                <EmptyState
                  icon={<GitCommitVertical className="h-5 w-5" />}
                  title="No evaluation questions"
                  hint="Author the first question above. Keyword-only questions are excluded from strict runs."
                />
              ) : (
                questions.map((q) => {
                  const isOpen = expanded === q.id;
                  const editable = q.status === "DRAFT" || q.status === "REVIEW";
                  return (
                    <div key={q.id} className="border-b border-line/50 last:border-0">
                      <button
                        onClick={() => setExpanded(isOpen ? null : q.id)}
                        className="flex w-full items-start gap-2 px-4 py-2.5 text-left hover:bg-surface-2"
                      >
                        {isOpen
                          ? <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" />
                          : <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" />}
                        <span className="min-w-0 flex-1">
                          <span className="block text-xs text-ink">{q.question}</span>
                          <span className="mt-0.5 flex flex-wrap items-center gap-2 text-2xs text-ink-faint">
                            <Badge tone={LIFECYCLE_TONE[q.status]}>{q.status}</Badge>
                            {q.expected_chunk_ids.length > 0 && <span>{q.expected_chunk_ids.length} chunks</span>}
                            {q.expected_document_ids.length > 0 && <span>{q.expected_document_ids.length} docs</span>}
                            {q.expected_chunk_ids.length === 0 && q.expected_document_ids.length === 0 && (
                              <span className="text-warn">keywords only — diagnostic</span>
                            )}
                            {q.revision > 1 && <span>rev {q.revision}{q.supersedes ? ` · supersedes ${q.supersedes.slice(0, 10)}…` : ""}</span>}
                            {q.reviewer && <span>by {q.reviewer}</span>}
                          </span>
                        </span>
                      </button>
                      {isOpen && (
                        <div className="space-y-3 px-4 pb-4 pl-10">
                          {q.notes && <p className="text-2xs leading-relaxed text-ink-muted">{q.notes}</p>}
                          {q.expected_chunk_ids.length > 0 && (
                            <div>
                              <div className="section-label mb-1">Expected chunks</div>
                              <ul className="space-y-1">
                                {q.expected_chunk_ids.map((cid) => (
                                  <li key={cid} className="rounded border border-line bg-surface-2 px-2 py-1 text-2xs text-ink-muted">
                                    <span className="text-ink-faint">{cid.slice(0, 12)}…</span>{" "}
                                    {chunkById.get(cid)?.text.slice(0, 120) ?? <span className="text-warn">chunk not in first 500 — check by ID</span>}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {q.expected_document_ids.length > 0 && (
                            <div>
                              <div className="section-label mb-1">Expected documents</div>
                              <ul className="space-y-1">
                                {q.expected_document_ids.map((did) => (
                                  <li key={did} className="text-2xs text-ink-muted">{docTitle(did)}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {q.expected_keywords.length > 0 && (
                            <div className="text-2xs text-ink-muted">
                              <span className="section-label mb-1 block">Keywords (diagnostic)</span>
                              {q.expected_keywords.join(", ")}
                            </div>
                          )}
                          <div className="flex flex-wrap items-center gap-1.5 border-t border-line/50 pt-2.5">
                            {editable && (
                              <>
                                <Button size="sm" variant="subtle" onClick={() => setStatus(q.id, "REVIEW")}>Send to review</Button>
                                <Button size="sm" variant="primary" onClick={() => setStatus(q.id, "APPROVED")}>Approve</Button>
                                <Button size="sm" variant="ghost" onClick={() => startEdit(q)}>Edit</Button>
                              </>
                            )}
                            {(q.status === "APPROVED" || q.status === "REVIEW" || q.status === "DRAFT") && (
                              <Button size="sm" variant="outline" onClick={() => setStatus(q.id, "FROZEN")}>
                                <Lock className="h-3 w-3" /> Freeze
                              </Button>
                            )}
                            {q.status !== "FROZEN" && (
                              <Button size="sm" variant="danger" onClick={() => del(q.id)}>Delete</Button>
                            )}
                            {q.status === "FROZEN" && (
                              <span className="flex items-center gap-1 text-2xs text-ink-faint">
                                <Lock className="h-3 w-3" /> frozen — immutable; revise via a new benchmark version
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </Panel>
        </div>

        {/* ------------------------------ Right: versions + runs ------------------------------ */}
        <div className="space-y-4">
          <Panel>
            <PanelHeader
              title="Benchmark versions"
              right={
                <span className="flex items-center gap-1 text-2xs text-ink-faint">
                  <ShieldCheck className="h-3 w-3" /> only FROZEN runs officially
                </span>
              }
            />
            <form onSubmit={createVersion} className="flex flex-wrap gap-2 border-b border-line p-3">
              <input
                value={versionName}
                onChange={(e) => setVersionName(e.target.value)}
                className="min-w-0 flex-1 rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                placeholder="version id — e.g. automobile-engineering-v2"
              />
              <input
                value={versionLabel}
                onChange={(e) => setVersionLabel(e.target.value)}
                className="min-w-0 flex-1 rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                placeholder="label (optional)"
              />
              <Button type="submit" size="sm">Snapshot all questions</Button>
            </form>
            <div>
              {versions.length === 0 ? (
                <div className="px-4 py-5 text-center text-2xs text-ink-faint">
                  Snapshot the reviewed questions into a version, then freeze it.
                </div>
              ) : (
                versions.map((v) => (
                  <div key={v.id} className="border-b border-line/50 px-4 py-2.5 last:border-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="min-w-0">
                        <span className="block truncate font-mono text-xs text-ink">{v.version}</span>
                        <span className="text-2xs text-ink-faint">
                          {v.question_ids.length} questions · {new Date(v.created_at).toLocaleDateString()}
                          {v.frozen_at ? ` · frozen ${new Date(v.frozen_at).toLocaleDateString()}` : ""}
                        </span>
                      </span>
                      <span className="flex items-center gap-1.5">
                        <Badge tone={v.status === "FROZEN" ? "accent" : v.status === "APPROVED" ? "ok" : "neutral"}>
                          {v.status}
                        </Badge>
                        {v.status !== "FROZEN" && (
                          <>
                            <Button size="sm" variant="subtle" onClick={() => freezeVersion(v)}>Freeze</Button>
                            <Button size="sm" variant="danger" onClick={() => deleteVersion(v)}>Delete</Button>
                          </>
                        )}
                      </span>
                    </div>
                    {v.notes && <p className="mt-1 text-2xs text-ink-faint">{v.notes}</p>}
                  </div>
                ))
              )}
            </div>
          </Panel>

          <Panel>
            <PanelHeader title="Run evaluation" />
            <div className="space-y-3 p-4 pt-3">
              <div className="grid grid-cols-2 gap-2">
                <label className="block">
                  <span className="section-label mb-1 block">Top K</span>
                  <input
                    type="number" min={1} max={50} value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink focus:border-line-focus focus:outline-none"
                  />
                </label>
                <label className="block">
                  <span className="section-label mb-1 block">Benchmark version</span>
                  <select
                    value={runVersion}
                    onChange={(e) => setRunVersion(e.target.value)}
                    className="w-full rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink focus:border-line-focus focus:outline-none"
                  >
                    <option value="">— live questions (working set) —</option>
                    {frozenVersions.map((v) => (
                      <option key={v.id} value={v.id}>{v.version} (frozen)</option>
                    ))}
                  </select>
                </label>
              </div>
              <input
                value={runLabel}
                onChange={(e) => setRunLabel(e.target.value)}
                className="w-full rounded border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink placeholder:text-ink-faint focus:border-line-focus focus:outline-none"
                placeholder="Run label, e.g. baseline k=5 strict"
              />
              <label className="flex items-center gap-2 text-2xs text-ink-muted">
                <input
                  type="checkbox" checked={diagnosticMode}
                  onChange={(e) => setDiagnosticMode(e.target.checked)}
                  className="accent-[var(--accent)]"
                />
                Diagnostic mode — score keyword-only questions with the disclosed heuristic
              </label>
              <Button variant="primary" loading={busy} onClick={runEval} className="w-full">
                <RefreshCw className="h-3 w-3" /> Run evaluation
              </Button>
              <p className="text-2xs leading-relaxed text-ink-faint">
                Strict runs skip keyword-only questions. Frozen-version runs score the immutable
                snapshot, never the live working set.
              </p>
            </div>
          </Panel>

          {latestRun && (
            <Panel>
              <PanelHeader
                title="Latest run"
                right={
                  <span className="font-mono text-2xs text-ink-faint">
                    {latestRun.config.benchmark_version
                      ? frozenVersions.find((v) => v.id === latestRun.config.benchmark_version)?.version ?? "frozen version"
                      : "live questions"}
                  </span>
                }
              />
              <div className="grid grid-cols-4 gap-3 border-b border-line px-4 py-3">
                <Metric label={`Recall@${latestRun.config.top_k}`} value={fmt(latestRun.aggregate.recall_at_k)} tone="accent" size="lg" />
                <Metric label="MRR" value={fmt(latestRun.aggregate.mrr)} size="lg" />
                <Metric label="NDCG" value={fmt(latestRun.aggregate.ndcg)} size="lg" />
                <Metric label={`P@${latestRun.config.top_k}`} value={fmt(latestRun.aggregate.precision_at_k)} size="lg" />
              </div>
              <div className="space-y-1 px-4 py-3 text-2xs text-ink-muted">
                <div className="flex items-center gap-1.5">
                  <StatusDot tone={latestRun.aggregate.strict_mode ? "ok" : "warn"} />
                  {latestRun.aggregate.strict_mode
                    ? "Strict mode — explicit ground truth only"
                    : "Diagnostic run — keyword heuristic disclosed"}
                </div>
                <div>
                  {latestRun.aggregate.questions_with_explicit_gt} explicit ·{" "}
                  {latestRun.aggregate.questions_with_keyword_fallback} keyword ·{" "}
                  {latestRun.aggregate.questions_skipped_no_gt} skipped
                </div>
                {latestRun.aggregate.notes && <div className="text-ink-faint">{latestRun.aggregate.notes}</div>}
              </div>
              <div className="border-t border-line">
                {latestRun.per_question.map((pq) => {
                  const q = questions.find((x) => x.id === pq.question_id);
                  return (
                    <div key={pq.question_id} className="border-b border-line/50 px-4 py-2 last:border-0">
                      <div className="flex items-center justify-between gap-3">
                        <span className="min-w-0 flex-1 truncate text-xs text-ink-muted">{pq.question}</span>
                        <span className="flex shrink-0 items-center gap-2 font-mono text-2xs text-ink-faint">
                          {pq.recall_at_k != null ? (
                            <>
                              <span>R {pq.recall_at_k.toFixed(2)}</span>
                              <span>MRR {pq.mrr?.toFixed(2) ?? "—"}</span>
                            </>
                          ) : (
                            <span className="text-warn">skipped</span>
                          )}
                        </span>
                      </div>
                      {pq.note && (
                        <div className="mt-1 text-2xs text-ink-faint">{pq.note}</div>
                      )}
                      {!pq.note && q?.status && (
                        <div className="mt-1 flex items-center gap-1.5 text-2xs text-ink-faint">
                          <Badge tone={LIFECYCLE_TONE[q.status]}>{q.status}</Badge>
                        </div>
                      )}
                      {pq.recall_at_k != null && (
                        <ScoreBar value={pq.recall_at_k} tone={pq.recall_at_k >= 1 ? "ok" : "accent"} className="mt-1.5" />
                      )}
                    </div>
                  );
                })}
              </div>
            </Panel>
          )}

          {runs.length > 1 && (
            <Panel>
              <PanelHeader title={`Run history (${runs.length})`} />
              <div>
                {runs.slice(1, 8).map((r) => (
                  <div key={r.id} className="flex items-center justify-between gap-3 border-b border-line/50 px-4 py-2 text-2xs last:border-0">
                    <span className="min-w-0">
                      <span className="block truncate text-ink-muted">
                        {r.aggregate.run_label || "unlabelled run"}
                        {r.config.benchmark_version ? " · frozen benchmark" : ""}
                      </span>
                      <span className="text-ink-faint">{new Date(r.started_at).toLocaleString()}</span>
                    </span>
                    <span className="shrink-0 font-mono text-ink-faint">
                      R@{r.config.top_k} {fmt(r.aggregate.recall_at_k)} · MRR {fmt(r.aggregate.mrr)}
                    </span>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
