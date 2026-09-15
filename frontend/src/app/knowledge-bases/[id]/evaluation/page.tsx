"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { api, Chunk, Document, EvaluationQuestion, EvaluationRun } from "@/lib/api";

const fmt = (v: number | null | undefined) => (v == null ? "n/a" : (v * 100).toFixed(1) + "%");

type GroundTruthMode = "keywords" | "chunks" | "documents";

export default function EvaluationPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [questions, setQuestions] = useState<EvaluationQuestion[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);

  // Authoring state
  const [qText, setQText] = useState("");
  const [mode, setMode] = useState<GroundTruthMode>("keywords");
  const [keywords, setKeywords] = useState("");
  const [selectedChunkIds, setSelectedChunkIds] = useState<string[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [chunkFilter, setChunkFilter] = useState("");
  const [gtNotes, setGtNotes] = useState("");

  const [topK, setTopK] = useState(5);
  const [runLabel, setRunLabel] = useState("");
  const [diagnosticMode, setDiagnosticMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api.listQuestions(kbId).then(setQuestions).catch(() => null);
    api.listEvaluationRuns(kbId).then(setRuns).catch(() => null);
    api.listChunks(kbId, 500).then(setChunks).catch(() => null);
    api.listDocuments(kbId).then(setDocuments).catch(() => null);
  }, [kbId]);
  useEffect(load, [load]);

  const filteredChunks = useMemo(() => {
    const f = chunkFilter.toLowerCase().trim();
    const base = f
      ? chunks.filter(
          (c) =>
            c.text.toLowerCase().includes(f) ||
            (c.section ?? "").toLowerCase().includes(f) ||
            (c.document_title ?? "").toLowerCase().includes(f)
        )
      : chunks;
    return base.slice(0, 60);
  }, [chunks, chunkFilter]);

  const toggle = (list: string[], id: string) =>
    list.includes(id) ? list.filter((x) => x !== id) : [...list, id];

  async function addQuestion(e: React.FormEvent) {
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
    try {
      await api.addQuestion(kbId, {
        question: qText,
        expected_keywords:
          mode === "keywords" ? keywords.split(",").map((k) => k.trim()).filter(Boolean) : [],
        expected_chunk_ids: mode === "chunks" ? selectedChunkIds : [],
        expected_document_ids: mode === "documents" ? selectedDocIds : [],
        notes: gtNotes,
      });
      setQText("");
      setKeywords("");
      setSelectedChunkIds([]);
      setSelectedDocIds([]);
      setGtNotes("");
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  async function runEval() {
    setBusy(true);
    setError(null);
    try {
      await api.evaluate(kbId, topK, runLabel, diagnosticMode);
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  async function del(qId: string) {
    await api.deleteQuestion(kbId, qId);
    load();
  }

  const docTitle = (docId: string | null | undefined) =>
    documents.find((d) => d.id === docId)?.title ?? chunks.find((c) => c.document_id === docId)?.document_title ?? docId;

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-4">Evaluation</h1>

      {/* ------------------------------ Authoring ------------------------------ */}
      <form onSubmit={addQuestion} className="mb-6 p-4 rounded border border-slate-800 bg-slate-900/40 space-y-3">
        <div className="text-sm font-medium text-slate-300">Add evaluation question (author ground truth)</div>
        <input
          value={qText}
          onChange={(e) => setQText(e.target.value)}
          required
          className="w-full px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm"
          placeholder="What are the major functions of an EV battery management system?"
        />

        <div className="flex gap-2 text-xs">
          {(
            [
              ["keywords", "Keywords (heuristic)"],
              ["chunks", "Expected chunks (rigorous)"],
              ["documents", "Expected documents (doc-level)"],
            ] as [GroundTruthMode, string][]
          ).map(([m, label]) => (
            <button
              type="button"
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded border ${
                mode === m
                  ? "bg-sky-900/60 border-sky-600 text-sky-200"
                  : "bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {mode === "keywords" && (
          <div className="space-y-1">
            <input
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              className="w-full px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm"
              placeholder="Expected keywords (comma-separated) — transparent relevance heuristic, not rigorous"
            />
            <p className="text-[11px] text-slate-500">
              A retrieved chunk counts as relevant if it contains any keyword. Scores are indicative only —
              prefer chunk IDs for rigorous measurement.
            </p>
          </div>
        )}

        {mode === "chunks" && (
          <div className="space-y-2">
            <input
              value={chunkFilter}
              onChange={(e) => setChunkFilter(e.target.value)}
              className="w-full px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm"
              placeholder="Search chunks by text / section / document…"
            />
            {chunks.length === 0 && (
              <p className="text-xs text-amber-400">No chunks yet — ingest + index first to author chunk-level ground truth.</p>
            )}
            <div className="max-h-64 overflow-y-auto rounded border border-slate-800 divide-y divide-slate-800/60">
              {filteredChunks.map((c) => (
                <label key={c.id} className="flex items-start gap-2 p-2 hover:bg-slate-800/40 cursor-pointer text-xs">
                  <input
                    type="checkbox"
                    className="mt-0.5"
                    checked={selectedChunkIds.includes(c.id)}
                    onChange={() => setSelectedChunkIds((prev) => toggle(prev, c.id))}
                  />
                  <span className="min-w-0">
                    {c.section ? <span className="text-sky-400">[{c.section}] </span> : null}
                    <span className="text-slate-300">{c.text.slice(0, 140)}</span>
                    <span className="block text-slate-600 truncate">
                      {c.document_title} · {c.source_url}
                    </span>
                  </span>
                </label>
              ))}
              {filteredChunks.length === 0 && chunks.length > 0 && (
                <p className="p-2 text-xs text-slate-500">No chunks match “{chunkFilter}”.</p>
              )}
            </div>
            <p className="text-[11px] text-slate-500">{selectedChunkIds.length} chunk(s) selected (showing first 60 matches)</p>
          </div>
        )}

        {mode === "documents" && (
          <div className="space-y-2">
            {documents.length === 0 && (
              <p className="text-xs text-amber-400">No documents yet — ingest first to author document-level ground truth.</p>
            )}
            <div className="max-h-48 overflow-y-auto rounded border border-slate-800 divide-y divide-slate-800/60">
              {documents.map((d) => (
                <label key={d.id} className="flex items-center gap-2 p-2 hover:bg-slate-800/40 cursor-pointer text-xs">
                  <input
                    type="checkbox"
                    checked={selectedDocIds.includes(d.id)}
                    onChange={() => setSelectedDocIds((prev) => toggle(prev, d.id))}
                  />
                  <span className="text-slate-300 truncate">
                    {d.title} <span className="text-slate-600">· {d.source_type} · {d.url}</span>
                  </span>
                </label>
              ))}
            </div>
            <p className="text-[11px] text-slate-500">{selectedDocIds.length} document(s) selected</p>
          </div>
        )}

        <input
          value={gtNotes}
          onChange={(e) => setGtNotes(e.target.value)}
          className="w-full px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm"
          placeholder="Ground-truth provenance (required for rigorous work): who selected this evidence, from what passage/document, on what date"
        />
        <button className="px-4 py-2 rounded bg-sky-700 hover:bg-sky-600 text-white text-sm">Add question</button>
      </form>

      {/* ------------------------------ Run ------------------------------ */}
      <div className="flex items-center gap-3 mb-6">
        <select
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="px-2 py-2 rounded bg-slate-900 border border-slate-700 text-sm"
        >
          {[3, 5, 10].map((k) => (
            <option key={k} value={k}>
              k={k}
            </option>
          ))}
        </select>
        <input
          value={runLabel}
          onChange={(e) => setRunLabel(e.target.value)}
          className="px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm w-64"
          placeholder="Run label (e.g. automobile-baseline-v1)"
        />
        <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={diagnosticMode}
            onChange={(e) => setDiagnosticMode(e.target.checked)}
          />
          diagnostic mode (allow keyword heuristic)
        </label>
        <button
          onClick={runEval}
          disabled={busy || questions.length === 0}
          className="px-4 py-2 rounded bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-white text-sm"
        >
          {busy ? "Running…" : "Run evaluation"}
        </button>
        <span className="text-xs text-slate-500">
          {diagnosticMode
            ? "DIAGNOSTIC: keyword-only questions will be scored and are NOT headline metrics."
            : "Strict mode: only explicit chunk/document ground truth is scored."}
        </span>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded bg-red-900/40 border border-red-800 text-sm text-red-200">{error}</div>
      )}

      {/* ------------------------------ Questions ------------------------------ */}
      <h2 className="text-sm font-semibold text-slate-300 mb-2">Questions ({questions.length})</h2>
      <div className="space-y-2 mb-6">
        {questions.map((q) => (
          <div key={q.id} className="p-3 rounded border border-slate-800 bg-slate-900/40 text-sm flex justify-between gap-3">
            <div className="min-w-0">
              <div className="text-slate-200">{q.question}</div>
              <div className="text-xs text-slate-500 mt-1 space-x-3">
                {q.expected_chunk_ids.length > 0 && (
                  <span className="text-emerald-400">rigorous: {q.expected_chunk_ids.length} expected chunk(s)</span>
                )}
                {q.expected_document_ids.length > 0 && (
                  <span className="text-emerald-400">doc-level: {q.expected_document_ids.length} expected document(s)</span>
                )}
                {q.expected_keywords.length > 0 && <span>keywords: {q.expected_keywords.join(", ")}</span>}
                {q.notes && <span className="text-slate-400">provenance: {q.notes}</span>}
              </div>
            </div>
            <button onClick={() => del(q.id)} className="text-xs text-red-400 hover:text-red-300 shrink-0">
              delete
            </button>
          </div>
        ))}
      </div>

      {/* ------------------------------ Runs ------------------------------ */}
      <h2 className="text-sm font-semibold text-slate-300 mb-2">Evaluation runs</h2>
      {runs.length === 0 && <p className="text-slate-400 text-sm">No runs yet.</p>}
      <div className="space-y-4">
        {runs.map((run) => (
          <div key={run.id} className="p-4 rounded border border-slate-800 bg-slate-900/40">
            <div className="grid grid-cols-5 gap-3 mb-2 text-sm">
              <Metric label={`Recall@${run.config?.top_k ?? 5}`} value={fmt(run.aggregate.recall_at_k)} />
              <Metric label="Precision@K" value={fmt(run.aggregate.precision_at_k)} />
              <Metric label="MRR" value={fmt(run.aggregate.mrr)} />
              <Metric label="NDCG" value={fmt(run.aggregate.ndcg)} />
              <Metric label="Questions" value={String(run.aggregate.questions_evaluated)} />
            </div>
            <div className="text-[11px] text-slate-500 mb-2">
              explicit GT: {run.aggregate.questions_with_explicit_gt} · keyword heuristic: {run.aggregate.questions_with_keyword_fallback} · skipped: {run.aggregate.questions_skipped_no_gt}
              {run.aggregate.strict_mode ? " · strict mode" : " · DIAGNOSTIC mode"}
              {run.aggregate.run_label ? ` · label: ${run.aggregate.run_label}` : ""}
            </div>
            {(run.aggregate.doc_recall_at_k != null ||
              run.aggregate.doc_precision_at_k != null ||
              run.aggregate.doc_mrr != null ||
              run.aggregate.doc_ndcg != null) && (
              <div className="grid grid-cols-4 gap-3 mb-3 text-sm">
                <Metric label="Doc Recall@K" value={fmt(run.aggregate.doc_recall_at_k)} />
                <Metric label="Doc Precision@K" value={fmt(run.aggregate.doc_precision_at_k)} />
                <Metric label="Doc MRR" value={fmt(run.aggregate.doc_mrr)} />
                <Metric label="Doc NDCG" value={fmt(run.aggregate.doc_ndcg)} />
              </div>
            )}
            {run.aggregate.notes && (
              <div className="mb-3 p-2 rounded bg-amber-900/30 border border-amber-800/60 text-[11px] text-amber-200">
                {run.aggregate.notes}
              </div>
            )}
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 text-left">
                  <th className="py-1 pr-3 font-normal">Question</th>
                  <th className="py-1 pr-3 font-normal">R@K</th>
                  <th className="py-1 pr-3 font-normal">P@K</th>
                  <th className="py-1 pr-3 font-normal">MRR</th>
                  <th className="py-1 pr-3 font-normal">Doc R@K</th>
                  <th className="py-1 font-normal">Note</th>
                </tr>
              </thead>
              <tbody>
                {run.per_question.map((p) => (
                  <tr key={p.question_id} className="border-t border-slate-800/60">
                    <td className="py-1.5 pr-3 text-slate-300">{p.question}</td>
                    <td className="py-1.5 pr-3 text-slate-400">{fmt(p.recall_at_k)}</td>
                    <td className="py-1.5 pr-3 text-slate-400">{fmt(p.precision_at_k)}</td>
                    <td className="py-1.5 pr-3 text-slate-400">{fmt(p.mrr)}</td>
                    <td className="py-1.5 pr-3 text-slate-400">{fmt(p.doc_recall_at_k)}</td>
                    <td className="py-1.5 text-slate-500">{p.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[11px] text-slate-500 mt-2">
              backend: {run.retrieval_backend} · embedding: {run.embedding_model}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-800/60 rounded p-2">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-slate-100 font-medium">{value}</div>
    </div>
  );
}
