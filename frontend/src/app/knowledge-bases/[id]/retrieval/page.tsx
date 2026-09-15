"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Search } from "lucide-react";
import { api, RetrievalResponse } from "@/lib/api";
import { Badge, Button, EmptyState, Panel, ScoreBar } from "@/components/ui";
import { fmtScore } from "@/lib/utils";

export default function RetrievalLab() {
  const { id: kbId } = useParams<{ id: string }>();
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [resp, setResp] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResp(await api.retrieve(kbId, query, topK));
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  const maxScore = resp && resp.results.length > 0 ? Math.max(...resp.results.map((r) => r.score)) : 1;

  return (
    <div className="mx-auto max-w-4xl">
      <div className="mb-2">
        <div className="section-label mb-2">Research</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Retrieval Lab</h1>
      </div>

      <form onSubmit={search} className="mb-3 flex gap-2">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-ghost" />
          <input
            value={query} onChange={(e) => setQuery(e.target.value)} required
            className="h-11 w-full rounded-lg border border-line-strong bg-surface-2 pl-9 pr-3 text-sm text-ink placeholder:text-ink-ghost focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
            placeholder="Ask the knowledge base…"
          />
        </div>
        <select
          value={topK} onChange={(e) => setTopK(Number(e.target.value))}
          className="h-11 rounded-lg border border-line-strong bg-surface-2 px-2 text-xs text-ink"
        >
          {[3, 5, 10, 20].map((k) => <option key={k} value={k}>top {k}</option>)}
        </select>
        <Button type="submit" variant="primary" loading={busy} className="h-11 px-5">Search</Button>
      </form>

      <p className="mb-6 text-2xs leading-4 text-ink-faint">
        Dense vector retrieval via Qdrant. Scores are real cosine similarities from the vector
        store — not a fabricated relevance percentage.
      </p>

      {error && <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>}

      {resp && (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-ink-faint">
            <span>backend: <span className="data-value text-ink-muted">{resp.retrieval_backend}</span></span>
            <span>embedding: <span className="data-value text-ink-muted">{resp.embedding_model}</span></span>
            <span className="data-value">{resp.results.length} results</span>
          </div>

          {resp.error && (
            <div className="mb-4 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">{resp.error}</div>
          )}

          {resp.results.length === 0 ? (
            <EmptyState title="No results" hint="No chunks matched this query in the vector store." />
          ) : (
            <div className="space-y-2.5">
              <AnimatePresence initial={false}>
                {resp.results.map((r, i) => (
                  <motion.div
                    key={r.chunk_id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.28, delay: i * 0.05 }}
                  >
                    <Panel className="p-4">
                      <div className="mb-2 flex items-center gap-3">
                        <span className="data-value text-2xs text-ink-ghost">#{i + 1}</span>
                        <div className="min-w-0 flex-1">
                          <ScoreBar value={r.score} max={maxScore} tone="accent" delayMs={i * 60} />
                        </div>
                        <span className="data-value shrink-0 text-xs text-accent-soft">{fmtScore(r.score, 3)}</span>
                      </div>
                      <p className="whitespace-pre-wrap text-xs leading-5 text-ink">{r.text}</p>
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {Object.entries(r.provenance).map(([k, v]) =>
                          v != null && v !== "" ? (
                            <Badge key={k} tone="neutral">
                              <span className="normal-case tracking-normal text-ink-faint">{k}:</span> {String(v).slice(0, 60)}
                            </Badge>
                          ) : null
                        )}
                      </div>
                    </Panel>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </>
      )}
    </div>
  );
}
