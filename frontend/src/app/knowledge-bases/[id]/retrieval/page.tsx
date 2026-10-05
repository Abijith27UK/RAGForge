"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { ExternalLink, FileText, Search, Sparkles } from "lucide-react";
import { api, AnswerResponse, RetrievalResponse, RetrievalResult } from "@/lib/api";
import { Badge, Button, EmptyState, Panel, ScoreBar } from "@/components/ui";
import { fmtScore } from "@/lib/utils";

const STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_reranked"] as const;

type Prov = RetrievalResult["provenance"];

function provenanceFacts(p: Prov): { label: string; value: string }[] {
  const facts: { label: string; value: string }[] = [];
  if (p.document_title) facts.push({ label: "Document", value: String(p.document_title) });
  if (p.slide != null) {
    facts.push({
      label: "Slide",
      value: p.slide_title ? `${p.slide} · ${p.slide_title}` : String(p.slide),
    });
  }
  if (p.page != null) facts.push({ label: "Page", value: String(p.page) });
  if (p.section_path) facts.push({ label: "Section", value: String(p.section_path) });
  else if (p.section) facts.push({ label: "Section", value: String(p.section) });
  if (p.source_title && p.source_title !== p.document_title) {
    facts.push({ label: "Source", value: String(p.source_title) });
  }
  if (p.trust_score != null) {
    facts.push({ label: "Trust", value: `${(Number(p.trust_score) * 100).toFixed(0)}% (heuristic)` });
  }
  if (p.user_provided === true) facts.push({ label: "Origin", value: "User provided" });
  else if (p.user_provided === false) facts.push({ label: "Origin", value: "Externally discovered" });
  if (p.document_version != null) facts.push({ label: "Doc version", value: `v${p.document_version}` });
  return facts;
}

function ProvenanceTrail({ r }: { r: RetrievalResult }) {
  const p = r.provenance;
  const steps = [
    { label: "Knowledge base", value: p.kb_id != null ? String(p.kb_id) : null },
    { label: "Document", value: p.document_id != null ? String(p.document_id) : null },
    { label: "Source", value: p.source_id != null ? String(p.source_id) : p.source_url },
    {
      label: p.slide != null ? "Slide" : "Page / Section",
      value: p.slide != null ? `slide ${p.slide}` : p.page != null ? `page ${p.page}` : p.section_path ?? p.section ?? null,
    },
    { label: "Chunk", value: r.chunk_id },
  ];
  return (
    <div className="mt-3 rounded border border-line bg-surface-3 px-2.5 py-2">
      <div className="section-label mb-1.5">Provenance</div>
      <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-2xs text-ink-faint">
        {steps.map((s, i) => (
          <li key={s.label} className="flex items-center gap-1.5">
            {i > 0 && <span className="text-line-strong">→</span>}
            <span>
              <span className="text-ink-muted">{s.label}:</span>{" "}
              <code className="data-value text-ink">{s.value ?? "—"}</code>
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function RetrievalLab() {
  const { id: kbId } = useParams<{ id: string }>();
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [strategy, setStrategy] = useState<string>("dense");
  const [resp, setResp] = useState<RetrievalResponse | null>(null);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [answering, setAnswering] = useState(false);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      setResp(await api.retrieve(kbId, query, topK, strategy));
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  // Retrieval only vs Retrieval + grounded answer: same question, same
  // strategy, so the two can be compared side by side.
  async function generateAnswer() {
    setAnswering(true);
    setError(null);
    try {
      setAnswer(await api.answer(kbId, query, { strategy, topK }));
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setAnswering(false);
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
            placeholder="What is the effect of free surface on ship stability?"
          />
        </div>
        <select
          value={strategy} onChange={(e) => setStrategy(e.target.value)}
          className="h-11 rounded-lg border border-line-strong bg-surface-2 px-2 text-xs text-ink"
        >
          {STRATEGIES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select
          value={topK} onChange={(e) => setTopK(Number(e.target.value))}
          className="h-11 rounded-lg border border-line-strong bg-surface-2 px-2 text-xs text-ink"
        >
          {[3, 5, 10, 20].map((k) => <option key={k} value={k}>top {k}</option>)}
        </select>
        <Button type="submit" variant="primary" loading={busy} className="h-11 px-5">Search</Button>
      </form>

      <div className="mb-6 flex items-center gap-3">
        <Button
          type="button"
          variant="outline"
          loading={answering}
          disabled={!query}
          onClick={generateAnswer}
        >
          <Sparkles className="h-3 w-3" /> Generate grounded answer
        </Button>
        <Link href={`/knowledge-bases/${kbId}/answer`} className="text-2xs text-accent-soft hover:underline">
          open full Answer page →
        </Link>
      </div>

      <p className="mb-6 text-2xs leading-4 text-ink-faint">
        Retrieval uses the selected registry strategy (dense, bm25, hybrid or hybrid_reranked). Scores
        are real retrieval scores from the strategy that produced them — not a fabricated relevance
        percentage. Page, slide and section numbers are shown only when the source file actually
        provides them; they are never inferred.
      </p>

      {/* Retrieval only vs Retrieval + grounded answer */}
      {answer && (
        <Panel className="mb-4">
          <div className="border-b border-line px-4 py-2.5 flex items-center justify-between">
            <span className="section-label">Retrieval + grounded answer</span>
            <div className="flex items-center gap-2">
              <Badge tone={answer.status === "grounded" ? "ok" : answer.status === "partial" ? "warn" : "bad"}>
                {answer.status}
              </Badge>
              <Link
                href={`/knowledge-bases/${kbId}/answer`}
                className="text-2xs text-accent-soft hover:underline"
              >
                full trace →
              </Link>
            </div>
          </div>
          <div className="px-4 py-3">
            <p className="whitespace-pre-wrap text-xs leading-5 text-ink">{answer.answer.text}</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {answer.citations.map((c) => (
                <Badge key={c.citation_id} tone="accent">{c.citation_id}</Badge>
              ))}
              <span className="ml-auto text-2xs text-ink-faint">
                {answer.grounding_assessment.decision} · {answer.grounding_assessment.reason_code}
              </span>
            </div>
          </div>
        </Panel>
      )}

      {error && <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>}

      {resp && (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-ink-faint">
            <span>query: <span className="data-value text-ink-muted">{resp.query}</span></span>
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
                {resp.results.map((r, i) => {
                  const facts = provenanceFacts(r.provenance);
                  return (
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

                        {/* Document / slide / page / section line */}
                        {facts.length > 0 && (
                          <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1">
                            {facts.map((f) => (
                              <span key={f.label} className="inline-flex items-baseline gap-1 text-2xs">
                                <span className="text-ink-faint">{f.label}:</span>
                                <span className="text-ink">{f.value}</span>
                              </span>
                            ))}
                            {r.provenance.source_url && !String(r.provenance.source_url).startsWith("upload://") && (
                              <a
                                href={String(r.provenance.source_url)}
                                target="_blank"
                                rel="noreferrer noopener"
                                className="ml-auto inline-flex items-center gap-1 text-2xs text-accent-soft hover:underline"
                              >
                                <ExternalLink className="h-3 w-3" /> open source
                              </a>
                            )}
                          </div>
                        )}

                        <p className="whitespace-pre-wrap text-xs leading-5 text-ink">{r.text}</p>

                        <div className="mt-3 flex flex-wrap items-center gap-1.5">
                          {r.provenance.content_hash && (
                            <Badge tone="neutral">
                              <span className="normal-case tracking-normal text-ink-faint">hash:</span>{" "}
                              <code className="data-value">{String(r.provenance.content_hash).slice(0, 10)}</code>
                            </Badge>
                          )}
                          {r.provenance.chunking_strategy && (
                            <Badge tone="neutral">{String(r.provenance.chunking_strategy)}</Badge>
                          )}
                          {r.provenance.chunk_index != null && (
                            <Badge tone="neutral">chunk #{String(r.provenance.chunk_index)}</Badge>
                          )}
                          <span className="ml-auto inline-flex items-center gap-1 text-2xs text-ink-faint">
                            <FileText className="h-3 w-3" /> {r.chunk_id}
                          </span>
                        </div>

                        <ProvenanceTrail r={r} />
                      </Panel>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}
        </>
      )}
    </div>
  );
}