"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { ChevronDown, FileSearch } from "lucide-react";
import { api, Source } from "@/lib/api";
import {
  Badge, Button, EmptyState, Panel, PanelHeader, Skeleton, StatusPill,
} from "@/components/ui";
import { cn, fmtScore } from "@/lib/utils";

export default function SourcesPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [sources, setSources] = useState<Source[] | null>(null);
  const [provider, setProvider] = useState("user-url");
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = useCallback(() => {
    api.listSources(kbId).then(setSources).catch((e) => { setError(String(e.message)); setSources([]); });
  }, [kbId]);
  useEffect(load, [load]);

  async function discover(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.discoverSources(kbId, { provider, query, limit: 8 });
      setQuery("");
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  async function decide(sourceId: string, decision: string) {
    try {
      await api.setDecision(kbId, sourceId, decision);
      load();
    } catch (err: unknown) {
      setError(String((err as Error).message));
    }
  }

  // Ranked by real composite score where present; unscored last.
  const ranked = useMemo(() => {
    if (!sources) return [];
    return [...sources].sort((a, b) => (b.trust_score ?? -1) - (a.trust_score ?? -1));
  }, [sources]);

  const scatterPoints = useMemo(
    () =>
      ranked
        .map((s) => ({
          s,
          relevance: s.quality?.signals?.relevance ?? null,
          authority: s.quality?.signals?.authority ?? null,
        }))
        .filter((p) => p.relevance != null && p.authority != null) as
        { s: Source; relevance: number; authority: number }[],
    [ranked]
  );

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-2">
        <div className="section-label mb-2">Research</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Source Intelligence</h1>
        <p className="mt-1 max-w-2xl text-xs leading-4 text-ink-faint">
          Candidates receive an automated, explainable quality assessment. It is an estimate, not
          ground truth — review the evidence and override decisions.
        </p>
      </div>

      {/* Discovery */}
      <form onSubmit={discover} className="my-4 flex flex-wrap gap-2">
        <select
          value={provider} onChange={(e) => setProvider(e.target.value)}
          className="h-8 rounded border border-line-strong bg-surface-2 px-2 text-xs text-ink"
        >
          <option value="user-url">URLs (user provided)</option>
          <option value="arxiv">arXiv search</option>
        </select>
        <input
          value={query} onChange={(e) => setQuery(e.target.value)} required
          className="h-8 min-w-0 flex-1 rounded border border-line-strong bg-surface-2 px-2.5 text-xs text-ink placeholder:text-ink-ghost focus:border-line-focus focus:outline-none"
          placeholder={provider === "user-url" ? "https://example.com/doc.pdf, https://another.org/page" : "electric vehicle battery management"}
        />
        <Button type="submit" variant="primary" size="sm" loading={busy}>Discover</Button>
      </form>

      {error && <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>}

      {!sources && <Skeleton className="h-72 w-full" />}

      {sources && sources.length === 0 && (
        <EmptyState icon={<FileSearch className="h-8 w-8" />} title="No sources yet"
          hint="Discover candidate sources above; each will receive an explainable quality assessment." />
      )}

      {sources && sources.length > 0 && (
        <div className="space-y-4">
          {/* Quality map */}
          {scatterPoints.length >= 3 && (
            <Panel>
              <PanelHeader title="Source quality map" right={<span className="text-2xs text-ink-faint">authority × relevance — click a point</span>} />
              <div className="p-4">
                <ScatterMap points={scatterPoints} selectedId={selectedId} onSelect={setSelectedId} />
              </div>
            </Panel>
          )}

          {/* Ranking */}
          <Panel>
            <PanelHeader title="Candidate ranking" right={<span className="data-value text-2xs text-ink-faint">{sources.length} sources</span>} />
            <div className="divide-y divide-line/60">
              {ranked.map((s, i) => (
                <div key={s.id}>
                  <button
                    onClick={() => setExpanded(expanded === s.id ? null : s.id)}
                    className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-surface-3"
                  >
                    <span className="data-value w-6 shrink-0 text-2xs text-ink-ghost">#{String(i + 1).padStart(2, "0")}</span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="truncate text-xs font-medium text-ink">{s.title ?? s.url}</span>
                        <StatusPill status={s.decision} />
                      </span>
                      <span className="mt-0.5 block truncate text-2xs text-ink-faint">{s.url}</span>
                    </span>
                    <span className="hidden w-40 shrink-0 flex-col gap-1 sm:flex">
                      <SignalBar label="relevance" value={s.quality?.signals?.relevance} tone="accent" />
                      <SignalBar label="authority" value={s.quality?.signals?.authority} tone="violet" />
                    </span>
                    <span className="data-value w-12 shrink-0 text-right text-sm text-ink">
                      {s.trust_score != null ? fmtScore(s.trust_score, 2) : "—"}
                    </span>
                    <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-ink-ghost transition-transform", expanded === s.id && "rotate-180")} />
                  </button>

                  {expanded === s.id && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="overflow-hidden border-t border-line/60 bg-surface-1">
                      <div className="space-y-3 px-4 py-3">
                        <div className="grid grid-cols-2 gap-x-6 gap-y-2 md:grid-cols-4">
                          <SignalBar label="relevance" value={s.quality?.signals?.relevance} tone="accent" />
                          <SignalBar label="authority" value={s.quality?.signals?.authority} tone="violet" />
                          <SignalBar label="recency" value={s.quality?.signals?.recency} tone="ink" />
                          <SignalBar label="source type" value={s.quality?.signals?.source_type} tone="ink" />
                          <SignalBar label="accessibility" value={s.quality?.signals?.accessibility} tone="ok" />
                          <SignalBar label="duplication" value={s.quality?.signals?.duplication} tone="warn" />
                          <SignalBar label="evidence" value={s.quality?.signals?.evidence_quality} tone="ok" />
                          <div>
                            <div className="section-label mb-0.5">composite</div>
                            <div className="data-value text-lg text-ink">{s.trust_score != null ? fmtScore(s.trust_score) : "—"}</div>
                          </div>
                        </div>

                        {s.quality?.reasons?.length ? (
                          <div>
                            <div className="section-label mb-1">Why this assessment</div>
                            <ul className="space-y-0.5">
                              {s.quality.reasons.map((r, j) => (
                                <li key={j} className="text-2xs leading-4 text-ink-muted">· {r}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {s.quality?.warnings?.length ? (
                          <div>
                            <div className="section-label mb-1 text-warn">Warnings</div>
                            <ul className="space-y-0.5">
                              {s.quality.warnings.map((w, j) => (
                                <li key={j} className="text-2xs leading-4 text-warn">· {w}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {s.notes && <p className="text-2xs text-ink-faint">notes: {s.notes}</p>}

                        <div className="flex items-center gap-1.5 pt-1">
                          <span className="section-label mr-1">Decision</span>
                          {["ACCEPT", "REVIEW", "REJECT"].map((d) => (
                            <Button key={d} size="sm" variant={s.decision === d ? "primary" : "outline"} onClick={() => decide(s.id, d)}>
                              {d}
                            </Button>
                          ))}
                          <span className="ml-2 text-2xs text-ink-faint">
                            assessed_by: {s.quality?.assessed_by ?? "—"}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}

function SignalBar({ label, value, tone }: { label: string; value: number | null | undefined; tone: "accent" | "violet" | "ok" | "warn" | "ink" }) {
  const v = value ?? 0;
  const bg = {
    accent: "bg-accent/70", violet: "bg-violet/70", ok: "bg-ok/70", warn: "bg-warn/70", ink: "bg-ink-faint",
  }[tone];
  return (
    <div>
      <div className="mb-0.5 flex items-baseline justify-between gap-2">
        <span className="section-label">{label}</span>
        <span className="data-value text-2xs text-ink-muted">{value != null ? fmtScore(value, 2) : "—"}</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-sm bg-surface-4">
        <motion.div
          className={cn("h-full rounded-sm", bg)}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

function ScatterMap({
  points, selectedId, onSelect,
}: {
  points: { s: Source; relevance: number; authority: number }[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const W = 640, H = 240, PAD = 34;
  const px = (v: number) => PAD + v * (W - PAD * 2);
  const py = (v: number) => H - PAD - v * (H - PAD * 2);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-60 w-full" role="img" aria-label="Authority vs relevance scatter">
      {/* grid */}
      {[0.25, 0.5, 0.75].map((g) => (
        <g key={g}>
          <line x1={px(g)} y1={H - PAD} x2={px(g)} y2={PAD} stroke="#1C2431" strokeDasharray="3 4" />
          <line x1={PAD} y1={py(g)} x2={W - PAD} y2={py(g)} stroke="#1C2431" strokeDasharray="3 4" />
        </g>
      ))}
      {/* axes */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#2A3546" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#2A3546" />
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="9" className="fill-[#5D6B80]">RELEVANCE →</text>
      <text x={12} y={H / 2} textAnchor="middle" fontSize="9" className="fill-[#5D6B80]" transform={`rotate(-90 12 ${H / 2})`}>AUTHORITY →</text>
      {/* quadrant hint */}
      <text x={px(0.82)} y={py(0.9)} textAnchor="middle" fontSize="8" className="fill-[#3D4859]">high authority · high relevance</text>
      {/* points */}
      {points.map(({ s, relevance, authority }) => {
        const sel = s.id === selectedId;
        const decisionTone =
          s.decision === "ACCEPT" ? "#34D399" : s.decision === "REJECT" ? "#F87171" : s.decision === "REVIEW" ? "#FBBF24" : "#5D6B80";
        return (
          <g key={s.id} className="cursor-pointer" onClick={() => onSelect(sel ? null : s.id)}>
            <motion.circle
              cx={px(relevance)} cy={py(authority)}
              r={sel ? 7 : 5}
              fill={sel ? "#38BDF8" : decisionTone}
              fillOpacity={sel ? 0.95 : 0.45}
              stroke={sel ? "#7DD3FC" : decisionTone}
              strokeWidth="1"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.35 }}
              whileHover={{ scale: 1.25 }}
            />
            <title>{s.title ?? s.url}</title>
            <text
              x={px(relevance) + 9} y={py(authority) + 3}
              fontSize="8.5"
              className={cn(sel ? "fill-[#7DD3FC]" : "fill-[#5D6B80]")}
            >
              {(s.title ?? s.url).slice(0, 24)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
