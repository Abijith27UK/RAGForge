"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { GitBranch } from "lucide-react";
import { api, DomainSpec } from "@/lib/api";
import { Badge, Button, EmptyState, Panel, PanelHeader, Skeleton } from "@/components/ui";
import { cn } from "@/lib/utils";

const CX = 300;
const CY = 260;
const RADIUS = 190;

type Node = {
  area: string;
  description?: string;
  priority?: string;
  x: number;
  y: number;
};

export default function DomainPage() {
  const { id: kbId } = useParams<{ id: string }>();
  const [spec, setSpec] = useState<DomainSpec | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api.getDomainSpec(kbId)
      .then(setSpec)
      .catch(() => setSpec(null))
      .finally(() => setLoading(false));
  }, [kbId]);

  async function analyze() {
    setBusy(true);
    setError(null);
    try {
      setSpec(await api.analyzeDomain(kbId));
      setSelected(null);
    } catch (e: unknown) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  // Radial layout over the real knowledge_requirements (fallback: subdomains).
  const nodes = useMemo<Node[]>(() => {
    const areas =
      spec && spec.knowledge_requirements.length > 0
        ? spec.knowledge_requirements.map((r) => ({ area: r.area, description: r.description, priority: r.priority }))
        : spec
        ? spec.subdomains.map((s) => ({ area: s }))
        : [];
    return areas.map((a, i) => {
      const angle = (i / Math.max(1, areas.length)) * Math.PI * 2 - Math.PI / 2;
      return { ...a, x: CX + Math.cos(angle) * RADIUS, y: CY + Math.sin(angle) * RADIUS * 0.78 };
    });
  }, [spec]);

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6 flex items-end justify-between">
        <div>
          <div className="section-label mb-2">Research</div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Domain Analysis</h1>
        </div>
        <Button variant="primary" size="sm" loading={busy} onClick={analyze}>
          {spec ? "Re-run analysis" : "Run analysis"}
        </Button>
      </div>

      {error && (
        <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>
      )}

      {loading && <Skeleton className="h-96 w-full" />}

      {!loading && !spec && (
        <EmptyState
          icon={<GitBranch className="h-8 w-8" />}
          title="No domain specification yet"
          hint="Run the analysis to generate a structured domain specification from this KB's domain, purpose, and audience."
          action={<Button variant="primary" size="sm" loading={busy} onClick={analyze}>Run analysis</Button>}
        />
      )}

      {spec && (
        <div className="space-y-4">
          {spec.is_mock && (
            <div className="rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
              Development mock output — configure an LLM provider (backend/.env) for real analysis.
              Labelled is_mock=true; all downstream data remains real.
            </div>
          )}

          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
            <span className="text-sm text-ink-muted">{spec.description}</span>
            <span className="text-2xs text-ink-faint">generated_by: {spec.generated_by}</span>
          </div>

          <Panel>
            <PanelHeader
              title="Knowledge map"
              right={<span className="text-2xs text-ink-faint">{nodes.length} requirement area{nodes.length === 1 ? "" : "s"} — click a node</span>}
            />
            <div className="overflow-x-auto p-2">
              <svg viewBox={`0 0 ${CX * 2} ${CY * 2}`} className="mx-auto block h-[440px] min-w-[560px]" role="img" aria-label="Domain knowledge map">
                {/* spokes */}
                {nodes.map((n, i) => (
                  <motion.line
                    key={n.area}
                    x1={CX} y1={CY} x2={n.x} y2={n.y}
                    stroke={selected === n.area ? "#38BDF8" : "#2A3546"}
                    strokeWidth={selected === n.area ? 1.5 : 1}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                  />
                ))}
                {/* center */}
                <g>
                  <circle cx={CX} cy={CY} r={54} fill="#131924" stroke="#3D4E66" />
                  <text x={CX} y={CY - 4} textAnchor="middle" className="fill-[#E7ECF3]" fontSize="12" fontWeight="600">
                    {spec.domain.length > 22 ? spec.domain.slice(0, 21) + "…" : spec.domain}
                  </text>
                  <text x={CX} y={CY + 12} textAnchor="middle" className="fill-[#5D6B80]" fontSize="9">
                    domain spec
                  </text>
                </g>
                {/* nodes */}
                {nodes.map((n) => {
                  const isSel = selected === n.area;
                  return (
                    <g
                      key={n.area}
                      transform={`translate(${n.x}, ${n.y})`}
                      onClick={() => setSelected(isSel ? null : n.area)}
                      className="cursor-pointer"
                    >
                      <motion.circle
                        r={isSel ? 9 : 6}
                        fill={isSel ? "#38BDF8" : "#0E131B"}
                        stroke={isSel ? "#7DD3FC" : "#2A3546"}
                        strokeWidth="1.5"
                        whileHover={{ r: 8 }}
                      />
                      <text
                        y={-14}
                        textAnchor="middle"
                        fontSize="9.5"
                        className={cn(isSel ? "fill-[#7DD3FC]" : "fill-[#93A0B4]")}
                      >
                        {n.area.length > 26 ? n.area.slice(0, 25) + "…" : n.area}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            {selected && (
              <motion.div
                initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
                className="border-t border-line px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <Badge tone="violet">{selected}</Badge>
                  {nodes.find((n) => n.area === selected)?.priority && (
                    <Badge tone="neutral">priority: {nodes.find((n) => n.area === selected)?.priority}</Badge>
                  )}
                </div>
                {nodes.find((n) => n.area === selected)?.description && (
                  <p className="mt-2 text-xs leading-5 text-ink-muted">
                    {nodes.find((n) => n.area === selected)?.description}
                  </p>
                )}
              </motion.div>
            )}
          </Panel>

          {/* Spec details */}
          <div className="grid gap-4 md:grid-cols-2">
            <SpecList title="Subdomains" items={spec.subdomains} />
            <SpecList title="Key concepts" items={spec.key_concepts} />
            <SpecList title="Entities" items={spec.entities} />
            <SpecList title="Terminology" items={spec.terminology} />
            <SpecList title="Recommended source categories" items={spec.recommended_source_categories} />
          </div>
        </div>
      )}
    </div>
  );
}

function SpecList({ title, items }: { title: string; items: string[] }) {
  if (!items || items.length === 0) return null;
  return (
    <Panel>
      <PanelHeader title={title} right={<span className="data-value text-2xs text-ink-faint">{items.length}</span>} />
      <div className="flex flex-wrap gap-1.5 p-3.5">
        {items.map((it) => (
          <Badge key={it} tone="neutral">{it}</Badge>
        ))}
      </div>
    </Panel>
  );
}
