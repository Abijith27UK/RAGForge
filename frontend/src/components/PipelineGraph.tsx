"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Check, ChevronRight, CircleDashed, Database, FileSearch, GitBranch,
  ListTree, Search, Sigma, TriangleAlert, Workflow,
} from "lucide-react";
import { BuildRun } from "@/lib/api";
import { cn } from "@/lib/utils";

export type PipelineCounts = {
  documents: number | null;
  chunks: number | null;
  sourcesAccepted: number | null;
  questions: number | null;
  hasSpec: boolean;
  evals: number | null;
  hasBuildRun: boolean;
};

type Stage = {
  key: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  href?: string;
  state: "done" | "active" | "empty" | "blocked";
  detail: string;
};

export default function PipelineGraph({ counts, kbId }: { counts: PipelineCounts; kbId: string }) {
  const stages: Stage[] = [
    {
      key: "domain", label: "Domain", icon: GitBranch, href: `/knowledge-bases/${kbId}/domain`,
      state: counts.hasSpec ? "done" : "active",
      detail: counts.hasSpec ? "specification generated" : "run domain analysis",
    },
    {
      key: "sources", label: "Sources", icon: FileSearch, href: `/knowledge-bases/${kbId}/sources`,
      state: counts.sourcesAccepted == null ? "empty" : counts.sourcesAccepted > 0 ? "done" : "active",
      detail: counts.sourcesAccepted == null ? "…" : `${counts.sourcesAccepted} accepted`,
    },
    {
      key: "ingestion", label: "Ingestion", icon: Workflow, href: `/knowledge-bases/${kbId}/processing`,
      state: counts.documents == null ? "empty" : counts.documents > 0 ? "done" : counts.sourcesAccepted ? "active" : "blocked",
      detail: counts.documents == null ? "…" : `${counts.documents} documents`,
    },
    {
      key: "chunks", label: "Chunks", icon: ListTree, href: `/knowledge-bases/${kbId}/chunks`,
      state: counts.chunks == null ? "empty" : counts.chunks > 0 ? "done" : "blocked",
      detail: counts.chunks == null ? "…" : `${counts.chunks} chunks`,
    },
    {
      key: "embeddings", label: "Embeddings", icon: Database, href: `/knowledge-bases/${kbId}/processing`,
      state: counts.chunks == null ? "empty" : counts.chunks > 0 ? "done" : "blocked",
      detail: counts.chunks != null && counts.chunks > 0 ? "MiniLM 384-d vectors" : "no vectors yet",
    },
    {
      key: "vectordb", label: "Vector DB", icon: Database, href: `/knowledge-bases/${kbId}/processing`,
      state: counts.chunks == null ? "empty" : counts.chunks > 0 ? "done" : "blocked",
      detail: "Qdrant collection",
    },
    {
      key: "retrieval", label: "Retrieval", icon: Search, href: `/knowledge-bases/${kbId}/retrieval`,
      state: counts.chunks != null && counts.chunks > 0 ? "active" : "blocked",
      detail: counts.chunks != null && counts.chunks > 0 ? "dense top-k ready" : "index first",
    },
    {
      key: "evaluation", label: "Evaluation", icon: Sigma, href: `/knowledge-bases/${kbId}/evaluation`,
      state: counts.evals == null ? "empty" : counts.evals > 0 ? "done" : counts.chunks ? "active" : "blocked",
      detail: counts.evals == null ? "…" : counts.evals > 0 ? `${counts.evals} run${counts.evals === 1 ? "" : "s"}` : "no runs yet",
    },
  ];

  // Ingestion is covered by the build run (if any) — show its latest state.
  void counts.hasBuildRun;

  return (
    <div className="relative">
      {/* Horizontal scroll on narrow screens; grid on wide. */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
        {stages.map((s, i) => (
          <motion.div
            key={s.key}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: i * 0.05 }}
          >
            <StageNode stage={s} last={i === stages.length - 1} />
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function StageNode({ stage, last }: { stage: Stage; last: boolean }) {
  const Icon = stage.icon;
  const tone = {
    done: { ring: "border-ok/30 bg-ok/5", icon: "text-ok", dot: "bg-ok" },
    active: { ring: "border-accent/40 bg-accent/5", icon: "text-accent-soft", dot: "bg-accent" },
    empty: { ring: "border-line bg-surface-2", icon: "text-ink-ghost", dot: "bg-ink-ghost" },
    blocked: { ring: "border-line bg-surface-1", icon: "text-ink-ghost", dot: "bg-ink-ghost" },
  }[stage.state];

  const body = (
    <div className={cn(
      "relative flex h-full flex-col gap-2 rounded-lg border p-3 transition-colors",
      tone.ring,
      stage.href && "hover:border-line-focus hover:bg-surface-3"
    )}>
      <div className="flex items-center justify-between">
        <Icon className={cn("h-4 w-4", tone.icon)} />
        {stage.state === "done" && <Check className="h-3.5 w-3.5 text-ok" />}
        {stage.state === "active" && <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent" />}
        {stage.state === "blocked" && <TriangleAlert className="h-3.5 w-3.5 text-ink-ghost" />}
        {stage.state === "empty" && <CircleDashed className="h-3.5 w-3.5 text-ink-ghost" />}
      </div>
      <div>
        <div className="text-xs font-medium text-ink">{stage.label}</div>
        <div className="data-value mt-0.5 text-2xs text-ink-faint">{stage.detail}</div>
      </div>
      {!last && (
        <ChevronRight className="absolute -right-2 top-1/2 hidden h-3.5 w-3.5 -translate-y-1/2 text-ink-ghost xl:block" />
      )}
    </div>
  );

  return stage.href ? <Link href={stage.href}>{body}</Link> : body;
}

/** Compact build-run stage strip used on the Processing page. */
export function BuildRunStages({ run }: { run: BuildRun }) {
  return (
    <div className="divide-y divide-line/60">
      {run.stages.map((s, i) => (
        <div key={i} className="flex items-center justify-between gap-3 px-4 py-2 text-xs">
          <span className="flex items-center gap-2">
            <StageDot status={s.status} />
            <span className="text-ink">{s.stage}</span>
          </span>
          <span className="min-w-0 truncate text-right text-2xs text-ink-faint">
            {s.items_processed > 0 && <span className="data-value mr-2 text-ink-muted">{s.items_processed} items</span>}
            {s.message}
          </span>
        </div>
      ))}
    </div>
  );
}

export function StageDot({ status }: { status: string }) {
  const cls =
    status === "done" ? "bg-ok" :
    status === "error" ? "bg-bad" :
    status === "running" ? "bg-accent animate-pulse-dot" :
    "bg-ink-ghost";
  return <span className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", cls)} />;
}
