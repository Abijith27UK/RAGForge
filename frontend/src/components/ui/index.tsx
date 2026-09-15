"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn, fmtCount } from "@/lib/utils";

/* ---------------------------------------------------------------- Button */

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost" | "outline" | "danger" | "subtle";
  size?: "sm" | "md";
  loading?: boolean;
};

export function Button({ variant = "outline", size = "md", loading, className, children, disabled, ...rest }: ButtonProps) {
  return (
    <button
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded border font-medium transition-all duration-150",
        "focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-45",
        size === "sm" ? "h-7 px-2.5 text-xs" : "h-8 px-3 text-xs",
        {
          primary: "border-accent-dim bg-accent/15 text-accent-soft hover:bg-accent/25 active:bg-accent/30",
          outline: "border-line-strong bg-surface-2 text-ink hover:border-line-focus hover:bg-surface-3 active:bg-surface-4",
          subtle: "border-transparent bg-surface-2 text-ink-muted hover:bg-surface-3 hover:text-ink",
          ghost: "border-transparent text-ink-muted hover:bg-surface-3 hover:text-ink",
          danger: "border-bad/40 bg-bad/10 text-bad hover:bg-bad/20",
        }[variant],
        className
      )}
      {...rest}
    >
      {loading && (
        <span className="h-3 w-3 animate-spin rounded-full border border-current border-t-transparent" aria-hidden />
      )}
      {children}
    </button>
  );
}

/* ----------------------------------------------------------------- Badge */

const BADGE_TONES = {
  neutral: "border-line-strong bg-surface-3 text-ink-muted",
  accent: "border-accent-dim bg-accent/10 text-accent-soft",
  violet: "border-violet-dim bg-violet/10 text-violet",
  ok: "border-ok-dim bg-ok/10 text-ok",
  warn: "border-warn-dim bg-warn/10 text-warn",
  bad: "border-bad-dim bg-bad/10 text-bad",
} as const;

export type BadgeTone = keyof typeof BADGE_TONES;

export function Badge({ tone = "neutral", className, children }: { tone?: BadgeTone; className?: string; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-medium uppercase tracking-wider",
        BADGE_TONES[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

/* ------------------------------------------------------------- StatusDot */

export function StatusDot({ tone, pulse, className }: { tone: "ok" | "warn" | "bad" | "neutral" | "accent"; pulse?: boolean; className?: string }) {
  const color = { ok: "bg-ok", warn: "bg-warn", bad: "bg-bad", neutral: "bg-ink-ghost", accent: "bg-accent" }[tone];
  return (
    <span className={cn("relative inline-flex h-1.5 w-1.5 shrink-0", className)}>
      {pulse && <span className={cn("absolute inset-0 animate-pulse-dot rounded-full", color)} aria-hidden />}
      <span className={cn("relative inline-flex h-1.5 w-1.5 rounded-full", color)} />
    </span>
  );
}

/* ------------------------------------------------------------ StatusPill */

const PILL_MAP: Record<string, { tone: BadgeTone; label?: string }> = {
  ready: { tone: "ok" },
  analyzed: { tone: "accent" },
  draft: { tone: "neutral" },
  building: { tone: "warn" },
  needs_review: { tone: "warn", label: "needs review" },
  error: { tone: "bad" },
  failed: { tone: "bad" },
  done: { tone: "ok" },
  completed: { tone: "ok" },
  running: { tone: "warn" },
  pending: { tone: "neutral" },
  PENDING: { tone: "neutral" },
  ACCEPT: { tone: "ok" },
  REVIEW: { tone: "warn" },
  REJECT: { tone: "bad" },
};

export function StatusPill({ status, className }: { status: string; className?: string }) {
  const { tone, label } = PILL_MAP[status] ?? { tone: "neutral" as BadgeTone };
  const live = status === "running" || status === "building";
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <StatusDot tone={tone === "neutral" ? "neutral" : (tone as "ok" | "warn" | "bad" | "accent")} pulse={live} />
      <span className={cn("text-2xs font-medium uppercase tracking-wider", {
        "text-ok": tone === "ok",
        "text-warn": tone === "warn",
        "text-bad": tone === "bad",
        "text-accent-soft": tone === "accent",
        "text-ink-faint": tone === "neutral",
      })}>
        {label ?? status.replace(/_/g, " ")}
      </span>
    </span>
  );
}

/* ----------------------------------------------------------------- Panel */

export function Panel({ className, children, ...rest }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-lg border border-line bg-surface-2 shadow-panel transition-colors duration-150",
        className
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function PanelHeader({ title, right, className }: { title: React.ReactNode; right?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-center justify-between border-b border-line px-4 py-2.5", className)}>
      <span className="section-label">{title}</span>
      {right}
    </div>
  );
}

/* ---------------------------------------------------------------- Metric */

export function Metric({
  label,
  value,
  hint,
  tone = "default",
  size = "sm",
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: "default" | "accent" | "ok" | "warn";
  size?: "sm" | "lg";
}) {
  return (
    <div>
      <div className="section-label mb-1">{label}</div>
      <div
        className={cn(
          "data-value font-medium",
          size === "lg" ? "text-3xl leading-9" : "text-lg leading-6",
          tone === "accent" && "text-accent-soft",
          tone === "ok" && "text-ok",
          tone === "warn" && "text-warn"
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 text-2xs text-ink-faint">{hint}</div>}
    </div>
  );
}

/* -------------------------------------------------------------- ScoreBar */

/** Animated horizontal score bar. Colors map to the semantic palette. */
export function ScoreBar({
  value,
  max = 1,
  tone = "accent",
  className,
  delayMs = 0,
}: {
  value: number;
  max?: number;
  tone?: "accent" | "ok" | "warn" | "bad" | "violet" | "ink";
  className?: string;
  delayMs?: number;
}) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(Math.max(0, Math.min(1, value / max)) * 100), delayMs);
    return () => clearTimeout(t);
  }, [value, max, delayMs]);
  const bg = {
    accent: "bg-accent/70",
    ok: "bg-ok/70",
    warn: "bg-warn/70",
    bad: "bg-bad/70",
    violet: "bg-violet/70",
    ink: "bg-ink-faint",
  }[tone];
  return (
    <div className={cn("h-1 w-full overflow-hidden rounded-sm bg-surface-4", className)}>
      <div
        className={cn("h-full rounded-sm transition-[width] duration-700 ease-out", bg)}
        style={{ width: `${w}%` }}
      />
    </div>
  );
}

/* ------------------------------------------------------------- Sparkline */

/** Minimal SVG sparkline; draws from real values only. */
export function Sparkline({ values, className, tone = "#38BDF8" }: { values: number[]; className?: string; tone?: string }) {
  if (values.length < 2) return null;
  const W = 120;
  const H = 28;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - 3 - ((v - min) / span) * (H - 6);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={cn("h-7 w-[120px]", className)} aria-hidden preserveAspectRatio="none">
      <polyline points={pts.join(" ")} fill="none" stroke={tone} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" opacity="0.85" />
      <circle cx={pts[pts.length - 1].split(",")[0]} cy={pts[pts.length - 1].split(",")[1]} r="2" fill={tone} />
    </svg>
  );
}

/* ------------------------------------------------------------ CopyButton */

export function CopyButton({ text, className, label }: { text: string; className?: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  return (
    <button
      type="button"
      title="Copy to clipboard"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          if (timer.current) clearTimeout(timer.current);
          timer.current = setTimeout(() => setCopied(false), 1200);
        } catch { /* clipboard unavailable */ }
      }}
      className={cn(
        "inline-flex h-6 w-6 items-center justify-center rounded border border-transparent text-ink-faint transition-colors hover:border-line-strong hover:bg-surface-3 hover:text-ink",
        className
      )}
    >
      {copied ? <Check className="h-3.5 w-3.5 text-ok" /> : <Copy className="h-3.5 w-3.5" />}
      {label && <span className="ml-1 text-2xs">{copied ? "Copied" : label}</span>}
    </button>
  );
}

/* -------------------------------------------------------------- Skeleton */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded", className)} />;
}

/* ------------------------------------------------------------ EmptyState */

export function EmptyState({ icon, title, hint, action }: { icon?: React.ReactNode; title: string; hint?: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line-strong px-6 py-12 text-center">
      {icon && <div className="text-ink-ghost">{icon}</div>}
      <div className="text-sm text-ink-muted">{title}</div>
      {hint && <div className="max-w-md text-xs leading-5 text-ink-faint">{hint}</div>}
      {action}
    </div>
  );
}

/* -------------------------------------------------------------- Progress */

export function Progress({ value, tone = "accent", className }: { value: number; tone?: "accent" | "ok" | "warn"; className?: string }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const bg = { accent: "bg-accent/70", ok: "bg-ok/70", warn: "bg-warn/70" }[tone];
  return (
    <div className={cn("h-1 w-full overflow-hidden rounded-sm bg-surface-4", className)}>
      <div className={cn("h-full transition-[width] duration-700 ease-out", bg)} style={{ width: `${pct}%` }} />
    </div>
  );
}

/** Animated integer counter (for stat counts, no fabrication — starts at 0, ends at the real value). */
export function Counter({ value, className }: { value: number; className?: string }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    const dur = 600;
    const t0 = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / dur);
      setV(Math.round(value * (1 - Math.pow(1 - p, 3))));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return <span className={cn("data-value", className)}>{fmtCount(v)}</span>;
}
