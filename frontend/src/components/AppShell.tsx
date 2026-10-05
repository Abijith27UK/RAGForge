"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity, Boxes, ChevronRight, Database, FileSearch, Files, FlaskConical, GitBranch,
  HardDriveDownload, LayoutDashboard, ListTree, Menu, MessagesSquare, PanelLeftClose,
  PanelLeftOpen, Plus, Quote, Search, ShieldAlert, ShieldCheck, Sigma, Workflow, X,
} from "lucide-react";
import CommandPalette from "@/components/CommandPalette";
import { StatusDot } from "@/components/ui";
import { api, KnowledgeBase, SystemStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = {
  workspace: [
    { href: "/", label: "Overview", icon: LayoutDashboard, hint: null },
    { href: "/knowledge-bases", label: "Knowledge Bases", icon: Database, hint: null },
    { href: "/knowledge-bases/new", label: "Create Knowledge Base", icon: Plus, hint: null },
  ],
  research: [
    { suffix: "/domain", label: "Domain Analysis", icon: GitBranch },
    { suffix: "/sources", label: "Sources", icon: FileSearch },
    { suffix: "/corpus", label: "Corpus Command Center", icon: ShieldCheck },
    { suffix: "/documents", label: "Documents", icon: Files },
    { suffix: "/processing", label: "Processing", icon: Workflow },
    { suffix: "/chunks", label: "Chunks", icon: ListTree },
    { suffix: "/retrieval", label: "Retrieval Lab", icon: Search, hint: "⌘K then query" },
    { suffix: "/answer", label: "Answer", icon: Quote },
    { suffix: "/chat", label: "Grounded Chat", icon: MessagesSquare, hint: "inspect every claim" },
    { suffix: "/evaluation", label: "Evaluation", icon: Sigma },
    { suffix: "/answer-quality", label: "Answer Quality", icon: ShieldAlert, hint: "citations, grounding, abstention" },
    { suffix: "/reliability", label: "Reliability", icon: Activity, hint: "strategies compared, no combined score" },
  ],
  global: [{ href: "/experiments", label: "Experiments", icon: FlaskConical, hint: null }],
} as const;

const SYSTEM_ICON = { qdrant: Database, embedding: Boxes, llm: FlaskConical } as const;

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [kbName, setKbName] = useState<string | null>(null);
  const [lastKb, setLastKb] = useState<{ id: string; name: string } | null>(null);

  // Active KB from the URL; remember the last one so Research links stay contextual.
  const kbMatch = pathname.match(/^\/knowledge-bases\/(?!new)([^/]+)/);
  const activeKbId = kbMatch ? kbMatch[1] : null;

  useEffect(() => {
    const saved = localStorage.getItem("ragforge:lastKB");
    if (saved) { try { setLastKb(JSON.parse(saved)); } catch { /* ignore */ } }
    setCollapsed(localStorage.getItem("ragforge:sidebarCollapsed") === "1");
  }, []);

  useEffect(() => {
    if (activeKbId && (!lastKb || lastKb.id !== activeKbId)) {
      api.getKB(activeKbId).then((kb: KnowledgeBase) => {
        const rec = { id: kb.id, name: kb.name };
        setLastKb(rec);
        localStorage.setItem("ragforge:lastKB", JSON.stringify(rec));
      }).catch(() => null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeKbId]);

  useEffect(() => {
    api.systemStatus().then(setStatus).catch(() => setStatus(null));
    const t = setInterval(() => api.systemStatus().then(setStatus).catch(() => null), 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    setKbName(null);
    if (activeKbId) api.getKB(activeKbId).then((kb) => setKbName(kb.name)).catch(() => null);
  }, [activeKbId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => setMobileOpen(false), [pathname]);

  const toggleCollapse = useCallback(() => {
    setCollapsed((c) => {
      localStorage.setItem("ragforge:sidebarCollapsed", c ? "0" : "1");
      return !c;
    });
  }, []);

  const researchKb = activeKbId ? { id: activeKbId, name: kbName } : lastKb;

  const systemRows = [
    { key: "qdrant" as const, label: "Qdrant", ok: status?.qdrant.reachable ?? null, value: status?.qdrant.reachable ? "Connected" : status ? "Unreachable" : "—" },
    { key: "embedding" as const, label: "Embeddings", ok: true, value: status?.embedding.model ?? "—" },
    {
      key: "llm" as const,
      label: "LLM",
      ok: !(status && status.llm.allow_mock && (status.llm.provider.startsWith("not configured") || status.llm.provider === "mock")),
      value: status ? (status.llm.provider.startsWith("not configured") ? "Dev mock" : status.llm.provider) : "—",
    },
  ];

  const sidebar = (
    <SidebarContent
      collapsed={collapsed && !mobileOpen}
      onToggleCollapse={toggleCollapse}
      researchKb={researchKb}
      activeKbId={activeKbId}
      systemRows={systemRows}
      onMobileNavigate={() => setMobileOpen(false)}
    />
  );

  const crumbs: { label: string; href?: string }[] = [];
  if (pathname === "/knowledge-bases/new") crumbs.push({ label: "Create Knowledge Base" });
  else if (kbName) crumbs.push({ label: kbName, href: `/knowledge-bases/${activeKbId}` });
  else if (pathname.startsWith("/knowledge-bases")) crumbs.push({ label: "Knowledge Bases", href: "/knowledge-bases" });

  const section = NAV.research.find((r) => pathname.endsWith(r.suffix));
  if (kbName && section) crumbs.push({ label: section.label });

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside
        className={cn(
          "sticky top-0 hidden h-screen shrink-0 flex-col border-r border-line bg-surface-1 transition-[width] duration-200 ease-out md:flex",
          collapsed ? "w-[56px]" : "w-60"
        )}
      >
        {sidebar}
      </aside>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}
            className="fixed inset-0 z-40 bg-black/60 md:hidden" onClick={() => setMobileOpen(false)}
          >
            <motion.aside
              initial={{ x: -280 }} animate={{ x: 0 }} exit={{ x: -280 }}
              transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
              className="flex h-full w-64 flex-col border-r border-line bg-surface-1"
              onClick={(e) => e.stopPropagation()}
            >
              {sidebar}
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Topbar */}
        <header className="sticky top-0 z-30 flex h-12 items-center gap-3 border-b border-line bg-canvas/85 px-4 backdrop-blur">
          <button
            className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-ink md:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="h-4.5 w-4.5 h-5 w-5" />
          </button>
          <nav className="flex min-w-0 items-center gap-1 text-xs" aria-label="Breadcrumb">
            <Link href="/" className="shrink-0 text-ink-faint transition-colors hover:text-ink-muted">RAGForge</Link>
            {crumbs.map((c, i) => (
              <span key={i} className="flex min-w-0 items-center gap-1">
                <ChevronRight className="h-3 w-3 shrink-0 text-ink-ghost" />
                {c.href ? (
                  <Link href={c.href} className="truncate text-ink-muted transition-colors hover:text-ink">{c.label}</Link>
                ) : (
                  <span className="truncate text-ink">{c.label}</span>
                )}
              </span>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setPaletteOpen(true)}
              className="hidden items-center gap-2 rounded border border-line-strong bg-surface-2 px-2.5 py-1 text-xs text-ink-faint transition-colors hover:border-line-focus hover:text-ink-muted sm:flex"
            >
              <Search className="h-3.5 w-3.5" />
              Search…
              <kbd className="ml-2 rounded border border-line bg-surface-3 px-1 py-px text-2xs">⌘K</kbd>
            </button>
            <button
              className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-ink sm:hidden"
              onClick={() => setPaletteOpen(true)}
              aria-label="Search"
            >
              <Search className="h-4 w-4" />
            </button>
            <StatusDot tone={status ? (status.qdrant.reachable ? "ok" : "bad") : "neutral"} pulse={false} />
          </div>
        </header>

        <main className="min-w-0 flex-1 px-4 py-6 md:px-8">{children}</main>

        <footer className="border-t border-line px-4 py-3 text-2xs leading-4 text-ink-ghost md:px-8">
          RAGForge — knowledge engineering tool. Every score, metric, provenance field and status
          shown comes from a real pipeline execution or is explicitly labelled dev/mock. Never
          fabricated. Uploaded documents are processed locally.
        </footer>
      </div>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}

/* ------------------------------------------------------------------ */

function SidebarContent({
  collapsed, onToggleCollapse, researchKb, activeKbId, systemRows, onMobileNavigate,
}: {
  collapsed: boolean;
  onToggleCollapse: () => void;
  researchKb: { id?: string; name: string | null } | null;
  activeKbId: string | null;
  systemRows: { key: "qdrant" | "embedding" | "llm"; label: string; ok: boolean | null; value: string }[];
  onMobileNavigate: () => void;
}) {
  const pathname = usePathname();

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");

  const item = (href: string, label: string, Icon: React.ComponentType<{ className?: string }>, hint?: string | null) => {
    const active = isActive(href);
    return (
      <Link
        key={href}
        href={href}
        onClick={onMobileNavigate}
        title={collapsed ? label : undefined}
        className={cn(
          "group relative flex h-8 items-center gap-2.5 rounded-md px-2.5 text-sm transition-colors duration-100",
          active ? "bg-surface-3 text-ink" : "text-ink-muted hover:bg-surface-2 hover:text-ink",
          collapsed && "justify-center px-0"
        )}
      >
        {active && (
          <motion.span
            layoutId="nav-indicator"
            className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-accent"
            transition={{ type: "spring", stiffness: 500, damping: 40 }}
          />
        )}
        <Icon className={cn("h-4 w-4 shrink-0", active ? "text-accent-soft" : "text-ink-faint group-hover:text-ink-muted")} />
        {!collapsed && <span className="min-w-0 flex-1 truncate">{label}</span>}
        {!collapsed && hint && (
          <span className="shrink-0 text-2xs text-ink-ghost opacity-0 transition-opacity group-hover:opacity-100">{hint}</span>
        )}
        {collapsed && (
          <span className="pointer-events-none absolute left-full z-50 ml-2 hidden whitespace-nowrap rounded border border-line-strong bg-surface-3 px-2 py-1 text-2xs text-ink shadow-raise group-hover:block">
            {label}
          </span>
        )}
      </Link>
    );
  };

  const groupLabel = (label: string) =>
    collapsed ? (
      <div className="mx-auto my-2 h-px w-6 bg-line" />
    ) : (
      <div className="section-label mb-1 mt-5 px-2.5 first:mt-0">{label}</div>
    );

  return (
    <>
      {/* Wordmark */}
      <div className={cn("flex h-12 shrink-0 items-center gap-2.5 border-b border-line px-4", collapsed && "justify-center px-0")}>
        <Link href="/" onClick={onMobileNavigate} className="flex items-center gap-2.5 overflow-hidden">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-accent-dim bg-accent/10">
            <Workflow className="h-3.5 w-3.5 text-accent-soft" />
          </span>
          {!collapsed && (
            <span className="min-w-0">
              <span className="block text-sm font-semibold leading-4 tracking-tight text-ink">RAGForge</span>
              <span className="block text-2xs leading-3 text-ink-faint">Knowledge Engineering</span>
            </span>
          )}
        </Link>
        <button
          className="ml-auto hidden rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-ink md:block"
          onClick={onToggleCollapse}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </button>
        <button className="ml-auto rounded p-1 text-ink-faint hover:text-ink md:hidden" onClick={onMobileNavigate} aria-label="Close">
          <X className="h-4 w-4" />
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3">
        {groupLabel("Workspace")}
        <div className="space-y-0.5">
          {NAV.workspace.map((n) => item(n.href, n.label, n.icon, n.hint))}
        </div>

        {groupLabel("Research")}
        <div className="space-y-0.5">
          {researchKb?.id ? (
            NAV.research.map((n) => item(`/knowledge-bases/${researchKb.id}${n.suffix}`, n.label, n.icon, (n as { hint?: string }).hint))
          ) : (
            <p className="px-2.5 py-1 text-2xs leading-4 text-ink-ghost">
              {collapsed ? "…" : "Open or create a knowledge base to enable the research views."}
            </p>
          )}
          {NAV.global.map((n) => item(n.href, n.label, n.icon, n.hint))}
        </div>

        {groupLabel("System")}
        <div className="space-y-0.5">
          {systemRows.map((row) => {
            const Icon = SYSTEM_ICON[row.key];
            const tone = row.ok === null ? "neutral" : row.ok ? "ok" : "bad";
            return (
              <div
                key={row.key}
                title={collapsed ? `${row.label}: ${row.value}` : undefined}
                className={cn(
                  "flex h-8 items-center gap-2.5 rounded-md px-2.5 text-sm text-ink-faint",
                  collapsed && "justify-center px-0"
                )}
              >
                <Icon className="h-4 w-4 shrink-0 text-ink-ghost" />
                {!collapsed ? (
                  <>
                    <span className="min-w-0 flex-1 truncate">{row.label}</span>
                    <span className="data-value truncate text-2xs text-ink-faint" title={row.value}>{row.value}</span>
                    <StatusDot tone={tone} pulse={tone === "bad"} />
                  </>
                ) : (
                  <span className="pointer-events-none absolute left-full z-50 ml-2 hidden items-center gap-1.5 whitespace-nowrap rounded border border-line-strong bg-surface-3 px-2 py-1 text-2xs text-ink shadow-raise group-hover:flex">
                    {row.label}: {row.value}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </nav>

      {!collapsed && (
        <div className="border-t border-line px-4 py-3 text-2xs leading-4 text-ink-ghost">
          <span className="flex items-center gap-1"><kbd className="rounded border border-line-strong px-1">⌘K</kbd> command palette</span>
        </div>
      )}
    </>
  );
}
