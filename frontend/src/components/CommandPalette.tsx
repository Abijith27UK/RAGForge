"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Boxes, Database, FileSearch, FlaskConical, GitBranch, LayoutDashboard,
  ListTree, Plus, Search, Sigma,
} from "lucide-react";
import { api, KnowledgeBase } from "@/lib/api";
import { cn } from "@/lib/utils";

type Item = {
  id: string;
  label: string;
  hint?: string;
  icon: React.ComponentType<{ className?: string }>;
  group: string;
  keywords?: string;
  run: (router: ReturnType<typeof useRouter>) => void;
};

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard, group: "Navigate", keywords: "dashboard home workspace" },
  { href: "/knowledge-bases", label: "Knowledge Bases", icon: Database, group: "Navigate", keywords: "kb list corpora" },
  { href: "/knowledge-bases/new", label: "Create Knowledge Base", icon: Plus, group: "Actions", keywords: "new add kb domain" },
];

const KB_TABS = [
  { suffix: "", label: "open", icon: Database, keywords: "overview pipeline" },
  { suffix: "/domain", label: "domain analysis", icon: GitBranch, keywords: "spec map requirements" },
  { suffix: "/sources", label: "sources", icon: FileSearch, keywords: "quality scores discover" },
  { suffix: "/chunks", label: "chunks", icon: ListTree, keywords: "inspect provenance" },
  { suffix: "/retrieval", label: "retrieval lab", icon: Search, keywords: "query search test" },
  { suffix: "/evaluation", label: "evaluation", icon: Sigma, keywords: "metrics recall mrr ndcg ground truth" },
];

export default function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      api.listKBs().then(setKbs).catch(() => setKbs([]));
      const t = setTimeout(() => inputRef.current?.focus(), 20);
      return () => clearTimeout(t);
    }
  }, [open]);

  const items = useMemo<Item[]>(() => {
    const out: Item[] = NAV.map((n) => ({
      id: `nav-${n.href}`,
      label: n.label,
      hint: n.href,
      icon: n.icon,
      group: n.group,
      keywords: n.keywords,
      run: (r) => r.push(n.href),
    }));
    for (const kb of kbs) {
      for (const tab of KB_TABS) {
        out.push({
          id: `kb-${kb.id}-${tab.suffix}`,
          label: tab.suffix ? `${tab.label} — ${kb.name}` : kb.name,
          hint: kb.domain,
          icon: tab.icon,
          group: "Knowledge bases",
          keywords: `${kb.domain} ${tab.keywords}`,
          run: (r) => r.push(`/knowledge-bases/${kb.id}${tab.suffix}`),
        });
      }
    }
    const q = query.trim().toLowerCase();
    if (!q) return out.slice(0, 12);
    return out
      .filter((it) => (it.label + " " + (it.keywords ?? "") + " " + (it.hint ?? "")).toLowerCase().includes(q))
      .slice(0, 14);
  }, [kbs, query]);

  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-idx="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  function choose(it: Item) {
    onOpenChange(false);
    it.run(router);
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[12vh] backdrop-blur-[2px]"
          onClick={() => onOpenChange(false)}
        >
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.99 }}
            transition={{ duration: 0.15, ease: [0.2, 0.8, 0.2, 1] }}
            className="w-full max-w-xl overflow-hidden rounded-lg border border-line-strong bg-surface-2 shadow-raise"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-label="Command palette"
          >
            <div className="flex items-center gap-2.5 border-b border-line px-4">
              <Search className="h-4 w-4 text-ink-faint" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(a + 1, items.length - 1)); }
                  else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
                  else if (e.key === "Enter" && items[active]) { e.preventDefault(); choose(items[active]); }
                  else if (e.key === "Escape") { onOpenChange(false); }
                }}
                placeholder="Search knowledge bases, pages, actions…"
                className="h-12 w-full bg-transparent text-sm text-ink outline-none placeholder:text-ink-ghost"
              />
              <kbd className="rounded border border-line-strong bg-surface-3 px-1.5 py-0.5 text-2xs text-ink-faint">esc</kbd>
            </div>
            <div ref={listRef} className="max-h-[46vh] overflow-y-auto p-1.5">
              {items.length === 0 && (
                <div className="px-3 py-8 text-center text-xs text-ink-faint">No matches for “{query}”</div>
              )}
              {items.map((it, i) => (
                <button
                  key={it.id}
                  data-idx={i}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => choose(it)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors",
                    i === active ? "bg-accent/10 text-ink" : "text-ink-muted hover:bg-surface-3"
                  )}
                >
                  <it.icon className={cn("h-4 w-4 shrink-0", i === active ? "text-accent-soft" : "text-ink-faint")} />
                  <span className="min-w-0 flex-1 truncate text-sm">{it.label}</span>
                  {it.hint && <span className="hidden shrink-0 text-2xs text-ink-faint sm:block">{it.hint}</span>}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-3 border-t border-line px-4 py-2 text-2xs text-ink-faint">
              <span><kbd className="rounded border border-line-strong px-1">↑↓</kbd> navigate</span>
              <span><kbd className="rounded border border-line-strong px-1">↵</kbd> open</span>
              <span className="ml-auto flex items-center gap-1">
                <Boxes className="h-3 w-3" /> RAGForge
              </span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
