"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Database } from "lucide-react";
import { api } from "@/lib/api";
import { Button, Panel, PanelHeader } from "@/components/ui";

const INPUT =
  "h-9 w-full rounded border border-line-strong bg-surface-3 px-3 text-sm text-ink placeholder:text-ink-ghost focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30";
const AREA =
  "w-full rounded border border-line-strong bg-surface-3 px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30";
const LABEL = "section-label mb-1.5 block";

export default function CreateKB() {
  const router = useRouter();
  const [form, setForm] = useState({
    name: "",
    domain: "",
    purpose: "",
    target_audience: "",
    depth: "intermediate",
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const set = (k: keyof typeof form) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => setForm({ ...form, [k]: e.target.value });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const kb = await api.createKB(form);
      router.push(`/knowledge-bases/${kb.id}`);
    } catch (err: unknown) {
      setError(String((err as Error).message));
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-6">
        <div className="section-label mb-2">Workspace</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Create Knowledge Base</h1>
        <p className="mt-1 text-xs text-ink-faint">
          The domain specification drives source discovery, relevance scoring, and evaluation —
          be specific about the domain and intended use.
        </p>
      </div>

      {error && (
        <div className="mb-4 rounded border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</div>
      )}

      <Panel>
        <PanelHeader title="Definition" right={<Database className="h-3.5 w-3.5 text-ink-ghost" />} />
        <form onSubmit={submit} className="space-y-4 p-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className={LABEL}>Name</label>
              <input className={INPUT} required value={form.name} onChange={set("name")} placeholder="Automobile Engineering KB" />
            </div>
            <div>
              <label className={LABEL}>Domain</label>
              <input className={INPUT} required value={form.domain} onChange={set("domain")} placeholder="Automobile Engineering" />
            </div>
          </div>
          <div>
            <label className={LABEL}>Purpose / use case</label>
            <textarea className={AREA} required rows={3} value={form.purpose} onChange={set("purpose")}
              placeholder="Build a study assistant for automobile engineering students" />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className={LABEL}>Target audience</label>
              <input className={INPUT} required value={form.target_audience} onChange={set("target_audience")}
                placeholder="Engineering students" />
            </div>
            <div>
              <label className={LABEL}>Depth</label>
              <select className={INPUT} value={form.depth} onChange={set("depth")}>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="technical">Technical</option>
                <option value="expert">Expert</option>
              </select>
            </div>
          </div>
          <div className="flex justify-end border-t border-line pt-4">
            <Button type="submit" variant="primary" loading={busy}>Create Knowledge Base</Button>
          </div>
        </form>
      </Panel>
    </div>
  );
}
