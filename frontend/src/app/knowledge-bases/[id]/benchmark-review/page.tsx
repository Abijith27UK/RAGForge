"use client";

/**
 * Benchmark Review — the V10 human-ground-truth workflow.
 *
 * QUESTION → CORPUS EVIDENCE → REFERENCE ANSWER → KEY POINTS
 *         → ACCEPTABLE ELEMENTS → PROVENANCE → HUMAN REVIEW
 *         → APPROVED BENCHMARK → FROZEN BENCHMARK VERSION
 *
 * Research-integrity rules mirrored from the backend:
 * - A question is PENDING / IN_REVIEW / APPROVED / REJECTED / AMBIGUOUS /
 *   INSUFFICIENT_EVIDENCE. Opening a question is NOT reviewing it: only an
 *   explicit reviewer outcome moves a question out of PENDING.
 * - Only APPROVED questions carry scorable ground truth. AMBIGUOUS and
 *   INSUFFICIENT_EVIDENCE are excluded from scoring BY POLICY and are named,
 *   never silently dropped.
 * - Coverage rates with no denominator render as UNKNOWN, never 0.
 * - Every write is append-only. Correcting a reference answer or a review
 *   creates a NEW record with `supersedes`; history stays visible below.
 * - The CENTER panel is SOURCE EVIDENCE, not ground truth: it is the actual
 *   indexed chunk text, with the provenance we have (a missing page renders
 *   as "not recorded").
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  FileText,
  Lock,
  RefreshCw,
  ShieldQuestion,
} from "lucide-react";
import {
  benchmarkReviewApi,
  answerEvaluationApi,
  DEFAULT_ANSWER_BENCHMARK_PATH,
  BenchmarkQuestionDetail,
  BenchmarkReviewPacket,
  DIMENSION_VERDICTS,
  EvidenceItem,
  FrozenVersionSummary,
  GroundTruthAnnotation,
  Measured,
  QUESTION_STATES,
  REQUIRED_REVIEW_DIMENSIONS,
  ReviewDimension,
  REVIEW_DIMENSIONS,
  QuestionState,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, Button, EmptyState, Panel, PanelHeader } from "@/components/ui";

const STATE_TONE: Record<string, "ok" | "warn" | "bad" | "accent" | "neutral"> = {
  pending: "neutral",
  in_review: "accent",
  approved: "ok",
  rejected: "bad",
  ambiguous: "warn",
  insufficient_evidence: "warn",
};

function StateBadge({ state }: { state: string }) {
  return <Badge tone={STATE_TONE[state] ?? "neutral"}>{state.replace(/_/g, " ")}</Badge>;
}

/** A rate with no denominator is UNKNOWN — never rendered as 0. */
function MeasuredValue({ m, digits = 3 }: { m: Measured | undefined; digits?: number }) {
  if (!m || !m.measured || m.value === null) {
    return (
      <span className="text-ink-faint" title={m?.reason ?? ""}>
        UNKNOWN
      </span>
    );
  }
  return (
    <span title={m.reason}>
      {m.value.toFixed(digits)}
      {m.sample_size != null && (
        <span className="ml-1 text-2xs text-ink-faint">n={m.sample_size}</span>
      )}
    </span>
  );
}

function ProvenanceChip({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <span className="text-2xs">
      <span className="text-ink-faint">{label}: </span>
      <span className={value === null || value === undefined || value === "" ? "text-ink-faint italic" : "text-ink-muted"}>
        {value === null || value === undefined || value === "" ? "not recorded" : String(value)}
      </span>
    </span>
  );
}

function EvidenceCard({ item }: { item: EvidenceItem }) {
  const source = item.source_title || item.label_source_title;
  return (
    <div
      className={cn(
        "rounded-md border bg-surface-2 p-3",
        item.problems.length > 0 ? "border-warn/50" : "border-line"
      )}
    >
      <div className="mb-1.5 flex flex-wrap items-center gap-2">
        <FileText className="h-3.5 w-3.5 text-ink-faint" />
        <span className="font-mono text-2xs text-ink-faint">{item.chunk_id}</span>
        {item.required && <Badge tone="accent">required</Badge>}
        {!item.found && <Badge tone="bad">not in corpus</Badge>}
        {item.found && <Badge tone="neutral">{item.char_count} chars</Badge>}
      </div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
        <ProvenanceChip label="source" value={source} />
        <ProvenanceChip label="publisher" value={item.publisher} />
        <ProvenanceChip label="document" value={item.document_title || item.document_id} />
        <ProvenanceChip label="section" value={item.section_path || item.section} />
        <ProvenanceChip label="page" value={item.page} />
        <ProvenanceChip label="slide" value={item.slide ? `slide ${item.slide}${item.slide_title ? ` — ${item.slide_title}` : ""}` : null} />
        <ProvenanceChip label="hash" value={item.content_hash ? item.content_hash.slice(0, 12) : null} />
        {item.source_url && (
          <span className="text-2xs">
            <span className="text-ink-faint">url: </span>
            <a className="text-accent underline-offset-2 hover:underline" href={item.source_url} target="_blank" rel="noreferrer">
              {item.source_url.slice(0, 60)}
            </a>
          </span>
        )}
      </div>
      {item.found ? (
        <p className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded border border-line bg-surface-3 p-2 text-sm leading-6 text-ink-muted">
          {item.text}
        </p>
      ) : (
        <p className="text-2xs text-bad">
          This chunk is not in the live corpus, so no reference answer can be grounded in it.
        </p>
      )}
      {item.problems.map((p, i) => (
        <p key={i} className="mt-1 text-2xs text-warn">
          {p}
        </p>
      ))}
    </div>
  );
}

/** LEFT panel — the reviewer's own ground truth for this question. */
function AuthoringPanel({
  kbId,
  benchmarkPath,
  detail,
  onSaved,
}: {
  kbId: string;
  benchmarkPath: string;
  detail: BenchmarkQuestionDetail;
  onSaved: () => void;
}) {
  const effective = detail.effective_annotation;
  const [author, setAuthor] = useState(effective?.author ?? "");
  const [reference, setReference] = useState(effective?.expected_answer ?? "");
  const [points, setPoints] = useState((effective?.key_points ?? []).join("\n"));
  const [elements, setElements] = useState(
    (effective?.acceptable_answer_elements ?? []).join("\n")
  );
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [problems, setProblems] = useState<string[]>([]);

  useEffect(() => {
    setAuthor(effective?.author ?? "");
    setReference(effective?.expected_answer ?? "");
    setPoints((effective?.key_points ?? []).join("\n"));
    setElements((effective?.acceptable_answer_elements ?? []).join("\n"));
    setNote("");
    setProblems([]);
    setErr(null);
  }, [detail.summary.question_id, effective?.annotation_id]);

  const save = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await benchmarkReviewApi.authorGroundTruth(
        kbId,
        detail.summary.question_id,
        {
          benchmark_path: benchmarkPath,
          author: author.trim(),
          expected_answer: reference,
          key_points: points.split("\n").map((p) => p.trim()).filter(Boolean),
          acceptable_answer_elements: elements
            .split("\n")
            .map((p) => p.trim())
            .filter(Boolean),
          evidence_chunk_ids: detail.evidence.map((e) => e.chunk_id),
          note: note.trim(),
        }
      );
      setProblems(res.problems);
      onSaved();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div>
        <div className="section-label">Question</div>
        <p className="text-sm text-ink">{detail.summary.question}</p>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="font-mono text-2xs text-ink-faint">{detail.summary.question_id}</span>
          <StateBadge state={detail.summary.state} />
          <Badge tone="neutral">{detail.summary.subdomain || "no subdomain"}</Badge>
          <Badge tone="neutral">answerability: {detail.summary.answerability}</Badge>
        </div>
      </div>

      {detail.summary.reasons.length > 0 && (
        <div className="rounded border border-line bg-surface-3 px-2.5 py-2">
          <div className="section-label mb-1">Why this state</div>
          {detail.summary.reasons.map((r, i) => (
            <p key={i} className="text-2xs text-ink-muted">
              {r}
            </p>
          ))}
          {detail.summary.problems.map((p, i) => (
            <p key={`p${i}`} className="text-2xs text-warn">
              blocked: {p}
            </p>
          ))}
        </div>
      )}

      <label className="block">
        <span className="section-label">Author (required)</span>
        <input
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          placeholder="e.g. ada@example.edu"
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        />
      </label>

      <label className="block">
        <span className="section-label">Reference answer (human-authored, from the evidence)</span>
        <textarea
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          rows={5}
          placeholder="Write the answer the evidence actually supports. Never paste a RAG answer here."
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        />
      </label>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block">
          <span className="section-label">Key points (one per line)</span>
          <textarea
            value={points}
            onChange={(e) => setPoints(e.target.value)}
            rows={4}
            placeholder="Facts a correct answer must state"
            className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
          />
        </label>
        <label className="block">
          <span className="section-label">Acceptable answer elements (one per line)</span>
          <textarea
            value={elements}
            onChange={(e) => setElements(e.target.value)}
            rows={4}
            placeholder="Alternative phrasings a human judged acceptable"
            className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
          />
        </label>
      </div>

      <label className="block">
        <span className="section-label">Note (recorded in provenance)</span>
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="which chunk did you read?"
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        />
      </label>

      {err && <p className="text-2xs text-bad">{err}</p>}
      {problems.length > 0 && (
        <div className="rounded border border-warn/40 bg-warn/5 px-2.5 py-2">
          <div className="section-label mb-1 text-warn">Stored, but not approvable yet</div>
          {problems.map((p, i) => (
            <p key={i} className="text-2xs text-warn">
              {p}
            </p>
          ))}
        </div>
      )}
      <div className="flex items-center gap-2">
        <Button onClick={save} loading={busy} disabled={!author.trim()}>
          Save reference answer
        </Button>
        <span className="text-2xs text-ink-faint">
          Append-only: saving again adds a new revision, nothing is overwritten.
        </span>
      </div>

      {detail.annotation_history.length > 0 && (
        <div>
          <div className="section-label mb-1">
            Annotation history ({detail.annotation_history.length})
          </div>
          <div className="space-y-1.5">
            {detail.annotation_history.map((a: GroundTruthAnnotation) => (
              <div key={a.annotation_id} className="rounded border border-line bg-surface-3 px-2.5 py-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-2xs text-ink-faint">{a.annotation_id}</span>
                  <span className="text-2xs text-ink-faint">{a.author}</span>
                  <span className="text-2xs text-ink-faint">{a.created_at}</span>
                  {a.annotation_id === effective?.annotation_id && <Badge tone="accent">effective</Badge>}
                  {a.supersedes && <Badge tone="neutral">supersedes {a.supersedes}</Badge>}
                </div>
                <p className="mt-1 whitespace-pre-wrap text-2xs text-ink-muted">
                  {a.expected_answer || "(no reference answer)"}
                </p>
                <p className="mt-0.5 text-2xs text-ink-faint">
                  {a.key_points.length} key point(s) · {a.acceptable_answer_elements.length} acceptable element(s) · evidence:{" "}
                  {a.evidence.map((e) => e.chunk_id).join(", ") || "none"}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** RIGHT panel — the reviewer's verdict (append-only). */
function ReviewPanel({
  kbId,
  benchmarkPath,
  detail,
  onSaved,
}: {
  kbId: string;
  benchmarkPath: string;
  detail: BenchmarkQuestionDetail;
  onSaved: () => void;
}) {
  const [reviewer, setReviewer] = useState("");
  const [outcome, setOutcome] = useState<QuestionState>("approved");
  const [verdicts, setVerdicts] = useState<Record<string, string>>(() =>
    Object.fromEntries(REVIEW_DIMENSIONS.map((d) => [d, "ok"]))
  );
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [unresolvedText, setUnresolvedText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setVerdicts(Object.fromEntries(REVIEW_DIMENSIONS.map((d) => [d, "ok"])));
    setNotes({});
    setUnresolvedText("");
    setErr(null);
  }, [detail.summary.question_id]);

  const required = useMemo(
    () => REQUIRED_REVIEW_DIMENSIONS.filter((d) => verdicts[d] !== "ok" && verdicts[d] !== "ok_with_note"),
    [verdicts]
  );

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await benchmarkReviewApi.createReview(kbId, detail.summary.question_id, {
        benchmark_path: benchmarkPath,
        reviewer: reviewer.trim(),
        outcome,
        verdicts,
        notes: Object.fromEntries(Object.entries(notes).filter(([, v]) => v.trim())),
        evidence_checked: detail.evidence.map((e) => e.chunk_id),
        annotation_id: detail.effective_annotation?.annotation_id ?? "",
        unresolved: unresolvedText.split("\n").map((s) => s.trim()).filter(Boolean),
        supersedes: detail.effective_review?.review_id ?? "",
      });
      onSaved();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="rounded border border-line bg-surface-3 px-2.5 py-2">
        <div className="section-label mb-1">Reviewer input — not ground truth until saved</div>
        <p className="text-2xs text-ink-faint">
          Only a saved review by a named human moves this question. An
          &quot;approved&quot; outcome is refused unless every required dimension is assessed
          affirmatively.
        </p>
      </div>

      <label className="block">
        <span className="section-label">Reviewer identity (required)</span>
        <input
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
          placeholder="e.g. ada@example.edu"
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        />
      </label>

      <label className="block">
        <span className="section-label">Outcome</span>
        <select
          value={outcome}
          onChange={(e) => setOutcome(e.target.value as QuestionState)}
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        >
          {QUESTION_STATES.filter((s) => s !== "pending").map((s) => (
            <option key={s} value={s}>
              {s.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </label>

      <div>
        <div className="section-label mb-1.5">Dimensions</div>
        <div className="space-y-2">
          {REVIEW_DIMENSIONS.map((d) => (
            <div key={d}>
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="w-36 text-2xs text-ink-muted">
                  {d.replace(/_/g, " ")}
                  {REQUIRED_REVIEW_DIMENSIONS.includes(d) && (
                    <span className="ml-1 text-2xs text-ink-faint">required</span>
                  )}
                </span>
                {DIMENSION_VERDICTS.map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setVerdicts((cur) => ({ ...cur, [d]: v }))}
                    className={cn(
                      "rounded-full border px-2 py-0.5 text-2xs transition-colors",
                      verdicts[d] === v
                        ? "border-accent/60 bg-accent/10 text-ink"
                        : "border-line text-ink-faint hover:text-ink-muted"
                    )}
                  >
                    {v.replace(/_/g, " ")}
                  </button>
                ))}
              </div>
              <input
                value={notes[d] ?? ""}
                onChange={(e) => setNotes((cur) => ({ ...cur, [d]: e.target.value }))}
                placeholder="note for this dimension (optional)"
                className="mt-1 w-full rounded border border-line bg-surface-3 px-2 py-1 text-2xs text-ink"
              />
            </div>
          ))}
        </div>
      </div>

      <label className="block">
        <span className="section-label">Unresolved (one per line)</span>
        <textarea
          value={unresolvedText}
          onChange={(e) => setUnresolvedText(e.target.value)}
          rows={2}
          placeholder="things this question does not settle"
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        />
      </label>

      <div className="rounded border border-line bg-surface-3 px-2.5 py-1.5">
        <p className="text-2xs text-ink-faint">
          evidence checked: {detail.evidence.map((e) => e.chunk_id).join(", ") || "none"}
          {detail.effective_review && (
            <>
              {" "}· supersedes {detail.effective_review.review_id}
            </>
          )}
        </p>
      </div>

      {outcome === "approved" && required.length > 0 && (
        <p className="text-2xs text-warn">
          an &quot;approved&quot; outcome will be refused: {required.join(", ")} not affirmative
        </p>
      )}
      {err && <p className="text-2xs text-bad">{err}</p>}
      <Button onClick={submit} loading={busy} disabled={!reviewer.trim()}>
        Record review
      </Button>

      {detail.review_history.length > 0 && (
        <div>
          <div className="section-label mb-1">Review history ({detail.review_history.length})</div>
          <div className="space-y-1.5">
            {detail.review_history.map((rv) => (
              <div key={rv.review_id} className="rounded border border-line bg-surface-3 px-2.5 py-1.5">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone={rv.outcome ? STATE_TONE[rv.outcome] ?? "neutral" : "neutral"}>
                    {rv.outcome ?? "dimensions only"}
                  </Badge>
                  <span className="font-mono text-2xs text-ink-faint">{rv.reviewer}</span>
                  <span className="text-2xs text-ink-faint">{rv.created_at}</span>
                  {rv.review_id === detail.effective_review?.review_id && <Badge tone="accent">effective</Badge>}
                </div>
                <p className="mt-1 text-2xs text-ink-faint">
                  {Object.entries(rv.verdicts).map(([k, v]) => `${k}=${v}`).join(" · ") || "no dimensions assessed"}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Approval gate + freeze + frozen versions (bottom of the review surface). */
function FreezePanel({
  kbId,
  benchmarkPath,
  packet,
  onFrozen,
}: {
  kbId: string;
  benchmarkPath: string;
  packet: BenchmarkReviewPacket;
  onFrozen: () => void;
}) {
  const [frozenBy, setFrozenBy] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [verify, setVerify] = useState<Record<string, { intact: boolean; problems: string[] }>>({});
  const [scoring, setScoring] = useState<string | null>(null);
  const [scoreResult, setScoreResult] = useState<Record<string, string>>({});

  const c = packet.completeness;
  const gate = packet.gate;

  const freeze = async () => {
    setBusy(true);
    setErr(null);
    try {
      await benchmarkReviewApi.freeze(kbId, {
        benchmark_path: benchmarkPath,
        frozen_by: frozenBy.trim(),
      });
      onFrozen();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const doVerify = async (versionId: string) => {
    try {
      const res = await benchmarkReviewApi.verifyVersion(kbId, versionId);
      setVerify((cur) => ({ ...cur, [versionId]: { intact: res.intact, problems: res.problems } }));
    } catch (e) {
      setVerify((cur) => ({
        ...cur,
        [versionId]: { intact: false, problems: [e instanceof Error ? e.message : String(e)] },
      }));
    }
  };

  const score = async (v: FrozenVersionSummary) => {
    setScoring(v.version_id);
    try {
      const run = await answerEvaluationApi.run(kbId, {
        benchmark_path: benchmarkPath,
        benchmark_version_id: v.version_id,
        evaluator: "reference",
        limit: 5,
      });
      const corr = run.aggregate.correctness;
      setScoreResult((cur) => ({
        ...cur,
        [v.version_id]: corr.measured
          ? `correctness ${corr.value?.toFixed(3)} (n=${corr.sample_size ?? "?"}) — reference-key-point-coverage-v1`
          : `correctness UNKNOWN — ${corr.reason}`,
      }));
    } catch (e) {
      setScoreResult((cur) => ({
        ...cur,
        [v.version_id]: `failed: ${e instanceof Error ? e.message : String(e)}`,
      }));
    } finally {
      setScoring(null);
    }
  };

  const rows: [string, number][] = [
    ["total", c.total],
    ["pending", c.pending],
    ["in review", c.in_review],
    ["approved", c.approved],
    ["rejected", c.rejected],
    ["ambiguous", c.ambiguous],
    ["insufficient evidence", c.insufficient_evidence],
  ];

  return (
    <Panel>
      <PanelHeader
        title="Review completeness & approval gate"
        right={
          gate.approved ? (
            <Badge tone="ok">gate passed</Badge>
          ) : (
            <Badge tone="warn">gate blocked</Badge>
          )
        }
      />
      <div className="grid gap-4 lg:grid-cols-3">
        <div>
          <div className="section-label mb-1.5">Question states</div>
          <div className="space-y-0.5">
            {rows.map(([label, value]) => (
              <div key={label} className="flex items-center justify-between text-2xs">
                <span className="text-ink-muted">{label}</span>
                <span className="data-value text-ink">{value}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="section-label mb-1.5">Authoring coverage</div>
          <div className="space-y-0.5">
            {([
              ["reference answers", c.reference_answer_coverage, c.authored_reference_answer_count],
              ["key points", c.key_point_coverage, c.authored_key_point_count],
              ["acceptable elements", c.acceptable_element_coverage, c.authored_acceptable_element_count],
              ["corpus-valid provenance", c.provenance_coverage, c.provenance_valid_count],
            ] as [string, Measured, number][]).map(([label, m, count]) => (
              <div key={label} className="flex items-center justify-between text-2xs">
                <span className="text-ink-muted">{label}</span>
                <span className="text-ink">
                  <MeasuredValue m={m} /> <span className="text-ink-faint">({count})</span>
                </span>
              </div>
            ))}
          </div>
          <p className="mt-1.5 text-2xs text-ink-faint">
            {c.review_count} review(s) by {c.reviewers.join(", ") || "nobody"}
          </p>
        </div>
        <div>
          <div className="section-label mb-1.5">Approval policy {gate.policy.policy_version}</div>
          <div className="max-h-40 space-y-1 overflow-y-auto">
            {gate.reasons.map((r, i) => (
              <p key={i} className="flex gap-1 text-2xs text-ink-muted">
                {gate.approved ? (
                  <CircleCheck className="mt-0.5 h-3 w-3 shrink-0 text-ok" />
                ) : (
                  <CircleAlert className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
                )}
                <span>{r}</span>
              </p>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-end gap-2 border-t border-line pt-3">
        <label className="block">
          <span className="section-label">Freeze as (required)</span>
          <input
            value={frozenBy}
            onChange={(e) => setFrozenBy(e.target.value)}
            placeholder="your name"
            className="mt-0.5 w-56 rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
          />
        </label>
        <Button onClick={freeze} loading={busy} disabled={!gate.approved || !frozenBy.trim()}>
          <Lock className="mr-1 h-3.5 w-3.5" /> Freeze approved benchmark
        </Button>
        <span className="text-2xs text-ink-faint">
          Freezing creates an immutable version. It cannot be edited; a correction
          becomes a new version.
        </span>
      </div>
      {err && <p className="mt-1 text-2xs text-bad">{err}</p>}

      {packet.versions.length > 0 && (
        <div className="mt-3 border-t border-line pt-3">
          <div className="section-label mb-1.5">Frozen versions ({packet.versions.length})</div>
          <div className="space-y-1.5">
            {packet.versions.map((v) => (
              <div key={v.version_id} className="rounded border border-line bg-surface-3 px-2.5 py-2">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="accent">v{v.version}</Badge>
                  <span className="font-mono text-2xs text-ink-faint">{v.version_id}</span>
                  <span className="text-2xs text-ink-faint">{v.frozen_by}</span>
                  <span className="text-2xs text-ink-faint">{v.created_at}</span>
                  <Badge tone="ok">{v.scoring_question_count} scored</Badge>
                  {v.excluded_question_count > 0 && (
                    <Badge tone="warn">{v.excluded_question_count} excluded</Badge>
                  )}
                  <button
                    type="button"
                    onClick={() => doVerify(v.version_id)}
                    className="text-2xs text-accent underline-offset-2 hover:underline"
                  >
                    verify integrity
                  </button>
                  <button
                    type="button"
                    onClick={() => score(v)}
                    disabled={scoring === v.version_id}
                    className="text-2xs text-accent underline-offset-2 hover:underline disabled:text-ink-faint"
                  >
                    {scoring === v.version_id ? "scoring…" : "score (first 5 questions)"}
                  </button>
                </div>
                <p className="mt-1 text-2xs text-ink-faint">
                  content {v.benchmark_fingerprint} · labels {v.ground_truth_fingerprint} · artifact {v.artifact_fingerprint.slice(0, 16)}…
                </p>
                {verify[v.version_id] && (
                  <p className={cn("mt-1 text-2xs", verify[v.version_id].intact ? "text-ok" : "text-bad")}>
                    {verify[v.version_id].intact
                      ? "intact — fingerprints recomputed and match"
                      : `TAMPERED: ${verify[v.version_id].problems.join("; ")}`}
                  </p>
                )}
                {scoreResult[v.version_id] && (
                  <p className="mt-1 text-2xs text-ink-muted">{scoreResult[v.version_id]}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}

export default function BenchmarkReviewPage() {
  const params = useParams<{ id: string }>();
  const kbId = typeof params?.id === "string" ? params.id : "";

  const [packet, setPacket] = useState<BenchmarkReviewPacket | null>(null);
  const [detail, setDetail] = useState<BenchmarkQuestionDetail | null>(null);
  const [index, setIndex] = useState(0);
  const [tick, setTick] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  // Whole-benchmark packet
  useEffect(() => {
    if (!kbId) return;
    let alive = true;
    (async () => {
      try {
        const p = await benchmarkReviewApi.packet(kbId);
        if (!alive) return;
        setPacket(p);
        setIndex((cur) => Math.min(cur, Math.max(0, p.questions.length - 1)));
        setErr(null);
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [kbId, tick]);

  // Current question detail (evidence + history)
  const currentId = packet?.questions[index]?.question_id;
  useEffect(() => {
    if (!kbId || !currentId) {
      setDetail(null);
      return;
    }
    let alive = true;
    (async () => {
      try {
        const d = await benchmarkReviewApi.question(kbId, currentId);
        if (alive) setDetail(d);
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [kbId, currentId, tick]);

  if (!kbId) return null;

  if (loading && !packet) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-muted">
        <RefreshCw className="h-4 w-4 animate-spin" /> loading the benchmark review packet…
      </div>
    );
  }
  if (err && !packet) {
    return (
      <Panel>
        <PanelHeader title="Benchmark review" />
        <EmptyState title="Could not load the review packet" hint={err} />
      </Panel>
    );
  }
  if (!packet) return null;

  const total = packet.questions.length;
  // A question counts as REVIEWED only when a human recorded a verdict for it;
  // authoring a draft does not. The backend's `reviewed` property is not part
  // of the serialized payload, so it is derived from the four terminal states.
  const reviewed =
    packet.completeness.approved +
    packet.completeness.rejected +
    packet.completeness.ambiguous +
    packet.completeness.insufficient_evidence;
  const identity = packet.identity;

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          title={`Benchmark review — ${identity.name}`}
          right={
            <div className="flex items-center gap-2">
              <Badge tone={identity.lifecycle === "frozen" ? "ok" : identity.lifecycle === "approved" ? "accent" : "warn"}>
                lifecycle: {identity.lifecycle}
              </Badge>
              <Badge tone={identity.human_review === "human_reviewed" ? "ok" : "warn"}>
                {identity.human_review === "human_reviewed" ? "human reviewed" : "human review pending"}
              </Badge>
            </div>
          }
        />
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-2xs text-ink-faint">
          <span>
            knowledge base: <span className="font-mono">{identity.kb_id}</span>
          </span>
          <span>
            content fingerprint: <span className="font-mono">{identity.fingerprint}</span>
          </span>
          <span>
            derived from: <span className="font-mono">{identity.derived_from || "—"}</span>
          </span>
          <span>questions: {identity.question_count}</span>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <div className="h-1.5 min-w-40 flex-1 overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: `${total ? (reviewed / total) * 100 : 0}%` }}
            />
          </div>
          <span className="text-2xs text-ink-muted">
            {reviewed} / {total} reviewed · {packet.completeness.approved} approved ·{" "}
            {packet.completeness.ambiguous} ambiguous · {packet.completeness.insufficient_evidence} insufficient evidence
          </span>
        </div>
        {packet.warnings.length > 0 && (
          <div className="mt-2 space-y-0.5">
            {packet.warnings.map((w, i) => (
              <p key={i} className="flex items-start gap-1 text-2xs text-warn">
                <ShieldQuestion className="mt-0.5 h-3 w-3 shrink-0" />
                <span>{w}</span>
              </p>
            ))}
          </div>
        )}
        {err && <p className="mt-1 text-2xs text-bad">{err}</p>}
      </Panel>

      <Panel>
        <PanelHeader
          title={
            detail
              ? `Question ${index + 1} of ${total} — ${detail.summary.question_id}`
              : `Question ${index + 1} of ${total}`
          }
          right={
            <div className="flex items-center gap-1.5">
              <Button
                variant="ghost"
                onClick={() => setIndex((i) => Math.max(0, i - 1))}
                disabled={index === 0}
              >
                <ChevronLeft className="mr-1 h-3.5 w-3.5" /> Previous
              </Button>
              <Button
                variant="ghost"
                onClick={() => setIndex((i) => Math.min(total - 1, i + 1))}
                disabled={index >= total - 1}
              >
                Next <ChevronRight className="ml-1 h-3.5 w-3.5" />
              </Button>
            </div>
          }
        />

        <div className="mb-3 flex flex-wrap gap-1.5">
          {packet.questions.map((q, i) => (
            <button
              key={q.question_id}
              type="button"
              onClick={() => setIndex(i)}
              title={`${q.question_id}: ${q.question}`}
              className={cn(
                "h-5 w-5 rounded-sm border text-2xs transition-colors",
                i === index ? "border-accent ring-1 ring-accent" : "border-line",
                q.state === "approved" && "bg-ok/40",
                q.state === "rejected" && "bg-bad/40",
                (q.state === "ambiguous" || q.state === "insufficient_evidence") && "bg-warn/40",
                q.state === "in_review" && "bg-accent/30"
              )}
            >
              {!["approved", "in_review", "rejected", "ambiguous", "insufficient_evidence"].includes(q.state) && (
                <span className="text-ink-faint">·</span>
              )}
            </button>
          ))}
        </div>

        {detail ? (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)_minmax(0,2fr)]">
            <div className="rounded-md border border-line bg-surface-2 p-3">
              <div className="section-label mb-2">Ground truth (human-authored)</div>
              <AuthoringPanel
                kbId={kbId}
                benchmarkPath={DEFAULT_ANSWER_BENCHMARK_PATH}
                detail={detail}
                onSaved={refresh}
              />
            </div>
            <div className="rounded-md border border-line bg-surface-2 p-3">
              <div className="section-label mb-2">Source evidence (what the corpus actually says)</div>
              <div className="space-y-2.5">
                {detail.evidence.length === 0 ? (
                  <p className="text-2xs text-ink-faint">this question names no required evidence</p>
                ) : (
                  detail.evidence.map((e) => <EvidenceCard key={e.chunk_id} item={e} />)
                )}
              </div>
            </div>
            <div className="rounded-md border border-line bg-surface-2 p-3">
              <div className="section-label mb-2">Human review</div>
              <ReviewPanel
                kbId={kbId}
                benchmarkPath={DEFAULT_ANSWER_BENCHMARK_PATH}
                detail={detail}
                onSaved={refresh}
              />
            </div>
          </div>
        ) : (
          <p className="text-sm text-ink-faint">loading question evidence…</p>
        )}
      </Panel>

      <FreezePanel
        kbId={kbId}
        benchmarkPath={DEFAULT_ANSWER_BENCHMARK_PATH}
        packet={packet}
        onFrozen={refresh}
      />
    </div>
  );
}
