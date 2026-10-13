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

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  FileText,
  Filter,
  Keyboard,
  Lock,
  RefreshCw,
  Save,
  ShieldQuestion,
  Sparkles,
  Undo2,
} from "lucide-react";
import {
  benchmarkReviewApi,
  answerEvaluationApi,
  DEFAULT_ANSWER_BENCHMARK_PATH,
  BenchmarkQualityAudit,
  BenchmarkQualityFinding,
  BenchmarkQuestionDetail,
  BenchmarkReviewPacket,
  DIMENSION_VERDICTS,
  DraftSuggestion,
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

/** Where the reviewer's own identity is remembered between visits. */
const REVIEWER_KEY = "ragforge.answerReview.reviewer";
/**
 * Browser-local key for an IN-PROGRESS authoring form, one per question.
 *
 * This is the "save and resume" mechanism, and it is deliberately browser-only:
 * it is never sent to the server, never becomes an annotation, and the UI says
 * so. Only "Save reference answer" writes to the database, and only a named
 * human pressing it can create ground truth.
 */
const localDraftKey = (kbId: string, questionId: string) =>
  `ragforge.answerReview.draft.${kbId}.${questionId}`;

type LocalAuthoringDraft = {
  reference: string;
  points: string;
  elements: string;
  note: string;
  savedAt: string;
};

function readLocal<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeLocal(key: string, value: unknown): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable (private mode) — the form still works, just not resumed */
  }
}

function clearLocal(key: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* nothing to do */
  }
}

type AuthoringForm = Pick<LocalAuthoringDraft, "reference" | "points" | "elements" | "note">;

/** Split a textarea into trimmed, non-empty lines. */
function linesOf(value: string): string[] {
  return value
    .split("\n")
    .map((p) => p.trim())
    .filter(Boolean);
}

/** True when the in-progress form differs from what is stored on the server. */
function formDiffers(
  form: AuthoringForm,
  stored: { expected_answer: string; key_points: string[]; acceptable_answer_elements: string[] } | null
): boolean {
  const empty: AuthoringForm = { reference: "", points: "", elements: "", note: "" };
  const base: AuthoringForm = stored
    ? {
        reference: stored.expected_answer ?? "",
        points: (stored.key_points ?? []).join("\n"),
        elements: (stored.acceptable_answer_elements ?? []).join("\n"),
        note: "",
      }
    : empty;
  return (
    form.reference !== base.reference ||
    form.points !== base.points ||
    form.elements !== base.elements
  );
}

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
/**
 * V11 draft assist. Produces a STARTING POINT for the human author.
 *
 * Deliberately placed in the ground-truth column with its machine origin
 * shouted, not whispered: it must be impossible to mistake a draft for a
 * reviewed reference answer. "Use as starting point" only fills the editable
 * fields above — nothing is stored until the author saves it under their own
 * name, and no draft can satisfy the approval gate on its own.
 */
function DraftAssistPanel({
  kbId,
  benchmarkPath,
  questionId,
  onUse,
  auditFindings,
}: {
  kbId: string;
  benchmarkPath: string;
  questionId: string;
  onUse: (draft: DraftSuggestion) => void;
  auditFindings: BenchmarkQualityFinding[];
}) {
  const [draft, setDraft] = useState<DraftSuggestion | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [history, setHistory] = useState<DraftSuggestion[]>([]);
  const [showCaveats, setShowCaveats] = useState(false);

  useEffect(() => {
    setDraft(null);
    setErr(null);
    let alive = true;
    (async () => {
      try {
        const h = await benchmarkReviewApi.draftHistory(kbId, questionId, benchmarkPath);
        if (alive) setHistory(h);
      } catch {
        if (alive) setHistory([]);
      }
    })();
    return () => {
      alive = false;
    };
  }, [kbId, questionId, benchmarkPath]);

  const generate = async () => {
    setBusy(true);
    setErr(null);
    try {
      const d = await benchmarkReviewApi.generateDraft(kbId, questionId, {
        benchmark_path: benchmarkPath,
      });
      setDraft(d);
      setHistory((h) => [...h, d]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const statusTone =
    draft?.status === "draft_ready" ? "neutral" : draft ? "warn" : "neutral";

  return (
    <div className="rounded border border-warn/40 bg-warn/5 px-2.5 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="section-label text-warn">Draft assist (machine-generated)</span>
        <Button variant="ghost" onClick={generate} loading={busy}>
          <Sparkles className="mr-1 h-3 w-3" />
          {draft ? "Regenerate draft" : "Generate draft"}
        </Button>
      </div>
      <p className="mt-1 text-2xs text-ink-muted">
        Assembles a starting point from <strong>verbatim sentences of the evidence</strong> in the
        centre panel. It adds no outside knowledge, it is not ground truth, and it cannot be
        approved. You still author, sign and save the reference answer.
      </p>

      {err && <p className="mt-1.5 text-2xs text-bad">{err}</p>}

      {!draft && history.length > 0 && (
        <p className="mt-1.5 text-2xs text-ink-faint">
          {history.length} draft(s) already generated for this question — generate again to see the
          latest, or check the history below.
        </p>
      )}

      {draft && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone="warn">{draft.label}</Badge>
            <Badge tone={statusTone as "warn" | "neutral"}>
              {draft.status.replace(/_/g, " ")}
            </Badge>
            <Badge tone="neutral">
              coverage:{" "}
              {draft.coverage_measured
                ? `${Math.round(draft.coverage * 100)}% (IDF-weighted)`
                : "UNKNOWN (no corpus statistics)"}
            </Badge>
            {draft.agrees_with_benchmark_quote === true && (
              <Badge tone="ok">matches the sentence recorded in the benchmark</Badge>
            )}
            {draft.agrees_with_benchmark_quote === false && (
              <Badge tone="warn">differs from the recorded sentence — compare both</Badge>
            )}
          </div>

          <p className="text-2xs text-ink-muted">{draft.reason}</p>

          {draft.expected_answer_draft ? (
            <div>
              <div className="section-label">Draft reference answer (verbatim corpus sentences)</div>
              <p className="mt-0.5 whitespace-pre-wrap rounded border border-line bg-surface-3 px-2 py-1.5 text-2xs text-ink">
                {draft.expected_answer_draft}
              </p>
            </div>
          ) : (
            <p className="text-2xs text-warn">
              No draft was produced: the evidence does not state an answer this selector can quote.
              Do not infer one — either cite additional evidence or classify the question.
            </p>
          )}

          {draft.key_points_draft.length > 0 && (
            <div>
              <div className="section-label">
                Draft key points ({draft.key_points_draft.length}) — each traceable to a sentence
              </div>
              <div className="mt-0.5 space-y-1">
                {draft.key_points_draft.map((p, i) => (
                  <div key={i} className="rounded border border-line bg-surface-3 px-2 py-1">
                    <p className="text-2xs text-ink">{p.text}</p>
                    <p className="mt-0.5 font-mono text-2xs text-ink-faint">
                      {p.source_chunk_id} · sentence #{p.source_sentence_index} · term coverage{" "}
                      {(p.coverage * 100).toFixed(0)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {draft.uncovered_terms.length > 0 && (
            <div>
              <div className="section-label">Question terms absent from this evidence</div>
              <p className="text-2xs text-ink-muted">
                {draft.uncovered_terms.join(", ")}
              </p>
              <p className="mt-0.5 text-2xs text-ink-faint">
                Ordered by significance. Several are usually the question&apos;s own phrasing — read
                the evidence before treating any as missing knowledge.
              </p>
            </div>
          )}

          <div className="flex items-center gap-2">
            <Button onClick={() => onUse(draft)} disabled={!draft.expected_answer_draft}>
              Use as starting point
            </Button>
            <span className="text-2xs text-ink-faint">
              Copies the text into the form above. Nothing is saved until you press Save.
            </span>
          </div>

          <button
            type="button"
            className="text-2xs text-ink-faint underline"
            onClick={() => setShowCaveats((s) => !s)}
          >
            {showCaveats ? "hide" : "show"} the draft&apos;s own caveats and generator identity
          </button>
          {showCaveats && (
            <div className="space-y-0.5">
              <p className="font-mono text-2xs text-ink-faint">
                {draft.generator} · {draft.method} · {draft.suggestion_id}
              </p>
              {draft.caveats.map((c, i) => (
                <p key={i} className="text-2xs text-warn">
                  {c}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {auditFindings.length > 0 && (
        <div className="mt-2 border-t border-line pt-2">
          <div className="section-label">Quality audit findings for this question</div>
          {auditFindings.map((f, i) => (
            <p
              key={i}
              className={cn(
                "text-2xs",
                f.severity === "error" ? "text-bad" : f.severity === "warning" ? "text-warn" : "text-ink-muted"
              )}
            >
              <span className="font-mono">{f.severity}</span> {f.code.split("_").join(" ").toLowerCase()}:{" "}
              {f.message}
            </p>
          ))}
        </div>
      )}

      {history.length > 1 && (
        <p className="mt-1.5 text-2xs text-ink-faint">
          Draft history: {history.length} record(s), newest {history[history.length - 1]?.suggestion_id}.
          Append-only — regenerating never overwrites an earlier draft.
        </p>
      )}
    </div>
  );
}

/**
 * V11 pre-freeze benchmark quality audit. READ-ONLY.
 *
 * Reports findings for a human to act on; it never repairs them, and it says
 * which checks could not run instead of reporting them as clean.
 */
function QualityAuditPanel({
  audit,
  busy,
  err,
  onRun,
}: {
  audit: BenchmarkQualityAudit | null;
  busy: boolean;
  err: string | null;
  onRun: () => void;
}) {
  const [showAll, setShowAll] = useState(false);

  const benchmarkFindings = audit?.findings.filter((f) => !f.question_id) ?? [];
  const errorFindings = audit?.findings.filter((f) => f.severity === "error") ?? [];
  const warnFindings = audit?.findings.filter((f) => f.severity === "warning") ?? [];
  // A benchmark-level finding appears in both the severity list and the
  // benchmark list; dedupe by object identity so "show all" never prints one twice.
  const seen = new Set<BenchmarkQualityFinding>();
  const panelFindings = [...errorFindings, ...warnFindings, ...benchmarkFindings].filter(
    (f) => !seen.has(f) && (seen.add(f), true)
  );
  const collapsed = [...errorFindings, ...benchmarkFindings]
    .filter((f, i, arr) => arr.indexOf(f) === i)
    .slice(0, 4);
  const shown = showAll ? panelFindings : collapsed;
  const hiddenCount = panelFindings.length - shown.length;

  return (
    <Panel>
      <PanelHeader
        title="Benchmark quality audit (read-only)"
        right={
          <Button variant="ghost" onClick={onRun} loading={busy}>
            <ShieldQuestion className="mr-1 h-3.5 w-3.5" />
            {audit ? "Re-run audit" : "Run audit"}
          </Button>
        }
      />
      {!audit && !err && (
        <p className="text-2xs text-ink-muted">
          Checks the benchmark against the live corpus before freezing: is every required chunk
          still present with a matching content hash, does the pinned evidence actually cover each
          question, are any questions near-duplicates, and are authored key points supported by the
          evidence they cite. It reports problems for you to fix — it never edits the benchmark.
        </p>
      )}
      {err && <p className="text-2xs text-bad">{err}</p>}
      {audit && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={audit.error_count > 0 ? "bad" : audit.warning_count > 0 ? "warn" : "ok"}>
              {audit.error_count} error · {audit.warning_count} warning · {audit.info_count} info
            </Badge>
            <Badge tone={audit.ready_to_freeze ? "ok" : "warn"}>
              {audit.ready_to_freeze ? "audit: ready to freeze" : "audit: not ready (see findings)"}
            </Badge>
            <span className="text-2xs text-ink-faint">
              {audit.checks_run.length} checks ran · {audit.checks_not_run.length} could not run
            </span>
          </div>
          <p className="text-2xs text-ink">{audit.verdict}</p>

          {shown.map((f, i) => (
            <p
              key={i}
              className={cn(
                "flex items-start gap-1 text-2xs",
                f.severity === "error" ? "text-bad" : "text-warn"
              )}
            >
              <CircleAlert className="mt-0.5 h-3 w-3 shrink-0" />
              <span>
                <span className="font-mono">
                  [{f.severity}] {f.question_id || "benchmark"} {f.code}
                </span>{" "}
                {f.message}
              </span>
            </p>
          ))}
          {!showAll && hiddenCount > 0 && (
            <button type="button" className="text-2xs text-ink-faint underline" onClick={() => setShowAll(true)}>
              show all {panelFindings.length} finding(s) — {hiddenCount} hidden, including every
              question-level finding
            </button>
          )}

          {audit.checks_not_run.length > 0 && (
            <div className="rounded border border-line bg-surface-3 px-2 py-1.5">
              <div className="section-label">Checks that did NOT run</div>
              {audit.checks_not_run.map((c, i) => (
                <p key={i} className="text-2xs text-ink-muted">
                  {c}
                </p>
              ))}
            </div>
          )}
          <div className="rounded border border-line bg-surface-3 px-2 py-1.5">
            <div className="section-label">What this audit cannot decide</div>
            {audit.limitations.map((l, i) => (
              <p key={i} className="text-2xs text-ink-faint">
                {l}
              </p>
            ))}
          </div>
          <p className="text-2xs text-ink-faint">
            {audit.audited_by} · {audit.audited_at}
          </p>
        </div>
      )}
    </Panel>
  );
}

function AuthoringPanel({
  kbId,
  benchmarkPath,
  detail,
  onSaved,
  auditFindings,
  reviewer,
}: {
  kbId: string;
  benchmarkPath: string;
  detail: BenchmarkQuestionDetail;
  onSaved: () => void;
  auditFindings: BenchmarkQualityFinding[];
  /** Shared, stored reviewer identity. Authorship is attributed to it. */
  reviewer: string;
}) {
  const effective = detail.effective_annotation;
  const questionId = detail.summary.question_id;
  const [reference, setReference] = useState(effective?.expected_answer ?? "");
  const [points, setPoints] = useState((effective?.key_points ?? []).join("\n"));
  const [elements, setElements] = useState(
    (effective?.acceptable_answer_elements ?? []).join("\n")
  );
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  //: When the form was filled from a browser-local unsaved edit, its timestamp.
  const [resumedFrom, setResumedFrom] = useState<string | null>(null);

  const storedSummary = useMemo(
    () =>
      effective
        ? {
            expected_answer: effective.expected_answer,
            key_points: effective.key_points,
            acceptable_answer_elements: effective.acceptable_answer_elements,
          }
        : null,
    [effective]
  );

  const storageKey = localDraftKey(kbId, questionId);
  const form: AuthoringForm = { reference, points, elements, note };
  const dirty = formDiffers(form, storedSummary);

  // Resume: a browser-local unsaved edit wins over the stored annotation ONLY
  // when it differs from it, and the reviewer is told what was restored.
  useEffect(() => {
    const local = readLocal<LocalAuthoringDraft>(localDraftKey(kbId, questionId));
    if (local && formDiffers(local, storedSummary)) {
      setReference(local.reference ?? "");
      setPoints(local.points ?? "");
      setElements(local.elements ?? "");
      setNote(local.note ?? "");
      setResumedFrom(local.savedAt ?? null);
    } else {
      setReference(storedSummary?.expected_answer ?? "");
      setPoints((storedSummary?.key_points ?? []).join("\n"));
      setElements((storedSummary?.acceptable_answer_elements ?? []).join("\n"));
      setNote("");
      setResumedFrom(null);
    }
    setProblems([]);
    setErr(null);
    setBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbId, questionId, effective?.annotation_id]);

  // Save-and-resume: keep the in-progress form in THIS BROWSER only. No request
  // is made, nothing becomes an annotation, and the UI says so next to it.
  useEffect(() => {
    if (dirty) {
      writeLocal(storageKey, {
        ...form,
        savedAt: new Date().toISOString(),
      } satisfies LocalAuthoringDraft);
    } else {
      clearLocal(storageKey);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbId, questionId, reference, points, elements, note, dirty]);

  const discardLocal = () => {
    clearLocal(storageKey);
    setReference(storedSummary?.expected_answer ?? "");
    setPoints((storedSummary?.key_points ?? []).join("\n"));
    setElements((storedSummary?.acceptable_answer_elements ?? []).join("\n"));
    setNote("");
    setResumedFrom(null);
  };

  const save = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await benchmarkReviewApi.authorGroundTruth(
        kbId,
        questionId,
        {
          benchmark_path: benchmarkPath,
          author: reviewer.trim(),
          expected_answer: reference,
          key_points: linesOf(points),
          acceptable_answer_elements: linesOf(elements),
          evidence_chunk_ids: detail.evidence.map((e) => e.chunk_id),
          note: note.trim(),
        }
      );
      setProblems(res.problems);
      clearLocal(storageKey);
      setResumedFrom(null);
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

      <div className="rounded border border-line bg-surface-3 px-2.5 py-2">
        <div className="section-label mb-1">Intended scope &amp; what the artifact pins</div>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <ProvenanceChip label="subdomain" value={detail.summary.subdomain} />
          <ProvenanceChip label="answerability" value={detail.summary.answerability} />
          <ProvenanceChip
            label="expected grounding"
            value={detail.summary.expected_grounding_state || "not recorded"}
          />
          <ProvenanceChip
            label="abstention"
            value={detail.summary.abstention_required ? "required" : "not required"}
          />
          <ProvenanceChip
            label="units"
            value={detail.summary.requires_units ? "required" : "not required"}
          />
          <ProvenanceChip
            label="numeric tolerance"
            value={
              detail.summary.numerical_tolerance === null
                ? null
                : detail.summary.numerical_tolerance
            }
          />
        </div>
        {Object.keys(detail.summary.citation_requirements ?? {}).length > 0 && (
          <p className="mt-1 text-2xs text-ink-muted">
            citations:{" "}
            {Object.entries(detail.summary.citation_requirements)
              .filter(([k]) => k !== "note")
              .map(([k, v]) => `${k.replace(/_/g, " ")}=${String(v)}`)
              .join(" · ")}
            {typeof detail.summary.citation_requirements.note === "string" && (
              <span className="text-ink-faint"> — {detail.summary.citation_requirements.note}</span>
            )}
          </p>
        )}
        {detail.summary.artifact_provenance && (
          <p className="mt-1 whitespace-pre-wrap text-2xs text-ink-faint">
            pinned: {detail.summary.artifact_provenance}
          </p>
        )}
        {(detail.summary.artifact_reference_answer || detail.summary.artifact_key_points.length > 0) && (
          <p className="mt-1 text-2xs text-ink-faint">
            the artifact also carries an unreviewed reference answer/key points of its own; they are
            informational and are not ground truth.
          </p>
        )}
      </div>

      <div className="rounded border border-line bg-surface-3 px-2.5 py-1.5">
        <span className="text-2xs text-ink-faint">Authoring as </span>
        {reviewer.trim() ? (
          <span className="font-mono text-2xs text-ink">{reviewer.trim()}</span>
        ) : (
          <span className="text-2xs text-warn">
            set your reviewer identity in the right-hand panel first — an unattributed
            reference answer cannot be distinguished from an agent draft
          </span>
        )}
      </div>

      {resumedFrom && (
        <div className="rounded border border-accent/40 bg-accent/5 px-2.5 py-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-2xs text-ink-muted">
              Restored an unsaved edit you left on this question ({resumedFrom.slice(0, 19)}). It is
              stored <strong>in this browser only</strong> — nothing was saved to the server and it
              is not ground truth until you save it.
            </span>
            <Button variant="ghost" onClick={discardLocal}>
              <Undo2 className="mr-1 h-3 w-3" /> Discard local edit
            </Button>
          </div>
        </div>
      )}

      <DraftAssistPanel
        kbId={kbId}
        benchmarkPath={benchmarkPath}
        questionId={detail.summary.question_id}
        auditFindings={auditFindings}
        onUse={(draft) => {
          setReference(draft.expected_answer_draft);
          setPoints(draft.key_points_draft.map((p) => p.text).join("\n"));
          setNote(
            `edited from draft ${draft.suggestion_id} (${draft.generator}); verified against the evidence`
          );
        }}
      />

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
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={save} loading={busy} disabled={!reviewer.trim()}>
          <Save className="mr-1 h-3.5 w-3.5" /> Save reference answer
        </Button>
        {dirty && (
          <Badge tone="warn">unsaved edit — kept in this browser, not saved to the server</Badge>
        )}
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
  reviewer,
  onReviewerChange,
}: {
  kbId: string;
  benchmarkPath: string;
  detail: BenchmarkQuestionDetail;
  onSaved: () => void;
  /** Shared reviewer identity; every recorded review is attributed to it. */
  reviewer: string;
  onReviewerChange: (value: string) => void;
}) {
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
        <span className="section-label">Reviewer identity (required, remembered)</span>
        <input
          value={reviewer}
          onChange={(e) => onReviewerChange(e.target.value)}
          placeholder="e.g. ada@example.edu"
          className="mt-0.5 w-full rounded border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink"
        />
        <span className="mt-1 block text-2xs text-ink-faint">
          Stored with every annotation and review you record, and kept in this browser so you do not
          have to retype it per question. An anonymous review cannot be audited, so it is refused.
        </span>
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
  // V11: the quality audit is run on demand, and its per-question findings are
  // shown next to the question they concern.
  const [audit, setAudit] = useState<BenchmarkQualityAudit | null>(null);
  const [auditBusy, setAuditBusy] = useState(false);
  const [auditErr, setAuditErr] = useState<string | null>(null);

  // --- V16: one reviewer identity, remembered between visits ---------------
  const [reviewer, setReviewer] = useState("");
  useEffect(() => {
    setReviewer(readLocal<string>(REVIEWER_KEY) ?? "");
  }, []);
  const changeReviewer = useCallback((value: string) => {
    setReviewer(value);
    if (value.trim()) writeLocal(REVIEWER_KEY, value);
    else clearLocal(REVIEWER_KEY);
  }, []);

  // --- V16: review queue / state filter + keyboard navigation --------------
  const [queueOnly, setQueueOnly] = useState(false);
  const [stateFilter, setStateFilter] = useState<"all" | QuestionState>("all");
  const [showShortcuts, setShowShortcuts] = useState(false);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  const runAudit = useCallback(async () => {
    if (!kbId) return;
    setAuditBusy(true);
    setAuditErr(null);
    try {
      setAudit(await benchmarkReviewApi.qualityAudit(kbId, DEFAULT_ANSWER_BENCHMARK_PATH));
    } catch (e) {
      setAuditErr(e instanceof Error ? e.message : String(e));
    } finally {
      setAuditBusy(false);
    }
  }, [kbId]);

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

  // --- V16 review queue, navigation and keyboard shortcuts ------------------
  // "Awaiting a decision" means no human verdict yet: pending (nothing
  // authored) and in_review (authored, not yet judged).
  const visibleQuestions = useMemo(() => {
    if (!packet) return [];
    return packet.questions.filter((q) =>
      queueOnly
        ? q.state === "pending" || q.state === "in_review"
        : stateFilter === "all" || q.state === stateFilter
    );
  }, [packet, queueOnly, stateFilter]);
  const visibleIds = useMemo(
    () => visibleQuestions.map((q) => q.question_id),
    [visibleQuestions]
  );
  //: -1 when the open question is not in the filtered queue (the effect below
  //: then lands on the first visible question).
  const positionInVisible = visibleIds.indexOf(currentId ?? "");
  const stateCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const q of packet?.questions ?? []) counts[q.state] = (counts[q.state] ?? 0) + 1;
    return counts;
  }, [packet]);
  const goByOffset = useCallback(
    (offset: number) => {
      if (!packet) return;
      const nextId = visibleIds[positionInVisible + offset];
      if (!nextId) return;
      const nextIndex = packet.questions.findIndex((q) => q.question_id === nextId);
      if (nextIndex >= 0) setIndex(nextIndex);
    },
    [packet, visibleIds, positionInVisible]
  );

  // If the filter changes under the reviewer, land on a question they can see.
  useEffect(() => {
    if (!packet || visibleIds.length === 0) return;
    if (currentId && visibleIds.includes(currentId)) return;
    const firstIndex = packet.questions.findIndex((q) => q.question_id === visibleIds[0]);
    if (firstIndex >= 0) setIndex(firstIndex);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [packet, queueOnly, stateFilter]);

  // Keyboard navigation. Navigation only: no shortcut records a decision, so
  // nothing here can weaken the review requirement.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        target?.isContentEditable
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      switch (event.key) {
        case "ArrowRight":
        case "j":
          event.preventDefault();
          goByOffset(1);
          break;
        case "ArrowLeft":
        case "k":
          event.preventDefault();
          goByOffset(-1);
          break;
        case "?":
          setShowShortcuts((s) => !s);
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goByOffset]);

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
                onClick={() => goByOffset(-1)}
                disabled={positionInVisible <= 0}
              >
                <ChevronLeft className="mr-1 h-3.5 w-3.5" /> Previous
              </Button>
              <Button
                variant="ghost"
                onClick={() => goByOffset(1)}
                disabled={positionInVisible >= visibleIds.length - 1}
              >
                Next <ChevronRight className="ml-1 h-3.5 w-3.5" />
              </Button>
            </div>
          }
        />

        {/* V16 review queue: filter to the questions still awaiting a human
            decision, jump by dot, and see where you are in the queue. */}
        <div className="mb-3 space-y-2">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <span className="section-label flex items-center gap-1">
              <Filter className="h-3 w-3" /> Queue
            </span>
            <label className="flex items-center gap-1.5 text-2xs text-ink-muted">
              <input
                type="checkbox"
                checked={queueOnly}
                onChange={(e) => {
                  const on = e.target.checked;
                  setQueueOnly(on);
                  if (on) setStateFilter("all");
                }}
              />
              awaiting a decision only (
              {packet.completeness.pending + packet.completeness.in_review})
            </label>
            <select
              value={stateFilter}
              onChange={(e) => {
                setStateFilter(e.target.value as "all" | QuestionState);
                setQueueOnly(false);
              }}
              disabled={queueOnly}
              className="rounded border border-line bg-surface-3 px-1.5 py-0.5 text-2xs text-ink disabled:opacity-40"
            >
              <option value="all">all states ({total})</option>
              {QUESTION_STATES.map((s) => (
                <option key={s} value={s}>
                  {s.replace(/_/g, " ")} ({stateCounts[s] ?? 0})
                </option>
              ))}
            </select>
            <span className="text-2xs text-ink-faint">
              showing {visibleQuestions.length} of {total}
              {positionInVisible >= 0 && ` · ${positionInVisible + 1} of ${visibleIds.length} in queue`}
            </span>
            <button
              type="button"
              onClick={() => setShowShortcuts((s) => !s)}
              className="ml-auto flex items-center gap-1 text-2xs text-ink-faint underline-offset-2 hover:underline"
            >
              <Keyboard className="h-3 w-3" /> keyboard shortcuts
            </button>
          </div>

          {showShortcuts && (
            <div className="rounded border border-line bg-surface-3 px-2.5 py-1.5 text-2xs text-ink-muted">
              <p>
                <span className="font-mono">←</span> / <span className="font-mono">k</span> previous
                question · <span className="font-mono">→</span> /{" "}
                <span className="font-mono">j</span> next question ·{" "}
                <span className="font-mono">?</span> toggle this help
              </p>
              <p className="mt-0.5 text-ink-faint">
                Navigation only — no shortcut records a decision, approves a question or saves a
                reference answer. Shortcuts are ignored while you are typing in a field.
              </p>
            </div>
          )}

          {visibleQuestions.length === 0 ? (
            <p className="text-2xs text-ink-muted">
              No question matches this filter. Clear the filter to see the {total} question(s) in the
              benchmark.
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {visibleQuestions.map((q) => {
                const qi = packet.questions.findIndex((p) => p.question_id === q.question_id);
                return (
                  <button
                    key={q.question_id}
                    type="button"
                    onClick={() => setIndex(qi)}
                    title={`${q.question_id}: ${q.question} — ${q.state}`}
                    className={cn(
                      "h-5 w-5 rounded-sm border text-2xs transition-colors",
                      qi === index ? "border-accent ring-1 ring-accent" : "border-line",
                      q.state === "approved" && "bg-ok/40",
                      q.state === "rejected" && "bg-bad/40",
                      (q.state === "ambiguous" || q.state === "insufficient_evidence") &&
                        "bg-warn/40",
                      q.state === "in_review" && "bg-accent/30"
                    )}
                  >
                    {!["approved", "in_review", "rejected", "ambiguous", "insufficient_evidence"].includes(q.state) && (
                      <span className="text-ink-faint">·</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
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
                reviewer={reviewer}
                auditFindings={
                  audit?.questions.find((q) => q.question_id === detail.summary.question_id)
                    ?.findings ?? []
                }
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
                reviewer={reviewer}
                onReviewerChange={changeReviewer}
              />
            </div>
          </div>
        ) : (
          <p className="text-sm text-ink-faint">loading question evidence…</p>
        )}
      </Panel>

      <QualityAuditPanel
        audit={audit}
        busy={auditBusy}
        err={auditErr}
        onRun={runAudit}
      />

      <FreezePanel
        kbId={kbId}
        benchmarkPath={DEFAULT_ANSWER_BENCHMARK_PATH}
        packet={packet}
        onFrozen={refresh}
      />
    </div>
  );
}
