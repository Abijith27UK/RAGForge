export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export type SourceMode = "external" | "user_provided" | "mixed";
export type DocumentStatus =
  | "uploaded" | "parsing" | "parsed" | "chunking" | "indexing" | "ready" | "failed";

export interface KnowledgeBase {
  id: string;
  name: string;
  domain: string;
  purpose: string;
  target_audience: string;
  depth: string;
  status: string;
  source_mode?: SourceMode;
  version?: number;
  last_build_at?: string | null;
  vector_backend?: string;
  chunking_strategy?: string;
  chunking_config?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface KBOverview {
  kb: KnowledgeBase;
  documents: number;
  documents_ready: number;
  documents_failed: number;
  user_provided_documents: number;
  external_documents: number;
  chunks: number;
  vectors: number | null;
  sources: number;
  sources_user_provided: number;
  sources_discovered: number;
  embedding_model: string;
  vector_backend: string;
  vector_store_status: string;
  chunking_strategy: string;
  chunking_config: Record<string, unknown>;
  last_build_at: string | null;
  last_build_status: string;
  version: number;
  build_status: string;
  evaluation_status:
    | "not_configured" | "questions_only" | "benchmark_draft" | "benchmark_frozen" | "evaluated";
  benchmark_versions: number;
  frozen_benchmark_versions: number;
  evaluation_questions: number;
  evaluation_runs: number;
  last_evaluation: {
    id: string;
    started_at: string;
    top_k: number;
    questions_evaluated: number;
    recall_at_k: number | null;
    mrr: number | null;
    ndcg: number | null;
    benchmark_version: string | null;
  } | null;
  /** Always false: ground truth is optional and never blocks READY. */
  evaluation_required: boolean;
}

export interface DomainSpec {
  kb_id: string;
  domain: string;
  description: string;
  subdomains: string[];
  key_concepts: string[];
  entities: string[];
  terminology: string[];
  knowledge_requirements: { area: string; description: string; priority: string }[];
  recommended_source_categories: string[];
  generated_by: string;
  is_mock: boolean;
}

export interface QualitySignals {
  authority: number;
  relevance: number;
  recency: number;
  source_type: number;
  accessibility: number;
  duplication: number;
  evidence_quality: number;
  /** V4 user-file integrity signals (absent on web-scored sources). */
  file_validity?: number;
  content_extraction?: number;
  structure?: number;
  user_relevance?: number;
}

export interface Source {
  id: string;
  kb_id: string;
  url: string;
  title: string | null;
  source_type: string;
  publisher: string | null;
  discovered_via: string;
  decision: "PENDING" | "ACCEPT" | "REVIEW" | "REJECT";
  trust_score: number | null;
  quality: {
    signals: QualitySignals;
    weights: Record<string, number>;
    score: number;
    decision: string;
    reasons: string[];
    warnings: string[];
    assessed_by: string;
  } | null;
  notes: string | null;
  /** V4: true for files/URLs the user supplied, false for discovered sources. */
  user_provided?: boolean;
  provenance?: "discovered" | "user_upload" | "user_url";
  file_name?: string | null;
  file_size?: number | null;
  integrity?: Record<string, unknown> | null;
}

export interface Document {
  id: string;
  kb_id: string;
  source_id: string;
  url: string;
  title: string | null;
  source_type: string;
  publisher: string | null;
  content_hash: string;
  text_length: number;
  page_count: number | null;
  parse_error: string | null;
  ingestion_timestamp: string;
  /** V4 document library fields. */
  status?: DocumentStatus;
  user_provided?: boolean;
  document_version?: number;
  replaces_document_id?: string | null;
  file_name?: string | null;
  file_size?: number | null;
  mime_type?: string | null;
  raw_file_path?: string | null;
  parser?: string | null;
  slide_count?: number | null;
  section_count?: number | null;
  chunk_count?: number;
  parse_metadata?: Record<string, unknown>;
  indexed_at?: string | null;
}

export interface DocumentDetail {
  document: Document;
  source: Source | null;
  chunk_count: number;
  integrity: Record<string, unknown> | null;
  kb_version: number;
  retrieval_enabled: boolean;
}

export interface DocumentLibrary {
  kb_id: string;
  documents: number;
  chunks: number;
  user_provided: number;
  external: number;
  by_status: Record<string, number>;
  supported_extensions: string[];
  max_upload_bytes: number;
}

export interface UploadedFileResult {
  file_name: string;
  size_bytes: number;
  status: "uploaded" | "duplicate" | "rejected" | "failed" | "indexed";
  message: string;
  document: Document | null;
  source: Source | null;
  duplicate_of: string | null;
  integrity: Record<string, unknown> | null;
}

export interface UploadResult {
  kb_id: string;
  source_mode: SourceMode;
  files: UploadedFileResult[];
  uploaded: number;
  duplicates: number;
  rejected: number;
  failed: number;
  indexed: boolean;
  indexing: {
    documents_indexed?: number;
    chunk_count?: number;
    vectors_indexed?: number;
    stale_vectors_removed?: number;
    /** Human label; "unknown" when the vector store could not confirm a count. */
    stale_vectors_removed_label?: string;
    seconds?: number;
    error?: string;
  } | null;
}

export interface Chunk {
  id: string;
  document_id: string;
  kb_id: string;
  chunk_index: number;
  text: string;
  source_url: string | null;
  source_title: string | null;
  source_type: string | null;
  publisher: string | null;
  document_title: string | null;
  section: string | null;
  section_path: string | null;
  page: number | null;
  slide: number | null;
  slide_title: string | null;
  domain: string | null;
  subdomain: string | null;
  trust_score: number | null;
  content_hash: string;
  char_count: number;
  chunking_strategy?: string | null;
  chunking_config?: Record<string, unknown>;
  document_version?: number | null;
  user_provided?: boolean;
  kb_version?: number | null;
}

export interface DocumentText {
  document_id: string;
  file_name: string | null;
  parser: string | null;
  char_count: number;
  offset: number;
  returned: number;
  truncated: boolean;
  text: string;
}

export interface RetrievalResult {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  provenance: Record<string, string | number | boolean | null>;
}

export interface RetrievalResponse {
  query: string;
  top_k: number;
  results: RetrievalResult[];
  embedding_model: string;
  retrieval_backend: string;
  error: string | null;
}

export type QuestionLifecycle = "DRAFT" | "REVIEW" | "APPROVED" | "FROZEN";

export interface EvaluationQuestion {
  id: string;
  kb_id: string;
  question: string;
  expected_chunk_ids: string[];
  expected_document_ids: string[];
  expected_keywords: string[];
  generated_by: string;
  notes?: string;
  // --- V3 Phase A lifecycle ---
  status: QuestionLifecycle;
  author: string;
  reviewer: string | null;
  created_at: string;
  reviewed_at: string | null;
  revision: number;
  supersedes: string | null;
  provenance: Record<string, unknown>;
}

export interface BenchmarkVersion {
  id: string;
  kb_id: string;
  version: string;
  label: string;
  status: "DRAFT" | "APPROVED" | "FROZEN";
  question_ids: string[];
  questions_snapshot: EvaluationQuestion[];
  created_by: string;
  created_at: string;
  frozen_at: string | null;
  notes: string;
}

export interface EvaluationRun {
  id: string;
  kb_id: string;
  config: { top_k: number; question_ids?: string[]; benchmark_version?: string | null };
  aggregate: {
    recall_at_k: number | null;
    precision_at_k: number | null;
    mrr: number | null;
    ndcg: number | null;
    doc_recall_at_k: number | null;
    doc_precision_at_k: number | null;
    doc_mrr: number | null;
    doc_ndcg: number | null;
    questions_evaluated: number;
    questions_with_explicit_gt: number;
    questions_with_keyword_fallback: number;
    questions_skipped_no_gt: number;
    strict_mode: boolean;
    run_label: string;
    notes: string;
  };
  per_question: {
    question_id: string;
    question: string;
    recall_at_k: number | null;
    precision_at_k: number | null;
    mrr: number | null;
    ndcg: number | null;
    num_relevant_found: number;
    num_relevant_total: number;
    doc_recall_at_k: number | null;
    doc_precision_at_k: number | null;
    doc_mrr: number | null;
    doc_ndcg: number | null;
    doc_num_relevant_found: number;
    doc_num_relevant_total: number;
    note: string;
  }[];
  retrieval_backend: string;
  embedding_model: string;
  started_at: string;
}

export interface BuildRun {
  id: string;
  kb_id: string;
  stages: {
    stage: string;
    status: string;
    message: string;
    items_processed: number;
  }[];
  status: string;
  started_at: string;
  finished_at: string | null;
}

export interface SystemStatus {
  qdrant: { url: string; reachable: boolean; error?: string };
  embedding: { provider: string; model: string };
  llm: { provider: string; model: string | null; allow_mock: boolean };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BACKEND_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...init,
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep default */ }
    throw new Error(detail);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

export const api = {
  // system
  systemStatus: () => request<SystemStatus>("/api/system/status"),
  health: () => request<{ status: string }>("/api/system/health"),

  // knowledge bases
  createKB: (body: Partial<KnowledgeBase>) =>
    request<KnowledgeBase>("/api/knowledge-bases", { method: "POST", body: JSON.stringify(body) }),
  listKBs: () => request<KnowledgeBase[]>("/api/knowledge-bases"),
  getKB: (id: string) => request<KnowledgeBase>(`/api/knowledge-bases/${id}`),
  kbOverview: (id: string) => request<KBOverview>(`/api/knowledge-bases/${id}/overview`),
  deleteKB: (id: string) => request<void>(`/api/knowledge-bases/${id}`, { method: "DELETE" }),
  buildStatus: (id: string) => request<BuildRun | null>(`/api/knowledge-bases/${id}/build-status`),

  // domain
  analyzeDomain: (id: string) =>
    request<DomainSpec>(`/api/knowledge-bases/${id}/analyze-domain`, { method: "POST" }),
  getDomainSpec: (id: string) => request<DomainSpec>(`/api/knowledge-bases/${id}/domain-spec`),

  // sources
  discoverSources: (id: string, body: { provider: string; query: string; limit?: number }) =>
    request<Source[]>(`/api/knowledge-bases/${id}/discover-sources`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listSources: (id: string) => request<Source[]>(`/api/knowledge-bases/${id}/sources`),
  setDecision: (kbId: string, sourceId: string, decision: string) =>
    request<Source>(`/api/knowledge-bases/${kbId}/sources/${sourceId}/decision`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    }),

  // build
  ingest: (id: string, sourceIds: string[] = []) =>
    request<BuildRun>(`/api/knowledge-bases/${id}/ingest`, {
      method: "POST",
      body: JSON.stringify({ source_ids: sourceIds }),
    }),
  index: (id: string, opts?: { chunker?: string; target_size?: number; overlap?: number }) =>
    request<BuildRun>(`/api/knowledge-bases/${id}/index`, {
      method: "POST",
      body: JSON.stringify(opts ?? {}),
    }),
  listDocuments: (id: string) => request<Document[]>(`/api/knowledge-bases/${id}/documents`),
  listChunks: (id: string, limit = 100) =>
    request<Chunk[]>(`/api/knowledge-bases/${id}/chunks?limit=${limit}`),

  // --- V4: user-provided document library ---
  documentLibrary: (id: string) =>
    request<DocumentLibrary>(`/api/knowledge-bases/${id}/document-library`),
  inspectDocument: (kbId: string, docId: string) =>
    request<DocumentDetail>(`/api/knowledge-bases/${kbId}/documents/${docId}`),
  documentText: (kbId: string, docId: string, limit = 20000) =>
    request<DocumentText>(
      `/api/knowledge-bases/${kbId}/documents/${docId}/text?limit=${limit}`,
    ),
  documentChunks: (kbId: string, docId: string) =>
    request<Chunk[]>(`/api/knowledge-bases/${kbId}/documents/${docId}/chunks`),
  rebuildDocument: (kbId: string, docId: string, reindex = true) =>
    request<BuildRun>(`/api/knowledge-bases/${kbId}/documents/${docId}/rebuild`, {
      method: "POST",
      body: JSON.stringify({ reindex }),
    }),
  indexDocument: (kbId: string, docId: string, opts?: { chunker?: string; target_size?: number; overlap?: number }) =>
    request<BuildRun>(`/api/knowledge-bases/${kbId}/documents/${docId}/index`, {
      method: "POST",
      body: JSON.stringify(opts ?? {}),
    }),
  deleteDocument: (kbId: string, docId: string) =>
    request<{
      deleted: string;
      file_name: string | null;
      /** -1 means the deletion ran but the count could not be confirmed. */
      vectors_removed: number;
      vectors_removed_label: string;
      vectors_removal_confirmed: boolean;
      vector_store_error: string | null;
    }>(
      `/api/knowledge-bases/${kbId}/documents/${docId}`,
      { method: "DELETE" },
    ),

  /** Multipart upload. Progress is reported via the XHR upload event. */
  uploadDocuments: (
    id: string,
    files: File[],
    opts: { index?: boolean; chunker?: string; target_size?: number; overlap?: number } = {},
    onProgress?: (fraction: number) => void,
  ) =>
    new Promise<UploadResult>((resolve, reject) => {
      const form = new FormData();
      files.forEach((f) => form.append("files", f, f.name));
      form.append("index", String(opts.index ?? false));
      form.append("chunker", opts.chunker ?? "section-aware");
      form.append("target_size", String(opts.target_size ?? 1200));
      form.append("overlap", String(opts.overlap ?? 150));

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BACKEND_URL}/api/knowledge-bases/${id}/documents/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let payload: unknown = null;
        try {
          payload = JSON.parse(xhr.responseText);
        } catch {
          payload = null;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress?.(1);
          resolve(payload as UploadResult);
        } else {
          const detail = (payload as { detail?: string } | null)?.detail ?? `${xhr.status}`;
          reject(new Error(detail));
        }
      };
      xhr.onerror = () => reject(new Error("Upload failed: could not reach the backend"));
      xhr.send(form);
    }),

  replaceDocument: (kbId: string, docId: string, file: File, opts: { index?: boolean; reason?: string } = {}) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("index", String(opts.index ?? true));
    if (opts.reason) form.append("reason", opts.reason);
    return request<{ document: Document; superseded: string; indexing: Record<string, unknown> | null }>(
      `/api/knowledge-bases/${kbId}/documents/${docId}/replace`,
      { method: "POST", body: form },
    );
  },

  addUserURLs: (kbId: string, urls: string[]) =>
    request<Source[]>(`/api/knowledge-bases/${kbId}/sources/user-urls`, {
      method: "POST",
      body: JSON.stringify({ urls }),
    }),

  // retrieval
  retrieve: (id: string, query: string, topK = 5) =>
    request<RetrievalResponse>(`/api/knowledge-bases/${id}/retrieve`, {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK }),
    }),

  // evaluation
  addQuestion: (id: string, body: Partial<EvaluationQuestion>) =>
    request<EvaluationQuestion>(`/api/knowledge-bases/${id}/evaluation-questions`, {
      method: "POST",
      body: JSON.stringify({
        question: body.question,
        expected_chunk_ids: body.expected_chunk_ids ?? [],
        expected_document_ids: body.expected_document_ids ?? [],
        expected_keywords: body.expected_keywords ?? [],
        notes: body.notes ?? "",
      }),
    }),
  listQuestions: (id: string) =>
    request<EvaluationQuestion[]>(`/api/knowledge-bases/${id}/evaluation-questions`),
  deleteQuestion: (kbId: string, qId: string) =>
    request<void>(`/api/knowledge-bases/${kbId}/evaluation-questions/${qId}`, { method: "DELETE" }),
  evaluate: (id: string, topK = 5, label = "", allowKeywordFallback = false, benchmarkVersion?: string | null) =>
    request<EvaluationRun>(`/api/knowledge-bases/${id}/evaluate`, {
      method: "POST",
      body: JSON.stringify({
        top_k: topK,
        label,
        allow_keyword_fallback: allowKeywordFallback,
        ...(benchmarkVersion ? { benchmark_version: benchmarkVersion } : {}),
      }),
    }),
  listEvaluationRuns: (id: string) =>
    request<EvaluationRun[]>(`/api/knowledge-bases/${id}/evaluation-runs`),

  // --- V3 Phase A: benchmark lifecycle & versions ---
  setQuestionStatus: (kbId: string, qId: string, status: QuestionLifecycle, reviewer = "") =>
    request<EvaluationQuestion>(
      `/api/knowledge-bases/${kbId}/evaluation-questions/${qId}/status`,
      { method: "POST", body: JSON.stringify({ status, reviewer }) },
    ),
  editQuestion: (kbId: string, qId: string, patch: Partial<EvaluationQuestion>) =>
    request<EvaluationQuestion>(
      `/api/knowledge-bases/${kbId}/evaluation-questions/${qId}`,
      { method: "PATCH", body: JSON.stringify(patch) },
    ),
  createBenchmarkVersion: (
    kbId: string,
    body: { version: string; label?: string; question_ids?: string[]; created_by?: string; notes?: string },
  ) =>
    request<BenchmarkVersion>(`/api/knowledge-bases/${kbId}/benchmark-versions`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listBenchmarkVersions: (kbId: string) =>
    request<BenchmarkVersion[]>(`/api/knowledge-bases/${kbId}/benchmark-versions`),
  getBenchmarkVersion: (kbId: string, bvId: string) =>
    request<BenchmarkVersion>(`/api/knowledge-bases/${kbId}/benchmark-versions/${bvId}`),
  freezeBenchmarkVersion: (kbId: string, bvId: string, reviewer = "") =>
    request<BenchmarkVersion>(
      `/api/knowledge-bases/${kbId}/benchmark-versions/${bvId}/freeze${reviewer ? `?reviewer=${encodeURIComponent(reviewer)}` : ""}`,
      { method: "POST" },
    ),
  deleteBenchmarkVersion: (kbId: string, bvId: string) =>
    request<void>(`/api/knowledge-bases/${kbId}/benchmark-versions/${bvId}`, { method: "DELETE" }),
};

/* ---------------------------------------------------------------------------
 * Corpus engineering (V5): batches, manifest, integrity, repair, versions.
 *
 * Every count below may legitimately be -1, which the backend uses for
 * "unknown / could not be measured". `countLabel` renders that as "unknown"
 * rather than 0 - never substitute a zero we did not measure.
 * ------------------------------------------------------------------------- */

export const countLabel = (n: number | null | undefined): string =>
  n === null || n === undefined || n === -1 ? "unknown" : String(n);

export type IngestionItemStatus =
  | "pending" | "processing" | "complete" | "failed"
  | "duplicate" | "rejected" | "cancelled" | "skipped";

export type IngestionStage =
  | "queued" | "validating" | "parsing" | "chunking"
  | "embedding" | "indexing" | "complete" | "failed" | "cancelled";

export interface IngestionItem {
  id: string;
  batch_id: string;
  kb_id: string;
  item_key: string;
  file_name: string;
  size_bytes: number;
  mime_type: string | null;
  content_hash: string | null;
  status: IngestionItemStatus;
  stage: IngestionStage;
  progress: number;
  error_code: string | null;
  error_message: string | null;
  attempts: number;
  document_id: string | null;
  duplicate_of: string | null;
  parser_used: string | null;
  chunk_count: number;
  vector_count: number;
  warnings: string[];
  created_at: string;
  completed_at: string | null;
}

export interface IngestionBatch {
  id: string;
  kb_id: string;
  status: "pending" | "running" | "complete" | "partial" | "incomplete" | "failed" | "cancelled";
  total_items: number;
  completed_items: number;
  failed_items: number;
  duplicate_items: number;
  rejected_items: number;
  resumed_at: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface IngestionBatchDetail {
  batch: IngestionBatch;
  items: IngestionItem[];
  resumable_items: number;
}

export interface CorpusManifestEntry {
  document_id: string;
  file_name: string | null;
  content_hash: string;
  document_version: number;
  source_mode: string;
  source_url: string | null;
  file_type: string;
  file_size: number | null;
  parser: string | null;
  page_count: number | null;
  slide_count: number | null;
  section_count: number | null;
  text_length: number;
  chunk_count: number;
  vector_count: number;
  embedding_model: string | null;
  embedding_dimension: number | null;
  chunking_strategy: string | null;
  chunking_config_hash: string | null;
  indexed_at: string | null;
  status: string;
  error_message: string | null;
  provenance: Record<string, unknown>;
}

export interface CorpusSummary {
  total_documents: number;
  completed_documents: number;
  failed_documents: number;
  processing_documents: number;
  total_pages: number;
  total_slides: number;
  total_sections: number;
  total_chunks: number;
  total_vectors: number;
  total_corpus_size_bytes: number;
  duplicate_documents: number;
  stale_documents: number;
  orphan_vectors: number;
  embedding_identity: string | null;
  chunking_identity: string | null;
  vector_backend: string | null;
  last_successful_index_at: string | null;
  vector_count_confirmed: boolean;
}

export interface CorpusManifest {
  kb_id: string;
  kb_name: string;
  kb_version: number;
  entries: CorpusManifestEntry[];
  summary: CorpusSummary;
  generated_at: string;
}

export type IntegrityCheck =
  | "missing_vectors" | "orphan_vectors" | "stale_vectors"
  | "embedding_mismatch" | "chunking_mismatch" | "duplicate_documents"
  | "duplicate_chunks" | "failed_documents" | "partial_documents"
  | "provenance_gaps" | "broken_source_references";

export interface IntegrityFinding {
  check: IntegrityCheck;
  severity: "info" | "warning" | "error";
  message: string;
  document_id: string | null;
  chunk_id: string | null;
  vector_id: string | null;
  details: Record<string, unknown>;
}

export interface IntegrityReport {
  kb_id: string;
  overall_status: "HEALTHY" | "WARNING" | "ERROR";
  documents_checked: number;
  chunks_checked: number;
  vectors_checked: number;
  findings: IntegrityFinding[];
  missing_vectors: IntegrityFinding[];
  orphan_vectors: IntegrityFinding[];
  stale_vectors: IntegrityFinding[];
  embedding_mismatches: IntegrityFinding[];
  chunking_mismatches: IntegrityFinding[];
  duplicate_documents: IntegrityFinding[];
  duplicate_chunks: IntegrityFinding[];
  failed_documents: IntegrityFinding[];
  partial_documents: IntegrityFinding[];
  provenance_gaps: IntegrityFinding[];
  broken_source_references: IntegrityFinding[];
  unknown_counts: string[];
  generated_at: string;
  duration_seconds: number;
}

export type RepairAction =
  | "reindex_document" | "reindex_failed_documents" | "remove_orphan_vectors"
  | "rebuild_document_vectors" | "rebuild_kb" | "recompute_embeddings" | "reindex_batch";

export interface RepairPlan {
  action: RepairAction;
  kb_id: string;
  description: string;
  affected_documents: string[];
  expected_chunks: number;
  expected_vectors: number;
  destructive: boolean;
  counts_confirmed: boolean;
}

export interface RepairResult {
  action: RepairAction;
  kb_id: string;
  ok: boolean;
  documents_affected: number;
  chunks_written: number;
  vectors_removed: number;
  vectors_written: number;
  removal_confirmed: boolean;
  message: string;
  errors: string[];
}

export interface CorpusVersion {
  id: string;
  kb_id: string;
  version: string;
  fingerprint: string;
  document_count: number;
  chunk_count: number;
  vector_backend: string;
  embedding_identity: string;
  chunking_identity: string;
  created_at: string;
  note: string | null;
}

export interface CorpusFingerprint {
  kb_id: string;
  fingerprint: string;
  document_count: number;
  chunk_count: number;
  embedding_identity: string | null;
  chunking_identity: string | null;
}

export interface CorpusDiff {
  kb_id: string;
  from_version: string | null;
  to_version: string | null;
  added: unknown[];
  removed: unknown[];
  changed: unknown[];
  unchanged: unknown[];
  identical: boolean;
}

export const corpusApi = {
  createBatch: (kbId: string, files: File[], opts: { index?: boolean; chunker?: string } = {}) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    fd.append("index", String(opts.index ?? true));
    fd.append("chunker", opts.chunker ?? "section-aware");
    return request<IngestionBatchDetail>(
      `/api/knowledge-bases/${kbId}/ingestion-batches`,
      { method: "POST", body: fd },
    );
  },
  listBatches: (kbId: string) =>
    request<IngestionBatch[]>(`/api/knowledge-bases/${kbId}/ingestion-batches`),
  getBatch: (kbId: string, batchId: string) =>
    request<IngestionBatchDetail>(`/api/knowledge-bases/${kbId}/ingestion-batches/${batchId}`),
  resumeBatch: (kbId: string, batchId: string, body: { include_completed?: boolean; retry_failed?: boolean; index?: boolean } = {}) =>
    request<IngestionBatchDetail>(
      `/api/knowledge-bases/${kbId}/ingestion-batches/${batchId}/resume`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  manifest: (kbId: string) =>
    request<CorpusManifest>(`/api/knowledge-bases/${kbId}/corpus-manifest`),
  integrity: (kbId: string) =>
    request<IntegrityReport>(`/api/knowledge-bases/${kbId}/corpus-integrity`),
  health: (kbId: string) =>
    request<{
      overall_status: "HEALTHY" | "WARNING" | "ERROR";
      documents_checked: number;
      chunks_checked: number;
      vectors_checked: number;
      counts: Record<string, number>;
      unknown_counts: string[];
      generated_at: string;
    }>(`/api/knowledge-bases/${kbId}/corpus-health`),

  planRepair: (kbId: string, body: { action: RepairAction; document_ids?: string[]; batch_id?: string }) =>
    request<RepairPlan>(`/api/knowledge-bases/${kbId}/corpus-repair/plan`, {
      method: "POST", body: JSON.stringify(body),
    }),
  runRepair: (kbId: string, body: { action: RepairAction; document_ids?: string[]; batch_id?: string; confirm_action?: RepairAction }) =>
    request<RepairResult>(`/api/knowledge-bases/${kbId}/corpus-repair`, {
      method: "POST", body: JSON.stringify(body),
    }),

  versions: (kbId: string) =>
    request<CorpusVersion[]>(`/api/knowledge-bases/${kbId}/corpus-versions`),
  snapshot: (kbId: string, note?: string) =>
    request<CorpusVersion>(`/api/knowledge-bases/${kbId}/corpus-versions`, {
      method: "POST", body: JSON.stringify({ note: note ?? null }),
    }),
  fingerprint: (kbId: string) =>
    request<CorpusFingerprint>(`/api/knowledge-bases/${kbId}/corpus-fingerprint`),
};
