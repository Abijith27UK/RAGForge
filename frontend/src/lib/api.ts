/* ------------------------------------------------------------------ */
/* Answer quality evaluation (V8)                                        */
/*                                                                      */
/* A metric is either MEASURED (with a value and a sample size) or       */
/* UNKNOWN (no value, plus the reason it could not be measured).         */
/* UNKNOWN is never rendered as 0 — that is the whole point.              */
/* ------------------------------------------------------------------ */

export interface Measured {
  value: number | null;
  measured: boolean;
  reason: string;
  sample_size: number | null;
}

export interface ClaimCitationVerdict {
  claim_id: string;
  claim_text: string;
  support_status: string;
  /** Five-state evaluator verdict: SUPPORTED | PARTIALLY_SUPPORTED |
   * UNSUPPORTED | CONTRADICTED | UNVERIFIABLE. */
  evaluated_state: string;
  cited_evidence_ids: string[];
  resolved_chunk_ids: string[];
  missing_required_chunk_ids: string[];
  irrelevant_chunk_ids: string[];
  entailment: "SUPPORTED" | "NOT_SUPPORTED" | "UNKNOWN";
  entailment_detail: string;
  entailment_method: string;
  has_invalid_citation: boolean;
  uncited: boolean;
  problems: string[];
}

export interface AnswerQualityResult {
  question_id: string;
  question: string;
  answerability: string;
  answer_id: string;
  answer_trace_id: string | null;
  answer_status: string;
  answer_text: string;
  citation_precision: Measured;
  citation_recall: Measured;
  citation_completeness: Measured;
  evidence_support_rate: Measured;
  unsupported_claim_rate: Measured;
  contradiction_rate: Measured;
  retrieval_hit_rate: Measured;
  correctness: Measured;
  key_point_recall: Measured;
  /** Spec name for completeness — the same measurement as key_point_recall. */
  expected_information_coverage: Measured;
  /** Lexical similarity to a HUMAN reference answer — NOT correctness. */
  reference_answer_similarity: Measured;
  question_answer_relevance: Measured;
  excess_information: Measured;
  supported_claim_ratio: Measured;
  partial_claim_ratio: Measured;
  unsupported_claim_ratio: Measured;
  fabricated_citation_rate: Measured;
  unsupported_citation_rate: Measured;
  abstention_expected: boolean;
  abstention_performed: boolean;
  abstention_correct: boolean | null;
  expected_grounding_state: string;
  actual_grounding_state: string;
  grounding_state_correct: boolean | null;
  false_supported: boolean;
  false_unsupported: boolean;
  claim_verdicts: ClaimCitationVerdict[];
  required_chunk_ids: string[];
  cited_chunk_ids: string[];
  retrieved_chunk_ids: string[];
  retrieved_rank_of_required: number | null;
  passed: boolean;
  final_score: number | null;
  final_score_computable: boolean;
  warnings: string[];
  problems: string[];
  evaluator_name: string;
  evaluator_version: string;
  evaluator_is_model_based: boolean;
  evaluator_detail: string;
  judge_raw: string;
  entailment_provider: string;
  entailment_is_model_based: boolean;
  /** Relevance provenance: method, weight source, model-based flag, close-call. */
  relevance_method: string;
  relevance_is_model_based: boolean;
  relevance_close_call: boolean;
  relevance_weight_source: string;
  relevance_passed: boolean | null;
}

export interface AnswerQualityAggregate {
  question_count: number;
  answerable_count: number;
  unanswerable_count: number;
  abstention_expected_count: number;
  citation_precision: Measured;
  citation_recall: Measured;
  citation_completeness: Measured;
  evidence_support_rate: Measured;
  unsupported_claim_rate: Measured;
  contradiction_rate: Measured;
  retrieval_hit_rate: Measured;
  correctness: Measured;
  key_point_recall: Measured;
  expected_information_coverage: Measured;
  reference_answer_similarity: Measured;
  question_answer_relevance: Measured;
  excess_information: Measured;
  supported_claim_ratio: Measured;
  partial_claim_ratio: Measured;
  unsupported_claim_ratio: Measured;
  fabricated_citation_rate: Measured;
  unsupported_citation_rate: Measured;
  relevance_failure_rate: Measured;
  abstention_accuracy: Measured;
  grounding_state_accuracy: Measured;
  false_supported_rate: Measured;
  false_unsupported_rate: Measured;
  hallucination_rate: Measured;
  pass_rate: Measured;
  grounding_state_distribution: Record<string, number>;
  confusion_matrix: Record<string, Record<string, number>>;
  unknown_metrics: string[];
  warnings: string[];
}

export interface AnswerEvaluationRunSummary {
  id: string;
  created_at: string;
  benchmark_name: string;
  benchmark_version: number;
  benchmark_fingerprint: string;
  /** Authoring lifecycle: draft | approved | frozen. Only frozen may be official. */
  benchmark_lifecycle: string;
  official: boolean;
  strategy: string;
  retrieval_params: Record<string, unknown>;
  generator: string;
  model: string;
  is_mock: boolean;
  /** Summary view joins the evaluator name and version into one string. */
  evaluator: string;
  evaluator_is_model_based: boolean;
  evaluator_detail: string;
  entailment_provider: string;
  entailment_is_model_based: boolean;
  relevance_method: string;
  relevance_weight_source: string;
  question_count: number;
  answerable_count: number;
  unanswerable_count: number;
  failed_question_ids: string[];
  unknown_metrics: string[];
  warnings: string[];
  aggregate: AnswerQualityAggregate;
}

/**
 * The full run record. NOTE: this is NOT the summary — producers are separate
 * fields here (`evaluator_name` + `evaluator_version`), so reading `evaluator`
 * off a full run silently yields undefined and renders a blank provenance field.
 */
export interface AnswerEvaluationRun extends Omit<AnswerEvaluationRunSummary, "evaluator"> {
  kb_id: string;
  benchmark_path: string;
  corpus_version: string | null;
  corpus_fingerprint: string | null;
  answer_mode: string;
  prompt_version: string;
  answerer_version: string;
  evaluator_name: string;
  evaluator_version: string;
  retrieval_run_ids: string[];
  question_ids: string[];
  subset_note: string;
  per_question: AnswerQualityResult[];
  notes: string[];
}

export interface AnswerEvaluationComparison {
  comparable: boolean;
  /** COMPARABLE | INCONCLUSIVE | NOT_COMPARABLE (V8 STEP 11). */
  verdict: string;
  /** identical | intersection | no_overlap */
  mode: string;
  /** Shared question ids the differences were computed over (intersection). */
  shared_question_ids: string[];
  reason: string;
  left: AnswerEvaluationRunSummary;
  right: AnswerEvaluationRunSummary;
  differences: Record<string, { left: Measured; right: Measured; delta: number | null }>;
}

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
  // V6: strategy provenance (present on all responses since V6)
  strategy?: string;
  retrieval_run_id?: string | null;
  notes?: string[];
}

// ---------------------------------------------------------------- V7 answers

export type AnswerStatus =
  | "grounded"
  | "partial"
  | "abstained"
  | "clarification_required"
  | "generation_failed";

export type GateDecision = "ANSWER" | "PARTIAL_ANSWER" | "ABSTAIN" | "ASK_CLARIFICATION";

/** The five explicit grounding outcomes exposed by the GroundingGate. */
export type GroundingState =
  | "ANSWERED"
  | "PARTIALLY_SUPPORTED"
  | "INSUFFICIENT_EVIDENCE"
  | "CONFLICTING_EVIDENCE"
  | "NO_RELEVANT_EVIDENCE";

/** Coarse classification of what KIND of turn a message is. */
export type QueryNature =
  | "knowledge"
  | "conversational"
  | "non_knowledge"
  | "underspecified"
  | "multi_hop";

/** Grounding detail attached to a chat turn. Never contains a fake
 *  confidence percentage — see docs/grounding-and-citations.md. */
export interface Grounding {
  state: GroundingState;
  decision: GateDecision;
  sufficient: boolean;
  confidence: "high" | "moderate" | "low" | "none";
  reason_code: string;
  reason: string;
  evidence_count: number;
  document_count: number;
  supporting_evidence_ids: string[];
  unsupported_aspects: string[];
  missing_information: string[];
  recommended_action: string;
  signals: GateSignal[];
  notes: string[];
}

export interface QueryTrace {
  original_query: string;
  normalized_query: string;
  /** null when the question passed through unchanged (never an empty string) */
  rewritten_query: string | null;
  subqueries: string[];
  processor: string;
  processor_version: string;
  query_type: string;
  nature: QueryNature;
  classification_method: string;
  classification_signals: string[];
  transformations: string[];
  expanded_terms: string[];
  warnings: string[];
  notes: string[];
  timing_ms: number | null;
  enabled: boolean;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  kb_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  answer_id: string | null;
  answer_trace_id: string | null;
  answer_run_id: string | null;
  retrieval_run_id: string | null;
  grounding_state: GroundingState | null;
  citation_count: number | null;
  /** prior user message this turn was expanded from, when applicable */
  resolved_from: string | null;
  used_as_knowledge: boolean;
}

export interface ChatGeneration {
  generated_by: string;
  model: string;
  is_mock: boolean;
  prompt_version: string;
  answerer_version: string;
  answer_mode: string;
  /** true when the configured generator was unavailable and a fallback ran */
  degraded: boolean;
  /** "live" | "mock" | "fallback" */
  generator_status: string;
}

export interface ChatResponse {
  answer: string;
  answer_id: string;
  status: AnswerStatus;
  citations: Citation[];
  claims: Claim[];
  grounding: Grounding;
  evidence: Evidence[];
  retrieval_run_id: string | null;
  answer_run_id: string;
  answer_trace_id: string;
  query_trace: QueryTrace;
  conversation_id: string;
  user_message_id: string;
  assistant_message_id: string;
  generation: ChatGeneration;
  warnings: string[];
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  kb_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail {
  conversation: {
    id: string;
    kb_id: string;
    title: string;
    created_at: string;
    updated_at: string;
    message_count: number;
  };
  messages: ChatMessage[];
}

export interface AnswerRun {
  id: string;
  kb_id: string;
  created_at: string;
  question: string;
  conversation_id: string | null;
  message_id: string | null;
  answer_id: string;
  answer_trace_id: string;
  retrieval_run_id: string | null;
  strategy: string;
  retrieval_params: Record<string, unknown>;
  selected_evidence_ids: string[];
  evidence_count: number;
  document_count: number;
  grounding_decision: string;
  grounding_state: GroundingState;
  grounding_reasons: string[];
  grounding_reason_code: string;
  grounding_sufficient: boolean;
  citation_count: number;
  claim_count: number;
  unsupported_claim_count: number;
  /** measured latencies in ms; null means "not measured", never 0 */
  query_processing_ms: number | null;
  retrieval_ms: number | null;
  evidence_selection_ms: number | null;
  grounding_ms: number | null;
  generation_ms: number | null;
  citation_validation_ms: number | null;
  total_ms: number | null;
  model: string;
  provider: string;
  is_mock: boolean;
  prompt_version: string;
  answerer_version: string;
  query_processor: string;
  query_processor_version: string;
  evidence_selector: string;
  grounding_gate: string;
  answer_status: AnswerStatus | null;
  warnings: string[];
  notes: string[];
}

export interface GateSignal {
  name: string;
  measured: boolean;
  value: number | null;
  threshold: number | null;
  passed: boolean | null;
  interpretation: string;
  not_performed_reason: string;
}

export interface EvidenceAssessment {
  sufficient: boolean;
  decision: GateDecision;
  /** the five-state grounding outcome (present on every assessment) */
  grounding_state?: GroundingState;
  confidence: "high" | "moderate" | "low" | "none";
  reason_code: string;
  reason: string;
  evidence_count: number;
  document_count: number;
  supporting_evidence_ids: string[];
  unsupported_aspects: string[];
  missing_information: string[];
  recommended_action: string;
  signals: GateSignal[];
  notes: string[];
}

export interface Evidence {
  evidence_id: string;
  chunk_id: string;
  document_id: string;
  source_id: string | null;
  kb_id: string;
  title: string;
  source_type: string | null;
  content: string;
  retrieval_score: number;
  retrieval_strategy: string;
  rank: number;
  original_rank: number;
  page: number | null;
  slide: number | null;
  section: string | null;
  section_path: string | null;
  url: string | null;
  publisher: string | null;
  document_version: number | null;
  content_hash: string;
  trust_score: number | null;
  provenance: Record<string, string | number | boolean | null>;
  retrieval_run_id: string | null;
  dedup_reason: string;
  overlap_fraction: number | null;
}

export interface Citation {
  citation_id: string;
  evidence_id: string;
  chunk_id: string;
  document_id: string;
  source_title: string;
  source_type: string | null;
  title: string;
  page: number | null;
  page_number: number | null;
  slide: number | null;
  slide_number: number | null;
  section: string | null;
  section_path: string | null;
  content_hash: string;
  url: string | null;
  publisher: string | null;
  snippet: string;
  validation: string;
  validation_detail: string;
}

export interface Claim {
  claim_id: string;
  text: string;
  claim_type: string;
  citation_ids: string[];
  evidence_ids: string[];
  support_status: "supported" | "partially_supported" | "unsupported";
  support_check: string;
  support_note: string;
}

export interface Answer {
  answer_id: string;
  kb_id: string;
  question: string;
  status: AnswerStatus;
  text: string;
  claims: Claim[];
  citations: Citation[];
  evidence_ids: string[];
  confidence: "high" | "moderate" | "low" | "none";
  confidence_basis: string;
  generated_by: string;
  model: string;
  is_mock: boolean;
  prompt_version: string;
  answer_mode: string;
  retrieval_run_id: string | null;
  answer_trace_id: string;
  assessment: EvidenceAssessment | null;
  generation_notes: string[];
  warnings: string[];
  created_at: string;
}

export interface AnswerResponse {
  answer: Answer;
  status: AnswerStatus;
  citations: Citation[];
  claims: Claim[];
  evidence: Evidence[];
  retrieval_run_id: string | null;
  answer_trace_id: string;
  grounding_assessment: EvidenceAssessment;
  generation_metadata: {
    generated_by: string;
    model: string;
    is_mock: boolean;
    prompt_version: string;
    answer_mode: string;
  };
  warnings: string[];
}

export interface AnswerTraceStage {
  name: string;
  status: "ok" | "skipped" | "unavailable" | "error";
  detail: string;
  count: number | null;
  ms: number | null;
}

export interface QueryPlan {
  original_query: string;
  normalized_query: string;
  detected_language: string | null;
  query_type: string;
  classification_method: string;
  classification_signals: string[];
  extracted_terms: string[];
  domain_terms: string[];
  retrieval_queries: string[];
  filters: Record<string, unknown>;
  requested_answer_format: string;
  notes: string[];
}

export interface AnswerTrace {
  id: string;
  kb_id: string;
  answer_id: string;
  created_at: string;
  question: string;
  answer_mode: string;
  query_plan: QueryPlan | null;
  retrieval_strategy: string;
  retrieval_run_id: string | null;
  retrieval_params: Record<string, unknown> | null;
  retrieval_stages: Record<string, unknown>[];
  evidence_ids: string[];
  evidence_items: Evidence[];
  evidence_dropped: { evidence_id: string; chunk_id: string; reason: string; original_rank: number }[];
  evidence_notes: string[];
  assessment: EvidenceAssessment | null;
  generator: string;
  generator_model: string;
  is_mock: boolean;
  generation_notes: string[];
  generation_warnings: string[];
  raw_generated_text: string;
  validation_actions: string[];
  citation_problems: Record<string, unknown>[];
  status: AnswerStatus | null;
  stages: AnswerTraceStage[];
  notes: string[];
  total_ms: number | null;
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
  retrieve: (id: string, query: string, topK = 5, strategy?: string) =>
    request<RetrievalResponse>(`/api/knowledge-bases/${id}/retrieve`, {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: topK,
        ...(strategy ? { strategy } : {}),
      }),
    }),

  // --- V7: grounded answers ---
  answer: (
    id: string,
    question: string,
    opts: { strategy?: string; mode?: string; topK?: number } = {},
  ) =>
    request<AnswerResponse>(`/api/knowledge-bases/${id}/answer`, {
      method: "POST",
      body: JSON.stringify({
        question,
        ...(opts.strategy ? { retrieval_strategy: opts.strategy } : {}),
        ...(opts.mode ? { answer_mode: opts.mode } : {}),
        ...(opts.topK ? { retrieval_params: { top_k: opts.topK } } : {}),
      }),
    }),
  getAnswer: (id: string, answerId: string) =>
    request<Answer>(`/api/knowledge-bases/${id}/answers/${answerId}`),
  getAnswerTrace: (id: string, traceId: string) =>
    request<AnswerTrace>(`/api/knowledge-bases/${id}/answer-traces/${traceId}`),

  // --- V7 Phase 13: grounded chat ---
  chat: (
    id: string,
    message: string,
    opts: {
      retrieval_strategy?: string;
      retrieval_params?: Record<string, unknown>;
      conversation_id?: string;
      answer_mode?: string;
    } = {},
  ) =>
    request<ChatResponse>(`/api/knowledge-bases/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({
        message,
        ...(opts.retrieval_strategy ? { retrieval_strategy: opts.retrieval_strategy } : {}),
        ...(opts.retrieval_params ? { retrieval_params: opts.retrieval_params } : {}),
        ...(opts.conversation_id ? { conversation_id: opts.conversation_id } : {}),
        ...(opts.answer_mode ? { answer_mode: opts.answer_mode } : {}),
      }),
    }),

  listConversations: (id: string) =>
    request<ConversationSummary[]>(`/api/knowledge-bases/${id}/conversations`),

  getConversation: (id: string, conversationId: string) =>
    request<ConversationDetail>(
      `/api/knowledge-bases/${id}/conversations/${conversationId}`,
    ),

  deleteConversation: (id: string, conversationId: string) =>
    request<void>(`/api/knowledge-bases/${id}/conversations/${conversationId}`, {
      method: "DELETE",
    }),

  listAnswerRuns: (id: string) =>
    request<AnswerRun[]>(`/api/knowledge-bases/${id}/answer-runs`),

  getAnswerRun: (id: string, runId: string) =>
    request<AnswerRun>(`/api/knowledge-bases/${id}/answer-runs/${runId}`),

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

/* Answer-quality evaluation (V8). */

/** One append-only human review of one evaluated answer (V8 STEP 4). */
export interface AnswerReview {
  id: string;
  kb_id: string;
  run_id: string;
  question_id: string;
  answer_id: string;
  reviewer: string;
  verdict: string;
  labels: string[];
  notes: string;
  created_at: string;
}

export const REVIEW_VERDICTS = [
  "correct",
  "mostly_correct",
  "partially_correct",
  "incorrect",
  "should_have_abstained",
  "correctly_abstained",
] as const;

export const REVIEW_LABELS = [
  "factual_error",
  "unsupported_claim",
  "missing_information",
  "wrong_citation",
  "irrelevant_evidence",
  "contradiction",
  "incomplete_answer",
  "excessive_answer",
  "correct_answer",
] as const;

export const answerEvaluationApi = {
  /** Run an answer-quality evaluation. persist=false stores no run row. */
  run: (
    kbId: string,
    body: {
      benchmark_path: string;
      strategy?: string;
      retrieval_params?: Record<string, unknown>;
      answer_mode?: string;
      question_ids?: string[];
      limit?: number;
      persist?: boolean;
      /** OFFICIAL result; refused 400 unless the benchmark lifecycle is frozen. */
      official?: boolean;
      /** deterministic (default) | human | llm */
      evaluator?: string;
    }
  ) =>
    request<AnswerEvaluationRun>(`/api/knowledge-bases/${kbId}/answer-evaluation/runs`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  list: (kbId: string, limit = 50) =>
    request<AnswerEvaluationRunSummary[]>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/runs?limit=${limit}`
    ),

  get: (kbId: string, runId: string) =>
    request<AnswerEvaluationRun>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/runs/${runId}`
    ),

  question: (kbId: string, runId: string, questionId: string) =>
    request<AnswerQualityResult>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/runs/${runId}/questions/${questionId}`
    ),

  /**
   * Compare two runs. The API refuses to compare different question subsets, so
   * `comparable` must be checked before any delta is rendered. Subsets that
   * PARTIALLY overlap come back verdict="INCONCLUSIVE" with differences over
   * the shared questions only.
   */
  compare: (kbId: string, left: string, right: string) =>
    request<AnswerEvaluationComparison>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/compare?left=${left}&right=${right}`
    ),

  /** Record ONE human review. Append-only: submitting twice keeps both. */
  createReview: (
    kbId: string,
    runId: string,
    questionId: string,
    body: {
      reviewer: string;
      verdict: string;
      labels?: string[];
      notes?: string;
      answer_id?: string;
    },
  ) =>
    request<AnswerReview>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/runs/${runId}/questions/${questionId}/reviews`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  /** Full review history for one question, oldest first. */
  reviews: (kbId: string, runId: string, questionId: string) =>
    request<AnswerReview[]>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/runs/${runId}/questions/${questionId}/reviews`
    ),

  /**
   * Derive a NEW run whose correctness comes from this run's human reviews.
   * Source run and reviews are never modified.
   */
  humanEvaluation: (kbId: string, runId: string, persist = true) =>
    request<AnswerEvaluationRun>(
      `/api/knowledge-bases/${kbId}/answer-evaluation/runs/${runId}/human-evaluation?persist=${persist}`,
      { method: "POST" },
    ),

  /** Top-level (cross-KB) run listing (V8 STEP 10). */
  listAll: (limit = 100, kbId?: string) =>
    request<AnswerEvaluationRunSummary[]>(
      `/api/answer-evaluation-runs?limit=${limit}${kbId ? `&kb_id=${kbId}` : ""}`
    ),

  /** One run by its global id, regardless of knowledge base. */
  getById: (runId: string) =>
    request<AnswerEvaluationRun>(`/api/answer-evaluation-runs/${runId}`),
};
