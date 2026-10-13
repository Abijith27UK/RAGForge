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
      /**
       * deterministic (default) | reference | human | llm.
       * 'reference' measures correctness from a HUMAN-REVIEWED benchmark
       * (requires lifecycle approved/frozen); 'deterministic' keeps
       * correctness UNKNOWN by design.
       */
      evaluator?: string;
      /**
       * V10: score an immutable frozen benchmark version (created by the
       * benchmark review workflow) instead of a file path.
       */
      benchmark_version_id?: string;
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

/* ------------------------------------------------------------------ */
/* Benchmark review — human ground truth (V10)                          */
/*                                                                      */
/* A question is PENDING / IN_REVIEW / APPROVED / REJECTED / AMBIGUOUS  */
/* / INSUFFICIENT_EVIDENCE. Only APPROVED questions carry scorable      */
/* ground truth; ambiguous and insufficient-evidence questions are      */
/* excluded LOUDLY by the published approval policy, never silently.    */
/* Coverage rates with no denominator come back as Measured.unknown.    */
/* ------------------------------------------------------------------ */

export const QUESTION_STATES = [
  "pending",
  "in_review",
  "approved",
  "rejected",
  "ambiguous",
  "insufficient_evidence",
] as const;
export type QuestionState = (typeof QUESTION_STATES)[number];

/** Review dimensions a reviewer must assess (mirrors the backend enum). */
export const REVIEW_DIMENSIONS = [
  "answerability",
  "evidence_sufficiency",
  "reference_answer",
  "key_points",
  "acceptable_elements",
  "ambiguity",
] as const;
export type ReviewDimension = (typeof REVIEW_DIMENSIONS)[number];

/** Dimensions the approval gate REQUIRES to be assessed affirmatively. */
export const REQUIRED_REVIEW_DIMENSIONS: ReviewDimension[] = [
  "answerability",
  "evidence_sufficiency",
  "reference_answer",
  "key_points",
  "ambiguity",
];

export const DIMENSION_VERDICTS = [
  "ok",
  "ok_with_note",
  "needs_revision",
  "wrong",
  "ambiguous",
  "unknown",
] as const;
export type DimensionVerdict = (typeof DIMENSION_VERDICTS)[number];

export interface BenchmarkIdentity {
  name: string;
  version: number;
  kb_id: string;
  kb_name: string;
  path: string;
  fingerprint: string;
  lifecycle: "draft" | "approved" | "frozen";
  human_review: "human_reviewed" | "pending";
  derived_from: string;
  question_count: number;
  authorship: Record<string, unknown>;
}

export interface ReviewCompleteness {
  benchmark_name: string;
  benchmark_fingerprint: string;
  total: number;
  pending: number;
  in_review: number;
  approved: number;
  rejected: number;
  ambiguous: number;
  insufficient_evidence: number;
  authored_reference_answer_count: number;
  authored_key_point_count: number;
  authored_acceptable_element_count: number;
  provenance_valid_count: number;
  provenance_invalid_count: number;
  review_count: number;
  reviewers: string[];
  /** Rates are Measured: no denominator means UNKNOWN, never 0. */
  reference_answer_coverage: Measured;
  key_point_coverage: Measured;
  acceptable_element_coverage: Measured;
  provenance_coverage: Measured;
  unknown_metrics: string[];
}

export interface ApprovalPolicy {
  policy_version: string;
  scoring_states: string[];
  permitted_non_scoring_states: string[];
  blocking_states: string[];
  minimum_approved_questions: number;
  require_key_points: boolean;
  require_reference_answer: boolean;
  require_provenance_valid: boolean;
}

export interface ApprovalGateResult {
  approved: boolean;
  policy: ApprovalPolicy;
  evaluated_at: string;
  counts: Record<string, number>;
  approved_question_ids: string[];
  excluded_question_ids: string[];
  blocking: Record<string, string[]>;
  problems: Record<string, string[]>;
  reasons: string[];
}

export interface EvidenceItem {
  chunk_id: string;
  required: boolean;
  found: boolean;
  document_id: string | null;
  text: string;
  source_title: string | null;
  label_source_title: string | null;
  source_url: string | null;
  source_type: string | null;
  publisher: string | null;
  document_title: string | null;
  section: string | null;
  section_path: string | null;
  page: number | null;
  slide: number | null;
  slide_title: string | null;
  content_hash: string;
  char_count: number;
  problems: string[];
}

export interface ReviewQuestionSummary {
  question_id: string;
  question: string;
  subdomain: string;
  answerability: string;
  state: QuestionState;
  reasons: string[];
  problems: string[];
  review_count: number;
  reviewers: string[];
  has_annotation: boolean;
  annotation_id: string;
  annotation_author: string;
  expected_answer: string;
  key_points: string[];
  acceptable_answer_elements: string[];
  evidence_chunk_ids: string[];
  artifact_reference_answer: string;
  artifact_key_points: string[];
  /** The question's own provenance text: which chunk/section the evidence was
   * selected from, and by whom. Informational — never ground truth. */
  artifact_provenance: string;
  /** The benchmark's citation requirements for this question. */
  citation_requirements: Record<string, unknown>;
  /** Which grounding outcome the question expects (ANY_ACCEPTABLE by default). */
  expected_grounding_state: string;
  abstention_required: boolean;
  requires_units: boolean;
  numerical_tolerance: number | null;
}

export interface ProvenanceBlock {
  kind: string;
  level: string;
  method: string;
  source: string;
  justification: string;
  timestamp: string;
  tags: string[];
  data: Record<string, unknown>;
}

export interface GroundTruthAnnotation {
  annotation_id: string;
  kb_id: string;
  benchmark_name: string;
  benchmark_fingerprint: string;
  question_id: string;
  author: string;
  created_at: string;
  expected_answer: string;
  key_points: string[];
  acceptable_answer_elements: string[];
  evidence: { chunk_id: string; document_id: string | null; content_hash: string; section: string | null; source_title: string | null; required: boolean }[];
  provenance: ProvenanceBlock[];
  supersedes: string;
  note: string;
}

export interface BenchmarkQuestionReview {
  review_id: string;
  kb_id: string;
  benchmark_name: string;
  benchmark_fingerprint: string;
  question_id: string;
  reviewer: string;
  created_at: string;
  verdicts: Record<string, DimensionVerdict>;
  notes: Record<string, string>;
  evidence_checked: string[];
  answerability_verdict: string | null;
  outcome: QuestionState | null;
  annotation_id: string;
  unresolved: string[];
  supersedes: string;
}

export interface BenchmarkQuestionDetail {
  identity: BenchmarkIdentity;
  summary: ReviewQuestionSummary;
  evidence: EvidenceItem[];
  annotation_history: GroundTruthAnnotation[];
  effective_annotation: GroundTruthAnnotation | null;
  review_history: BenchmarkQuestionReview[];
  effective_review: BenchmarkQuestionReview | null;
}

export interface BenchmarkReviewPacket {
  identity: BenchmarkIdentity;
  questions: ReviewQuestionSummary[];
  completeness: ReviewCompleteness;
  gate: ApprovalGateResult;
  policy: ApprovalPolicy;
  versions: FrozenVersionSummary[];
  warnings: string[];
}

export interface FrozenVersionSummary {
  version_id: string;
  version: number;
  benchmark_name: string;
  benchmark_fingerprint: string;
  ground_truth_fingerprint: string;
  artifact_fingerprint: string;
  created_at: string;
  frozen_by: string;
  scoring_question_count: number;
  excluded_question_count: number;
}

export interface FrozenBenchmarkVersion extends Omit<FrozenVersionSummary, "scoring_question_count" | "excluded_question_count"> {
  kb_id: string;
  gate: ApprovalGateResult;
  benchmark: Record<string, unknown>;
  annotations: GroundTruthAnnotation[];
  reviews: BenchmarkQuestionReview[];
  notes: string[];
}

export interface VersionVerifyResult {
  version_id: string;
  intact: boolean;
  problems: string[];
}

// ---------------------------------------------------------------------------
// V11 draft assist and benchmark quality audit
//
// A DraftSuggestion is NOT ground truth. It carries `is_draft`, a conspicuous
// `label`, and `requires_human_review`, and the backend stores it in a table no
// ground-truth query can read. The UI must render it as a suggestion the author
// edits and submits under their own name — never as an approved label.
// ---------------------------------------------------------------------------

export type DraftStatus = "draft_ready" | "partial" | "insufficient_evidence";

export interface DraftKeyPoint {
  /** Verbatim text of the cited chunk. Guaranteed by the backend. */
  text: string;
  source_chunk_id: string;
  source_sentence_index: number;
  score: number;
  coverage: number;
  method: string;
}

export interface DraftEvidence {
  chunk_id: string;
  found: boolean;
  document_id: string | null;
  source_title: string | null;
  section: string | null;
  content_hash: string;
  char_count: number;
  sentence_count: number;
  problems: string[];
}

export interface DraftSuggestion {
  suggestion_id: string;
  kb_id: string;
  benchmark_name: string;
  benchmark_fingerprint: string;
  question_id: string;
  created_at: string;
  method: string;
  generator: string;
  is_draft: boolean;
  label: string;
  requires_human_review: boolean;
  status: DraftStatus;
  reason: string;
  caveats: string[];
  question: string;
  subdomain: string;
  expected_answer_draft: string;
  key_points_draft: DraftKeyPoint[];
  evidence: DraftEvidence[];
  question_terms: string[];
  covered_terms: string[];
  uncovered_terms: string[];
  coverage: number;
  coverage_measured: boolean;
  coverage_method: string;
  benchmark_quote: string;
  agrees_with_benchmark_quote: boolean | null;
}

export interface DraftBatchResponse {
  suggestions: DraftSuggestion[];
  summary: Record<string, unknown>;
  label: string;
  notice: string;
}

export type AuditSeverity = "error" | "warning" | "info";

export interface BenchmarkQualityFinding {
  code: string;
  severity: AuditSeverity;
  message: string;
  question_id: string;
  also_question_id: string;
  evidence_chunk_id: string;
  action: string;
  detail: Record<string, unknown>;
}

export interface BenchmarkQualityQuestion {
  question_id: string;
  question: string;
  subdomain: string;
  answerability: string;
  evidence_chunk_ids: string[];
  has_annotation: boolean;
  key_points_checked: number;
  key_points_supported: number;
  key_points_unsupported: string[];
  question_terms: string[];
  uncovered_terms: string[];
  /** null when the KB has no corpus statistics — NEVER a substituted 0. */
  coverage: number | null;
  recommended_action: string;
  findings: BenchmarkQualityFinding[];
}

export interface BenchmarkQualityAudit {
  kb_id: string;
  benchmark_name: string;
  benchmark_fingerprint: string;
  benchmark_path: string;
  audited_at: string;
  audited_by: string;
  questions: BenchmarkQualityQuestion[];
  findings: BenchmarkQualityFinding[];
  error_count: number;
  warning_count: number;
  info_count: number;
  ready_to_freeze: boolean;
  verdict: string;
  checks_run: string[];
  checks_not_run: string[];
  limitations: string[];
}

export const DEFAULT_ANSWER_BENCHMARK_PATH =
  "benchmarks/answer-quality-automobile-v1.json";

export const benchmarkReviewApi = {
  /** The whole review surface: states, completeness, gate, frozen versions. */
  packet: (kbId: string, benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH) =>
    request<BenchmarkReviewPacket>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/review-packet?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  /** One question with the ACTUAL corpus evidence and full human input history. */
  question: (kbId: string, questionId: string, benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH) =>
    request<BenchmarkQuestionDetail>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  completeness: (kbId: string, benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH) =>
    request<ReviewCompleteness>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/completeness?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  /** Append a human-authored reference answer. Append-only: never an update. */
  authorGroundTruth: (
    kbId: string,
    questionId: string,
    body: {
      benchmark_path?: string;
      author: string;
      expected_answer?: string;
      key_points?: string[];
      acceptable_answer_elements?: string[];
      evidence_chunk_ids?: string[];
      method?: string;
      note?: string;
      supersedes?: string;
    },
  ) =>
    request<{ annotation: GroundTruthAnnotation; problems: string[]; stored: boolean }>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}/ground-truth`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  groundTruthHistory: (kbId: string, questionId: string, benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH) =>
    request<GroundTruthAnnotation[]>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}/ground-truth?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  /** Append ONE human review. An 'approved' outcome requires every required
   * dimension to be assessed affirmatively — the API refuses otherwise. */
  createReview: (
    kbId: string,
    questionId: string,
    body: {
      benchmark_path?: string;
      reviewer: string;
      outcome?: string | null;
      verdicts: Record<string, string>;
      notes?: Record<string, string>;
      evidence_checked?: string[];
      annotation_id?: string;
      answerability_verdict?: string | null;
      unresolved?: string[];
      supersedes?: string;
    },
  ) =>
    request<BenchmarkQuestionReview>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}/reviews`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  reviewHistory: (kbId: string, questionId: string, benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH) =>
    request<BenchmarkQuestionReview[]>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}/reviews?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  /** Freeze an APPROVED benchmark into a new immutable version. The API
   * returns 409 with the gate's own reasons when it is not approvable. */
  freeze: (kbId: string, body: { benchmark_path?: string; frozen_by: string; notes?: string[] }) =>
    request<FrozenBenchmarkVersion>(`/api/knowledge-bases/${kbId}/answer-benchmark/freeze`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  versions: (kbId: string) =>
    request<FrozenVersionSummary[]>(`/api/knowledge-bases/${kbId}/answer-benchmark/versions`),

  version: (kbId: string, versionId: string) =>
    request<FrozenBenchmarkVersion>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/versions/${versionId}`
    ),

  verifyVersion: (kbId: string, versionId: string) =>
    request<VersionVerifyResult>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/versions/${versionId}/verify`,
      { method: "POST" }
    ),

  /** V11: generate a corpus-grounded DRAFT for one question.
   * The result is a starting point for a human, never a label. */
  generateDraft: (
    kbId: string,
    questionId: string,
    body: { benchmark_path?: string; persist?: boolean } = {},
  ) =>
    request<DraftSuggestion>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}/draft-suggestions`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  /** Every draft ever generated for a question, oldest first (append-only). */
  draftHistory: (
    kbId: string,
    questionId: string,
    benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH,
  ) =>
    request<DraftSuggestion[]>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/questions/${questionId}/draft-suggestions?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  /** Generate drafts for the whole benchmark (or a subset). */
  generateDrafts: (
    kbId: string,
    body: { benchmark_path?: string; question_ids?: string[]; persist?: boolean } = {},
  ) =>
    request<DraftBatchResponse>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/draft-suggestions`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  /** V11: read-only pre-freeze quality audit. Findings are reported, never applied. */
  qualityAudit: (kbId: string, benchmarkPath = DEFAULT_ANSWER_BENCHMARK_PATH) =>
    request<BenchmarkQualityAudit>(
      `/api/knowledge-bases/${kbId}/answer-benchmark/quality-audit?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),
};

// ---------------------------------------------------------------------------
// V12 retrieval diagnostics
//
// MEASURED vs INFERRED vs PROPOSED, kept apart everywhere below:
//   * a metric with `measured: false` is UNKNOWN and must render as such —
//     never as 0.0, and never as a pass;
//   * a `classification` is a MEASURED verdict produced by a named rule from
//     the observations in `evidence`;
//   * a `recommendation` is a PROPOSED experiment with the measurements that
//     support it, never an applied configuration change.
// ---------------------------------------------------------------------------

export const DEFAULT_RETRIEVAL_BENCHMARK_PATH =
  "benchmarks/automobile-engineering-baseline-v1.json";

/** Every published retrieval outcome. LOW_PRECISION is documented as reserved. */
export type RetrievalFailureCode =
  | "RETRIEVAL_SUCCESS"
  | "RANKING_FAILURE"
  | "LOW_RECALL"
  | "LOW_PRECISION"
  | "NO_RELEVANT_EVIDENCE"
  | "RETRIEVAL_FAILURE"
  | "EXPECTED_EVIDENCE_UNAVAILABLE"
  | "CORPUS_OR_SOURCE_DEFICIENCY"
  | "UNKNOWN";

export type EvidenceAvailabilityStatus =
  | "indexed"
  | "corpus_only"
  | "missing_from_index"
  | "missing_from_corpus"
  | "unknown";

export interface EvidenceAvailability {
  chunk_id: string;
  expected_document_ids: string[];
  in_corpus: boolean | null;
  in_index: boolean | null;
  status: EvidenceAvailabilityStatus;
  actual_document_id: string | null;
  document_matches_benchmark: boolean | null;
  section: string | null;
  source_title: string | null;
  content_hash: string | null;
  notes: string[];
  /** null when the index could not be probed — never assume a vector exists. */
  retrievable: boolean | null;
}

export interface RetrievalClassification {
  code: RetrievalFailureCode;
  rule: string;
  confidence: "deterministic" | "partial" | "unknown";
  evidence: string[];
  reason: string;
  limitations: string[];
  is_critical: boolean;
}

export interface RetrievalProbe {
  candidate_depth: number;
  candidate_chunk_ids: string[];
  required_ranks_in_probe: Record<string, number>;
  required_beyond_cutoff: Record<string, number>;
  absent_from_probe: string[];
  required_recall_in_probe: number | null;
  truncated: boolean;
}

export interface RetrievedItem {
  rank: number;
  chunk_id: string;
  score: number | null;
  score_breakdown: Record<string, unknown>;
  document_id: string | null;
  source_id: string | null;
  source_title: string | null;
  source_url: string | null;
  source_type: string | null;
  publisher: string | null;
  section: string | null;
  page: number | null;
  trust_score: number | null;
  content_hash: string | null;
  is_required: boolean;
  selected_for_generation: boolean;
}

export interface QuestionDiagnosticSummaryMetrics {
  recall_at_k: Measured;
  precision_at_k: Measured;
  mrr: Measured;
  ndcg_at_k: Measured;
  required_recall_at_depth: Measured;
  first_relevant_rank: Measured;
  latency_ms: Measured;
  required_recall_in_probe: Measured;
}

export interface StrategyDiagnosticsSummary extends QuestionDiagnosticSummaryMetrics {
  question_count: number;
  ground_truth_question_count: number;
  no_ground_truth_question_count: number;
  evaluated_question_count: number;
  code_counts: Record<string, number>;
  availability_counts: Record<string, number>;
  first_relevant_rank_histogram: Record<string, number>;
  critical_question_ids: string[];
  not_retrieved_question_ids: string[];
  evidence_gap_question_ids: string[];
  no_candidate_question_ids: string[];
  unknown_metrics: string[];
  warnings: string[];
}

export interface QuestionRetrievalDiagnostic {
  question_id: string;
  question: string;
  subdomain: string;
  strategy: string;
  top_k: number | null;
  expected_chunk_ids: string[];
  expected_document_ids: string[];
  availability: EvidenceAvailability[];
  availability_rollup: string;
  probe: RetrievalProbe;
  classification: RetrievalClassification;
  latency_ms: number | null;
  evidence: {
    question_id: string;
    question: string;
    subdomain: string;
    strategy: string;
    top_k: number | null;
    retrieval_params: Record<string, unknown>;
    retrieved: RetrievedItem[];
    retrieval_depth: number;
    required_chunk_ids: string[];
    required_document_ids: string[];
    ground_truth_available: boolean;
    first_relevant_rank: number | null;
    required_retrieved: boolean | null;
    recall_at_k: Measured;
    precision_at_k: Measured;
    mrr: Measured;
    ndcg_at_k: Measured;
    required_recall_at_depth: Measured;
    latency_ms: number | null;
    warnings: string[];
  };
}

export interface StrategyDiagnostics {
  strategy: string;
  status: "completed" | "failed" | "skipped" | string;
  failure_reason: string;
  top_k: number;
  candidate_depth: number;
  retrieval_params: Record<string, unknown>;
  config_fingerprint: string;
  embedding_model: string;
  retrieval_backend: string;
  reranker_status: string;
  corpus_version: string | null;
  corpus_fingerprint: string | null;
  index_points: number | null;
  notes: string[];
  questions: QuestionRetrievalDiagnostic[];
  summary: StrategyDiagnosticsSummary;
  duration_ms: number | null;
}

export interface RetrievalIntegritySnapshot {
  kb_id: string;
  index_points_before: number | null;
  index_points_after: number | null;
  chunk_rows_before: number | null;
  chunk_rows_after: number | null;
  corpus_revision_before: number | null;
  corpus_revision_after: number | null;
  benchmark_sha256_before: string;
  benchmark_sha256_after: string;
  retrieval_runs_persisted: number;
  protected_kb_unchanged: boolean | null;
  benchmark_unchanged: boolean | null;
}

export interface RetrievalRecommendation {
  action: string;
  reason: string;
  supporting_measurements: Record<string, unknown>;
  expected_effect: string;
  risk: string;
  strength: "strong" | "moderate" | "weak" | "insufficient_evidence";
  affected_question_ids: string[];
  proposed_config_change: Record<string, unknown>;
  experiment_required: boolean;
  thresholds_used: Record<string, unknown>;
}

export interface RetrievalDiagnosticReport {
  id: string;
  kb_id: string;
  kb_name: string;
  created_at: string;
  benchmark_name: string;
  benchmark_version: number;
  benchmark_path: string;
  benchmark_fingerprint: string;
  relevance_model: string;
  benchmark_authorship: Record<string, unknown>;
  top_k: number;
  question_count: number;
  matched_question_ids: string[];
  strategies: StrategyDiagnostics[];
  comparisons: StrategyComparison[];
  recommendations: {
    recommendations: RetrievalRecommendation[];
    total_questions: number;
    questions_with_ground_truth: number;
    questions_analysed: number;
    unknown_failure_count: number;
    data_gaps: string[];
    notes: string[];
  };
  integrity: RetrievalIntegritySnapshot;
  limitations: string[];
  notes: string[];
}

export interface MetricComparisonRow {
  metric: string;
  direction: "higher_is_better" | "lower_is_better";
  questions: number;
  paired_count: number;
  unknown_count: number;
  baseline_mean: number | null;
  candidate_mean: number | null;
  mean_delta: number | null;
  improved_count: number;
  regressed_count: number;
  tie_count: number;
  outcome: "IMPROVED" | "REGRESSED" | "UNCHANGED" | "UNKNOWN";
  reason: string;
  deltas: Array<{
    question_id: string;
    baseline: number | null;
    candidate: number | null;
    delta: number | null;
    outcome: string;
    unknown_reason: string;
  }>;
}

export interface StrategyComparison {
  baseline_label: string;
  candidate_label: string;
  baseline_config_fingerprint: string;
  candidate_config_fingerprint: string;
  benchmark_fingerprint: string;
  pairing: "COMPARABLE" | "INCONCLUSIVE" | "NOT_COMPARABLE";
  pairing_reason: string;
  comparable_question_ids: string[];
  protocol_differences: string[];
  metric_comparisons: MetricComparisonRow[];
  significance_tests: Array<{
    name: string;
    statistic: number | null;
    p_value: number | null;
    n: number;
    exploratory: boolean;
    note: string;
    assumptions: string;
  }>;
  improved_metric_count: number;
  regressed_metric_count: number;
  unchanged_metric_count: number;
  unknown_metric_count: number;
  exploratory: boolean;
  notes: string[];
}

export interface RetrievalDiagnosticsRunSummary {
  id: string;
  kb_id: string;
  created_at: string;
  benchmark_name: string;
  benchmark_fingerprint: string;
  top_k: number;
  question_count: number;
  strategies: Array<{
    strategy: string;
    status: string;
    failure_reason: string;
    code_counts: Record<string, number>;
    recall_at_k: Measured;
    mrr: Measured;
    ndcg_at_k: Measured;
    latency_ms: Measured;
  }>;
  protected_kb_unchanged: boolean | null;
  benchmark_unchanged: boolean | null;
  limitations: string[];
}

export interface EvidenceAuditQuestion {
  question_id: string;
  availability_rollup: string;
  expected_chunk_ids: string[];
  availability: EvidenceAvailability[];
}

export interface EvidenceAudit {
  kb_id: string;
  benchmark: string;
  benchmark_fingerprint: string;
  benchmark_path: string;
  benchmark_sha256: string;
  relevance_model: string;
  authorship: Record<string, unknown>;
  question_count: number;
  expected_chunk_count: number;
  expected_document_count: number;
  index_probe: "verified" | "unavailable" | string;
  index_points: number | null;
  index_probe_notes: string[];
  probe_notes: string[];
  availability_counts: Record<string, number>;
  questions: EvidenceAuditQuestion[];
  verdict: string;
  note: string;
}

export interface RetrievalBenchmarkIdentity {
  benchmark: string;
  version: number;
  created_at: string;
  kb_id: string;
  kb_name: string;
  relevance_model: string;
  authorship: Record<string, unknown>;
  fingerprint: string;
  path: string;
  question_count: number;
  matches_this_kb: boolean;
  questions: Array<{
    id: string;
    subdomain: string;
    question: string;
    expected_chunk_ids: string[];
    expected_document_ids: string[];
    provenance: string;
  }>;
  note: string;
}

export interface RetrievalDiagnosticsTaxonomy {
  codes: RetrievalFailureCode[];
  critical_codes: RetrievalFailureCode[];
  not_emitted: Record<string, string>;
  measured_vs_inferred: string;
}

/**
 * V12 retrieval diagnostics. Every call is READ-ONLY against the knowledge base,
 * the vector index and both benchmark artifacts; creating a run only INSERTs the
 * run's own report row.
 */
export const retrievalDiagnosticsApi = {
  benchmark: (kbId: string, benchmarkPath = DEFAULT_RETRIEVAL_BENCHMARK_PATH) =>
    request<RetrievalBenchmarkIdentity>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/benchmark?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  evidenceAudit: (kbId: string, benchmarkPath = DEFAULT_RETRIEVAL_BENCHMARK_PATH) =>
    request<EvidenceAudit>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/evidence-audit?benchmark_path=${encodeURIComponent(benchmarkPath)}`
    ),

  strategies: (kbId: string) =>
    request<{ strategies: Array<Record<string, unknown>>; note: string }>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/strategies`
    ),

  taxonomy: (kbId: string) =>
    request<RetrievalDiagnosticsTaxonomy>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/taxonomy`
    ),

  runDiagnostics: (
    kbId: string,
    body: {
      benchmark_path?: string;
      strategies?: string[];
      top_k?: number;
      candidate_depth?: number | null;
      question_ids?: string[];
      persist?: boolean;
      persist_retrieval_runs?: boolean;
    } = {},
  ) =>
    request<RetrievalDiagnosticReport>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/runs`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  runs: (kbId: string, benchmarkName?: string, limit = 50) => {
    const qs = new URLSearchParams();
    if (benchmarkName) qs.set("benchmark_name", benchmarkName);
    qs.set("limit", String(limit));
    return request<RetrievalDiagnosticsRunSummary[]>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/runs?${qs.toString()}`
    );
  },

  run: (kbId: string, runId: string) =>
    request<RetrievalDiagnosticReport>(
      `/api/knowledge-bases/${kbId}/retrieval-diagnostics/runs/${runId}`
    ),

  question: (kbId: string, runId: string, questionId: string) =>
    request<{
      run_id: string;
      question_id: string;
      benchmark_name: string;
      benchmark_fingerprint: string;
      strategies: Array<{
        strategy: string;
        status: string;
        top_k?: number;
        classification?: RetrievalClassification;
        availability_rollup?: string;
        availability?: EvidenceAvailability[];
        expected_chunk_ids?: string[];
        retrieved_chunk_ids?: string[];
        rank_of_expected?: Record<string, number | null>;
        required_ranks_in_probe?: Record<string, number>;
        required_beyond_cutoff?: Record<string, number>;
        absent_from_probe?: string[];
        required_recall_in_probe?: number | null;
        candidate_depth?: number;
        retrieved?: RetrievedItem[];
        metrics?: Record<string, Measured>;
        latency_ms?: number | null;
        warnings?: string[];
        note?: string;
      }>;
    }>(`/api/knowledge-bases/${kbId}/retrieval-diagnostics/runs/${runId}/questions/${questionId}`),

  recommendations: (kbId: string, runId: string) =>
    request<{
      run_id: string;
      recommendations: RetrievalRecommendation[];
      data_gaps: string[];
      notes: string[];
      limitations: string[];
      notice: string;
    }>(`/api/knowledge-bases/${kbId}/retrieval-diagnostics/runs/${runId}/recommendations`),

  comparisons: (kbId: string, runId: string) =>
    request<{
      run_id: string;
      comparisons: StrategyComparison[];
      notice: string;
    }>(`/api/knowledge-bases/${kbId}/retrieval-diagnostics/runs/${runId}/comparisons`),
};

// ---------------------------------------------------------------------------
// V13 controlled retrieval experiments
// ---------------------------------------------------------------------------

/** One arm of an experiment: a strategy plus the configuration under test. */
export interface ExperimentArmConfig {
  label: string;
  strategy: string;
  top_k: number;
  candidate_depth: number | null;
  params: Record<string, unknown>;
  role: string;
  rationale: string;
}

/** Descriptive statistics for ONE metric over the measured questions. */
export interface ExperimentMetricStats {
  metric: string;
  mean: number | null;
  median: number | null;
  min: number | null;
  max: number | null;
  sample_size: number;
  direction: "higher_is_better" | "lower_is_better" | string;
  measured: boolean;
  reason: string;
}

/** One question's baseline vs candidate value for the PRIMARY metric. */
export interface ExperimentPerQuestion {
  question_id: string;
  question: string;
  baseline: number | null;
  candidate: number | null;
  delta: number | null;
  direction: string;
  outcome: "IMPROVED" | "REGRESSED" | "UNCHANGED" | "UNKNOWN" | string;
  note: string;
}

/** One measured arm, recorded exactly as it ran. */
export interface ExperimentArm {
  label: string;
  strategy: string;
  role: string;
  config: ExperimentArmConfig;
  status: string;
  failure_reason: string;
  top_k: number;
  candidate_depth: number;
  config_fingerprint: string;
  embedding_model: string;
  retrieval_backend: string;
  reranker_status: string;
  index_points: number | null;
  duration_ms: number | null;
  question_count: number;
  code_counts: Record<string, number>;
  latency_ms: number | null;
  metric_stats: Record<string, ExperimentMetricStats>;
  per_question: ExperimentPerQuestion[];
  errors: string[];
  notes: string[];
}

/** One candidate's verdict, with the rules that produced it. */
export interface ExperimentCandidateOutcome {
  label: string;
  strategy: string;
  config_fingerprint: string;
  acceptance: "ACCEPT" | "REJECT" | "INCONCLUSIVE" | "NOT_VALID" | string;
  acceptance_reason: string;
  outcome: RetrievalExperimentOutcome;
  outcome_explanation: string;
  pairing: string;
  pairing_reason: string;
  primary_metric: string;
  primary_mean_delta: number | null;
  primary_required_delta: number;
  primary_improved: boolean;
  paired_question_count: number;
  paired_wins: number | null;
  paired_losses: number | null;
  guardrails_regressed: string[];
  guardrail_assessment: string[];
  rules_applied: string[];
}

export type RetrievalExperimentOutcome =
  | "PENDING"
  | "BLOCKED"
  | "INCONCLUSIVE"
  | "NO_MEANINGFUL_IMPROVEMENT"
  | "REGRESSION"
  | "IMPROVED";

/**
 * A stored experiment: the declaration, the measurements and the derived verdict.
 * `promoted_to_default` is always false — an experiment never changes the
 * default retrieval configuration.
 */
export interface RetrievalExperiment {
  id: string;
  kb_id: string;
  kb_name: string;
  name: string;
  slug: string;

  hypothesis: string;
  targeted_failure_code: string;
  targeted_question_ids: string[];
  proposed_change: string;
  expected_effect: string;
  primary_metric: string;
  guardrail_metrics: string[];
  comparison_method: string;
  rejection_conditions: string[];

  baseline: ExperimentArmConfig;
  candidates: ExperimentArmConfig[];
  benchmark_name: string;
  benchmark_fingerprint: string;
  benchmark_path: string;
  benchmark_sha256_before: string;
  benchmark_sha256_after: string;
  corpus_fingerprint: string | null;
  corpus_version: string | null;
  matched_question_ids: string[];
  question_set_fingerprint: string;

  status: string;
  arms: ExperimentArm[];
  comparisons: StrategyComparison[];
  candidate_outcomes: ExperimentCandidateOutcome[];
  outcome: RetrievalExperimentOutcome;
  outcome_explanation: string;
  accepted_candidate: string;
  promoted_to_default: boolean;

  integrity: Record<string, unknown>;
  limitations: string[];
  notes: string[];
  errors: string[];

  created_at: string;
  completed_at: string | null;
  created_by: string;
}

/** A row of the experiment list: the declaration plus the verdict, no arms. */
export interface RetrievalExperimentSummary {
  id: string;
  name: string;
  kb_id: string;
  created_at: string;
  completed_at: string | null;
  hypothesis: string;
  primary_metric: string;
  guardrail_metrics: string[];
  targeted_failure_code: string;
  targeted_question_ids: string[];
  benchmark_name: string;
  benchmark_fingerprint: string;
  question_set_fingerprint: string;
  matched_question_count: number;
  baseline: ExperimentArmConfig;
  candidates: ExperimentArmConfig[];
  outcome: RetrievalExperimentOutcome;
  outcome_explanation: string;
  accepted_candidate: string;
  promoted_to_default: boolean;
  candidate_outcomes: Array<{
    label: string;
    outcome: RetrievalExperimentOutcome;
    acceptance: string;
    pairing: string;
    primary_mean_delta: number | null;
    paired_question_count: number;
  }>;
  integrity: Record<string, unknown>;
  errors: string[];
}

export interface RetrievalExperimentMeta {
  outcomes: RetrievalExperimentOutcome[];
  meaningful_delta: Record<string, number>;
  comparison_method: string;
  promotion_policy: string;
  notice: string;
}

/**
 * V13 controlled experiments.
 *
 * `run` executes real retrieval over the benchmark and is READ-ONLY with respect
 * to the index; `list`/`get`/`meta` are pure reads. Nothing here changes the
 * saved retrieval configuration.
 */
export const retrievalExperimentsApi = {
  meta: (kbId: string) =>
    request<RetrievalExperimentMeta>(
      `/api/knowledge-bases/${kbId}/retrieval-experiments/meta`
    ),

  list: (kbId: string) =>
    request<RetrievalExperimentSummary[]>(
      `/api/knowledge-bases/${kbId}/retrieval-experiments`
    ),

  get: (kbId: string, experimentId: string) =>
    request<RetrievalExperiment>(
      `/api/knowledge-bases/${kbId}/retrieval-experiments/${experimentId}`
    ),

  run: (
    kbId: string,
    body: {
      name: string;
      hypothesis: string;
      baseline: Omit<ExperimentArmConfig, "role" | "rationale"> | Partial<ExperimentArmConfig>;
      candidates: Array<Partial<ExperimentArmConfig>>;
      primary_metric: string;
      guardrail_metrics?: string[];
      proposed_change?: string;
      expected_effect?: string;
      targeted_failure_code?: string;
      targeted_question_ids?: string[];
      rejection_conditions?: string[];
      benchmark_path?: string;
      question_ids?: string[];
      persist?: boolean;
    },
  ) =>
    request<RetrievalExperiment>(
      `/api/knowledge-bases/${kbId}/retrieval-experiments`,
      { method: "POST", body: JSON.stringify(body) },
    ),
};

// ---------------------------------------------------------------------------
// V14 held-out validation
// ---------------------------------------------------------------------------

/** Where a held-out question stands. `draft` means nobody has decided yet. */
export type HeldOutCandidateStatus = "draft" | "approved" | "edited" | "rejected";

/**
 * The two-tier lifecycle. PROVISIONAL means the artifact is sealed but at least
 * one question was never reviewed, so every result from it is exploratory.
 * VALIDATED requires a human decision on every surviving question, recorded
 * BEFORE the freeze.
 */
export type HeldOutLifecycle = "draft" | "provisional" | "validated";
export type HeldOutClaimStrength = "exploratory" | "confirmatory";

/** One validation finding. `error` blocks a freeze; `warning` does not. */
export interface HeldOutFinding {
  severity: "error" | "warning" | "info" | string;
  code: string;
  detail: string;
  action: string;
  question_id: string;
  applies_to_frozen: boolean;
}

/** One candidate's validation verdict. */
export interface HeldOutQuestionValidation {
  question_id: string;
  status: string;
  subdomain: string;
  question: string;
  expected_chunk_ids: string[];
  evidence_resolved: number;
  evidence_total: number;
  evidence_in_index: number | null;
  needs_review: boolean;
  findings: HeldOutFinding[];
}

/** Everything the validator established, and what it could not run. */
export interface HeldOutValidationReport {
  benchmark: string;
  kb_id: string;
  candidate_fingerprint: string;
  question_count: number;
  surviving_count: number;
  rejected_count: number;
  checks_run: string[];
  checks_not_run: Array<{ check: string; reason: string }>;
  errors: string[];
  warnings: string[];
  questions: HeldOutQuestionValidation[];
  separation: {
    development_benchmark_questions: number;
    development_benchmark_chunk_ids: number;
    development_document_ids: string[];
    thresholds: { token_jaccard: number; char_ngram_jaccard: number };
    candidates_sharing_expected_chunk_with_development: number;
    max_token_jaccard_vs_development: number;
    max_char_ngram_jaccard_vs_development: number;
    per_question: Record<
      string,
      {
        shared_expected_chunk_ids: string[];
        shared_with_development_questions: string[];
        nearest_development_question_id: string;
        nearest_development_question: string;
        token_jaccard: number;
        char_ngram_jaccard: number;
        paraphrase_suspected: boolean;
      }
    >;
    note: string;
  };
  composition: {
    subdomains: Record<string, number>;
    query_types: Record<string, number>;
    difficulty: Record<string, number>;
    documents_covered: string[];
    documents_not_covered_by_development: string[];
  };
  review: HeldOutReviewCompleteness;
  verdict: string;
  limitations: string[];
}

export interface HeldOutReviewCompleteness {
  candidates_total: number;
  candidates_surviving: number;
  candidates_rejected: number;
  candidates_reviewed: number;
  candidates_unreviewed: number;
  unreviewed_question_ids: string[];
  reviewers: string[];
  complete: boolean;
}

/** The held-out instrument's identity, validation state and freeze tier. */
export interface HeldOutStatus {
  kb_id: string;
  candidate_path: string;
  benchmark_path: string;
  dev_benchmark_path: string;
  candidate_fingerprint: string;
  candidate_sha256: string;
  candidate_count: number;
  surviving_count: number;
  rejected_count: number;
  review: HeldOutReviewCompleteness;
  composition: {
    subdomains: Record<string, number>;
    query_types: Record<string, number>;
    difficulty: Record<string, number>;
  };
  validation: HeldOutValidationReport;
  lifecycle: HeldOutLifecycle;
  claim_strength: HeldOutClaimStrength;
  frozen: null | {
    benchmark: string;
    version: number;
    fingerprint: string;
    sha256: string;
    frozen_at: string;
    frozen_by: string;
    question_count: number;
    lifecycle: HeldOutLifecycle;
    claim_strength: HeldOutClaimStrength;
    review: HeldOutReviewCompleteness;
    composition: Record<string, unknown>;
    authorship: Record<string, unknown>;
    protocol: Record<string, string>;
    can_be_run: boolean;
    supports_confirmatory_claim: boolean;
  };
  notice: string;
}

/** A candidate with the FULL text of the evidence its label rests on. */
export interface HeldOutReviewQuestion {
  id: string;
  question: string;
  subdomain: string;
  query_type: string;
  difficulty: string;
  status: HeldOutCandidateStatus;
  review: null | {
    decision: string;
    reviewer: string;
    reviewed_at: string;
    notes: string;
    previous_question: string;
    previous_chunk_ids: string[];
    previous_subdomain: string;
    results_seen: boolean;
  };
  expected_chunk_ids: string[];
  expected_document_ids: string[];
  evidence_quote: string;
  coverage_rationale: string;
  provenance: string;
  draft_origin: string;
  needs_review: boolean;
  evidence: Array<{
    chunk_id: string;
    document_id: string;
    document_title: string;
    section: string;
    char_count: number;
    text: string;
    resolves_in_corpus: boolean;
    present_in_index: boolean | null;
  }>;
  findings: HeldOutFinding[];
}

export interface HeldOutReviewView {
  kb_id: string;
  benchmark: string;
  candidate_fingerprint: string;
  question_count: number;
  review: HeldOutReviewCompleteness;
  questions: HeldOutReviewQuestion[];
  notice: string;
}

/** What the artifacts allow right now, so the UI never offers a dead action. */
export interface HeldOutConfig {
  kb_id: string;
  can_review: boolean;
  can_freeze: boolean;
  can_run: boolean;
  supports_confirmatory_claim: boolean;
  review_complete: boolean;
  lifecycle: HeldOutLifecycle;
  question_count: number;
  notice: string;
}

/** A cross-instrument generalization summary: dev vs held-out, unpaired. */
export interface HeldOutGeneralizationSummary {
  experiment: string;
  heldout_artifact: string;
  dev_artifact: string;
  retests: string;
  primary_metric: string;
  heldout_outcome: RetrievalExperimentOutcome;
  dev_outcome: RetrievalExperimentOutcome | "";
  heldout_claim_strength: string;
  heldout_benchmark_lifecycle: string;
  heldout_question_count: number;
  dev_question_count: number;
  arms: Record<
    string,
    {
      heldout: Record<string, number | null | string> | null;
      development: Record<string, number | null | string> | null;
    }
  >;
  agreement: Record<
    string,
    {
      comparable: boolean;
      reason?: string;
      heldout_delta?: number;
      heldout_outcome?: string;
      development_delta?: number;
      dev_outcome?: string;
      direction_agrees?: boolean;
    }
  >;
  note: string;
}

/** A held-out experiment row, including the arms and per-question movement. */
export interface HeldOutExperimentRow {
  id: string;
  name: string;
  instrument: string;
  benchmark_name: string;
  benchmark_fingerprint: string;
  benchmark_lifecycle: string;
  claim_strength: string;
  retests: string;
  hypothesis: string;
  primary_metric: string;
  guardrail_metrics: string[];
  rejection_conditions: string[];
  outcome: RetrievalExperimentOutcome;
  outcome_explanation: string;
  accepted_candidate: string;
  promoted_to_default: boolean;
  question_count: number;
  question_set_fingerprint: string;
  status: string;
  created_at: string;
  integrity: Record<string, unknown>;
  limitations: string[];
  arms: Array<{
    label: string;
    strategy: string;
    top_k: number;
    candidate_depth: number;
    params: Record<string, unknown>;
    status: string;
    failure_reason: string;
    reranker_status: string;
    metrics: Record<string, ExperimentMetricStats>;
  }>;
  candidate_outcomes: Array<{
    label: string;
    outcome: RetrievalExperimentOutcome;
    acceptance: string;
    acceptance_reason: string;
    outcome_explanation: string;
    pairing: string;
    primary_mean_delta: number | null;
    primary_required_delta: number;
    primary_sign_test_p: number | null;
    primary_sign_test_n: number;
    paired_question_count: number;
    paired_wins: number | null;
    paired_losses: number | null;
    guardrails_regressed: string[];
    guardrail_assessment: string[];
  }>;
  per_question: Array<{
    label: string;
    question_id: string;
    question: string;
    baseline: number | null;
    candidate: number | null;
    delta: number | null;
    outcome: string;
    direction: string;
  }>;
}

/**
 * V14 held-out validation. Reads are pure reads; `review` and `freeze` write only
 * the two held-out benchmark artifacts under `benchmarks/`. The dev benchmark and
 * the knowledge base are never written.
 */
export const heldOutApi = {
  status: (kbId: string) =>
    request<HeldOutStatus>(`/api/knowledge-bases/${kbId}/held-out/benchmark`),

  reviewView: (kbId: string) =>
    request<HeldOutReviewView>(`/api/knowledge-bases/${kbId}/held-out/review-view`),

  config: (kbId: string) =>
    request<HeldOutConfig>(`/api/knowledge-bases/${kbId}/held-out/config`),

  review: (
    kbId: string,
    decisions: Array<{
      question_id: string;
      decision: "approve" | "edit" | "reject";
      reviewer: string;
      notes?: string;
      question?: string | null;
      expected_chunk_ids?: string[] | null;
      subdomain?: string | null;
      results_seen?: boolean;
    }>,
  ) =>
    request<{
      candidates: Record<string, unknown>;
      review: HeldOutReviewCompleteness;
      validation: HeldOutValidationReport;
      notice: string;
    }>(`/api/knowledge-bases/${kbId}/held-out/review`, {
      method: "POST",
      body: JSON.stringify({ decisions }),
    }),

  freeze: (kbId: string, body: { frozen_by: string; notes?: string[] }) =>
    request<{
      benchmark: Record<string, unknown>;
      fingerprint: string;
      sha256: string;
      lifecycle: HeldOutLifecycle;
      claim_strength: HeldOutClaimStrength;
      path: string;
      validation: HeldOutValidationReport;
    }>(`/api/knowledge-bases/${kbId}/held-out/freeze`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  experiments: (kbId: string) =>
    request<{
      kb_id: string;
      heldout: null | {
        benchmark: string;
        fingerprint: string;
        lifecycle: HeldOutLifecycle;
        claim_strength: HeldOutClaimStrength;
        question_count: number;
        frozen_at: string;
        frozen_by: string;
      };
      experiments: HeldOutExperimentRow[];
      development_references: HeldOutExperimentRow[];
      generalization_summaries: HeldOutGeneralizationSummary[];
      notice: string;
    }>(`/api/knowledge-bases/${kbId}/held-out/experiments`),
};
