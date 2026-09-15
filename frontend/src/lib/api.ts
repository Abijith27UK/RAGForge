export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export interface KnowledgeBase {
  id: string;
  name: string;
  domain: string;
  purpose: string;
  target_audience: string;
  depth: string;
  status: string;
  created_at: string;
  updated_at: string;
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
  domain: string | null;
  subdomain: string | null;
  trust_score: number | null;
  content_hash: string;
  char_count: number;
}

export interface RetrievalResult {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  provenance: Record<string, string | number | null>;
}

export interface RetrievalResponse {
  query: string;
  top_k: number;
  results: RetrievalResult[];
  embedding_model: string;
  retrieval_backend: string;
  error: string | null;
}

export interface EvaluationQuestion {
  id: string;
  kb_id: string;
  question: string;
  expected_chunk_ids: string[];
  expected_document_ids: string[];
  expected_keywords: string[];
  generated_by: string;
  notes?: string;
}

export interface EvaluationRun {
  id: string;
  kb_id: string;
  config: { top_k: number; question_ids?: string[] };
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
  evaluate: (id: string, topK = 5, label = "", allowKeywordFallback = false) =>
    request<EvaluationRun>(`/api/knowledge-bases/${id}/evaluate`, {
      method: "POST",
      body: JSON.stringify({ top_k: topK, label, allow_keyword_fallback: allowKeywordFallback }),
    }),
  listEvaluationRuns: (id: string) =>
    request<EvaluationRun[]>(`/api/knowledge-bases/${id}/evaluation-runs`),
};
