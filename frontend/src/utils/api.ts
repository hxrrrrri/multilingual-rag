import axios from "axios";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE, timeout: 60000 });

// ── Types ──────────────────────────────────────────────────────────────────

export interface Document {
  document_id: string;
  filename: string;
  status: "pending" | "processing" | "ready" | "failed";
  language_detected?: string;
  page_count?: number;
  chunk_count?: number;
  ocr_confidence?: number;
  processing_time_seconds?: number;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface SourceChunk {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  text: string;
  similarity_score: number;
  retrieval_method: string;
}

export interface QueryResponse {
  query_id: string;
  query: string;
  answer: string;
  confidence: number;
  language_detected: string;
  sources: SourceChunk[];
  faithfulness_score?: number;
  latency_ms: number;
  model_used: string;
  created_at: string;
}

// ── Documents ──────────────────────────────────────────────────────────────

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/api/documents/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listDocuments(page = 1, pageSize = 20) {
  const { data } = await api.get("/api/documents", { params: { page, page_size: pageSize } });
  return data as { documents: Document[]; total: number };
}

export async function getDocument(id: string): Promise<Document> {
  const { data } = await api.get(`/api/documents/${id}`);
  return data;
}

export async function deleteDocument(id: string): Promise<void> {
  await api.delete(`/api/documents/${id}`);
}

// ── Query ──────────────────────────────────────────────────────────────────

export async function queryDocuments(
  query: string,
  documentIds?: string[],
  topK = 5
): Promise<QueryResponse> {
  const { data } = await api.post("/api/query", {
    query,
    document_ids: documentIds ?? null,
    top_k: topK,
    include_sources: true,
  });
  return data;
}

export async function getHistory(limit = 20) {
  const { data } = await api.get("/api/query/history", { params: { limit } });
  return data as { queries: any[] };
}

// ── Feedback ───────────────────────────────────────────────────────────────

export async function submitFeedback(
  queryId: string,
  sentiment: "positive" | "negative",
  comment?: string
) {
  const { data } = await api.post("/api/feedback", { query_id: queryId, sentiment, comment });
  return data;
}

// ── Health ─────────────────────────────────────────────────────────────────

export async function getHealth() {
  const { data } = await api.get("/health");
  return data;
}
