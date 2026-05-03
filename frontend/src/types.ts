import type { AxiosProgressEvent } from 'axios';

export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const API_KEY = import.meta.env.VITE_API_KEY || '';
export const PUBLIC_DEMO_MODE = import.meta.env.VITE_PUBLIC_DEMO_MODE === 'true';
export const DEMO_SESSION_STORAGE_KEY = 'ragit_demo_session_id';
export const AUTH_TOKEN_STORAGE_KEY = 'ragit_auth_token';
export const RESET_REQUEST_TIMEOUT_MS = 15000;
export const STARTER_PROMPTS = [
  'Summarize leave policy changes this year',
  'What are the vendor onboarding steps?',
  'Where is the SLA escalation path documented?',
];

export interface DemoSessionResponse {
  token: string;
  expires_at: number;
}

export interface Source {
  file_hash?: string | null;
  document: string;
  page: number;
  chunk_index?: number | null;
  relevance_score?: number | null;
  snippet?: string | null;
  section?: string | null;
  vector_score?: number | null;
  lexical_score?: number | null;
  bm25_score?: number | null;
  final_score?: number | null;
  reranker_applied?: boolean | null;
}

export interface RetrievalDiagnostics {
  query_variants_used: string[];
  is_broad_question: boolean;
  is_simple_query?: boolean;
  fast_mode_applied?: boolean;
  fallback_applied: boolean;
  candidates_considered: number;
  reranker_applied?: boolean;
  reranker_skipped_reason?: string;
  retrieval_ms?: number;
  rerank_ms?: number;
  context_build_ms?: number;
  generation_ms?: number;
  total_pipeline_ms?: number;
  low_confidence_fallback_used?: boolean;
  verification_applied?: boolean;
}

export interface UploadItemResult {
  filename: string;
  chunks_indexed: number;
  status: 'success' | 'duplicate' | 'failed' | string;
  message: string;
  file_hash?: string | null;
  document_id?: string | null;
  parsing_method?: string | null;
  vision_calls_used?: number;
}

export interface UploadBatchResponse {
  files?: UploadItemResult[];
  total_files?: number;
  processed_files?: number;
  total_chunks_indexed?: number;
}

export interface KnowledgeBaseFilesResponse {
  files?: KnowledgeBaseFileApiItem[];
}

export interface DocumentChunk {
  id: string;
  content: string;
  page: number;
  section: string;
  chunk_index: number;
  metadata: Record<string, unknown>;
}

export interface DocumentChunksResponse {
  file_hash: string;
  filename: string;
  chunks: DocumentChunk[];
  focus_chunk_index?: number | null;
}

export interface IngestionJobStatus {
  job_id: string;
  status: string;
  stage: string;
  message: string;
  total_files: number;
  processed_files: number;
  total_chunks_indexed: number;
  results: UploadItemResult[];
}

export interface ModelHealthItem {
  name: string;
  status: string;
  detail: string;
}

export interface AdminOverview {
  document_count: number;
  chunk_count: number;
  feedback_count: number;
  chat_session_count: number;
  eval_run_count: number;
  user_count: number;
  audit_event_count: number;
  recent_feedback: Array<{
    id: number;
    created_at: number;
    question: string;
    rating: string;
    reason?: string;
    confidence_score?: number | null;
  }>;
  metadata_db_path: string;
  embedding_model: string;
  embedding_device: string;
  docling_enabled: boolean;
  reranker_enabled: boolean;
  openai_enabled: boolean;
}

export interface AuthStatus {
  auth_enabled: boolean;
  has_users: boolean;
  bootstrap_required: boolean;
}

export interface AuthUser {
  id: string;
  email: string;
  display_name: string;
  role: 'admin' | 'user' | string;
  disabled: number;
  created_at: number;
  updated_at: number;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_at: number;
  user: AuthUser;
}

export interface AuditEvent {
  id: number;
  created_at: number;
  actor_email: string;
  action: string;
  resource_type: string;
  resource_id: string;
  detail: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface ChatMessageApiItem {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence_score?: number | null;
  confidence_level?: Message['confidence_level'];
  diagnostics?: RetrievalDiagnostics | null;
}

export interface EvalRun {
  id: string;
  created_at: number;
  status: string;
  total: number;
  passed: number;
  failed: number;
  message: string;
  results: Array<{
    eval_id: string;
    passed: boolean;
    message: string;
  }>;
}

export interface KnowledgeBaseFileApiItem {
  file_hash: string;
  document_id?: string;
  filename: string;
  chunk_count: number;
  indexed_at: number;
  parsing_method?: string;
  upload_status?: string;
  vision_calls_used?: number;
  embedding_model?: string;
  owner_user_id?: string;
  visibility?: string;
  allowed_roles?: string[];
}

export interface KnowledgeFile {
  fileHash: string;
  documentId?: string;
  filename: string;
  chunks: number;
  indexedAt: number;
  parsingMethod: string;
  uploadStatus: string;
  visionCallsUsed: number;
  embeddingModel: string;
  ownerUserId: string;
  visibility: string;
  allowedRoles: string[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence_score?: number | null;
  confidence_level?: 'high' | 'medium' | 'low' | null;
  diagnostics?: RetrievalDiagnostics | null;
  timestamp?: number;
}

export interface StreamDonePayload {
  answer?: string;
  sources?: Source[];
  confidence_score?: number | null;
  confidence_level?: Message['confidence_level'];
  diagnostics?: RetrievalDiagnostics | null;
}

export type StreamEventPayload = StreamDonePayload & {
  text?: string;
  detail?: string;
};

export type UploadProgressHandler = (event: AxiosProgressEvent) => void;
