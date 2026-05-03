import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Loader2,
} from 'lucide-react';
import '../App.css';
import { ChatWindow } from './Chat/ChatWindow';
import { SourcePanel } from './Chat/SourcePanel';
import { DocumentSidebar } from './Documents/DocumentSidebar';

// API Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';
const PUBLIC_DEMO_MODE = import.meta.env.VITE_PUBLIC_DEMO_MODE === 'true';
const DEMO_SESSION_STORAGE_KEY = 'ragit_demo_session_id';
const API_HEADERS: Record<string, string> = API_KEY ? { 'X-API-Key': API_KEY } : {};
const AUTH_TOKEN_STORAGE_KEY = 'ragit_auth_token';
const RESET_REQUEST_TIMEOUT_MS = 15000;
interface DemoSessionResponse {
  token: string;
  expires_at: number;
}

function getStoredDemoSessionId(): string {
  if (!PUBLIC_DEMO_MODE) return '';
  return window.localStorage.getItem(DEMO_SESSION_STORAGE_KEY) || '';
}

async function ensureDemoSessionId(): Promise<string> {
  const existing = getStoredDemoSessionId();
  if (existing) return existing;
  const response = await axios.post<DemoSessionResponse>(
    `${API_BASE}/demo/session`,
    undefined,
    { headers: API_HEADERS }
  );
  window.localStorage.setItem(DEMO_SESSION_STORAGE_KEY, response.data.token);
  return response.data.token;
}

interface Source {
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

interface RetrievalDiagnostics {
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

interface UploadItemResult {
  filename: string;
  chunks_indexed: number;
  status: 'success' | 'duplicate' | 'failed' | string;
  message: string;
  file_hash?: string | null;
  document_id?: string | null;
  parsing_method?: string | null;
  vision_calls_used?: number;
}

interface UploadBatchResponse {
  files?: UploadItemResult[];
  total_files?: number;
  processed_files?: number;
  total_chunks_indexed?: number;
}

interface KnowledgeBaseFilesResponse {
  files?: KnowledgeBaseFileApiItem[];
}

interface DocumentChunk {
  id: string;
  content: string;
  page: number;
  section: string;
  chunk_index: number;
  metadata: Record<string, unknown>;
}

interface DocumentChunksResponse {
  file_hash: string;
  filename: string;
  chunks: DocumentChunk[];
  focus_chunk_index?: number | null;
}

interface IngestionJobStatus {
  job_id: string;
  status: string;
  stage: string;
  message: string;
  total_files: number;
  processed_files: number;
  total_chunks_indexed: number;
  results: UploadItemResult[];
}

interface ModelHealthItem {
  name: string;
  status: string;
  detail: string;
}

interface AdminOverview {
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

interface AuthStatus {
  auth_enabled: boolean;
  has_users: boolean;
  bootstrap_required: boolean;
}

interface AuthUser {
  id: string;
  email: string;
  display_name: string;
  role: 'admin' | 'user' | string;
  disabled: number;
  created_at: number;
  updated_at: number;
}

interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_at: number;
  user: AuthUser;
}

interface AuditEvent {
  id: number;
  created_at: number;
  actor_email: string;
  action: string;
  resource_type: string;
  resource_id: string;
  detail: Record<string, unknown>;
}

interface ChatSession {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
}

interface ChatMessageApiItem {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence_score?: number | null;
  confidence_level?: Message['confidence_level'];
  diagnostics?: RetrievalDiagnostics | null;
}

interface EvalRun {
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

interface KnowledgeBaseFileApiItem {
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

interface KnowledgeFile {
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

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence_score?: number | null;
  confidence_level?: 'high' | 'medium' | 'low' | null;
  diagnostics?: RetrievalDiagnostics | null;
}

interface StreamDonePayload {
  answer?: string;
  sources?: Source[];
  confidence_score?: number | null;
  confidence_level?: Message['confidence_level'];
  diagnostics?: RetrievalDiagnostics | null;
}

type StreamEventPayload = StreamDonePayload & {
  text?: string;
  detail?: string;
};

function getApiErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    if (error.response?.status === 401) {
      return 'Login expired or API key missing. Sign in again.';
    }
    if (error.response?.status === 403) {
      return 'Your account does not have permission for that action.';
    }
    if (error.response?.data?.detail) {
      return String(error.response.data.detail);
    }
  }
  return fallback;
}

function getStoredAuthToken(): string {
  return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || '';
}

function buildApiHeaders(token: string, demoSessionId: string): Record<string, string> {
  return {
    ...API_HEADERS,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(demoSessionId ? { 'X-Demo-Session-Id': demoSessionId } : {}),
  };
}

function mergeKnowledgeFiles(current: KnowledgeFile[], incoming: KnowledgeFile[]): KnowledgeFile[] {
  const byKey = new Map<string, KnowledgeFile>();
  for (const item of current) {
    byKey.set(item.fileHash || item.filename, item);
  }
  for (const item of incoming) {
    const key = item.fileHash || item.filename;
    const existing = byKey.get(key);
    if (!existing) {
      byKey.set(key, item);
      continue;
    }
    byKey.set(key, {
      ...existing,
      ...item,
      filename: item.filename,
      chunks: Math.max(existing.chunks, item.chunks),
    });
  }
  return [...byKey.values()].sort((a, b) => a.filename.localeCompare(b.filename));
}

function mapKnowledgeFile(item: KnowledgeBaseFileApiItem): KnowledgeFile {
  return {
    fileHash: item.file_hash,
    documentId: item.document_id,
    filename: item.filename,
    chunks: Number(item.chunk_count || 0),
    indexedAt: Number(item.indexed_at || 0),
    parsingMethod: item.parsing_method || 'unknown',
    uploadStatus: item.upload_status || 'indexed',
    visionCallsUsed: Number(item.vision_calls_used || 0),
    embeddingModel: item.embedding_model || '',
    ownerUserId: item.owner_user_id || '',
    visibility: item.visibility || 'shared',
    allowedRoles: item.allowed_roles || [],
  };
}

function mapUploadResult(item: UploadItemResult): KnowledgeFile | null {
  if (!item.file_hash) return null;
  return {
    fileHash: item.file_hash,
    documentId: item.document_id || undefined,
    filename: item.filename,
    chunks: Number(item.chunks_indexed || 0),
    indexedAt: Math.floor(Date.now() / 1000),
    parsingMethod: item.parsing_method || 'unknown',
    uploadStatus: item.status === 'duplicate' ? 'duplicate' : 'indexed',
    visionCallsUsed: Number(item.vision_calls_used || 0),
    embeddingModel: '',
    ownerUserId: '',
    visibility: 'shared',
    allowedRoles: [],
  };
}

function formatIndexedAt(value: number): string {
  if (!value) return 'Indexed time unknown';
  return new Date(value * 1000).toLocaleString();
}

function formatPercent(value?: number | null): string {
  if (typeof value !== 'number') return 'n/a';
  return `${Math.round(value * 100)}%`;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function AppShell() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputTitle, setInputTitle] = useState('');
  const [knowledgeFiles, setKnowledgeFiles] = useState<KnowledgeFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [documentActionId, setDocumentActionId] = useState<string | null>(null);
  const [uploadSteps, setUploadSteps] = useState<string[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<DocumentChunksResponse | null>(null);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [adminOverview, setAdminOverview] = useState<AdminOverview | null>(null);
  const [modelHealth, setModelHealth] = useState<ModelHealthItem[]>([]);
  const [isLoadingAdmin, setIsLoadingAdmin] = useState(false);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [evalRuns, setEvalRuns] = useState<EvalRun[]>([]);
  const [isRunningEval, setIsRunningEval] = useState(false);
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);
  const [authToken, setAuthToken] = useState(getStoredAuthToken);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authDisplayName, setAuthDisplayName] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [demoSessionId, setDemoSessionId] = useState(getStoredDemoSessionId);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryInputRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const lastScrollTopRef = useRef(0);
  const apiHeaders = React.useMemo(() => buildApiHeaders(authToken, demoSessionId), [authToken, demoSessionId]);
  const canUseApi = Boolean(
    authStatus && ((PUBLIC_DEMO_MODE && demoSessionId) || !authStatus.auth_enabled || authUser || API_KEY)
  );

  // Auto-scroll only when user is already near bottom.
  useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleChatScroll = () => {
    const el = chatContainerRef.current;
    if (!el) return;

    const previousTop = lastScrollTopRef.current;
    const currentTop = el.scrollTop;
    const scrolledUp = currentTop < previousTop;

    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom < 60;

    // If user actively scrolls upward, release bottom-lock immediately.
    if (scrolledUp && !nearBottom) {
      shouldAutoScrollRef.current = false;
    } else if (nearBottom) {
      // Re-enable only when user returns close to the latest content.
      shouldAutoScrollRef.current = true;
    }

    lastScrollTopRef.current = currentTop;
  };

  const handleChatWheel: React.WheelEventHandler<HTMLDivElement> = (event) => {
    // If user scrolls upward, stop pinning to bottom while streaming.
    if (event.deltaY < 0) {
      shouldAutoScrollRef.current = false;
    }
  };

  const handleChatTouchStart: React.TouchEventHandler<HTMLDivElement> = () => {
    // Mobile/manual interaction should also release auto-scroll pinning.
    shouldAutoScrollRef.current = false;
  };

  const handleChatMouseDown: React.MouseEventHandler<HTMLDivElement> = () => {
    // Dragging scrollbar or selecting content should not force-follow output.
    shouldAutoScrollRef.current = false;
  };

  useEffect(() => {
    const loadAuthStatus = async () => {
      try {
        const response = await axios.get<AuthStatus>(`${API_BASE}/auth/status`);
        setAuthStatus(response.data);
      } catch (error) {
        console.warn('Failed to load auth status.', error);
        setAuthStatus({ auth_enabled: false, has_users: false, bootstrap_required: false });
      }
    };
    void loadAuthStatus();
  }, []);

  useEffect(() => {
    if (!PUBLIC_DEMO_MODE) return;
    let cancelled = false;
    const loadDemoSession = async () => {
      try {
        const token = await ensureDemoSessionId();
        if (!cancelled) setDemoSessionId(token);
      } catch (error) {
        console.warn('Failed to create demo session.', error);
        window.localStorage.removeItem(DEMO_SESSION_STORAGE_KEY);
        if (!cancelled) setDemoSessionId('');
      }
    };
    void loadDemoSession();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (PUBLIC_DEMO_MODE || !authStatus || (!authToken && !API_KEY)) return;
    const loadCurrentUser = async () => {
      try {
        const response = await axios.get<{ user: AuthUser }>(`${API_BASE}/auth/me`, {
          headers: apiHeaders,
        });
        setAuthUser(response.data.user);
      } catch (error) {
        console.warn('Failed to restore login.', error);
        window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
        setAuthToken('');
        setAuthUser(null);
      }
    };
    void loadCurrentUser();
  }, [apiHeaders, authStatus, authToken]);

  const handleAuthSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!authEmail.trim() || !authPassword.trim()) return;

    setIsAuthenticating(true);
    setAuthError(null);
    try {
      const endpoint = authStatus?.bootstrap_required ? '/auth/bootstrap' : '/auth/login';
      const payload = authStatus?.bootstrap_required
        ? { email: authEmail, password: authPassword, display_name: authDisplayName }
        : { email: authEmail, password: authPassword };
      const response = await axios.post<AuthTokenResponse>(`${API_BASE}${endpoint}`, payload);
      window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, response.data.access_token);
      setAuthToken(response.data.access_token);
      setAuthUser(response.data.user);
      setAuthPassword('');
      setAuthStatus((prev) => prev ? { ...prev, has_users: true, bootstrap_required: false } : prev);
    } catch (error) {
      setAuthError(getApiErrorMessage(error, 'Sign-in failed.'));
    } finally {
      setIsAuthenticating(false);
    }
  };

  const handleLogout = async () => {
    try {
      await axios.post(`${API_BASE}/auth/logout`, undefined, { headers: apiHeaders });
    } catch (error) {
      console.warn('Logout request failed.', error);
    }
    window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    setAuthToken('');
    setAuthUser(null);
    setMessages([]);
    setKnowledgeFiles([]);
    setChatSessions([]);
    setActiveSessionId(null);
  };

  const loadIndexedFiles = useCallback(async () => {
    if (!canUseApi) return;
    try {
      const response = await axios.get<KnowledgeBaseFilesResponse>(
        `${API_BASE}/knowledge-base/files`,
        { headers: apiHeaders }
      );
      const files = (response.data?.files || []).map(mapKnowledgeFile);
      setKnowledgeFiles(files);
    } catch (error) {
      // Keep UI usable even if this optional hydration call fails.
      console.warn('Failed to load indexed files on startup.', error);
    }
  }, [apiHeaders, canUseApi]);

  useEffect(() => {
    if (!canUseApi) return;
    void loadIndexedFiles();
  }, [canUseApi, loadIndexedFiles]);

  const loadChatSessions = useCallback(async () => {
    if (!canUseApi || PUBLIC_DEMO_MODE) return;
    try {
      const response = await axios.get<{ sessions: ChatSession[] }>(
        `${API_BASE}/chat/sessions`,
        { headers: apiHeaders }
      );
      setChatSessions(response.data?.sessions || []);
    } catch (error) {
      console.warn('Failed to load chat sessions.', error);
    }
  }, [apiHeaders, canUseApi]);

  useEffect(() => {
    if (!canUseApi || PUBLIC_DEMO_MODE) return;
    void loadChatSessions();
  }, [canUseApi, loadChatSessions]);

  const ensureChatSession = async (): Promise<string> => {
    if (activeSessionId) return activeSessionId;
    const response = await axios.post<ChatSession>(
      `${API_BASE}/chat/sessions`,
      { title: 'New chat' },
      { headers: apiHeaders }
    );
    setActiveSessionId(response.data.id);
    setChatSessions((prev) => [response.data, ...prev]);
    return response.data.id;
  };

  const saveChatMessage = async (sessionId: string, message: Message) => {
    try {
      await axios.post(
        `${API_BASE}/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
        {
          id: message.id,
          role: message.role,
          content: message.content,
          sources: message.sources || [],
          diagnostics: message.diagnostics || undefined,
          confidence_score: message.confidence_score,
          confidence_level: message.confidence_level,
        },
        { headers: apiHeaders }
      );
      void loadChatSessions();
    } catch (error) {
      console.warn('Failed to save chat message.', error);
    }
  };

  const loadChatMessages = async (session: ChatSession) => {
    try {
      const response = await axios.get<{ messages: ChatMessageApiItem[] }>(
        `${API_BASE}/chat/sessions/${encodeURIComponent(session.id)}/messages`,
        { headers: apiHeaders }
      );
      const loaded = (response.data?.messages || []).map((item) => ({
        id: item.id,
        role: item.role,
        content: item.content,
        sources: item.sources,
        confidence_score: item.confidence_score,
        confidence_level: item.confidence_level,
        diagnostics: item.diagnostics,
      }));
      setActiveSessionId(session.id);
      setMessages(loaded);
      setFeedbackStatus(null);
    } catch (error) {
      console.warn('Failed to load chat messages.', error);
    }
  };

  const startNewChat = async () => {
    const response = await axios.post<ChatSession>(
      `${API_BASE}/chat/sessions`,
      { title: 'New chat' },
      { headers: apiHeaders }
    );
    setActiveSessionId(response.data.id);
    setChatSessions((prev) => [response.data, ...prev]);
    setMessages([]);
  };

  const deleteChatSession = async (session: ChatSession) => {
    const confirmed = window.confirm(`Delete chat "${session.title || 'New chat'}"?`);
    if (!confirmed) return;

    try {
      await axios.delete(
        `${API_BASE}/chat/sessions/${encodeURIComponent(session.id)}`,
        { headers: apiHeaders }
      );
      setChatSessions((prev) => prev.filter((item) => item.id !== session.id));
      if (activeSessionId === session.id) {
        setActiveSessionId(null);
        setMessages([]);
        setFeedbackStatus(null);
      }
    } catch (error) {
      console.warn('Failed to delete chat session.', error);
      setFeedbackStatus(getApiErrorMessage(error, 'Failed to delete chat.'));
    }
  };

  const loadEvalRuns = useCallback(async () => {
    try {
      const response = await axios.get<{ runs: EvalRun[] }>(
        `${API_BASE}/evals/runs`,
        { headers: apiHeaders }
      );
      setEvalRuns(response.data?.runs || []);
    } catch (error) {
      console.warn('Failed to load eval runs.', error);
    }
  }, [apiHeaders]);

  const loadAdminDebug = useCallback(async () => {
    if (!canUseApi) return;
    setIsLoadingAdmin(true);
    try {
      const [overviewResponse, healthResponse, auditResponse] = await Promise.all([
        axios.get<AdminOverview>(`${API_BASE}/admin/overview`, { headers: apiHeaders }),
        axios.get<{ checks: ModelHealthItem[] }>(`${API_BASE}/health/models`, { headers: apiHeaders }),
        axios.get<{ events: AuditEvent[] }>(`${API_BASE}/admin/audit-log`, { headers: apiHeaders }),
      ]);
      setAdminOverview(overviewResponse.data);
      setModelHealth(healthResponse.data?.checks || []);
      setAuditEvents(auditResponse.data?.events || []);
      await loadEvalRuns();
    } catch (error) {
      console.warn('Failed to load admin/debug data.', error);
    } finally {
      setIsLoadingAdmin(false);
    }
  }, [apiHeaders, canUseApi, loadEvalRuns]);

  const startEvalRun = async () => {
    setIsRunningEval(true);
    try {
      const response = await axios.post<{ run_id: string }>(
        `${API_BASE}/evals/runs`,
        undefined,
        { headers: apiHeaders }
      );
      const runId = response.data.run_id;
      for (let i = 0; i < 120; i += 1) {
        const runResponse = await axios.get<EvalRun>(
          `${API_BASE}/evals/runs/${encodeURIComponent(runId)}`,
          { headers: apiHeaders }
        );
        setEvalRuns((prev) => [runResponse.data, ...prev.filter((run) => run.id !== runId)]);
        if (runResponse.data.status !== 'running') break;
        await delay(1500);
      }
      void loadAdminDebug();
    } catch (error) {
      console.warn('Failed to run evals.', error);
    } finally {
      setIsRunningEval(false);
    }
  };

  const uploadFiles = async (files: File[]) => {
    if (!files.length) return;

    const pdfFiles = files.filter((file) => file.name.toLowerCase().endsWith('.pdf'));
    if (!pdfFiles.length) {
      setUploadStatus('Only PDF files are supported. Please upload .pdf files.');
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Indexing ${pdfFiles.length} file(s)...`);
    setUploadSteps(['Queued files', 'Uploading PDFs']);
    
    const formData = new FormData();
    pdfFiles.forEach((file) => formData.append('files', file));

    try {
      const response = await axios.post<{ job_id: string }>(`${API_BASE}/upload/jobs`, formData, {
        headers: {
          ...apiHeaders,
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (event) => {
          if (!event.total) return;
          const percent = Math.round((event.loaded / event.total) * 100);
          setUploadSteps([
            'Queued files',
            `Uploading PDFs (${percent}%)`,
            'Parsing, chunking, embedding, and saving index',
          ]);
        },
      });
      const jobId = response.data.job_id;
      let data: UploadBatchResponse = {
        files: [],
        total_files: pdfFiles.length,
        processed_files: 0,
        total_chunks_indexed: 0,
      };
      for (let i = 0; i < 720; i += 1) {
        const jobResponse = await axios.get<IngestionJobStatus>(
          `${API_BASE}/upload/jobs/${encodeURIComponent(jobId)}`,
          { headers: apiHeaders }
        );
        const job = jobResponse.data;
        setUploadStatus(job.message || `${job.stage}: ${job.status}`);
        setUploadSteps([
          'Uploaded PDFs',
          `${job.stage || 'processing'}: ${job.status}`,
          `Processed ${job.processed_files}/${job.total_files} file(s)`,
          `Indexed ${job.total_chunks_indexed} chunks`,
        ]);
        data = {
          files: job.results,
          total_files: job.total_files,
          processed_files: job.processed_files,
          total_chunks_indexed: job.total_chunks_indexed,
        };
        if (job.status === 'completed' || job.status === 'failed') break;
        await delay(1500);
      }
      const totalFiles = Number(data?.total_files ?? pdfFiles.length);
      const processedFiles = Number(data?.processed_files ?? 0);
      const totalChunks = Number(data?.total_chunks_indexed ?? 0);
      setUploadStatus(
        `Upload completed. Processed ${processedFiles}/${totalFiles} files. Indexed ${totalChunks} chunks.`
      );
      const methods = [...new Set((data?.files || []).map((item) => item.parsing_method).filter(Boolean))];
      setUploadSteps([
        'Queued files',
        'Uploaded PDFs',
        methods.length ? `Parsed with ${methods.join(', ')}` : 'Parsed documents',
        `Indexed ${totalChunks} chunks`,
        'Complete',
      ]);

      const incoming: KnowledgeFile[] = (data?.files || [])
        .filter((item) => item.status === 'success' || item.status === 'duplicate')
        .map(mapUploadResult)
        .filter((item): item is KnowledgeFile => item !== null);
      if (incoming.length) {
        setKnowledgeFiles((prev) => mergeKnowledgeFiles(prev, incoming));
      } else {
        void loadIndexedFiles();
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus(getApiErrorMessage(error, "Failed to upload files. Is the backend running?"));
      setUploadSteps((prev) => [...prev, 'Failed']);
    } finally {
      setIsUploading(false);
      // Reset input
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    await uploadFiles(files);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!isUploading) {
      setIsDragActive(true);
    }
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!event.currentTarget.contains(event.relatedTarget as Node)) {
      setIsDragActive(false);
    }
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragActive(false);
    if (isUploading || isResetting) return;

    const files = Array.from(event.dataTransfer.files || []);
    if (!files.length) return;
    await uploadFiles(files);
  };

  const handleStarterPrompt = (prompt: string) => {
    setInputTitle(prompt);
    queryInputRef.current?.focus();
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputTitle.trim() || isQuerying) return;

    const sessionId = PUBLIC_DEMO_MODE ? '' : await ensureChatSession();
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: inputTitle };
    const assistantId = (Date.now() + 1).toString();
    let assistantDraft = '';
    let finalAssistant: Message | null = null;
    shouldAutoScrollRef.current = true;
    setMessages(prev => [
      ...prev,
      userMsg,
      { id: assistantId, role: 'assistant', content: '' },
    ]);
    setInputTitle('');
    setIsQuerying(true);
    if (sessionId) void saveChatMessage(sessionId, userMsg);

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...apiHeaders },
        body: JSON.stringify({ question: userMsg.content }),
      });

      if (response.status === 401) {
        throw new Error('Login expired or API key missing. Sign in again.');
      }

      if (!response.ok || !response.body) {
        throw new Error(`Streaming request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const applyChunk = (text: string) => {
        if (!text) return;
        assistantDraft += text;
        setMessages(prev => prev.map(msg => (
          msg.id === assistantId
            ? { ...msg, content: `${msg.content}${text}` }
            : msg
        )));
      };

      const applyDone = (payload: StreamDonePayload) => {
        finalAssistant = {
          id: assistantId,
          role: 'assistant',
          content: payload.answer || assistantDraft,
          sources: payload.sources,
          confidence_score: payload.confidence_score,
          confidence_level: payload.confidence_level,
          diagnostics: payload.diagnostics,
        };
        setMessages(prev => prev.map(msg => (
          msg.id === assistantId
            ? {
                ...msg,
                content: payload.answer || msg.content,
                sources: payload.sources,
                confidence_score: payload.confidence_score,
                confidence_level: payload.confidence_level,
                diagnostics: payload.diagnostics,
              }
            : msg
        )));
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const eventBlock of events) {
          const lines = eventBlock.split('\n');
          const eventName = lines.find(line => line.startsWith('event:'))?.replace('event:', '').trim();
          const dataLine = lines.find(line => line.startsWith('data:'))?.replace('data:', '').trim();
          if (!eventName || !dataLine) continue;

          const payload = JSON.parse(dataLine) as StreamEventPayload;

          if (eventName === 'chunk') {
            applyChunk(String(payload.text || ''));
          } else if (eventName === 'done') {
            applyDone(payload);
          } else if (eventName === 'error') {
            throw new Error(String(payload.detail || 'Streaming query failed'));
          }
        }
      }

      if (finalAssistant && sessionId) {
        void saveChatMessage(sessionId, finalAssistant);
      }
    } catch (error) {
      console.error("Query error:", error);
      const errorText = error instanceof Error && error.message
        ? error.message
        : "Sorry, there was an error processing your request. Please ensure the backend server is running.";

      setMessages(prev => prev.map(msg => (
        msg.id === assistantId
          ? { ...msg, content: errorText }
          : msg
      )));
    } finally {
      setIsQuerying(false);
    }
  };

  const handleResetKnowledgeBase = async () => {
    if (isResetting) return;

    const confirmed = window.confirm(
      'This will remove all uploaded PDFs and clear the index. Continue?'
    );
    if (!confirmed) return;

    setIsResetting(true);
    setUploadStatus('Resetting knowledge base...');

    try {
      const response = await axios.post(
        `${API_BASE}/knowledge-base/reset`,
        undefined,
        { timeout: RESET_REQUEST_TIMEOUT_MS, headers: apiHeaders }
      );
      const deleted = Number(response.data?.uploads_deleted ?? 0);
      setMessages([]);
      setKnowledgeFiles([]);
      setUploadStatus(`Knowledge base reset complete. Deleted ${deleted} uploaded file(s).`);
    } catch (error) {
      console.error('Reset error:', error);
      if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
        setUploadStatus('Reset request timed out. Check backend status and try again.');
        return;
      }
      setUploadStatus(getApiErrorMessage(error, 'Failed to reset knowledge base.'));
    } finally {
      setIsResetting(false);
    }
  };

  const handleDeleteKnowledgeFile = async (file: KnowledgeFile) => {
    if (!file.fileHash || documentActionId) return;

    const confirmed = window.confirm(`Delete ${file.filename} from the knowledge base?`);
    if (!confirmed) return;

    const actionId = `delete:${file.fileHash}`;
    setDocumentActionId(actionId);
    setUploadStatus(`Deleting ${file.filename}...`);

    try {
      const response = await axios.delete(
        `${API_BASE}/knowledge-base/files/${encodeURIComponent(file.fileHash)}`,
        { headers: apiHeaders }
      );
      const chunksDeleted = Number(response.data?.chunks_deleted ?? 0);
      setKnowledgeFiles((prev) => prev.filter((item) => item.fileHash !== file.fileHash));
      setUploadStatus(`Deleted ${file.filename}. Removed ${chunksDeleted} chunk(s).`);
    } catch (error) {
      console.error('Delete error:', error);
      setUploadStatus(getApiErrorMessage(error, `Failed to delete ${file.filename}.`));
    } finally {
      setDocumentActionId(null);
    }
  };

  const handleReindexKnowledgeFile = async (file: KnowledgeFile) => {
    if (!file.fileHash || documentActionId) return;

    const actionId = `reindex:${file.fileHash}`;
    setDocumentActionId(actionId);
    setUploadStatus(`Reindexing ${file.filename}...`);

    try {
      const response = await axios.post<UploadItemResult>(
        `${API_BASE}/knowledge-base/files/${encodeURIComponent(file.fileHash)}/reindex`,
        undefined,
        { headers: apiHeaders }
      );
      const updated = mapUploadResult(response.data);
      if (updated) {
        setKnowledgeFiles((prev) => mergeKnowledgeFiles(prev, [updated]));
      } else {
        void loadIndexedFiles();
      }
      setUploadStatus(
        `Reindexed ${file.filename}. Indexed ${Number(response.data?.chunks_indexed || 0)} chunk(s).`
      );
    } catch (error) {
      console.error('Reindex error:', error);
      setUploadStatus(getApiErrorMessage(error, `Failed to reindex ${file.filename}.`));
    } finally {
      setDocumentActionId(null);
    }
  };

  const handleUpdateDocumentVisibility = async (file: KnowledgeFile, visibility: string) => {
    if (!file.fileHash || documentActionId || visibility === file.visibility) return;

    const actionId = `permissions:${file.fileHash}`;
    setDocumentActionId(actionId);
    setUploadStatus(`Updating access for ${file.filename}...`);
    try {
      const response = await axios.patch<KnowledgeBaseFilesResponse>(
        `${API_BASE}/knowledge-base/files/${encodeURIComponent(file.fileHash)}/permissions`,
        { visibility, allowed_roles: file.allowedRoles || [] },
        { headers: apiHeaders }
      );
      const files = (response.data?.files || []).map(mapKnowledgeFile);
      setKnowledgeFiles(files);
      setUploadStatus(`Updated ${file.filename} access to ${visibility}.`);
    } catch (error) {
      console.error('Permission update error:', error);
      setUploadStatus(getApiErrorMessage(error, `Failed to update ${file.filename} access.`));
    } finally {
      setDocumentActionId(null);
    }
  };

  const openDocumentChunks = async (
    fileHash: string | null | undefined,
    focusChunkIndex?: number | null,
  ) => {
    if (!fileHash) return;
    setIsLoadingChunks(true);
    try {
      const query = typeof focusChunkIndex === 'number'
        ? `?focus_chunk_index=${focusChunkIndex}&neighbor_window=2`
        : '';
      const response = await axios.get<DocumentChunksResponse>(
        `${API_BASE}/knowledge-base/files/${encodeURIComponent(fileHash)}/chunks${query}`,
        { headers: apiHeaders }
      );
      setSelectedChunks(response.data);
    } catch (error) {
      console.error('Chunk load error:', error);
      setUploadStatus(getApiErrorMessage(error, 'Failed to load source chunks.'));
    } finally {
      setIsLoadingChunks(false);
    }
  };

  const openSourceChunks = async (source: Source) => {
    const fileHash = source.file_hash || knowledgeFiles.find((file) => file.filename === source.document)?.fileHash;
    await openDocumentChunks(fileHash, source.chunk_index);
  };

  const submitFeedback = async (
    assistantMessage: Message,
    rating: string,
    reason = '',
  ) => {
    const index = messages.findIndex((msg) => msg.id === assistantMessage.id);
    const previousUser = [...messages.slice(0, index)]
      .reverse()
      .find((msg) => msg.role === 'user');

    if (!previousUser) return;

    setFeedbackStatus('Saving feedback...');
    try {
      await axios.post(
        `${API_BASE}/feedback`,
        {
          question: previousUser.content,
          answer: assistantMessage.content,
          rating,
          reason,
          confidence_score: assistantMessage.confidence_score,
          sources: assistantMessage.sources || [],
          diagnostics: assistantMessage.diagnostics || undefined,
        },
        { headers: apiHeaders }
      );
      setFeedbackStatus('Feedback saved.');
      void loadAdminDebug();
    } catch (error) {
      console.error('Feedback error:', error);
      setFeedbackStatus(getApiErrorMessage(error, 'Failed to save feedback.'));
    }
  };

  const latestAssistant = [...messages]
    .reverse()
    .find((msg) => msg.role === 'assistant' && msg.sources && msg.sources.length > 0);

  const contextCards = latestAssistant?.sources?.slice(0, 4) || [];
  const latestDiagnostics = latestAssistant?.diagnostics || null;

  if (!authStatus) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <h1>RAGiT</h1>
          <p>Checking access...</p>
        </div>
      </div>
    );
  }

  if (!PUBLIC_DEMO_MODE && authStatus?.auth_enabled && !authUser && !API_KEY) {
    return (
      <div className="auth-shell">
        <form className="auth-card" onSubmit={handleAuthSubmit}>
          <h1>RAGiT</h1>
          <p>{authStatus.bootstrap_required ? 'Create the first admin account.' : 'Sign in to your workspace.'}</p>
          <label>
            Email
            <input
              type="email"
              value={authEmail}
              onChange={(event) => setAuthEmail(event.target.value)}
              autoComplete="email"
            />
          </label>
          {authStatus.bootstrap_required && (
            <label>
              Display name
              <input
                type="text"
                value={authDisplayName}
                onChange={(event) => setAuthDisplayName(event.target.value)}
                autoComplete="name"
              />
            </label>
          )}
          <label>
            Password
            <input
              type="password"
              value={authPassword}
              onChange={(event) => setAuthPassword(event.target.value)}
              autoComplete={authStatus.bootstrap_required ? 'new-password' : 'current-password'}
            />
          </label>
          {authError && <p className="auth-error">{authError}</p>}
          <button type="submit" disabled={isAuthenticating || !authEmail || !authPassword}>
            {isAuthenticating ? <Loader2 size={15} className="animate-spin" /> : null}
            {authStatus.bootstrap_required ? 'Create admin' : 'Sign in'}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="ui-shell">
      <DocumentSidebar
        fileInputRef={fileInputRef}
        user={authUser}
        isDemoMode={PUBLIC_DEMO_MODE}
        hasApiKey={Boolean(API_KEY)}
        chatSessions={chatSessions}
        activeSessionId={activeSessionId}
        knowledgeFiles={knowledgeFiles}
        isDragActive={isDragActive}
        isUploading={isUploading}
        isResetting={isResetting}
        uploadStatus={uploadStatus}
        uploadSteps={uploadSteps}
        documentActionId={documentActionId}
        isLoadingChunks={isLoadingChunks}
        adminOverview={adminOverview}
        modelHealth={modelHealth}
        isLoadingAdmin={isLoadingAdmin}
        auditEvents={auditEvents}
        evalRuns={evalRuns}
        isRunningEval={isRunningEval}
        formatIndexedAt={formatIndexedAt}
        onLogout={() => void handleLogout()}
        onStartNewChat={() => void startNewChat()}
        onLoadChatMessages={(session) => void loadChatMessages(session)}
        onDeleteChatSession={(session) => void deleteChatSession(session)}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={(event) => void handleDrop(event)}
        onFileUpload={(event) => void handleFileUpload(event)}
        onResetKnowledgeBase={() => void handleResetKnowledgeBase()}
        onOpenDocumentChunks={(fileHash) => void openDocumentChunks(fileHash)}
        onReindexKnowledgeFile={(file) => void handleReindexKnowledgeFile(file)}
        onDeleteKnowledgeFile={(file) => void handleDeleteKnowledgeFile(file)}
        onUpdateDocumentVisibility={(file, visibility) => void handleUpdateDocumentVisibility(file, visibility)}
        onLoadAdminDebug={() => void loadAdminDebug()}
        onStartEvalRun={() => void startEvalRun()}
      />
      <ChatWindow
        messages={messages}
        inputTitle={inputTitle}
        isQuerying={isQuerying}
        isDemoMode={PUBLIC_DEMO_MODE}
        queryInputRef={queryInputRef}
        chatContainerRef={chatContainerRef}
        chatEndRef={chatEndRef}
        onInputChange={setInputTitle}
        onSubmit={handleQuery}
        onStarterPrompt={handleStarterPrompt}
        onFeedback={(message, rating, reason) => void submitFeedback(message, rating, reason)}
        onAttach={() => fileInputRef.current?.click()}
        onScroll={handleChatScroll}
        onWheel={handleChatWheel}
        onTouchStart={handleChatTouchStart}
        onMouseDown={handleChatMouseDown}
      />
      <SourcePanel
        latestAssistant={latestAssistant}
        latestDiagnostics={latestDiagnostics}
        contextCards={contextCards}
        selectedChunks={selectedChunks}
        isLoadingChunks={isLoadingChunks}
        feedbackStatus={feedbackStatus}
        formatPercent={formatPercent}
        onOpenSourceChunks={(source) => void openSourceChunks(source)}
        onCloseChunks={() => setSelectedChunks(null)}
      />
    </div>
  );
}

export default AppShell;
