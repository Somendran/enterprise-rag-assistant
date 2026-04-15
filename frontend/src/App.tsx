import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  UploadCloud,
  Send,
  Loader2,
  Paperclip,
  RotateCcw,
  RefreshCw,
  Trash2,
  Eye,
  Heart,
  ThumbsDown,
  AlertTriangle,
  FileSearch,
} from 'lucide-react';
import './App.css';

// API Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';
const API_HEADERS: Record<string, string> = API_KEY ? { 'X-API-Key': API_KEY } : {};
const RESET_REQUEST_TIMEOUT_MS = 15000;
const STARTER_PROMPTS = [
  'Summarize leave policy changes this year',
  'What are the vendor onboarding steps?',
  'Where is the SLA escalation path documented?',
];

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
      return 'API key missing or invalid. Check VITE_API_KEY and backend APP_API_KEY.';
    }
    if (error.response?.data?.detail) {
      return String(error.response.data.detail);
    }
  }
  return fallback;
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

function App() {
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
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryInputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const lastScrollTopRef = useRef(0);

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

  const loadIndexedFiles = useCallback(async () => {
    try {
      const response = await axios.get<KnowledgeBaseFilesResponse>(
        `${API_BASE}/knowledge-base/files`,
        { headers: API_HEADERS }
      );
      const files = (response.data?.files || []).map(mapKnowledgeFile);
      setKnowledgeFiles(files);
    } catch (error) {
      // Keep UI usable even if this optional hydration call fails.
      console.warn('Failed to load indexed files on startup.', error);
    }
  }, []);

  useEffect(() => {
    void loadIndexedFiles();
  }, [loadIndexedFiles]);

  const loadChatSessions = useCallback(async () => {
    try {
      const response = await axios.get<{ sessions: ChatSession[] }>(
        `${API_BASE}/chat/sessions`,
        { headers: API_HEADERS }
      );
      setChatSessions(response.data?.sessions || []);
    } catch (error) {
      console.warn('Failed to load chat sessions.', error);
    }
  }, []);

  useEffect(() => {
    void loadChatSessions();
  }, [loadChatSessions]);

  const ensureChatSession = async (): Promise<string> => {
    if (activeSessionId) return activeSessionId;
    const response = await axios.post<ChatSession>(
      `${API_BASE}/chat/sessions`,
      { title: 'New chat' },
      { headers: API_HEADERS }
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
        { headers: API_HEADERS }
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
        { headers: API_HEADERS }
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
      { headers: API_HEADERS }
    );
    setActiveSessionId(response.data.id);
    setChatSessions((prev) => [response.data, ...prev]);
    setMessages([]);
  };

  const loadEvalRuns = useCallback(async () => {
    try {
      const response = await axios.get<{ runs: EvalRun[] }>(
        `${API_BASE}/evals/runs`,
        { headers: API_HEADERS }
      );
      setEvalRuns(response.data?.runs || []);
    } catch (error) {
      console.warn('Failed to load eval runs.', error);
    }
  }, []);

  const loadAdminDebug = useCallback(async () => {
    setIsLoadingAdmin(true);
    try {
      const [overviewResponse, healthResponse] = await Promise.all([
        axios.get<AdminOverview>(`${API_BASE}/admin/overview`, { headers: API_HEADERS }),
        axios.get<{ checks: ModelHealthItem[] }>(`${API_BASE}/health/models`, { headers: API_HEADERS }),
      ]);
      setAdminOverview(overviewResponse.data);
      setModelHealth(healthResponse.data?.checks || []);
      await loadEvalRuns();
    } catch (error) {
      console.warn('Failed to load admin/debug data.', error);
    } finally {
      setIsLoadingAdmin(false);
    }
  }, [loadEvalRuns]);

  const startEvalRun = async () => {
    setIsRunningEval(true);
    try {
      const response = await axios.post<{ run_id: string }>(
        `${API_BASE}/evals/runs`,
        undefined,
        { headers: API_HEADERS }
      );
      const runId = response.data.run_id;
      for (let i = 0; i < 120; i += 1) {
        const runResponse = await axios.get<EvalRun>(
          `${API_BASE}/evals/runs/${encodeURIComponent(runId)}`,
          { headers: API_HEADERS }
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
          ...API_HEADERS,
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
          { headers: API_HEADERS }
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

    const sessionId = await ensureChatSession();
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
    void saveChatMessage(sessionId, userMsg);

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...API_HEADERS },
        body: JSON.stringify({ question: userMsg.content }),
      });

      if (response.status === 401) {
        throw new Error('API key missing or invalid. Check VITE_API_KEY and backend APP_API_KEY.');
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

      if (finalAssistant) {
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
        { timeout: RESET_REQUEST_TIMEOUT_MS, headers: API_HEADERS }
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
        { headers: API_HEADERS }
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
        { headers: API_HEADERS }
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
        { headers: API_HEADERS }
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
        { headers: API_HEADERS }
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

  return (
    <div className="ui-shell">
      <aside className="kb-pane">
        <div className="brand-block">
          <h1>RAGiT</h1>
          <p>Grounded knowledge workspace</p>
        </div>

        <div className="sessions-panel">
          <div className="admin-header">
            <h2>Chats</h2>
            <button type="button" onClick={() => void startNewChat()}>New</button>
          </div>
          {chatSessions.length === 0 ? (
            <p className="empty-kb">No saved chats yet.</p>
          ) : (
            <div className="session-list">
              {chatSessions.slice(0, 6).map((session) => (
                <button
                  key={session.id}
                  type="button"
                  className={session.id === activeSessionId ? 'active' : ''}
                  onClick={() => void loadChatMessages(session)}
                >
                  {session.title || 'New chat'}
                </button>
              ))}
            </div>
          )}
        </div>

        <div
          className={`upload-dropzone ${isDragActive ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <h2>Knowledge Base</h2>
          <div className="drop-inner">
            <UploadCloud size={24} />
            <p>
              Drag and drop PDFs here,
              <br />
              or <button type="button" className="inline-link" onClick={() => fileInputRef.current?.click()}>browse</button>.
            </p>
          </div>

          <input 
            type="file" 
            accept=".pdf" 
            multiple
            ref={fileInputRef} 
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />

          {uploadStatus && (
            <div className={`status-note ${uploadStatus.includes('Failed') || uploadStatus.includes('API key') ? 'error' : 'success'}`}>
              {uploadStatus}
            </div>
          )}

          {uploadSteps.length > 0 && (
            <ol className="upload-steps">
              {uploadSteps.map((step, index) => (
                <li key={`${step}-${index}`}>{step}</li>
              ))}
            </ol>
          )}

          <button
            type="button"
            className="reset-kb-btn"
            onClick={handleResetKnowledgeBase}
            disabled={isResetting || isUploading}
          >
            {isResetting ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
            {isResetting ? 'Resetting...' : 'Reset Knowledge Base'}
          </button>
        </div>

        <div className="knowledge-list">
          {knowledgeFiles.length === 0 ? (
            <p className="empty-kb">No indexed files yet.</p>
          ) : (
            knowledgeFiles.map((file) => (
              <div key={file.fileHash || file.filename} className="kb-item">
                <div className="kb-item-main">
                  <strong>{file.filename}</strong>
                  <span>{file.chunks} chunks</span>
                </div>
                <div className="kb-meta">
                  <span>{file.parsingMethod}</span>
                  <span>{file.uploadStatus}</span>
                  {file.visionCallsUsed > 0 && <span>{file.visionCallsUsed} vision calls</span>}
                </div>
                <p className="kb-indexed-at">{formatIndexedAt(file.indexedAt)}</p>
                <div className="kb-actions">
                  <button
                    type="button"
                    onClick={() => openDocumentChunks(file.fileHash)}
                    disabled={isLoadingChunks}
                  >
                    <Eye size={13} />
                    Chunks
                  </button>
                  <button
                    type="button"
                    onClick={() => handleReindexKnowledgeFile(file)}
                    disabled={Boolean(documentActionId) || isUploading || isResetting}
                  >
                    {documentActionId === `reindex:${file.fileHash}` ? (
                      <Loader2 size={13} className="animate-spin" />
                    ) : (
                      <RefreshCw size={13} />
                    )}
                    Reindex
                  </button>
                  <button
                    type="button"
                    className="danger"
                    onClick={() => handleDeleteKnowledgeFile(file)}
                    disabled={Boolean(documentActionId) || isUploading || isResetting}
                  >
                    {documentActionId === `delete:${file.fileHash}` ? (
                      <Loader2 size={13} className="animate-spin" />
                    ) : (
                      <Trash2 size={13} />
                    )}
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="admin-panel">
          <div className="admin-header">
            <h2>Debug</h2>
            <button type="button" onClick={() => void loadAdminDebug()} disabled={isLoadingAdmin}>
              {isLoadingAdmin ? <Loader2 size={13} className="animate-spin" /> : <FileSearch size={13} />}
              Refresh
            </button>
          </div>
          {adminOverview ? (
            <div className="admin-stats">
              <span>{adminOverview.document_count} docs</span>
              <span>{adminOverview.chunk_count} chunks</span>
              <span>{adminOverview.feedback_count} feedback</span>
              <span>{adminOverview.chat_session_count} chats</span>
              <span>{adminOverview.eval_run_count} evals</span>
              <span>Docling {adminOverview.docling_enabled ? 'on' : 'off'}</span>
              <span>Reranker {adminOverview.reranker_enabled ? 'on' : 'off'}</span>
            </div>
          ) : (
            <p className="empty-kb">Refresh to inspect runtime status.</p>
          )}
          {modelHealth.length > 0 && (
            <div className="health-list">
              {modelHealth.map((item) => (
                <div key={item.name} className={`health-item ${item.status}`}>
                  <strong>{item.name}</strong>
                  <span>{item.status}</span>
                </div>
              ))}
            </div>
          )}
          <div className="eval-panel">
            <button type="button" onClick={() => void startEvalRun()} disabled={isRunningEval}>
              {isRunningEval ? <Loader2 size={13} className="animate-spin" /> : <FileSearch size={13} />}
              Run evals
            </button>
            {evalRuns.slice(0, 3).map((run) => (
              <div key={run.id} className="eval-run">
                <strong>{run.status}: {run.passed}/{run.total}</strong>
                <span>{run.message || `${run.failed} failed`}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <main className="workspace-pane">
        <header className="workspace-header">
          <div>
            <h1>Assistant Workspace</h1>
            <p>Ask questions and inspect confidence before acting.</p>
          </div>
        </header>

        <div
          className="chat-scroll"
          ref={chatContainerRef}
          onScroll={handleChatScroll}
          onWheel={handleChatWheel}
          onTouchStart={handleChatTouchStart}
          onMouseDown={handleChatMouseDown}
        >
          {messages.length === 0 ? (
            <div className="empty-workspace">
              <h2>Ask anything about your indexed workspace.</h2>
              <div className="starter-prompts">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="prompt-chip"
                    onClick={() => handleStarterPrompt(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="chat-list">
              {messages.map((msg) => (
                <div key={msg.id} className={`chat-row ${msg.role}`}>
                  <div className={`chat-bubble ${msg.role}`}>
                    {msg.role === 'assistant' ? <ReactMarkdown>{msg.content}</ReactMarkdown> : <p>{msg.content}</p>}
                    {msg.role === 'assistant' && msg.content && (
                      <div className="feedback-row">
                        <button type="button" onClick={() => void submitFeedback(msg, 'helpful')}>
                          <Heart size={13} />
                          Helpful
                        </button>
                        <button type="button" onClick={() => void submitFeedback(msg, 'not_helpful')}>
                          <ThumbsDown size={13} />
                          Not helpful
                        </button>
                        <button type="button" onClick={() => void submitFeedback(msg, 'wrong_source', 'wrong_source')}>
                          <AlertTriangle size={13} />
                          Wrong source
                        </button>
                        <button type="button" onClick={() => void submitFeedback(msg, 'missing_info', 'missing_info')}>
                          Missing info
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isQuerying && (
                <div className="chat-row assistant">
                  <div className="chat-bubble assistant loading-bubble">
                    <Loader2 className="animate-spin" size={16} /> Thinking...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <form onSubmit={handleQuery} className="composer">
            <input
              ref={queryInputRef}
              type="text"
              placeholder="Ask anything about your indexed documents..."
              value={inputTitle}
              onChange={(e) => setInputTitle(e.target.value)}
              disabled={isQuerying}
            />
            <button type="submit" className="send-btn" disabled={isQuerying || !inputTitle.trim()}>
              <Send size={15} />
              Send
            </button>
            <button type="button" className="attach-btn" onClick={() => fileInputRef.current?.click()}>
              <Paperclip size={15} />
            </button>
          </form>
        </div>
      </main>

      <aside className="context-pane">
        <h3>Retrieved Context</h3>
        {latestDiagnostics && (
          <div className="diagnostics-panel">
            <div>
              <span>Confidence</span>
              <strong>{formatPercent(latestAssistant?.confidence_score)}</strong>
            </div>
            <div>
              <span>Reranker</span>
              <strong>{latestDiagnostics.reranker_applied ? 'Applied' : latestDiagnostics.reranker_skipped_reason || 'Skipped'}</strong>
            </div>
            <div>
              <span>Candidates</span>
              <strong>{latestDiagnostics.candidates_considered}</strong>
            </div>
            <div>
              <span>Retrieval</span>
              <strong>{Math.round(latestDiagnostics.retrieval_ms || 0)} ms</strong>
            </div>
          </div>
        )}
        {contextCards.length === 0 ? (
          <p className="empty-context">Context cards appear here after a response with sources.</p>
        ) : (
          <div className="context-list">
            {contextCards.map((src, idx) => {
              return (
                <article key={`${src.document}-${src.page}-${idx}`} className="context-card">
                  <h4>{src.document.replace('.pdf', '')}, Page {src.page}</h4>
                  {src.section && <p className="context-section">{src.section}</p>}
                  <div className="score-grid">
                    <span>Final {formatPercent(src.final_score ?? src.relevance_score)}</span>
                    <span>Vector {formatPercent(src.vector_score)}</span>
                    <span>Lexical {formatPercent(src.lexical_score)}</span>
                    <span>BM25 {formatPercent(src.bm25_score)}</span>
                  </div>
                  <p className="context-snippet">
                    {src.snippet?.trim() || 'Snippet unavailable.'}
                  </p>
                  <button
                    type="button"
                    className="source-view-btn"
                    onClick={() => void openSourceChunks(src)}
                    disabled={isLoadingChunks}
                  >
                    <Eye size={13} />
                    Open source chunks
                  </button>
                </article>
              );
            })}
          </div>
        )}
        {feedbackStatus && <p className="feedback-status">{feedbackStatus}</p>}
        {selectedChunks && (
          <div className="chunk-viewer">
            <div className="chunk-viewer-header">
              <h4>{selectedChunks.filename}</h4>
              <button type="button" onClick={() => setSelectedChunks(null)}>Close</button>
            </div>
            {selectedChunks.chunks.length === 0 ? (
              <p className="empty-context">No chunks found for this document.</p>
            ) : (
              selectedChunks.chunks.map((chunk) => (
                <article
                  key={chunk.id}
                  className={`chunk-item ${selectedChunks.focus_chunk_index === chunk.chunk_index ? 'focused' : ''}`}
                >
                  <strong>Page {chunk.page || 'n/a'} · Chunk {chunk.chunk_index}</strong>
                  {chunk.section && <span>{chunk.section}</span>}
                  <p>{chunk.content}</p>
                </article>
              ))
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

export default App;
