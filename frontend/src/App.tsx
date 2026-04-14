import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  UploadCloud,
  Send,
  Loader2,
  Paperclip,
  RotateCcw,
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
  document: string;
  page: number;
  relevance_score?: number | null;
  snippet?: string | null;
}

interface RetrievalDiagnostics {
  query_variants_used: string[];
  is_broad_question: boolean;
  fallback_applied: boolean;
  candidates_considered: number;
}

interface UploadItemResult {
  filename: string;
  chunks_indexed: number;
  status: 'success' | 'duplicate' | 'failed' | string;
  message: string;
}

interface UploadBatchResponse {
  files?: UploadItemResult[];
  total_files?: number;
  processed_files?: number;
  total_chunks_indexed?: number;
}

interface KnowledgeBaseFilesResponse {
  files?: Array<{
    filename: string;
    chunk_count: number;
    indexed_at: number;
  }>;
}

interface KnowledgeFile {
  filename: string;
  chunks: number;
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
  const byName = new Map<string, KnowledgeFile>();
  for (const item of current) {
    byName.set(item.filename, item);
  }
  for (const item of incoming) {
    const existing = byName.get(item.filename);
    if (!existing) {
      byName.set(item.filename, item);
      continue;
    }
    byName.set(item.filename, {
      filename: item.filename,
      chunks: Math.max(existing.chunks, item.chunks),
    });
  }
  return [...byName.values()].sort((a, b) => a.filename.localeCompare(b.filename));
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

  useEffect(() => {
    const loadIndexedFiles = async () => {
      try {
        const response = await axios.get<KnowledgeBaseFilesResponse>(
          `${API_BASE}/knowledge-base/files`,
          { headers: API_HEADERS }
        );
        const files = (response.data?.files || []).map((item) => ({
          filename: item.filename,
          chunks: Number(item.chunk_count || 0),
        }));
        setKnowledgeFiles(files);
      } catch (error) {
        // Keep UI usable even if this optional hydration call fails.
        console.warn('Failed to load indexed files on startup.', error);
      }
    };

    void loadIndexedFiles();
  }, []);

  const uploadFiles = async (files: File[]) => {
    if (!files.length) return;

    const pdfFiles = files.filter((file) => file.name.toLowerCase().endsWith('.pdf'));
    if (!pdfFiles.length) {
      setUploadStatus('Only PDF files are supported. Please upload .pdf files.');
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Indexing ${pdfFiles.length} file(s)...`);
    
    const formData = new FormData();
    pdfFiles.forEach((file) => formData.append('files', file));

    try {
      const response = await axios.post<UploadBatchResponse>(`${API_BASE}/upload`, formData, {
        headers: {
          ...API_HEADERS,
          'Content-Type': 'multipart/form-data',
        },
      });
      const data = response.data;
      const totalFiles = Number(data?.total_files ?? pdfFiles.length);
      const processedFiles = Number(data?.processed_files ?? 0);
      const totalChunks = Number(data?.total_chunks_indexed ?? 0);
      setUploadStatus(
        `Upload completed. Processed ${processedFiles}/${totalFiles} files. Indexed ${totalChunks} chunks.`
      );

      const incoming: KnowledgeFile[] = (data?.files || [])
        .filter((item) => item.status === 'success' || item.status === 'duplicate')
        .map((item) => ({
          filename: item.filename,
          chunks: Number(item.chunks_indexed || 0),
        }));
      if (incoming.length) {
        setKnowledgeFiles((prev) => mergeKnowledgeFiles(prev, incoming));
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus(getApiErrorMessage(error, "Failed to upload files. Is the backend running?"));
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

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: inputTitle };
    const assistantId = (Date.now() + 1).toString();
    shouldAutoScrollRef.current = true;
    setMessages(prev => [
      ...prev,
      userMsg,
      { id: assistantId, role: 'assistant', content: '' },
    ]);
    setInputTitle('');
    setIsQuerying(true);

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
        setMessages(prev => prev.map(msg => (
          msg.id === assistantId
            ? { ...msg, content: `${msg.content}${text}` }
            : msg
        )));
      };

      const applyDone = (payload: StreamDonePayload) => {
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

  const latestAssistant = [...messages]
    .reverse()
    .find((msg) => msg.role === 'assistant' && msg.sources && msg.sources.length > 0);

  const contextCards = latestAssistant?.sources?.slice(0, 4) || [];

  return (
    <div className="ui-shell">
      <aside className="kb-pane">
        <div className="brand-block">
          <h1>RAGiT</h1>
          <p>Grounded knowledge workspace</p>
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
              <div key={file.filename} className="kb-item">
                <strong>{file.filename}</strong>
                <span>({file.chunks} chunks)</span>
              </div>
            ))
          )}
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
        {contextCards.length === 0 ? (
          <p className="empty-context">Context cards appear here after a response with sources.</p>
        ) : (
          <div className="context-list">
            {contextCards.map((src, idx) => {
              const score = typeof src.relevance_score === 'number' ? Math.round(src.relevance_score * 100) : null;
              return (
                <article key={`${src.document}-${src.page}-${idx}`} className="context-card">
                  <h4>{src.document.replace('.pdf', '')}, Page {src.page}</h4>
                  {score !== null && <p className="context-confidence">(Confidence: {score}%)</p>}
                  <p className="context-snippet">
                    {src.snippet?.trim() || 'Snippet unavailable.'}
                  </p>
                </article>
              );
            })}
          </div>
        )}
      </aside>
    </div>
  );
}

export default App;
