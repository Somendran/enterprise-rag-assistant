import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  UploadCloud,
  MessageSquare,
  Send,
  FileText,
  Loader2,
  ChevronDown,
  ChevronUp,
  Info,
  ShieldCheck,
  UserRound,
  Trash2,
  RotateCcw,
} from 'lucide-react';
import './App.css';

// API Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const STARTER_PROMPTS = [
  'Summarize leave policy changes this year',
  'What are the vendor onboarding steps?',
  'Where is the SLA escalation path documented?',
];

interface Source {
  document: string;
  page: number;
  relevance_score?: number | null;
}

interface RetrievalDiagnostics {
  query_variants_used: string[];
  is_broad_question: boolean;
  fallback_applied: boolean;
  candidates_considered: number;
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

function confidenceLabel(level?: Message['confidence_level']) {
  if (level === 'high') return 'High confidence';
  if (level === 'medium') return 'Medium confidence';
  if (level === 'low') return 'Low confidence';
  return 'Unknown confidence';
}

function confidenceAccent(level?: Message['confidence_level']) {
  if (level === 'high') return 'high';
  if (level === 'medium') return 'medium';
  if (level === 'low') return 'low';
  return 'unknown';
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputTitle, setInputTitle] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const uploadFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('Only PDF files are supported. Please upload a .pdf file.');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Uploading and indexing...');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      if (response.data?.message) {
        setUploadStatus(response.data.message);
      } else {
        setUploadStatus(`Success! Indexed ${response.data.chunks_indexed} chunks.`);
      }
    } catch (error) {
      console.error("Upload error:", error);
      if (axios.isAxiosError(error) && error.response?.data?.detail) {
        setUploadStatus(String(error.response.data.detail));
      } else {
        setUploadStatus("Failed to upload document. Is the backend running?");
      }
    } finally {
      setIsUploading(false);
      // Reset input
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await uploadFile(file);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!isUploading && !isResetting) {
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

    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    await uploadFile(file);
  };

  const handleStarterPrompt = (prompt: string) => {
    setInputTitle(prompt);
    queryInputRef.current?.focus();
  };

  const clearConversation = () => {
    setMessages([]);
    queryInputRef.current?.focus();
  };

  const handleResetKnowledgeBase = async () => {
    if (isResetting) return;

    const confirmed = window.confirm(
      'This will remove all uploaded PDFs and clear the vector index. Continue?'
    );
    if (!confirmed) return;

    setIsResetting(true);
    setUploadStatus('Resetting knowledge base...');

    try {
      const response = await axios.post(`${API_BASE}/knowledge-base/reset`);
      const deleted = Number(response.data?.uploads_deleted ?? 0);
      const indexCleared = Boolean(response.data?.index_cleared);
      setMessages([]);
      setUploadStatus(
        `Knowledge base reset complete. Deleted ${deleted} file(s). Index cleared: ${indexCleared ? 'yes' : 'no'}.`
      );
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.data?.detail) {
        setUploadStatus(String(error.response.data.detail));
      } else {
        setUploadStatus('Failed to reset knowledge base. Please try again.');
      }
    } finally {
      setIsResetting(false);
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputTitle.trim() || isQuerying) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: inputTitle };
    setMessages(prev => [...prev, userMsg]);
    setInputTitle('');
    setIsQuerying(true);

    try {
      const response = await axios.post(`${API_BASE}/query`, { question: userMsg.content });
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        confidence_score: response.data.confidence_score,
        confidence_level: response.data.confidence_level,
        diagnostics: response.data.diagnostics,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Query error:", error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: axios.isAxiosError(error) && error.response?.data?.detail
          ? String(error.response.data.detail)
          : "Sorry, there was an error processing your request. Please ensure the backend server is running."
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar glass">
        <div className="sidebar-header">
          <div className="logo-container">
            <h2>Enterprise RAG</h2>
          </div>
          <p className="subtitle">Grounded knowledge workspace</p>
        </div>

        <div
          className={`upload-section glass-panel ${isDragActive ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-title-row">
            <h3>Knowledge Base</h3>
          </div>
          <p>Upload company PDFs to expand searchable context.</p>
          <p className="drop-hint">Drag and drop a PDF here, or use the upload button.</p>
          
          <input 
            type="file" 
            accept=".pdf" 
            ref={fileInputRef} 
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          
          <button 
            className="upload-button" 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading || isResetting}
          >
            {isUploading ? <Loader2 className="animate-spin icon-primary" size={18} /> : <UploadCloud size={18} className="icon-primary" />}
            {isUploading ? 'Indexing Document...' : 'Upload PDF'}
          </button>

          <button
            type="button"
            className="reset-button"
            onClick={handleResetKnowledgeBase}
            disabled={isResetting || isUploading}
          >
            {isResetting ? <Loader2 className="animate-spin icon-muted" size={16} /> : <RotateCcw size={16} className="icon-muted" />}
            {isResetting ? 'Resetting...' : 'Reset Knowledge Base'}
          </button>

          {uploadStatus && (
            <div className={`status-message ${uploadStatus.includes('Failed') ? 'error' : 'success'} animate-slide-up`}>
              {uploadStatus}
            </div>
          )}
        </div>
        
      </aside>

      <main className="main-area">
        <header className="workspace-header glass-panel">
          <div className="workspace-title">
            <h1>Assistant Workspace</h1>
            <p>Ask questions and inspect confidence before acting.</p>
          </div>
          <div className="workspace-actions">
            <span className="message-count">{messages.length} messages</span>
            <button
              type="button"
              className="ghost-button"
              onClick={clearConversation}
              disabled={messages.length === 0 || isQuerying}
            >
              <Trash2 size={14} className="icon-muted" /> Clear
            </button>
          </div>
        </header>

        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="empty-state animate-slide-up">
              <MessageSquare size={48} className="empty-icon" />
              <h2>Ask your first question</h2>
              <p>Upload one or more PDFs, then ask for policies, summaries, or specific facts.</p>
              <div className="starter-prompts">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="starter-prompt"
                    onClick={() => handleStarterPrompt(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((msg) => (
                <div key={msg.id} className={`message-wrapper ${msg.role} animate-slide-up`}>
                  <div className="message-avatar">
                    {msg.role === 'assistant' ? <MessageSquare size={16} /> : <UserRound size={16} />}
                  </div>
                  <div className={`message-content glass-panel ${msg.role === 'assistant' ? 'markdown-body' : ''}`}>
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                    
                    {msg.role === 'assistant' && (
                      <div className="assistant-meta-stack">
                        <div className={`confidence-pill ${confidenceAccent(msg.confidence_level)}`}>
                          <ShieldCheck size={14} className="icon-confidence" />
                          <span>{confidenceLabel(msg.confidence_level)}</span>
                          {typeof msg.confidence_score === 'number' && (
                            <strong>{Math.round(msg.confidence_score * 100)}%</strong>
                          )}
                        </div>

                        {msg.diagnostics && (
                          <details className="diagnostics-panel">
                            <summary>
                              <Info size={14} className="icon-muted" />
                              Retrieval diagnostics
                              <ChevronDown size={14} className="summary-caret summary-caret-down" />
                              <ChevronUp size={14} className="summary-caret summary-caret-up" />
                            </summary>
                            <div className="diagnostics-grid">
                              <div>
                                <span>Question type</span>
                                <strong>{msg.diagnostics.is_broad_question ? 'Broad' : 'Specific'}</strong>
                              </div>
                              <div>
                                <span>Fallback</span>
                                <strong>{msg.diagnostics.fallback_applied ? 'Applied' : 'Not needed'}</strong>
                              </div>
                              <div>
                                <span>Candidates</span>
                                <strong>{msg.diagnostics.candidates_considered}</strong>
                              </div>
                              <div>
                                <span>Variants</span>
                                <strong>{msg.diagnostics.query_variants_used.length}</strong>
                              </div>
                            </div>
                            {msg.diagnostics.query_variants_used.length > 0 && (
                              <div className="variant-list">
                                {msg.diagnostics.query_variants_used.map((variant, index) => (
                                  <span key={`${variant}-${index}`} className="variant-chip">{variant}</span>
                                ))}
                              </div>
                            )}
                          </details>
                        )}

                        {msg.sources && msg.sources.length > 0 && (
                          <div className="sources-container">
                            <h4>Sources</h4>
                            <ul>
                              {msg.sources.map((src, i) => (
                                <li key={i}>
                                  <FileText size={12} className="icon-primary" />
                                  <span>{src.document} (Page {src.page})</span>
                                  {typeof src.relevance_score === 'number' && (
                                    <strong>{Math.round(src.relevance_score * 100)}%</strong>
                                  )}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isQuerying && (
                <div className="message-wrapper assistant animate-slide-up">
                  <div className="message-avatar"><MessageSquare size={18} /></div>
                  <div className="message-content glass-panel loading-dots">
                    <Loader2 className="animate-spin" size={20} /> Thinking...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        <div className="input-area glass">
          <form onSubmit={handleQuery} className="input-form">
            <input
              ref={queryInputRef}
              type="text"
              placeholder="Ask anything about your indexed documents..."
              value={inputTitle}
              onChange={(e) => setInputTitle(e.target.value)}
              disabled={isQuerying}
            />
            <button type="submit" className="primary" disabled={isQuerying || !inputTitle.trim()}>
              <Send size={16} className="icon-strong" />
            </button>
          </form>
          <div className="input-footer-row">
            <p className="input-helper">Answers include confidence and citations from indexed sources.</p>
            <span className="char-count">{inputTitle.length}/1000</span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
