import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { UploadCloud, MessageSquare, Send, FileText, Loader2, Link2 } from 'lucide-react';
import './App.css';

// API Configuration
const API_BASE = 'http://localhost:8000';

interface Source {
  document: string;
  page: number;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputTitle, setInputTitle] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

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
      setUploadStatus(`Success! Indexed ${response.data.chunks_indexed} chunks.`);
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus("Failed to upload document. Is the backend running?");
    } finally {
      setIsUploading(false);
      // Reset input
      if (fileInputRef.current) fileInputRef.current.value = '';
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
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Query error:", error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "Sorry, there was an error processing your request. Please ensure the backend server is running."
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar: File Upload */}
      <aside className="sidebar glass">
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon"><MessageSquare size={24} /></div>
            <h2>Enterprise RAG</h2>
          </div>
          <p className="subtitle">Knowledge Assistant</p>
        </div>

        <div className="upload-section glass-panel">
          <h3>Knowledge Base</h3>
          <p>Upload internal PDFs to expand the assistant's knowledge.</p>
          
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
            disabled={isUploading}
          >
            {isUploading ? <Loader2 className="animate-spin" size={20} /> : <UploadCloud size={20} />}
            {isUploading ? 'Indexing Document...' : 'Upload PDF'}
          </button>

          {uploadStatus && (
            <div className={`status-message ${uploadStatus.includes('Failed') ? 'error' : 'success'} animate-slide-up`}>
              {uploadStatus}
            </div>
          )}
        </div>
        
        <div className="features-list">
          <div className="feature-item">
            <FileText size={18} />
            <span>Parses PDF text</span>
          </div>
          <div className="feature-item">
            <Link2 size={18} />
            <span>Provides cited sources</span>
          </div>
        </div>
      </aside>

      {/* Main Area: Chat Interface */}
      <main className="main-area">
        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="empty-state animate-slide-up">
              <MessageSquare size={48} className="empty-icon" />
              <h2>How can I help you today?</h2>
              <p>Ask a question about your uploaded documents.</p>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((msg) => (
                <div key={msg.id} className={`message-wrapper ${msg.role} animate-slide-up`}>
                  <div className="message-avatar">
                    {msg.role === 'assistant' ? <MessageSquare size={18} /> : 'U'}
                  </div>
                  <div className={`message-content glass-panel ${msg.role === 'assistant' ? 'markdown-body' : ''}`}>
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                    
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources-container">
                        <h4>Sources</h4>
                        <ul>
                          {msg.sources.map((src, i) => (
                            <li key={i}><FileText size={12} /> {src.document} (Page {src.page})</li>
                          ))}
                        </ul>
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
              type="text"
              placeholder="Ask anything about your documents..."
              value={inputTitle}
              onChange={(e) => setInputTitle(e.target.value)}
              disabled={isQuerying}
            />
            <button type="submit" className="primary" disabled={isQuerying || !inputTitle.trim()}>
              <Send size={18} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
