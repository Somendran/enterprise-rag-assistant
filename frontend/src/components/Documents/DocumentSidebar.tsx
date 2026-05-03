import type { ChangeEvent, DragEvent, RefObject } from 'react';
import { FileSearch, Loader2, Trash2 } from 'lucide-react';
import { AuditLog } from '../Admin/AuditLog';
import { EvalDashboard } from '../Admin/EvalDashboard';
import { ModelHealth } from '../Admin/ModelHealth';
import { DocumentRow } from './DocumentRow';
import { UploadZone } from './UploadZone';
import type {
  AdminOverview,
  AuditEvent,
  AuthUser,
  ChatSession,
  EvalRun,
  KnowledgeFile,
  ModelHealthItem,
} from '../../types';

interface DocumentSidebarProps {
  fileInputRef: RefObject<HTMLInputElement | null>;
  user: AuthUser | null;
  isDemoMode: boolean;
  hasApiKey: boolean;
  chatSessions: ChatSession[];
  activeSessionId: string | null;
  knowledgeFiles: KnowledgeFile[];
  isDragActive: boolean;
  isUploading: boolean;
  isResetting: boolean;
  uploadStatus: string | null;
  uploadSteps: string[];
  documentActionId: string | null;
  isLoadingChunks: boolean;
  adminOverview: AdminOverview | null;
  modelHealth: ModelHealthItem[];
  isLoadingAdmin: boolean;
  auditEvents: AuditEvent[];
  evalRuns: EvalRun[];
  isRunningEval: boolean;
  formatIndexedAt: (value: number) => string;
  onLogout: () => void;
  onStartNewChat: () => void;
  onLoadChatMessages: (session: ChatSession) => void;
  onDeleteChatSession: (session: ChatSession) => void;
  onDragOver: (event: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (event: DragEvent<HTMLDivElement>) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onFileUpload: (event: ChangeEvent<HTMLInputElement>) => void;
  onResetKnowledgeBase: () => void;
  onOpenDocumentChunks: (fileHash: string) => void;
  onReindexKnowledgeFile: (file: KnowledgeFile) => void;
  onDeleteKnowledgeFile: (file: KnowledgeFile) => void;
  onUpdateDocumentVisibility: (file: KnowledgeFile, visibility: string) => void;
  onLoadAdminDebug: () => void;
  onStartEvalRun: () => void;
}

export function DocumentSidebar({
  fileInputRef,
  user,
  isDemoMode,
  hasApiKey,
  chatSessions,
  activeSessionId,
  knowledgeFiles,
  isDragActive,
  isUploading,
  isResetting,
  uploadStatus,
  uploadSteps,
  documentActionId,
  isLoadingChunks,
  adminOverview,
  modelHealth,
  isLoadingAdmin,
  auditEvents,
  evalRuns,
  isRunningEval,
  formatIndexedAt,
  onLogout,
  onStartNewChat,
  onLoadChatMessages,
  onDeleteChatSession,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileUpload,
  onResetKnowledgeBase,
  onOpenDocumentChunks,
  onReindexKnowledgeFile,
  onDeleteKnowledgeFile,
  onUpdateDocumentVisibility,
  onLoadAdminDebug,
  onStartEvalRun,
}: DocumentSidebarProps) {
  return (
    <aside className="kb-pane">
      <div className="brand-block">
        <h1>RAGiT</h1>
        <p>Grounded knowledge workspace</p>
        {user && !isDemoMode && (
          <div className="user-chip">
            <span>{user.display_name || user.email}</span>
            <small>{user.role}</small>
            {!hasApiKey && <button type="button" onClick={onLogout}>Logout</button>}
          </div>
        )}
      </div>

      {!isDemoMode && (
        <div className="sessions-panel">
          <div className="admin-header">
            <h2>Chats</h2>
            <button type="button" onClick={onStartNewChat}>New</button>
          </div>
          {chatSessions.length === 0 ? (
            <p className="empty-kb">No saved chats yet.</p>
          ) : (
            <div className="session-list">
              {chatSessions.slice(0, 6).map((session) => (
                <div key={session.id} className="session-list-item">
                  <button
                    type="button"
                    className={session.id === activeSessionId ? 'active' : ''}
                    onClick={() => onLoadChatMessages(session)}
                  >
                    {session.title || 'New chat'}
                  </button>
                  <button
                    type="button"
                    className="session-delete-btn"
                    aria-label={`Delete chat ${session.title || 'New chat'}`}
                    onClick={() => onDeleteChatSession(session)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <UploadZone
        fileInputRef={fileInputRef}
        isDragActive={isDragActive}
        isUploading={isUploading}
        isResetting={isResetting}
        uploadStatus={uploadStatus}
        uploadSteps={uploadSteps}
        isDemoMode={isDemoMode}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onFileUpload={onFileUpload}
        onReset={onResetKnowledgeBase}
      />

      <div className="knowledge-list">
        {knowledgeFiles.length === 0 ? (
          <p className="empty-kb">No indexed files yet.</p>
        ) : (
          knowledgeFiles.map((file) => (
            <DocumentRow
              key={file.fileHash || file.filename}
              file={file}
              actionId={documentActionId}
              disabled={Boolean(documentActionId) || isUploading || isResetting}
              isDemoMode={isDemoMode}
              isLoadingChunks={isLoadingChunks}
              formatIndexedAt={formatIndexedAt}
              onOpenChunks={onOpenDocumentChunks}
              onReindex={onReindexKnowledgeFile}
              onDelete={onDeleteKnowledgeFile}
              onVisibilityChange={onUpdateDocumentVisibility}
            />
          ))
        )}
      </div>

      {!isDemoMode && (
        <div className="admin-panel">
          <div className="admin-header">
            <h2>Debug</h2>
            <button type="button" onClick={onLoadAdminDebug} disabled={isLoadingAdmin}>
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
              <span>{adminOverview.user_count} users</span>
              <span>{adminOverview.audit_event_count} audit events</span>
              <span>Docling {adminOverview.docling_enabled ? 'on' : 'off'}</span>
              <span>Reranker {adminOverview.reranker_enabled ? 'on' : 'off'}</span>
            </div>
          ) : (
            <p className="empty-kb">Refresh to inspect runtime status.</p>
          )}
          <ModelHealth items={modelHealth} />
          <AuditLog events={auditEvents} />
          <EvalDashboard runs={evalRuns} isRunning={isRunningEval} onRun={onStartEvalRun} />
        </div>
      )}
    </aside>
  );
}
