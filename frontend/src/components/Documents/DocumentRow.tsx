import { Eye, Loader2, RefreshCw, Trash2 } from 'lucide-react';
import type { KnowledgeFile } from '../../types';

interface DocumentRowProps {
  file: KnowledgeFile;
  actionId: string | null;
  disabled: boolean;
  isDemoMode: boolean;
  isLoadingChunks: boolean;
  formatIndexedAt: (value: number) => string;
  onOpenChunks: (fileHash: string) => void;
  onReindex: (file: KnowledgeFile) => void;
  onDelete: (file: KnowledgeFile) => void;
  onVisibilityChange: (file: KnowledgeFile, visibility: string) => void;
}

export function DocumentRow({
  file,
  actionId,
  disabled,
  isDemoMode,
  isLoadingChunks,
  formatIndexedAt,
  onOpenChunks,
  onReindex,
  onDelete,
  onVisibilityChange,
}: DocumentRowProps) {
  return (
    <div className="kb-item">
      <div className="kb-item-main">
        <strong>{file.filename}</strong>
        <span>{file.chunks} chunks</span>
      </div>
      <div className="kb-meta">
        <span>{file.parsingMethod}</span>
        <span>{file.uploadStatus}</span>
        <span>{file.visibility}</span>
        {file.visionCallsUsed > 0 && <span>{file.visionCallsUsed} vision calls</span>}
      </div>
      <p className="kb-indexed-at">{formatIndexedAt(file.indexedAt)}</p>
      <div className="kb-actions">
        <button type="button" onClick={() => onOpenChunks(file.fileHash)} disabled={isLoadingChunks}>
          <Eye size={13} />
          Chunks
        </button>
        {!isDemoMode && (
          <>
            <button type="button" onClick={() => onReindex(file)} disabled={disabled}>
              {actionId === `reindex:${file.fileHash}` ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <RefreshCw size={13} />
              )}
              Reindex
            </button>
            <select
              value={file.visibility}
              onChange={(event) => onVisibilityChange(file, event.target.value)}
              disabled={disabled}
              aria-label={`Access for ${file.filename}`}
            >
              <option value="shared">Shared</option>
              <option value="private">Private</option>
            </select>
            <button type="button" className="danger" onClick={() => onDelete(file)} disabled={disabled}>
              {actionId === `delete:${file.fileHash}` ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Trash2 size={13} />
              )}
              Delete
            </button>
          </>
        )}
      </div>
    </div>
  );
}
