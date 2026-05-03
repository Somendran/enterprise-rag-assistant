import type { KnowledgeFile } from '../../types';

interface PermissionsModalProps {
  file: KnowledgeFile | null;
  onClose: () => void;
}

export function PermissionsModal({ file, onClose }: PermissionsModalProps) {
  if (!file) return null;
  return (
    <div className="chunk-viewer">
      <div className="chunk-viewer-header">
        <h4>{file.filename}</h4>
        <button type="button" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
