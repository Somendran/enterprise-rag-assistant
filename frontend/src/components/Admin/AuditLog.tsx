import type { AuditEvent } from '../../types';

interface AuditLogProps {
  events: AuditEvent[];
}

export function AuditLog({ events }: AuditLogProps) {
  if (events.length === 0) return null;
  return (
    <div className="audit-list">
      {events.slice(0, 5).map((event) => (
        <div key={event.id} className="audit-item">
          <strong>{event.action}</strong>
          <span>{event.actor_email || 'system'} · {new Date(event.created_at * 1000).toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}
