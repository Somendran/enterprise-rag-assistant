import type { ModelHealthItem } from '../../types';

interface ModelHealthProps {
  items: ModelHealthItem[];
}

export function ModelHealth({ items }: ModelHealthProps) {
  if (items.length === 0) return null;
  return (
    <div className="health-list">
      {items.map((item) => (
        <div key={item.name} className={`health-item ${item.status}`}>
          <strong>{item.name}</strong>
          <span>{item.status}</span>
        </div>
      ))}
    </div>
  );
}
