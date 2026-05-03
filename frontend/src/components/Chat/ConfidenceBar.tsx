interface ConfidenceBarProps {
  score?: number | null;
  formatPercent: (value?: number | null) => string;
}

export function ConfidenceBar({ score, formatPercent }: ConfidenceBarProps) {
  return <strong>{formatPercent(score)}</strong>;
}
