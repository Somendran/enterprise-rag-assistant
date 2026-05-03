import { FileSearch, Loader2 } from 'lucide-react';
import type { EvalRun } from '../../types';

interface EvalDashboardProps {
  runs: EvalRun[];
  isRunning: boolean;
  onRun: () => void;
}

export function EvalDashboard({ runs, isRunning, onRun }: EvalDashboardProps) {
  return (
    <div className="eval-panel">
      <button type="button" onClick={onRun} disabled={isRunning}>
        {isRunning ? <Loader2 size={13} className="animate-spin" /> : <FileSearch size={13} />}
        Run evals
      </button>
      {runs.slice(0, 3).map((run) => (
        <div key={run.id} className="eval-run">
          <strong>{run.status}: {run.passed}/{run.total}</strong>
          <span>{run.message || `${run.failed} failed`}</span>
        </div>
      ))}
    </div>
  );
}
