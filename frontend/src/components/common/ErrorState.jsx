import { CircleAlert, RefreshCw } from 'lucide-react';

export function ErrorState({ error, onRetry }) {
  return (
    <div className="state-view" role="alert">
      <CircleAlert size={44} />
      <h2>Backend connection failed</h2>
      <p>{error?.message || 'The common EDA service is not available.'}</p>
      <button className="button" type="button" onClick={onRetry}>
        <RefreshCw size={16} /> Retry
      </button>
    </div>
  );
}
