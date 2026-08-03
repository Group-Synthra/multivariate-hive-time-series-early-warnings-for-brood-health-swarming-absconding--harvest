import { LoaderCircle } from 'lucide-react';

export function LoadingState({ message = 'Loading common hive analytics…' }) {
  return (
    <div className="state-view" role="status" aria-live="polite">
      <LoaderCircle className="spin" size={40} />
      <h2>{message}</h2>
      <p>Retrieving aggregated results from the backend API.</p>
    </div>
  );
}
