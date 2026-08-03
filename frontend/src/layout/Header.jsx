import { RefreshCw } from 'lucide-react';
import HivoraXLogo from '../assets/HivoraX.png';

export function Header({ onRefresh, refreshing }) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-logo-container">
          <img
            src={HivoraXLogo}
            alt="HivoraX logo"
            className="brand-logo"
          />
        </div>

        <div>
          <h1>Hive Analytics</h1>
          <p>Shared data foundation for four predictive modules</p>
        </div>
      </div>

      <div className="header-actions">
        <span className="connection-badge">
          <i />
          Common backend API
        </span>

        <button
          className="button button-secondary"
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
        >
          <RefreshCw
            className={refreshing ? 'spin' : ''}
            size={16}
          />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>
    </header>
  );
}