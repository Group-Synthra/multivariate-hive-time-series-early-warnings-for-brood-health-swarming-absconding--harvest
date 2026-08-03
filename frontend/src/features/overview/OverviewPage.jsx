import { Activity, CalendarRange, Database, Timer } from 'lucide-react';
import { MODULES } from '../../config/modules';
import { Panel } from '../../components/common/Panel';
import { StatCard } from '../../components/common/StatCard';
import { formatDate } from '../../utils/formatters';

export function OverviewPage({ edaData, onOpenModule }) {
  const summary = edaData?.summary || {};

  return (
    <div className="page-stack">
      <section className="hero">
        <div>
          <span className="eyebrow">TEAM SHARED LAYER</span>
          <h2>Common hive-data preparation and analytics</h2>
        </div>
      </section>

      <div className="stats-grid">
        <StatCard label="Records" value={summary.total_records} icon={Database} note="Cleaned common observations" />
        <StatCard label="Hives" value={summary.total_hives} icon={Activity} note="Hive-wise chronological streams" />
        <StatCard label="Sampling" value={summary.sampling_frequency} icon={Timer} note="Historical training frequency" />
        <StatCard
          label="Coverage"
          value={`${formatDate(summary.analysis_start)} – ${formatDate(summary.analysis_end)}`}
          icon={CalendarRange}
          note="Available historical period"
        />
      </div>

      <Panel
        title="Module ownership"
      >
        <div className="module-grid">
          {MODULES.map((module) => (
            <button
              key={module.id}
              type="button"
              className="module-card"
              onClick={() => onOpenModule(module.id)}
            >
              <span>{module.owner}</span>
              <h3>{module.title}</h3>
              <p>{module.description}</p>
              <code>{module.target}</code>
            </button>
          ))}
        </div>
      </Panel>
    </div>
  );
}
