import { formatNumber } from '../../utils/formatters';

export function StatCard({ label, value, unit, icon: Icon, note }) {
  return (
    <article className="stat-card">
      <div className="stat-card-header">
        <span>{label}</span>
        {Icon && <Icon size={18} />}
      </div>
      <div className="stat-card-value">
        {typeof value === 'number' ? formatNumber(value) : value ?? '—'}
        {unit && <small>{unit}</small>}
      </div>
      {note && <p>{note}</p>}
    </article>
  );
}
