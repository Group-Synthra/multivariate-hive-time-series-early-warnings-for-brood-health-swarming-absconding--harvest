import { healthClass, numberValue } from '../utils/broodHealth';

export function HealthScoreGauge({ score, level, label, detail }) {
  const value = Math.max(0, Math.min(100, Number(score || 0)));
  const rotation = -90 + value * 1.8;
  return (
    <article className={`health-gauge-card ${healthClass(level)}`}>
      <div className="health-gauge-title">{label}</div>
      <div className="health-gauge" aria-label={`${label}: ${numberValue(value, 1)} out of 100`}>
        <div className="health-gauge-track" />
        <div className="health-gauge-needle" style={{ transform: `rotate(${rotation}deg)` }} />
        <div className="health-gauge-centre">
          <strong>{numberValue(value, 1)}</strong>
          <span>/ 100</span>
        </div>
      </div>
      <div className="health-gauge-level">{level || 'Unavailable'}</div>
      {detail && <p>{detail}</p>}
    </article>
  );
}
