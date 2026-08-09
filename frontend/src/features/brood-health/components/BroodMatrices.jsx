import { numberValue, percentValue } from '../utils/broodHealth';

export function TransitionMatrix({ data }) {
  const lookup = new Map((data || []).map((row) => [`${row.from}-${row.to}`, row]));
  const labels = ['Unhealthy', 'Healthy'];
  return (
    <div className="brood-matrix-wrap">
      <div className="brood-matrix brood-matrix-three">
        <div className="brood-matrix-corner">Current → next</div>
        {labels.map((label) => <div className="brood-matrix-axis" key={`top-${label}`}>{label}</div>)}
        {labels.map((from) => [
          <div className="brood-matrix-axis" key={`side-${from}`}>{from}</div>,
          ...labels.map((to) => {
            const row = lookup.get(`${from}-${to}`) || {};
            const intensity = Math.min(0.88, 0.12 + Number(row.probability || 0) / 125);
            return (
              <div className="brood-matrix-cell" key={`${from}-${to}`} style={{ background: `rgba(37, 99, 235, ${intensity})` }}>
                <strong>{percentValue(row.probability, 2)}</strong>
                <span>{numberValue(row.count, 0)} transitions</span>
              </div>
            );
          }),
        ])}
      </div>
    </div>
  );
}

export function ConfusionMatrix({ matrix, labels = ['Critical', 'Poor', 'Good', 'Excellent'] }) {
  const safeLabels = labels?.length ? labels : ['Critical', 'Poor', 'Good', 'Excellent'];
  const size = safeLabels.length;
  const values = matrix || Array.from({ length: size }, () => Array(size).fill(0));
  const maximum = Math.max(1, ...values.flat().map(Number));
  const rowTotals = values.map((row) => Math.max(1, row.reduce((sum, value) => sum + Number(value || 0), 0)));

  return (
    <div className="brood-matrix-wrap">
      <div
        className="brood-dynamic-matrix"
        style={{ gridTemplateColumns: `110px repeat(${size}, minmax(86px, 1fr))` }}
      >
        <div className="brood-matrix-corner">Actual → predicted</div>
        {safeLabels.map((label) => (
          <div className="brood-matrix-axis" key={`top-${label}`}>{label}</div>
        ))}

        {safeLabels.flatMap((actual, rowIndex) => [
          <div className="brood-matrix-axis" key={`side-${actual}`}>
            <strong>{actual}</strong>
            <small>100% of this row</small>
          </div>,
          ...safeLabels.map((predicted, columnIndex) => {
            const value = Number(values[rowIndex]?.[columnIndex] || 0);
            const percentage = value / rowTotals[rowIndex] * 100;
            const opacity = 0.10 + value / maximum * 0.78;
            return (
              <div
                className="brood-matrix-cell"
                key={`${actual}-${predicted}`}
                style={{ background: `rgba(15, 118, 110, ${opacity})` }}
                title={`${actual} predicted as ${predicted}: ${numberValue(value, 0)} cases (${percentage.toFixed(2)}%)`}
              >
                <strong>{percentValue(percentage, 2)}</strong>
                <span>{numberValue(value, 0)} cases</span>
                <small>{actual === predicted ? 'Correct level' : 'Different level'}</small>
              </div>
            );
          }),
        ])}
      </div>
    </div>
  );
}

export function CorrelationMatrix({ data }) {
  const variables = [...new Set((data || []).map((row) => row.row))];
  const lookup = new Map((data || []).map((row) => [`${row.row}-${row.column}`, Number(row.value || 0)]));
  const short = {
    temperature_c: 'Temp', humidity_pct: 'Humidity', co2_ppm: 'CO₂', weight_kg: 'Weight',
    condition_score: 'Condition', brood_health_healthy_1: 'Target',
  };
  if (!variables.length) return null;
  return (
    <div className="brood-correlation-scroll">
      <div className="brood-correlation-grid" style={{ gridTemplateColumns: `100px repeat(${variables.length}, minmax(78px, 1fr))` }}>
        <div />
        {variables.map((name) => <div className="brood-matrix-axis" key={`head-${name}`}>{short[name] || name}</div>)}
        {variables.flatMap((row) => [
          <div className="brood-matrix-axis" key={`row-${row}`}>{short[row] || row}</div>,
          ...variables.map((column) => {
            const value = lookup.get(`${row}-${column}`) || 0;
            const positive = value >= 0;
            const opacity = 0.10 + Math.abs(value) * 0.75;
            return (
              <div className="brood-correlation-cell" key={`${row}-${column}`} style={{ background: positive ? `rgba(37,99,235,${opacity})` : `rgba(220,38,38,${opacity})` }} title={`${row} vs ${column}: ${value.toFixed(3)}`}>
                {value.toFixed(2)}
              </div>
            );
          }),
        ])}
      </div>
    </div>
  );
}
