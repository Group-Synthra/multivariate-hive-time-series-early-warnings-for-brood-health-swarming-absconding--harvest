import { EmptyState } from '../../components/common/EmptyState';
import { SENSOR_OPTIONS } from './chartConfig';

function cellBackground(value) {
  const number = Number(value || 0);
  const strength = Math.min(Math.abs(number), 1);
  if (number >= 0) return `rgba(37, 99, 235, ${0.10 + strength * 0.72})`;
  return `rgba(220, 38, 38, ${0.10 + strength * 0.72})`;
}

export function CorrelationHeatmap({ data }) {
  if (!data || !Object.keys(data).length) {
    return <EmptyState message="Correlation results are not available from the API." />;
  }

  return (
    <div className="heatmap-scroll">
      <div className="heatmap-grid" style={{ '--heatmap-size': SENSOR_OPTIONS.length }}>
        <div className="heatmap-corner" />
        {SENSOR_OPTIONS.map((sensor) => (
          <div className="heatmap-axis heatmap-axis-top" key={`top-${sensor.key}`}>{sensor.label}</div>
        ))}
        {SENSOR_OPTIONS.map((rowSensor) => (
          <div className="heatmap-row" key={rowSensor.key}>
            <div className="heatmap-axis">{rowSensor.label}</div>
            {SENSOR_OPTIONS.map((columnSensor) => {
              const value = Number(data?.[rowSensor.key]?.[columnSensor.key] ?? 0);
              return (
                <div
                  className="heatmap-cell"
                  key={`${rowSensor.key}-${columnSensor.key}`}
                  style={{ background: cellBackground(value) }}
                  title={`${rowSensor.label} vs ${columnSensor.label}: ${value.toFixed(3)}`}
                >
                  {value.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="heatmap-legend">
        <span className="negative">−1 negative</span>
        <span>0 no linear relationship</span>
        <span className="positive">+1 positive</span>
      </div>
    </div>
  );
}
