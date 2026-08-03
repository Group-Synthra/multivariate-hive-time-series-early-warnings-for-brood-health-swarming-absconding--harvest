import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { EmptyState } from '../../components/common/EmptyState';
import { ChartSelect } from './ChartSelect';
import { SENSOR_OPTIONS, TARGET_OPTIONS, targetOption } from './chartConfig';

export function TargetSensorEffectChart({ data }) {
  const [target, setTarget] = useState('swarming_happened_1');
  const selected = targetOption(target);
  const targetData = data?.[target];
  const chartData = (targetData?.effects || []).map((item) => ({
    sensor: SENSOR_OPTIONS.find((sensor) => sensor.key === item.sensor)?.label || item.sensor,
    difference: Number(item.standardized_difference || 0),
    positiveMean: item.positive_mean,
    negativeMean: item.negative_mean,
  }));

  if (!data || !Object.keys(data).length) {
    return <EmptyState message="Target-conditioned sensor analysis is not available." />;
  }

  const evidence = Number(targetData?.positive_count || 0);

  return (
    <div>
      <div className="chart-control-row chart-control-row-wrap">
        <ChartSelect label="Target" value={target} onChange={setTarget} options={TARGET_OPTIONS} />
        <span className={`evidence-badge ${evidence < 30 ? 'warning' : ''}`}>
          {evidence.toLocaleString()} positive records
          {evidence < 30 ? ' · interpret cautiously' : ''}
        </span>
      </div>
      <div className="chart-area chart-area-large">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 15, left: 5, bottom: 25 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="sensor" />
            <YAxis label={{ value: 'Standardized mean difference', angle: -90, position: 'insideLeft' }} />
            <ReferenceLine y={0} stroke="#475569" />
            <Tooltip
              formatter={(value, _, item) => [
                `${Number(value).toFixed(4)} SD · positive mean ${item.payload.positiveMean ?? '—'} · negative mean ${item.payload.negativeMean ?? '—'}`,
                selected.label,
              ]}
            />
            <Bar dataKey="difference" name="Standardized difference" fill="#2563eb" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-footnote">
        Positive values mean the sensor average was higher during positive-label records; negative values mean it was lower.
      </p>
    </div>
  );
}
