import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { EmptyState } from '../../components/common/EmptyState';
import { SENSOR_OPTIONS } from './chartConfig';

export function OutlierAnalysisChart({ data }) {
  const chartData = SENSOR_OPTIONS.map((sensor) => ({
    sensor: sensor.label,
    percentage: Number(data?.[sensor.key]?.percentage || 0),
    count: Number(data?.[sensor.key]?.count || 0),
    lower: data?.[sensor.key]?.lower_bound,
    upper: data?.[sensor.key]?.upper_bound,
  }));

  if (!data || !Object.keys(data).length) {
    return <EmptyState message="Outlier analysis is not available from the API." />;
  }

  return (
    <div className="chart-area">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 12, left: 0, bottom: 25 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="sensor" angle={-10} textAnchor="end" height={45} />
          <YAxis unit="%" />
          <Tooltip
            formatter={(value, name, item) => [
              `${Number(value).toFixed(3)}% (${item.payload.count.toLocaleString()} records)`,
              'Outliers',
            ]}
            labelFormatter={(label, payload) => {
              const row = payload?.[0]?.payload;
              if (!row) return label;
              return `${label} · IQR limits ${row.lower ?? '—'} to ${row.upper ?? '—'}`;
            }}
          />
          <Bar dataKey="percentage" name="Outlier percentage" fill="#7c3aed" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
