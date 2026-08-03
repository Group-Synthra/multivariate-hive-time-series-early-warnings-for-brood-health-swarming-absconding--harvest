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

const LABELS = {
  temperature: 'Temperature',
  humidity: 'Humidity',
  co2: 'CO₂',
  weight: 'Weight',
  brood_health_healthy_1: 'Brood label',
  swarming_happened_1: 'Swarming label',
  absconding_happened_1: 'Absconding label',
  honey_harvested_1: 'Harvest label',
};

export function DataQualityChart({ data }) {
  const chartData = Object.entries(data?.missing_by_column || {}).map(([column, count]) => ({
    column: LABELS[column] || column,
    missing: Number(count || 0),
  }));

  if (!chartData.length) {
    return <EmptyState message="Column-level missing-value data are not available." />;
  }

  return (
    <div className="chart-area">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 55 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="column" angle={-35} textAnchor="end" height={75} interval={0} />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="missing" name="Missing records" fill="#dc2626" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
