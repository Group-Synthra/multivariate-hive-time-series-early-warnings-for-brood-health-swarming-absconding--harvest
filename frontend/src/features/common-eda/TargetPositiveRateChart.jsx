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
import { TARGET_OPTIONS } from './chartConfig';

export function TargetPositiveRateChart({ data }) {
  const chartData = (data || []).map((item) => ({
    name: item.display_name || TARGET_OPTIONS.find((target) => target.key === item.target)?.label || item.target,
    rate: Math.max(Number(item.positive_percentage || 0), 0.000001),
    positive: Number(item.positive || 0),
    per10000: Number(item.positive_per_10000 || 0),
  }));

  if (!chartData.length) {
    return <EmptyState message="Target positive-rate results are not available." />;
  }

  return (
    <div className="chart-area chart-area-large">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" angle={-12} textAnchor="end" height={55} />
          <YAxis scale="log" domain={['auto', 'auto']} unit="%" />
          <Tooltip
            formatter={(value, _, item) => [
              `${Number(value).toFixed(8)}% · ${item.payload.positive.toLocaleString()} positives · ${item.payload.per10000.toFixed(4)} per 10,000`,
              'Positive rate',
            ]}
          />
          <Bar dataKey="rate" name="Positive rate" fill="#7c3aed" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
