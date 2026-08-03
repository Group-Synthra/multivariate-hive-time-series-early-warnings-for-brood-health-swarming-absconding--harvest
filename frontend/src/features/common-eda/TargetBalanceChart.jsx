import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { EmptyState } from '../../components/common/EmptyState';

const SHORT_NAMES = {
  brood_health_healthy_1: 'Brood healthy',
  swarming_happened_1: 'Swarming',
  absconding_happened_1: 'Absconding',
  honey_harvested_1: 'Harvesting',
};

export function TargetBalanceChart({ data }) {
  const chartData = (data || []).map((item) => ({
    name: item.display_name || SHORT_NAMES[item.target] || item.target,
    positive: Number(item.positive || 0),
    negative: Number(item.negative || 0),
    rate: Number(item.positive_percentage || 0),
  }));

  if (!chartData.length) {
    return <EmptyState message="Target-balance results are not available from the API." />;
  }

  return (
    <div className="chart-area chart-area-large">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 12, right: 10, left: 15, bottom: 35 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" angle={-12} textAnchor="end" height={55} />
          <YAxis tickFormatter={(value) => Number(value).toLocaleString()} />
          <Tooltip
            formatter={(value, name, item) => [
              Number(value).toLocaleString(),
              name === 'Positive records' ? `${name} (${item.payload.rate.toFixed(6)}%)` : name,
            ]}
          />
          <Legend />
          <Bar dataKey="negative" name="Negative records" stackId="records" fill="#cbd5e1" />
          <Bar dataKey="positive" name="Positive records" stackId="records" fill="#2563eb" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
