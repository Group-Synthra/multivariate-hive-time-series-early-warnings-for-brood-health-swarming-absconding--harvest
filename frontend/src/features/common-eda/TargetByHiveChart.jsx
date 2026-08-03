import { useMemo, useState } from 'react';
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
import { ChartSelect } from './ChartSelect';
import { TARGET_OPTIONS, targetOption } from './chartConfig';

export function TargetByHiveChart({ data }) {
  const [target, setTarget] = useState('swarming_happened_1');
  const selected = targetOption(target);
  const chartData = useMemo(
    () => (data || [])
      .map((item) => ({ hive: item.hive, count: Number(item[target] || 0) }))
      .sort((left, right) => right.count - left.count || String(left.hive).localeCompare(String(right.hive))),
    [data, target],
  );

  if (!data?.length) {
    return <EmptyState message="Hive-level target counts are not available from the API." />;
  }

  return (
    <div>
      <div className="chart-control-row">
        <ChartSelect label="Target" value={target} onChange={setTarget} options={TARGET_OPTIONS} />
        <span className="chart-note">Hives are ordered by positive-label count.</span>
      </div>
      <div className="chart-area chart-area-large">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 55 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="hive" angle={-45} textAnchor="end" height={75} interval={0} />
            <YAxis allowDecimals={false} />
            <Tooltip formatter={(value) => [Number(value).toLocaleString(), selected.label]} />
            <Bar dataKey="count" name={selected.label} fill="#0f766e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
