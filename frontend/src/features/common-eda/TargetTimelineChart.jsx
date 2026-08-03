import { useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { EmptyState } from '../../components/common/EmptyState';
import { ChartSelect } from './ChartSelect';
import { TARGET_OPTIONS, targetOption } from './chartConfig';

export function TargetTimelineChart({ data }) {
  const [target, setTarget] = useState('swarming_happened_1');
  const [metric, setMetric] = useState('count');
  const selected = targetOption(target);
  const dataKey = metric === 'rate' ? `${target}_per_10000` : target;

  if (!data?.length) {
    return <EmptyState message="Monthly target-event data are not available from the API." />;
  }

  return (
    <div>
      <div className="chart-control-row chart-control-row-wrap">
        <ChartSelect label="Target" value={target} onChange={setTarget} options={TARGET_OPTIONS} />
        <div className="segmented-control compact">
          <button className={metric === 'count' ? 'active' : ''} onClick={() => setMetric('count')}>Count</button>
          <button className={metric === 'rate' ? 'active' : ''} onClick={() => setMetric('rate')}>Per 10,000</button>
        </div>
      </div>
      <div className="chart-area chart-area-large">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 5, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" angle={-35} textAnchor="end" height={65} minTickGap={22} />
            <YAxis allowDecimals={metric === 'rate'} />
            <Tooltip
              formatter={(value) => [
                metric === 'rate' ? `${Number(value).toFixed(5)} per 10,000` : Number(value).toLocaleString(),
                selected.label,
              ]}
            />
            <Line type="monotone" dataKey={dataKey} name={selected.label} stroke="#7c3aed" strokeWidth={2.5} dot={{ r: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
