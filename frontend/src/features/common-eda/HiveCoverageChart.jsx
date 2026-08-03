import { useState } from 'react';
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

export function HiveCoverageChart({ data }) {
  const [metric, setMetric] = useState('records');
  const chartData = (data || []).map((item) => ({
    hive: item.hive || item.hive_id,
    records: Number(item.records || item.row_count || item.count || 0),
    coverage: Number(item.coverage_percentage || 0),
    duration: Number(item.duration_days || 0),
    start: item.start,
    end: item.end,
  }));

  if (!chartData.length) {
    return <EmptyState message="Hive-coverage data are not available from the API." />;
  }

  const config = {
    records: { dataKey: 'records', label: 'Records', unit: '', color: '#0f766e' },
    coverage: { dataKey: 'coverage', label: 'Coverage', unit: '%', color: '#2563eb' },
    duration: { dataKey: 'duration', label: 'Observed duration', unit: ' days', color: '#7c3aed' },
  }[metric];

  return (
    <div>
      <div className="segmented-control" aria-label="Hive coverage metric">
        <button className={metric === 'records' ? 'active' : ''} onClick={() => setMetric('records')}>Records</button>
        <button className={metric === 'coverage' ? 'active' : ''} onClick={() => setMetric('coverage')}>Coverage %</button>
        <button className={metric === 'duration' ? 'active' : ''} onClick={() => setMetric('duration')}>Duration</button>
      </div>
      <div className="chart-area chart-area-large">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 8, left: 5, bottom: 45 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="hive" angle={-50} textAnchor="end" height={75} interval={0} />
            <YAxis unit={config.unit.trim()} domain={metric === 'coverage' ? [0, 100] : ['auto', 'auto']} />
            <Tooltip
              formatter={(value) => [`${Number(value).toLocaleString()}${config.unit}`, config.label]}
              labelFormatter={(label, payload) => {
                const row = payload?.[0]?.payload;
                return row ? `${label} · ${row.start?.slice(0, 10) || '—'} to ${row.end?.slice(0, 10) || '—'}` : label;
              }}
            />
            <Bar dataKey={config.dataKey} name={config.label} fill={config.color} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
