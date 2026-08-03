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
import { ChartSelect } from './ChartSelect';
import { SENSOR_OPTIONS, sensorOption } from './chartConfig';

export function SensorDistributionChart({ data }) {
  const [sensor, setSensor] = useState('temperature');
  const selected = sensorOption(sensor);
  const rows = data?.[sensor] || [];

  if (!Object.keys(data || {}).length) {
    return <EmptyState message="Histogram data are not available from the API." />;
  }

  return (
    <div>
      <div className="chart-control-row">
        <ChartSelect label="Sensor" value={sensor} onChange={setSensor} options={SENSOR_OPTIONS} />
        <span className="chart-note">Distribution calculated from all cleaned records.</span>
      </div>
      <div className="chart-area">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 35 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="bin_center"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(value) => Number(value).toFixed(1)}
              label={{ value: `${selected.label} (${selected.unit})`, position: 'insideBottom', offset: -8 }}
            />
            <YAxis />
            <Tooltip
              labelFormatter={(_, payload) => {
                const row = payload?.[0]?.payload;
                return row ? `${row.bin_start} to ${row.bin_end} ${selected.unit}` : selected.label;
              }}
              formatter={(value) => [Number(value).toLocaleString(), 'Records']}
            />
            <Bar dataKey="count" name="Records" fill={selected.color} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
