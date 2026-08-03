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
import { SENSOR_OPTIONS, sensorOption } from './chartConfig';

export function TemporalPatternChart({ data, xKey, xFormatter, initialSensor = 'temperature' }) {
  const [sensor, setSensor] = useState(initialSensor);
  const selected = sensorOption(sensor);

  if (!data?.length) {
    return <EmptyState message="Temporal pattern data are not available from the API." />;
  }

  return (
    <div>
      <div className="chart-control-row">
        <ChartSelect label="Sensor" value={sensor} onChange={setSensor} options={SENSOR_OPTIONS} />
        <span className="chart-note">Showing one sensor at a time avoids misleading mixed scales.</span>
      </div>
      <div className="chart-area chart-area-large">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 25, left: 5, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey={xKey}
              tickFormatter={xFormatter}
              angle={data.length > 14 ? -35 : 0}
              textAnchor={data.length > 14 ? 'end' : 'middle'}
              height={data.length > 14 ? 65 : 35}
              minTickGap={18}
            />
            <YAxis unit={` ${selected.unit}`} />
            <Tooltip
              labelFormatter={(value) => xFormatter?.(value) ?? value}
              formatter={(value) => [`${Number(value).toFixed(3)} ${selected.unit}`, selected.label]}
            />
            <Line
              type="monotone"
              dataKey={sensor}
              name={selected.label}
              stroke={selected.color}
              strokeWidth={2.5}
              dot={data.length <= 12}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
