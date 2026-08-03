import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { EmptyState } from '../../components/common/EmptyState';
import { ChartSelect } from './ChartSelect';
import { SENSOR_OPTIONS, sensorOption } from './chartConfig';

export function SensorRelationshipChart({ data }) {
  const [xSensor, setXSensor] = useState('temperature');
  const [ySensor, setYSensor] = useState('humidity');
  const xOption = sensorOption(xSensor);
  const yOption = sensorOption(ySensor);

  const chartData = useMemo(
    () => (data || [])
      .map((item) => ({
        x: Number(item[xSensor]),
        y: Number(item[ySensor]),
        hive: item.hive,
        timestamp: item.timestamp,
      }))
      .filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y)),
    [data, xSensor, ySensor],
  );

  if (!data?.length) {
    return <EmptyState message="The sampled sensor relationship data are not available." />;
  }

  return (
    <div>
      <div className="chart-control-row chart-control-row-wrap">
        <ChartSelect label="X-axis" value={xSensor} onChange={setXSensor} options={SENSOR_OPTIONS} />
        <ChartSelect label="Y-axis" value={ySensor} onChange={setYSensor} options={SENSOR_OPTIONS} />
        <span className="chart-note">Deterministic sample used to keep the browser responsive.</span>
      </div>
      <div className="chart-area chart-area-large">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="x"
              name={xOption.label}
              unit={` ${xOption.unit}`}
              label={{ value: `${xOption.label} (${xOption.unit})`, position: 'insideBottom', offset: -12 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name={yOption.label}
              unit={` ${yOption.unit}`}
              label={{ value: `${yOption.label} (${yOption.unit})`, angle: -90, position: 'insideLeft' }}
            />
            <ZAxis range={[22, 22]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(value, name) => [`${Number(value).toFixed(3)}`, name]}
              labelFormatter={(_, payload) => {
                const row = payload?.[0]?.payload;
                return row ? `${row.hive} · ${row.timestamp}` : '';
              }}
            />
            <Scatter name={`${xOption.label} vs ${yOption.label}`} data={chartData} fill="#2563eb" fillOpacity={0.42} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
