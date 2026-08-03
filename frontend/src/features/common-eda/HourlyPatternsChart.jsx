import { TemporalPatternChart } from './TemporalPatternChart';

export function HourlyPatternsChart({ data }) {
  return (
    <TemporalPatternChart
      data={data}
      xKey="hour"
      xFormatter={(value) => `${value}:00`}
      initialSensor="temperature"
    />
  );
}
