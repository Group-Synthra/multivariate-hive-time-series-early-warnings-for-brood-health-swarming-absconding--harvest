import { TemporalPatternChart } from './TemporalPatternChart';

export function WeekdayPatternsChart({ data }) {
  return (
    <TemporalPatternChart
      data={data}
      xKey="day"
      xFormatter={(value) => value}
      initialSensor="co2"
    />
  );
}
