import { TemporalPatternChart } from './TemporalPatternChart';

export function MonthlyPatternsChart({ data }) {
  return (
    <TemporalPatternChart
      data={data}
      xKey="month"
      xFormatter={(value) => value}
      initialSensor="weight"
    />
  );
}
