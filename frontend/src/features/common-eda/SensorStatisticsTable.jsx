import { formatNumber } from '../../utils/formatters';

const ROWS = [
  ['Temperature', 'temperature', '°C'],
  ['Humidity', 'humidity', '%'],
  ['CO₂', 'co2', 'ppm'],
  ['Hive weight', 'weight', 'kg'],
];

export function SensorStatisticsTable({ statistics }) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Sensor</th>
            <th>Mean</th>
            <th>Std.</th>
            <th>Minimum</th>
            <th>Q1</th>
            <th>Median</th>
            <th>Q3</th>
            <th>Maximum</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(([label, key, unit]) => {
            const item = statistics?.[key] || {};
            return (
              <tr key={key}>
                <td><strong>{label}</strong> <small>{unit}</small></td>
                <td>{formatNumber(item.mean)}</td>
                <td>{formatNumber(item.std)}</td>
                <td>{formatNumber(item.min)}</td>
                <td>{formatNumber(item.q1)}</td>
                <td>{formatNumber(item.median)}</td>
                <td>{formatNumber(item.q3)}</td>
                <td>{formatNumber(item.max)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
