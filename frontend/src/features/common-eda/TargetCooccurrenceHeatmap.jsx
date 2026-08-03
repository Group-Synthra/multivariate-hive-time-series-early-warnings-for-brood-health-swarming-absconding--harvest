import { EmptyState } from '../../components/common/EmptyState';
import { TARGET_OPTIONS } from './chartConfig';

function background(value, max) {
  const strength = max > 0 ? Math.min(value / max, 1) : 0;
  return `rgba(124, 58, 237, ${0.08 + strength * 0.78})`;
}

export function TargetCooccurrenceHeatmap({ data }) {
  if (!data?.length) {
    return <EmptyState message="Target co-occurrence data are not available." />;
  }

  const lookup = new Map(data.map((item) => [`${item.row}:${item.column}`, Number(item.count || 0)]));
  const max = Math.max(...data.map((item) => Number(item.count || 0)), 0);

  return (
    <div className="cooccurrence-table-wrap">
      <table className="cooccurrence-table">
        <thead>
          <tr>
            <th />
            {TARGET_OPTIONS.map((target) => <th key={target.key}>{target.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {TARGET_OPTIONS.map((rowTarget) => (
            <tr key={rowTarget.key}>
              <th>{rowTarget.label}</th>
              {TARGET_OPTIONS.map((columnTarget) => {
                const count = lookup.get(`${rowTarget.key}:${columnTarget.key}`) || 0;
                return (
                  <td
                    key={columnTarget.key}
                    style={{ background: background(count, max) }}
                    title={`${rowTarget.label} and ${columnTarget.label}: ${count.toLocaleString()}`}
                  >
                    {count.toLocaleString()}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="chart-footnote">Diagonal cells show each target’s positive count. Off-diagonal cells show records where both labels are positive.</p>
    </div>
  );
}
