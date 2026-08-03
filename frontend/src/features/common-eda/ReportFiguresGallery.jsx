import { ExternalLink, Image as ImageIcon } from 'lucide-react';
import { apiAssetUrl } from '../../services/apiClient';
import { EmptyState } from '../../components/common/EmptyState';

const FIGURES = [
  ['target_balance', 'Positive labels by target'],
  ['sensor_distributions', 'Sensor-value distributions'],
  ['rows_per_hive', 'Records per hive'],
  ['correlation_heatmap', 'Sensor correlation matrix'],
  ['hourly_sensor_patterns', 'Hourly sensor patterns'],
  ['monthly_sensor_trends', 'Monthly sensor trends'],
  ['outlier_percentages', 'IQR outlier percentages'],
  ['target_positive_rates_log', 'Positive-label rates — logarithmic scale'],
  ['monthly_event_timeline', 'Monthly target-event timeline'],
  ['target_cooccurrence', 'Target-label co-occurrence'],
];

export function ReportFiguresGallery({ images }) {
  const available = FIGURES
    .map(([key, title]) => ({ key, title, path: images?.[key] }))
    .filter((item) => item.path);

  if (!available.length) {
    return (
      <EmptyState message="No generated report figures were found. Rerun the common pipeline after copying the enhanced backend EDA file." />
    );
  }

  return (
    <div className="report-figure-grid">
      {available.map((item) => {
        const src = apiAssetUrl(item.path);
        return (
          <figure className="report-figure" key={item.key}>
            <div className="report-figure-toolbar">
              <span><ImageIcon size={16} /> {item.title}</span>
              <a href={src} target="_blank" rel="noreferrer" aria-label={`Open ${item.title}`}>
                <ExternalLink size={16} />
              </a>
            </div>
            <img src={src} alt={item.title} loading="lazy" />
          </figure>
        );
      })}
    </div>
  );
}
