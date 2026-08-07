import { ExternalLink, Image } from 'lucide-react';
import { broodHealthApi } from '../services/broodHealthApi';

export function BroodReportGallery({ images }) {
  if (!images?.length) return <div className="empty-state">Generated Python report figures are not available yet.</div>;
  return (
    <div className="brood-report-grid">
      {images.map((item) => {
        const url = broodHealthApi.reportUrl(item.filename);
        return (
          <figure className="brood-report-figure" key={item.filename}>
            <figcaption>
              <span><Image size={15} /> {item.title}</span>
              <a href={url} target="_blank" rel="noreferrer" aria-label={`Open ${item.title}`}><ExternalLink size={15} /></a>
            </figcaption>
            <img src={url} alt={item.title} loading="lazy" />
          </figure>
        );
      })}
    </div>
  );
}
