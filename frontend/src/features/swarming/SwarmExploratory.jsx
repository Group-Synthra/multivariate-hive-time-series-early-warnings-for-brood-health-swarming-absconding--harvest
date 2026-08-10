import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  FileText,
  Maximize2,
  RefreshCw,
  Scale,
  X,
  XCircle,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const PALETTE = {
  navy: '#263b50',
  slate: '#657786',
  pale: '#d8dde3',
  burgundy: '#7a3343',
  charcoal: '#30363d',
  paper: '#ffffff',
  wash: '#f5f7f8',
};

const CHART_TITLES = {
  feature_distribution: 'Distribution of Swarming Sensor Variables',
  swarm_indicators: 'Swarming Outcome Prevalence and Hive Heterogeneity',
  correlation_matrix: 'Pearson Correlation Matrix',
  top_correlations: 'Sensor–Swarming Association Statistics',
  temporal_patterns: 'Mean Diurnal Sensor Profiles',
  data_quality: 'Swarming Predictor Data-Quality Diagnostics',
};

const titleCase = value => String(value || '')
  .replace(/_/g, ' ')
  .replace(/\b\w/g, character => character.toUpperCase());

const formatNumber = (value, digits = 2) => (
  Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }) : '—'
);

const formatInteger = value => (
  Number.isFinite(Number(value)) ? Number(value).toLocaleString() : '—'
);

function SummaryCard({ label, value, note, icon: Icon }) {
  return (
    <article style={{
      border: `1px solid ${PALETTE.pale}`,
      borderTop: `3px solid ${PALETTE.navy}`,
      background: PALETTE.paper,
      borderRadius: 4,
      padding: '1rem',
      minHeight: 112,
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.45rem', color: PALETTE.slate,
        fontSize: '0.69rem', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase',
      }}>
        <Icon size={14} strokeWidth={1.8} />
        {label}
      </div>
      <div style={{ marginTop: '0.55rem', color: PALETTE.charcoal, fontSize: '1.45rem', fontWeight: 650 }}>
        {value}
      </div>
      <div style={{ marginTop: '0.25rem', color: PALETTE.slate, fontSize: '0.72rem', lineHeight: 1.45 }}>
        {note}
      </div>
    </article>
  );
}

function Section({ title, subtitle, children }) {
  return (
    <section style={{
      border: `1px solid ${PALETTE.pale}`,
      background: PALETTE.paper,
      borderRadius: 4,
      overflow: 'hidden',
    }}>
      <header style={{ padding: '0.9rem 1rem', borderBottom: `1px solid ${PALETTE.pale}` }}>
        <h3 style={{ margin: 0, color: PALETTE.charcoal, fontSize: '0.94rem', fontWeight: 650 }}>
          {title}
        </h3>
        {subtitle && (
          <p style={{ margin: '0.24rem 0 0', color: PALETTE.slate, fontSize: '0.72rem', lineHeight: 1.45 }}>
            {subtitle}
          </p>
        )}
      </header>
      <div style={{ padding: '1rem' }}>{children}</div>
    </section>
  );
}

function FigureCard({ filename }) {
  const [expanded, setExpanded] = useState(false);
  const [failed, setFailed] = useState(false);
  const key = filename.replace(/\.png$/i, '').toLowerCase();
  const title = CHART_TITLES[key] || titleCase(key);
  const source = `/api/eda-swarming/images/${encodeURIComponent(filename)}`;

  return (
    <>
      <article style={{ border: `1px solid ${PALETTE.pale}`, borderRadius: 4, background: '#fff' }}>
        <header style={{
          minHeight: 48, padding: '0.65rem 0.8rem', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', gap: '0.75rem', borderBottom: `1px solid ${PALETTE.pale}`,
          background: PALETTE.wash,
        }}>
          <div>
            <h4 style={{ margin: 0, color: PALETTE.charcoal, fontSize: '0.77rem', fontWeight: 650 }}>{title}</h4>
            <span style={{ color: PALETTE.slate, fontSize: '0.63rem' }}></span>
          </div>
          <button
            type="button"
            onClick={() => setExpanded(true)}
            aria-label={`Expand ${title}`}
            style={{
              width: 30, height: 30, display: 'grid', placeItems: 'center', borderRadius: 3,
              border: `1px solid ${PALETTE.pale}`, background: '#fff', color: PALETTE.charcoal,
              cursor: 'pointer',
            }}
          >
            <Maximize2 size={14} />
          </button>
        </header>
        <div style={{ height: 330, padding: '0.7rem', display: 'grid', placeItems: 'center' }}>
          {failed ? (
            <div style={{ textAlign: 'center', color: PALETTE.burgundy, fontSize: '0.75rem' }}>
              <XCircle size={22} />
              <div style={{ marginTop: '0.35rem' }}>Figure could not be loaded.</div>
            </div>
          ) : (
            <img
              src={source}
              alt={title}
              onError={() => setFailed(true)}
              onClick={() => setExpanded(true)}
              style={{ width: '100%', height: '100%', objectFit: 'contain', cursor: 'zoom-in' }}
            />
          )}
        </div>
      </article>

      {expanded && !failed && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={title}
          onClick={() => setExpanded(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 1000, padding: '2rem', display: 'grid',
            placeItems: 'center', background: 'rgba(22, 28, 34, 0.90)',
          }}
        >
          <div
            onClick={event => event.stopPropagation()}
            style={{
              position: 'relative', width: 'min(1280px, 96vw)', maxHeight: '92vh',
              padding: '3.2rem 1rem 1rem', background: '#fff', borderRadius: 4,
            }}
          >
            <strong style={{ position: 'absolute', top: '1rem', left: '1rem', color: PALETTE.charcoal, fontSize: '0.85rem' }}>
              {title}
            </strong>
            <button
              type="button"
              onClick={() => setExpanded(false)}
              aria-label="Close expanded figure"
              style={{
                position: 'absolute', top: '0.7rem', right: '0.8rem', width: 30, height: 30,
                display: 'grid', placeItems: 'center', border: `1px solid ${PALETTE.pale}`,
                background: '#fff', borderRadius: 3, cursor: 'pointer',
              }}
            >
              <X size={16} />
            </button>
            <img src={source} alt={title} style={{ width: '100%', maxHeight: '80vh', objectFit: 'contain' }} />
          </div>
        </div>
      )}
    </>
  );
}

function AssociationTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div style={{
      padding: '0.65rem 0.75rem', background: PALETTE.charcoal, color: '#fff',
      borderRadius: 3, fontSize: '0.7rem', lineHeight: 1.55,
    }}>
      <strong>{row.label}</strong>
      <div>Pearson: {formatNumber(row.pearson, 4)}</div>
      <div>Spearman: {formatNumber(row.spearman, 4)}</div>
      <div>Effect size: {formatNumber(row.effectSize, 4)}</div>
    </div>
  );
}

export default function SwarmExploratory() {
  const [dashboard, setDashboard] = useState(null);
  const [images, setImages] = useState([]);
  const [report, setReport] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadEDA = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [dashboardResponse, imagesResponse, reportResponse] = await Promise.all([
        fetch('/api/eda-swarming'),
        fetch('/api/eda-swarming/images'),
        fetch('/api/eda-swarming/report'),
      ]);
      if (!dashboardResponse.ok) {
        throw new Error(`EDA dashboard request failed (${dashboardResponse.status}).`);
      }
      const dashboardData = await dashboardResponse.json();
      const imageData = imagesResponse.ok ? await imagesResponse.json() : [];
      const reportData = reportResponse.ok ? await reportResponse.json() : {};
      setDashboard(dashboardData);
      setImages(
        imageData.filter(filename => (
          /\.png$/i.test(filename) && !/^pelt_regime_hive_/i.test(filename)
        ))
      );
      setReport(reportData.report || '');
    } catch (requestError) {
      setError(requestError.message || 'The EDA results could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadEDA(); }, [loadEDA]);

  const associations = useMemo(() => (
    (dashboard?.correlation_analysis?.top_features || []).map(row => ({
      label: titleCase(row.feature),
      pearson: Number(row.correlation),
      spearman: Number(row.spearman),
      effectSize: Number(row.effect_size),
    })).sort((left, right) => Math.abs(right.pearson) - Math.abs(left.pearson))
  ), [dashboard]);

  if (loading) {
    return (
      <div style={{ padding: '4rem 1rem', textAlign: 'center', color: PALETTE.slate }}>
        <RefreshCw size={28} style={{ animation: 'swarm-spin 1.1s linear infinite' }} />
        <p style={{ marginTop: '0.8rem', fontSize: '0.82rem' }}>Loading swarming EDA outputs…</p>
        <style>{`@keyframes swarm-spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '2rem', textAlign: 'center', border: `1px solid ${PALETTE.pale}`,
        background: PALETTE.paper, borderRadius: 4,
      }}>
        <XCircle size={32} color={PALETTE.burgundy} />
        <h3 style={{ color: PALETTE.charcoal }}>Swarming EDA is unavailable</h3>
        <p style={{ color: PALETTE.slate, fontSize: '0.8rem' }}>{error}</p>
        <p style={{ color: PALETTE.slate, fontSize: '0.73rem' }}>
          Run <code>python scripts/run_swarming_eda.py</code>, then restart the backend.
        </p>
        <button
          type="button"
          onClick={loadEDA}
          style={{
            marginTop: '0.6rem', padding: '0.5rem 0.9rem', border: 0, borderRadius: 3,
            background: PALETTE.navy, color: '#fff', cursor: 'pointer',
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  const summary = dashboard?.summary || {};
  const quality = dashboard?.data_quality || {};
  const prevalence = dashboard?.swarming_analysis?.distribution?.swarming_72h || {};
  const missing = quality.missing_values || [];
  const missingCount = missing.reduce((total, row) => total + Number(row.missing_count || 0), 0);
  const methodology = dashboard?.methodology || {};

  return (
    <main style={{ display: 'flex', flexDirection: 'column', gap: '1rem', color: PALETTE.charcoal }}>
      <header style={{
        padding: '1.15rem 1.2rem', border: `1px solid ${PALETTE.pale}`,
        borderLeft: `5px solid ${PALETTE.navy}`, background: PALETTE.paper, borderRadius: 4,
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem',
      }}>
        <div>
          <div style={{ color: PALETTE.slate, fontSize: '0.66rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Swarming prediction module
          </div>
          <h2 style={{ margin: '0.25rem 0 0', fontSize: '1.15rem', fontWeight: 650 }}>
            Exploratory Data Analysis
          </h2>
          <p style={{ margin: '0.35rem 0 0', maxWidth: 760, color: PALETTE.slate, fontSize: '0.76rem', lineHeight: 1.55 }}>
            Descriptive assessment of sensor distributions, data quality, class prevalence,
            between-hive heterogeneity, temporal structure, and sensor association with the
            72-hour swarming target.
          </p>
        </div>
        <button
          type="button"
          onClick={loadEDA}
          style={{
            display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 0.7rem',
            border: `1px solid ${PALETTE.pale}`, background: '#fff', color: PALETTE.charcoal,
            borderRadius: 3, cursor: 'pointer', fontSize: '0.72rem',
          }}
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
        <SummaryCard label="Records" value={formatInteger(summary.total_records)} note="Sensor observations analysed" icon={Database} />
        <SummaryCard label="Hives" value={formatInteger(summary.total_hives)} note="Independent hive identifiers" icon={BarChart3} />
        <SummaryCard label="Observation period" value={`${formatInteger(summary.time_days)} days`} note="Calendar coverage of the dataset" icon={FileText} />
        <SummaryCard label="72-hour positive rate" value={`${formatNumber(summary.swarm_rate, 3)}%`} note={`${formatInteger(prevalence?.count?.positive)} positive records`} icon={Scale} />
      </div>

      <Section
        title="Analytical figures"
        
      >
        {images.length ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(440px, 1fr))', gap: '0.9rem' }}>
            {images.map(filename => <FigureCard key={filename} filename={filename} />)}
          </div>
        ) : (
          <p style={{ color: PALETTE.slate, fontSize: '0.76rem' }}>No PNG figures were returned by the backend.</p>
        )}
      </Section>

      <Section
        title="Sensor association with swarming within 72 hours"
        
      >
        <div style={{ height: 310 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={associations} layout="vertical" margin={{ top: 5, right: 20, bottom: 20, left: 160 }}>
              <CartesianGrid stroke={PALETTE.pale} strokeDasharray="2 3" horizontal={false} />
              <XAxis type="number" domain={['auto', 'auto']} tick={{ fontSize: 10, fill: PALETTE.slate }} label={{ value: 'Pearson correlation coefficient', position: 'insideBottom', offset: -10, fontSize: 10 }} />
              <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 10, fill: PALETTE.charcoal }} />
              <Tooltip content={<AssociationTooltip />} />
              <ReferenceLine x={0} stroke={PALETTE.charcoal} />
              <Bar dataKey="pearson" fill={PALETTE.navy} maxBarSize={24} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 0.8fr) minmax(340px, 1.2fr)', gap: '1rem' }}>
       

       
      </div>

      
    </main>
  );
}