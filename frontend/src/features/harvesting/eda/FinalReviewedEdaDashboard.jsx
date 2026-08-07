import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Database,
  Droplets,
  Gauge,
  Scale,
  ShieldCheck,
  Sparkles,
  Thermometer,
  Wind,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { loadFinalHarvestEdaDashboard } from "../../../services/finalHarvestEdaService";
import "./FinalReviewedEdaDashboard.css";

const FAMILY_META = {
  weight: { label: "Weight", icon: Scale },
  humidity: { label: "Humidity", icon: Droplets },
  temperature: { label: "Temperature", icon: Thermometer },
  co2: { label: "CO₂", icon: Wind },
};

function integer(value) {
  return Number(value ?? 0).toLocaleString();
}

function decimal(value, digits = 2) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
}

function percent(value, digits = 3) {
  return `${(Number(value ?? 0) * 100).toFixed(digits)}%`;
}

function dateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SummaryCard({ label, value, note, icon: Icon }) {
  return (
    <article className="final-eda-summary-card">
      <span className="final-eda-summary-icon"><Icon size={19} /></span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
        <span>{note}</span>
      </div>
    </article>
  );
}

function IntegrityItem({ title, note }) {
  return (
    <div className="final-eda-integrity-item">
      <CheckCircle2 size={19} />
      <div>
        <strong>{title}</strong>
        <small>{note}</small>
      </div>
    </div>
  );
}

function SensorCard({ item }) {
  const meta = FAMILY_META[item.family] ?? {
    label: item.label,
    icon: Activity,
  };
  const Icon = meta.icon;
  return (
    <article className="final-eda-sensor-card">
      <div className="final-eda-sensor-card-heading">
        <span className={`final-eda-family-icon family-${item.family}`}>
          <Icon size={20} />
        </span>
        <div>
          <small>{meta.label}</small>
          <strong>{item.strength} separation</strong>
        </div>
      </div>
      <div className="final-eda-sensor-score">
        <strong>{decimal(item.maximum_absolute_smd, 2)}</strong>
        <span>|SMD| at 72h</span>
      </div>
      <p>{item.feature_display_name}</p>
    </article>
  );
}

function CoverageRing({ value }) {
  const bounded = Math.min(Math.max(Number(value) || 0, 0), 100);
  return (
    <div
      className="final-eda-coverage-ring"
      style={{ "--coverage": `${bounded * 3.6}deg` }}
    >
      <div>
        <strong>{bounded.toFixed(0)}%</strong>
        <span>coverage</span>
      </div>
    </div>
  );
}

function FeatureTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="final-eda-tooltip">
      <strong>{row.display_name}</strong>
      <span>|SMD| {decimal(row.absolute_smd, 2)}</span>
      <span>{row.smd >= 0 ? "Higher in event samples" : "Lower in event samples"}</span>
    </div>
  );
}

export default function FinalReviewedEdaDashboard() {
  const [dashboard, setDashboard] = useState(null);
  const [error, setError] = useState("");
  const [selectedLead, setSelectedLead] = useState(72);

  useEffect(() => {
    let active = true;
    loadFinalHarvestEdaDashboard()
      .then((payload) => {
        if (!active) return;
        setDashboard(payload);
        setSelectedLead(Number(payload.lead_hours?.[0] ?? 72));
      })
      .catch((requestError) => {
        if (active) setError(requestError.message);
      });
    return () => { active = false; };
  }, []);

  const selectedFeatures = useMemo(() => (
    dashboard?.top_sensor_features_by_lead?.[String(selectedLead)] ?? []
  ), [dashboard, selectedLead]);

  const evolution = useMemo(() => (
    [...(dashboard?.signal_evolution ?? [])].sort(
      (a, b) => Number(b.lead_hours) - Number(a.lead_hours),
    )
  ), [dashboard]);

  if (error) {
    return (
      <div className="final-eda-state is-error">
        <h3>Final EDA dashboard data is unavailable</h3>
        <p>{error}</p>
        <code>python scripts/export_final_harvest_eda_dashboard.py</code>
      </div>
    );
  }

  if (!dashboard) {
    return (
      <div className="final-eda-state">
        <div className="final-eda-spinner" />
        <p>Loading reviewed-event analysis…</p>
      </div>
    );
  }

  const summary = dashboard.summary;
  const integrity = dashboard.integrity;
  const coverage = dashboard.coverage;

  return (
    <section className="final-eda-dashboard">
      <header className="final-eda-heading">
        <div>
          <span className="final-eda-eyebrow">TIME-OPTIMAL HONEY HARVESTING</span>
          <h2>Reviewed Event Exploratory Analysis</h2>
          <p>Leakage safe exploratory evidence supporting the 72 hour harvest event forecasting task.</p>
        </div>
        <div className="final-eda-export-badge">
          <small>ANALYSIS READY</small>
          <strong>{dateTime(dashboard.generated_at)}</strong>
        </div>
      </header>

      <div className="final-eda-summary-grid">
        <SummaryCard icon={Database} label="Modelling rows" value={integer(summary.modelling_rows)} note={`${integer(summary.positive_rows)} positive rows`} />
        <SummaryCard icon={CheckCircle2} label="Reviewed events" value={integer(summary.reviewed_events)} note={`${integer(summary.positive_hives)} positive hives`} />
        <SummaryCard icon={Sparkles} label="Engineered features" value={integer(summary.engineered_features)} note={`${integer(summary.minimum_history_hours)}h past history`} />
        <SummaryCard icon={Gauge} label="Rare-event target" value={percent(summary.target_prevalence)} note="72-hour horizon" />
        <SummaryCard icon={ShieldCheck} label="Analysis coverage" value={`${decimal(coverage.coverage_percent, 0)}%`} note="Event + matched-control samples" />
        <SummaryCard icon={BarChart3} label="Sensor families" value={`${dashboard.strong_sensor_family_count}/4`} note="Large 72h separation" />
      </div>

      <article className="final-eda-panel final-eda-integrity-panel">
        <div className="final-eda-section-heading">
          <div>
            <span className="final-eda-eyebrow">DATASET INTEGRITY</span>
          </div>
          <span className="final-eda-success-badge"><CheckCircle2 size={16} /> Ready for modelling</span>
        </div>
        <div className="final-eda-integrity-grid">
          <IntegrityItem title="Past-only features" note={`${summary.minimum_history_hours}h historical windows`} />
          <IntegrityItem title="No prohibited leakage features" note={integrity.no_prohibited_leakage_features ? "Target/event indicators excluded" : "Review feature manifest"} />
          <IntegrityItem title="Complete event-lead coverage" note={`${integrity.available_event_samples}/${integrity.expected_samples} samples`} />
          <IntegrityItem title="Complete matched controls" note={`${integrity.available_controls}/${integrity.expected_samples} controls`} />
          <IntegrityItem title="No missing analysis samples" note={`${integrity.missing_event_samples + integrity.missing_controls} missing`} />
          <IntegrityItem title="Chronological split" note={`${summary.events_by_split?.train ?? 0} train · ${summary.events_by_split?.validation ?? 0} validation · ${summary.events_by_split?.test ?? 0} test`} />
        </div>
      </article>

      <section>
        <div className="final-eda-section-heading">
          <div>
            <span className="final-eda-eyebrow">HARVEST-RELATED SENSOR EVIDENCE</span>
            <h3>Strongest sensor derived signals at 72 hours</h3>
            <p>Event periods are compared with matched normal periods using standardized mean differences.</p>
          </div>
        </div>
        <div className="final-eda-sensor-grid">
          {(dashboard.sensor_summary_72h ?? []).map((item) => (
            <SensorCard key={item.family} item={item} />
          ))}
        </div>
      </section>

      <div className="final-eda-analysis-grid">
        <article className="final-eda-panel">
          <div className="final-eda-chart-heading">
            <div>
              <span className="final-eda-eyebrow">LEAD TIME ANALYSIS</span>
              <h3>Top Sensor Derived Differences</h3>
            </div>
            <div className="final-eda-tabs">
              {(dashboard.lead_hours ?? []).map((lead) => (
                <button
                  key={lead}
                  type="button"
                  className={Number(lead) === selectedLead ? "is-active" : ""}
                  onClick={() => setSelectedLead(Number(lead))}
                >
                  {lead}h
                </button>
              ))}
            </div>
          </div>
          <div className="final-eda-bar-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={selectedFeatures} layout="vertical" margin={{ top: 8, right: 20, bottom: 8, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="display_name" type="category" width={170} tick={{ fontSize: 11 }} />
                <Tooltip content={<FeatureTooltip />} />
                <Bar dataKey="absolute_smd" radius={[0, 7, 7, 0]}>
                  {selectedFeatures.map((item) => (
                    <Cell key={`${item.feature}-${item.lead_hours}`} className={`family-${item.sensor_family}`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="final-eda-caption">Larger |SMD| means stronger descriptive separation between reviewed-event and matched-control periods.</p>
        </article>

        <article className="final-eda-panel">
          <div className="final-eda-chart-heading">
            <div>
              <span className="final-eda-eyebrow">SIGNAL EVOLUTION</span>
              <h3>Sensor Evidence Across Prediction Horizons</h3>
            </div>
          </div>
          <div className="final-eda-line-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={evolution} margin={{ top: 12, right: 20, bottom: 8, left: -6 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="lead_hours" tickFormatter={(value) => `${value}h`} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value, name) => [decimal(value, 2), FAMILY_META[name]?.label ?? name]} labelFormatter={(value) => `${value}h lead`} />
                <Legend formatter={(value) => FAMILY_META[value]?.label ?? value} />
                <Line type="monotone" dataKey="weight" stroke="#2563eb" strokeWidth={3} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="humidity" stroke="#0891b2" strokeWidth={3} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="temperature" stroke="#d97706" strokeWidth={3} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="co2" stroke="#7c3aed" strokeWidth={3} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="final-eda-caption">Each line shows the strongest feature within that sensor family at each lead time.</p>
        </article>
      </div>

      {/* <article className="final-eda-panel final-eda-coverage-panel">
        <div className="final-eda-coverage-copy">
          <CoverageRing value={coverage.coverage_percent} />
          <div>
            <span className="final-eda-eyebrow">REVIEWED-EVENT COVERAGE</span>
            <h3>Complete event-control analysis</h3>
            <p>Every reviewed event was evaluated at {coverage.lead_times_per_event} lead times with a matched control.</p>
          </div>
        </div>
        <div className="final-eda-coverage-stats">
          <div><small>Reviewed events</small><strong>{coverage.reviewed_events}</strong></div>
          <div><small>Expected samples</small><strong>{coverage.expected_event_lead_samples}</strong></div>
          <div><small>Event samples</small><strong>{coverage.available_event_lead_samples}</strong></div>
          <div><small>Matched controls</small><strong>{coverage.available_matched_controls}</strong></div>
        </div>
      </article> */}

      <article className="final-eda-takeaway">

        <div>
          <h3>{dashboard.takeaway.title}</h3>
          <p>{dashboard.takeaway.text}</p>
        </div>
      </article>

      <details className="final-eda-details">
        <summary><span>Detailed sensor statistics</span><ChevronDown size={18} /></summary>
        <div className="final-eda-table-wrap">
          <table>
            <thead><tr><th>Lead</th><th>Feature</th><th>Event mean</th><th>Control mean</th><th>SMD</th><th>Event N</th><th>Control N</th></tr></thead>
            <tbody>
              {(dashboard.detailed_sensor_statistics ?? []).map((row) => (
                <tr key={`${row.lead_hours}-${row.feature}`}>
                  <td>{row.lead_hours}h</td><td>{row.feature}</td><td>{decimal(row.event_mean, 3)}</td><td>{decimal(row.control_mean, 3)}</td><td>{decimal(row.standardized_mean_difference, 3)}</td><td>{row.event_n}</td><td>{row.control_n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
