import { useMemo, useState } from 'react';
import { Activity, AlertTriangle, CalendarDays, Database, HeartPulse, Users } from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { StatCard } from '../../../components/common/StatCard';
import { EmptyState } from '../../../components/common/EmptyState';
import { useBroodEDA } from '../hooks/useBroodHealthData';
import { SENSOR_META, numberValue, percentValue, timestampValue } from '../utils/broodHealth';
import {
  ConditionLevelChart,
  HiveHealthyRateChart,
  PrecursorChart,
  RelationshipScatterChart,
  SensorDistributionChart,
  SensorEffectChart,
  TargetBalanceChart,
  TemporalHealthyRateChart,
} from './BroodHealthCharts';
import { CorrelationMatrix, TransitionMatrix } from './BroodMatrices';
import { BroodReportGallery } from './BroodReportGallery';

const SENSOR_KEYS = Object.keys(SENSOR_META);

function InlineState({ loading, error, onRetry }) {
  if (loading) return <div className="brood-inline-state">Loading comprehensive brood-health analysis…</div>;
  if (error) return <div className="brood-inline-state error"><strong>EDA could not be loaded.</strong><span>{error.message}</span><button className="button" onClick={onRetry}>Retry</button></div>;
  return null;
}

function SensorStatisticsTable({ rows }) {
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Sensor</th><th>Healthy mean</th><th>Unhealthy mean</th><th>Difference</th><th>Cohen’s d</th><th>Target correlation</th><th>Missing</th></tr></thead>
        <tbody>{(rows || []).map((row) => (
          <tr key={row.sensor}>
            <td><strong>{row.label}</strong> <small>{row.unit}</small></td>
            <td>{numberValue(row.healthy_mean, 2)}</td><td>{numberValue(row.unhealthy_mean, 2)}</td>
            <td>{numberValue(row.mean_difference, 2)}</td><td>{numberValue(row.cohens_d, 3)}</td>
            <td>{numberValue(row.target_correlation, 3)}</td><td>{numberValue(row.missing, 0)}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}


function ScoreDefinitionPanel({ definition, components }) {
  const weights = definition?.weights || {};
  const componentLookup = new Map((components || []).map((row) => [row.component, row]));
  const rows = [
    ['Temperature', 'temperature_component', weights.temperature],
    ['Humidity', 'humidity_component', weights.humidity],
    ['CO₂', 'co2_component', weights.co2],
    ['Relative weight stability', 'weight_component', weights.weight_stability],
  ];
  return (
    <div>
      <div className="table-scroll">
        <table><thead><tr><th>Current-score component</th><th>Calibrated weight</th><th>Mean component score</th><th>Median</th><th>Range</th></tr></thead>
          <tbody>{rows.map(([label, key, weight]) => { const item = componentLookup.get(key) || {}; return (
            <tr key={key}><td><strong>{label}</strong></td><td>{percentValue(Number(weight || 0) * 100, 0)}</td><td>{numberValue(item.mean, 2)}</td><td>{numberValue(item.median, 2)}</td><td>{numberValue(item.minimum, 1)}–{numberValue(item.maximum, 1)}</td></tr>
          ); })}</tbody>
        </table>
      </div>
      <p className="chart-footnote">The weights are calibrated on training hives only and are treated as research coefficients, not universal biological percentages. Absolute hive weight is not transferred as a forecast feature.</p>
    </div>
  );
}

export function BroodExploratoryTab() {
  const resource = useBroodEDA(true);
  const [sensor, setSensor] = useState('temperature_c');
  const [precursorSensor, setPrecursorSensor] = useState('temperature_c');
  const [temporalMode, setTemporalMode] = useState('hourly');
  const [xSensor, setXSensor] = useState('temperature_c');
  const [ySensor, setYSensor] = useState('humidity_pct');
  const data = resource.data;
  const meta = data?.meta || {};
  const temporal = data?.temporal?.[temporalMode] || [];
  const temporalKey = temporalMode === 'hourly' ? 'hour' : temporalMode === 'weekday' ? 'weekday' : 'month';
  const strongestSensor = useMemo(() => [...(data?.sensor_statistics || [])].sort((a, b) => Math.abs(b.cohens_d || 0) - Math.abs(a.cohens_d || 0))[0], [data]);

  if (!data) return <InlineState loading={resource.loading} error={resource.error} onRetry={resource.refetch} />;

  return (
    <div className="page-stack">
      <div className="brood-section-heading">
        <div><span className="eyebrow">OBSERVED BROOD-HEALTH TARGET</span><h3>Exploratory analysis from biological, temporal, hive and data-quality perspectives</h3><p>The analysis uses the observed <code>brood_health_healthy_1</code> label. The transparent 1–100 Brood Health Score is shown separately and is not treated as direct veterinary ground truth.</p></div>
        <div className="brood-date-range"><CalendarDays size={17} /><span>{timestampValue(meta.analysis_start)}<br />to {timestampValue(meta.analysis_end)}</span></div>
      </div>

      <div className="stats-grid stats-grid-six">
        <StatCard label="Records" value={meta.records} icon={Database} note="Hourly observations" />
        <StatCard label="Hives" value={meta.hives} icon={Users} note="Independent hive streams" />
        <StatCard label="Healthy rate" value={percentValue(meta.healthy_rate)} icon={HeartPulse} />
        <StatCard label="Unhealthy records" value={meta.unhealthy_count} icon={AlertTriangle} />
        <StatCard label="Unhealthy onsets" value={data?.transitions?.unhealthy_onsets} icon={Activity} note="Healthy → unhealthy" />
        <StatCard label="Strongest effect" value={strongestSensor?.label || '—'} note={strongestSensor ? `|Cohen’s d| ${numberValue(Math.abs(strongestSensor.cohens_d), 2)}` : undefined} />
      </div>

      <div className="two-column-grid">
        <Panel title="Observed target balance" subtitle="Healthy and unhealthy records retained for EDA and secondary biological alignment checks."><TargetBalanceChart data={data.class_balance} /></Panel>
        <Panel title="Sensor separation by status" subtitle="Absolute Cohen’s d indicates how strongly each sensor separates healthy and unhealthy records."><SensorEffectChart data={data.sensor_statistics} /></Panel>
      </div>

      <Panel title="Sensor distributions by observed status" subtitle="Normalized class distributions reveal overlap, skew and possible decision boundaries." action={<select className="brood-select" value={sensor} onChange={(e) => setSensor(e.target.value)}>{SENSOR_KEYS.map((key) => <option key={key} value={key}>{SENSOR_META[key].label}</option>)}</select>}>
        <SensorDistributionChart data={data.sensor_distributions?.[sensor]} sensor={sensor} />
      </Panel>

      <Panel title="Descriptive and inferential sensor comparison" subtitle="Healthy/unhealthy means, standardized effect size, target correlation and missingness."><SensorStatisticsTable rows={data.sensor_statistics} /></Panel>

      <div className="two-column-grid">
        <Panel title="Temporal brood-health pattern" subtitle="Compare healthy rate and unhealthy record volume across ordered time periods." action={<div className="segmented-control compact">{['hourly', 'weekday', 'monthly'].map((mode) => <button className={temporalMode === mode ? 'active' : ''} key={mode} onClick={() => setTemporalMode(mode)}>{mode}</button>)}</div>}>
          <TemporalHealthyRateChart data={temporal} xKey={temporalKey} labelFormatter={(v) => temporalMode === 'hourly' ? `${v}:00` : v} />
        </Panel>
        <Panel title="Current Brood Health Score levels" subtitle="Transparent 1–100 score categories used for the current and future module outputs."><ConditionLevelChart data={data.condition_level_balance} /></Panel>
      </div>

      <Panel title="Transparent current-score definition" subtitle="Component distributions and calibrated coefficients used to calculate the present 1–100 Brood Health Score.">
        <ScoreDefinitionPanel definition={data.score_definition} components={data.score_component_summary} />
      </Panel>

      <Panel title="Between-hive variation" subtitle="Per-hive observed healthy rate identifies persistent differences and supports group-aware interpretation."><HiveHealthyRateChart data={data.hive_profiles} /></Panel>

      <div className="two-column-grid">
        <Panel title="One-hour transition matrix" subtitle="Persistence, deterioration and recovery probabilities between consecutive observations."><TransitionMatrix data={data.transitions?.counts} /></Panel>
        <Panel title="Episode-duration profile" subtitle="Duration of contiguous healthy and unhealthy periods.">
          <div className="table-scroll"><table><thead><tr><th>Status</th><th>Episodes</th><th>Mean hours</th><th>Median</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody>{(data.transitions?.episode_summary || []).map((row) => <tr key={row.status}><td><strong>{row.status}</strong></td><td>{numberValue(row.count, 0)}</td><td>{numberValue(row.mean, 1)}</td><td>{numberValue(row.median, 1)}</td><td>{numberValue(row.min, 0)}</td><td>{numberValue(row.max, 0)}</td></tr>)}</tbody></table></div>
          <div className="brood-kpi-strip"><span><strong>{numberValue(data.transitions?.unhealthy_onsets, 0)}</strong> unhealthy onsets</span><span><strong>{numberValue(data.transitions?.recoveries, 0)}</strong> recoveries</span></div>
        </Panel>
      </div>

      <Panel title="Precursor analysis before unhealthy onset" subtitle={`Sensor shifts relative to the ${data.precursor_analysis?.baseline_window || 'earlier'} baseline across ${numberValue(data.precursor_analysis?.accepted_onsets, 0)} accepted onsets.`} action={<select className="brood-select" value={precursorSensor} onChange={(e) => setPrecursorSensor(e.target.value)}>{SENSOR_KEYS.map((key) => <option key={key} value={key}>{SENSOR_META[key].label}</option>)}</select>}>
        <PrecursorChart data={data.precursor_analysis?.rows} sensor={precursorSensor} />
      </Panel>

      <div className="two-column-grid">
        <Panel title="Sensor correlation matrix" subtitle="Pearson correlations among sensors, condition index and observed target."><CorrelationMatrix data={data.correlation} /></Panel>
        <Panel title="Sensor relationship by status" subtitle="Deterministic sample for visually assessing nonlinear separation and overlap." action={<div className="brood-select-pair"><select className="brood-select" value={xSensor} onChange={(e) => setXSensor(e.target.value)}>{SENSOR_KEYS.map((key) => <option key={key} value={key}>X: {SENSOR_META[key].label}</option>)}</select><select className="brood-select" value={ySensor} onChange={(e) => setYSensor(e.target.value)}>{SENSOR_KEYS.filter((key) => key !== xSensor).map((key) => <option key={key} value={key}>Y: {SENSOR_META[key].label}</option>)}</select></div>}><RelationshipScatterChart data={data.scatter_sample} xSensor={xSensor} ySensor={ySensor} /></Panel>
      </div>

      <div className="two-column-grid">
        <Panel title="Data-quality audit" subtitle="Missing values, duplicate hive timestamps and temporal gaps.">
          <div className="brood-quality-grid"><div><strong>{numberValue(data.data_quality?.total_missing, 0)}</strong><span>Total missing cells</span></div><div><strong>{numberValue(data.data_quality?.duplicate_hive_timestamps, 0)}</strong><span>Duplicate hive timestamps</span></div><div><strong>{numberValue(data.data_quality?.detected_time_gaps, 0)}</strong><span>Detected time gaps</span></div></div>
          <div className="table-scroll"><table><thead><tr><th>Sensor</th><th>IQR outliers</th><th>Percentage</th><th>Lower bound</th><th>Upper bound</th></tr></thead><tbody>{(data.data_quality?.outliers || []).map((row) => <tr key={row.sensor}><td>{row.label}</td><td>{numberValue(row.count, 0)}</td><td>{percentValue(row.percentage)}</td><td>{numberValue(row.lower_bound, 2)}</td><td>{numberValue(row.upper_bound, 2)}</td></tr>)}</tbody></table></div>
        </Panel>
        <Panel title="Leakage-safe analytical design" subtitle="Controls applied before model evaluation."><ul className="brood-check-list">{(data.methodology?.leakage_controls || []).map((item) => <li key={item}>{item}</li>)}</ul><div className="brood-method-box"><strong>Split:</strong> {data.methodology?.split_rule}<br /><strong>Forecast horizon:</strong> {data.methodology?.forecast_horizon_hours} hours</div></Panel>
      </div>

      <Panel title="Exact Python-generated analysis figures" subtitle="These images are served directly from the backend report artifacts."><BroodReportGallery images={data.generated_images} /></Panel>
      {!data.generated_images?.length && <EmptyState message="Run the EDA endpoint once to generate report figures." />}
    </div>
  );
}
