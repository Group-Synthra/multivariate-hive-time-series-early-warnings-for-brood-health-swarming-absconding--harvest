import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Battery,
  CheckCircle2,
  Clock3,
  Database,
  Droplets,
  RefreshCw,
  Scale,
  Thermometer,
  Wifi,
  WifiOff,
  Wind,
  Zap,
} from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { StatCard } from '../../../components/common/StatCard';
import { useBroodIoT } from '../hooks/useBroodHealthData';
import { numberValue, timestampValue } from '../utils/broodHealth';
import { HealthScoreComparisonChart, LiveEarlyWarningTimeline } from './BroodHealthCharts';
import { HealthScoreGauge } from './HealthScoreGauge';

const DEFAULT_REFRESH_SECONDS = 600;

const SENSOR_CARDS = [
  ['temperature_c', 'Internal Temperature', '°C', Thermometer],
  ['humidity_pct', 'Internal Humidity', '% RH', Droplets],
  ['co2_ppm', 'Internal CO₂', 'ppm', Wind],
  ['weight_kg', 'Total Hive Weight', 'kg', Scale],
  ['external_temp', 'External Temperature', '°C', Thermometer],
  ['external_humidity', 'External Humidity', '% RH', Droplets],
];

function RoDCard({ value, label }) {
  const numeric = Number(value || 0);
  const maximum = 6;
  const position = Math.max(0, Math.min(100, ((numeric + maximum) / (maximum * 2)) * 100));
  return (
    <article className="health-gauge-card">
      <div className="health-gauge-title">Rate of Development (RoD)</div>
      <div style={{ padding: '2.4rem 1rem 1.1rem' }}>
        <div style={{ height: 14, borderRadius: 999, background: 'linear-gradient(90deg,#dc2626,#f59e0b,#94a3b8,#34d399,#0f766e)', position: 'relative' }}>
          <span style={{ position: 'absolute', left: `${position}%`, top: -13, width: 4, height: 40, borderRadius: 4, background: '#0f172a', transform: 'translateX(-50%)', transition: 'left .8s ease' }} />
        </div>
        <strong style={{ display: 'block', marginTop: 22, fontSize: '1.8rem', color: numeric < -0.5 ? '#dc2626' : numeric > 0.5 ? '#0f766e' : '#475569' }}>{numberValue(numeric, 2)}</strong>
        <span style={{ color: 'var(--muted)', fontSize: '.75rem' }}>score points per hour</span>
      </div>
      <div className="health-gauge-level">{label || 'Stable'}</div>
    </article>
  );
}

export function BroodIoTTab() {
  const iot = useBroodIoT(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [now, setNow] = useState(Date.now());
  const [lastRefresh, setLastRefresh] = useState(Date.now());

  const health = iot.health.data;
  const prediction = iot.prediction.data;
  const databaseReady = Boolean(health?.database?.connected);
  const modelReady = Boolean(health?.model?.ready);
  const current = prediction?.current_condition || {};
  const forecast = prediction?.prediction || {};
  const warning = prediction?.warning || {};
  const refreshSeconds = Number(iot.devices.data?.refresh_seconds || health?.database?.refresh_seconds || DEFAULT_REFRESH_SECONDS);
  const refreshMs = Math.max(30, refreshSeconds) * 1000;

  useEffect(() => {
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(clock);
  }, []);

  useEffect(() => {
    if (!autoRefresh || !iot.selectedDevice) return undefined;
    const timer = window.setInterval(async () => {
      await iot.loadPrediction();
      setLastRefresh(Date.now());
    }, refreshMs);
    return () => window.clearInterval(timer);
  }, [autoRefresh, iot.loadPrediction, iot.selectedDevice, refreshMs]);

  const countdown = useMemo(() => {
    const seconds = Math.max(0, Math.ceil((lastRefresh + refreshMs - now) / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  }, [lastRefresh, now, refreshMs]);

  const latestSensors = { ...(prediction?.latest_sensors || {}), ...(prediction?.context || {}) };
  const scoreChange = Number(forecast.forecast_change_points || 0);

  return (
    <div className="page-stack">
      <div className="brood-section-heading">
        <div>
          <span className="eyebrow">SRI LANKAN HIVE IOT DEPLOYMENT</span>
          <h3>Live current score, future score, BHSI and RoD</h3>
          <p>Supabase/PostgreSQL readings are mapped from your configured columns, aggregated from approximately 10-minute intervals to hourly medians, transformed with the saved feature schema and passed to the selected score-forecasting model.</p>
        </div>
        <div className="brood-action-group">
          <label className="brood-switch"><input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} /><span /> Auto refresh</label>
          <button className="button button-outline" onClick={async () => { await iot.refresh(); setLastRefresh(Date.now()); }}><RefreshCw size={16} /> Refresh</button>
        </div>
      </div>

      <div className={`brood-live-status ${databaseReady ? 'connected' : 'disconnected'}`}>
        {databaseReady ? <Wifi size={24} /> : <WifiOff size={24} />}
        <div>
          <strong>{databaseReady ? 'Live PostgreSQL connection active' : 'Live PostgreSQL connection unavailable'}</strong>
          <p>{databaseReady ? `Source: ${health.database.schema}.${health.database.table} · Latest stored reading: ${timestampValue(health.database.latest_recorded_at)}` : health?.database?.error}</p>
        </div>
        <Database size={22} />
      </div>

      {!modelReady && <div className="brood-alert danger"><AlertTriangle size={20} /><div><strong>The corrected score-forecasting model is not available</strong><p>{health?.model?.error || 'Retrain the brood-health model after applying the corrected files.'}</p></div></div>}

      <Panel title="Live hive selection" subtitle="Devices are discovered directly from the configured IoT table." action={
        <div className="brood-device-actions">
          <select className="brood-select" value={iot.selectedDevice} onChange={(event) => iot.setSelectedDevice(event.target.value)}>
            <option value="">Select hive</option>
            {(iot.devices.data?.devices || []).map((item) => <option key={item.device_id} value={item.device_id}>{item.device_id}</option>)}
          </select>
          <button className="button" disabled={!iot.selectedDevice || iot.prediction.loading || !modelReady} onClick={async () => { await iot.loadPrediction(); setLastRefresh(Date.now()); }}>
            {iot.prediction.loading ? 'Predicting…' : 'Run live prediction'}
          </button>
        </div>
      }>
        <div className="brood-live-meta">
          <span><Clock3 size={15} /> Next refresh: {countdown}</span>
          <span>Latest sensor: {timestampValue(prediction?.latest_timestamp)}</span>
          <span>Raw readings: {numberValue(prediction?.raw_rows, 0)}</span>
          <span>Hourly observations: {numberValue(prediction?.hourly_rows, 0)}</span>
        </div>
      </Panel>

      {iot.prediction.error && <div className="brood-alert danger"><AlertTriangle size={20} /><div><strong>Live prediction failed</strong><p>{iot.prediction.error.message}</p></div></div>}

      {prediction && <>
        <div className={`brood-warning-panel ${String(warning.level || 'good').toLowerCase()}`}>
          <Zap size={31} />
          <div>
            <span>BROOD HEALTH EARLY-WARNING STATUS</span>
            <h3>{warning.title || warning.level || forecast.forecast_level}</h3>
            <p>{warning.summary || prediction.disclaimer}</p>
            <div className="brood-warning-columns">
              <div><strong>Why this warning was generated</strong><ul>{(warning.reasons || []).map((reason) => <li key={reason}>{reason}</li>)}</ul></div>
              <div><strong>Recommended beekeeper action</strong><ol>{(warning.recommended_actions || []).map((action) => <li key={action}>{action}</li>)}</ol></div>
            </div>
          </div>
        </div>

        <div className="brood-four-gauge-grid">
          <HealthScoreGauge score={current.score} level={current.level} label="Current Brood Health Score" detail="Transparent 1–100 score from the latest live sensor history." />
          <HealthScoreGauge score={forecast.forecast_score} level={forecast.forecast_level} label={`Predicted Minimum (${forecast.horizon_hours || 6} h)`} detail={`Selected model: ${prediction.model?.model_name || health?.model?.model_name || 'trained model'}`} />
          <HealthScoreGauge score={current.bhsi} level={current.stability_level} label="BHSI" detail="Stability of the recent Brood Health Score trajectory." />
          <RoDCard value={current.rod_points_per_hour} label={current.trend_label} />
        </div>

        <div className="stats-grid stats-grid-six">
          <StatCard label="Expected score change" value={`${scoreChange >= 0 ? '+' : ''}${numberValue(scoreChange, 1)}`} unit="points" note="Predicted minimum minus current" />
          <StatCard label="Forecast drop" value={numberValue(forecast.forecast_drop_points, 1)} unit="points" note="Zero when no decline is forecast" />
          <StatCard label="Risk index" value={numberValue(forecast.risk_index, 1)} unit="/100" note="100 minus forecast score; not a probability" />
          <StatCard label="Feature completeness" value={numberValue(prediction.feature_completeness_percentage, 1)} unit="%" />
          <StatCard label="History sufficiency" value={prediction.history_sufficiency || '—'} note={`Recommended: ${prediction.minimum_recommended_history_hours || 72} h`} />
          <StatCard label="Data freshness" value={numberValue(prediction.data_freshness_minutes, 0)} unit="min" />
        </div>

        <div className="two-column-grid">
          <Panel title="Current versus predicted minimum score" subtitle="The future value is the minimum score expected anywhere inside the forecast window, not an exact measurement at one future timestamp.">
            <HealthScoreComparisonChart currentScore={current.score} predictedScore={forecast.forecast_score} />
          </Panel>
          <Panel title="Deployment information" subtitle="Saved model, source mapping and live-data readiness.">
            <div className="brood-info-list">
              <span>Selected model <strong>{prediction.model?.model_name || '—'}</strong></span>
              <span>Prediction target <strong>Future-window minimum score</strong></span>
              <span>Prediction horizon <strong>{forecast.horizon_hours || 6} hours</strong></span>
              <span>Latest sensor time <strong>{timestampValue(prediction.latest_timestamp)}</strong></span>
              <span>Hourly aggregation <strong>Median of raw readings</strong></span>
              <span>Battery voltage <strong><Battery size={14} /> {numberValue(prediction.context?.battery_voltage, 2)} V</strong></span>
            </div>
          </Panel>
        </div>

        <div className="stats-grid stats-grid-six">
          {SENSOR_CARDS.map(([key, label, unit, Icon]) => <StatCard key={key} label={label} value={numberValue(latestSensors[key], key === 'co2_ppm' ? 0 : 2)} unit={unit} icon={Icon} />)}
        </div>

        {prediction.domain_shift_warnings?.length > 0
          ? <div className="brood-alert warning"><AlertTriangle size={20} /><div><strong>Live readings outside the historical training range</strong><ul>{prediction.domain_shift_warnings.map((item) => <li key={item}>{item}</li>)}</ul></div></div>
          : <div className="brood-alert success"><CheckCircle2 size={20} /><div><strong>No central-range domain-shift warning detected</strong><p>This does not replace sensor calibration or field validation.</p></div></div>}

        <Panel title="Recent live score and warning timeline" subtitle="Current score, predicted minimum score, BHSI and RoD generated from the same hourly history.">
          <LiveEarlyWarningTimeline data={prediction.history} />
        </Panel>

        <div className="brood-alert info"><AlertTriangle size={19} /><div><strong>Interpretation boundary</strong><p>{prediction.disclaimer} The risk index is a score transformation, not a calibrated disease probability.</p></div></div>
      </>}
    </div>
  );
}
