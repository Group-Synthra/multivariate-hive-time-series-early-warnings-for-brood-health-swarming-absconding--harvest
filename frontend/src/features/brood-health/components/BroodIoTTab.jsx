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
  ShieldCheck,
  Thermometer,
  Wifi,
  WifiOff,
  Wind,
  Zap,
} from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { StatCard } from '../../../components/common/StatCard';
import { useBroodIoT } from '../hooks/useBroodHealthData';
import {
  freshnessLabel,
  numberValue,
  signedNumber,
  timestampValue,
} from '../utils/broodHealth';
import {
  ForecastTrajectoryChart,
  HealthScoreComparisonChart,
  LiveEarlyWarningTimeline,
} from './BroodHealthCharts';
import {
  HealthScoreGauge,
  RoDMeter,
  StabilityGauge,
} from './HealthScoreGauge';

const DEFAULT_REFRESH_SECONDS = 600;

const SENSOR_CARDS = [
  ['temperature_c', 'Internal temperature', '°C', Thermometer],
  ['humidity_pct', 'Internal humidity', '% RH', Droplets],
  ['co2_ppm', 'Internal CO₂', 'ppm', Wind],
  ['weight_kg', 'Hive weight', 'kg', Scale],
  ['external_temp', 'External temperature', '°C', Thermometer],
  ['external_humidity', 'External humidity', '% RH', Droplets],
];

function WarningPanel({ warning, disclaimer }) {
  const level = String(warning?.level || 'Good').toLowerCase();
  return (
    <section className={`brood-warning-panel ${level}`}>
      <Zap size={31} />
      <div>
        <span>BROOD HEALTH EARLY-WARNING STATUS</span>
        <h3>{warning?.title || `${warning?.level || 'Good'} brood-health warning`}</h3>
        <p>{warning?.summary || disclaimer}</p>
        <div className="brood-warning-columns">
          <div>
            <strong>Why this warning was generated</strong>
            <ul>{(warning?.reasons || ['No negative warning rule was triggered.']).map((reason) => <li key={reason}>{reason}</li>)}</ul>
          </div>
          <div>
            <strong>Recommended beekeeper action</strong>
            <ol>{(warning?.recommended_actions || ['Continue routine monitoring.']).map((action) => <li key={action}>{action}</li>)}</ol>
          </div>
        </div>
      </div>
    </section>
  );
}

function IntervalCard({ interval80, interval90, score }) {
  return (
    <div className="brood-interval-card">
      <ShieldCheck size={22} />
      <div>
        <span>Forecast uncertainty around exact +6 h score</span>
        <strong>{numberValue(score, 1)} / 100</strong>
        <p>
          80% residual interval: <b>{numberValue(interval80?.[0], 1)}–{numberValue(interval80?.[1], 1)}</b><br />
          90% residual interval: <b>{numberValue(interval90?.[0], 1)}–{numberValue(interval90?.[1], 1)}</b>
        </p>
      </div>
    </div>
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
  const forecastIndicators = prediction?.forecast_indicators || {};
  const warning = prediction?.warning || {};
  const refreshSeconds = Number(
    iot.devices.data?.refresh_seconds
      || health?.database?.refresh_seconds
      || DEFAULT_REFRESH_SECONDS,
  );
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

  const latestSensors = {
    ...(prediction?.latest_sensors || {}),
    ...(prediction?.latest_raw_sensors || {}),
    ...(prediction?.context || {}),
  };
  const conversion = prediction?.database_weight_conversion || {};
  const contextWarnings = [
    ...(prediction?.domain_shift_warnings || []),
    ...(prediction?.context_warnings || []),
  ];

  return (
    <div className="page-stack">
      <div className="brood-section-heading">
        <div>
          <span className="eyebrow">SRI LANKAN HIVE IOT DEPLOYMENT</span>
          <h3>Rolling exact +6-hour forecast, Future BHSI and Forecast RoD</h3>
          <p>
            Every new IoT reading moves the rolling one-hour aggregation window. A reading at
            10:40 therefore produces a forecast target at 16:40. The native model remains hourly;
            ten-minute points between native forecasts are clearly marked display interpolation.
          </p>
        </div>
        <div className="brood-action-group">
          <label className="brood-switch">
            <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
            <span /> Auto refresh
          </label>
          <button className="button button-outline" onClick={async () => { await iot.refresh(); setLastRefresh(Date.now()); }}>
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
      </div>

      <div className={`brood-live-status ${databaseReady ? 'connected' : 'disconnected'}`}>
        {databaseReady ? <Wifi size={24} /> : <WifiOff size={24} />}
        <div>
          <strong>{databaseReady ? 'Live PostgreSQL connection active' : 'Live PostgreSQL connection unavailable'}</strong>
          <p>
            {databaseReady
              ? `Source: ${health.database.schema}.${health.database.table} · Latest stored reading: ${timestampValue(health.database.latest_recorded_at)}`
              : health?.database?.error}
          </p>
        </div>
        <Database size={22} />
      </div>

      {!modelReady && (
        <div className="brood-alert danger">
          <AlertTriangle size={20} />
          <div><strong>The Brood Health v6 model is not available</strong><p>{health?.model?.error || 'Delete the old model artifacts and retrain.'}</p></div>
        </div>
      )}

      <Panel
        title="Live hive selection"
        subtitle="Devices are discovered directly from the configured PostgreSQL table."
        action={(
          <div className="brood-device-actions">
            <select className="brood-select" value={iot.selectedDevice} onChange={(event) => iot.setSelectedDevice(event.target.value)}>
              <option value="">Select hive</option>
              {(iot.devices.data?.devices || []).map((item) => (
                <option key={item.device_id} value={item.device_id}>{item.device_id}</option>
              ))}
            </select>
            <button
              className="button"
              disabled={!iot.selectedDevice || iot.prediction.loading || !modelReady}
              onClick={async () => { await iot.loadPrediction(); setLastRefresh(Date.now()); }}
            >
              {iot.prediction.loading ? 'Predicting…' : 'Run live prediction'}
            </button>
          </div>
        )}
      >
        <div className="brood-live-meta">
          <span><Clock3 size={15} /> Next refresh: {countdown}</span>
          <span>Latest IoT reading: {timestampValue(prediction?.live_latest_timestamp || prediction?.latest_timestamp)}</span>
          <span>Rolling model anchor: {timestampValue(forecast.forecast_anchor_timestamp || prediction?.latest_timestamp)}</span>
          <span>Forecast target: {timestampValue(forecast.exact_forecast_timestamp)}</span>
          <span>Observed interval: {numberValue(prediction?.reading_interval_minutes, 0)} min</span>
          <span>Raw readings: {numberValue(prediction?.raw_rows, 0)}</span>
          <span>Hourly windows: {numberValue(prediction?.hourly_rows, 0)}</span>
        </div>
      </Panel>

      {iot.prediction.error && (
        <div className="brood-alert danger"><AlertTriangle size={20} /><div><strong>Live prediction failed</strong><p>{iot.prediction.error.message}</p></div></div>
      )}

      {prediction && (
        <>
          <WarningPanel warning={warning} disclaimer={prediction.disclaimer} />

          <div className="brood-four-gauge-grid">
            <HealthScoreGauge
              score={current.score}
              level={current.level}
              label="Current Brood Health Score"
              badge="Now"
              detail="Transparent 1–100 score calculated from the latest internal conditions."
            />
            <HealthScoreGauge
              score={forecast.exact_score}
              level={forecast.exact_level}
              label={`Exact Brood Health Score at +${forecast.horizon_hours || 6} h`}
              badge={timestampValue(forecast.exact_forecast_timestamp)}
              detail={`Selected model: ${prediction.model?.model_name || health?.model?.model_name || 'trained model'}`}
            />
            <StabilityGauge
              score={forecastIndicators.bhsi}
              level={forecastIndicators.stability_level}
              label="Forecast BHSI"
              badge="Next 6 h"
              detail="Stability of the predicted health-score path from now to the exact +6-hour score."
            />
            <RoDMeter
              value={forecastIndicators.rod_points_per_hour}
              label={forecastIndicators.trend_label}
              title="Forecast RoD"
              badge="Next 6 h"
              detail="Predicted health-score slope from current condition through the next six hours."
            />
          </div>

          <div className="stats-grid stats-grid-six">
            <StatCard label="Exact score change" value={signedNumber(forecast.exact_change_points, 1)} unit="points" note="+6 h forecast minus current" />
            <StatCard label="Exact forecast drop" value={numberValue(forecast.exact_drop_points, 1)} unit="points" note="Zero when no decline is forecast" />
            <StatCard label="Safety minimum" value={numberValue(forecast.safety_minimum_score, 1)} unit="/100" note={`${forecast.safety_minimum_level || '—'} · lowest predicted 1–6 h point`} />
            <StatCard label="Feature completeness" value={numberValue(prediction.feature_completeness_percentage, 1)} unit="%" />
            <StatCard label="History sufficiency" value={prediction.history_sufficiency || '—'} note={`Recommended: ${prediction.minimum_recommended_history_hours || 72} h`} />
            <StatCard label="Data freshness" value={freshnessLabel(prediction.data_freshness_minutes)} note={`${numberValue(prediction.data_freshness_minutes, 0)} minutes`} />
          </div>

          <Panel
            title="Observed condition versus six-hour forecast indicators"
            subtitle="Observed indicators describe the recent past. Forecast indicators describe the predicted path from the current score to the exact +6-hour score."
          >
            <div className="stats-grid stats-grid-six">
              <StatCard label="Observed BHSI" value={numberValue(current.bhsi, 1)} unit="/100" note={`${current.stability_level || '—'} · recent historical stability`} />
              <StatCard label="Forecast BHSI" value={numberValue(forecastIndicators.bhsi, 1)} unit="/100" note={`${forecastIndicators.stability_level || '—'} · predicted next-six-hour stability`} />
              <StatCard label="Observed RoD" value={signedNumber(current.rod_points_per_hour, 2)} unit="points/h" note={current.trend_label || '—'} />
              <StatCard label="Forecast RoD" value={signedNumber(forecastIndicators.rod_points_per_hour, 2)} unit="points/h" note={forecastIndicators.trend_label || '—'} />
              <StatCard label="Forecast window start" value={timestampValue(forecastIndicators.window_start_timestamp)} note="Latest rolling IoT anchor" />
              <StatCard label="Forecast window end" value={timestampValue(forecastIndicators.window_end_timestamp)} note="Exact +6-hour target" />
            </div>
          </Panel>

          <div className="two-column-grid">
            <Panel
              title="Current and Predicted Health Comparison"
              subtitle={`Current score compared with the exact +${forecast.horizon_hours || 6}-hour forecast across the Critical, Poor, Good and Excellent ranges.`}
            >
              <HealthScoreComparisonChart
                currentScore={current.score}
                exactScore={forecast.exact_score}
                safetyScore={forecast.safety_minimum_score}
                forecastHorizonHours={forecast.horizon_hours || 6}
                currentTimestamp={forecast.forecast_anchor_timestamp || prediction?.latest_timestamp}
                forecastTimestamp={forecast.exact_forecast_timestamp}
              />
            </Panel>
            <Panel title="Prediction interval and deployment information" subtitle="Residual uncertainty is estimated on validation hives and transferred with the saved model.">
              <IntervalCard
                interval80={forecast.prediction_interval_80}
                interval90={forecast.prediction_interval_90}
                score={forecast.exact_score}
              />
              <div className="brood-info-list">
                <span>Selected model <strong>{prediction.model?.model_name || '—'}</strong></span>
                <span>Primary target <strong>Exact score at +{forecast.horizon_hours || 6} hours</strong></span>
                <span>Secondary target <strong>Minimum predicted score in 1–6 hours</strong></span>
                <span>Forecast anchor <strong>{timestampValue(forecast.forecast_anchor_timestamp)}</strong></span>
                <span>Forecast target <strong>{timestampValue(forecast.exact_forecast_timestamp)}</strong></span>
                <span>Aggregation <strong>Rolling one-hour median aligned to latest reading</strong></span>
                <span>Native model resolution <strong>{forecast.native_model_resolution_minutes || 60} minutes</strong></span>
                <span>Dashboard trajectory <strong>{forecast.display_resolution_minutes || 10} minutes · interpolated between native outputs</strong></span>
                <span>Battery <strong><Battery size={14} /> {numberValue(prediction.context?.battery_voltage, 2)} V</strong></span>
              </div>
            </Panel>
          </div>

          <Panel
            title="Rolling current-to-+6-hour forecast trajectory"
            subtitle="Native model outputs occur hourly. Intermediate ten-minute points make the chart easier to follow and are display interpolation, not separately trained predictions."
          >
            <ForecastTrajectoryChart
              data={forecast.display_trajectory || forecast.trajectory}
              currentScore={current.score}
              exactHorizon={forecast.horizon_hours || 6}
              anchorTimestamp={forecast.forecast_anchor_timestamp}
              targetTimestamp={forecast.exact_forecast_timestamp}
            />
          </Panel>

          <section>
            <div className="brood-section-heading compact">
              <div><h3>Latest live sensor inputs</h3><p>External readings are contextual unless the saved model explicitly lists them as features.</p></div>
            </div>
            <div className="stats-grid stats-grid-six">
              {SENSOR_CARDS.map(([key, label, unit, Icon]) => (
                <StatCard
                  key={key}
                  label={label}
                  value={numberValue(latestSensors[key], key === 'co2_ppm' ? 0 : 2)}
                  unit={unit}
                  icon={Icon}
                />
              ))}
            </div>
          </section>

          <div className="two-column-grid">
            <Panel title="Current score component evidence" subtitle="The current score remains explainable even when the forecast model is complex.">
              <div className="brood-component-grid">
                {Object.entries(prediction.score_components || {}).map(([key, value]) => (
                  <div key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{numberValue(value, 1)}</strong><div><i style={{ width: `${Math.max(0, Math.min(100, Number(value || 0)))}%` }} /></div></div>
                ))}
              </div>
            </Panel>
            <Panel title="Weight transfer and unit handling" subtitle="The model excludes absolute hive weight and uses relative weight changes and stability.">
              <div className="brood-info-list">
                <span>Live scale factor <strong>{numberValue(conversion.scale_factor ?? 1, 4)}</strong></span>
                <span>Live offset <strong>{numberValue(conversion.offset_kg ?? 0, 3)} kg</strong></span>
                <span>Forecast feature strategy <strong>Relative change and coefficient of variation</strong></span>
                <span>Absolute weight <strong>Display and context only</strong></span>
              </div>
            </Panel>
          </div>

          {contextWarnings.length > 0 ? (
            <div className="brood-alert warning"><AlertTriangle size={20} /><div><strong>Transfer or historical-domain warning</strong><ul>{contextWarnings.map((item) => <li key={item}>{item}</li>)}</ul></div></div>
          ) : (
            <div className="brood-alert success"><CheckCircle2 size={20} /><div><strong>No central-range transfer warning detected</strong><p>This still requires local sensor calibration and physical brood validation.</p></div></div>
          )}

          <Panel title="Recent live health and rolling six-hour forecast timeline" subtitle="Observed BHSI/RoD describe past conditions; Forecast BHSI/RoD describe each predicted current-to-+6-hour path.">
            <LiveEarlyWarningTimeline data={prediction.history} />
          </Panel>

          <div className="brood-alert info">
            <AlertTriangle size={19} />
            <div><strong>Decision-support boundary</strong><p>{prediction.disclaimer}</p></div>
          </div>
        </>
      )}
    </div>
  );
}
