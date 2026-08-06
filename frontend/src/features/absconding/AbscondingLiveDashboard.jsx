import React, { useMemo, useState } from 'react';
import {
  AlertTriangle, CheckCircle, Info, RefreshCw, ShieldCheck, TrendingUp
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts';

const RANGE_OPTIONS = [
  { label: '1H', hours: 1 },
  { label: '6H', hours: 6 },
  { label: '12H', hours: 12 },
  { label: '24H', hours: 24 },
  { label: '7D', hours: 168 },
];

function clamp(value, min = 0, max = 100) {
  const n = Number(value);
  if (Number.isNaN(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function getNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isNaN(n) ? fallback : n;
}

function getOptionalNum(value) {
  if (value === undefined || value === null || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(value, digits = 1, suffix = '') {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function signed(value, digits = 1, suffix = '') {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return '—';
  const n = Number(value);
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}${suffix}`;
}

function probabilityPercent(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}%`;
}

function trendFromChange(changePercentagePoints) {
  const change = Number(changePercentagePoints);
  if (!Number.isFinite(change)) return 'Not available';
  if (change >= 5) return 'Rapidly Increasing';
  if (change >= 1) return 'Increasing';
  if (change >= 0.005) return 'Slightly Increasing';
  if (change <= -5) return 'Rapidly Decreasing';
  if (change <= -1) return 'Decreasing';
  if (change <= -0.005) return 'Slightly Decreasing';
  return 'Stable';
}

function adaptiveRiskDomain(rows) {
  const values = (rows || [])
    .map((row) => Number(row.risk))
    .filter((value) => Number.isFinite(value));
  if (!values.length) return [0, 5];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(max - min, 0.05);
  const padding = Math.max(spread * 0.3, 0.05);
  let lower = Math.max(0, min - padding);
  let upper = Math.min(100, max + padding);

  if (upper - lower < 0.2) {
    const centre = (upper + lower) / 2;
    lower = Math.max(0, centre - 0.1);
    upper = Math.min(100, centre + 0.1);
  }

  lower = Math.floor(lower * 100) / 100;
  upper = Math.ceil(upper * 100) / 100;
  return [lower, upper];
}

function safeDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function dateTime(value) {
  const d = safeDate(value);
  if (!d) return '—';
  return d.toLocaleString([], {
    month: 'numeric',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function timeLabel(value) {
  const d = safeDate(value);
  if (!d) return String(value || '—').slice(0, 10);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMinutes(value, minutes = 10) {
  const d = safeDate(value);
  return d ? new Date(d.getTime() + minutes * 60 * 1000).toISOString() : null;
}

function addHours(value, hours = 24) {
  const d = safeDate(value);
  return d ? new Date(d.getTime() + hours * 60 * 60 * 1000).toISOString() : null;
}

function predictionHorizonHours(data, fallback = 24) {
  const direct = getOptionalNum(data?.prediction_horizon_hours);
  if (direct !== null && direct > 0) return direct;

  const windowText = String(data?.prediction_window || '');
  const match = windowText.match(/next[_\s-]?(\d+)[_\s-]?hours?/i);
  if (match) {
    const parsed = Number(match[1]);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }

  return fallback;
}

function isIncreasingTrend(trend) {
  return ['slightly increasing', 'increasing', 'rapidly increasing'].includes(
    String(trend || '').trim().toLowerCase(),
  );
}

function forecastAlertState(riskLevel, armTrend, notification = {}) {
  const level = String(riskLevel || 'Low').trim().toLowerCase();
  const increasing = isIncreasingTrend(armTrend);

  if (level === 'high') {
    return {
      label: 'HIGH RISK',
      message: notification.message || 'Immediate hive inspection is required.',
      className: 'critical',
    };
  }

  if (notification.should_notify || (level === 'medium' && increasing)) {
    return {
      label: 'WARNING',
      message: notification.message || 'Absconding risk is increasing. Inspect the colony soon.',
      className: 'warning',
    };
  }

  if (level === 'medium') {
    return {
      label: 'WATCH',
      message: notification.message || 'Elevated risk detected. Continue close monitoring.',
      className: 'watch',
    };
  }

  return {
    label: 'NORMAL',
    message: notification.message || 'Continue monitoring.',
    className: 'normal',
  };
}

function riskColor(level) {
  const clean = String(level || '').toLowerCase();
  if (clean === 'high') return 'var(--accent-crimson)';
  if (clean === 'medium') return 'var(--accent-gold)';
  return 'var(--accent-emerald)';
}

function displayLevel(level) {
  return String(level || 'Low').toUpperCase();
}

function freshnessText(status) {
  if (!status) return 'Checking';
  if (status === 'Fresh') return 'Live';
  return status;
}

function freshnessClass(status) {
  if (status === 'Fresh') return 'live';
  if (status === 'Stale') return 'stale';
  if (status === 'Delayed') return 'delayed';
  return 'checking';
}

function modelNameFromPath(path, fallback = 'Trained Model') {
  if (!path) return fallback;
  const file = String(path).split(/[\\/]/).pop() || '';
  if (file.includes('rf')) return 'Random Forest';
  if (file.includes('extratrees')) return 'Extra Trees';
  if (file.includes('fast')) return 'Logistic Regression';
  if (file.includes('gnb') || file.includes('nb')) return 'Naive Bayes';
  return fallback;
}

function statusForSensor(type, value, latest) {
  const n = getNum(value, null);
  if (n === null) return { label: 'Unknown', className: 'neutral' };
  if (type === 'temperature') {
    const deviation = Math.abs(getNum(latest.temp_deviation_from_35, n - 35));
    if (deviation >= 2.5 || n < 31 || n > 38) return { label: 'Unstable', className: 'bad' };
    if (deviation >= 1.2) return { label: 'Watch', className: 'warn' };
    return { label: 'Stable', className: 'good' };
  }
  if (type === 'humidity') {
    if (n > 75 || n < 40) return { label: n > 75 ? 'High' : 'Low', className: 'bad' };
    if (n > 70 || n < 45) return { label: 'Watch', className: 'warn' };
    return { label: 'Normal', className: 'good' };
  }
  if (type === 'co2') {
    if (n >= 1800) return { label: 'High', className: 'bad' };
    if (n >= 1200) return { label: 'Elevated', className: 'warn' };
    return { label: 'Normal', className: 'good' };
  }
  if (type === 'weight') {
    const sixHour = getNum(latest.weight_change_6h, 0);
    const day = getNum(latest.weight_change_24h, 0);
    if (sixHour < -0.8 || day < -1.5) return { label: 'Declining', className: 'bad' };
    if (sixHour < -0.2 || day < -0.5) return { label: 'Slight drop', className: 'warn' };
    return { label: 'Stable', className: 'good' };
  }
  return { label: 'Normal', className: 'good' };
}

function rangeFilter(rows, hours) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  const enriched = rows.map((row, index) => ({ ...row, _date: safeDate(row.timestamp), _index: index }));
  const dateRows = enriched.filter(row => row._date);
  if (dateRows.length >= 2) {
    const latestTime = dateRows[dateRows.length - 1]._date.getTime();
    const minTime = latestTime - hours * 60 * 60 * 1000;
    const filtered = dateRows.filter(row => row._date.getTime() >= minTime);
    return filtered.length ? filtered : dateRows.slice(-Math.max(6, Math.round(hours * 6)));
  }
  return enriched.slice(-Math.max(6, Math.round(hours * 6)));
}

function makeChartRows(timeline, hours) {
  return rangeFilter(timeline || [], hours).map((row) => ({
    time: timeLabel(row.timestamp),
    fullTime: dateTime(row.timestamp),
    risk: getNum(row.risk_percentage, getNum(row.risk_probability) * 100),
    arm: getNum(row.arm) * 100,
    temp: getNum(row.temperature_c),
    humidity: getNum(row.humidity_pct),
    co2: getNum(row.co2_ppm),
    co2Scaled: getNum(row.co2_ppm) / 100,
    weight: getNum(row.weight_kg),
  }));
}

function DashboardTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="iot-tooltip">
      <strong>{payload[0]?.payload?.fullTime || label}</strong>
      {payload.map((entry) => (
        <p key={entry.dataKey} style={{ color: entry.color }}>
          {entry.name}: {Number(entry.value).toFixed(entry.dataKey === 'risk' ? 3 : 2)}{entry.unit || ''}
        </p>
      ))}
    </div>
  );
}

function OutputStatusCard({ title, icon, children, className = '' }) {
  return (
    <section className={`iot-card ${className}`}>
      <div className="iot-card-title-row">
        <h3>{title}</h3>
        <span className="iot-info-dot"><Info size={14} /></span>
      </div>
      {icon && <div className="iot-card-icon-large">{icon}</div>}
      {children}
    </section>
  );
}

function Forecast24HourCard({
  horizonHours,
  probability,
  previousProbability,
  changePercentagePoints,
  riskLevel,
  armTrend,
  alertState,
  windowStart,
  windowEnd,
  modelName,
  color,
}) {
  const levelClass = String(riskLevel || 'Low').toLowerCase();
  const changeClass = changePercentagePoints > 0
    ? 'increasing'
    : changePercentagePoints < 0
    ? 'decreasing'
    : 'stable';
  const AlertIcon = alertState.className === 'normal' ? ShieldCheck : AlertTriangle;

  return (
    <section
      className={`iot-card iot-forecast-card forecast-${levelClass}`}
      style={{ '--forecast-color': color }}
      aria-live="polite"
    >
      <div className="iot-forecast-header">
        <div className="iot-forecast-title-wrap">
          <span className="iot-forecast-title-icon"><ShieldCheck size={24} /></span>
          <div>
            <p className="iot-forecast-eyebrow">Predictive early-warning output</p>
            <h2>Next {horizonHours}-Hour Absconding Forecast</h2>
            <p>
              Estimated probability that an Absconding event will begin at any point
              during the next {horizonHours} hours.
            </p>
          </div>
        </div>
        <span className={`iot-forecast-risk-badge ${levelClass}`}>
          {displayLevel(riskLevel)} RISK
        </span>
      </div>

      <div className="iot-forecast-body">
        <div className="iot-forecast-probability-panel">
          <span>Predicted risk probability</span>
          <strong style={{ color }}>{probabilityPercent(probability, 2)}</strong>
          <small>Generated by {modelName}</small>
        </div>

        <div className="iot-forecast-stat-panel">
          <span>Previous forecast</span>
          <strong>{probabilityPercent(previousProbability, 2)}</strong>
          <small>Comparison value used for ARM</small>
        </div>

        <div className={`iot-forecast-stat-panel change-${changeClass}`}>
          <span>Probability change</span>
          <strong>{signed(changePercentagePoints, 3, ' pp')}</strong>
          <small>Percentage-point movement</small>
        </div>

        <div className="iot-forecast-stat-panel">
          <span>ARM trend</span>
          <strong>{armTrend}</strong>
          <small>Direction of forecast risk</small>
        </div>

        <div className={`iot-forecast-alert-panel ${alertState.className}`}>
          <AlertIcon size={28} />
          <div>
            <span>Early Warning Alert</span>
            <strong>{alertState.label}</strong>
            <p>{alertState.message}</p>
          </div>
        </div>
      </div>

      <div className="iot-forecast-window">
        <span><strong>Forecast starts:</strong> {dateTime(windowStart)}</span>
        <span className="iot-forecast-window-arrow">→</span>
        <span><strong>Forecast ends:</strong> {dateTime(windowEnd)}</span>
      </div>
    </section>
  );
}

function RangeButtons({ active, onChange, options = RANGE_OPTIONS }) {
  return (
    <div className="iot-range-buttons">
      {options.map((item) => (
        <button
          key={item.label}
          className={active === item.hours ? 'active' : ''}
          onClick={() => onChange(item.hours)}
          type="button"
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function EmptyLiveState({ error, onRetry, loading }) {
  return (
    <div className="iot-dashboard-shell">
      <section className="iot-card iot-empty-state">
        <div className="iot-card-title-row">
          <h3>Live Prediction (IoT) is waiting for backend data</h3>
          <AlertTriangle size={24} color="var(--accent-gold)" />
        </div>
        <p>{error || 'The dashboard is checking /api/absconding/iot/live for the latest Supabase IoT prediction.'}</p>
        <div className="iot-empty-steps">
          <span>1. IoT sensor inserts to Supabase every 10 minutes</span>
          <span>2. Backend monitor pulls real data and saves prediction cache</span>
          <span>3. Dashboard displays probability, risk level, ARM, alert, and insights</span>
        </div>
        <button className="iot-refresh-button" onClick={() => onRetry?.(true)} disabled={loading}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} /> Pull IoT Now
        </button>
      </section>
    </div>
  );
}

export default function AbscondingLiveDashboard({
  iotLiveData,
  iotLiveLoading,
  iotLiveError,
  refetchIotLive,
  dashboardRefreshIntervalMinutes = 10,
  lastIotFetchAt,
  nextIotRefreshAt,
}) {
  const [riskRange, setRiskRange] = useState(6);
  const [sensorRange, setSensorRange] = useState(6);
  const [riskScaleMode, setRiskScaleMode] = useState('zoomed');
  const [showInsights, setShowInsights] = useState(false);
  const riskRows = useMemo(() => makeChartRows(iotLiveData?.timeline || [], riskRange), [iotLiveData?.timeline, riskRange]);
  const sensorRows = useMemo(() => makeChartRows(iotLiveData?.timeline || [], sensorRange), [iotLiveData?.timeline, sensorRange]);
  const zoomedRiskDomain = useMemo(() => adaptiveRiskDomain(riskRows), [riskRows]);
  const armMiniRows = useMemo(() => {
    const rows = riskRows.slice(-8);
    const values = rows.map((row) => row.risk).filter(Number.isFinite);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 0;
    const spread = Math.max(max - min, 0.001);
    return rows.map((row, index) => ({
      ...row,
      left: rows.length <= 1 ? 0 : (index / (rows.length - 1)) * 100,
      top: 78 - ((row.risk - min) / spread) * 42,
    }));
  }, [riskRows]);

  if (iotLiveError && !iotLiveData) {
    return <EmptyLiveState error={iotLiveError} onRetry={refetchIotLive} loading={iotLiveLoading} />;
  }

  if (!iotLiveData) {
    return <EmptyLiveState onRetry={refetchIotLive} loading={iotLiveLoading} />;
  }

  if (iotLiveData.status === 'collecting_history') {
    return (
      <EmptyLiveState
        error={iotLiveData.notification?.message || iotLiveData.reason}
        onRetry={refetchIotLive}
        loading={iotLiveLoading}
      />
    );
  }

  const latest = iotLiveData.latest_sensor_readings || {};
  const notification = iotLiveData.notification || {};
  const monitor = iotLiveData.backend_iot_monitor || {};
  const source = iotLiveData.data_source || {};
  const interval = getNum(iotLiveData.sampling_interval_minutes, 10);
  const lastUpdated = iotLiveData.last_updated;
  const nextExpected = iotLiveData.next_expected_reading || addMinutes(lastUpdated, interval);
  const riskProbability = getNum(iotLiveData.risk_probability, getNum(iotLiveData.risk_percentage) / 100);
  const riskPercentage = clamp(getNum(iotLiveData.risk_percentage, riskProbability * 100));
  const comparisonHours = Math.max(1, getNum(iotLiveData.comparison_hours, 24));
  const responsePreviousProbability = getOptionalNum(iotLiveData.previous_probability);
  const previousProbability = responsePreviousProbability ?? (
    Number.isFinite(riskProbability) && getOptionalNum(iotLiveData.arm) !== null
      ? riskProbability - getNum(iotLiveData.arm, 0)
      : null
  );
  const responsePreviousPercentage = getOptionalNum(iotLiveData.previous_probability_percent);
  const previousPercentage = responsePreviousPercentage
    ?? (previousProbability !== null ? previousProbability * 100 : null);
  const responseChangePercentagePoints = getOptionalNum(
    iotLiveData.probability_change_percentage_points,
  );
  const probabilityChangePercentagePoints = responseChangePercentagePoints
    ?? (previousPercentage !== null ? riskPercentage - previousPercentage : null);
  const riskLevel = iotLiveData.risk_level
    || (riskPercentage > 70 ? 'High' : riskPercentage > 35 ? 'Medium' : 'Low');
  const color = riskColor(riskLevel);
  const arm = getNum(iotLiveData.arm, 0);
  const armPerHour = getNum(iotLiveData.arm_per_hour, arm / comparisonHours);
  const armTrend = iotLiveData.arm_trend || trendFromChange(probabilityChangePercentagePoints);
  const thresholds = iotLiveData.thresholds || {};
  const mediumThreshold = getNum(thresholds.medium_percentage, 35);
  const highThreshold = getNum(thresholds.high_percentage, 70);
  const riskYAxisDomain = riskScaleMode === 'full' ? [0, 100] : zoomedRiskDomain;
  const freshness = iotLiveData.data_freshness_status || (getNum(iotLiveData.data_age_minutes, 0) <= interval * 2 ? 'Fresh' : 'Stale');
  const modelName = iotLiveData.active_model_name || modelNameFromPath(iotLiveData.active_model_path, 'Trained Model');
  const forecastHours = predictionHorizonHours(iotLiveData, 24);
  const forecastWindowStart = iotLiveData.forecast_window_start
    || iotLiveData.prediction_timestamp
    || lastUpdated;
  const forecastWindowEnd = iotLiveData.forecast_window_end
    || addHours(forecastWindowStart, forecastHours);
  const forecastAlert = forecastAlertState(riskLevel, armTrend, notification);

  const sensorStatuses = {
    temperature: statusForSensor('temperature', latest.temperature_c, latest),
    humidity: statusForSensor('humidity', latest.humidity_pct, latest),
    co2: statusForSensor('co2', latest.co2_ppm, latest),
    weight: statusForSensor('weight', latest.weight_kg, latest),
  };

  const factorCards = [
    {
      key: 'weight',
      icon: '📉',
      title: 'Weight Decline',
      impact: getNum(latest.weight_change_6h) < -0.8 || getNum(latest.weight_change_24h) < -1.5 ? 'High Impact' : 'Medium Impact',
      className: getNum(latest.weight_change_6h) < -0.8 || getNum(latest.weight_change_24h) < -1.5 ? 'bad' : 'warn',
      text: `Weight changed ${signed(latest.weight_change_6h, 2, ' kg')} in 6h`,
    },
    {
      key: 'co2',
      icon: '☁️',
      title: 'CO₂ Buildup',
      impact: getNum(latest.co2_ppm) >= 1800 ? 'High Impact' : getNum(latest.co2_ppm) >= 1200 ? 'Medium Impact' : 'Low Impact',
      className: getNum(latest.co2_ppm) >= 1800 ? 'bad' : getNum(latest.co2_ppm) >= 1200 ? 'warn' : 'good',
      text: `CO₂ change ${signed(latest.co2_change_6h, 0, ' ppm')} in 6h`,
    },
    {
      key: 'temp',
      icon: '🌡️',
      title: 'Temperature Instability',
      impact: Math.abs(getNum(latest.temp_deviation_from_35)) >= 2.5 ? 'High Impact' : 'Medium Impact',
      className: Math.abs(getNum(latest.temp_deviation_from_35)) >= 2.5 ? 'bad' : 'warn',
      text: `Deviation from 35°C is ${fmt(latest.temp_deviation_from_35, 2, '°C')}`,
    },
    {
      key: 'humidity',
      icon: '💧',
      title: 'Humidity Deviation',
      impact: Math.abs(getNum(latest.humidity_deviation_from_optimal)) >= 15 ? 'High Impact' : 'Medium Impact',
      className: Math.abs(getNum(latest.humidity_deviation_from_optimal)) >= 15 ? 'bad' : 'warn',
      text: `Humidity deviation ${fmt(latest.humidity_deviation_from_optimal, 2, '%')}`,
    },
  ];

  const actions = [
    notification.should_notify || riskLevel === 'High'
      ? 'Inspect hive within the next 12 hours.'
      : riskLevel === 'Medium'
      ? 'Inspect hive within the next 24 hours if trend keeps increasing.'
      : 'Continue normal inspection schedule and keep monitoring.',
    'Ensure adequate ventilation to reduce CO₂ buildup.',
    'Stabilize hive temperature and reduce fluctuations.',
    'Monitor weight, food stores, queen status, pests, and disturbances closely.',
  ];

  return (
    <div className="iot-dashboard-shell">
      <div className="iot-top-strip">
        <div className="iot-meta-card">
          <span className="iot-meta-icon hive">▦</span>
          <div><p>Hive ID</p><strong>{iotLiveData.hive_id || 'Verification Hive'}</strong></div>
        </div>
        <div className="iot-meta-card">
          <span className="iot-meta-icon">◷</span>
          <div><p>Last Updated</p><strong>{dateTime(lastUpdated)}</strong></div>
        </div>
        <div className="iot-meta-card">
          <span className="iot-meta-icon">▣</span>
          <div><p>Next Expected Reading</p><strong>{dateTime(nextExpected)}</strong></div>
        </div>
        <div className="iot-meta-card">
          <span className="iot-meta-icon shield">✓</span>
          <div><p>Data Freshness</p><strong className={`freshness-${freshnessClass(freshness)}`}>{freshnessText(freshness)} <span className="iot-live-dot" /></strong></div>
        </div>
      </div>

      <Forecast24HourCard
        horizonHours={forecastHours}
        probability={riskPercentage}
        previousProbability={previousPercentage}
        changePercentagePoints={probabilityChangePercentagePoints}
        riskLevel={riskLevel}
        armTrend={armTrend}
        alertState={forecastAlert}
        windowStart={forecastWindowStart}
        windowEnd={forecastWindowEnd}
        modelName={modelName}
        color={color}
      />

      <div className="iot-kpi-grid">
        <OutputStatusCard title="Absconding Probability" className="iot-probability-card">
          <div className="iot-gauge-wrap" style={{ '--risk-color': color, '--risk-deg': `${riskPercentage * 1.8}deg` }}>
            <div className="iot-gauge-arc" />
            <div className="iot-gauge-needle" style={{ transform: `rotate(${riskPercentage * 1.8 - 90}deg)` }} />
            <div className="iot-gauge-value">
              <strong style={{ color }}>{probabilityPercent(riskPercentage, 2)}</strong>
              <span>Current probability</span>
            </div>
          </div>
          <div className="iot-mini-message" style={{ display: 'grid', gap: '0.35rem', textAlign: 'left' }}>
            <span style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
              <span>Previous ({comparisonHours}h earlier)</span>
              <strong>{probabilityPercent(previousPercentage, 2)}</strong>
            </span>
            <span style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
              <span>Change</span>
              <strong style={{ color: probabilityChangePercentagePoints > 0 ? 'var(--accent-gold)' : probabilityChangePercentagePoints < 0 ? 'var(--accent-emerald)' : 'inherit' }}>
                {signed(probabilityChangePercentagePoints, 3, ' pp')}
              </strong>
            </span>
          </div>
          <div className="iot-model-pill">Model: {modelName} <CheckCircle size={14} /></div>
        </OutputStatusCard>

        <OutputStatusCard title="Risk Level">
          <div className="iot-risk-level" style={{ color }}>{displayLevel(riskLevel)}</div>
          <p className="iot-risk-caption">{riskLevel === 'High' ? 'Urgent Risk' : riskLevel === 'Medium' ? 'Elevated Risk' : 'Normal Risk'}</p>
          <div className="iot-risk-scale"><span>LOW</span><span>MEDIUM</span><span>HIGH</span><b style={{ left: `${riskPercentage}%` }} /></div>
          <p className="iot-mini-message">
            {riskLevel === 'Low' ? 'Conditions are currently stable.' : 'Conditions indicate increased absconding likelihood. Monitor closely.'}
            <br />
            <small>Medium: {probabilityPercent(mediumThreshold, 2)} · High: {probabilityPercent(highThreshold, 2)}</small>
          </p>
        </OutputStatusCard>

        <OutputStatusCard title="ARM Trend Behavior">
          <div className="iot-mini-arm-chart">
            {armMiniRows.map((row, index) => (
              <span
                key={`${row.time}-${index}`}
                style={{ left: `${row.left}%`, top: `${row.top}%` }}
              />
            ))}
            <TrendingUp className="iot-arm-trend-icon" size={28} />
          </div>
          <p className="iot-trend-text">Trend: {armTrend}</p>
          <p className="iot-trend-sub">
            {signed(probabilityChangePercentagePoints, 3, ' pp')} over {comparisonHours}h
          </p>
          <p className="iot-trend-sub">
            {signed(armPerHour * 100, 4, ' pp/hour')}
          </p>
        </OutputStatusCard>

        <OutputStatusCard title="Early Warning Alert" className="iot-alert-status-card">
          <div className="iot-warning-symbol"><AlertTriangle size={82} /></div>
          <h4 style={{ color }}>{notification.should_notify || riskLevel !== 'Low' ? (riskLevel === 'High' ? 'HIGH RISK' : 'ELEVATED RISK') : 'NORMAL'}</h4>
          <p>{riskLevel === 'High' ? 'Act immediately' : riskLevel === 'Medium' ? 'Watch closely' : 'Continue monitoring'}</p>
          <div className="iot-mini-message">{notification.should_notify ? 'Alert threshold reached or ARM is increasing quickly.' : 'Risk crossing HIGH threshold is monitored continuously.'}</div>
        </OutputStatusCard>

        <OutputStatusCard title="Explainable Environmental Insights">
          <p className="iot-insight-text">
            {factorCards.filter(f => f.className !== 'good').length > 0
              ? `${factorCards.filter(f => f.className !== 'good').map(f => f.title.toLowerCase()).slice(0, 3).join(', ')} are key indicators increasing absconding risk.`
              : 'Current environmental indicators are not showing severe absconding triggers.'}
          </p>
          <button type="button" className="iot-secondary-button" onClick={() => setShowInsights(!showInsights)}>
            {showInsights ? 'Hide Full Insights' : 'View Full Insights'} →
          </button>
        </OutputStatusCard>
      </div>

      <div className="iot-chart-grid">
        <section className="iot-card iot-risk-chart-card">
          <div className="iot-card-title-row">
            <h3>Absconding Risk Timeline</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div className="iot-range-buttons">
                <button type="button" className={riskScaleMode === 'zoomed' ? 'active' : ''} onClick={() => setRiskScaleMode('zoomed')}>Zoomed</button>
                <button type="button" className={riskScaleMode === 'full' ? 'active' : ''} onClick={() => setRiskScaleMode('full')}>0–100%</button>
              </div>
              <span className="iot-info-dot"><Info size={14} /></span>
            </div>
          </div>
          <div className="iot-chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={riskRows} margin={{ top: 18, right: 14, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="iotRiskGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-emerald)" stopOpacity={0.45} />
                    <stop offset="95%" stopColor="var(--accent-emerald)" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                <XAxis dataKey="time" stroke="var(--text-secondary)" tick={{ fontSize: 11 }} minTickGap={18} />
                <YAxis
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11 }}
                  domain={riskYAxisDomain}
                  tickFormatter={(value) => `${Number(value).toFixed(riskScaleMode === 'zoomed' ? 2 : 0)}%`}
                  allowDataOverflow={false}
                />
                <Tooltip content={<DashboardTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {riskScaleMode === 'full' && (
                  <>
                    <ReferenceLine y={mediumThreshold} stroke="var(--accent-gold)" strokeDasharray="4 4" label={{ value: `Medium threshold (${mediumThreshold.toFixed(2)}%)`, fill: 'var(--accent-gold)', fontSize: 11 }} />
                    <ReferenceLine y={highThreshold} stroke="var(--accent-crimson)" strokeDasharray="4 4" label={{ value: `High threshold (${highThreshold.toFixed(2)}%)`, fill: 'var(--accent-crimson)', fontSize: 11 }} />
                  </>
                )}
                <Area type="monotone" dataKey="risk" name="Absconding Probability" unit="%" stroke="var(--accent-emerald)" fill="url(#iotRiskGradient)" strokeWidth={2.5} dot={{ r: 2.5 }} activeDot={{ r: 5 }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <RangeButtons active={riskRange} onChange={setRiskRange} />
        </section>

        <section className="iot-card iot-sensor-chart-card">
          <div className="iot-card-title-row"><h3>Sensor Trend (Live)</h3><span className="iot-info-dot"><Info size={14} /></span></div>
          <div className="iot-chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sensorRows} margin={{ top: 18, right: 20, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                <XAxis dataKey="time" stroke="var(--text-secondary)" tick={{ fontSize: 11 }} minTickGap={16} />
                <YAxis yAxisId="left" stroke="var(--text-secondary)" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" stroke="var(--accent-gold)" tick={{ fontSize: 11 }} />
                <Tooltip content={<DashboardTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line yAxisId="left" type="monotone" dataKey="temp" name="Temp (°C)" stroke="#ff4d4d" strokeWidth={2} dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="humidity" name="Humidity (%)" stroke="#4aa3ff" strokeWidth={2} dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="co2" name="CO₂ (ppm)" stroke="var(--accent-gold)" strokeWidth={2} dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="weight" name="Weight (kg)" stroke="var(--accent-emerald)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <RangeButtons active={sensorRange} onChange={setSensorRange} options={RANGE_OPTIONS.slice(0, 4)} />
        </section>

        <section className="iot-card iot-readings-card">
          <div className="iot-card-title-row"><h3>Latest IoT Readings</h3><span className="iot-info-dot"><Info size={14} /></span></div>
          <div className="iot-reading-list">
            <div className="iot-reading-row"><span className="iot-reading-icon temp">♨</span><div><strong>Temperature</strong></div><b className="temp-value">{fmt(latest.temperature_c, 1, ' °C')}</b><em className={sensorStatuses.temperature.className}>Status<br />{sensorStatuses.temperature.label}</em></div>
            <div className="iot-reading-row"><span className="iot-reading-icon humidity">◖</span><div><strong>Humidity</strong></div><b className="humidity-value">{fmt(latest.humidity_pct, 1, ' %')}</b><em className={sensorStatuses.humidity.className}>Status<br />{sensorStatuses.humidity.label}</em></div>
            <div className="iot-reading-row"><span className="iot-reading-icon co2">☁</span><div><strong>CO₂ Level</strong></div><b className="co2-value">{fmt(latest.co2_ppm, 0, ' ppm')}</b><em className={sensorStatuses.co2.className}>Status<br />{sensorStatuses.co2.label}</em></div>
            <div className="iot-reading-row"><span className="iot-reading-icon weight">▣</span><div><strong>Weight</strong></div><b className="weight-value">{fmt(latest.weight_kg, 2, ' kg')}</b><em className={sensorStatuses.weight.className}>Status<br />{sensorStatuses.weight.label}</em></div>
            {latest.external_temperature_c !== null && latest.external_temperature_c !== undefined && (
              <div className="iot-reading-row"><span className="iot-reading-icon temp">↗</span><div><strong>External Temperature</strong></div><b className="temp-value">{fmt(latest.external_temperature_c, 1, ' °C')}</b><em className="neutral">Context<br />Outside</em></div>
            )}
            {latest.external_humidity_pct !== null && latest.external_humidity_pct !== undefined && (
              <div className="iot-reading-row"><span className="iot-reading-icon humidity">↗</span><div><strong>External Humidity</strong></div><b className="humidity-value">{fmt(latest.external_humidity_pct, 1, ' %')}</b><em className="neutral">Context<br />Outside</em></div>
            )}
            {latest.battery_voltage !== null && latest.battery_voltage !== undefined && (
              <div className="iot-reading-row"><span className="iot-reading-icon weight">⚡</span><div><strong>Battery Voltage</strong></div><b className="weight-value">{fmt(latest.battery_voltage, 2, ' V')}</b><em className={getNum(latest.battery_voltage) < 3.4 ? 'warn' : 'good'}>Device<br />{getNum(latest.battery_voltage) < 3.4 ? 'Low' : 'Normal'}</em></div>
            )}
          </div>
        </section>
      </div>

      {showInsights && (
        <section className="iot-card iot-full-insights-card">
          <div className="iot-card-title-row"><h3>Full Live Prediction Diagnostics</h3><span className="iot-info-dot"><Info size={14} /></span></div>
          <div className="iot-diagnostics-grid">
            <span><strong>Prediction window</strong>{iotLiveData.prediction_window || 'next_24_hours'}</span>
            <span><strong>API delivery mode</strong>{iotLiveData.api_delivery_mode || 'backend_cached_real_iot'}</span>
            <span><strong>Backend poller</strong>{monitor.enabled ? `Running every ${monitor.interval_minutes || interval} min` : 'Not enabled'}</span>
            <span><strong>Last DB pull</strong>{dateTime(monitor.last_success_at || monitor.last_poll_finished_at)}</span>
            <span><strong>Next DB pull</strong>{dateTime(monitor.next_poll_at || nextIotRefreshAt)}</span>
            <span><strong>Data source</strong>{source.source || 'supabase_postgres'}</span>
            <span><strong>Schema/table</strong>{source.schema || 'public'}.{source.table || source.path || 'beehive_readings'}</span>
            <span><strong>Records used</strong>{iotLiveData.records_used_for_prediction || 0} / expected 24h {iotLiveData.expected_records_for_24h || 144}</span>
            <span><strong>Last dashboard refresh</strong>{dateTime(lastIotFetchAt)}</span>
            <span><strong>Next dashboard refresh</strong>{dateTime(nextIotRefreshAt)}</span>
          </div>
        </section>
      )}

      <div className="iot-bottom-grid">
        <section className="iot-card iot-factors-card">
          <div className="iot-card-title-row"><h3>Key Contributing Factors</h3><span className="iot-info-dot"><Info size={14} /></span></div>
          <div className="iot-factor-grid">
            {factorCards.map((factor) => (
              <div className={`iot-factor-card ${factor.className}`} key={factor.key}>
                <span>{factor.icon}</span>
                <h4>{factor.title}</h4>
                <b>{factor.impact}</b>
                <p>{factor.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="iot-card iot-actions-card">
          <div className="iot-card-title-row"><h3>Recommended Action</h3><span className="iot-info-dot"><Info size={14} /></span></div>
          <div className="iot-action-content">
            <div className="iot-action-list">
              {actions.map((item) => (
                <div key={item}><CheckCircle size={18} /> <span>{item}</span></div>
              ))}
            </div>
            <div className="iot-shield-illustration">
              <ShieldCheck size={128} />
            </div>
          </div>
        </section>
      </div>

      <div className="iot-footer-note">
        <Info size={15} /> Predictions are based on live sensor data and machine learning models. Use insights to support, not replace, expert judgement.
        <button className="iot-refresh-mini" onClick={() => refetchIotLive?.(true)} disabled={iotLiveLoading}>
          <RefreshCw size={13} className={iotLiveLoading ? 'spin' : ''} /> Pull IoT Now
        </button>
      </div>
    </div>
  );
}