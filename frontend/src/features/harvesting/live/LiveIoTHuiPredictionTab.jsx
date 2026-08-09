import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BatteryCharging,
  CheckCircle2,
  Clock3,
  CloudSun,
  Database,
  Droplets,
  Gauge,
  Leaf,
  RefreshCw,
  Scale,
  ShieldCheck,
  Thermometer,
  TrendingDown,
  TrendingUp,
  Wind,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  loadLiveHuiHistory,
  loadLiveHuiPrediction,
  loadLiveHuiStatus,
  loadLiveSensorSnapshot,
  refreshLiveHuiPrediction,
} from "../../../services/classifierDerivedHuiService";

import "./LiveIoTHuiPredictionTab.css";

const CURRENT_REQUIRED_HOURS = 168;
const FUTURE_REQUIRED_HOURS = 192;

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatNumber(value, digits = 1) {
  const parsed = asNumber(value);
  return parsed === null ? "—" : parsed.toFixed(digits);
}

function formatTime(value) {
  if (!value) return "—";

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);

  return parsed.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSavedTime(value) {
  if (!value) return "—";

  const parsed = new Date(value);

  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }

  return parsed.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function selectHive(items, hiveId) {
  return (items ?? []).find((item) => item.hive_id === hiveId) ?? null;
}

function readinessTone(readinessClass) {
  if (readinessClass === "Approaching Harvest") return "approaching";
  if (readinessClass === "Ready") return "ready";
  if (readinessClass === "High-Priority Harvest") return "high";
  return "not-ready";
}

function getTrendIcon(value) {
  if (value === "Increasing") return TrendingUp;
  if (value === "Decreasing") return TrendingDown;
  return Activity;
}

function getCleanRecommendation(latest) {
  if (!latest) {
    return {
      title: "Collecting model history",
      text: "Live sensor monitoring is active. HUI values will appear automatically when the input window is ready.",
    };
  }

  const current = asNumber(latest.current_hui);
  const future = [
    { hour: 24, value: asNumber(latest.predicted_hui_24h) },
    { hour: 48, value: asNumber(latest.predicted_hui_48h) },
    { hour: 72, value: asNumber(latest.predicted_hui_72h) },
  ].filter((item) => item.value !== null);

  if (current !== null && current >= 80) {
    return {
      title: "Inspect immediately",
      text: "Current HUI is in the high-priority harvest range.",
    };
  }

  if (current !== null && current >= 60) {
    return {
      title: "Inspect for harvest readiness",
      text: "Current HUI is in the Ready range.",
    };
  }

  const readyForecast = future.find((item) => item.value >= 60);
  if (readyForecast) {
    return {
      title: `Plan inspection within ${readyForecast.hour} hours`,
      text: "The forecast reaches the Ready range.",
    };
  }

  const approachingForecast = future.find((item) => item.value >= 40);
  if (approachingForecast) {
    return {
      title: "Continue close monitoring",
      text: `The HUI is projected to enter the Approaching Harvest range within ${approachingForecast.hour} hours.`,
    };
  }

  return {
    title: "Continue routine monitoring",
    text: "Current and forecast HUI remain below the approaching-harvest threshold.",
  };
}

function StatusPill({ icon: Icon, label, value, tone = "neutral" }) {
  return (
    <div className={`live-pill live-pill-${tone}`}>
      <Icon size={16} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SensorTile({ icon: Icon, label, value, suffix, subtext }) {
  return (
    <article className="live-sensor-tile">
      <span className="live-sensor-icon">
        <Icon size={20} />
      </span>

      <div>
        <small>{label}</small>
        <strong>
          {formatNumber(value, 1)}
          {asNumber(value) !== null ? suffix : ""}
        </strong>
        {subtext ? <span>{subtext}</span> : null}
      </div>
    </article>
  );
}

function ForecastTile({ hour, value, readinessClass, model }) {
  return (
    <article
      className={`live-forecast-tile is-${readinessTone(readinessClass)}`}
    >
      <div className="live-forecast-topline">
        <span>+{hour} HOURS</span>
        <small>{model}</small>
      </div>

      <strong>{formatNumber(value, 1)}</strong>
      <b>{readinessClass ?? "Pending"}</b>
    </article>
  );
}

function HuiGauge({ value, readinessClass }) {
  const numeric = asNumber(value);
  const bounded =
    numeric === null ? 0 : Math.min(Math.max(numeric, 0), 100);

  return (
    <div
      className={`live-hui-ring is-${readinessTone(readinessClass)}`}
      style={{ "--hui-angle": `${bounded * 3.6}deg` }}
    >
      <div className="live-hui-ring-inner">
        <strong>
          {numeric === null ? "—" : formatNumber(bounded, 1)}
        </strong>
        <span>HUI / 100</span>
      </div>
    </div>
  );
}

function HistoryProgress({ label, value, maximum }) {
  const amount = Math.min(Math.max(Number(value) || 0, 0), maximum);
  const percentage = (amount / maximum) * 100;

  return (
    <div className="live-history-progress">
      <div className="live-history-progress-heading">
        <span>{label}</span>
        <strong>
          {amount}/{maximum}h
        </strong>
      </div>

      <div className="live-history-progress-track">
        <span style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

export default function LiveIoTHuiPredictionTab() {
  const [prediction, setPrediction] = useState(null);
  const [sensors, setSensors] = useState(null);
  const [monitor, setMonitor] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedHive, setSelectedHive] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const refreshMilliseconds = Number(
    import.meta.env.VITE_IOT_REFRESH_MS ?? 60000,
  );

  const loadDashboard = useCallback(
    async ({ force = false } = {}) => {
      force ? setRefreshing(true) : setLoading(true);
      setError("");

      try {
       const [
      predictionResult,
      sensorResult,
      monitorResult,
      historyResult,
   ] = await Promise.allSettled([
  force
    ? refreshLiveHuiPrediction(selectedHive)
    : loadLiveHuiPrediction({ hiveId: selectedHive }),
  loadLiveSensorSnapshot(selectedHive),
  loadLiveHuiStatus(),
  loadLiveHuiHistory(selectedHive, 100),
]);

        if (predictionResult.status === "fulfilled") {
  setPrediction(predictionResult.value);
} else {
  setPrediction(null);
  setError(
    predictionResult.reason?.message ??
      "Unable to load live HUI data.",
  );
}

if (sensorResult.status === "fulfilled") {
  setSensors(sensorResult.value);
}

if (monitorResult.status === "fulfilled") {
  setMonitor(monitorResult.value);
} else {
  setMonitor(null);
}

if (historyResult.status === "fulfilled") {
  setHistory(historyResult.value?.history ?? []);
}

        if (sensorResult.status === "fulfilled") {
          setSensors(sensorResult.value);
        }

        if (historyResult.status === "fulfilled") {
        setHistory(historyResult.value?.history ?? []);
       }
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [selectedHive],
  );

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  useEffect(() => {
    if (
      !Number.isFinite(refreshMilliseconds) ||
      refreshMilliseconds < 10000
    ) {
      return undefined;
    }

    const interval = window.setInterval(
      loadDashboard,
      refreshMilliseconds,
    );

    return () => window.clearInterval(interval);
  }, [loadDashboard, refreshMilliseconds]);

  const hives = useMemo(() => {
    const diagnosticHives = (prediction?.hive_diagnostics ?? [])
      .map((item) => item?.hive_id)
      .filter(Boolean);

    return [
      ...new Set([
        ...(prediction?.available_hives ?? []),
        ...(sensors?.available_hives ?? []),
        ...diagnosticHives,
      ]),
    ].sort();
  }, [prediction, sensors]);

  useEffect(() => {
    if (!selectedHive && hives.length > 0) {
      setSelectedHive(hives[0]);
    }
  }, [hives, selectedHive]);

  const latest = useMemo(
    () => selectHive(prediction?.latest_by_hive, selectedHive),
    [prediction, selectedHive],
  );

  const sensor = useMemo(
    () =>
      selectHive(sensors?.latest_sensor_by_hive, selectedHive) ??
      latest?.sensor_status ??
      null,
    [sensors, latest, selectedHive],
  );

  const diagnostic = useMemo(() => {
    const diagnostics = prediction?.hive_diagnostics ?? [];

    return (
      selectHive(diagnostics, selectedHive) ??
      diagnostics[0] ??
      null
    );
  }, [prediction, selectedHive]);

  const contiguousHours =
    diagnostic?.latest_contiguous_hourly_rows ??
    diagnostic?.contiguous_hourly_rows ??
    diagnostic?.latest_contiguous_rows ??
    0;

  const predictionReady = Boolean(latest);
  const imputationActive = Boolean(latest?.imputation_applied);
  const imputedHours = Number(latest?.imputed_hourly_rows ?? 0);
  const domainShift = (sensors?.domain_warnings ?? []).length > 0;

  const freshnessMinutes = asNumber(sensor?.freshness_minutes);
  const isFresh =
    sensor?.freshness_label === "Fresh" ||
    (freshnessMinutes !== null && freshnessMinutes <= 30);

  const TrendIcon = getTrendIcon(latest?.rate_of_change);
  const recommendation = getCleanRecommendation(latest);

  const trajectory = predictionReady
    ? [
        { label: "Now", hui: latest.current_hui },
        { label: "+24h", hui: latest.predicted_hui_24h },
        { label: "+48h", hui: latest.predicted_hui_48h },
        { label: "+72h", hui: latest.predicted_hui_72h },
      ]
    : [];
  const historyRows = history ?? [];

  if (loading && !prediction && !sensors) {
    return (
      <div className="live-dashboard-loading">
        <RefreshCw className="is-spinning" size={22} />
        Loading live hive data…
      </div>
    );
  }

  return (
    <section className="live-hui-dashboard">
      <header className="live-dashboard-header">
        <div>
          <span className="live-eyebrow">
            LIVE IOT HARVEST INTELLIGENCE
          </span>
          <h2>Harvest Urgency Dashboard</h2>
          <p>Current hive state and 72-hour HUI outlook.</p>
        </div>

        <div className="live-dashboard-actions">
          <select
            value={selectedHive}
            onChange={(event) =>
              setSelectedHive(event.target.value)
            }
            aria-label="Hive"
          >
            {hives.map((hiveId) => (
              <option key={hiveId} value={hiveId}>
                {hiveId}
              </option>
            ))}
          </select>

          <button
            type="button"
            disabled={refreshing}
            onClick={() => loadDashboard({ force: true })}
          >
            <RefreshCw
              size={17}
              className={refreshing ? "is-spinning" : ""}
            />
            {refreshing ? "Refreshing" : "Refresh"}
          </button>
        </div>
      </header>

      <div className="live-quick-status">
        <StatusPill
          icon={Database}
          label="Hive"
          value={selectedHive || "—"}
          tone="blue"
        />
        <StatusPill
          icon={isFresh ? CheckCircle2 : AlertTriangle}
          label="Data"
          value={isFresh ? "Live" : "Review"}
          tone={isFresh ? "green" : "amber"}
        />
        <StatusPill
          icon={Clock3}
          label="Updated"
          value={formatTime(
            sensor?.timestamp ?? latest?.source_timestamp_utc,
          )}
        />
        <StatusPill
          icon={Activity}
          label="Monitor"
          value={monitor?.thread_running ? "Running" : "Offline"}
          tone={monitor?.thread_running ? "green" : "amber"}
        />
      </div>

      {error ? (
        <div className="live-compact-alert is-error">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      <div className="live-main-grid">
        <article className="live-main-card live-current-card">
          <div className="live-card-label">
            <Gauge size={18} />
            Current HUI
          </div>

          <div className="live-current-layout">
            <HuiGauge
              value={latest?.current_hui}
              readinessClass={latest?.current_class}
            />

            <div className="live-current-copy">
              <span>Current status</span>
              <h3>
                {latest?.current_class ?? "Collecting history"}
              </h3>
              <p>
                {predictionReady
                  ? `Trend: ${latest?.rate_of_change ?? "Stable"}`
                  : "Waiting for model-ready hourly history."}
              </p>
            </div>
          </div>

          {predictionReady ? (
            <div className="live-current-metrics">
              <div>
                <small>Stability</small>
                <strong>{formatNumber(latest?.hrsi, 1)}</strong>
                <span>
                  {latest?.hrsi_interpretation ?? "—"}
                </span>
              </div>

              <div>
                <small>Trend</small>
                <strong className="live-inline-icon">
                  <TrendIcon size={16} />
                  {latest?.rate_of_change ?? "—"}
                </strong>
                <span>
                  {formatNumber(
                    latest?.rate_of_change_points_per_hour,
                    2,
                  )}{" "}
                  pts/h
                </span>
              </div>

              <div>
                <small>Confidence</small>
                <strong>
                  {latest?.prediction_confidence ?? "—"}
                </strong>
                <span>
                  {formatNumber(
                    latest?.confidence_score,
                    1,
                  )}
                  /100
                </span>
              </div>
            </div>
          ) : null}
        </article>

        <article className="live-main-card live-forecast-card">
          <div className="live-card-header-row">
            <div className="live-card-label">
              <TrendingUp size={18} />
              72-hour HUI outlook
            </div>

            <span
              className={`live-badge ${
                imputationActive ? "is-purple" : "is-green"
              }`}
            >
              {imputationActive
                ? "Imputed input"
                : "Observed input"}
            </span>
          </div>

          <div className="live-forecast-grid">
            <ForecastTile
              hour={24}
              value={latest?.predicted_hui_24h}
              readinessClass={latest?.predicted_class_24h}
              model="LightGBM"
            />
            <ForecastTile
              hour={48}
              value={latest?.predicted_hui_48h}
              readinessClass={latest?.predicted_class_48h}
              model="XGBoost"
            />
            <ForecastTile
              hour={72}
              value={latest?.predicted_hui_72h}
              readinessClass={latest?.predicted_class_72h}
              model="LightGBM"
            />
          </div>

          {predictionReady ? (
            <div className="live-forecast-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={trajectory}
                  margin={{
                    top: 8,
                    right: 12,
                    left: -16,
                    bottom: 0,
                  }}
                >
                  <ReferenceArea
                    y1={0}
                    y2={40}
                    fill="#f8fafc"
                  />
                  <ReferenceArea
                    y1={40}
                    y2={60}
                    fill="#fffbeb"
                  />
                  <ReferenceArea
                    y1={60}
                    y2={80}
                    fill="#f0fdf4"
                  />
                  <ReferenceArea
                    y1={80}
                    y2={100}
                    fill="#fef2f2"
                  />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip
                    formatter={(value) => [
                      formatNumber(value, 1),
                      "HUI",
                    ]}
                  />
                  <ReferenceLine
                    y={40}
                    strokeDasharray="4 4"
                  />
                  <ReferenceLine
                    y={60}
                    strokeDasharray="4 4"
                  />
                  <ReferenceLine
                    y={80}
                    strokeDasharray="4 4"
                  />
                  <Line
                    type="monotone"
                    dataKey="hui"
                    stroke="#2563eb"
                    strokeWidth={3}
                    dot={{
                      r: 5,
                      fill: "#ffffff",
                      strokeWidth: 3,
                    }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="live-history-box">
              <HistoryProgress
                label="Current HUI"
                value={contiguousHours}
                maximum={CURRENT_REQUIRED_HOURS}
              />
              <HistoryProgress
                label="Future HUI"
                value={contiguousHours}
                maximum={FUTURE_REQUIRED_HOURS}
              />
            </div>
          )}
        </article>

        <article className="live-main-card live-action-card">
          <div className="live-card-label">
            <Leaf size={18} />
            Recommendation
          </div>

          <h3>{recommendation.title}</h3>
          <p>{recommendation.text}</p>

          <div className="live-action-footer">
            <ShieldCheck size={16} />
            Verify by physical hive inspection.
          </div>
        </article>
      </div>

      <article className="live-sensors-panel">
        <div className="live-panel-heading">
          <div>
            <span className="live-eyebrow">
              CURRENT VARIABLES
            </span>
            <h3>Latest IoT readings</h3>
          </div>

          <span
            className={`live-live-indicator ${
              isFresh ? "is-live" : ""
            }`}
          >
            <i />
            {isFresh ? "Live" : "Review"}
          </span>
        </div>

        <div className="live-sensor-grid">
          <SensorTile
            icon={Thermometer}
            label="Internal temp."
            value={sensor?.internal_temperature_c}
            suffix=" °C"
          />
          <SensorTile
            icon={Droplets}
            label="Internal humidity"
            value={sensor?.internal_humidity_pct}
            suffix="%"
          />
          <SensorTile
            icon={Wind}
            label="CO₂"
            value={sensor?.co2_ppm}
            suffix=" ppm"
          />
          <SensorTile
            icon={Scale}
            label="Hive weight"
            value={sensor?.weight_kg}
            suffix=" kg"
          />
          <SensorTile
            icon={CloudSun}
            label="External temp."
            value={sensor?.external_temperature_c}
            suffix=" °C"
          />
          <SensorTile
            icon={Droplets}
            label="External humidity"
            value={sensor?.external_humidity_pct}
            suffix="%"
          />
          <SensorTile
            icon={BatteryCharging}
            label="Battery"
            value={sensor?.battery_voltage}
            suffix=" V"
          />
          <SensorTile
            icon={Clock3}
            label="Freshness"
            value={sensor?.freshness_minutes}
            suffix=" min"
            subtext={sensor?.freshness_label}
          />
        </div>
      </article>

      {(imputationActive || domainShift) ? (
        <div className="live-quality-strip">
          <details className="live-saved-history">
  <summary className="live-saved-history-summary">
    <div>
      <strong>Saved Live Prediction History</strong>
      <span>
        {historyRows.length} model-ready records
      </span>
    </div>
  </summary>

  <div className="live-saved-history-body">
    {historyRows.length > 0 ? (
      <div className="live-history-table-wrap">
        <table className="live-history-table">
          <thead>
            <tr>
              <th>Saved At</th>
              <th>Current HUI</th>
              <th>Status</th>
              <th>+24h</th>
              <th>+48h</th>
              <th>+72h</th>
              <th>HRSI</th>
              <th>Trend</th>
              <th>Confidence</th>
              <th>Recommendation</th>
            </tr>
          </thead>

          <tbody>
            {historyRows.map((row, index) => (
            <tr key={`${row.hive_id}-${row.timestamp}-${row.saved_at_utc ?? "legacy"}-${index}`}>
                <td>{formatSavedTime(row.saved_at_utc)}</td>

                <td>
                  <strong>
                    {formatNumber(row.current_hui, 1)}
                  </strong>
                </td>

                <td>
                  <span
                    className={`live-history-status is-${readinessTone(
                      row.current_class,
                    )}`}
                  >
                    {row.current_class ?? "—"}
                  </span>
                </td>

                <td>
                  {formatNumber(
                    row.predicted_hui_24h,
                    1,
                  )}
                </td>

                <td>
                  {formatNumber(
                    row.predicted_hui_48h,
                    1,
                  )}
                </td>

                <td>
                  {formatNumber(
                    row.predicted_hui_72h,
                    1,
                  )}
                </td>

                <td>
                  {formatNumber(row.hrsi, 1)}
                </td>

                <td>
                  {row.rate_of_change ?? "—"}
                </td>

                <td>
                  <strong>
                    {row.prediction_confidence ?? "—"}
                  </strong>
                  <small>
                    {formatNumber(
                      row.confidence_score,
                      1,
                    )}
                    /100
                  </small>
                </td>

                <td>
                  {row.recommended_window ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ) : (
      <div className="live-history-empty">
        No saved live predictions are available yet.
      </div>
    )}
  </div>
</details>
</div>
          
      ) : null}
    </section>
  );
}
