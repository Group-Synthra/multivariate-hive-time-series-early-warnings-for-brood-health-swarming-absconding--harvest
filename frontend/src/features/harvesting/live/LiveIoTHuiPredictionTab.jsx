import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BatteryCharging,
  CheckCircle2,
  Clock3,
  CloudSun,
  Database,
  Download,
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

function HuiMeter({
  value,
  readinessClass,
  title,
  subtitle,
  model,
  large = false,
}) {
  const numeric = asNumber(value);

  const bounded = numeric === null ? 0 : Math.min(Math.max(numeric, 0), 100);

  return (
    <div
      className={`live-ux-hui-meter ${
        large ? "is-large" : ""
      } is-${readinessTone(readinessClass)}`}
    >
      <div className="live-ux-meter-heading">
        <div>
          <span className="live-ux-meter-title">{title}</span>

          {subtitle ? <small>{subtitle}</small> : null}
        </div>

        {model ? <span className="live-ux-model-badge">{model}</span> : null}
      </div>

      <div className="live-ux-meter-value-row">
        <strong className="live-ux-meter-value">
          {numeric === null ? "—" : formatNumber(bounded, 1)}
        </strong>

        <div>
          <span className="live-ux-meter-unit">HUI / 100</span>

          <b className="live-ux-readiness-label">
            {readinessClass ?? "Pending"}
          </b>
        </div>
      </div>

      <div className="live-ux-scale">
        <div className="live-ux-scale-track">
          <span className="segment not-ready" />
          <span className="segment approaching" />
          <span className="segment ready" />
          <span className="segment priority" />

          {numeric !== null ? (
            <span
              className="live-ux-scale-marker"
              style={{
                left: `${bounded}%`,
              }}
            >
              <i />
            </span>
          ) : null}
        </div>

        <div className="live-ux-scale-numbers">
          <span>0</span>
          <span>40</span>
          <span>60</span>
          <span>80</span>
          <span>100</span>
        </div>
      </div>

      {large ? (
        <div className="live-ux-scale-legend">
          <span className="not-ready">
            <i />
            Not Ready
          </span>

          <span className="approaching">
            <i />
            Approaching
          </span>

          <span className="ready">
            <i />
            Ready
          </span>

          <span className="priority">
            <i />
            High Priority
          </span>
        </div>
      ) : null}
    </div>
  );
}

function HealthMetric({
  label,
  value,
  suffix = "",
  description,
  icon: Icon,
  progress,
}) {
  const numericProgress =
    progress === null || progress === undefined
      ? null
      : Math.min(Math.max(Number(progress) || 0, 0), 100);

  return (
    <article className="live-ux-health-card">
      <div className="live-ux-health-icon">
        <Icon size={19} />
      </div>

      <div className="live-ux-health-content">
        <span>{label}</span>

        <strong>
          {value}
          {suffix}
        </strong>

        <small>{description}</small>

        {numericProgress !== null ? (
          <div className="live-ux-health-progress">
            <span
              style={{
                width: `${numericProgress}%`,
              }}
            />
          </div>
        ) : null}
      </div>
    </article>
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
const HOUR_MS = 60 * 60 * 1000;

/*
  A future HUI does not always have a record at exactly
  +24h / +48h / +72h.

  Allow the nearest stored HUI within 90 minutes.
*/
const FUTURE_MATCH_TOLERANCE_MS = 90 * 60 * 1000;

function getHistoryTimestampMs(row) {
  const value =
    row?.source_timestamp_utc ??
    row?.timestamp ??
    row?.prediction_timestamp ??
    row?.saved_at ??
    row?.created_at ??
    null;

  if (!value) {
    return null;
  }

  const parsed = new Date(value);

  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return parsed.getTime();
}

function csvValue(value) {
  if (value === null || value === undefined) {
    return "";
  }

  const stringValue = String(value);

  return `"${stringValue.replace(/"/g, '""')}"`;
}

function csvNumber(value, digits = 3) {
  const numeric = asNumber(value);

  return numeric === null ? "" : numeric.toFixed(digits);
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
        const [predictionResult, sensorResult, monitorResult, historyResult] =
          await Promise.allSettled([
            force
              ? refreshLiveHuiPrediction(selectedHive)
              : loadLiveHuiPrediction({ hiveId: selectedHive }),
            loadLiveSensorSnapshot(selectedHive),
            loadLiveHuiStatus(),
            loadLiveHuiHistory(selectedHive, 1000),
          ]);

        if (predictionResult.status === "fulfilled") {
          setPrediction(predictionResult.value);
        } else {
          setPrediction(null);
          setError(
            predictionResult.reason?.message ?? "Unable to load live HUI data.",
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
    if (!Number.isFinite(refreshMilliseconds) || refreshMilliseconds < 10000) {
      return undefined;
    }

    const interval = window.setInterval(loadDashboard, refreshMilliseconds);

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

    return selectHive(diagnostics, selectedHive) ?? diagnostics[0] ?? null;
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
 const evaluatedHistoryRows = useMemo(() => {
  const ordered = historyRows
    .map((row) => ({
      row,
      time: getHistoryTimestampMs(row),
    }))
    .filter((item) => item.time !== null)
    .sort((a, b) => a.time - b.time);

  const latestAvailableTime =
    ordered.length > 0
      ? ordered[ordered.length - 1].time
      : null;

  function evaluateHorizon(origin, horizonHours) {
    const expected = asNumber(
      origin.row[
        `predicted_hui_${horizonHours}h`
      ],
    );

    const targetTime =
      origin.time +
      horizonHours * HOUR_MS;

    if (expected === null) {
      return {
        horizonHours,
        targetTime,
        expected: null,
        actual: null,
        gap: null,
        absoluteError: null,
        status: "No forecast",
      };
    }

    if (
      latestAvailableTime === null ||
      latestAvailableTime < targetTime
    ) {
      return {
        horizonHours,
        targetTime,
        expected,
        actual: null,
        gap: null,
        absoluteError: null,
        status: "Pending",
      };
    }

    let closest = null;
    let closestDifference =
      Number.POSITIVE_INFINITY;

    for (const candidate of ordered) {
      const difference = Math.abs(
        candidate.time - targetTime,
      );

      if (difference < closestDifference) {
        closest = candidate;
        closestDifference = difference;
      }
    }

    if (
      !closest ||
      closestDifference >
        FUTURE_MATCH_TOLERANCE_MS
    ) {
      return {
        horizonHours,
        targetTime,
        expected,
        actual: null,
        gap: null,
        absoluteError: null,
        status: "Actual unavailable",
      };
    }

    const actual = asNumber(
      closest.row?.current_hui,
    );

    if (actual === null) {
      return {
        horizonHours,
        targetTime,
        expected,
        actual: null,
        gap: null,
        absoluteError: null,
        status: "Actual unavailable",
      };
    }

    const gap =
      actual - expected;

    const absoluteError =
      Math.abs(gap);

    return {
      horizonHours,
      targetTime,
      expected,
      actual,
      gap,
      absoluteError,
      actualTime: closest.time,
      status: "Evaluated",
    };
  }

  return [...ordered]
    .reverse()
    .map((origin) => ({
      row: origin.row,
      predictionTime: origin.time,

      horizon24:
        evaluateHorizon(origin, 24),

      horizon48:
        evaluateHorizon(origin, 48),

      horizon72:
        evaluateHorizon(origin, 72),
    }));
}, [historyRows]);
  const evaluationByPredictionTime = useMemo(() => {
  const map = new Map();

  evaluatedHistoryRows.forEach((item) => {
    map.set(
      item.predictionTime,
      item,
    );
  });

  return map;
}, [evaluatedHistoryRows]);


  const downloadForecastEvaluationCsv = useCallback(() => {
    if (evaluatedHistoryRows.length === 0) {
      return;
    }

    const headers = [
      "prediction_time",
      "hive_id",

      "current_hui",
      "current_class",

      "expected_hui_24h",
      "actual_hui_24h",
      "gap_24h",
      "absolute_error_24h",
      "status_24h",

      "expected_hui_48h",
      "actual_hui_48h",
      "gap_48h",
      "absolute_error_48h",
      "status_48h",

      "expected_hui_72h",
      "actual_hui_72h",
      "gap_72h",
      "absolute_error_72h",
      "status_72h",
    ];

    const rows = evaluatedHistoryRows.map(
      ({ row, predictionTime, horizon24, horizon48, horizon72 }) => [
        new Date(predictionTime).toISOString(),

        row?.hive_id ?? selectedHive ?? "",

        csvNumber(row?.current_hui),

        row?.current_class ?? "",

        csvNumber(horizon24.expected),

        csvNumber(horizon24.actual),

        csvNumber(horizon24.gap),

        csvNumber(horizon24.absoluteError),

        horizon24.status,

        csvNumber(horizon48.expected),

        csvNumber(horizon48.actual),

        csvNumber(horizon48.gap),

        csvNumber(horizon48.absoluteError),

        horizon48.status,

        csvNumber(horizon72.expected),

        csvNumber(horizon72.actual),

        csvNumber(horizon72.gap),

        csvNumber(horizon72.absoluteError),

        horizon72.status,
      ],
    );

    const csv = [headers, ...rows]
      .map((row) => row.map(csvValue).join(","))
      .join("\r\n");

    const blob = new Blob(["\uFEFF", csv], {
      type: "text/csv;charset=utf-8",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    const safeHive = (selectedHive || "hive").replace(/[^a-zA-Z0-9_-]/g, "_");

    link.href = url;

    link.download = `${safeHive}_live_hui_forecast_evaluation.csv`;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  }, [evaluatedHistoryRows, selectedHive]);

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
          <span className="live-eyebrow">LIVE IOT HARVEST INTELLIGENCE</span>
          <h2>Harvest Urgency Dashboard</h2>
          <p>Current hive state and 72-hour HUI outlook.</p>
        </div>

        <div className="live-dashboard-actions">
          <select
            value={selectedHive}
            onChange={(event) => setSelectedHive(event.target.value)}
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
            <RefreshCw size={17} className={refreshing ? "is-spinning" : ""} />
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
          value={formatTime(sensor?.timestamp ?? latest?.source_timestamp_utc)}
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

      <div className="live-ux-decision-layout">
        {/* CURRENT HUI */}
        <section className="live-ux-current-section">
          <div className="live-ux-section-heading">
            <div>
              <span className="live-eyebrow">CURRENT HARVEST URGENCY</span>

              <h3>Where is the hive now?</h3>
            </div>

            <Gauge size={22} />
          </div>

          <h2>
            <HuiMeter
              value={latest?.current_hui}
              readinessClass={latest?.current_class}
              title="Current HUI"
              large
            />
          </h2>
        </section>

        {/* RECOMMENDATION */}
        <aside className="live-ux-recommendation">
          <div className="live-ux-recommendation-heading">
            <span className="live-ux-recommendation-icon">
              <Leaf size={21} />
            </span>

            <div>
              <span>RECOMMENDED ACTION</span>
              <h3>{recommendation.title}</h3>
            </div>
          </div>
          <p>{recommendation.text}</p>
        </aside>
      </div>

      {/* =========================================================
    FUTURE HUI
========================================================= */}

      <section className="live-ux-panel">
        <div className="live-ux-panel-heading">
          <div>
            <span className="live-eyebrow">72-HOUR FORECAST</span>

            <h2>Predicted HUI at Three Future Horizons.</h2>
          </div>
        </div>

        {predictionReady ? (
          <>
            <div className="live-ux-forecast-grid">
              <HuiMeter
                title="+24 hours"
                subtitle="Short Term Forecast"
                model="LightGBM"
                value={latest?.predicted_hui_24h}
                readinessClass={latest?.predicted_class_24h}
              />

              <HuiMeter
                title="+48 hours"
                subtitle="Medium Term Forecast"
                model="XGBoost"
                value={latest?.predicted_hui_48h}
                readinessClass={latest?.predicted_class_48h}
              />

              <HuiMeter
                title="+72 hours"
                subtitle="Extended Term Forecast"
                model="LightGBM"
                value={latest?.predicted_hui_72h}
                readinessClass={latest?.predicted_class_72h}
              />
            </div>

            <div className="live-ux-chart-section">
              <div className="live-ux-chart-heading">
                <div>
                  <strong>HUI Trajectory</strong>
                  <span>Current value compared with the next 72 hours</span>
                </div>

                <TrendingUp size={19} />
              </div>

              <div className="live-ux-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={trajectory}
                    margin={{
                      top: 16,
                      right: 24,
                      left: 0,
                      bottom: 4,
                    }}
                  >
                    <ReferenceArea
                      y1={0}
                      y2={40}
                      fill="#eff6ff"
                      fillOpacity={0.55}
                    />

                    <ReferenceArea
                      y1={40}
                      y2={60}
                      fill="#fefce8"
                      fillOpacity={0.65}
                    />

                    <ReferenceArea
                      y1={60}
                      y2={80}
                      fill="#f0fdf4"
                      fillOpacity={0.7}
                    />

                    <ReferenceArea
                      y1={80}
                      y2={100}
                      fill="#fef2f2"
                      fillOpacity={0.7}
                    />

                    <CartesianGrid
                      strokeDasharray="4 4"
                      vertical={false}
                      stroke="#dbe3ec"
                    />

                    <XAxis
                      dataKey="label"
                      tick={{
                        fontSize: 13,
                        fill: "#475569",
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      domain={[0, 100]}
                      ticks={[0, 20, 40, 60, 80, 100]}
                      width={42}
                      tick={{
                        fontSize: 12,
                        fill: "#64748b",
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <Tooltip
                      formatter={(value) => [
                        `${formatNumber(value, 1)} / 100`,
                        "HUI",
                      ]}
                    />

                    <ReferenceLine
                      y={40}
                      stroke="#d97706"
                      strokeDasharray="5 5"
                    />

                    <ReferenceLine
                      y={60}
                      stroke="#16a34a"
                      strokeDasharray="5 5"
                    />

                    <ReferenceLine
                      y={80}
                      stroke="#dc2626"
                      strokeDasharray="5 5"
                    />

                    <Line
                      type="monotone"
                      dataKey="hui"
                      stroke="#2563eb"
                      strokeWidth={4}
                      dot={{
                        r: 6,
                        fill: "#ffffff",
                        stroke: "#2563eb",
                        strokeWidth: 3,
                      }}
                      activeDot={{
                        r: 8,
                      }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        ) : (
          <div className="live-history-box">
            <HistoryProgress
              label="Current HUI history"
              value={contiguousHours}
              maximum={CURRENT_REQUIRED_HOURS}
            />

            <HistoryProgress
              label="Future HUI history"
              value={contiguousHours}
              maximum={FUTURE_REQUIRED_HOURS}
            />
          </div>
        )}
      </section>

      {/* =========================================================
    SUPPORTING INDICATORS
========================================================= */}

      {predictionReady ? (
        <section className="live-ux-panel">
          <div className="live-ux-panel-heading">
            <div>
              <span className="live-eyebrow">DECISION SUPPORT INDICATORS</span>

              <h3>How strong is the current signal?</h3>

              <p>
                Stability, Recent Movement and Supporting Research Evidence.
              </p>
            </div>
          </div>

          <div className="live-ux-health-grid">
            <HealthMetric
              icon={ShieldCheck}
              label="HUI Stability"
              value={formatNumber(latest?.hrsi, 1)}
              suffix="/100"
              progress={latest?.hrsi}
              description={
                latest?.hrsi_interpretation ?? "Stability unavailable"
              }
            />

            <HealthMetric
              icon={TrendIcon}
              label="Recent HUI Trend"
              value={latest?.rate_of_change ?? "—"}
              description={
                asNumber(latest?.rate_of_change_points_per_hour) === null
                  ? "Rate unavailable"
                  : `${formatNumber(
                      latest?.rate_of_change_points_per_hour,
                      2,
                    )} HUI points per hour`
              }
            />

            <HealthMetric
              icon={CheckCircle2}
              label="Research Confidence"
              value={latest?.prediction_confidence ?? "—"}
              progress={latest?.confidence_score}
              description={
                asNumber(latest?.confidence_score) === null
                  ? "Evidence score unavailable"
                  : `${formatNumber(
                      latest?.confidence_score,
                      1,
                    )}/100 evidence score`
              }
            />
          </div>

          <div className="live-ux-confidence-note">
            <AlertTriangle size={16} />

            <span>
              Confidence is a research evidence score, not a probability that
              the prediction is correct.
            </span>
          </div>
        </section>
      ) : null}

      <article className="live-sensors-panel">
        <div className="live-panel-heading">
          <div>
            <span className="live-eyebrow">CURRENT VARIABLES</span>
            <h3>Latest IoT readings</h3>
          </div>

          <span className={`live-live-indicator ${isFresh ? "is-live" : ""}`}>
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

      {imputationActive || domainShift ? (
        <div className="live-quality-strip">
          <details className="live-saved-history">
            <summary className="live-saved-history-summary">
              <div>
                <strong>Saved Live Prediction History</strong>
                <span>{historyRows.length} model-ready records</span>
                <button
                  type="button"
                  className="live-download-csv-button"
                  disabled={evaluatedHistoryRows.length === 0}
                  onClick={downloadForecastEvaluationCsv}
                >
                  <Download size={17} />
                  Download CSV
                </button>
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

                        <th>+24h Expected</th>
                        <th>+24h Actual</th>
                        <th>+24h Gap</th>
                        <th>+24h Error</th>

                        <th>+48h Expected</th>
                        <th>+48h Actual</th>
                        <th>+48h Gap</th>
                        <th>+48h Error</th>

                        <th>+72h Expected</th>
                        <th>+72h Actual</th>
                        <th>+72h Gap</th>
                        <th>+72h Error</th>

                        <th>HRSI</th>
                        <th>Trend</th>
                        <th>Confidence</th>
                        <th>Recommendation</th>
                      </tr>
                    </thead>

                    <tbody>
                      {historyRows.map((row, index) => {
                        const evaluation = evaluationByPredictionTime.get(
                          getHistoryTimestampMs(row),
                        );

                        const horizon24 = evaluation?.horizon24 ?? null;

                        const horizon48 = evaluation?.horizon48 ?? null;

                        const horizon72 = evaluation?.horizon72 ?? null;

                        return (
                          <tr
                            key={`${row.hive_id}-${row.timestamp}-${row.saved_at_utc ?? "legacy"}-${index}`}
                          >
                            {/* SAVED AT */}
                            <td>{formatSavedTime(row.saved_at_utc)}</td>

                            {/* CURRENT HUI */}
                            <td>
                              <strong>
                                {formatNumber(row.current_hui, 1)}
                              </strong>
                            </td>

                            {/* CURRENT STATUS */}
                            <td>
                              <span
                                className={`live-history-status is-${readinessTone(
                                  row.current_class,
                                )}`}
                              >
                                {row.current_class ?? "—"}
                              </span>
                            </td>

                            {/* =========================
                    +24 HOURS
                ========================= */}

                            {/* EXPECTED */}
                            <td>
                              <strong>
                                {formatNumber(row.predicted_hui_24h, 1)}
                              </strong>
                            </td>

                            {/* ACTUAL */}
                            <td>
                              {horizon24?.actual != null ? (
                                <strong>
                                  {formatNumber(horizon24.actual, 1)}
                                </strong>
                              ) : (
                                <span className="live-eval-pending">
                                  {horizon24?.status ?? "Pending"}
                                </span>
                              )}
                            </td>

                            {/* GAP */}
                            <td>
                              {horizon24?.gap != null ? (
                                <span
                                  className={`live-eval-gap ${
                                    horizon24.gap > 0
                                      ? "is-positive"
                                      : horizon24.gap < 0
                                        ? "is-negative"
                                        : ""
                                  }`}
                                >
                                  {horizon24.gap > 0 ? "+" : ""}
                                  {formatNumber(horizon24.gap, 2)}
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>

                            {/* ERROR */}
                            <td>{formatNumber(horizon24?.absoluteError, 2)}</td>

                            {/* =========================
                    +48 HOURS
                ========================= */}

                            {/* EXPECTED */}
                            <td>
                              <strong>
                                {formatNumber(row.predicted_hui_48h, 1)}
                              </strong>
                            </td>

                            {/* ACTUAL */}
                            <td>
                              {horizon48?.actual != null ? (
                                <strong>
                                  {formatNumber(horizon48.actual, 1)}
                                </strong>
                              ) : (
                                <span className="live-eval-pending">
                                  {horizon48?.status ?? "Pending"}
                                </span>
                              )}
                            </td>

                            {/* GAP */}
                            <td>
                              {horizon48?.gap != null ? (
                                <span
                                  className={`live-eval-gap ${
                                    horizon48.gap > 0
                                      ? "is-positive"
                                      : horizon48.gap < 0
                                        ? "is-negative"
                                        : ""
                                  }`}
                                >
                                  {horizon48.gap > 0 ? "+" : ""}
                                  {formatNumber(horizon48.gap, 2)}
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>

                            {/* ERROR */}
                            <td>{formatNumber(horizon48?.absoluteError, 2)}</td>

                            {/* =========================
                    +72 HOURS
                ========================= */}

                            {/* EXPECTED */}
                            <td>
                              <strong>
                                {formatNumber(row.predicted_hui_72h, 1)}
                              </strong>
                            </td>

                            {/* ACTUAL */}
                            <td>
                              {horizon72?.actual != null ? (
                                <strong>
                                  {formatNumber(horizon72.actual, 1)}
                                </strong>
                              ) : (
                                <span className="live-eval-pending">
                                  {horizon72?.status ?? "Pending"}
                                </span>
                              )}
                            </td>

                            {/* GAP */}
                            <td>
                              {horizon72?.gap != null ? (
                                <span
                                  className={`live-eval-gap ${
                                    horizon72.gap > 0
                                      ? "is-positive"
                                      : horizon72.gap < 0
                                        ? "is-negative"
                                        : ""
                                  }`}
                                >
                                  {horizon72.gap > 0 ? "+" : ""}
                                  {formatNumber(horizon72.gap, 2)}
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>

                            {/* ERROR */}
                            <td>{formatNumber(horizon72?.absoluteError, 2)}</td>

                            {/* =========================
                    EXISTING VALUES
                ========================= */}

                            {/* HRSI */}
                            <td>{formatNumber(row.hrsi, 1)}</td>

                            {/* TREND */}
                            <td>{row.rate_of_change ?? "—"}</td>

                            {/* CONFIDENCE */}
                            <td>
                              <strong>
                                {row.prediction_confidence ?? "—"}
                              </strong>

                              <small>
                                {formatNumber(row.confidence_score, 1)}
                                /100
                              </small>
                            </td>

                            {/* RECOMMENDATION */}
                            <td>{row.recommended_window ?? "—"}</td>
                          </tr>
                        );
                      })}
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

          <details className="live-hui-calc">
            <summary className="live-hui-calc-summary">
              <div>
                <span>HUI METHODOLOGY</span>
                <strong>How Current HUI Is Calculated</strong>
                <small>Sensor data → XGBoost → calibration → HUI</small>
              </div>

              <span className="live-hui-calc-chevron">›</span>
            </summary>

            <div className="live-hui-calc-body">
              <div className="live-hui-calc-intro">
                <h3>Current Harvest Urgency Index Calculation</h3>
                <p>
                  The HUI is not calculated directly from one sensor. It is
                  produced through several steps using the recent hive
                  time-series history.
                </p>
              </div>

              {/* STEP 1 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">1</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">INPUT</span>

                  <h4>Collect recent hive sensor history</h4>

                  <div className="live-hui-calc-sensors">
                    <span>Hive weight</span>
                    <span>Internal temperature</span>
                    <span>CO₂</span>
                    <span>Timestamp</span>
                  </div>

                  <p>
                    The model uses the current reading together with previous
                    readings. It therefore understands how the hive has been
                    changing over time rather than looking at only one instant.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 2 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">2</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    FEATURE ENGINEERING
                  </span>

                  <h4>Create 53 time-series features</h4>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>Current values</strong>
                      <small>Current weight, temperature and CO₂</small>
                    </div>

                    <div>
                      <strong>Changes</strong>
                      <small>1h, 6h, 24h and 72h changes</small>
                    </div>

                    <div>
                      <strong>Rolling behaviour</strong>
                      <small>Means, standard deviations and ranges</small>
                    </div>

                    <div>
                      <strong>Trends</strong>
                      <small>
                        Direction of recent weight and sensor behaviour
                      </small>
                    </div>

                    <div>
                      <strong>Recent maximum</strong>
                      <small>Current weight compared with recent maximum</small>
                    </div>

                    <div>
                      <strong>Time information</strong>
                      <small>Hour and day cyclical features</small>
                    </div>
                  </div>

                  <div className="live-hui-calc-formula-small">
                    Xₜ = [x₁, x₂, x₃, ... , x₅₃]
                  </div>

                  <p>
                    <strong>Xₜ</strong> represents all engineered information
                    available at the current time.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 3 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">3</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">CLASSIFICATION</span>

                  <h4>XGBoost produces the raw harvest score</h4>

                  <p>
                    During training, XGBoost learned patterns associated with
                    reviewed harvest events occurring within the next 72 hours.
                  </p>

                  <div className="live-hui-calc-formula">
                    p<sub>raw,t</sub> = f<sub>XGB</sub>(X<sub>t</sub>)
                  </div>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>Xₜ</strong>
                      <span>53 engineered sensor features</span>
                    </div>

                    <div>
                      <strong>fXGB</strong>
                      <span>Selected XGBoost classifier</span>
                    </div>

                    <div>
                      <strong>praw</strong>
                      <span>Raw classifier score</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 4 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">4</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">CALIBRATION</span>

                  <h4>Adjust the XGBoost score using Platt calibration</h4>

                  <p>
                    XGBoost first produces a raw classifier score. Although this
                    score is useful for ranking hives, its numerical value
                    should not automatically be treated as a reliable
                    probability.
                  </p>

                  <p>
                    Therefore, Platt calibration is applied to adjust the scale
                    of the raw model output before it is converted into the
                    Harvest Urgency Index.
                  </p>

                  <div className="live-hui-calc-formula">
                    p<sub>cal,t</sub>= 1 / (1 + e<sup>−(a·s+b)</sup>)
                  </div>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>s</strong>
                      <span>Raw score produced by the XGBoost classifier</span>
                    </div>

                    <div>
                      <strong>a, b</strong>
                      <span>
                        Calibration parameters learned from model data
                      </span>
                    </div>

                    <div>
                      <strong>pcal</strong>
                      <span>Platt-adjusted classifier score</span>
                    </div>
                  </div>

                  <div className="live-hui-calc-rule">
                    Raw XGBoost score
                    <span>→</span>
                    Platt scaling
                    <span>→</span>
                    Adjusted classifier score
                  </div>

                  <p>
                    In simple terms, Platt calibration acts like a
                    <strong> correction layer</strong>. It does not change the
                    sensor data or retrain XGBoost. It only adjusts the scale of
                    the classifier output so that the next HUI calculation uses
                    a more suitable score.
                  </p>

                  <p>
                    The calibrated value is still not the final HUI. It is
                    passed to the next step, where the score is mapped onto the
                    0–100 HUI scale.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 5 */}
              <div className="live-hui-calc-step is-highlight">
                <div className="live-hui-calc-number">5</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">HUI GENERATION</span>

                  <h4>Convert the calibrated score to the 0–100 HUI scale</h4>

                  <div className="live-hui-calc-main-formula">
                    HUI<sub>t</sub> = M(p<sub>cal,t</sub>)
                  </div>

                  <p>
                    After Platt calibration, the model still produces a very
                    small numerical score because harvest events are rare.
                    Therefore, simply multiplying the calibrated score by 100
                    would not create a useful 0–100 urgency scale.
                  </p>

                  <p>
                    Instead, the calibrated score is compared with
                    <strong>
                      {" "}
                      fixed reference points obtained from the training data
                    </strong>
                    . These reference points show what relatively low, medium
                    and high classifier scores looked like during model
                    development.
                  </p>

                  <div className="live-hui-calc-rule">
                    Calibrated score
                    <span>→</span>
                    Compare with training reference points
                    <span>→</span>
                    Position on 0–100 HUI scale
                  </div>

                  <p>
                    The mapping function <strong>M</strong> performs this
                    conversion. A score that is relatively low compared with the
                    training scores receives a low HUI, while a score that is
                    relatively high receives a higher HUI.
                  </p>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>pcal</strong>
                      <span>Platt-adjusted classifier score</span>
                    </div>

                    <div>
                      <strong>M</strong>
                      <span>Fixed training-derived score-to-HUI mapping</span>
                    </div>

                    <div>
                      <strong>HUI</strong>
                      <span>Relative harvest urgency from 0 to 100</span>
                    </div>
                  </div>

                  <p>
                    The mapping is <strong>monotonic</strong>. This means that
                    if the classifier score becomes higher, the HUI cannot
                    become lower.
                  </p>

                  <div className="live-hui-calc-rule">
                    Lower classifier evidence
                    <span>→</span>
                    Lower HUI
                    <span>│</span>
                    Higher classifier evidence
                    <span>→</span>
                    Higher HUI
                  </div>

                  <p>
                    For example, if today's calibrated score lies in a
                    relatively low part of the training-score distribution, it
                    may be mapped to an HUI such as <strong>32</strong>. An HUI
                    of 32 is then interpreted as
                    <strong> Not Ready</strong>.
                  </p>

                  <p>
                    Therefore, HUI should be understood as a
                    <strong> relative urgency index</strong>, not as a literal
                    probability percentage. An HUI of 32 does not mean “32%
                    probability of harvest” or “32% honey maturity”.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 6 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">6</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">INTERPRETATION</span>

                  <h4>Convert HUI into a readiness class</h4>

                  <div className="live-hui-calc-scale">
                    <div className="is-low">
                      <strong>0–39</strong>
                      <span>Not Ready</span>
                    </div>

                    <div className="is-approaching">
                      <strong>40–59</strong>
                      <span>Approaching Harvest</span>
                    </div>

                    <div className="is-ready">
                      <strong>60–79</strong>
                      <span>Ready</span>
                    </div>

                    <div className="is-high">
                      <strong>80–100</strong>
                      <span>High-Priority Harvest</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* FINAL FLOW */}
              <div className="live-hui-calc-final">
                <span>Complete calculation</span>

                <strong>
                  Sensor History
                  <b>→</b>
                  53 Features
                  <b>→</b>
                  XGBoost
                  <b>→</b>
                  Raw Score
                  <b>→</b>
                  Platt Calibration
                  <b>→</b>
                  HUI 0–100
                  <b>→</b>
                  Readiness Class
                </strong>
              </div>
            </div>
          </details>

          <details className="live-hui-calc">
            <summary className="live-hui-calc-summary">
              <div>
                <span>FUTURE HUI FORECAST</span>
                <strong>How 24h, 48h and 72h HUI Are Predicted</strong>
                <small>
                  Current hive condition → regression models → future HUI
                </small>
              </div>

              <span className="live-hui-calc-chevron">›</span>
            </summary>

            <div className="live-hui-calc-body">
              <div className="live-hui-calc-intro">
                <h3>Future Harvest Urgency Forecasting</h3>

                <p>
                  After calculating the current HUI, the system predicts what
                  the HUI may be after 24, 48 and 72 hours.
                </p>
              </div>

              {/* STEP 1 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">1</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    CURRENT INFORMATION
                  </span>

                  <h4>Use the current HUI and current sensor features</h4>

                  <p>
                    The forecasting models receive information describing the
                    hive at the current time, including the current HUI and
                    recent engineered sensor behaviour.
                  </p>

                  <div className="live-hui-calc-rule">
                    Current sensor features
                    <span>+</span>
                    Current HUI
                    <span>→</span>
                    Future-HUI model input
                  </div>

                  <div className="live-hui-calc-formula-small">
                    Zₜ = [Xₜ, HUIₜ]
                  </div>

                  <p>
                    <strong>Xₜ</strong> represents the current engineered sensor
                    features and <strong>HUIₜ</strong> represents the current
                    Harvest Urgency Index.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 2 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">2</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">MODEL SELECTION</span>

                  <h4>
                    Select the best model separately for each forecast horizon
                  </h4>

                  <p>
                    The same regression model was not automatically used for all
                    three horizons. The 24-hour, 48-hour and 72-hour prediction
                    problems were trained and evaluated separately.
                  </p>

                  <p>
                    For each horizon, candidate regression models were compared
                    using the <strong>validation dataset</strong>. The main
                    error measure used for model selection was
                    <strong> Mean Absolute Error (MAE)</strong>.
                  </p>

                  <div className="live-hui-calc-formula">
                    MAE = (1 / n) Σ |HUI<sub>actual</sub> − HUI
                    <sub>predicted</sub>|
                  </div>

                  <p>
                    MAE tells us, on average, how many HUI points the forecast
                    differs from the true future HUI.
                    <strong> Lower MAE is better.</strong>
                  </p>

                  <div className="live-hui-calc-rule">
                    Train candidate models
                    <span>→</span>
                    Predict validation HUI
                    <span>→</span>
                    Calculate MAE
                    <span>→</span>
                    Select the better-performing model
                  </div>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>+24 hours — LightGBM</strong>

                      <small>Validation MAE: 3.071 HUI points</small>

                      <small>Persistence baseline: 4.641</small>

                      <small>Improvement: ≈33.8%</small>
                    </div>

                    <div>
                      <strong>+48 hours — XGBoost</strong>

                      <small>Validation MAE: 3.851 HUI points</small>

                      <small>Persistence baseline: 7.322</small>

                      <small>Improvement: ≈47.4%</small>
                    </div>

                    <div>
                      <strong>+72 hours — LightGBM</strong>

                      <small>Validation MAE: 4.654 HUI points</small>

                      <small>Persistence baseline: 9.017</small>

                      <small>Improvement: ≈48.4%</small>
                    </div>
                  </div>

                  <p>
                    Therefore, <strong>LightGBM</strong> was retained for the
                    24-hour horizon, <strong>XGBoost</strong> for the 48-hour
                    horizon, and <strong>LightGBM</strong> again for the 72-hour
                    horizon.
                  </p>

                  <p>
                    This happens because prediction difficulty changes with
                    forecast distance. A model that works best for the next 24
                    hours is not necessarily the model that works best for 48 or
                    72 hours.
                  </p>

                  <div className="live-hui-calc-rule">
                    24h problem
                    <span>→</span>
                    LightGBM
                    <span>│</span>
                    48h problem
                    <span>→</span>
                    XGBoost
                    <span>│</span>
                    72h problem
                    <span>→</span>
                    LightGBM
                  </div>

                  <p>
                    Model selection was performed using
                    <strong>
                      {" "}
                      validation results, not the final test results
                    </strong>
                    . The test set was kept for checking how the
                    already-selected model generalized to unseen data.
                  </p>

                  <div className="live-hui-calc-note">
                    <strong>Why compare with persistence?</strong>

                    <p>
                      Persistence is a simple baseline that assumes the future
                      HUI will remain approximately the same as the current HUI.
                      A trained forecasting model should improve on this simple
                      baseline to demonstrate that it is learning useful future
                      behaviour.
                    </p>
                  </div>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 3 */}
              <div className="live-hui-calc-step is-highlight">
                <div className="live-hui-calc-number">3</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    REGRESSION PREDICTION
                  </span>

                  <h4>Predict future continuous HUI values</h4>

                  <div className="live-hui-calc-main-formula">
                    ĤUI<sub>t+h</sub> = f<sub>h</sub>(Z<sub>t</sub>)
                  </div>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>h</strong>
                      <span>Forecast horizon: 24, 48 or 72 hours</span>
                    </div>

                    <div>
                      <strong>Zₜ</strong>
                      <span>Information available now</span>
                    </div>

                    <div>
                      <strong>ĤUI</strong>
                      <span>Predicted future HUI</span>
                    </div>
                  </div>

                  <p>
                    In simple terms, each regression model learned from
                    historical examples how today's hive condition was related
                    to the HUI observed later.
                  </p>

                  <div className="live-hui-calc-rule">
                    Hive condition now
                    <span>→</span>
                    learned historical relationship
                    <span>→</span>
                    estimated HUI later
                  </div>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 4 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">4</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">OUTPUT</span>

                  <h4>Produce three continuous HUI forecasts</h4>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>ĤUIₜ₊₂₄</strong>
                      <small>Estimated HUI after 24 hours</small>
                    </div>

                    <div>
                      <strong>ĤUIₜ₊₄₈</strong>
                      <small>Estimated HUI after 48 hours</small>
                    </div>

                    <div>
                      <strong>ĤUIₜ₊₇₂</strong>
                      <small>Estimated HUI after 72 hours</small>
                    </div>
                  </div>

                  <p>
                    These are continuous values, so predictions such as
                    <strong> 38.5</strong> or <strong>41.8</strong> are valid.
                  </p>

                  <p>
                    Each predicted HUI is then interpreted using the same
                    readiness boundaries as the current HUI.
                  </p>

                  <div className="live-hui-calc-scale">
                    <div className="is-low">
                      <strong>0 – &lt;40</strong>
                      <span>Not Ready</span>
                    </div>

                    <div className="is-approaching">
                      <strong>40 – &lt;60</strong>
                      <span>Approaching Harvest</span>
                    </div>

                    <div className="is-ready">
                      <strong>60 – &lt;80</strong>
                      <span>Ready</span>
                    </div>

                    <div className="is-high">
                      <strong>80 – 100</strong>
                      <span>High-Priority Harvest</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="live-hui-calc-final">
                <span>Complete forecast process</span>

                <strong>
                  Current HUI + Sensor Features
                  <b>→</b>
                  Separate Regression Models
                  <b>→</b>
                  +24h / +48h / +72h HUI
                  <b>→</b>
                  Future Readiness Classes
                </strong>
              </div>
            </div>
          </details>
          <details className="live-hui-calc">
            <summary className="live-hui-calc-summary">
              <div>
                <span>READINESS STABILITY</span>
                <strong>How HRSI Is Calculated</strong>
                <small>
                  Recent HUI variation → stability score from 0 to 100
                </small>
              </div>

              <span className="live-hui-calc-chevron">›</span>
            </summary>

            <div className="live-hui-calc-body">
              <div className="live-hui-calc-intro">
                <h3>Harvest Readiness Stability Index</h3>

                <p>
                  HRSI checks whether the recent HUI values have remained
                  consistent or have been changing strongly.
                </p>
              </div>

              {/* STEP 1 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">1</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">RECENT HISTORY</span>

                  <h4>Take the most recent 24 HUI values</h4>

                  <p>
                    The system does not judge stability from only the current
                    HUI. It looks at the recent HUI history.
                  </p>

                  <div className="live-hui-calc-rule">
                    HUI₁, HUI₂, HUI₃, ... , HUI₂₄
                  </div>

                  <p>
                    With hourly HUI observations, this represents the recent
                    24-hour readiness behaviour.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 2 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">2</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">VARIATION</span>

                  <h4>Calculate the standard deviation of recent HUI</h4>

                  <div className="live-hui-calc-formula">
                    σ = √[ Σ(HUI<sub>i</sub> − HUĪ)² / N ]
                  </div>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>HUIᵢ</strong>
                      <span>Each recent HUI value</span>
                    </div>

                    <div>
                      <strong>HUĪ</strong>
                      <span>Average recent HUI</span>
                    </div>

                    <div>
                      <strong>σ</strong>
                      <span>Amount of HUI variation</span>
                    </div>
                  </div>

                  <p>
                    Standard deviation simply tells us how spread out the recent
                    HUI values are.
                  </p>

                  <div className="live-hui-calc-rule">
                    Similar HUI values
                    <span>→</span>
                    small σ<span>│</span>
                    strongly changing HUI values
                    <span>→</span>
                    large σ
                  </div>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 3 */}
              <div className="live-hui-calc-step is-highlight">
                <div className="live-hui-calc-number">3</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">HRSI CALCULATION</span>

                  <h4>Convert HUI variation into a 0–100 stability score</h4>

                  <div className="live-hui-calc-main-formula">
                    HRSI = clip[ 100 × (1 − σ / 20), 0, 100 ]
                  </div>

                  <p>
                    The calculation starts from 100 and reduces the score as the
                    recent HUI becomes more variable.
                  </p>

                  <div className="live-hui-calc-rule">
                    Small variation
                    <span>→</span>
                    HRSI close to 100
                    <span>│</span>
                    Large variation
                    <span>→</span>
                    Lower HRSI
                  </div>

                  <p>
                    The <strong>clip</strong> operation simply ensures that HRSI
                    can never go below 0 or above 100.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 4 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">4</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">INTERPRETATION</span>

                  <h4>Translate HRSI into a stability label</h4>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>75 – 100</strong>
                      <small>Stable</small>
                    </div>

                    <div>
                      <strong>50 – &lt;75</strong>
                      <small>Moderately stable</small>
                    </div>

                    <div>
                      <strong>0 – &lt;50</strong>
                      <small>Fluctuating</small>
                    </div>
                  </div>

                  <p>
                    For example, an HRSI of <strong>92.4</strong> means that the
                    recent HUI values have remained quite consistent, so the
                    dashboard reports
                    <strong> Stable</strong>.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-final">
                <span>Complete HRSI process</span>

                <strong>
                  Last 24 HUI Values
                  <b>→</b>
                  Standard Deviation
                  <b>→</b>
                  HRSI 0–100
                  <b>→</b>
                  Stability Label
                </strong>
              </div>
            </div>
          </details>

          <details className="live-hui-calc">
            <summary className="live-hui-calc-summary">
              <div>
                <span>READINESS TREND</span>
                <strong>
                  How Harvest Readiness Rate of Change Is Calculated
                </strong>
                <small>
                  Recent HUI values → trend slope → Increasing / Stable /
                  Decreasing
                </small>
              </div>

              <span className="live-hui-calc-chevron">›</span>
            </summary>

            <div className="live-hui-calc-body">
              <div className="live-hui-calc-intro">
                <h3>Harvest Readiness Rate of Change</h3>

                <p>
                  The rate of change shows whether the HUI is currently moving
                  upward, downward or remaining approximately stable.
                </p>
              </div>

              {/* STEP 1 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">1</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    RECENT HUI HISTORY
                  </span>

                  <h4>Take the latest 6 HUI values</h4>

                  <p>
                    Instead of comparing only the current HUI with the previous
                    HUI, the system looks at the latest six hourly HUI
                    observations.
                  </p>

                  <div className="live-hui-calc-rule">
                    HUI₁
                    <span>→</span>
                    HUI₂
                    <span>→</span>
                    HUI₃
                    <span>→</span>
                    ...
                    <span>→</span>
                    HUI₆
                  </div>

                  <p>
                    Using several observations gives a better indication of the
                    recent overall direction and reduces the effect of a single
                    temporary change.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 2 */}
              <div className="live-hui-calc-step is-highlight">
                <div className="live-hui-calc-number">2</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">TREND CALCULATION</span>

                  <h4>
                    Fit a straight trend line through the recent HUI values
                  </h4>

                  <p>
                    A least-squares line is fitted through the six recent HUI
                    observations. The slope of that line tells us how quickly
                    HUI is changing over time.
                  </p>

                  <div className="live-hui-calc-main-formula">
                    HRRoC = Σ[(tᵢ − t̄)(HUIᵢ − HUĪ)] / Σ[(tᵢ − t̄)²]
                  </div>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>tᵢ</strong>
                      <span>Time of each recent HUI observation</span>
                    </div>

                    <div>
                      <strong>HUIᵢ</strong>
                      <span>HUI value at that time</span>
                    </div>

                    <div>
                      <strong>HRRoC</strong>
                      <span>HUI change in points per hour</span>
                    </div>
                  </div>

                  <p>
                    If the fitted line slopes upward, HUI is increasing. If it
                    slopes downward, HUI is decreasing. A nearly flat line means
                    readiness is relatively stable.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 3 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">3</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">INTERPRETATION</span>

                  <h4>Convert the slope into a trend label</h4>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>HRRoC &gt; +0.5</strong>
                      <small>Increasing</small>
                    </div>

                    <div>
                      <strong>−0.5 to +0.5</strong>
                      <small>Stable</small>
                    </div>

                    <div>
                      <strong>HRRoC &lt; −0.5</strong>
                      <small>Decreasing</small>
                    </div>
                  </div>

                  <p>
                    For example, if the dashboard reports
                    <strong> +0.67 HUI points/hour</strong>:
                  </p>

                  <div className="live-hui-calc-rule">
                    +0.67
                    <span>&gt;</span>
                    +0.5
                    <span>→</span>
                    Increasing
                  </div>

                  <p>
                    This means the harvest-urgency index has been moving upward
                    during the recent HUI history.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-final">
                <span>Complete rate-of-change process</span>

                <strong>
                  Latest 6 HUI Values
                  <b>→</b>
                  Fit Trend Line
                  <b>→</b>
                  Calculate Slope
                  <b>→</b>
                  HUI Points / Hour
                  <b>→</b>
                  Increasing / Stable / Decreasing
                </strong>
              </div>
            </div>
          </details>

          <details className="live-hui-calc">
            <summary className="live-hui-calc-summary">
              <div>
                <span>RESEARCH CONFIDENCE</span>
                <strong>How Prediction Confidence Is Calculated</strong>
                <small>
                  Calibration evidence + HUI stability + data completeness
                </small>
              </div>

              <span className="live-hui-calc-chevron">›</span>
            </summary>

            <div className="live-hui-calc-body">
              <div className="live-hui-calc-intro">
                <h3>Prediction Confidence</h3>

                <p>
                  The confidence score summarizes how much supporting evidence
                  is available for the current prediction. It combines three
                  different components.
                </p>
              </div>

              {/* STEP 1 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">1</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    CALIBRATION EVIDENCE
                  </span>

                  <h4>Check the strength of classifier calibration</h4>

                  <p>
                    The first component looks at the calibration evidence.
                    Stronger calibration evidence receives a larger score.
                  </p>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>100</strong>
                      <small>Calibration research gate passed</small>
                    </div>

                    <div>
                      <strong>50</strong>
                      <small>
                        A non-identity calibration method was selected
                      </small>
                    </div>

                    <div>
                      <strong>25</strong>
                      <small>Calibration evidence remains limited</small>
                    </div>
                  </div>

                  <p>
                    This component receives the largest weight because the
                    current HUI starts from the classifier score, so the quality
                    of that score calibration is important.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 2 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">2</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">HUI STABILITY</span>

                  <h4>Use the HRSI as the stability component</h4>

                  <p>
                    The second component is the HRSI calculated from the recent
                    HUI history.
                  </p>

                  <div className="live-hui-calc-rule">
                    Stable recent HUI
                    <span>→</span>
                    High HRSI
                    <span>→</span>
                    Stronger confidence
                  </div>

                  <p>
                    This prevents a highly fluctuating HUI from receiving the
                    same confidence as a consistently stable HUI.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 3 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">3</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    INPUT COMPLETENESS
                  </span>

                  <h4>
                    Check whether the required sensor information is available
                  </h4>

                  <p>
                    The third component checks how complete the sensor inputs
                    are. Missing sensor information reduces the evidence
                    supporting the prediction.
                  </p>

                  <div className="live-hui-calc-formula">
                    Completeness = (Available Inputs / Required Inputs) × 100
                  </div>

                  <p>
                    For example, if all required sensor values are available:
                  </p>

                  <div className="live-hui-calc-rule">
                    Complete sensor inputs
                    <span>→</span>
                    100% completeness
                  </div>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 4 */}
              <div className="live-hui-calc-step is-highlight">
                <div className="live-hui-calc-number">4</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">
                    CONFIDENCE CALCULATION
                  </span>

                  <h4>Combine the three components</h4>

                  <div className="live-hui-calc-main-formula">
                    Confidence = 0.40 × Calibration + 0.35 × HRSI + 0.25 ×
                    Completeness
                  </div>

                  <div className="live-hui-calc-definition">
                    <div>
                      <strong>40%</strong>
                      <span>Calibration evidence</span>
                    </div>

                    <div>
                      <strong>35%</strong>
                      <span>Recent HUI stability</span>
                    </div>

                    <div>
                      <strong>25%</strong>
                      <span>Input-data completeness</span>
                    </div>
                  </div>

                  <p>
                    Therefore, confidence becomes stronger when the model has
                    better calibration evidence, the recent HUI is stable, and
                    the required sensor information is available.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-down">↓</div>

              {/* STEP 5 */}
              <div className="live-hui-calc-step">
                <div className="live-hui-calc-number">5</div>

                <div className="live-hui-calc-step-content">
                  <span className="live-hui-calc-label">INTERPRETATION</span>

                  <h4>Convert the confidence score into a label</h4>

                  <div className="live-hui-calc-feature-grid">
                    <div>
                      <strong>0 – &lt;50</strong>
                      <small>Low</small>
                    </div>

                    <div>
                      <strong>50 – &lt;75</strong>
                      <small>Moderate</small>
                    </div>

                    <div>
                      <strong>75 – 100</strong>
                      <small>High</small>
                    </div>
                  </div>

                  <p>
                    For example, a score of
                    <strong> 68</strong> would be interpreted as
                    <strong> Moderate research confidence</strong>.
                  </p>
                </div>
              </div>

              <div className="live-hui-calc-final">
                <span>Complete confidence process</span>

                <strong>
                  Calibration Evidence
                  <b>+</b>
                  HRSI
                  <b>+</b>
                  Input Completeness
                  <b>→</b>
                  Confidence 0–100
                  <b>→</b>
                  Low / Moderate / High
                </strong>
              </div>
            </div>
          </details>
        </div>
      ) : null}
    </section>
  );
}
