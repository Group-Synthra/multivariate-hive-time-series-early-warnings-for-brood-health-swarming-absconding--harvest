import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Database,
  Gauge,
  Info,
  LineChart as LineChartIcon,
  Scale,
  Sparkles,
  Thermometer,
  Waves,
  Wind,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Panel } from "../../../components/common/Panel";
import { loadClassifierDerivedHuiDashboard } from "../../../services/classifierDerivedHuiService";

import "./ClassifierDerivedHuiPredictionTab.css";

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined) {
    return "—";
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function classTone(value) {
  if (value === "Not Ready") {
    return "not-ready";
  }
  if (value === "Approaching Harvest") {
    return "approaching";
  }
  if (value === "Ready") {
    return "ready";
  }
  return "high-priority";
}

function classShortLabel(value) {
  if (value === "Approaching Harvest") {
    return "Approaching";
  }
  if (value === "High-Priority Harvest") {
    return "High Priority";
  }
  return value ?? "Unavailable";
}

function ForecastCard({ horizon, hui, readinessClass, model, current = false }) {
  return (
    <article
      className={`decision-hui-forecast-card is-${classTone(
        readinessClass,
      )} ${current ? "is-current" : ""}`}
    >
      <div className="decision-hui-forecast-heading">
        <span>{horizon}</span>
        <small>{current ? "Classifier" : model}</small>
      </div>
      <strong>{formatNumber(hui, 1)}</strong>
      <b>{classShortLabel(readinessClass)}</b>
    </article>
  );
}

function MiniMetric({ icon: Icon, label, value, note }) {
  return (
    <div className="decision-hui-mini-metric">
      <span className="decision-hui-mini-icon">
        <Icon size={17} />
      </span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
        {note ? <span>{note}</span> : null}
      </div>
    </div>
  );
}

function SensorCard({ icon: Icon, label, value, suffix = "" }) {
  return (
    <div className="decision-hui-sensor-card">
      <span className="decision-hui-sensor-icon">
        <Icon size={18} />
      </span>
      <div>
        <small>{label}</small>
        <strong>
          {value === null || value === undefined
            ? "Unavailable"
            : `${formatNumber(value, 1)}${suffix}`}
        </strong>
      </div>
    </div>
  );
}

function processStep(number, title, text) {
  return (
    <div className="decision-hui-process-step" key={number}>
      <span>{number}</span>
      <div>
        <strong>{title}</strong>
        <small>{text}</small>
      </div>
    </div>
  );
}

function historyCutoff(rows, hours) {
  if (!rows.length || hours === "all") {
    return rows;
  }

  const timestamps = rows
    .map((row) => new Date(row.timestamp).getTime())
    .filter(Number.isFinite);

  if (!timestamps.length) {
    return rows;
  }

  const latest = Math.max(...timestamps);
  const minimum = latest - Number(hours) * 60 * 60 * 1000;

  return rows.filter(
    (row) => new Date(row.timestamp).getTime() >= minimum,
  );
}

export default function ClassifierDerivedHuiPredictionTab() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedHive, setSelectedHive] = useState("");
  const [error, setError] = useState("");
  const [historyRange, setHistoryRange] = useState(168);
  const [visibleSeries, setVisibleSeries] = useState({
    current: true,
    h24: true,
    h48: true,
    h72: true,
  });

  useEffect(() => {
    let cancelled = false;

    loadClassifierDerivedHuiDashboard()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setDashboard(payload);
        setSelectedHive(payload.available_hives?.[0] ?? "");
        setError("");
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(loadError.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const latest = useMemo(() => {
    if (!dashboard || !selectedHive) {
      return null;
    }

    return (
      dashboard.latest_by_hive.find(
        (row) => row.hive_id === selectedHive,
      ) ?? null
    );
  }, [dashboard, selectedHive]);

  const selectedSeries = useMemo(() => {
    if (!dashboard || !selectedHive) {
      return [];
    }

    const rows = dashboard.historical_test_series
      .filter((row) => row.hive_id === selectedHive)
      .map((row) => ({
        ...row,
        displayTime: new Date(row.timestamp).toLocaleString([], {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }),
      }));

    return historyCutoff(rows, historyRange);
  }, [dashboard, selectedHive, historyRange]);

  if (error) {
    return (
      <Panel title="HUI Decision Support">
        <p className="decision-hui-error">{error}</p>
      </Panel>
    );
  }

  if (!dashboard) {
    return (
      <Panel title="HUI Decision Support">
        <p>Loading HUI decision-support dashboard…</p>
      </Panel>
    );
  }

  const horizonSummary =
    dashboard.future_hui_regression?.summary?.horizons ?? {};
  const model24 = horizonSummary?.["24"]?.selected_model ?? "LightGBM";
  const model48 = horizonSummary?.["48"]?.selected_model ?? "XGBoost";
  const model72 = horizonSummary?.["72"]?.selected_model ?? "LightGBM";
  const gatePassed = Boolean(
    dashboard.future_hui_regression?.gate?.gate_passed,
  );

  const trajectory = latest
    ? [
        {
          horizon: "Current",
          hui: Number(latest.current_hui),
          readinessClass: latest.current_class,
          model: "Classifier",
        },
        {
          horizon: "+24h",
          hui: Number(latest.predicted_hui_24h),
          readinessClass: latest.predicted_class_24h,
          model: model24,
        },
        {
          horizon: "+48h",
          hui: Number(latest.predicted_hui_48h),
          readinessClass: latest.predicted_class_48h,
          model: model48,
        },
        {
          horizon: "+72h",
          hui: Number(latest.predicted_hui_72h),
          readinessClass: latest.predicted_class_72h,
          model: model72,
        },
      ]
    : [];

  const highestForecast = trajectory
    .slice(1)
    .reduce(
      (best, item) =>
        item.hui > (best?.hui ?? -Infinity) ? item : best,
      null,
    );

  const recommendationTitle =
    latest?.recommended_window ?? "No harvest window indicated";
  const recommendationText =
    latest?.final_recommendation ??
    "Continue routine monitoring until the HUI enters a higher readiness range.";

  const factors = (latest?.contributing_factors ?? []).slice(0, 2);

  return (
    <section className="classifier-hui-tab decision-hui-tab">
      <header className="decision-hui-header">
        <div>
          <span className="eyebrow">HISTORICAL HUI DECISION SUPPORT</span>
          <h2>Current and 72-Hour Harvest Urgency</h2>
          <p>
            This view demonstrates how the selected classifier produces the
            current HUI and how the selected regressors forecast HUI over the
            next 24, 48 and 72 hours.
          </p>
        </div>

        <label className="decision-hui-hive-select">
          <span>Hive</span>
          <select
            value={selectedHive}
            onChange={(event) => setSelectedHive(event.target.value)}
          >
            {dashboard.available_hives.map((hive) => (
              <option key={hive} value={hive}>
                {hive}
              </option>
            ))}
          </select>
        </label>
      </header>

      <section className="decision-hui-process-panel">
        <div className="decision-hui-process-title">
          <Info size={19} />
          <div>
            <strong>What is happening in this tab?</strong>
            <small>Historical test data is used here for demonstration; live IoT prediction is shown in the separate Live IoT Prediction tab.</small>
          </div>
        </div>

        <div className="decision-hui-process-grid">
          {processStep(
            "1",
            "Current HUI",
            "XGBoost classifier score is transformed into a 0–100 urgency index.",
          )}
          {processStep(
            "2",
            "Future HUI",
            "Selected regressors forecast the index at +24h, +48h and +72h.",
          )}
          {processStep(
            "3",
            "Decision support",
            "Forecast readiness classes are converted into a monitoring or inspection recommendation.",
          )}
        </div>
      </section>

      <div className="decision-hui-overview-grid">
        <article className={`decision-hui-current-card is-${classTone(latest?.current_class)}`}>
          <div className="decision-hui-current-heading">
            <span className="decision-hui-current-icon">
              <Gauge size={26} />
            </span>
            <div>
              <small>CURRENT HUI</small>
              <strong>{formatNumber(latest?.current_hui, 1)}</strong>
              <b>{classShortLabel(latest?.current_class)}</b>
            </div>
          </div>

          <div className="decision-hui-current-scale">
            <span>0</span>
            <div>
              <i
                style={{
                  width: `${Math.min(
                    100,
                    Math.max(0, Number(latest?.current_hui ?? 0)),
                  )}%`,
                }}
              />
            </div>
            <span>100</span>
          </div>

          <p>Relative harvest urgency from the selected classifier.</p>
        </article>

        <article className="decision-hui-forecast-overview">
          <div className="decision-hui-section-heading compact">
            <div>
              <small>UPCOMING 72 HOURS</small>
              <h3>Future-HUI forecast</h3>
            </div>
            <span className="decision-hui-gate-pill">
              <CheckCircle2 size={15} />
              {gatePassed ? "3/3 horizons passed" : "Research forecast"}
            </span>
          </div>

          <div className="decision-hui-forecast-grid">
            {trajectory.slice(1).map((item) => (
              <ForecastCard
                key={item.horizon}
                horizon={item.horizon}
                hui={item.hui}
                readinessClass={item.readinessClass}
                model={item.model}
              />
            ))}
          </div>
        </article>

        <article className="decision-hui-action-card">
          <div className="decision-hui-section-heading compact">
            <div>
              <small>RECOMMENDED ACTION</small>
              <h3>{recommendationTitle}</h3>
            </div>
            <span className="decision-hui-action-icon">
              <ArrowRight size={21} />
            </span>
          </div>

          <p>{recommendationText}</p>

          {highestForecast ? (
            <div className="decision-hui-action-summary">
              <span>Highest forecast</span>
              <strong>
                {formatNumber(highestForecast.hui, 1)} · {highestForecast.horizon}
              </strong>
              <small>{classShortLabel(highestForecast.readinessClass)}</small>
            </div>
          ) : null}
        </article>
      </div>

      <div className="decision-hui-mini-grid">
        <MiniMetric
          icon={Waves}
          label="Readiness stability"
          value={formatNumber(latest?.hrsi, 1)}
          note={latest?.hrsi_interpretation ?? "Unavailable"}
        />
        <MiniMetric
          icon={Activity}
          label="Recent HUI trend"
          value={latest?.rate_of_change ?? "—"}
          note={`${formatNumber(
            latest?.rate_of_change_points_per_hour,
            2,
          )} points/hour`}
        />
        <MiniMetric
          icon={Clock3}
          label="Forecast coverage"
          value="72 hours"
          note={`${model24} · ${model48} · ${model72}`}
        />
      </div>

      <Panel
        title="Current and future HUI trajectory"
        subtitle="Readiness thresholds: 40 Approaching, 60 Ready, 80 High Priority."
      >
        <div className="decision-hui-trajectory-layout">
          <div className="decision-hui-current-plus-forecast">
            <ForecastCard
              horizon="Current"
              hui={latest?.current_hui}
              readinessClass={latest?.current_class}
              model="Classifier"
              current
            />
            {trajectory.slice(1).map((item) => (
              <ForecastCard
                key={`trajectory-${item.horizon}`}
                horizon={item.horizon}
                hui={item.hui}
                readinessClass={item.readinessClass}
                model={item.model}
              />
            ))}
          </div>

          <div className="classifier-hui-trajectory-chart decision-hui-trajectory-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={trajectory}
                margin={{ top: 16, right: 34, left: 5, bottom: 8 }}
              >
                <ReferenceArea y1={0} y2={40} fill="#f8fafc" fillOpacity={0.9} />
                <ReferenceArea y1={40} y2={60} fill="#fffbeb" fillOpacity={0.8} />
                <ReferenceArea y1={60} y2={80} fill="#f0fdf4" fillOpacity={0.8} />
                <ReferenceArea y1={80} y2={100} fill="#fef2f2" fillOpacity={0.75} />
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="horizon" />
                <YAxis domain={[0, 100]} />
                <Tooltip
                  formatter={(value) => [
                    formatNumber(value, 1),
                    "HUI",
                  ]}
                />
                <ReferenceLine y={40} stroke="#d97706" strokeDasharray="5 5" />
                <ReferenceLine y={60} stroke="#16a34a" strokeDasharray="5 5" />
                <ReferenceLine y={80} stroke="#dc2626" strokeDasharray="5 5" />
                <Line
                  type="monotone"
                  dataKey="hui"
                  name="Harvest Urgency Index"
                  stroke="#2563eb"
                  strokeWidth={3}
                  dot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Panel>

      <div className="decision-hui-explanation-grid">
        <article className="decision-hui-simple-card">
          <div className="decision-hui-section-heading compact">
            <div>
              <small>SIMPLE INTERPRETATION</small>
              <h3>What does this result mean?</h3>
            </div>
            <Sparkles size={20} />
          </div>

          <p>
            The current hive state is <strong>{classShortLabel(latest?.current_class)}</strong> at HUI {formatNumber(latest?.current_hui, 1)}.
            The highest forecast over the next 72 hours is <strong>{formatNumber(highestForecast?.hui, 1)}</strong>, classified as <strong>{classShortLabel(highestForecast?.readinessClass)}</strong>.
          </p>

          <div className="decision-hui-class-key">
            <span className="is-not-ready">0–39 Not Ready</span>
            <span className="is-approaching">40–59 Approaching</span>
            <span className="is-ready">60–79 Ready</span>
            <span className="is-high-priority">80–100 High Priority</span>
          </div>
        </article>

        <article className="decision-hui-simple-card">
          <div className="decision-hui-section-heading compact">
            <div>
              <small>WHY THIS OUTPUT?</small>
              <h3>Key contributing factors</h3>
            </div>
            <LineChartIcon size={20} />
          </div>

          <ul className="decision-hui-factor-list">
            {factors.length ? (
              factors.map((factor) => <li key={factor}>{factor}</li>)
            ) : (
              <li>No additional contributing factors were exported.</li>
            )}
          </ul>
        </article>
      </div>

      <section className="decision-hui-sensor-section">
        <div className="decision-hui-section-heading">
          <div>
            <span className="eyebrow">SENSOR SNAPSHOT</span>
            <h3>Variables behind the demonstrated decision</h3>
            <p>{latest?.timestamp ?? "Latest historical record"}</p>
          </div>
        </div>

        <div className="decision-hui-sensor-grid">
          <SensorCard
            icon={Scale}
            label="Hive weight"
            value={latest?.sensor_status?.weight_kg}
            suffix=" kg"
          />
          <SensorCard
            icon={Thermometer}
            label="Internal temperature"
            value={latest?.sensor_status?.internal_temperature_c}
            suffix=" °C"
          />
          <SensorCard
            icon={Waves}
            label="Internal humidity"
            value={latest?.sensor_status?.internal_humidity_pct}
            suffix="%"
          />
          <SensorCard
            icon={Wind}
            label="CO₂ concentration"
            value={latest?.sensor_status?.co2_ppm}
            suffix=" ppm"
          />
        </div>
      </section>

      <Panel
        title="Historical HUI trajectory"
        subtitle="Held-out historical test rows for the selected hive."
      >
        <div className="decision-hui-history-toolbar">
          <div className="decision-hui-range-buttons">
            {[
              [24, "24h"],
              [72, "72h"],
              [168, "7d"],
              ["all", "All"],
            ].map(([value, label]) => (
              <button
                key={label}
                type="button"
                className={historyRange === value ? "is-active" : ""}
                onClick={() => setHistoryRange(value)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="decision-hui-series-buttons">
            {[
              ["current", "Current"],
              ["h24", "+24h"],
              ["h48", "+48h"],
              ["h72", "+72h"],
            ].map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={visibleSeries[key] ? "is-active" : ""}
                onClick={() =>
                  setVisibleSeries((current) => ({
                    ...current,
                    [key]: !current[key],
                  }))
                }
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="classifier-hui-history-chart decision-hui-history-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={selectedSeries}
              margin={{ top: 10, right: 25, left: 5, bottom: 55 }}
            >
              <ReferenceArea y1={0} y2={40} fill="#f8fafc" fillOpacity={0.75} />
              <ReferenceArea y1={40} y2={60} fill="#fffbeb" fillOpacity={0.55} />
              <ReferenceArea y1={60} y2={80} fill="#f0fdf4" fillOpacity={0.55} />
              <ReferenceArea y1={80} y2={100} fill="#fef2f2" fillOpacity={0.5} />
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="displayTime"
                angle={-30}
                textAnchor="end"
                interval="preserveStartEnd"
                height={70}
                tick={{ fontSize: 10 }}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <ReferenceLine y={40} stroke="#d97706" strokeDasharray="5 5" />
              <ReferenceLine y={60} stroke="#16a34a" strokeDasharray="5 5" />
              <ReferenceLine y={80} stroke="#dc2626" strokeDasharray="5 5" />
              {visibleSeries.current ? (
                <Line
                  type="monotone"
                  dataKey="classifier_derived_hui"
                  name="Current HUI"
                  dot={false}
                  stroke="#1d4ed8"
                  strokeWidth={2.5}
                />
              ) : null}
              {visibleSeries.h24 ? (
                <Line
                  type="monotone"
                  dataKey="predicted_hui_24h"
                  name="Predicted +24h HUI"
                  dot={false}
                  stroke="#0891b2"
                  strokeWidth={2}
                />
              ) : null}
              {visibleSeries.h48 ? (
                <Line
                  type="monotone"
                  dataKey="predicted_hui_48h"
                  name="Predicted +48h HUI"
                  dot={false}
                  stroke="#7c3aed"
                  strokeWidth={2}
                />
              ) : null}
              {visibleSeries.h72 ? (
                <Line
                  type="monotone"
                  dataKey="predicted_hui_72h"
                  name="Predicted +72h HUI"
                  dot={false}
                  stroke="#d97706"
                  strokeWidth={2}
                />
              ) : null}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="decision-hui-footnote">
        <Database size={16} />
        <span>
          Historical decision-support demonstration. HUI is a relative urgency index, not a direct honey-maturity percentage.
        </span>
      </div>
    </section>
  );
}
