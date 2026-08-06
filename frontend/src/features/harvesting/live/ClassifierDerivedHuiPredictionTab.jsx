import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Gauge,
  ShieldAlert,
  Thermometer,
  Waves,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Panel } from "../../../components/common/Panel";
import { StatCard } from "../../../components/common/StatCard";
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

function formatModelName(value) {
  const model = String(value ?? "").toLowerCase();

  if (model === "lightgbm") {
    return "LightGBM";
  }
  if (model === "xgboost") {
    return "XGBoost";
  }
  if (model === "random_forest") {
    return "Random Forest";
  }
  if (model === "ridge") {
    return "Ridge";
  }
  if (model === "persistence") {
    return "Persistence";
  }

  return value || "Unavailable";
}

function ProjectionCard({ horizon, hui, readinessClass, model }) {
  return (
    <div
      className={`classifier-hui-projection-card is-${classTone(
        readinessClass,
      )}`}
    >
      <span>{horizon}</span>
      <strong>{formatNumber(hui, 1)}</strong>
      <b>{readinessClass}</b>
      <small>{formatModelName(model)}</small>
    </div>
  );
}

function SensorValue({ label, value, suffix = "" }) {
  return (
    <div className="classifier-hui-sensor-value">
      <span>{label}</span>
      <strong>
        {value === null || value === undefined
          ? "Not available"
          : `${formatNumber(value, 1)}${suffix}`}
      </strong>
    </div>
  );
}

export default function ClassifierDerivedHuiPredictionTab() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedHive, setSelectedHive] = useState("");
  const [error, setError] = useState("");

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

    return dashboard.historical_test_series
      .filter((row) => row.hive_id === selectedHive)
      .map((row) => ({
        ...row,
        displayTime: new Date(row.timestamp).toLocaleString(),
      }));
  }, [dashboard, selectedHive]);

  if (error) {
    return (
      <Panel title="Classifier-Derived HUI Prediction">
        <p className="classifier-hui-error">{error}</p>
      </Panel>
    );
  }

  if (!dashboard) {
    return (
      <Panel title="Classifier-Derived HUI Prediction">
        <p>Loading final HUI research dashboard…</p>
      </Panel>
    );
  }

  const horizonSummary =
    dashboard.future_hui_regression?.summary?.horizons ?? {};
  const model24 = horizonSummary?.["24"]?.selected_model ?? "Unavailable";
  const model48 = horizonSummary?.["48"]?.selected_model ?? "Unavailable";
  const model72 = horizonSummary?.["72"]?.selected_model ?? "Unavailable";
  const gatePassed = Boolean(
    dashboard.future_hui_regression?.gate?.gate_passed,
  );

  const trajectory = latest
    ? [
        {
          horizon: "Current",
          hui: Number(latest.current_hui),
          readinessClass: latest.current_class,
          model: "Classifier-derived HUI",
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

  return (
    <section className="classifier-hui-tab">
      <div className="classifier-hui-warning">
        <ShieldAlert size={22} aria-hidden="true" />
        <div>
          <strong>Viva research prototype</strong>
          <p>
            HUI is a classifier-derived relative urgency index. It is
            not a literal probability percentage and does not directly
            measure honey maturity. The current screen uses held-out
            historical records to demonstrate the same output package
            that will be produced from live IoT sensor history.
          </p>
        </div>
      </div>

      <div className="classifier-hui-heading">
        <div>
          <span className="eyebrow">
            CURRENT AND THREE-DAY HARVEST URGENCY
          </span>
          <h2>Classifier-Derived HUI Decision Support</h2>
          <p>
            The current HUI is derived from the selected classifier.
            LightGBM and XGBoost regressors forecast the HUI after 24,
            48 and 72 hours.
          </p>
        </div>

        <label>
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
      </div>

      <div className="stats-grid stats-grid-six">
        <StatCard
          label="Current HUI"
          value={formatNumber(latest?.current_hui, 1)}
          icon={Gauge}
          note={latest?.current_class ?? "Unavailable"}
        />
        <StatCard
          label="24h predicted HUI"
          value={formatNumber(latest?.predicted_hui_24h, 1)}
          icon={Clock3}
          note={latest?.predicted_class_24h ?? "Unavailable"}
        />
        <StatCard
          label="Readiness stability"
          value={formatNumber(latest?.hrsi, 1)}
          icon={Waves}
          note={latest?.hrsi_interpretation ?? "Unavailable"}
        />
        <StatCard
          label="Rate of change"
          value={latest?.rate_of_change ?? "—"}
          icon={Activity}
          note={`${formatNumber(
            latest?.rate_of_change_points_per_hour,
            2,
          )} HUI points/hour`}
        />
        <StatCard
          label="Evidence confidence"
          value={latest?.prediction_confidence ?? "—"}
          icon={ShieldAlert}
          note={`${formatNumber(latest?.confidence_score, 1)}/100 prototype evidence; operational confidence not established`}
        />
        <StatCard
          label="Future-HUI gate"
          value={gatePassed ? "Passed" : "Limited"}
          icon={gatePassed ? CheckCircle2 : AlertTriangle}
          note="Viva dashboard only; operational deployment disabled"
        />
      </div>

      <Panel
        title="Current and future Harvest Urgency Index"
        subtitle="Fixed readiness classes: 0–39 Not Ready, 40–59 Approaching Harvest, 60–79 Ready, and 80–100 High-Priority Harvest."
      >
        <div className="classifier-hui-projection-grid">
          {trajectory.map((item) => (
            <ProjectionCard
              key={item.horizon}
              horizon={item.horizon}
              hui={item.hui}
              readinessClass={item.readinessClass}
              model={item.model}
            />
          ))}
        </div>

        <div className="classifier-hui-trajectory-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={trajectory}
              margin={{ top: 15, right: 30, left: 5, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="horizon" />
              <YAxis domain={[0, 100]} />
              <Tooltip
                formatter={(value) => [
                  formatNumber(value, 1),
                  "HUI",
                ]}
              />
              <ReferenceLine y={40} strokeDasharray="5 5" label="Approaching" />
              <ReferenceLine y={60} strokeDasharray="5 5" label="Ready" />
              <ReferenceLine
                y={80}
                strokeDasharray="5 5"
                label="High-Priority Harvest"
              />
              <Line
                type="monotone"
                dataKey="hui"
                name="Harvest Urgency Index"
                strokeWidth={3}
                dot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="two-column-grid">
        <Panel
          title="Recommended inspection / potential harvest window"
          subtitle="The earliest forecast horizon entering the Ready range determines the beekeeper inspection window; harvesting still requires physical confirmation."
        >
          <div className="classifier-hui-recommendation">
            <ArrowRight size={24} aria-hidden="true" />
            <div>
              <strong>{latest?.recommended_window ?? "Unavailable"}</strong>
              <p>{latest?.final_recommendation ?? "No recommendation available."}</p>
            </div>
          </div>
        </Panel>

        <Panel
          title="Explanation of contributing factors"
          subtitle="Data-derived reasons supporting the current decision-support output."
        >
          <ol className="classifier-hui-factor-list">
            {(latest?.contributing_factors ?? []).map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ol>
        </Panel>
      </div>

      <Panel
        title="Environmental and sensor status"
        subtitle={`Latest demonstrated record: ${latest?.timestamp ?? "Unavailable"}`}
      >
        <div className="classifier-hui-sensor-grid">
          <SensorValue
            label="Hive weight"
            value={latest?.sensor_status?.weight_kg}
            suffix=" kg"
          />
          <SensorValue
            label="Internal temperature"
            value={latest?.sensor_status?.internal_temperature_c}
            suffix=" °C"
          />
          <SensorValue
            label="Internal humidity"
            value={latest?.sensor_status?.internal_humidity_pct}
            suffix="%"
          />
          <SensorValue
            label="CO₂ concentration"
            value={latest?.sensor_status?.co2_ppm}
            suffix=" ppm"
          />
          <SensorValue
            label="Input completeness"
            value={latest?.sensor_status?.input_completeness_percent}
            suffix="%"
          />
          <div className="classifier-hui-sensor-value">
            <span>Sensor freshness</span>
            <strong>{latest?.sensor_status?.sensor_freshness ?? "Unavailable"}</strong>
          </div>
          <div className="classifier-hui-sensor-value">
            <span>Battery status</span>
            <strong>{latest?.sensor_status?.battery_status ?? "Unavailable"}</strong>
          </div>
        </div>
      </Panel>

      <Panel
        title="Held-out historical HUI trajectory"
        subtitle="Held-out historical test rows for the selected hive; this chart is a demonstration, not a live IoT stream."
      >
        <div className="classifier-hui-history-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={selectedSeries}
              margin={{ top: 10, right: 25, left: 5, bottom: 75 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="displayTime"
                angle={-35}
                textAnchor="end"
                interval="preserveStartEnd"
                height={85}
                tick={{ fontSize: 10 }}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="classifier_derived_hui"
                name="Current HUI"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="predicted_hui_24h"
                name="Predicted +24h HUI"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="predicted_hui_48h"
                name="Predicted +48h HUI"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="predicted_hui_72h"
                name="Predicted +72h HUI"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="classifier-hui-method-note">
        <Thermometer size={20} aria-hidden="true" />
        <p>
          Current HUI is derived from the Platt-adjusted classifier score
          through training-only monotonic anchors. Future HUI forecasts
          passed the predefined viva research gate at all three horizons.
          Evidence confidence is capped at Moderate while the probability-calibration gate remains limited. Independent biological and operational validation remains future work.
        </p>
      </div>
    </section>
  );
}
