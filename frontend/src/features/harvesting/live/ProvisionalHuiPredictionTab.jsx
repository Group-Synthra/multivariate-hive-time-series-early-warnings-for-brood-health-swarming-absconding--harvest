import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Gauge,
  ShieldAlert,
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
import { loadProvisionalHuiDashboard } from "../../../services/provisionalHuiService";

import "./ProvisionalHuiPredictionTab.css";

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined) {
    return "—";
  }

  const numeric = Number(value);
  return Number.isFinite(numeric)
    ? numeric.toFixed(digits)
    : "—";
}

function classTone(value) {
  if (value === "Not Ready") {
    return "not-ready";
  }
  if (value === "Approaching Harvest") {
    return "approaching";
  }
  if (value === "Ready — Inspection Recommended") {
    return "ready";
  }
  return "high-priority";
}

function ProjectionCard({
  horizon,
  hui,
  readinessClass,
  model,
}) {
  return (
    <div
      className={`hui-projection-card is-${classTone(
        readinessClass,
      )}`}
    >
      <span>{horizon}</span>
      <strong>{formatNumber(hui, 1)}</strong>
      <b>{readinessClass}</b>
      <small>{model}</small>
    </div>
  );
}

export default function ProvisionalHuiPredictionTab() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedHive, setSelectedHive] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    loadProvisionalHuiDashboard()
      .then((payload) => {
        setDashboard(payload);
        setSelectedHive(
          payload.available_hives?.[0] ?? "",
        );
      })
      .catch((loadError) => {
        setError(loadError.message);
      });
  }, []);

  const selectedSeries = useMemo(() => {
    if (!dashboard || !selectedHive) {
      return [];
    }

    return dashboard.historical_test_series
      .filter((row) => row.hive_id === selectedHive)
      .map((row) => ({
        ...row,
        displayTime: new Date(
          row.timestamp,
        ).toLocaleString(),
      }));
  }, [dashboard, selectedHive]);

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

  if (error) {
    return (
      <Panel title="Provisional HUI Prediction">
        <p className="hui-prediction-error">{error}</p>
      </Panel>
    );
  }

  if (!dashboard) {
    return (
      <Panel title="Provisional HUI Prediction">
        <p>Loading Provisional HUI regression results…</p>
      </Panel>
    );
  }

  const model24 =
    dashboard.models?.["24"]?.selected_model ?? "Unavailable";
  const model48 =
    dashboard.models?.["48"]?.selected_model ?? "Unavailable";
  const model72 =
    dashboard.models?.["72"]?.selected_model ?? "Unavailable";

  const trajectory = latest
    ? [
        {
          horizon: "Current",
          hui: Number(latest.provisional_hui),
          readinessClass:
            latest.provisional_hui_class,
          model: "Observed current index",
        },
        {
          horizon: "+24h",
          hui: Number(latest.predicted_hui_24h),
          readinessClass:
            latest.predicted_class_24h,
          model: model24,
        },
        {
          horizon: "+48h",
          hui: Number(latest.predicted_hui_48h),
          readinessClass:
            latest.predicted_class_48h,
          model: model48,
        },
        {
          horizon: "+72h",
          hui: Number(latest.predicted_hui_72h),
          readinessClass:
            latest.predicted_class_72h,
          model: model72,
        },
      ]
    : [];

  return (
    <section className="provisional-hui-tab">
      <div className="hui-research-warning">
        <ShieldAlert size={22} aria-hidden="true" />
        <div>
          <strong>Provisional research index</strong>
          <p>
            Current and future HUI values are predictions of an
            engineered research index. They are not calibrated
            harvest probabilities and do not directly measure
            honey maturity. A beekeeper inspection is required.
          </p>
        </div>
      </div>

      <div className="hui-tab-heading">
        <div>
          <span className="eyebrow">
            MULTI-HORIZON REGRESSION PROTOTYPE
          </span>
          <h2>Current and Future Provisional HUI</h2>
          <p>
            Current HUI is calculated from present and past hive
            telemetry. Future HUI is predicted for 24, 48 and
            72 hours using regression models selected on the
            validation split.
          </p>
        </div>

        <label>
          <span>Hive</span>
          <select
            value={selectedHive}
            onChange={(event) =>
              setSelectedHive(event.target.value)
            }
          >
            {dashboard.available_hives.map((hive) => (
              <option key={hive} value={hive}>
                {hive}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="stats-grid">
        <StatCard
          label="Current Provisional HUI"
          value={formatNumber(
            latest?.provisional_hui,
            1,
          )}
          icon={Gauge}
          note={
            latest?.provisional_hui_class ?? "Unavailable"
          }
        />
        <StatCard
          label="24h predicted HUI"
          value={formatNumber(
            latest?.predicted_hui_24h,
            1,
          )}
          icon={Clock3}
          note={
            latest?.predicted_class_24h ?? "Unavailable"
          }
        />
        <StatCard
          label="24h HUI change"
          value={
            latest
              ? `${formatNumber(
                  Number(latest.predicted_hui_24h) -
                    Number(latest.provisional_hui),
                  1,
                )} points`
              : "—"
          }
          icon={ArrowRight}
          note={`Selected model: ${model24}`}
        />
        <StatCard
          label="Research gate"
          value={
            dashboard.research_gate.gate_passed
              ? "Passed"
              : "Limited"
          }
          icon={
            dashboard.research_gate.gate_passed
              ? CheckCircle2
              : AlertTriangle
          }
          note="Operational HUI remains disabled"
        />
      </div>

      <Panel
        title="HUI trajectory and readiness classes"
        subtitle="Fixed display classes: 0–39 Not Ready, 40–59 Approaching, 60–79 Ready, 80–100 High Priority."
      >
        <div className="hui-projection-grid">
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

        <div className="hui-trajectory-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={trajectory}
              margin={{
                top: 15,
                right: 30,
                left: 5,
                bottom: 10,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="horizon" />
              <YAxis domain={[0, 100]} />
              <Tooltip
                formatter={(value) => [
                  formatNumber(value, 1),
                  "Provisional HUI",
                ]}
              />
              <ReferenceLine
                y={40}
                strokeDasharray="5 5"
                label="Approaching"
              />
              <ReferenceLine
                y={60}
                strokeDasharray="5 5"
                label="Ready"
              />
              <ReferenceLine
                y={80}
                strokeDasharray="5 5"
                label="High Priority"
              />
              <Line
                type="monotone"
                dataKey="hui"
                name="Provisional HUI"
                stroke="#2563eb"
                strokeWidth={3}
                dot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel
        title="Historical held-out HUI predictions"
        subtitle="Current HUI and 24-hour predicted HUI across the latest exported test rows for the selected hive."
      >
        <div className="hui-history-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={selectedSeries}
              margin={{
                top: 10,
                right: 25,
                left: 5,
                bottom: 75,
              }}
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
                dataKey="provisional_hui"
                name="Current Provisional HUI"
                stroke="#0f766e"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="predicted_hui_24h"
                name="Predicted +24h HUI"
                stroke="#2563eb"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel
        title="Interpretation"
        subtitle="The readiness class is a display category derived from the predicted index."
      >
        <div className="hui-interpretation-grid">
          <div>
            <strong>Not Ready</strong>
            <span>HUI below 40</span>
          </div>
          <div>
            <strong>Approaching Harvest</strong>
            <span>HUI from 40 to below 60</span>
          </div>
          <div>
            <strong>Ready — Inspection Recommended</strong>
            <span>HUI from 60 to below 80</span>
          </div>
          <div>
            <strong>High-Priority Harvest Review</strong>
            <span>HUI 80 or above</span>
          </div>
        </div>
      </Panel>
    </section>
  );
}
