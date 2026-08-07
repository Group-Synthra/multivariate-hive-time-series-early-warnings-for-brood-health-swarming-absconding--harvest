import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  FlaskConical,
  Scale,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Panel } from "../../../components/common/Panel";
import { StatCard } from "../../../components/common/StatCard";
import { loadHarvestingBenchmarkDashboard } from "../../../services/harvestingBenchmarkService";

import "./ExploratoryWeightForecastTab.css";

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

export default function ExploratoryWeightForecastTab() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedHive, setSelectedHive] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    loadHarvestingBenchmarkDashboard()
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

  const rows = useMemo(() => {
    if (!dashboard || !selectedHive) {
      return [];
    }

    return dashboard.exploratory_24h_series
      .filter((row) => row.hive_id === selectedHive)
      .map((row) => ({
        ...row,
        displayTime: new Date(
          row.timestamp,
        ).toLocaleString(),
      }));
  }, [dashboard, selectedHive]);

  const meanAbsoluteError = useMemo(() => {
    if (rows.length === 0) {
      return null;
    }
    return (
      rows.reduce(
        (total, row) =>
          total + Number(row.absolute_error_kg ?? 0),
        0,
      ) / rows.length
    );
  }, [rows]);

  const latest = rows.at(-1);

  if (error) {
    return (
      <Panel title="Exploratory 24-hour weight forecast">
        <p className="forecast-tab-error">{error}</p>
      </Panel>
    );
  }

  if (!dashboard) {
    return (
      <Panel title="Exploratory 24-hour weight forecast">
        <p>Loading forecast benchmark…</p>
      </Panel>
    );
  }

  return (
    <section className="exploratory-forecast-tab">
      <div className="forecast-research-warning">
        <AlertTriangle size={21} aria-hidden="true" />
        <div>
          <strong>Research-only forecast</strong>
          <p>
            This screen estimates 24-hour hive-weight change.
            It does not estimate honey maturity, harvest
            readiness, HUI, or a recommended harvesting time.
          </p>
        </div>
      </div>

      <div className="forecast-tab-header">
        <div>
          <span className="eyebrow">
            HISTORICAL TEST-SPLIT CASE STUDY
          </span>
          <h2>Exploratory 24-Hour Weight Forecast</h2>
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
          label="Current weight"
          value={
            latest
              ? `${formatNumber(
                  latest.current_weight_kg,
                  2,
                )} kg`
              : "—"
          }
          icon={Scale}
          note="Latest exported test row"
        />
        <StatCard
          label="Predicted 24h change"
          value={
            latest
              ? `${formatNumber(
                  latest.predicted_delta_kg,
                  3,
                )} kg`
              : "—"
          }
          icon={FlaskConical}
          note="Exploratory model output"
        />
        <StatCard
          label="Displayed MAE"
          value={
            meanAbsoluteError === null
              ? "—"
              : `${formatNumber(
                  meanAbsoluteError,
                  3,
                )} kg`
          }
          icon={AlertTriangle}
          note="Selected hive and displayed period"
        />
      </div>

      <Panel
        title="Actual versus predicted future weight"
        subtitle="Most recent exported test rows for the selected hive."
      >
        <div className="forecast-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={rows}
              margin={{
                top: 10,
                right: 20,
                left: 10,
                bottom: 70,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="displayTime"
                angle={-35}
                textAnchor="end"
                interval="preserveStartEnd"
                height={80}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                label={{
                  value: "Weight (kg)",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="actual_future_weight_kg"
                name="Actual future weight"
                stroke="#0f766e"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="predicted_future_weight_kg"
                name="Predicted future weight"
                stroke="#2563eb"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel
        title="Prospective validation required"
        subtitle="Operational harvesting decisions remain disabled."
      >
        <p className="forecast-prospective-note">
          Collect beekeeper-confirmed inspections, actual harvest
          timestamps, pre- and post-harvest weight, harvested honey
          mass, comb capping and honey moisture before building or
          calibrating a readiness model.
        </p>
      </Panel>
    </section>
  );
}
