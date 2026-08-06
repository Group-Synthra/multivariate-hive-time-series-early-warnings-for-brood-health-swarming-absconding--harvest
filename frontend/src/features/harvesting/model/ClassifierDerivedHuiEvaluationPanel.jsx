import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Gauge,
  ShieldCheck,
} from "lucide-react";

import { Panel } from "../../../components/common/Panel";
import { StatCard } from "../../../components/common/StatCard";
import { loadClassifierDerivedHuiDashboard } from "../../../services/classifierDerivedHuiService";

import "./ClassifierDerivedHuiEvaluationPanel.css";

function formatDecimal(value, digits = 3) {
  if (value === null || value === undefined) {
    return "—";
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function modelLabel(value) {
  return String(value ?? "")
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function ClassifierDerivedHuiEvaluationPanel() {
  const [dashboard, setDashboard] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    loadClassifierDerivedHuiDashboard()
      .then((payload) => {
        if (!cancelled) {
          setDashboard(payload);
          setError("");
        }
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

  const calibrationRows = useMemo(() => {
    if (!dashboard) {
      return {};
    }

    const rows = dashboard.calibration?.comparison ?? [];
    const result = {};
    rows.forEach((row) => {
      result[`${row.method}-${row.split}`] = row;
    });
    return result;
  }, [dashboard]);

  if (error) {
    return (
      <Panel title="Classifier-Derived HUI Evaluation">
        <p className="classifier-hui-evaluation-error">{error}</p>
      </Panel>
    );
  }

  if (!dashboard) {
    return (
      <Panel title="Classifier-Derived HUI Evaluation">
        <p>Loading HUI evaluation values…</p>
      </Panel>
    );
  }

  const regression = dashboard.future_hui_regression;
  const gate = regression.gate;
  const horizons = regression.summary.horizons;
  const calibrationGate = dashboard.calibration.gate;
  const identityValidation = calibrationRows["identity-validation"];
  const plattValidation = calibrationRows["platt-validation"];
  const plattTest = calibrationRows["platt-test"];

  return (
    <section className="classifier-hui-evaluation-section">
      <div className="classifier-hui-evaluation-heading">
        <div>
          <span className="eyebrow">FINAL VIVA EVALUATION</span>
          <h3>Classifier-Derived HUI and Future-HUI Validation</h3>
          <p>
            The existing classifier comparison remains unchanged. This
            section evaluates the score transformation used for current HUI
            and the regression models used for 24-, 48- and 72-hour HUI
            forecasts.
          </p>
        </div>
        <span
          className={`classifier-hui-gate-badge ${
            gate.gate_passed ? "is-success" : "is-warning"
          }`}
        >
          {gate.gate_passed
            ? "Future-HUI research gate passed"
            : "Future-HUI evidence limited"}
        </span>
      </div>

      <div className="stats-grid stats-grid-six">
        <StatCard
          label="Calibration method"
          value={modelLabel(calibrationGate.selected_method)}
          icon={Gauge}
          note="Research-stage score transformation"
        />
        <StatCard
          label="Validation Brier"
          value={formatDecimal(plattValidation?.brier_score, 4)}
          icon={ShieldCheck}
          note={`${formatPercent(
            calibrationGate.validation_brier_improvement_fraction,
            1,
          )} improvement over raw score`}
        />
        <StatCard
          label="Calibration gate"
          value={calibrationGate.gate_passed ? "Passed" : "Limited"}
          icon={calibrationGate.gate_passed ? CheckCircle2 : AlertTriangle}
          note="Validation ECE prevented an operational claim"
        />
        <StatCard
          label="24h test MAE"
          value={formatDecimal(horizons["24"]?.test?.mae, 2)}
          icon={Gauge}
          note={`${formatPercent(
            horizons["24"]?.test?.readiness_class_agreement_fraction,
            1,
          )} class agreement`}
        />
        <StatCard
          label="48h test MAE"
          value={formatDecimal(horizons["48"]?.test?.mae, 2)}
          icon={Gauge}
          note={`${formatPercent(
            horizons["48"]?.test?.readiness_class_agreement_fraction,
            1,
          )} class agreement`}
        />
        <StatCard
          label="72h test MAE"
          value={formatDecimal(horizons["72"]?.test?.mae, 2)}
          icon={Gauge}
          note={`${formatPercent(
            horizons["72"]?.test?.readiness_class_agreement_fraction,
            1,
          )} class agreement`}
        />
      </div>

      <div className="two-column-grid">
        <Panel
          title="Probability-calibration evaluation"
          subtitle="Platt scaling was selected using training OOF and validation evidence; the test split was not used for selection."
        >
          <div className="table-scroll">
            <table className="model-results-table">
              <thead>
                <tr>
                  <th>Evaluation</th>
                  <th>Brier</th>
                  <th>Log loss</th>
                  <th>ECE</th>
                  <th>Slope</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Raw validation score</td>
                  <td>{formatDecimal(identityValidation?.brier_score, 6)}</td>
                  <td>{formatDecimal(identityValidation?.log_loss, 6)}</td>
                  <td>
                    {formatDecimal(
                      identityValidation?.expected_calibration_error,
                      6,
                    )}
                  </td>
                  <td>{formatDecimal(identityValidation?.calibration_slope, 3)}</td>
                </tr>
                <tr className="is-selected">
                  <td>Platt validation</td>
                  <td>{formatDecimal(plattValidation?.brier_score, 6)}</td>
                  <td>{formatDecimal(plattValidation?.log_loss, 6)}</td>
                  <td>
                    {formatDecimal(
                      plattValidation?.expected_calibration_error,
                      6,
                    )}
                  </td>
                  <td>{formatDecimal(plattValidation?.calibration_slope, 3)}</td>
                </tr>
                <tr>
                  <td>Platt held-out test</td>
                  <td>{formatDecimal(plattTest?.brier_score, 6)}</td>
                  <td>{formatDecimal(plattTest?.log_loss, 6)}</td>
                  <td>
                    {formatDecimal(
                      plattTest?.expected_calibration_error,
                      6,
                    )}
                  </td>
                  <td>{formatDecimal(plattTest?.calibration_slope, 3)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="classifier-hui-evaluation-note">
            Brier score improved, but validation ECE increased slightly.
            Therefore, Platt output is used only to construct a provisional
            relative HUI and is not presented as an operational probability.
          </p>
        </Panel>

        <Panel
          title="Future-HUI regression evaluation"
          subtitle="Models were selected by validation MAE and compared with current-HUI persistence."
        >
          <div className="table-scroll">
            <table className="model-results-table">
              <thead>
                <tr>
                  <th>Horizon</th>
                  <th>Selected model</th>
                  <th>Validation MAE</th>
                  <th>Test MAE</th>
                  <th>Test R²</th>
                  <th>Within ±5</th>
                  <th>Class agreement</th>
                </tr>
              </thead>
              <tbody>
                {[24, 48, 72].map((horizon) => {
                  const result = horizons[String(horizon)];
                  return (
                    <tr key={horizon}>
                      <td>{horizon}h</td>
                      <td>{modelLabel(result.selected_model)}</td>
                      <td>{formatDecimal(result.validation.mae, 3)}</td>
                      <td>{formatDecimal(result.test.mae, 3)}</td>
                      <td>{formatDecimal(result.test.r2, 3)}</td>
                      <td>
                        {formatPercent(
                          result.test.within_5_points_fraction,
                          1,
                        )}
                      </td>
                      <td>
                        {formatPercent(
                          result.test.readiness_class_agreement_fraction,
                          1,
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel
        title="Validation conclusion for the viva"
        subtitle="The result supports a research prototype while preserving operational limitations."
      >
        <div className="classifier-hui-validation-conclusion">
          <CheckCircle2 size={22} aria-hidden="true" />
          <p>
            All three future-HUI horizons improved validation MAE by more
            than the predefined 5% requirement and maintained acceptable
            test-to-validation error ratios. The gate therefore supports the
            final viva dashboard. It does not establish independent honey
            maturity, biological validity or operational deployment.
          </p>
        </div>
      </Panel>
    </section>
  );
}
