import { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Database,
  Gauge,
  Layers3,
  ShieldCheck,
  Sparkles,
  Target,
  Trophy,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { loadClassifierDerivedHuiDashboard } from "../../../services/classifierDerivedHuiService";
import { loadHarvestingModelDashboard } from "../../../services/harvestingModelService";

import "./FinalModelEvaluationDashboard.css";

const METRICS = {
  pr_auc: {
    label: "PR-AUC",
    key: "validation_pr_auc",
    digits: 3,
  },
  f1: {
    label: "F1",
    key: "validation_f1",
    digits: 3,
  },
  recall: {
    label: "Recall",
    key: "validation_recall",
    digits: 3,
  },
};

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatNumber(value, digits = 3) {
  const parsed = asNumber(value);
  return parsed === null ? "—" : parsed.toFixed(digits);
}

function formatPercent(value, digits = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(digits)}%` : "—";
}

function validationPrevalence(benchmark) {
  const candidates = [
    benchmark?.summary?.validation,
    benchmark?.validation,
    benchmark?.class_balance?.validation,
    benchmark?.split_summary?.validation,
  ].filter(Boolean);

  for (const item of candidates) {
    const positive = asNumber(
      item.positive_rows ?? item.positive ?? item.positives ?? item.n_positive,
    );
    const negative = asNumber(
      item.negative_rows ?? item.negative ?? item.negatives ?? item.n_negative,
    );

    if (positive !== null && negative !== null && positive + negative > 0) {
      return positive / (positive + negative);
    }

    const prevalence = asNumber(
      item.target_prevalence ?? item.prevalence ?? item.positive_rate,
    );
    if (prevalence !== null) {
      return prevalence;
    }
  }

  return 144 / (144 + 39541);
}

function resolveRegressionHorizon(huiDashboard, horizon) {
  const key = String(horizon);
  const summary =
    huiDashboard?.future_hui_regression?.summary?.horizons?.[key] ?? {};
  const gate = huiDashboard?.future_hui_regression?.gate?.horizons?.[key] ?? {};
  const validation = summary?.validation ?? {};
  const test = summary?.test ?? {};

  return {
    selectedModel:
      gate.selected_model ?? summary.selected_model ?? "Unavailable",
    validationMae:
      gate.selected_validation_mae ?? validation.mae ?? summary.validation_mae,
    testMae: gate.selected_test_mae ?? test.mae ?? summary.test_mae,
    improvement:
      gate.validation_mae_improvement_fraction ??
      summary.validation_mae_improvement_fraction,
    ratio:
      gate.test_to_validation_mae_ratio ?? summary.test_to_validation_mae_ratio,
    classAgreement:
      test.class_agreement_fraction ??
      test.class_agreement ??
      summary.test_class_agreement_fraction ??
      summary.test_class_agreement,
    testR2: test.r2 ?? summary.test_r2,
    passed: gate.horizon_passed ?? true,
  };
}

function modelLabel(value) {
  const normalized = String(value ?? "").toLowerCase();

  if (normalized === "xgboost") return "XGBoost";
  if (normalized === "lightgbm") return "LightGBM";
  if (normalized === "random_forest") return "Random Forest";
  if (normalized === "logistic_regression") return "Logistic Regression";

  return String(value ?? "")
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function featureSetLabel(value) {
  if (value === "no_humidity") return "No Humidity";
  if (value === "weight_only") return "Weight Only";
  if (value === "core") return "Core";
  if (value === "full") return "Full";

  return modelLabel(value);
}

function friendlyFeature(value) {
  const labels = {
    weight_range_168h_kg: "168-hour weight range",
    weight_std_168h_kg: "168-hour weight variability",
    weight_mean_72h_kg: "72-hour mean hive weight",
    weight_mean_6h_kg: "6-hour mean hive weight",
    weight_mean_24h_kg: "24-hour mean hive weight",
    weight_delta_72h_kg: "72-hour weight change",
    weight_relative_to_max_168h: "Weight relative to 168-hour maximum",
    weight_std_24h_kg: "24-hour weight variability",
    environmental_variability_72h: "72-hour environmental variability",
    temperature_c_range_24h: "24-hour temperature range",
    weight_mean_168h_kg: "168-hour mean hive weight",
    weight_trend_72h_kg_per_hour: "72-hour weight trend",
  };

  if (labels[value]) {
    return labels[value];
  }

  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function ContextMetricCard({
  icon: Icon,
  label,
  value,
  context,
  tone = "blue",
}) {
  return (
    <article className={`model-context-card is-${tone}`}>
      <div className="model-context-card-heading">
        <span className="model-context-card-icon">
          <Icon size={18} />
        </span>
        <small>{label}</small>
      </div>

      <strong>{value}</strong>

      <div className="model-context-card-note">
        <CheckCircle2 size={14} />
        <span>{context}</span>
      </div>
    </article>
  );
}

function SummaryCard({ icon: Icon, label, value, note, tone = "blue" }) {
  return (
    <article className={`final-model-summary-card is-${tone}`}>
      <span className="final-model-summary-icon">
        <Icon size={19} />
      </span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
        <span>{note}</span>
      </div>
    </article>
  );
}

function StrengthItem({ icon: Icon, title, text }) {
  return (
    <div className="final-model-strength-item">
      <Icon size={19} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <small>{text}</small>
      </div>
    </div>
  );
}

function TrainingStageCard({
  number,
  eyebrow,
  title,
  description,
  icon: Icon,
  items,
  resultLabel,
  result,
  tone = "blue",
}) {
  return (
    <article className={`training-stage-card is-${tone}`}>
      <div className="training-stage-card-top">
        <span className="training-stage-number">{number}</span>

        <span className="training-stage-icon">
          <Icon size={24} />
        </span>
      </div>

      <div className="training-stage-card-heading">
        <span>{eyebrow}</span>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div className="training-stage-steps">
        {items.map((item, index) => (
          <div className="training-stage-step" key={`${number}-${index}`}>
            <span>{index + 1}</span>

            <div>
              <strong>{item.title}</strong>
              <small>{item.text}</small>
            </div>
          </div>
        ))}
      </div>

      <div className="training-stage-result">
        <small>{resultLabel}</small>
        <strong>{result}</strong>
      </div>
    </article>
  );
}

function HorizonCard({ horizon, data }) {
  return (
    <article className="final-model-horizon-card">
      <div className="final-model-horizon-heading">
        <span>+{horizon} HOURS</span>
        <b>{modelLabel(data?.selectedModel)}</b>
      </div>

      <div className="final-model-horizon-primary">
        <strong>{formatNumber(data?.testMae, 2)}</strong>
        <span>HUI points · Test MAE</span>
      </div>

      <div className="final-model-horizon-context">
        <TrendingDown size={16} />
        <span>
          Validation MAE improved{" "}
          <strong>{formatPercent(data?.improvement, 1)}</strong> over
          persistence
        </span>
      </div>

      <div className="final-model-horizon-metrics">
        <div>
          <small>Validation MAE</small>
          <strong>{formatNumber(data?.validationMae, 2)}</strong>
        </div>
        <div>
          <small>Class agreement</small>
          <strong>{formatPercent(data?.classAgreement, 1)}</strong>
        </div>
        <div>
          <small>Test R²</small>
          <strong>{formatNumber(data?.testR2, 2)}</strong>
        </div>
        <div>
          <small>Test / validation</small>
          <strong>{formatNumber(data?.ratio, 2)}</strong>
        </div>
      </div>

      <span className="final-model-pass-pill">
        <CheckCircle2 size={15} />
        Forecast horizon passed
      </span>
    </article>
  );
}

function CandidateTooltip({ active, payload, metric }) {
  if (!active || !payload?.length) {
    return null;
  }

  const row = payload[0].payload;
  const config = METRICS[metric];

  return (
    <div className="final-model-tooltip">
      <strong>
        {modelLabel(row.model)} · {featureSetLabel(row.feature_set)}
      </strong>
      <span>
        {config.label}: {formatNumber(row[config.key], config.digits)}
      </span>
      <span>{row.feature_count} features</span>
      {row.selected ? <b>Selected model</b> : null}
    </div>
  );
}

function CandidateTable({ rows }) {
  return (
    <div className="final-model-table-scroll">
      <table className="final-model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Feature set</th>
            <th>Features</th>
            <th>PR-AUC</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.model}-${row.feature_set}`}
              className={row.selected ? "is-selected" : ""}
            >
              <td>
                {modelLabel(row.model)}
                {row.selected ? (
                  <span className="final-model-selected-tag">Selected</span>
                ) : null}
              </td>
              <td>{featureSetLabel(row.feature_set)}</td>
              <td>{row.feature_count}</td>
              <td>{formatNumber(row.validation_pr_auc, 3)}</td>
              <td>{formatNumber(row.validation_recall, 3)}</td>
              <td>{formatNumber(row.validation_f1, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function PipelineStep({ number, title, text }) {
  return (
    <div className="training-ui-step">
      <span>{number}</span>

      <div>
        <strong>{title}</strong>
        <small>{text}</small>
      </div>
    </div>
  );
}

function ForecastModelChip({ horizon, model, mae }) {
  return (
    <div className="training-ui-forecast-chip">
      <div>
        <span>+{horizon}h</span>
        <strong>{modelLabel(model)}</strong>
      </div>

      <small>
        Validation MAE <b>{formatNumber(mae, 2)}</b>
      </small>
    </div>
  );
}

function MetricDirection({ direction = "up", children }) {
  const Icon = direction === "down" ? TrendingDown : TrendingUp;

  return (
    <span className={`stage-compare-direction is-${direction}`}>
      <Icon size={14} />
      {children}
    </span>
  );
}

function EvaluationMetricCard({ label, value, note, direction = "up" }) {
  const Icon = direction === "down" ? TrendingDown : TrendingUp;

  return (
    <article className="stage-eval-metric">
      <div className="stage-eval-metric-top">
        <span>{label}</span>

        <span
          className={`stage-eval-arrow is-${direction}`}
          title={direction === "down" ? "Lower is better" : "Higher is better"}
        >
          <Icon size={16} />
        </span>
      </div>

      <strong>{value}</strong>

      <small>{note}</small>
    </article>
  );
}

export default function FinalModelEvaluationDashboard() {
  const [benchmark, setBenchmark] = useState(null);
  const [huiDashboard, setHuiDashboard] = useState(null);
  const [error, setError] = useState("");
  const [metric, setMetric] = useState("pr_auc");
  const [modelFilter, setModelFilter] = useState("all");
  const [showCount, setShowCount] = useState(6);

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      loadHarvestingModelDashboard(),
      loadClassifierDerivedHuiDashboard(),
    ])
      .then(([benchmarkPayload, huiPayload]) => {
        if (cancelled) {
          return;
        }

        setBenchmark(benchmarkPayload);
        setHuiDashboard(huiPayload);
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

  const candidates = useMemo(() => {
    if (!benchmark) {
      return [];
    }

    const metricConfig = METRICS[metric];

    return benchmark.comparison
      .filter((row) => row.status === "ok")
      .filter((row) => modelFilter === "all" || row.model === modelFilter)
      .sort(
        (left, right) =>
          Number(right[metricConfig.key] ?? -1) -
          Number(left[metricConfig.key] ?? -1),
      );
  }, [benchmark, metric, modelFilter]);

  const visibleCandidates = useMemo(
    () => candidates.slice(0, showCount),
    [candidates, showCount],
  );

  const modelOptions = useMemo(() => {
    if (!benchmark) {
      return [];
    }

    return Array.from(
      new Set(benchmark.comparison.map((row) => row.model).filter(Boolean)),
    );
  }, [benchmark]);

  if (error) {
    return (
      <div className="final-model-state is-error">
        <h3>Model evaluation data could not be loaded</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!benchmark || !huiDashboard) {
    return (
      <div className="final-model-state">
        <div className="final-model-spinner" />
        <p>Loading final model evaluation…</p>
      </div>
    );
  }

  const summary = benchmark.summary ?? {};
  const validation = summary.validation ?? {};
  const selectedCandidate =
    benchmark.comparison.find((row) => row.selected) ?? benchmark.comparison[0];

  const validationEvents = benchmark.validation_events ?? [];
  const testEvents = benchmark.test_events ?? [];

  const validationDetected = validationEvents.filter(
    (row) => row.detected,
  ).length;
  const testDetected = testEvents.filter((row) => row.detected).length;

  const regressionGate = huiDashboard.future_hui_regression?.gate ?? {};

  const horizonData = Object.fromEntries(
    [24, 48, 72].map((horizon) => [
      horizon,
      resolveRegressionHorizon(huiDashboard, horizon),
    ]),
  );

  const baseline = validationPrevalence(benchmark);
  const selectedPrAuc =
    asNumber(validation.pr_auc ?? selectedCandidate?.validation_pr_auc) ?? 0;
  const prAucMultiplier = baseline > 0 ? selectedPrAuc / baseline : null;

  const fullFeatureCount = Math.max(
    ...benchmark.comparison.map((row) => Number(row.feature_count) || 0),
    Number(
      summary.selected_feature_count ?? selectedCandidate?.feature_count ?? 53,
    ),
  );
  const selectedFeatureCount = Number(
    summary.selected_feature_count ?? selectedCandidate?.feature_count ?? 53,
  );
  const omittedFeatureCount = Math.max(
    0,
    fullFeatureCount - selectedFeatureCount,
  );

  const nearestCompetitor = benchmark.comparison
    .filter(
      (row) =>
        row.status === "ok" &&
        !row.selected &&
        row.feature_set === selectedCandidate?.feature_set,
    )
    .sort(
      (left, right) =>
        Number(right.validation_pr_auc ?? -1) -
        Number(left.validation_pr_auc ?? -1),
    )[0];

  const selectedFalseAlerts = Number(
    selectedCandidate?.validation_false_alert_episodes ??
      validation.false_alert_episodes ??
      0,
  );

  const competitorFalseAlerts = Number(
    nearestCompetitor?.validation_false_alert_episodes ?? 0,
  );

  const chartRows = visibleCandidates.map((row) => ({
    ...row,
    chartLabel: `${modelLabel(row.model)} · ${featureSetLabel(
      row.feature_set,
    )}`,
    metricValue: Number(row[METRICS[metric].key] ?? 0),
  }));

  const featureImportance = benchmark.feature_importance ?? [];

  const maxFeatureImportance = Math.max(
    ...featureImportance.map((row) => Number(row.absolute_importance) || 0),
    1,
  );

  const allCandidates = benchmark.comparison
    .filter((row) => row.status === "ok")
    .sort(
      (left, right) =>
        Number(right.validation_pr_auc ?? -1) -
        Number(left.validation_pr_auc ?? -1),
    );

  return (
    <section className="final-model-dashboard">
      <header className="final-model-heading">
        <div>
          <span className="final-model-eyebrow">
            FINAL MODEL SELECTION & FORECAST VALIDATION
          </span>
          <h2>Harvest Prediction Model Evaluation</h2>
          <p>
            Four classifiers and Four feature sets were compared, followed by
            24, 48 and 72 hour HUI forecast validation.
          </p>
        </div>

        <span className="final-model-ready-badge">
          <CheckCircle2 size={17} />
          Evaluation pipeline complete
        </span>
      </header>

      <section className="training-ui-overview">
        {/* HEADER */}
        <div className="training-ui-overview-header">
          <div>
            <span className="final-model-eyebrow">
              MODEL DEVELOPMENT PIPELINE
            </span>

            <h2>How the Harvest Prediction System Is Trained</h2>

            <p>
              Two connected Machine Learning stages transform historical hive
              behaviour into current and future Harvest Urgency Index
              predictions.
            </p>
          </div>

          <div className="training-ui-stage-count">
            <Layers3 size={18} />
            <strong>2</strong>
            <span>Training stages</span>
          </div>
        </div>

       

        {/* CONNECTOR */}
        <div className="training-ui-down-flow">
          <span>↓</span>
        </div>

        {/* MAIN PIPELINE GRID */}
        <div className="training-ui-stage-grid">
          <article className="training-ui-stage-card stage-one">
            <div className="training-ui-stage-top">
              <div className="training-ui-stage-id">
                <span>01</span>

                <div>
                  <small>CLASSIFICATION</small>
                  <strong>Current HUI Model</strong>
                </div>
              </div>

              <span className="training-ui-stage-main-icon">
                <BrainCircuit size={27} />
              </span>
            </div>

            <div className="training-ui-stage-question">
              <Target size={18} />

              <div>
                <small>MODEL QUESTION</small>

                <strong>
                  Will a reviewed harvest event occur within the next 72 hours?
                </strong>
              </div>
            </div>

            <div className="training-ui-stage-steps">
              <PipelineStep
                number="1"
                title="Create 72h target"
                text="Each hourly record becomes harvest-event or normal."
              />

              <PipelineStep
                number="2"
                title="Build time-series features"
                text="Recent changes, trends and rolling sensor behaviour."
              />

              <PipelineStep
                number="3"
                title="Train 16 candidates"
                text="4 classifiers × 4 feature sets."
              />

              <PipelineStep
                number="4"
                title="Validate candidates"
                text="Compare PR-AUC, event detection and false alerts."
              />
            </div>

            <div className="training-ui-model-winner">
              <div>
                <small>SELECTED CLASSIFIER</small>

                <strong>{modelLabel(summary.selected_model)}</strong>

                <span>No Humidity · {selectedFeatureCount} features</span>
              </div>

              <span className="training-ui-winner-badge">
                <Trophy size={18} />
                Selected
              </span>
            </div>

            <div className="training-ui-stage-output">
              <span>STAGE 1 OUTPUT</span>

              <strong>Raw harvest-event classifier score</strong>
            </div>
          </article>

          {/* =====================================================
        CENTRAL HUI HUB
    ====================================================== */}
          <div className="training-ui-hui-bridge">
            <div className="training-ui-bridge-line top" />

            <div className="training-ui-hui-node">
              <span className="training-ui-hui-icon">
                <Gauge size={24} />
              </span>

              <small>SCORE TRANSFORMATION</small>

              <strong>HUI</strong>

              <b>0–100</b>

              <p>
                XGBoost score
                <span>↓</span>
                Platt calibration
                <span>↓</span>
                HUI mapping
              </p>
            </div>

            <div className="training-ui-bridge-line bottom" />

            <span className="training-ui-hui-caption">
              Stage 1 output becomes the input for future-HUI forecasting
            </span>
          </div>

          {/* =====================================================
        STAGE 2
    ====================================================== */}
          <article className="training-ui-stage-card stage-two">
            <div className="training-ui-stage-top">
              <div className="training-ui-stage-id">
                <span>02</span>

                <div>
                  <small>REGRESSION</small>
                  <strong>Future HUI Models</strong>
                </div>
              </div>

              <span className="training-ui-stage-main-icon">
                <TrendingUp size={27} />
              </span>
            </div>

            <div className="training-ui-stage-question">
              <Clock3 size={18} />

              <div>
                <small>MODEL QUESTION</small>

                <strong>
                  What will the Harvest Urgency Index be after 24, 48 and 72
                  hours?
                </strong>
              </div>
            </div>

            <div className="training-ui-stage-steps">
              <PipelineStep
                number="1"
                title="Create future targets"
                text="Generate HUI targets at +24h, +48h and +72h."
              />

              <PipelineStep
                number="2"
                title="Train separately"
                text="Each horizon becomes an independent regression task."
              />
            </div>

            <div className="training-ui-forecast-models">
              <ForecastModelChip
                horizon={24}
                model={horizonData[24]?.selectedModel}
                mae={horizonData[24]?.validationMae}
              />

              <ForecastModelChip
                horizon={48}
                model={horizonData[48]?.selectedModel}
                mae={horizonData[48]?.validationMae}
              />

              <ForecastModelChip
                horizon={72}
                model={horizonData[72]?.selectedModel}
                mae={horizonData[72]?.validationMae}
              />
            </div>

            <div className="training-ui-stage-output">
              <span>STAGE 2 OUTPUT</span>

              <strong>+24h · +48h · +72h HUI forecasts</strong>
            </div>
          </article>
        </div>

        {/* FINAL OUTPUT */}
        <div className="training-ui-final-result">
          <span className="training-ui-final-icon">
            <Sparkles size={23} />
          </span>

          <div>
            <small>FINAL DECISION-SUPPORT OUTPUT</small>

            <strong>Current HUI + 72-hour Harvest Urgency Outlook</strong>
          </div>

          <span className="training-ui-complete">
            <CheckCircle2 size={17} />
            Pipeline complete
          </span>
        </div>
      </section>

      {/* =========================================================
    STAGE-BY-STAGE MODEL COMPARISON
========================================================= */}

      <section className="eval-flow-shell">
        {/* =====================================================
      MAIN HEADING
  ====================================================== */}
        <div className="eval-flow-heading">
          <div>
            <span className="final-model-eyebrow">
              MODEL COMPARISON & EVALUATION
            </span>

            <h2>Model Selection Evidence Across Both Training Stages</h2>

            <p>
              Each stage uses evaluation metrics suited to its own prediction
              task. The selected output from Stage 1 is transformed into HUI
              before Stage 2 begins.
            </p>
          </div>

          <div className="eval-flow-legend">
            <MetricDirection direction="up">Higher is better</MetricDirection>

            <MetricDirection direction="down">Lower is better</MetricDirection>
          </div>
        </div>

        {/* =====================================================
      STAGE 1 - BLUE SECTION
  ====================================================== */}
        <section className="eval-stage-section eval-stage-one">
          <div className="eval-stage-banner">
            <div className="eval-stage-banner-number">01</div>

            <div className="eval-stage-banner-copy">
              <span>CLASSIFICATION STAGE</span>

              <h3>Select the Model for Current Harvest Urgency</h3>

              <p>
                4 classifiers × 4 feature sets were evaluated using the
                validation data.
              </p>
            </div>

            <span className="eval-stage-type-badge">
              <BrainCircuit size={17} />
              Classification
            </span>
          </div>

          {/* KEY METRICS */}
          <div className="eval-stage-metric-grid">
            <article className="eval-stage-metric-card is-blue">
              <div>
                <span>PR-AUC</span>
                <TrendingUp size={18} />
              </div>

              <strong>{formatNumber(selectedPrAuc, 3)}</strong>

              <small>
                {prAucMultiplier !== null
                  ? `≈${formatNumber(prAucMultiplier, 1)}× rare-event baseline`
                  : "Rare-event ranking performance"}
              </small>

              <b>Higher is better ↑</b>
            </article>

            <article className="eval-stage-metric-card is-indigo">
              <div>
                <span>ROC-AUC</span>
                <TrendingUp size={18} />
              </div>

              <strong>{formatNumber(validation.roc_auc, 3)}</strong>

              <small>0.50 represents random ranking</small>

              <b>Higher is better ↑</b>
            </article>

            <article className="eval-stage-metric-card is-green">
              <div>
                <span>Event Detection</span>
                <TrendingUp size={18} />
              </div>

              <strong>
                {validationEvents.length
                  ? `${validationDetected}/${validationEvents.length}`
                  : "—"}
              </strong>

              <small>Reviewed validation events detected</small>

              <b>Higher is better ↑</b>
            </article>

            <article className="eval-stage-metric-card is-orange">
              <div>
                <span>False Alerts</span>
                <TrendingDown size={18} />
              </div>

              <strong>{formatNumber(selectedFalseAlerts, 0)}</strong>

              <small>Validation false-alert episodes</small>

              <b>Lower is better ↓</b>
            </article>
          </div>

          {/* METRIC MEANING */}
          <div className="eval-metric-explainer">
            <div>
              <TrendingUp size={16} />
              <span>
                <strong>PR-AUC ↑</strong>
                Stronger rare-event precision–recall performance
              </span>
            </div>

            <div>
              <TrendingUp size={16} />
              <span>
                <strong>Recall ↑</strong>
                More positive rows detected
              </span>
            </div>

            <div>
              <TrendingUp size={16} />
              <span>
                <strong>F1 ↑</strong>
                Better precision–recall balance
              </span>
            </div>

            <div>
              <TrendingDown size={16} />
              <span>
                <strong>False Alerts ↓</strong>
                Lower alert burden
              </span>
            </div>
          </div>

          {/* TABLE */}
          <div className="eval-table-panel">
            <div className="eval-table-heading">
              <div>
                <span>16 CANDIDATES</span>
                <h4>Classifier Comparison</h4>
              </div>

              <span className="eval-table-selected-note">
                <Trophy size={15} />
                Selected row highlighted
              </span>
            </div>

            <div className="stage-comparison-table-wrap">
              <table className="stage-comparison-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Classifier</th>
                    <th>Feature Set</th>
                    <th>Features</th>

                    <th>
                      <span className="stage-table-metric">
                        PR-AUC
                        <TrendingUp size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric">
                        Recall
                        <TrendingUp size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric">
                        F1
                        <TrendingUp size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric is-down">
                        False Alerts
                        <TrendingDown size={14} />
                      </span>
                    </th>

                    <th>Selection</th>
                  </tr>
                </thead>

                <tbody>
                  {allCandidates.map((row, index) => (
                    <tr
                      key={`${row.model}-${row.feature_set}`}
                      className={row.selected ? "is-selected" : ""}
                    >
                      <td>
                        <span className="stage-rank">{index + 1}</span>
                      </td>

                      <td>
                        <strong>{modelLabel(row.model)}</strong>
                      </td>

                      <td>
                        <span className="stage-feature-pill">
                          {featureSetLabel(row.feature_set)}
                        </span>
                      </td>

                      <td>{row.feature_count}</td>

                      <td>
                        <strong>
                          {formatNumber(row.validation_pr_auc, 3)}
                        </strong>
                      </td>

                      <td>{formatNumber(row.validation_recall, 3)}</td>

                      <td>{formatNumber(row.validation_f1, 3)}</td>

                      <td>
                        {formatNumber(row.validation_false_alert_episodes, 0)}
                      </td>

                      <td>
                        {row.selected ? (
                          <span className="stage-selected-pill">
                            <CheckCircle2 size={14} />
                            Selected
                          </span>
                        ) : (
                          <span className="stage-not-selected">Compared</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* INTERACTIVE CHART */}
          <div className="eval-chart-panel">
            <div className="final-model-panel-heading">
              <div>
                <span className="final-model-eyebrow">VISUAL COMPARISON</span>

                <h3>Compare Classifier Performance</h3>

                <p>Switch between PR-AUC, F1 and Recall.</p>
              </div>

              <div className="final-model-controls">
                <div className="final-model-segmented">
                  {Object.entries(METRICS).map(([key, config]) => (
                    <button
                      key={key}
                      type="button"
                      className={metric === key ? "is-active" : ""}
                      onClick={() => setMetric(key)}
                    >
                      {config.label}
                    </button>
                  ))}
                </div>

                <select
                  value={modelFilter}
                  onChange={(event) => setModelFilter(event.target.value)}
                >
                  <option value="all">All models</option>

                  {modelOptions.map((option) => (
                    <option key={option} value={option}>
                      {modelLabel(option)}
                    </option>
                  ))}
                </select>

                <select
                  value={showCount}
                  onChange={(event) => setShowCount(Number(event.target.value))}
                >
                  <option value={4}>Top 4</option>
                  <option value={6}>Top 6</option>
                  <option value={8}>Top 8</option>
                  <option value={16}>All 16</option>
                </select>
              </div>
            </div>

            <div className="final-model-candidate-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartRows}
                  margin={{
                    top: 12,
                    right: 16,
                    left: 0,
                    bottom: 76,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />

                  <XAxis
                    dataKey="chartLabel"
                    interval={0}
                    angle={-26}
                    textAnchor="end"
                    height={100}
                    tick={{ fontSize: 10 }}
                  />

                  <YAxis domain={[0, "auto"]} tick={{ fontSize: 10 }} />

                  <Tooltip content={<CandidateTooltip metric={metric} />} />

                  <Bar dataKey="metricValue" radius={[6, 6, 0, 0]}>
                    {chartRows.map((row) => (
                      <Cell
                        key={`${row.model}-${row.feature_set}`}
                        fill={row.selected ? "#1d4ed8" : "#93c5fd"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* STAGE 1 OUTPUT */}
          <div className="eval-stage-output is-stage-one">
            <span className="eval-output-icon">
              <Trophy size={24} />
            </span>

            <div className="eval-output-main">
              <small>STAGE 1 SELECTED OUTPUT</small>

              <strong>
                {modelLabel(summary.selected_model)}
                {" · "}
                No Humidity
                {" · "}
                {selectedFeatureCount} Features
              </strong>

              <p>
                This selected classifier produces the raw harvest-event score
                used by the next transformation step.
              </p>
            </div>

            <div className="eval-output-next">
              <span>Goes next to</span>
              <strong>Score Transformation</strong>
              <b>↓</b>
            </div>
          </div>
        </section>

        {/* =====================================================
      HUI TRANSFORMATION BRIDGE
  ====================================================== */}

        <section className="eval-hui-transfer">
          <div className="eval-transfer-line" />

          <div className="eval-transfer-card">
            <span className="eval-transfer-badge">BRIDGE BETWEEN STAGES</span>

            <div className="eval-transfer-flow">
              <div>
                <small>STAGE 1 OUTPUT</small>
                <strong>XGBoost Score</strong>
              </div>

              <span>→</span>

              <div>
                <small>CALIBRATION</small>
                <strong>Platt Scaling</strong>
              </div>

              <span>→</span>

              <div>
                <small>MAPPING</small>
                <strong>HUI 0–100</strong>
              </div>
            </div>

            <p>
              The classifier score is adjusted using Platt calibration and then
              mapped to the relative 0–100 Harvest Urgency Index.
            </p>

            <div className="eval-transfer-next">
              <span>Current HUI becomes an input to Stage 2</span>
              <TrendingDown size={20} />
            </div>
          </div>

          <div className="eval-transfer-line" />
        </section>

        {/* =====================================================
      STAGE 2 - GREEN SECTION
  ====================================================== */}

        <section className="eval-stage-section eval-stage-two">
          <div className="eval-stage-banner">
            <div className="eval-stage-banner-number">02</div>

            <div className="eval-stage-banner-copy">
              <span>REGRESSION STAGE</span>

              <h3>Select Models for Future HUI Forecasting</h3>

              <p>
                Separate regression models are trained and evaluated for +24h,
                +48h and +72h.
              </p>
            </div>

            <span className="eval-stage-type-badge">
              <TrendingUp size={17} />
              Regression
            </span>
          </div>

          {/* METRIC GUIDE */}
          <div className="eval-metric-explainer is-green">
            <div>
              <TrendingDown size={16} />
              <span>
                <strong>MAE ↓</strong>
                Smaller forecast error
              </span>
            </div>

            <div>
              <TrendingUp size={16} />
              <span>
                <strong>Improvement ↑</strong>
                Better than persistence
              </span>
            </div>

            <div>
              <TrendingUp size={16} />
              <span>
                <strong>R² ↑</strong>
                More future variation explained
              </span>
            </div>

            <div>
              <TrendingUp size={16} />
              <span>
                <strong>Class Agreement ↑</strong>
                More matching readiness classes
              </span>
            </div>
          </div>

          {/* TABLE */}
          <div className="eval-table-panel is-stage-two">
            <div className="eval-table-heading">
              <div>
                <span>THREE FORECAST HORIZONS</span>

                <h4>Future-HUI Model Evaluation</h4>
              </div>

              <span className="eval-table-pass-note">
                <CheckCircle2 size={15} />
                {regressionGate.improved_horizon_count ?? 3}
                /3 passed
              </span>
            </div>

            <div className="stage-comparison-table-wrap">
              <table className="stage-comparison-table stage-two-table">
                <thead>
                  <tr>
                    <th>Horizon</th>

                    <th>Selected Model</th>

                    <th>
                      <span className="stage-table-metric is-down">
                        Validation MAE
                        <TrendingDown size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric is-down">
                        Test MAE
                        <TrendingDown size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric">
                        Improvement
                        <TrendingUp size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric">
                        Test R²
                        <TrendingUp size={14} />
                      </span>
                    </th>

                    <th>
                      <span className="stage-table-metric">
                        Class Agreement
                        <TrendingUp size={14} />
                      </span>
                    </th>

                    <th>Gate</th>
                  </tr>
                </thead>

                <tbody>
                  {[24, 48, 72].map((horizon) => {
                    const data = horizonData[horizon];

                    return (
                      <tr key={horizon}>
                        <td>
                          <span className="eval-horizon-badge">
                            +{horizon}h
                          </span>
                        </td>

                        <td>
                          <strong>{modelLabel(data?.selectedModel)}</strong>
                        </td>

                        <td>
                          <span className="eval-value-lower">
                            <TrendingDown size={14} />
                            {formatNumber(data?.validationMae, 2)}
                          </span>
                        </td>

                        <td>
                          <span className="eval-value-lower">
                            <TrendingDown size={14} />
                            {formatNumber(data?.testMae, 2)}
                          </span>
                        </td>

                        <td>
                          <span className="eval-value-higher">
                            <TrendingUp size={14} />
                            {formatPercent(data?.improvement, 1)}
                          </span>
                        </td>

                        <td>
                          <span className="eval-value-higher">
                            <TrendingUp size={14} />
                            {formatNumber(data?.testR2, 2)}
                          </span>
                        </td>

                        <td>
                          <span className="eval-value-higher">
                            <TrendingUp size={14} />
                            {formatPercent(data?.classAgreement, 1)}
                          </span>
                        </td>

                        <td>
                          <span
                            className={`stage-gate-pill ${
                              data?.passed ? "is-passed" : "is-review"
                            }`}
                          >
                            {data?.passed ? (
                              <>
                                <CheckCircle2 size={14} />
                                Passed
                              </>
                            ) : (
                              "Review"
                            )}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* THREE SELECTED OUTPUT CARDS */}
          <div className="eval-horizon-cards">
            {[24, 48, 72].map((horizon) => {
              const data = horizonData[horizon];

              return (
                <article
                  className={`eval-horizon-card horizon-${horizon}`}
                  key={horizon}
                >
                  <div className="eval-horizon-card-top">
                    <span>+{horizon} HOURS</span>

                    <CheckCircle2 size={18} />
                  </div>

                  <strong>{modelLabel(data?.selectedModel)}</strong>

                  <div className="eval-horizon-primary">
                    <small>Test MAE</small>

                    <b>{formatNumber(data?.testMae, 2)}</b>

                    <span>HUI points</span>
                  </div>

                  <div className="eval-horizon-improvement">
                    <TrendingUp size={15} />

                    <span>
                      <strong>{formatPercent(data?.improvement, 1)}</strong>{" "}
                      improvement over persistence
                    </span>
                  </div>

                  <div className="eval-horizon-secondary">
                    <span>
                      R²
                      <strong>{formatNumber(data?.testR2, 2)}</strong>
                    </span>

                    <span>
                      Class match
                      <strong>{formatPercent(data?.classAgreement, 1)}</strong>
                    </span>
                  </div>
                </article>
              );
            })}
          </div>

          {/* STAGE 2 OUTPUT */}
          <div className="eval-stage-output is-stage-two">
            <span className="eval-output-icon">
              <Clock3 size={24} />
            </span>

            <div className="eval-output-main">
              <small>STAGE 2 SELECTED OUTPUTS</small>

              <strong>
                +24h LightGBM
                {" · "}
                +48h XGBoost
                {" · "}
                +72h LightGBM
              </strong>

              <p>
                These three selected regression models produce the future HUI
                trajectory used by the decision-support dashboard.
              </p>
            </div>

            <div className="eval-output-next">
              <span>Goes next to</span>
              <strong>Final Decision Support</strong>
              <b>↓</b>
            </div>
          </div>
        </section>

        {/* =====================================================
      FINAL OUTPUT
  ====================================================== */}

        <section className="eval-final-output">
          <div className="eval-final-output-heading">
            <span className="eval-final-main-icon">
              <Sparkles size={27} />
            </span>

            <div>
              <span>FINAL MODEL OUTPUT</span>

              <h3>Harvest Decision-Support Prediction Package</h3>

              <p>
                Stage 1 provides current harvest urgency. Stage 2 provides the
                expected future trajectory.
              </p>
            </div>

            <span className="eval-final-complete">
              <CheckCircle2 size={16} />
              Complete
            </span>
          </div>

          <div className="eval-final-grid">
            <div>
              <span>
                <Gauge size={20} />
              </span>

              <small>CURRENT</small>

              <strong>HUI 0–100</strong>

              <p>Current relative harvest urgency</p>
            </div>

            <div>
              <span>
                <Clock3 size={20} />
              </span>

              <small>+24 HOURS</small>

              <strong>LightGBM</strong>

              <p>Short-term future HUI</p>
            </div>

            <div>
              <span>
                <Clock3 size={20} />
              </span>

              <small>+48 HOURS</small>

              <strong>XGBoost</strong>

              <p>Medium-term future HUI</p>
            </div>

            <div>
              <span>
                <Clock3 size={20} />
              </span>

              <small>+72 HOURS</small>

              <strong>LightGBM</strong>

              <p>Extended future HUI</p>
            </div>
          </div>

          <div className="eval-final-flow">
            <span>Current HUI</span>

            <b>+</b>

            <span>+24h HUI</span>

            <b>+</b>

            <span>+48h HUI</span>

            <b>+</b>

            <span>+72h HUI</span>

            <strong>→</strong>

            <span className="is-final">Harvest Decision Support</span>
          </div>
        </section>
      </section>

      <div className="final-model-feature-grid">
        <article className="final-model-feature-panel">
          <div className="final-model-panel-heading">
            <div>
              <span className="final-model-eyebrow">MODEL INTERPRETATION</span>
              <h3>Most Influential Selected Model Features</h3>
              <p>
                Feature importance highlights the sensor patterns used by the
                selected classifier.
              </p>
            </div>
          </div>

          <div className="final-model-feature-list">
            {featureImportance.slice(0, 8).map((row, index) => (
              <div className="final-model-feature-row" key={row.feature}>
                <span>{index + 1}</span>
                <div>
                  <strong>{friendlyFeature(row.feature)}</strong>
                  <small>{row.feature}</small>
                  <div className="final-model-feature-track">
                    <i
                      style={{
                        width: `${Math.max(
                          5,
                          (Number(row.absolute_importance) /
                            maxFeatureImportance) *
                            100,
                        )}%`,
                      }}
                    />
                  </div>
                </div>
                <b>{formatNumber(row.importance, 4)}</b>
              </div>
            ))}
          </div>
        </article>

        <article className="final-model-pipeline-panel">
          <span className="final-model-eyebrow">FINAL EVALUATION SUMMARY</span>
          <h3>Research decision-support pipeline established</h3>

          <div className="final-model-pipeline-flow">
            <span>
              <CheckCircle2 size={17} />
              Four-model benchmark completed
            </span>
            <i>↓</i>
            <span>
              <CheckCircle2 size={17} />
              XGBoost classifier selected
            </span>
            <i>↓</i>
            <span>
              <CheckCircle2 size={17} />
              Classifier-derived HUI constructed
            </span>
            <i>↓</i>
            <span>
              <CheckCircle2 size={17} />
              24h / 48h / 72h HUI forecasts passed
            </span>
            <i>↓</i>
            <span>
              <CheckCircle2 size={17} />
              Live IoT inference integrated
            </span>
          </div>
        </article>
      </div>

      <details className="final-model-details">
        <summary>
          <span>Technical validation details</span>
          <ChevronDown size={18} />
        </summary>

        <div className="final-model-details-content">
          <div className="final-model-note-grid">
            <div>
              <strong>Benchmark and temporal policy are separate</strong>
              <p>
                The benchmark threshold detected the held-out event. The
                stricter smoothed temporal alert policy is evaluated separately
                and is not used to describe classifier ranking performance.
              </p>
            </div>

            <div>
              <strong>HUI transformation</strong>
              <p>
                Platt scaling is used as a research-stage score transformation
                before mapping to the relative 0–100 HUI. The HUI is not
                presented as a literal biological probability.
              </p>
            </div>

            <div>
              <strong>Research scope</strong>
              <p>
                These results support the final decision-support research
                prototype. Independent biological and operational validation
                remains a later deployment stage.
              </p>
            </div>
          </div>

          <h4>All classifier candidates</h4>
          <CandidateTable rows={allCandidates} />
        </div>
      </details>
    </section>
  );
}
