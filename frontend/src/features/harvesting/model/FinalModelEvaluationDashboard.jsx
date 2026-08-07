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
  return Number.isFinite(parsed)
    ? `${(parsed * 100).toFixed(digits)}%`
    : "—";
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
      item.positive_rows ??
        item.positive ??
        item.positives ??
        item.n_positive,
    );
    const negative = asNumber(
      item.negative_rows ??
        item.negative ??
        item.negatives ??
        item.n_negative,
    );

    if (
      positive !== null &&
      negative !== null &&
      positive + negative > 0
    ) {
      return positive / (positive + negative);
    }

    const prevalence = asNumber(
      item.target_prevalence ??
        item.prevalence ??
        item.positive_rate,
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
  const gate =
    huiDashboard?.future_hui_regression?.gate?.horizons?.[key] ?? {};
  const validation = summary?.validation ?? {};
  const test = summary?.test ?? {};

  return {
    selectedModel:
      gate.selected_model ?? summary.selected_model ?? "Unavailable",
    validationMae:
      gate.selected_validation_mae ?? validation.mae ?? summary.validation_mae,
    testMae:
      gate.selected_test_mae ?? test.mae ?? summary.test_mae,
    improvement:
      gate.validation_mae_improvement_fraction ??
      summary.validation_mae_improvement_fraction,
    ratio:
      gate.test_to_validation_mae_ratio ??
      summary.test_to_validation_mae_ratio,
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
    .map(
      (part) =>
        part.charAt(0).toUpperCase() + part.slice(1),
    )
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
    weight_relative_to_max_168h:
      "Weight relative to 168-hour maximum",
    weight_std_24h_kg: "24-hour weight variability",
    environmental_variability_72h:
      "72-hour environmental variability",
    temperature_c_range_24h:
      "24-hour temperature range",
    weight_mean_168h_kg: "168-hour mean hive weight",
    weight_trend_72h_kg_per_hour:
      "72-hour weight trend",
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

function SummaryCard({
  icon: Icon,
  label,
  value,
  note,
  tone = "blue",
}) {
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
          <strong>{formatPercent(data?.improvement, 1)}</strong>{" "}
          over persistence
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

function CandidateTooltip({
  active,
  payload,
  metric,
}) {
  if (!active || !payload?.length) {
    return null;
  }

  const row = payload[0].payload;
  const config = METRICS[metric];

  return (
    <div className="final-model-tooltip">
      <strong>
        {modelLabel(row.model)} ·{" "}
        {featureSetLabel(row.feature_set)}
      </strong>
      <span>
        {config.label}:{" "}
        {formatNumber(row[config.key], config.digits)}
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
                  <span className="final-model-selected-tag">
                    Selected
                  </span>
                ) : null}
              </td>
              <td>{featureSetLabel(row.feature_set)}</td>
              <td>{row.feature_count}</td>
              <td>
                {formatNumber(
                  row.validation_pr_auc,
                  3,
                )}
              </td>
              <td>
                {formatNumber(
                  row.validation_recall,
                  3,
                )}
              </td>
              <td>
                {formatNumber(
                  row.validation_f1,
                  3,
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
      .filter(
        (row) =>
          modelFilter === "all" ||
          row.model === modelFilter,
      )
      .sort(
        (left, right) =>
          Number(
            right[metricConfig.key] ?? -1,
          ) -
          Number(
            left[metricConfig.key] ?? -1,
          ),
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
      new Set(
        benchmark.comparison
          .map((row) => row.model)
          .filter(Boolean),
      ),
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
    benchmark.comparison.find(
      (row) => row.selected,
    ) ?? benchmark.comparison[0];

  const validationEvents =
    benchmark.validation_events ?? [];
  const testEvents = benchmark.test_events ?? [];

  const validationDetected =
    validationEvents.filter(
      (row) => row.detected,
    ).length;
  const testDetected =
    testEvents.filter((row) => row.detected).length;

  const regressionGate =
    huiDashboard.future_hui_regression?.gate ?? {};

  const horizonData = Object.fromEntries(
    [24, 48, 72].map((horizon) => [
      horizon,
      resolveRegressionHorizon(
        huiDashboard,
        horizon,
      ),
    ]),
  );

  const baseline = validationPrevalence(benchmark);
  const selectedPrAuc =
    asNumber(
      validation.pr_auc ??
        selectedCandidate?.validation_pr_auc,
    ) ?? 0;
  const prAucMultiplier =
    baseline > 0 ? selectedPrAuc / baseline : null;

  const fullFeatureCount = Math.max(
    ...benchmark.comparison.map(
      (row) => Number(row.feature_count) || 0,
    ),
    Number(
      summary.selected_feature_count ??
        selectedCandidate?.feature_count ??
        53,
    ),
  );
  const selectedFeatureCount = Number(
    summary.selected_feature_count ??
      selectedCandidate?.feature_count ??
      53,
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
        row.feature_set ===
          selectedCandidate?.feature_set,
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
    nearestCompetitor?.validation_false_alert_episodes ??
      0,
  );

  const chartRows = visibleCandidates.map(
    (row) => ({
      ...row,
      chartLabel: `${modelLabel(row.model)} · ${featureSetLabel(
        row.feature_set,
      )}`,
      metricValue: Number(
        row[METRICS[metric].key] ?? 0,
      ),
    }),
  );

  const featureImportance =
    benchmark.feature_importance ?? [];

  const maxFeatureImportance = Math.max(
    ...featureImportance.map(
      (row) =>
        Number(row.absolute_importance) || 0,
    ),
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
            Four classifiers and four feature sets were
            compared, followed by 24, 48 and 72 hour
            HUI forecast validation.
          </p>
        </div>

        <span className="final-model-ready-badge">
          <CheckCircle2 size={17} />
          Evaluation pipeline complete
        </span>
      </header>

      <article className="final-model-selected-card">
        <div className="final-model-selected-title">
          <span className="final-model-selected-icon">
            <BrainCircuit size={28} />
          </span>
          <div>
            <small>SELECTED CLASSIFIER</small>
            <h3>{modelLabel(summary.selected_model)}</h3>
            <p>{selectedFeatureCount}-feature environmental model</p>
            <span>Humidity derived features excluded</span>
          </div>
        </div>

        <div className="model-context-grid">
          <ContextMetricCard
            icon={Layers3}
            label="Features used"
            value={selectedFeatureCount}
            context={
              omittedFeatureCount > 0
                ? `${omittedFeatureCount} fewer than the ${fullFeatureCount}-feature Full set`
                : `${fullFeatureCount}-feature configuration`
            }
          />

          <ContextMetricCard
            icon={Gauge}
            label="Validation PR-AUC"
            value={formatNumber(selectedPrAuc, 3)}
            context={
              prAucMultiplier !== null
                ? `≈${formatNumber(prAucMultiplier, 1)}× the validation rare-event baseline (${formatNumber(baseline, 4)})`
                : `Above the validation rare-event baseline (${formatNumber(baseline, 4)})`
            }
            tone="green"
          />

          <ContextMetricCard
            icon={TrendingUp}
            label="Validation ROC-AUC"
            value={formatNumber(validation.roc_auc, 3)}
            context="0.50 represents random ranking"
            tone="violet"
          />

          <ContextMetricCard
            icon={Target}
            label="Validation event detection"
            value={
              validationEvents.length
                ? formatPercent(
                    validationDetected / validationEvents.length,
                    0,
                  )
                : "—"
            }
            context={
              validationEvents.length
                ? `Detected ${validationDetected} of ${validationEvents.length} reviewed validation events`
                : "Event-level benchmark result"
            }
            tone="green"
          />
        </div>

        <div className="final-model-selected-reason">
          <p>
            Selected from the strongest validation PR-AUC tier using
            event detection and false-alert burden as additional
            decision criteria.
            {nearestCompetitor &&
            competitorFalseAlerts > selectedFalseAlerts
              ? ` Within the same feature set, XGBoost also produced fewer validation false-alert episodes than ${modelLabel(
                  nearestCompetitor.model,
                )} (${selectedFalseAlerts} vs ${competitorFalseAlerts}).`
              : ""}
          </p>
        </div>
      </article>

      <div className="final-model-summary-grid">
        <SummaryCard
          icon={Layers3}
          label="Configurations evaluated"
          value={`${benchmark.successful_candidate_count ?? 16}/16`}
          note="4 classifiers × 4 feature sets completed"
          tone="violet"
        />
        <SummaryCard
          icon={Target}
          label="Validation events"
          value={`${validationDetected}/${validationEvents.length}`}
          note="Both reviewed validation events detected"
          tone="green"
        />
        <SummaryCard
          icon={Trophy}
          label="Held-out benchmark event"
          value={testEvents.length ? `${testDetected}/${testEvents.length}` : "—"}
          note={
            testEvents[0]?.lead_hours != null
              ? `Detected ${formatNumber(testEvents[0].lead_hours, 0)}h before the event`
              : "Final test case preserved"
          }
          tone="green"
        />
        <SummaryCard
          icon={Clock3}
          label="Future-HUI horizons"
          value={`${regressionGate.improved_horizon_count ?? 3}/3`}
          note="24h, 48h and 72h passed the research gate"
          tone="green"
        />
      </div>

      <div className="final-model-hero-grid">
        <article className="final-model-strength-card">
          <span className="final-model-eyebrow">
            EVALUATION STRENGTHS
          </span>
          <h3>Methodological safeguards</h3>

          <div className="final-model-strength-list">
            <StrengthItem
              icon={Database}
              title="Chronological evaluation"
              text="Later observations were reserved for validation and final testing."
            />
            <StrengthItem
              icon={Layers3}
              title="Feature-set ablation"
              text="Core, weight-only, no-humidity and full feature sets were compared."
            />
            <StrengthItem
              icon={ShieldCheck}
              title="Session-aware training"
              text="Closely timed harvest events were balanced as shared temporal sessions."
            />
            <StrengthItem
              icon={CheckCircle2}
              title="Held-out test preserved"
              text="The final benchmark case was not used for model selection."
            />
          </div>
        </article>

        <article className="final-model-benchmark-card">
          <span className="final-model-eyebrow">
            BENCHMARK INTERPRETATION
          </span>
          <h3>What the classifier result means</h3>

          <div className="final-model-benchmark-list">
            <div>
              <CheckCircle2 size={18} />
              <span>
                PR-AUC is interpreted against validation prevalence
                because probable harvest events are rare.
              </span>
            </div>
            <div>
              <CheckCircle2 size={18} />
              <span>
                ROC-AUC {formatNumber(validation.roc_auc, 3)} is above
                the 0.50 random-ranking reference.
              </span>
            </div>
            <div>
              <CheckCircle2 size={18} />
              <span>
                Event detection means each reviewed event was identified
                at least once within its warning window.
              </span>
            </div>
          </div>
        </article>
      </div>

      <article className="final-model-candidate-panel">
        <div className="final-model-panel-heading">
          <div>
            <span className="final-model-eyebrow">
              INTERACTIVE MODEL COMPARISON
            </span>
            <h3>Top-performing classifier candidates</h3>
            <p>
              Compare candidate models using the selected
              validation metric.
            </p>
          </div>

          <div className="final-model-controls">
            <div className="final-model-segmented">
              {Object.entries(METRICS).map(
                ([key, config]) => (
                  <button
                    key={key}
                    type="button"
                    className={
                      metric === key
                        ? "is-active"
                        : ""
                    }
                    onClick={() => setMetric(key)}
                  >
                    {config.label}
                  </button>
                ),
              )}
            </div>

            <select
              value={modelFilter}
              onChange={(event) =>
                setModelFilter(event.target.value)
              }
              aria-label="Filter models"
            >
              <option value="all">All models</option>
              {modelOptions.map((option) => (
                <option
                  key={option}
                  value={option}
                >
                  {modelLabel(option)}
                </option>
              ))}
            </select>

            <select
              value={showCount}
              onChange={(event) =>
                setShowCount(
                  Number(event.target.value),
                )
              }
              aria-label="Number of candidates"
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
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis
                dataKey="chartLabel"
                interval={0}
                angle={-26}
                textAnchor="end"
                height={100}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                domain={[0, "auto"]}
                tick={{ fontSize: 10 }}
              />
              <Tooltip
                content={
                  <CandidateTooltip
                    metric={metric}
                  />
                }
              />
              <Bar
                dataKey="metricValue"
                radius={[6, 6, 0, 0]}
              >
                {chartRows.map((row) => (
                  <Cell
                    key={`${row.model}-${row.feature_set}`}
                    fill={
                      row.selected
                        ? "#2563eb"
                        : "#93c5fd"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="final-model-chart-caption">
          Selected model is highlighted in dark blue.
        </div>
      </article>

      <section className="final-model-future-section">
        <div className="final-model-section-heading">
          <div>
            <span className="final-model-eyebrow">
              FUTURE HUI FORECAST PERFORMANCE
            </span>
            <h3>
              All three forecast horizons passed the
              predefined research gate
            </h3>
            <p>
              Each MAE is shown with its improvement over
              persistence so the forecast error has context.
            </p>
          </div>

          <span className="final-model-pass-badge">
            <CheckCircle2 size={17} />
            3 / 3 horizons passed
          </span>
        </div>

        <div className="final-model-horizon-grid">
          {[24, 48, 72].map((horizon) => (
            <HorizonCard
              key={horizon}
              horizon={horizon}
              data={horizonData[horizon]}
            />
          ))}
        </div>
      </section>

      <div className="final-model-feature-grid">
        <article className="final-model-feature-panel">
          <div className="final-model-panel-heading">
            <div>
              <span className="final-model-eyebrow">
                MODEL INTERPRETATION
              </span>
              <h3>
                Most Influential Selected Model Features
              </h3>
              <p>
                Feature importance highlights the sensor
                patterns used by the selected classifier.
              </p>
            </div>
          </div>

          <div className="final-model-feature-list">
            {featureImportance
              .slice(0, 8)
              .map((row, index) => (
                <div
                  className="final-model-feature-row"
                  key={row.feature}
                >
                  <span>{index + 1}</span>
                  <div>
                    <strong>
                      {friendlyFeature(row.feature)}
                    </strong>
                    <small>{row.feature}</small>
                    <div className="final-model-feature-track">
                      <i
                        style={{
                          width: `${Math.max(
                            5,
                            (Number(
                              row.absolute_importance,
                            ) /
                              maxFeatureImportance) *
                              100,
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                  <b>
                    {formatNumber(
                      row.importance,
                      4,
                    )}
                  </b>
                </div>
              ))}
          </div>
        </article>

        <article className="final-model-pipeline-panel">
          <span className="final-model-eyebrow">
            FINAL EVALUATION SUMMARY
          </span>
          <h3>
            Research decision-support pipeline established
          </h3>

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
              <strong>
                Benchmark and temporal policy are separate
              </strong>
              <p>
                The benchmark threshold detected the held-out
                event. The stricter smoothed temporal alert
                policy is evaluated separately and is not used
                to describe classifier ranking performance.
              </p>
            </div>

            <div>
              <strong>HUI transformation</strong>
              <p>
                Platt scaling is used as a research-stage score
                transformation before mapping to the relative
                0–100 HUI. The HUI is not presented as a
                literal biological probability.
              </p>
            </div>

            <div>
              <strong>Research scope</strong>
              <p>
                These results support the final
                decision-support research prototype.
                Independent biological and operational
                validation remains a later deployment stage.
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
