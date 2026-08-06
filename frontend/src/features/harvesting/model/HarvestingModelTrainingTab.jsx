import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Database,
  Gauge,
  Layers3,
  ShieldCheck,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import HarvestingFinalResearchPanel from "./HarvestingFinalResearchPanel";
import ClassifierDerivedHuiEvaluationPanel from "./ClassifierDerivedHuiEvaluationPanel";

import { Panel } from "../../../components/common/Panel";
import { StatCard } from "../../../components/common/StatCard";
import { loadHarvestingBenchmarkDashboard } from "../../../services/harvestingBenchmarkService";
import { loadHarvestingModelDashboard } from "../../../services/harvestingModelService";

import "./HarvestingModelTrainingTab.css";

const NUMBER_FORMAT = new Intl.NumberFormat("en-US");

function formatNumber(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  return NUMBER_FORMAT.format(value);
}

function formatDecimal(value, digits = 3) {
  if (value === null || value === undefined) {
    return "—";
  }
  return Number(value).toFixed(digits);
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
    .map(
      (part) =>
        part.charAt(0).toUpperCase() + part.slice(1),
    )
    .join(" ");
}

function featureSetLabel(value) {
  return modelLabel(value);
}

function EventDetectionTable({ rows, title, emptyText }) {
  return (
    
    <Panel title={title}>
      {rows.length === 0 ? (
        <div className="model-empty-state">{emptyText}</div>
      ) : (
        <div className="table-scroll">
          <table className="model-results-table">
            <thead>
              <tr>
                <th>Hive</th>
                <th>Event time</th>
                <th>Detected</th>
                <th>Lead time</th>
                <th>Maximum score</th>
                <th>Alert rows</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={`${row.harvest_event_id}-${row.hive_id}`}
                >
                  <td>{row.hive_id}</td>
                  <td>{row.event_start ?? "—"}</td>
                  <td>
                    <span
                      className={`model-status-pill ${
                        row.detected ? "is-success" : "is-danger"
                      }`}
                    >
                      {row.detected ? "Detected" : "Missed"}
                    </span>
                  </td>
                  <td>
                    {row.lead_hours === null
                      ? "—"
                      : `${formatDecimal(row.lead_hours, 1)} h`}
                  </td>
                  <td>
                    {formatDecimal(
                      row.maximum_probability,
                      3,
                    )}
                  </td>
                  <td>{formatNumber(row.alert_rows)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function QualityItem({ icon: Icon, title, text, tone = "good" }) {
  return (
    <div className={`model-quality-item is-${tone}`}>
      <Icon size={19} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <p>{text}</p>
      </div>
    </div>
  );
}

export default function HarvestingModelTrainingTab() {
  const [dashboard, setDashboard] = useState(null);
  const [researchDashboard, setResearchDashboard] =
    useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [featureSetFilter, setFeatureSetFilter] =
    useState("all");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setIsLoading(true);
        const [modelResult, researchResult] =
          await Promise.all([
            loadHarvestingModelDashboard(),
            loadHarvestingBenchmarkDashboard(),
          ]);

        if (!cancelled) {
          setDashboard(modelResult);
          setResearchDashboard(researchResult);
          setError("");
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const successfulCandidates = useMemo(() => {
    if (!dashboard) {
      return [];
    }

    return dashboard.comparison
      .filter((row) => row.status === "ok")
      .filter(
        (row) =>
          featureSetFilter === "all" ||
          row.feature_set === featureSetFilter,
      )
      .sort(
        (left, right) =>
          Number(right.validation_pr_auc ?? -1) -
          Number(left.validation_pr_auc ?? -1),
      );
  }, [dashboard, featureSetFilter]);

  const chartRows = useMemo(
    () =>
      successfulCandidates.map((row) => ({
        name: `${modelLabel(row.model)} · ${featureSetLabel(
          row.feature_set,
        )}`,
        prAuc: Number(row.validation_pr_auc ?? 0),
        eventRecall: Number(
          row.validation_event_recall ?? 0,
        ),
        selected: Boolean(row.selected),
      })),
    [successfulCandidates],
  );

  const featureSetOptions = useMemo(() => {
    if (!dashboard) {
      return [];
    }

    return Array.from(
      new Set(
        dashboard.comparison
          .map((row) => row.feature_set)
          .filter(Boolean),
      ),
    );
  }, [dashboard]);

  if (isLoading) {
    return (
      <div className="model-dashboard-state">
        <div className="model-dashboard-spinner" />
        <p>Loading research model comparison…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="model-dashboard-state is-error">
        <AlertTriangle size={34} aria-hidden="true" />
        <h3>Model dashboard data is not available</h3>
        <p>{error}</p>
        <code>
          python scripts/export_harvest_model_results_for_frontend.py
        </code>
        <code>
          python scripts/export_harvesting_benchmark_dashboard.py
        </code>
      </div>
    );
  }

  const summary = dashboard.summary;
  const validation = summary.validation;
  const test = summary.test;
  const robustness = summary.grouped_hive_robustness;
  const classificationDecision = researchDashboard?.decision?.classification_branch;


  return (
    <section className="harvesting-model-dashboard">

      
      <div className="model-dashboard-heading">
        <div>
          <span className="eyebrow">
            SESSION-AWARE FOUR-MODEL COMPARISON
          </span>
          <h2>Reviewed Harvest Model Evaluation</h2>
          <p>
            Logistic Regression, Random Forest, XGBoost and
            LightGBM evaluated across core, weight-only,
            no-humidity and full feature sets.
          </p>
        </div>
        <span className="model-stage-badge">
          Benchmark evaluation complete
        </span>
      </div>

      <HarvestingFinalResearchPanel
        dashboard={researchDashboard}
      />

      <ClassifierDerivedHuiEvaluationPanel />

      <div className="stats-grid stats-grid-six">
        <StatCard
          label="Selected model"
          value={modelLabel(summary.selected_model)}
          icon={BrainCircuit}
          note={featureSetLabel(
            summary.selected_feature_set,
          )}
        />
        <StatCard
          label="Validation PR-AUC"
          value={formatDecimal(validation.pr_auc, 3)}
          icon={Gauge}
          note="Primary row-level ranking metric"
        />
        <StatCard
          label="Benchmark validation events"
          value={`${formatNumber(
            summary.event_counts.validation ?? 0,
          )}`}
          icon={CheckCircle2}
          note={`${formatPercent(
          validation.event_recall,
          0,
          )} selected-threshold event recall`}
         />

         
        <StatCard
          label="False-alert episodes"
          value={formatNumber(
            validation.false_alert_episodes,
          )}
          icon={AlertTriangle}
          note="Validation split"
        />
        <StatCard
  label="Held-out policy test"
  value={
    classificationDecision
      ? `${formatNumber(
          classificationDecision.test_detected_event_count,
        )}/${formatNumber(
          classificationDecision.test_event_count,
        )}`
      : "—"
  }
  icon={Activity}
  note="Research-safe temporal policy"
/>
       
        <StatCard
          label="Temporal sessions"
          value={formatNumber(
            Object.values(
              summary.session_counts ?? {},
            ).reduce(
              (total, value) => total + Number(value),
              0,
            ),
          )}
          icon={Clock3}
          note="Independent decision periods"
        />
      </div>

      <div className="two-column-grid">
        <Panel
          title="Selected candidate performance"
          subtitle="Benchmark threshold fixed from the official validation split; not an operational alert policy."
        >
          <div className="model-metric-grid">
            <div>
              <span>Precision</span>
              <strong>
                {formatDecimal(validation.precision)}
              </strong>
            </div>
            <div>
              <span>Recall</span>
              <strong>
                {formatDecimal(validation.recall)}
              </strong>
            </div>
            <div>
              <span>F1-score</span>
              <strong>
                {formatDecimal(validation.f1)}
              </strong>
            </div>
            <div>
              <span>ROC-AUC</span>
              <strong>
                {formatDecimal(validation.roc_auc)}
              </strong>
            </div>
            <div>
              <span>Brier score</span>
              <strong>
                {formatDecimal(validation.brier_score)}
              </strong>
            </div>
            <div>
              <span>Threshold</span>
              <strong>
                {formatDecimal(summary.selected_threshold)}
              </strong>
            </div>
          </div>
        </Panel>

        <Panel
          title="Research safeguards"
          subtitle="Checks used to reduce optimistic row-level conclusions."
        >
          <div className="model-quality-list">
            <QualityItem
              icon={Layers3}
              title="Feature ablation included"
              text="Core, weight-only, no-humidity and full feature sets were compared."
            />
            <QualityItem
              icon={Database}
              title="Harvest sessions balanced"
              text="Closely timed hive events were grouped and weighted as shared harvesting sessions."
            />
            <QualityItem
              icon={ShieldCheck}
              title="Official test preserved"
              text="The selected candidate and threshold were applied unchanged to the one-event test split."
            />
            <QualityItem
              icon={AlertTriangle}
              title="Classifier-derived HUI evaluated"
              text="Platt calibration remains research-stage, while future-HUI regression passed the predefined viva gate at 24, 48 and 72 hours."
              tone="warning"
            />
          </div>
        </Panel>
      </div>

      <Panel
        title="All model and feature-set candidates"
        subtitle={`${formatNumber(
          dashboard.successful_candidate_count,
        )} successful candidates from ${formatNumber(
          dashboard.candidate_count,
        )} attempted configurations.`}
        action={
          <label className="model-filter">
            <span>Feature set</span>
            <select
              value={featureSetFilter}
              onChange={(event) =>
                setFeatureSetFilter(event.target.value)
              }
            >
              <option value="all">All feature sets</option>
              {featureSetOptions.map((option) => (
                <option key={option} value={option}>
                  {featureSetLabel(option)}
                </option>
              ))}
            </select>
          </label>
        }
      >
        <div className="model-comparison-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartRows}
              margin={{
                top: 10,
                right: 20,
                left: 0,
                bottom: 95,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                angle={-38}
                textAnchor="end"
                interval={0}
                height={110}
                tick={{ fontSize: 11 }}
              />
              <YAxis domain={[0, 1]} />
              <Tooltip
                formatter={(value) =>
                  formatDecimal(value, 3)
                }
              />
              <Legend />
              <Bar
                dataKey="prAuc"
                name="Validation PR-AUC"
                fill="#2563eb"
                radius={[5, 5, 0, 0]}
              />
              <Bar
                dataKey="eventRecall"
                name="Validation event recall"
                fill="#d97706"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="table-scroll">
          <table className="model-results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Feature set</th>
                <th>Features</th>
                <th>PR-AUC</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Event recall</th>
                <th>False alerts</th>
              </tr>
            </thead>
            <tbody>
              {successfulCandidates.map((row) => (
                <tr
                  className={
                    row.selected ? "is-selected" : ""
                  }
                  key={`${row.model}-${row.feature_set}`}
                >
                  <td>
                    {modelLabel(row.model)}
                    {row.selected ? (
                      <span className="selected-candidate-badge">
                        Selected
                      </span>
                    ) : null}
                  </td>
                  <td>{featureSetLabel(row.feature_set)}</td>
                  <td>{formatNumber(row.feature_count)}</td>
                  <td>
                    {formatDecimal(
                      row.validation_pr_auc,
                    )}
                  </td>
                  <td>
                    {formatDecimal(
                      row.validation_precision,
                    )}
                  </td>
                  <td>
                    {formatDecimal(
                      row.validation_recall,
                    )}
                  </td>
                  <td>
                    {formatDecimal(row.validation_f1)}
                  </td>
                  <td>
                    {formatPercent(
                      row.validation_event_recall,
                      0,
                    )}
                  </td>
                  <td>
                    {formatNumber(
                      row.validation_false_alert_episodes,
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="two-column-grid">
        <EventDetectionTable
          rows={dashboard.validation_events}
          title="Benchmark validation event detection"
          emptyText="No validation-event output was exported."
        />
        <EventDetectionTable
          rows={dashboard.test_events}
          title="Benchmark test event case study"
          emptyText="No test-event output was exported."
        />
      </div>

      <div className="two-column-grid">
        <Panel
          title="Grouped hive robustness"
          subtitle="Leave-one-positive-hive-out secondary sensitivity analysis."
        >
          {dashboard.grouped_hive_robustness.length === 0 ? (
            <div className="model-empty-state">
              No grouped-hive robustness rows were exported.
            </div>
          ) : (
            <div className="table-scroll">
              <table className="model-results-table">
                <thead>
                  <tr>
                    <th>Held-out hive</th>
                    <th>PR-AUC</th>
                    <th>F1</th>
                    <th>Event recall</th>
                    <th>False alerts</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboard.grouped_hive_robustness.map(
                    (row) => (
                      <tr key={row.held_out_hive}>
                        <td>{row.held_out_hive}</td>
                        <td>{formatDecimal(row.pr_auc)}</td>
                        <td>{formatDecimal(row.f1)}</td>
                        <td>
                          {formatPercent(
                            row.event_recall,
                            0,
                          )}
                        </td>
                        <td>
                          {formatNumber(
                            row.false_alert_episodes,
                          )}
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel
          title="Top selected-model features"
          subtitle="Coefficients or tree-based feature importance, depending on the selected model."
        >
          <div className="model-feature-list">
            {dashboard.feature_importance
              .slice(0, 12)
              .map((row, index) => (
                <div
                  className="model-feature-row"
                  key={row.feature}
                >
                  <span>{index + 1}</span>
                  <div>
                    <strong>{row.feature}</strong>
                    <div className="model-feature-track">
                      <div
                        style={{
                          width: `${Math.max(
                            4,
                            Math.min(
                              100,
                              (Number(
                                row.absolute_importance,
                              ) /
                                Math.max(
                                  ...dashboard.feature_importance.map(
                                    (item) =>
                                      Number(
                                        item.absolute_importance,
                                      ) || 0,
                                  ),
                                  1,
                                )) *
                                100,
                            ),
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                  <b>
                    {formatDecimal(
                      row.importance,
                      4,
                    )}
                  </b>
                </div>
              ))}
          </div>
        </Panel>
      </div>

      <Panel
        title="Prospective validation pathway"
        subtitle="Operational readiness remains blocked until independent beekeeper-confirmed evidence is available."
      >
        <div className="model-next-step-flow">
          <span className="is-complete">
            Four-model benchmark complete
          </span>
          <i>→</i>
          <span className="is-complete">
            Generalization review complete
          </span>
          <i>→</i>
          <span>Prospective harvest records</span>
          <i>→</i>
          <span>Independent external validation</span>
          <i>→</i>
          <span>Future calibration review</span>
        </div>
      </Panel>
    </section>
  );
}
