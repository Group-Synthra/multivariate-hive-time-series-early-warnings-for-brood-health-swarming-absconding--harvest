import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Activity,
  ShieldCheck,
  Target,
  Trophy,
} from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { useBroodTraining } from '../hooks/useBroodHealthData';
import { asPercent, numberValue } from '../utils/broodHealth';
import {
  ActualPredictedScoreChart,
  FeatureImportanceChart,
  HorizonErrorChart,
  ModelComparisonChart,
  ModelErrorComparisonChart,
  PersistenceComparisonChart,
} from './BroodHealthCharts';
import { HealthLevelClassificationChart } from './HealthLevelClassificationChart';
import { BroodReportGallery } from './BroodReportGallery';
import { BroodFormulaReference } from './BroodFormulaReference';

function errorPoints(value, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '—';
  return `${numberValue(numeric, digits)} points`;
}

function ModelComparisonTable({ models, bestModel }) {
  return (
    <div className="table-scroll">
      <table className="brood-model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>MAE ↓</th>
            <th>RMSE ↓</th>
            <th>R² ↑</th>
            <th>Accuracy ↑</th>
            <th>Transition Accuracy ↑</th>
            <th>Deterioration Recall ↑</th>
            <th>Critical Recall ↑</th>
            <th>BHSI Accuracy ↑</th>
            <th>RoD Trend Accuracy ↑</th>
          </tr>
        </thead>
        <tbody>
          {(models || []).map((row) => {
            const exact = row.test?.exact_horizon || {};
            const transition = row.test?.transition || {};
            const deterioration = row.test?.deterioration || {};
            const selected = row.model === bestModel;
            return (
              <tr
                key={row.model}
                className={`${row.status !== 'ok' ? 'failed-row' : ''} ${selected ? 'selected-row' : ''}`}
              >
                <td>
                  <strong>{row.model}</strong>
                  {selected && <span className="brood-selected-badge">Selected</span>}
                </td>
                {row.status !== 'ok' ? (
                  <td colSpan={9}>{row.error || 'Model failed'}</td>
                ) : (
                  <>
                    <td>{errorPoints(exact.mae)}</td>
                    <td>{errorPoints(exact.rmse)}</td>
                    <td>{asPercent(exact.r2)}</td>
                    <td>{asPercent(exact.health_level_accuracy)}</td>
                    <td><strong>{asPercent(transition.health_level_accuracy)}</strong></td>
                    <td>{asPercent(deterioration.recall)}</td>
                    <td>{asPercent(exact.critical_recall)}</td>
                    <td>{asPercent(row.test?.forecast_indicators?.forecast_bhsi_level_accuracy)}</td>
                    <td>{asPercent(row.test?.forecast_indicators?.forecast_trend_accuracy)}</td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SelectedModelSummary({ modelName, exact, transition, deterioration, forecastIndicators }) {
  const groups = [
    {
      title: 'Score Prediction',
      icon: Target,
      metrics: [
        ['MAE ↓', errorPoints(exact.mae)],
        ['RMSE ↓', errorPoints(exact.rmse)],
        ['R² ↑', asPercent(exact.r2)],
      ],
    },
    {
      title: 'Health Classification',
      icon: ShieldCheck,
      metrics: [
        ['Accuracy ↑', asPercent(exact.health_level_accuracy)],
        ['Transition Accuracy ↑', asPercent(transition.health_level_accuracy)],
        ['Deterioration Recall ↑', asPercent(deterioration.recall)],
        ['Critical Recall ↑', asPercent(exact.critical_recall)],
      ],
    },
    {
      title: 'Future Indicators',
      icon: Activity,
      metrics: [
        ['BHSI Accuracy ↑', asPercent(forecastIndicators.forecast_bhsi_level_accuracy)],
        ['RoD Trend Accuracy ↑', asPercent(forecastIndicators.forecast_trend_accuracy)],
      ],
    },
  ];

  return (
    <Panel
      title="Selected Model Performance"
      subtitle="Final test performance of the model selected during validation."
    >
      <div className="brood-best-model-spotlight">
        <div className="brood-best-model-icon">
          <Trophy size={28} />
        </div>

        <div className="brood-best-model-copy">
          <span>Selected Best Model</span>
          <strong>{modelName || '—'}</strong>
          <small>Chosen after comparing all candidate models on validation performance.</small>
        </div>

        <div className="brood-best-model-status">
          <CheckCircle2 size={17} />
          Best Model
        </div>
      </div>

      <div className="brood-selected-performance-groups">
        {groups.map((group) => {
          const Icon = group.icon;
          return (
            <section className="brood-performance-group" key={group.title}>
              <div className="brood-performance-group-heading">
                <span><Icon size={18} /></span>
                <strong>{group.title}</strong>
              </div>

              <div className="brood-performance-metric-list">
                {group.metrics.map(([label, value]) => (
                  <div className="brood-performance-metric" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>

      <div className="brood-metric-direction-note">
      </div>
    </Panel>
  );
}

function BinaryPersistenceAudit({ audit }) {
  const rows = audit?.horizons || [];
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Horizon</th><th>Comparable rows</th><th>Unchanged labels</th><th>Transitions</th><th>Persistence accuracy</th><th>Transition rate</th></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.horizon_hours}>
              <td><strong>{row.horizon_hours} h</strong></td>
              <td>{numberValue(row.comparable_rows, 0)}</td>
              <td>{numberValue(row.same_status_rows, 0)}</td>
              <td>{numberValue(row.transition_rows, 0)}</td>
              <td>{asPercent(row.persistence_accuracy)}</td>
              <td>{asPercent(row.transition_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="chart-footnote">{audit?.interpretation}</p>
    </div>
  );
}

function LeakageAudit({ audit }) {
  const rows = [
    ['Observed target or target lags used as features', audit?.current_or_lagged_binary_target_used],
    ['Future sensor values used as features', audit?.future_sensor_values_used_as_features],
    ['Hive/device identifier used as a feature', audit?.hive_id_used_as_feature],
    ['Absolute date/day-of-year used as a feature', audit?.absolute_date_or_day_of_year_used_as_feature],
    ['Absolute hive weight used as a feature', audit?.absolute_weight_used_as_feature],
  ];
  return (
    <div className="brood-audit-list">
      {rows.map(([label, failed]) => (
        <div key={label} className={failed ? 'failed' : 'passed'}>
          {failed ? <AlertTriangle size={17} /> : <CheckCircle2 size={17} />}
          <span>{label}</span><strong>{failed ? 'Detected' : 'Excluded'}</strong>
        </div>
      ))}
      <p className="chart-footnote">
        Past-only rolling features: {audit?.rolling_features_shifted_before_aggregation ? 'verified' : 'not verified'} ·
        Whole hives held out: {audit?.whole_hives_held_out ? 'yes' : 'no'}
      </p>
    </div>
  );
}

function ScoreWeights({ definition, sensitivity }) {
  const weights = definition?.weights || {};
  const rows = [
    ['Temperature', weights.temperature],
    ['Humidity', weights.humidity],
    ['CO₂', weights.co2],
    ['Relative weight stability', weights.weight_stability],
  ];
  return (
    <>
      <div className="brood-weight-grid">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span><strong>{asPercent(value, 0)}</strong>
            <div><i style={{ width: `${Math.max(0, Number(value || 0) * 100)}%` }} /></div>
          </div>
        ))}
      </div>
      <p className="chart-footnote">
        Active score coefficients for this training run.
      </p>
      {!!sensitivity?.length && (
        <details className="brood-details">
          <summary>View top sensitivity-analysis alternatives</summary>
          <div className="table-scroll">
            <table><thead><tr><th>Temperature</th><th>Humidity</th><th>CO₂</th><th>Weight stability</th><th>Balanced accuracy</th><th>Macro F1</th></tr></thead>
              <tbody>{sensitivity.slice(0, 10).map((row, index) => (
                <tr key={`${index}-${row.temperature_weight}`}>
                  <td>{asPercent(row.temperature_weight, 0)}</td>
                  <td>{asPercent(row.humidity_weight, 0)}</td>
                  <td>{asPercent(row.co2_weight, 0)}</td>
                  <td>{asPercent(row.weight_stability_weight, 0)}</td>
                  <td>{asPercent(row.balanced_accuracy)}</td>
                  <td>{asPercent(row.macro_f1)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </details>
      )}
    </>
  );
}

function SectionHeading({ title, subtitle }) {
  return (
    <div className="brood-section-heading compact brood-admin-heading">
      <div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>
    </div>
  );
}

export function BroodTrainingTab() {
  const resource = useBroodTraining(true);
  const data = resource.data;
  const status = data?.training_status || {};
  const metrics = data?.best_metrics || {};
  const exact = metrics.exact_horizon || {};
  const transition = metrics.transition || {};
  const deterioration = metrics.deterioration || {};
  const forecastIndicators = metrics.forecast_indicators || {};
  const split = data?.split_summary || {};

  return (
    <div className="page-stack">
      <div className="brood-section-heading">
        <div>
          <span className="eyebrow">MODEL TRAINING</span>
          <h3>Brood Health Forecast Model</h3>
          <p>Train, compare and review the six-hour brood-health forecasting models.</p>
        </div>
        <div className="brood-action-group">
          <button className="button button-outline" disabled={resource.starting || status.running} onClick={() => resource.startTraining({ fastMode: true, horizonHours: 6 })}>Quick comparison</button>
          <button className="button" disabled={resource.starting || status.running} onClick={() => resource.startTraining({ fastMode: false, horizonHours: 6 })}><BrainCircuit size={17} /> Full training</button>
        </div>
      </div>

      {(resource.loading && !data) && <div className="brood-inline-state">Loading model summary…</div>}
      {(resource.error || resource.startError) && (
        <div className="brood-alert danger"><AlertTriangle size={18} /><div><strong>Model service error</strong><p>{resource.error?.message || resource.startError?.message}</p></div></div>
      )}

      {(status.running || resource.starting) && (
        <Panel title="Training in progress" subtitle={status.message || 'Preparing the leakage-safe model comparison.'}>
          <div className="brood-progress"><div style={{ width: `${Math.max(2, status.progress || 2)}%` }} /></div>
          <div className="brood-progress-meta"><span>{status.event || 'queued'}</span><strong>{status.progress || 0}%</strong><span>{status.model || ''}</span></div>
        </Panel>
      )}

      {!data?.trained && !status.running && !resource.loading && (
        <div className="brood-empty-workspace">
          <BrainCircuit size={48} /><h3>No v6 brood-health forecaster is available</h3>
          <p>Run the model comparison after deleting the older generated artifacts.</p>
        </div>
      )}

      {data?.trained && (
        <>
          <SectionHeading
            title="Selected Model Summary"
          />
          <SelectedModelSummary
            modelName={data.best_model}
            exact={exact}
            transition={transition}
            deterioration={deterioration}
            forecastIndicators={forecastIndicators}
          />

          <Panel title="Complete Model Comparison" subtitle="Compare regression error, classification accuracy and recall across all candidate models.">
            <ModelComparisonTable models={data.all_models} bestModel={data.best_model} />
          </Panel>

          <Panel title="What the model predicts" subtitle="The three main future outputs shown to the beekeeper.">
            <div className="brood-info-list">
              <span>Six-hour health <strong>Expected Brood Health Score exactly 6 hours ahead</strong></span>
              <span>Lowest expected point <strong>Lowest score predicted anywhere during the next 6 hours</strong></span>
              <span>Future condition pattern <strong>Forecast BHSI shows stability; Forecast RoD shows speed and direction of change</strong></span>
            </div>
          </Panel>

          <SectionHeading title="Model Comparison" subtitle="Compare how accurately each model predicts the future score and changing health conditions." />
          <div className="two-column-grid">
            <Panel title="Classification Performance" subtitle="Accuracy and recall metrics are shown as percentages. Higher values indicate better classification performance.">
              <ModelComparisonChart data={data.all_models} />
            </Panel>
            <Panel title="Regression Error Comparison" subtitle="MAE and RMSE are score-point errors; percentage labels show the same error relative to the 100-point Brood Health Score scale.">
              <ModelErrorComparisonChart data={data.all_models} />
            </Panel>
          </div>

          <SectionHeading title="Forecast Performance" subtitle="Detailed evaluation of the selected model across forecast horizons and health levels." />
          <div className="two-column-grid">
            <Panel title="Selected Model vs Persistence Baseline" subtitle="Compares the trained model with a baseline that assumes the current Brood Health Score will remain unchanged.">
              <PersistenceComparisonChart model={metrics} persistence={data.persistence_baseline} />
            </Panel>
            <Panel title="MAE and RMSE by Forecast Horizon" subtitle="Shows how score-prediction error changes from one hour ahead to six hours ahead.">
              <HorizonErrorChart data={metrics.per_horizon} />
            </Panel>
          </div>

          <div className="two-column-grid">
            <Panel title="Actual vs Predicted 6-Hour Score" subtitle="Points closer to the diagonal reference line indicate predictions closer to the actual score.">
              <ActualPredictedScoreChart data={data.prediction_sample} />
            </Panel>
            <Panel
              title="Health-Level Prediction Results"
              subtitle="For each actual health level, shows the percentage predicted as Critical, Poor, Good or Excellent."
            >
              <HealthLevelClassificationChart
                matrix={exact.confusion_matrix}
                labels={exact.level_labels}
              />
            </Panel>
          </div>

          <SectionHeading
            title="Score & Indicator Calculation"
            subtitle="Open the formula reference when you need to explain how the Brood Health Score, Forecast BHSI and Forecast RoD are produced."
          />
          <BroodFormulaReference
            scoreDefinition={data.score_definition}
            stabilityReference={data.forecast_stability_reference}
            weightCalibration={data.weight_calibration}
          />

          <SectionHeading title="Score & Feature Configuration" subtitle="Current-score contributions and the inputs that influenced the selected forecasting model." />
          <div className="two-column-grid">
            <Panel title="Current-score contributions" subtitle="The four percentages used to combine the sensor sub-scores into the 1–100 Brood Health Score.">
              <ScoreWeights definition={data.score_definition} sensitivity={data.weight_sensitivity_top} />
            </Panel>
            <Panel title="Most influential prediction inputs" subtitle="Shows which engineered inputs contributed most to the selected model.">
              <FeatureImportanceChart data={data.top_features} />
            </Panel>
          </div>

          <SectionHeading title="Training Setup" subtitle="Dataset split and optional technical checks." />
          <div className="two-column-grid">
            <Panel title="Training / Validation / Test Split" subtitle="Row and hive counts used for model development and final evaluation.">
              {(() => {
                const totalRows = Number(split.train_rows || 0)
                  + Number(split.validation_rows || 0)
                  + Number(split.test_rows || 0);
                const share = (value) => (
                  totalRows > 0
                    ? `${(Number(value || 0) / totalRows * 100).toFixed(1)}%`
                    : '—'
                );

                return (
                  <div className="brood-split-grid">
                    <div>
                      <span>Train</span>
                      <strong>{numberValue(split.train_rows, 0)}</strong>
                      <small>{share(split.train_rows)} of rows · {numberValue(split.train_hives, 0)} hives</small>
                    </div>
                    <div>
                      <span>Validation</span>
                      <strong>{numberValue(split.validation_rows, 0)}</strong>
                      <small>{share(split.validation_rows)} of rows · {numberValue(split.validation_hives, 0)} hives</small>
                    </div>
                    <div>
                      <span>Test</span>
                      <strong>{numberValue(split.test_rows, 0)}</strong>
                      <small>{share(split.test_rows)} of rows · {numberValue(split.test_hives, 0)} unseen hives</small>
                    </div>
                  </div>
                );
              })()}
              <p className="chart-footnote">Minimum causal history: {split.minimum_history_hours} hours.</p>
            </Panel>
            <Panel title="Technical Checks" subtitle="Detailed validation checks are collapsed by default.">
              <details className="brood-details brood-admin-details">
                <summary>View model data checks</summary>
                <LeakageAudit audit={data.leakage_audit} />
                <div className="brood-admin-divider" />
                <BinaryPersistenceAudit audit={data.binary_target_audit} />
              </details>
            </Panel>
          </div>

          <SectionHeading title="Model Charts" subtitle="Saved performance charts for the selected training run." />
          <Panel title="Model Report Figures" subtitle="Generated performance charts.">
            <BroodReportGallery images={data.generated_images} />
          </Panel>
        </>
      )}
    </div>
  );
}
