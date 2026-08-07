import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Crosshair,
  Gauge,
  ShieldCheck,
  Target,
  TrendingDown,
} from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { StatCard } from '../../../components/common/StatCard';
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
import { ConfusionMatrix } from './BroodMatrices';
import { BroodReportGallery } from './BroodReportGallery';

function ModelComparisonTable({ models, bestModel }) {
  return (
    <div className="table-scroll">
      <table className="brood-model-table">
        <thead>
          <tr>
            <th>Model</th><th>Exact +6 h MAE ↓</th><th>MSE ↓</th><th>RMSE ↓</th><th>R² ↑</th>
            <th>Level accuracy ↑</th><th>Transition accuracy ↑</th>
            <th>Deterioration recall ↑</th><th>Critical recall ↑</th>
            <th>Forecast BHSI accuracy ↑</th><th>Forecast trend accuracy ↑</th>
            <th>Group-CV MAE ↓</th>
          </tr>
        </thead>
        <tbody>
          {(models || []).map((row) => {
            const exact = row.test?.exact_horizon || {};
            const transition = row.test?.transition || {};
            const deterioration = row.test?.deterioration || {};
            const selected = row.model === bestModel;
            return (
              <tr key={row.model} className={`${row.status !== 'ok' ? 'failed-row' : ''} ${selected ? 'selected-row' : ''}`}>
                <td><strong>{row.model}</strong>{selected && <span className="brood-selected-badge">Selected</span>}</td>
                {row.status !== 'ok' ? (
                  <td colSpan={11}>{row.error || 'Model failed'}</td>
                ) : (
                  <>
                    <td>{numberValue(exact.mae, 3)}</td>
                    <td>{numberValue(exact.mse, 3)}</td>
                    <td>{numberValue(exact.rmse, 3)}</td>
                    <td>{numberValue(exact.r2, 4)}</td>
                    <td>{asPercent(exact.health_level_accuracy)}</td>
                    <td><strong>{asPercent(transition.health_level_accuracy)}</strong></td>
                    <td>{asPercent(deterioration.recall)}</td>
                    <td>{asPercent(exact.critical_recall)}</td>
                    <td>{asPercent(row.test?.forecast_indicators?.forecast_bhsi_level_accuracy)}</td>
                    <td>{asPercent(row.test?.forecast_indicators?.forecast_trend_accuracy)}</td>
                    <td>{numberValue(row.test?.cv_mae_mean, 3)}</td>
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
          <SectionHeading title="Selected Model Summary" subtitle="Key performance indicators for the selected model." />
          <div className="stats-grid stats-grid-six">
            <StatCard label="Selected model" value={data.best_model} icon={BrainCircuit} note="Selected on validation hives only" />
            <StatCard label="Exact +6 h MAE" value={numberValue(exact.mae, 2)} unit="points" icon={Gauge} note="Untouched test hives" />
            <StatCard label="Exact +6 h RMSE" value={numberValue(exact.rmse, 2)} unit="points" icon={TrendingDown} />
            <StatCard label="Transition accuracy" value={asPercent(transition.health_level_accuracy)} icon={Target} note="Difficult changing periods" />
            <StatCard label="Deterioration recall" value={asPercent(deterioration.recall)} icon={Crosshair} />
            <StatCard label="Critical recall" value={asPercent(exact.critical_recall)} icon={ShieldCheck} />
          </div>

          <Panel
            title="Future-indicator validation on unseen hives"
            subtitle="These metrics verify the derived six-hour stability and trend outputs, not only the final score."
          >
            <div className="stats-grid stats-grid-six">
              <StatCard label="Forecast BHSI MAE" value={numberValue(forecastIndicators.forecast_bhsi_mae, 2)} unit="points" />
              <StatCard label="Forecast BHSI RMSE" value={numberValue(forecastIndicators.forecast_bhsi_rmse, 2)} unit="points" />
              <StatCard label="BHSI level accuracy" value={asPercent(forecastIndicators.forecast_bhsi_level_accuracy)} note="Low, Moderate or High" />
              <StatCard label="Forecast RoD MAE" value={numberValue(forecastIndicators.forecast_rod_mae, 2)} unit="points/h" />
              <StatCard label="Forecast RoD RMSE" value={numberValue(forecastIndicators.forecast_rod_rmse, 2)} unit="points/h" />
              <StatCard label="Forecast trend accuracy" value={asPercent(forecastIndicators.forecast_trend_accuracy)} note="Declining, Stable or Improving classes" />
            </div>
          </Panel>

          <Panel title="Forecast Outputs" subtitle="Outputs returned by the selected model.">
            <div className="brood-info-list">
              <span>Primary forecast <strong>Exact Brood Health Score at +6 hours</strong></span>
              <span>Safety indicator <strong>Lowest predicted score within +1 to +6 hours</strong></span>
              <span>Future indicators <strong>Forecast BHSI and Forecast RoD</strong></span>
            </div>
          </Panel>

          <SectionHeading title="Model Comparison" subtitle="Candidate models compared on forecast error and health-transition performance." />
          <div className="two-column-grid">
            <Panel title="Model-level performance" subtitle="Exact +6-hour level accuracy, transition accuracy and deterioration recall on complete unseen hives.">
              <ModelComparisonChart data={data.all_models} />
            </Panel>
            <Panel title="Continuous-score errors" subtitle="Lower MAE and RMSE are better. Transition MAE focuses on changing conditions.">
              <ModelErrorComparisonChart data={data.all_models} />
            </Panel>
          </div>

          <Panel title="Complete model comparison" subtitle={data.metrics_note}>
            <ModelComparisonTable models={data.all_models} bestModel={data.best_model} />
          </Panel>

          <SectionHeading title="Forecast Performance" subtitle="Selected-model behaviour across horizons and health levels." />
          <div className="two-column-grid">
            <Panel title="Selected model versus current-score persistence" subtitle="A useful early-warning model should outperform simply repeating the current score.">
              <PersistenceComparisonChart model={metrics} persistence={data.persistence_baseline} />
            </Panel>
            <Panel title="Error by forecast horizon" subtitle="The same selected model predicts +1 through +6 hours directly.">
              <HorizonErrorChart data={metrics.per_horizon} />
            </Panel>
          </div>

          <div className="two-column-grid">
            <Panel title="Actual versus predicted exact +6-hour score" subtitle="Sample from the untouched whole-hive test partition.">
              <ActualPredictedScoreChart data={data.prediction_sample} />
            </Panel>
            <Panel title="Four-level confusion matrix" subtitle="Exact +6-hour Critical, Poor, Good and Excellent classifications.">
              <ConfusionMatrix matrix={exact.confusion_matrix} labels={exact.level_labels} />
            </Panel>
          </div>

          <SectionHeading title="Score & Feature Configuration" subtitle="Current-score coefficients and model feature importance." />
          <div className="two-column-grid">
            <Panel title="Current-score coefficients" subtitle={`${data.weight_calibration?.scope || 'training hives only'} · ${data.weight_calibration?.selection_metric || 'constrained sensitivity analysis'}`}>
              <ScoreWeights definition={data.score_definition} sensitivity={data.weight_sensitivity_top} />
            </Panel>
            <Panel title="Feature Importance" subtitle="Most influential forecasting inputs.">
              <FeatureImportanceChart data={data.top_features} />
            </Panel>
          </div>

          <SectionHeading title="Training Setup" subtitle="Dataset split and optional technical checks." />
          <div className="two-column-grid">
            <Panel title="Training / Validation / Test Split" subtitle="Row and hive counts used for model development and final evaluation.">
              <div className="brood-split-grid">
                <div><span>Train</span><strong>{numberValue(split.train_rows, 0)}</strong><small>{numberValue(split.train_hives, 0)} hives</small></div>
                <div><span>Validation</span><strong>{numberValue(split.validation_rows, 0)}</strong><small>{numberValue(split.validation_hives, 0)} hives</small></div>
                <div><span>Test</span><strong>{numberValue(split.test_rows, 0)}</strong><small>{numberValue(split.test_hives, 0)} unseen hives</small></div>
              </div>
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
