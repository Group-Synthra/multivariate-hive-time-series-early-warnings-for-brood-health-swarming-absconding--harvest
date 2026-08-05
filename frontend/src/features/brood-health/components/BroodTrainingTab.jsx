import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Crosshair,
  Gauge,
  Layers3,
  ShieldCheck,
  Target,
  TrendingDown,
} from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { StatCard } from '../../../components/common/StatCard';
import { useBroodTraining } from '../hooks/useBroodHealthData';
import { asPercent, numberValue, timestampValue } from '../utils/broodHealth';
import {
  ActualPredictedScoreChart,
  FeatureImportanceChart,
  ModelComparisonChart,
  ModelErrorComparisonChart,
  PersistenceComparisonChart,
} from './BroodHealthCharts';
import { ConfusionMatrix } from './BroodMatrices';
import { BroodReportGallery } from './BroodReportGallery';

function ModelComparisonTable({ models }) {
  return (
    <div className="table-scroll">
      <table className="brood-model-table">
        <thead>
          <tr>
            <th>Model</th><th>Status</th><th>MAE</th><th>RMSE</th><th>R²</th>
            <th>Overall level accuracy</th><th>Transition accuracy</th><th>Transition MAE</th>
            <th>Critical recall</th><th>Group-CV MAE</th><th>Fit time</th>
          </tr>
        </thead>
        <tbody>{(models || []).map((row) => (
          <tr key={row.model} className={row.status !== 'ok' ? 'failed-row' : ''}>
            <td><strong>{row.model}</strong></td>
            <td>{row.status}</td>
            <td>{numberValue(row.test?.test_mae, 3)}</td>
            <td>{numberValue(row.test?.test_rmse, 3)}</td>
            <td>{numberValue(row.test?.test_r2, 4)}</td>
            <td>{asPercent(row.test?.health_level_accuracy)}</td>
            <td><strong>{asPercent(row.test?.transition_level_accuracy)}</strong></td>
            <td>{numberValue(row.test?.transition_mae, 3)}</td>
            <td>{asPercent(row.test?.critical_recall)}</td>
            <td>{numberValue(row.test?.cv_mae_mean, 3)}</td>
            <td>{row.fit_seconds ? `${numberValue(row.fit_seconds, 1)} s` : '—'}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function BinaryPersistenceAudit({ audit }) {
  const rows = audit?.horizons || [];
  if (!rows.length) return <p className="chart-footnote">Run the corrected training pipeline to generate the binary persistence audit.</p>;
  return (
    <div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Forecast horizon</th><th>Comparable rows</th><th>Status unchanged</th><th>True transitions</th><th>Persistence accuracy</th><th>Transition rate</th></tr></thead>
          <tbody>{rows.map((row) => (
            <tr key={row.horizon_hours}>
              <td><strong>{row.horizon_hours} h</strong></td>
              <td>{numberValue(row.comparable_rows, 0)}</td>
              <td>{numberValue(row.same_status_rows, 0)}</td>
              <td>{numberValue(row.transition_rows, 0)}</td>
              <td>{asPercent(row.persistence_accuracy)}</td>
              <td>{asPercent(row.transition_rate)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="chart-footnote">{audit.interpretation}</p>
    </div>
  );
}

function LeakageAudit({ audit }) {
  const rows = [
    ['Target or target lags used as features', audit?.target_columns_in_features || audit?.current_or_lagged_binary_target_used],
    ['Future sensor values used as inputs', audit?.future_sensor_values_used_as_features],
    ['Hive ID used as an input feature', audit?.hive_id_used_as_feature],
    ['Absolute date used as an input feature', audit?.absolute_date_or_day_of_year_used_as_feature],
    ['Whole hives held out from training', audit?.whole_hives_held_out],
    ['Test hives used during model selection', audit?.test_hives_used_for_model_selection],
  ];
  return (
    <div className="brood-audit-grid">
      {rows.map(([label, value], index) => {
        const inverse = index < 4 || index === 5;
        const passed = inverse ? !value : Boolean(value);
        return <div key={label} className={passed ? 'pass' : 'fail'}><span>{label}</span><strong>{passed ? 'Passed' : 'Review'}</strong></div>;
      })}
    </div>
  );
}

export function BroodTrainingTab() {
  const resource = useBroodTraining(true);
  const data = resource.data;
  const status = data?.training_status || {};
  const metrics = data?.best_metrics || {};
  const split = data?.split_summary || {};
  const interpretation = data?.accuracy_interpretation || {};
  const persistence = data?.persistence_baseline || {};

  return (
    <div className="page-stack">
      <div className="brood-section-heading">
        <div>
          <span className="eyebrow">BROOD-SPECIFIC MODEL DEVELOPMENT</span>
          <h3>Predict the minimum Brood Health Score expected within the next six hours</h3>
          <p>The common cleaned dataset is the shared first phase. This module then applies brood-specific preprocessing, causal feature engineering, whole-hive holdout validation and continuous score forecasting.</p>
        </div>
        <div className="brood-action-group">
          <button className="button button-outline" disabled={resource.starting || status.running} onClick={() => resource.startTraining({ fastMode: true })}>Quick comparison</button>
          <button className="button" disabled={resource.starting || status.running} onClick={() => resource.startTraining({ fastMode: false })}><BrainCircuit size={17} /> Full training</button>
        </div>
      </div>

      {(resource.loading && !data) && <div className="brood-inline-state">Loading model summary…</div>}
      {(resource.error || resource.startError) && <div className="brood-alert danger"><AlertTriangle size={18} /><div><strong>Model service error</strong><p>{resource.error?.message || resource.startError?.message}</p></div></div>}

      {(status.running || resource.starting) && <Panel title="Training in progress" subtitle={status.message || 'Preparing the model comparison.'}><div className="brood-progress"><div style={{ width: `${Math.max(2, status.progress || 2)}%` }} /></div><div className="brood-progress-meta"><span>{status.event || 'queued'}</span><strong>{status.progress || 0}%</strong><span>{status.model || ''}</span></div></Panel>}

      {!data?.trained && !status.running ? <div className="brood-empty-workspace"><BrainCircuit size={48} /><h3>No trained score-forecasting model is available</h3><p>Run quick comparison or full training. The backend saves the selected regressor, exact feature schema, score definition, held-out-hive metrics and deployment artifacts.</p></div> : null}

      {data?.trained && <>
        <div className="stats-grid stats-grid-six">
          <StatCard label="Selected model" value={data.best_model} icon={BrainCircuit} note={`${data.horizon_hours}-hour future window`} />
          <StatCard label="Transition accuracy" value={asPercent(metrics.transition_level_accuracy)} icon={Target} note="Primary early-warning metric" />
          <StatCard label="Test MAE" value={numberValue(metrics.test_mae, 2)} unit="points" icon={Gauge} note="Unseen hives" />
          <StatCard label="Test RMSE" value={numberValue(metrics.test_rmse, 2)} unit="points" icon={TrendingDown} />
          <StatCard label="Critical recall" value={asPercent(metrics.critical_recall)} icon={Crosshair} />
          <StatCard label="Test R²" value={numberValue(metrics.test_r2, 4)} icon={ShieldCheck} />
        </div>

        <div className="brood-alert success">
          <CheckCircle2 size={20} />
          <div>
            <strong>Primary early-warning accuracy: {numberValue(interpretation.primary_early_warning_accuracy_percent, 2)}%</strong>
            <p>{interpretation.explanation}</p>
          </div>
        </div>

        <div className="brood-alert info">
          <Layers3 size={20} />
          <div>
            <strong>Corrected target formulation</strong>
            <p>{data.target_formulation?.why_changed}</p>
            <p><strong>Current output:</strong> {data.target_formulation?.current_output}<br /><strong>Future output:</strong> {data.target_formulation?.future_output}</p>
          </div>
        </div>

        <Panel title="Why the previous 98–99% binary accuracy was misleading" subtitle="A persistence rule predicts that the future healthy/unhealthy label will remain equal to the current label. These figures are calculated from the historical dataset during training.">
          <BinaryPersistenceAudit audit={data.binary_target_audit} />
        </Panel>

        <div className="two-column-grid">
          <Panel title="Early-warning accuracy comparison" subtitle="Transition rows contain a level crossing or a score drop of at least 10 points."><ModelComparisonChart data={data.all_models} /></Panel>
          <Panel title="Score-error comparison" subtitle="MAE is the main continuous-score error; transition MAE isolates difficult deterioration periods."><ModelErrorComparisonChart data={data.all_models} /></Panel>
        </div>

        <Panel title="Complete model comparison" subtitle={data.metrics_note}><ModelComparisonTable models={data.all_models} /></Panel>

        <div className="two-column-grid">
          <Panel title="Model versus persistence baseline" subtitle="Persistence predicts that the future score will equal the current score. A useful model must improve particularly during transitions."><PersistenceComparisonChart model={metrics} persistence={persistence} /></Panel>
          <Panel title="Actual versus predicted future score" subtitle="Sample from the untouched unseen-hive test partition."><ActualPredictedScoreChart data={data.prediction_sample} /></Panel>
        </div>

        <div className="two-column-grid">
          <Panel title="Four-level confusion matrix" subtitle="Critical, Poor, Good and Excellent future score levels on unseen hives."><ConfusionMatrix matrix={metrics.confusion_matrix} labels={metrics.level_labels} /></Panel>
          <Panel title="Top causal features" subtitle="Relative importance from the selected evaluation model. Target labels, hive IDs and future readings are excluded."><FeatureImportanceChart data={data.top_features} /></Panel>
        </div>

        <div className="two-column-grid">
          <Panel title="Whole-hive split audit" subtitle="Historical hives are separated so the test set represents colonies never seen during fitting.">
            <div className="brood-split-grid">
              <div><span>Train</span><strong>{numberValue(split.train_rows, 0)}</strong><small>{numberValue(split.train_hives, 0)} hives</small></div>
              <div><span>Validation</span><strong>{numberValue(split.validation_rows, 0)}</strong><small>{numberValue(split.validation_hives, 0)} hives</small></div>
              <div><span>Test</span><strong>{numberValue(split.test_rows, 0)}</strong><small>{numberValue(split.test_hives, 0)} unseen hives</small></div>
            </div>
            <p className="chart-footnote">Minimum causal history: {split.minimum_history_hours} hours. Target rows always occur after the feature timestamp.</p>
          </Panel>
          <Panel title="Leakage and transfer audit" subtitle="Checks that directly address the previous 98–99% binary-classification concern."><LeakageAudit audit={data.leakage_audit} /></Panel>
        </div>

        <div className="two-column-grid">
          <Panel title="Feature families and deployment schema" subtitle={`${numberValue(data.feature_count, 0)} causal features from the four IoT sensor streams.`}>
            <div className="brood-feature-groups">{(data.grouped_feature_importance || []).map((row) => <div key={row.sensor_group}><span>{row.sensor_group}</span><div><i style={{ width: `${Math.min(100, row.importance_percentage)}%` }} /></div><strong>{numberValue(row.importance_percentage, 2)}%</strong></div>)}</div>
            <details className="brood-details"><summary>View exact feature columns</summary><div className="brood-chip-list">{(data.feature_columns || []).map((feature) => <code key={feature}>{feature}</code>)}</div></details>
          </Panel>
          <Panel title="Model selection rule" subtitle="Validation hives select the model; test hives are used once for final reporting."><ul className="brood-check-list"><li>{data.selection_rule}</li><li>The selected model is refit on training and validation hives only.</li><li>Current score, future score, BHSI and RoD remain separate outputs.</li><li>Historical metrics do not replace live-field validation against physical inspections.</li></ul><p className="chart-footnote">Trained: {timestampValue(data.trained_at_utc)}</p></Panel>
        </div>

        <Panel title="Exact backend-generated model figures" subtitle="Figures produced from the same saved evaluation results used by the API."><BroodReportGallery images={data.generated_images} /></Panel>
        <Panel title="Research and deployment limitations" subtitle="These statements should remain visible in the report and demonstration."><ul className="brood-warning-list">{(data.model_limitations || []).map((item) => <li key={item}>{item}</li>)}</ul></Panel>
      </>}
    </div>
  );
}
