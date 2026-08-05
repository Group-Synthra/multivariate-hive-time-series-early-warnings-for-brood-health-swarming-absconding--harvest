import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Clock3,
  Database,
  RefreshCw,
  ShieldCheck,
  Wind,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { EmptyState } from '../../components/common/EmptyState';
import { Panel } from '../../components/common/Panel';
import { StatCard } from '../../components/common/StatCard';
import { formatDate, formatNumber, percentage } from '../../utils/formatters';
import { useAbscondingData } from '../../hooks/useAbscondingData';
import { ModuleTabs } from '../shared/ModuleTabs';
import AbscondingLiveDashboard from './AbscondingLiveDashboard';
import "../../styles/absconding.css";

const RISK_ORDER = { High: 0, Medium: 1, Low: 2 };
const RISK_COLOURS = { High: '#dc2626', Medium: '#d97706', Low: '#059669' };

function riskBadge(level) {
  return <span className={`risk-badge risk-${String(level).toLowerCase()}`}>{level}</span>;
}

function metric(value, digits = 4) {
  return value === null || value === undefined ? '—' : formatNumber(value, digits);
}

function metricPercentage(value, digits = 2) {
  if (value === null || value === undefined || value === '') {
    return '—';
  }

  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? `${(numericValue * 100).toFixed(digits)}%`
    : '—';
}

function SetupRequired({ error, onRetry }) {
  return (
    <div className="page-stack">
      <section className="hero compact">
        <div>
          <span className="eyebrow">MODULE 03</span>
          <h2>Absconding Early Warning</h2>
          <p>Run the module pipeline once to generate the trained model and dashboard artifacts.</p>
        </div>
        <Wind size={42} />
      </section>
      <Panel title="Absconding outputs are not ready" subtitle={error?.message || 'No generated dashboard was found.'}>
        <div className="module-command-box">
          <code>python scripts/run_absconding_pipeline.py</code>
          <button type="button" className="button" onClick={onRetry}>
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      </Panel>
    </div>
  );
}

export function AbscondingPage() {
  const [activeTab, setActiveTab] = useState('exploratory-analysis');
  const [selectedHive, setSelectedHive] = useState('');
  const {
    data, loading, error, refetch,
    iotLiveData, iotLiveLoading, iotLiveError, refetchIotLive,
    dashboardRefreshIntervalMinutes, lastIotFetchAt, nextIotRefreshAt,
  } = useAbscondingData();

  useEffect(() => {
    if (!selectedHive && data?.hive_options?.length) {
      setSelectedHive(data.hive_options[0]);
    }
  }, [data, selectedHive]);

  const selectedDetail = selectedHive ? data?.hive_details?.[selectedHive] : null;
  const sortedHiveRisk = useMemo(
    () =>
      [...(data?.latest_hive_risk || [])].sort(
        (a, b) =>
          (RISK_ORDER[a.risk_level] ?? 3) - (RISK_ORDER[b.risk_level] ?? 3) ||
          b.probability - a.probability,
      ),
    [data],
  );

  if (error && !data) {
    return <SetupRequired error={error} onRetry={refetch} />;
  }

  if (loading && !data) {
    return <EmptyState message="Loading the Absconding module…" />;
  }

  const summary = data?.summary || {};
  const exploratory = data?.exploratory_analysis || {};
  const training = data?.model_training || {};

  return (
    <div className="page-stack absconding-page">
      <section className="hero compact absconding-hero">
        <div>
          <span className="eyebrow">MODULE 03 · EVENT-AWARE TIME SERIES</span>
          <h2>Absconding Early Warning</h2>
          <p>
            Leakage-safe next-24-hour warning targets, long-term deterioration features, classical
            rare-event models, an anomaly baseline and per-hive risk monitoring.
          </p>
        </div>
        <div className="hero-status-stack">
          <span className="status-pill warning"><AlertTriangle size={15} /> Exploratory evidence</span>
          <button type="button" className="button" onClick={refetch} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin' : ''} /> Refresh
          </button>
        </div>
      </section>

      <div className="stats-grid stats-grid-six">
        <StatCard label="Raw event markers" value={summary.source_event_markers} icon={Database} note="Original absconding_happened_1 rows" />
        <StatCard label="Event episodes" value={summary.distinct_event_episodes} icon={AlertTriangle} note="Nearby markers merged within 24 hours" />
        <StatCard label="Future warning rows" value={summary.future_positive_rows} icon={Clock3} note={`${summary.prediction_horizon_hours}-hour target horizon`} />
        <StatCard label="Selected model" value={summary.selected_model_name} icon={BrainCircuit} note="Chosen on validation data only" />
        <StatCard label="Test PR-AUC" value={training.test_metrics?.pr_auc} icon={Activity} note="Preferred rare-event ranking metric" />
        <StatCard label="Test event recall" value={training.test_event_metrics?.event_recall} icon={ShieldCheck} note="Detected event episodes" />
      </div>

      <div className="eda-interpretation-note absconding-warning-note">
        <strong>Evidence limitation:</strong> {summary.methodology_note}
      </div>

      <ModuleTabs activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'exploratory-analysis' && (
        <ExploratoryAnalysis
          exploratory={exploratory}
          summary={summary}
          data={data}
          risks={sortedHiveRisk}
          selectedHive={selectedHive}
          onHiveChange={setSelectedHive}
          detail={selectedDetail}
        />
      )}
      {activeTab === 'model-training' && (
        <ModelTraining training={training} plots={data?.plots || {}} />
      )}
      {activeTab === 'live-early-warning' && (
        <AbscondingLiveDashboard
          iotLiveData={iotLiveData}
          iotLiveLoading={iotLiveLoading}
          iotLiveError={iotLiveError}
          refetchIotLive={refetchIotLive}
          dashboardRefreshIntervalMinutes={dashboardRefreshIntervalMinutes}
          lastIotFetchAt={lastIotFetchAt}
          nextIotRefreshAt={nextIotRefreshAt}
        />
      )}
    </div>
  );
}

function ExploratoryAnalysis({ exploratory, summary, data, risks, selectedHive, onHiveChange, detail }) {
  const splitRows = exploratory.split_summary || [];
  const effects = exploratory.sensor_effects || [];
  const latest = detail?.latest;
  const timeline = (detail?.timeline || []).map((row) => ({
    ...row,
    time: new Date(row.timestamp).toLocaleString(),
    risk: Number(row.risk_percentage || 0),
    armScaled: Number(row.arm || 0) * 100,
    stressScaled: Number(row.environmental_stress_score || 0) * 100,
    co2Scaled: Number(row.co2_ppm || 0) / 100,
  }));
  return (
    <>
      <Panel title="Absconding Risk Score — Meaning" subtitle="The dashboard combines model probability with ARM escalation.">
        <div className="absconding-risk-meaning-grid">
          <article><strong>Low</strong><span>Below the validation-derived medium threshold</span><p>Routine monitoring.</p></article>
          <article><strong>Medium</strong><span>Moderate probability or increasing ARM</span><p>Monitor closely and prepare an inspection.</p></article>
          <article><strong>High</strong><span>Validation-derived alert threshold reached</span><p>Urgent physical hive inspection.</p></article>
        </div>
      </Panel>

      <Panel
        title="Hive-wise historical risk analysis"
        subtitle="Select a hive to inspect risk, ARM, environmental stress and sensor behaviour."
        action={(
          <label className="chart-select-label">
            Hive
            <select value={selectedHive} onChange={(event) => onHiveChange(event.target.value)}>
              {(risks || []).map((item) => <option key={item.hive_id} value={item.hive_id}>{item.hive_id}</option>)}
            </select>
          </label>
        )}
      >
        {latest ? (
          <>
            <div className="stats-grid">
              <StatCard label="Absconding risk" value={latest.risk_percentage} unit="%" note={`${latest.risk_level} risk`} />
              <StatCard label="ARM" value={latest.arm} note={latest.arm_trend} />
              <StatCard label="Environmental stress" value={(latest.latest_sensor_readings?.environmental_stress_score || 0) * 100} unit="%" note="Explainable stress indicator" />
            </div>
            <div className="two-column-grid absconding-historical-grid">
              <div className="chart-area-large">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" minTickGap={55} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="risk" name="Risk %" stroke="#dc2626" dot={false} strokeWidth={3} />
                    <Line type="monotone" dataKey="armScaled" name="ARM ×100" stroke="#d97706" dot={false} />
                    <Line type="monotone" dataKey="stressScaled" name="Stress ×100" stroke="#2563eb" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div>
                <h4>Explainable risk factors</h4>
                <div className="absconding-factor-list">
                  {(latest.key_factors || latest.signal_explanations || []).map((factor) => (
                    <article key={factor.factor}>
                      <strong>{factor.factor}</strong>
                      <p>{factor.detail || factor.interpretation}</p>
                    </article>
                  ))}
                </div>
              </div>
            </div>
            <div className="chart-area-large">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" minTickGap={55} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="temperature_c" name="Temperature °C" stroke="#dc2626" dot={false} />
                  <Line type="monotone" dataKey="humidity_pct" name="Humidity %" stroke="#2563eb" dot={false} />
                  <Line type="monotone" dataKey="weight_kg" name="Weight kg" stroke="#059669" dot={false} />
                  <Line type="monotone" dataKey="co2Scaled" name="CO₂ /100" stroke="#d97706" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : <EmptyState message="No hive-level prediction is available yet." />}
      </Panel>
      <div className="two-column-grid">
        <Panel title="Target construction" subtitle="The label asks whether an event occurs after the current observation.">
          <p className="module-body-copy">{exploratory.target_definition}</p>
          <div className="target-flow">
            <span>Past 168h sensors</span><b>→</b><span>Current observation</span><b>→</b><span>Next {summary.prediction_horizon_hours || 24}h event window</span>
          </div>
          <ul className="responsibility-list compact-list">
            {(exploratory.leakage_controls || []).map((item) => <li key={item}>{item}</li>)}
          </ul>
        </Panel>

        <Panel title="Class balance by chronological split" subtitle="Positive rows are expanded warning windows, not independent events.">
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={splitRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="split" />
                <YAxis yAxisId="rows" />
                <YAxis yAxisId="rate" orientation="right" tickFormatter={(v) => `${(v * 100).toFixed(2)}%`} />
                <Tooltip formatter={(value, name) => name === 'positive_rate' ? percentage(value * 100) : formatNumber(value)} />
                <Legend />
                <Bar yAxisId="rows" dataKey="positive_rows" name="Future-event rows" fill="#2563eb" />
                <Bar yAxisId="rate" dataKey="positive_rate" name="Positive rate" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="two-column-grid">
        <Panel title="Pre-event sensor differences" subtitle={`Standardized mean differences between ${summary.prediction_horizon_hours || 24}-hour warning rows and normal rows.`}>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={effects} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="sensor" width={110} />
                <Tooltip />
                <Bar dataKey="standardized_difference" name="Standardized difference">
                  {effects.map((row) => <Cell key={row.sensor} fill={row.standardized_difference >= 0 ? '#2563eb' : '#dc2626'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Observed absconding episodes" subtitle={`${summary.distinct_event_episodes || 0} distinct episodes across the dataset.`}>
          <div className="table-scroll compact-table-scroll">
            <table>
              <thead><tr><th>Hive</th><th>Start</th><th>Markers</th><th>Split</th></tr></thead>
              <tbody>
                {(exploratory.event_episodes || []).map((event) => (
                  <tr key={event.episode_id}>
                    <td>{event.hive_id}</td><td>{formatDate(event.event_start)}</td><td>{event.marker_count}</td><td>{event.split}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel
        title="Generated module images"
        subtitle="These figures are generated automatically by the Absconding pipeline."
      >
        {Object.keys(data?.plots || {}).length ? (
          <div className="report-figure-grid">
            {Object.entries(data.plots).map(([label, imagePath]) => (
              <figure className="report-figure" key={label}>
                <figcaption className="report-figure-toolbar">
                  <span>{label.replaceAll('_', ' ')}</span>
                  <a href={imagePath} target="_blank" rel="noreferrer">
                    Open full image
                  </a>
                </figcaption>
                <img
                  src={imagePath}
                  alt={`Absconding ${label.replaceAll('_', ' ')}`}
                  loading="lazy"
                />
              </figure>
            ))}
          </div>
        ) : (
          <EmptyState message="No generated Absconding images are available yet." />
        )}
      </Panel>
    </>
  );
}

function ModelTraining({ training, plots }) {
  const comparison = training.model_comparison || [];
  const test = training.test_metrics || {};
  const event = training.test_event_metrics || {};
  const importance = training.feature_importance || [];
  return (
    <>
      <div className="stats-grid stats-grid-six">
        <StatCard label="Accuracy" value={metricPercentage(test.accuracy)} note="Overall row-level correctness" />
        <StatCard label="Balanced accuracy" value={metricPercentage(test.balanced_accuracy)} note="Average sensitivity across both classes" />
        <StatCard label="Precision" value={metricPercentage(test.precision)} note="Warning correctness on test" />
        <StatCard label="Recall" value={metricPercentage(test.recall)} note="Future-event rows detected" />
        <StatCard label="F2 score" value={metricPercentage(test.f2)} note="Recall-weighted threshold metric" />
        <StatCard label="Event recall" value={metricPercentage(event.event_recall)} note={`${event.detected_event_count || 0}/${event.event_count || 0} test episodes detected`} />
      </div>

      <Panel title="Report-aligned model design" subtitle="The current pipeline retrains compatible classical models on the new shared data contract.">
        <div className="absconding-model-design-grid">
          <article><strong>Classical comparisons</strong><p>Rule baseline, Gaussian NB, Logistic Regression, Ridge, Decision Tree, Random Forest and Extra Trees are trained and compared.</p></article>
          <article><strong>Temporal design</strong><p>Current/past-only lags, rolling windows, rates of change and 72-hour deterioration features preserve the report’s time-series rationale.</p></article>
          <article><strong>LSTM compatibility</strong><p>The past saved LSTM is not silently reused because its feature schema and target contract differ. It must be retrained separately before it can be claimed as a new-version result.</p></article>
        </div>
      </Panel>

      <Panel
        title="Validation model comparison"
        subtitle={`${training.selection_rule || ''} Accuracy is included for completeness, but PR-AUC, F2 and event recall are more informative for this rare-event target.`}
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Selection score</th>
                <th>Accuracy</th>
                <th>Balanced accuracy</th>
                <th>PR-AUC</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>F2</th>
                <th>Event recall</th>
                <th>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.model_key} className={row.model_key === training.selected_model?.model_key ? 'selected-model-row' : ''}>
                  <td><strong>{row.model_name}</strong>{row.model_key === training.selected_model?.model_key && <small> Selected</small>}</td>
                  <td>{metric(row.selection_score)}</td>
                  <td>{metricPercentage(row.validation_metrics?.accuracy)}</td>
                  <td>{metricPercentage(row.validation_metrics?.balanced_accuracy)}</td>
                  <td>{metricPercentage(row.validation_metrics?.pr_auc)}</td>
                  <td>{metricPercentage(row.validation_metrics?.precision)}</td>
                  <td>{metricPercentage(row.validation_metrics?.recall)}</td>
                  <td>{metricPercentage(row.validation_metrics?.f1 ?? row.validation_metrics?.f1_score)}</td>
                  <td>{metricPercentage(row.validation_metrics?.f2)}</td>
                  <td>{metricPercentage(row.validation_event_metrics?.event_recall)}</td>
                  <td>{metricPercentage(row.threshold_selection?.threshold)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="two-column-grid">
        <Panel title="Top selected-model features" subtitle="Importance is model-specific and should not be interpreted as causality.">
          {importance.some((item) => item.importance > 0) ? (
            <div className="chart-area-large">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={importance.filter((item) => item.importance > 0).slice(0, 15)} layout="vertical" margin={{ left: 45 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={160} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message="This selected model does not expose a stable feature-importance measure." />}
        </Panel>

        <Panel title="Unbiased test evaluation" subtitle="The validation-selected threshold is frozen before the test split is scored.">
          <dl className="metric-definition-grid">
            <div><dt>Test records</dt><dd>{formatNumber(test.records)}</dd></div>
            <div><dt>Positive rows</dt><dd>{formatNumber(test.positive_rows)}</dd></div>
            <div><dt>Accuracy</dt><dd>{metricPercentage(test.accuracy)}</dd></div>
            <div><dt>Balanced accuracy</dt><dd>{metricPercentage(test.balanced_accuracy)}</dd></div>
            <div><dt>Precision</dt><dd>{metricPercentage(test.precision)}</dd></div>
            <div><dt>Recall</dt><dd>{metricPercentage(test.recall)}</dd></div>
            <div><dt>F1</dt><dd>{metricPercentage(test.f1 ?? test.f1_score)}</dd></div>
            <div><dt>F2</dt><dd>{metricPercentage(test.f2)}</dd></div>
            <div><dt>PR-AUC</dt><dd>{metricPercentage(test.pr_auc)}</dd></div>
            <div><dt>ROC-AUC</dt><dd>{metricPercentage(test.roc_auc)}</dd></div>
            <div><dt>Alert fraction</dt><dd>{metricPercentage(test.alert_fraction)}</dd></div>
            <div><dt>Brier score</dt><dd>{metric(test.brier_score, 6)}</dd></div>
          </dl>
          <div className="confusion-grid">
            {Object.entries(test.confusion_matrix || {}).map(([key, value]) => <div key={key}><span>{key.toUpperCase()}</span><strong>{formatNumber(value)}</strong></div>)}
          </div>
          {plots.confusion_matrix && <a className="text-link" href={plots.confusion_matrix} target="_blank" rel="noreferrer">Open generated confusion-matrix image</a>}
        </Panel>
      </div>
    </>
  );
}

function LiveEarlyWarning({ risks, selectedHive, onHiveChange, detail, liveInference, thresholds }) {
  const timeline = (detail?.timeline || []).map((row) => ({ ...row, time: new Date(row.timestamp).toLocaleString() }));
  const latest = detail?.latest;
  return (
    <>
      <Panel
        title="Per-hive early-warning status"
        subtitle="The table shows the latest available historical observation. The same saved pipeline powers the POST inference endpoint."
        action={(
          <label className="chart-select-label">
            Selected hive
            <select value={selectedHive} onChange={(event) => onHiveChange(event.target.value)}>
              {risks.map((item) => <option key={item.hive_id} value={item.hive_id}>{item.hive_id}</option>)}
            </select>
          </label>
        )}
      >
        <div className="table-scroll hive-risk-table-wrap">
          <table>
            <thead><tr><th>Hive</th><th>Risk</th><th>Probability</th><th>ARM</th><th>Updated</th><th>Weight</th><th>CO₂</th></tr></thead>
            <tbody>
              {risks.map((item) => (
                <tr key={item.hive_id} className={item.hive_id === selectedHive ? 'selected-model-row' : ''} onClick={() => onHiveChange(item.hive_id)}>
                  <td><strong>{item.hive_id}</strong></td><td>{riskBadge(item.risk_level)}</td><td>{percentage(item.risk_percentage)}</td><td>{metric(item.arm, 6)} · {item.arm_trend}</td><td>{new Date(item.timestamp).toLocaleString()}</td><td>{metric(item.weight_kg)} kg</td><td>{metric(item.co2_ppm)} ppm</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {latest && (
        <div className="stats-grid">
          <StatCard label="Selected hive" value={latest.hive_id} note={new Date(latest.timestamp).toLocaleString()} />
          <StatCard label="Risk level" value={latest.risk_level} note={`High: ${thresholds.high || 'configured threshold'}`} />
          <StatCard label="Probability" value={latest.risk_percentage} unit="%" note={`${liveInference?.prediction_horizon_hours || 24}-hour future-event probability-like score`} />
          <StatCard label="ARM trend" value={latest.arm_trend} note={`24-hour change: ${metric(latest.arm, 6)}`} />
        </div>
      )}

      <div className="two-column-grid">
        <Panel title={`${selectedHive || 'Hive'} risk timeline`} subtitle="Latest 336 hourly observations with probability and ARM movement.">
          {timeline.length ? (
            <div className="chart-area-large">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="probability" name="Probability" stroke="#2563eb" dot={false} />
                  <Line type="monotone" dataKey="arm" name="ARM" stroke="#d97706" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message="No timeline is available for the selected hive." />}
        </Panel>

        <Panel title="Transparent signal explanations" subtitle="These are descriptive signals, not causal SHAP explanations.">
          <div className="signal-card-stack">
            {(latest?.signal_explanations || []).map((item) => (
              <article key={item.factor} className="signal-card">
                <div><strong>{item.factor}</strong><span>Strength {metric(item.signal_strength)}</span></div>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>
          <div className="api-contract-box">
            <strong>Live inference contract</strong>
            <code>{liveInference.method || 'POST'} {liveInference.endpoint || '/api/absconding/predict'}</code>
            <p>{liveInference.note}</p>
            <small>Required history: {liveInference.required_history_hours || 168} hourly observations per hive.</small>
          </div>
        </Panel>
      </div>
    </>
  );
}