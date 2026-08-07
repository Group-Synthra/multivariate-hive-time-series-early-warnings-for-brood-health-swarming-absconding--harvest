import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Battery,
  CheckCircle2,
  Clock3,
  Database,
  Droplets,
  RefreshCw,
  Scale,
  ShieldCheck,
  Thermometer,
  Wifi,
  WifiOff,
  Wind,
  Zap,
} from 'lucide-react';
import { Panel } from '../../../components/common/Panel';
import { StatCard } from '../../../components/common/StatCard';
import { useBroodIoT } from '../hooks/useBroodHealthData';
import {
  formatBhsi,
  formatHealthScore,
  freshnessLabel,
  healthLevelFromScore,
  numberValue,
  signedNumber,
  timestampValue,
} from '../utils/broodHealth';
import {
  ForecastTrajectoryChart,
  HealthScoreComparisonChart,
  LiveEarlyWarningTimeline,
} from './BroodHealthCharts';
import {
  HealthScoreGauge,
  RoDMeter,
  StabilityGauge,
} from './HealthScoreGauge';

const DEFAULT_REFRESH_SECONDS = 600;

const SENSOR_CARDS = [
  ['temperature_c', 'Internal temperature', '°C', Thermometer],
  ['humidity_pct', 'Internal humidity', '% RH', Droplets],
  ['co2_ppm', 'Internal CO₂', 'ppm', Wind],
  ['weight_kg', 'Hive weight', 'kg', Scale],
  ['external_temp', 'External temperature', '°C', Thermometer],
  ['external_humidity', 'External humidity', '% RH', Droplets],
];

function severitySlug(value) {
  return String(value || 'Normal')
    .toLowerCase()
    .replaceAll(' ', '-');
}

function WarningPanel({
  warning,
  current,
  forecast,
  forecastIndicators,
}) {
  const severity = warning?.severity || warning?.level || 'Normal';
  const slug = severitySlug(severity);

  const predictedLevel = healthLevelFromScore(
    forecast?.exact_score,
  );

  const reasons = warning?.reasons || [
    'No alert condition was triggered.',
  ];

  const actions = warning?.recommended_actions || [
    'Continue routine monitoring.',
  ];

  const confidenceNotes =
    warning?.confidence_notes || [];

  const SeverityIcon =
    severity === 'Normal'
      ? CheckCircle2
      : severity === 'Critical Alert'
        ? Zap
        : AlertTriangle;

  return (
    <section className={`brood-warning-panel ${slug}`}>
      <div className="brood-warning-accent">
        <SeverityIcon size={28} />
      </div>

      <div className="brood-warning-body">
        <div className="brood-warning-header">
          <div>
            <span className="brood-warning-eyebrow">
              COMPOSITE EARLY-WARNING
            </span>

            <h3>
              {warning?.title || 'Brood-health status'}
            </h3>

            <p>{warning?.summary}</p>
          </div>

          <div className="brood-warning-badges">
            <span
              className={`alert-severity ${slug}`}
            >
              {severity}
            </span>

            <span
              className={`health-level ${predictedLevel.toLowerCase()}`}
            >
              +6 h health: {predictedLevel}
            </span>
          </div>
        </div>

        <div className="brood-warning-metrics">
          <div>
            <span>Current</span>
            <strong>
              {formatHealthScore(current?.score)}
            </strong>
          </div>

          <div>
            <span>Exact +6 h</span>
            <strong>
              {formatHealthScore(
                forecast?.exact_score,
              )}
            </strong>
          </div>

          <div>
            <span>Safety minimum</span>
            <strong>
              {formatHealthScore(
                forecast?.safety_minimum_score,
              )}
            </strong>
          </div>

          <div>
            <span>Forecast BHSI</span>
            <strong>
              {formatBhsi(
                forecastIndicators?.bhsi,
              )}
            </strong>
          </div>

          <div>
            <span>Forecast RoD</span>
            <strong>
              {signedNumber(
                forecastIndicators?.rod_points_per_hour,
                2,
              )}
            </strong>
            <small> pts/h</small>
          </div>
        </div>

        <div className="brood-warning-columns">
          <div className="brood-warning-drivers">
            <strong>Alert drivers</strong>

            <ul>
              {reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </div>

          <div className="brood-warning-actions">
            <div className="brood-warning-action-heading">
              <strong>
                Recommended beekeeper actions
              </strong>

              {warning?.urgency && (
                <span>{warning.urgency}</span>
              )}
            </div>

            <ol>
              {actions.map((action) => (
                <li key={action}>{action}</li>
              ))}
            </ol>
          </div>
        </div>

        {confidenceNotes.length > 0 && (
          <details className="brood-warning-confidence">
            <summary>
              Data confidence notes (
              {confidenceNotes.length})
            </summary>

            <ul>
              {confidenceNotes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </section>
  );
}

function IntervalCard({
  interval80,
  interval90,
  score,
}) {
  return (
    <div className="brood-interval-card">
      <ShieldCheck size={22} />

      <div>
        <span>
          Exact +6 h prediction interval
        </span>

        <strong>
          {formatHealthScore(score)} / 100
        </strong>

        <p>
          80% interval:{' '}
          <b>
            {formatHealthScore(
              interval80?.[0],
            )}
            –
            {formatHealthScore(
              interval80?.[1],
            )}
          </b>

          <br />

          90% interval:{' '}
          <b>
            {formatHealthScore(
              interval90?.[0],
            )}
            –
            {formatHealthScore(
              interval90?.[1],
            )}
          </b>
        </p>
      </div>
    </div>
  );
}

export function BroodIoTTab() {
  const iot = useBroodIoT(true);

  const [autoRefresh, setAutoRefresh] =
    useState(true);

  const [now, setNow] = useState(
    Date.now(),
  );

  const [lastRefresh, setLastRefresh] =
    useState(Date.now());

  const health = iot.health.data;
  const prediction = iot.prediction.data;

  const databaseReady = Boolean(
    health?.database?.connected,
  );

  const modelReady = Boolean(
    health?.model?.ready,
  );

  const current =
    prediction?.current_condition || {};

  const forecast =
    prediction?.prediction || {};

  const forecastIndicators =
    prediction?.forecast_indicators || {};

  const warning =
    prediction?.warning || {};

  const refreshSeconds = Number(
    iot.devices.data?.refresh_seconds
      || health?.database?.refresh_seconds
      || DEFAULT_REFRESH_SECONDS,
  );

  const refreshMs =
    Math.max(30, refreshSeconds) * 1000;

  /*
   * UI clock.
   *
   * This does not call the API every second.
   * It only keeps the displayed countdown moving.
   */
  useEffect(() => {
    const clock = window.setInterval(
      () => setNow(Date.now()),
      1000,
    );

    return () =>
      window.clearInterval(clock);
  }, []);

  /*
   * Automatic live prediction refresh.
   */
  useEffect(() => {
    if (
      !autoRefresh
      || !iot.selectedDevice
    ) {
      return undefined;
    }

    const timer = window.setInterval(
      async () => {
        await iot.loadPrediction();

        const refreshedAt = Date.now();

        setLastRefresh(refreshedAt);
        setNow(refreshedAt);
      },
      refreshMs,
    );

    return () =>
      window.clearInterval(timer);
  }, [
    autoRefresh,
    iot.loadPrediction,
    iot.selectedDevice,
    refreshMs,
  ]);

  /*
   * MM:SS countdown.
   */
  const countdown = useMemo(() => {
    if (!autoRefresh) {
      return 'Paused';
    }

    const remainingMs =
      lastRefresh + refreshMs - now;

    const seconds = Math.max(
      0,
      Math.ceil(remainingMs / 1000),
    );

    const minutes =
      Math.floor(seconds / 60);

    const remainingSeconds =
      String(seconds % 60).padStart(
        2,
        '0',
      );

    return `${minutes}:${remainingSeconds}`;
  }, [
    autoRefresh,
    lastRefresh,
    now,
    refreshMs,
  ]);

  /*
   * Manual refresh used by the compact icon
   * next to the countdown timer.
   */
  const handleLivePredictionRefresh =
    async () => {
      if (
        !iot.selectedDevice
        || iot.prediction.loading
        || !modelReady
      ) {
        return;
      }

      await iot.loadPrediction();

      const refreshedAt = Date.now();

      setLastRefresh(refreshedAt);
      setNow(refreshedAt);
    };

  const latestSensors = {
    ...(prediction?.latest_sensors || {}),
    ...(prediction?.latest_raw_sensors || {}),
    ...(prediction?.context || {}),
  };

  return (
    <div className="page-stack">
      {/* ================================================================ */}
      {/* Live dashboard heading                                           */}
      {/* ================================================================ */}

      <div className="brood-section-heading">
        <div>
          <span className="eyebrow">
            LIVE HIVE MONITORING
          </span>

          <h3>
            Brood Health Early-Warning Dashboard
          </h3>

          <p>
            Current health, exact +6-hour
            forecast, forecast stability and
            deterioration trend.
          </p>
        </div>

        <div className="brood-action-group">
          <label className="brood-switch">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) =>
                setAutoRefresh(
                  event.target.checked,
                )
              }
            />

            <span />

            Auto refresh
          </label>

          <button
            type="button"
            className="button button-outline"
            onClick={async () => {
              await iot.refresh();

              const refreshedAt =
                Date.now();

              setLastRefresh(
                refreshedAt,
              );

              setNow(refreshedAt);
            }}
          >
            <RefreshCw size={16} />

            Refresh
          </button>
        </div>
      </div>

      {/* ================================================================ */}
      {/* Live data source                                                 */}
      {/* ================================================================ */}

      <div className="brood-section-heading compact brood-admin-heading">
        <div>
          <h3>Live Data Source</h3>
        </div>
      </div>

      <div
        className={`brood-live-status ${
          databaseReady
            ? 'connected'
            : 'disconnected'
        }`}
      >
        {databaseReady ? (
          <Wifi size={24} />
        ) : (
          <WifiOff size={24} />
        )}

        <div>
          <strong>
            {databaseReady
              ? 'Live PostgreSQL connection active'
              : 'Live PostgreSQL connection unavailable'}
          </strong>

          <p>
            {databaseReady
              ? `Source: ${health.database.schema}.${health.database.table} · Latest stored reading: ${timestampValue(
                health.database
                  .latest_recorded_at,
              )}`
              : health?.database?.error}
          </p>
        </div>

        <Database size={22} />
      </div>

      {!modelReady && (
        <div className="brood-alert danger">
          <AlertTriangle size={20} />

          <div>
            <strong>
              The Brood Health v6 model is not
              available
            </strong>

            <p>
              {health?.model?.error
                || 'Delete the old model artifacts and retrain.'}
            </p>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* Live hive selection                                              */}
      {/* ================================================================ */}

      <Panel
        title="Live hive selection"
        subtitle="Select a hive and monitor its latest forecast."
        action={(
          <div className="brood-device-actions">
            {/* Hive selector */}
            <select
              className="brood-select"
              value={iot.selectedDevice}
              onChange={(event) =>
                iot.setSelectedDevice(
                  event.target.value,
                )
              }
            >
              <option value="">
                Select hive
              </option>

              {(iot.devices.data?.devices || [])
                .map((item) => (
                  <option
                    key={item.device_id}
                    value={item.device_id}
                  >
                    {item.device_id}
                  </option>
                ))}
            </select>

            {/* Next automatic refresh timer */}
            <div
              className={`brood-next-update ${
                !autoRefresh
                  ? 'paused'
                  : ''
              }`}
              title={
                autoRefresh
                  ? 'Time remaining until the next automatic live prediction refresh'
                  : 'Automatic refresh is paused'
              }
            >
              <Clock3 size={16} />

              <div>
                <span>
                  Next update
                </span>

                <strong>
                  {countdown}
                </strong>
              </div>
            </div>

            {/* Compact manual refresh button */}
            <button
              type="button"
              className="brood-live-refresh-button"
              disabled={
                !iot.selectedDevice
                || iot.prediction.loading
                || !modelReady
              }
              onClick={
                handleLivePredictionRefresh
              }
              title="Refresh live prediction now"
              aria-label="Refresh live prediction now"
            >
              <RefreshCw
                size={18}
                className={
                  iot.prediction.loading
                    ? 'brood-refresh-spinning'
                    : ''
                }
              />
            </button>
          </div>
        )}
      >
        <div className="brood-live-meta">
          <span>
            Latest IoT reading:{' '}
            {timestampValue(
              prediction?.live_latest_timestamp
              || prediction?.latest_timestamp,
            )}
          </span>

          <span>
            Rolling model anchor:{' '}
            {timestampValue(
              forecast.forecast_anchor_timestamp
              || prediction?.latest_timestamp,
            )}
          </span>

          <span>
            Forecast target:{' '}
            {timestampValue(
              forecast.exact_forecast_timestamp,
            )}
          </span>

          <span>
            Observed interval:{' '}
            {numberValue(
              prediction?.reading_interval_minutes,
              0,
            )}{' '}
            min
          </span>

          <span>
            Raw readings:{' '}
            {numberValue(
              prediction?.raw_rows,
              0,
            )}
          </span>

          <span>
            Hourly windows:{' '}
            {numberValue(
              prediction?.hourly_rows,
              0,
            )}
          </span>
        </div>
      </Panel>

      {/* ================================================================ */}
      {/* API errors                                                       */}
      {/* ================================================================ */}

      {iot.prediction.error && (
        <div className="brood-alert danger">
          <AlertTriangle size={20} />

          <div>
            <strong>
              Live prediction failed
            </strong>

            <p>
              {iot.prediction.error.message}
            </p>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* Prediction dashboard                                             */}
      {/* ================================================================ */}

      {prediction && (
        <>
          {/* ------------------------------------------------------------ */}
          {/* Early warning                                                */}
          {/* ------------------------------------------------------------ */}

          <div className="brood-section-heading compact brood-admin-heading">
            <div>
              <h3>
                Early-Warning Status
              </h3>

              <p>
                Health level and combined
                deterioration alerts.
              </p>
            </div>
          </div>

          <WarningPanel
            warning={warning}
            current={current}
            forecast={forecast}
            forecastIndicators={
              forecastIndicators
            }
          />

          {/* ------------------------------------------------------------ */}
          {/* Main gauges                                                  */}
          {/* ------------------------------------------------------------ */}

          <div className="brood-section-heading compact brood-admin-heading">
            <div>
              <h3>
                Current & Future Health Indicators
              </h3>

              <p>
                Current score, exact +6-hour
                score, Forecast BHSI and
                Forecast RoD.
              </p>
            </div>
          </div>

          <div className="brood-four-gauge-grid">
            <HealthScoreGauge
              score={current.score}
              level={current.level}
              label="Current Brood Health Score"
              badge="Now"
              detail="Latest IoT brood-health condition score."
            />

            <HealthScoreGauge
              score={forecast.exact_score}
              level={forecast.exact_level}
              label={`Exact Brood Health Score at +${
                forecast.horizon_hours || 6
              } h`}
              badge={timestampValue(
                forecast.exact_forecast_timestamp,
              )}
              detail={`Selected model: ${
                prediction.model?.model_name
                || health?.model?.model_name
                || 'trained model'
              }`}
            />

            <StabilityGauge
              score={
                forecastIndicators.bhsi
              }
              level={
                forecastIndicators.stability_level
              }
              label="Forecast BHSI"
              badge="Next 6 h"
              detail="Predicted six-hour health-path stability."
            />

            <RoDMeter
              value={
                forecastIndicators
                  .rod_points_per_hour
              }
              label={
                forecastIndicators.trend_label
              }
              title="Forecast RoD"
              badge="Next 6 h"
              detail="Expected score-change rate over the next six hours."
            />
          </div>

          {/* ------------------------------------------------------------ */}
          {/* Forecast snapshot                                            */}
          {/* ------------------------------------------------------------ */}

          <div className="brood-section-heading compact brood-admin-heading">
            <div>
              <h3>
                Forecast Snapshot
              </h3>

              <p>
                Key forecast changes, safety
                minimum and data readiness.
              </p>
            </div>
          </div>

          <div className="stats-grid stats-grid-six">
            <StatCard
              label="Exact score change"
              value={signedNumber(
                forecast.exact_change_points,
                2,
              )}
              unit="points"
              note="+6 h minus current"
            />

            <StatCard
              label="Exact forecast drop"
              value={numberValue(
                forecast.exact_drop_points,
                2,
              )}
              unit="points"
              note="Expected decline"
            />

            <StatCard
              label="Safety minimum"
              value={formatHealthScore(
                forecast.safety_minimum_score,
              )}
              unit="/100"
              note={`${healthLevelFromScore(
                forecast.safety_minimum_score,
              )} · lowest predicted 1–6 h point`}
            />

            <StatCard
              label="Feature completeness"
              value={numberValue(
                prediction
                  .feature_completeness_percentage,
                1,
              )}
              unit="%"
            />

            <StatCard
              label="History sufficiency"
              value={
                prediction.history_sufficiency
                || '—'
              }
              note={`Recommended: ${
                prediction
                  .minimum_recommended_history_hours
                || 72
              } h`}
            />

            <StatCard
              label="Data freshness"
              value={freshnessLabel(
                prediction.data_freshness_minutes,
              )}
              note={`${numberValue(
                prediction.data_freshness_minutes,
                0,
              )} minutes`}
            />
          </div>

          {/* ------------------------------------------------------------ */}
          {/* Observed vs future                                           */}
          {/* ------------------------------------------------------------ */}

          <Panel
            title="Observed vs Forecast Indicators"
            subtitle="Recent observed values compared with the next-six-hour forecast."
          >
            <div className="stats-grid stats-grid-six">
              <StatCard
                label="Observed BHSI"
                value={formatBhsi(
                  current.bhsi,
                )}
                unit="/100"
                note={`${
                  current.stability_level || '—'
                } · recent historical stability`}
              />

              <StatCard
                label="Forecast BHSI"
                value={formatBhsi(
                  forecastIndicators.bhsi,
                )}
                unit="/100"
                note={`${
                  forecastIndicators
                    .stability_level
                  || '—'
                } · predicted next-six-hour stability`}
              />

              <StatCard
                label="Observed RoD"
                value={signedNumber(
                  current.rod_points_per_hour,
                  2,
                )}
                unit="points/h"
                note={
                  current.trend_label
                  || '—'
                }
              />

              <StatCard
                label="Forecast RoD"
                value={signedNumber(
                  forecastIndicators
                    .rod_points_per_hour,
                  2,
                )}
                unit="points/h"
                note={
                  forecastIndicators
                    .trend_label
                  || '—'
                }
              />

              <StatCard
                label="Forecast window start"
                value={timestampValue(
                  forecastIndicators
                    .window_start_timestamp,
                )}
                note="Latest rolling IoT anchor"
              />

              <StatCard
                label="Forecast window end"
                value={timestampValue(
                  forecastIndicators
                    .window_end_timestamp,
                )}
                note="Exact +6-hour target"
              />
            </div>
          </Panel>

          {/* ------------------------------------------------------------ */}
          {/* Comparison + forecast details                                */}
          {/* ------------------------------------------------------------ */}

          <div className="two-column-grid">
            <Panel
              title="Current vs +6-Hour Health"
              subtitle="Current, exact future and safety-minimum positions on the health scale."
            >
              <HealthScoreComparisonChart
                currentScore={
                  current.score
                }
                exactScore={
                  forecast.exact_score
                }
                safetyScore={
                  forecast
                    .safety_minimum_score
                }
                forecastHorizonHours={
                  forecast.horizon_hours || 6
                }
                currentTimestamp={
                  forecast
                    .forecast_anchor_timestamp
                  || prediction?.latest_timestamp
                }
                forecastTimestamp={
                  forecast
                    .exact_forecast_timestamp
                }
              />
            </Panel>

            <Panel
              title="Forecast Details"
              subtitle="Prediction interval, model and forecast timestamps."
            >
              <IntervalCard
                interval80={
                  forecast.prediction_interval_80
                }
                interval90={
                  forecast.prediction_interval_90
                }
                score={forecast.exact_score}
              />

              <div className="brood-info-list">
                <span>
                  Selected model

                  <strong>
                    {prediction.model?.model_name
                      || '—'}
                  </strong>
                </span>

                <span>
                  Primary target

                  <strong>
                    Exact score at +
                    {forecast.horizon_hours || 6}{' '}
                    hours
                  </strong>
                </span>

                <span>
                  Forecast anchor

                  <strong>
                    {timestampValue(
                      forecast
                        .forecast_anchor_timestamp,
                    )}
                  </strong>
                </span>

                <span>
                  Forecast target

                  <strong>
                    {timestampValue(
                      forecast
                        .exact_forecast_timestamp,
                    )}
                  </strong>
                </span>
              </div>
            </Panel>
          </div>

          {/* ------------------------------------------------------------ */}
          {/* Forecast trajectory                                          */}
          {/* ------------------------------------------------------------ */}

          <Panel
            title="Six-Hour Health Forecast"
            subtitle="Current score and predicted health path through the exact +6-hour target."
          >
            <ForecastTrajectoryChart
              data={
                forecast.display_trajectory
                || forecast.trajectory
              }
              currentScore={
                current.score
              }
              exactHorizon={
                forecast.horizon_hours || 6
              }
              anchorTimestamp={
                forecast
                  .forecast_anchor_timestamp
              }
              targetTimestamp={
                forecast
                  .exact_forecast_timestamp
              }
            />
          </Panel>

          {/* ------------------------------------------------------------ */}
          {/* Sensor cards                                                 */}
          {/* ------------------------------------------------------------ */}

          <section>
            <div className="brood-section-heading compact">
              <div>
                <h3>
                  Live Sensor Readings
                </h3>

                <p>
                  Latest internal, external and
                  hive-weight readings.
                </p>
              </div>
            </div>

            <div className="stats-grid stats-grid-six">
              {SENSOR_CARDS.map(
                ([
                  key,
                  label,
                  unit,
                  Icon,
                ]) => (
                  <StatCard
                    key={key}
                    label={label}
                    value={numberValue(
                      latestSensors[key],
                      key === 'co2_ppm'
                        ? 0
                        : 2,
                    )}
                    unit={unit}
                    icon={Icon}
                  />
                ),
              )}
            </div>
          </section>

          {/* ------------------------------------------------------------ */}
          {/* Components + data quality                                    */}
          {/* ------------------------------------------------------------ */}

          <div className="two-column-grid">
            <Panel
              title="Current Score Components"
              subtitle="Contribution of each monitored condition to the current score."
            >
              <div className="brood-component-grid">
                {Object.entries(
                  prediction.score_components
                  || {},
                ).map(
                  ([key, value]) => (
                    <div key={key}>
                      <span>
                        {key.replaceAll(
                          '_',
                          ' ',
                        )}
                      </span>

                      <strong>
                        {numberValue(
                          value,
                          2,
                        )}
                      </strong>

                      <div>
                        <i
                          style={{
                            width: `${
                              Math.max(
                                0,
                                Math.min(
                                  100,
                                  Number(
                                    value || 0,
                                  ),
                                ),
                              )
                            }%`,
                          }}
                        />
                      </div>
                    </div>
                  ),
                )}
              </div>
            </Panel>

            <Panel
              title="Data Quality & Readiness"
              subtitle="Operational checks for the current live prediction."
            >
              <div className="brood-info-list">
                <span>
                  Feature completeness

                  <strong>
                    {numberValue(
                      prediction
                        .feature_completeness_percentage,
                      1,
                    )}
                    %
                  </strong>
                </span>

                <span>
                  History

                  <strong>
                    {prediction
                      .history_sufficiency
                      || '—'}
                  </strong>
                </span>

                <span>
                  Data freshness

                  <strong>
                    {freshnessLabel(
                      prediction
                        .data_freshness_minutes,
                    )}
                  </strong>
                </span>

                <span>
                  Battery

                  <strong>
                    <Battery size={14} />

                    {numberValue(
                      prediction.context
                        ?.battery_voltage,
                      2,
                    )}{' '}
                    V
                  </strong>
                </span>
              </div>
            </Panel>
          </div>

          {/* ------------------------------------------------------------ */}
          {/* Historical timeline                                          */}
          {/* ------------------------------------------------------------ */}

          <Panel
            title="Recent Health & Forecast Timeline"
            subtitle="Recent current scores and forecast indicators for the selected hive."
          >
            <LiveEarlyWarningTimeline
              data={prediction.history}
            />
          </Panel>
        </>
      )}
    </div>
  );
}