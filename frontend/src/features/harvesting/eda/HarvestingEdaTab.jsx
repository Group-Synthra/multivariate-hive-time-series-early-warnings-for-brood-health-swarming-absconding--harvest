import { useEffect, useMemo, useState } from "react";

import { loadHarvestingEdaDashboard } from "../../../services/harvestingEdaService";

import "./HarvestingEdaTab.css";

const NUMBER_FORMAT = new Intl.NumberFormat("en-US");

function formatNumber(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  return NUMBER_FORMAT.format(value);
}

function formatPercent(value, digits = 3) {
  if (value === null || value === undefined) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDecimal(value, digits = 3) {
  if (value === null || value === undefined) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

function StatCard({ label, value, detail }) {
  return (
    <article className="harvest-stat-card">
      <span className="harvest-stat-card__label">{label}</span>
      <strong className="harvest-stat-card__value">{value}</strong>
      {detail ? (
        <span className="harvest-stat-card__detail">{detail}</span>
      ) : null}
    </article>
  );
}

function StatusItem({ ok, label, detail }) {
  return (
    <div className="harvest-status-item">
      <span
        aria-hidden="true"
        className={`harvest-status-item__indicator ${
          ok ? "is-ok" : "is-warning"
        }`}
      />
      <div>
        <strong>{label}</strong>
        {detail ? <p>{detail}</p> : null}
      </div>
    </div>
  );
}

function TargetDistribution({ rows }) {
  const maximumTotal = Math.max(
    1,
    ...rows.map((row) => row.total_rows),
  );

  return (
    <div className="harvest-distribution">
      {rows.map((row) => {
        const totalWidth =
          (row.total_rows / maximumTotal) * 100;
        const positiveWidth =
          row.total_rows > 0
            ? Math.max(
                0.8,
                (row.positive_rows / row.total_rows) * 100,
              )
            : 0;

        return (
          <div
            className="harvest-distribution__row"
            key={row.split}
          >
            <div className="harvest-distribution__heading">
              <strong>{row.split}</strong>
              <span>
                {formatPercent(row.positive_rate)} positive
              </span>
            </div>

            <div className="harvest-distribution__track">
              <div
                className="harvest-distribution__total"
                style={{ width: `${totalWidth}%` }}
              >
                <div
                  className="harvest-distribution__positive"
                  style={{ width: `${positiveWidth}%` }}
                />
              </div>
            </div>

            <div className="harvest-distribution__counts">
              <span>
                Positive: {formatNumber(row.positive_rows)}
              </span>
              <span>
                Negative: {formatNumber(row.negative_rows)}
              </span>
            </div>
          </div>
        );
      })}
      <p className="harvest-chart-note">
        Positive segments are given a minimum visible width. Exact
        counts are shown below each bar.
      </p>
    </div>
  );
}

function FeatureBars({ rows }) {
  const validRows = rows.filter((row) =>
    Number.isFinite(
      Number(row.absolute_standardized_mean_difference),
    ),
  );
  const maximum = Math.max(
    1,
    ...validRows.map((row) =>
      Math.abs(
        Number(row.absolute_standardized_mean_difference),
      ),
    ),
  );

  return (
    <div className="harvest-feature-bars">
      {validRows.slice(0, 12).map((row) => {
        const magnitude = Math.abs(
          Number(row.absolute_standardized_mean_difference),
        );
        const width = (magnitude / maximum) * 100;

        return (
          <div
            className="harvest-feature-bar"
            key={row.feature}
          >
            <div className="harvest-feature-bar__heading">
              <span title={row.feature}>{row.feature}</span>
              <strong>
                {formatDecimal(
                  row.standardized_mean_difference,
                  2,
                )}
              </strong>
            </div>
            <div className="harvest-feature-bar__track">
              <div
                className="harvest-feature-bar__fill"
                style={{ width: `${width}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function FeatureTable({ rows }) {
  return (
    <div className="harvest-table-wrapper">
      <table className="harvest-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Event mean</th>
            <th>Control mean</th>
            <th>SMD</th>
            <th>Event n</th>
            <th>Control n</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((row) => (
            <tr key={row.feature}>
              <td>{row.feature}</td>
              <td>{formatDecimal(row.event_mean)}</td>
              <td>{formatDecimal(row.control_mean)}</td>
              <td>
                {formatDecimal(
                  row.standardized_mean_difference,
                )}
              </td>
              <td>{formatNumber(row.event_n)}</td>
              <td>{formatNumber(row.control_n)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CoveragePanel({ summary }) {
  const eda = summary.eda;

  const rows = [
    {
      label: "Expected event-lead samples",
      value: eda.expected_event_lead_samples,
    },
    {
      label: "Available event-lead samples",
      value: eda.available_event_lead_samples,
    },
    {
      label: "Available matched controls",
      value: eda.available_matched_controls,
    },
    {
      label: "Missing event samples",
      value: eda.missing_event_lead_samples,
    },
    {
      label: "Missing controls",
      value: eda.missing_controls,
    },
  ];

  return (
    <div className="harvest-coverage-grid">
      {rows.map((row) => (
        <div className="harvest-coverage-item" key={row.label}>
          <span>{row.label}</span>
          <strong>{formatNumber(row.value)}</strong>
        </div>
      ))}
    </div>
  );
}

export default function HarvestingEdaTab() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedLead, setSelectedLead] = useState(72);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setIsLoading(true);
        const result = await loadHarvestingEdaDashboard();

        if (!cancelled) {
          setDashboard(result);
          const firstLead =
            result.summary.eda.lead_hours?.[0] ?? 72;
          setSelectedLead(firstLead);
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

  const selectedFeatures = useMemo(() => {
    if (!dashboard) {
      return [];
    }

    return (
      dashboard.topFeatures[String(selectedLead)] ??
      dashboard.featureComparison[String(selectedLead)] ??
      []
    );
  }, [dashboard, selectedLead]);

  const selectedFigure = useMemo(() => {
    if (!dashboard) {
      return null;
    }

    return dashboard.summary.eda.figures.find(
      (figure) => figure.lead_hours === selectedLead,
    );
  }, [dashboard, selectedLead]);

  if (isLoading) {
    return (
      <section className="harvesting-eda-state">
        <div className="harvesting-eda-spinner" />
        <p>Loading reviewed harvesting EDA…</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="harvesting-eda-state is-error">
        <h2>Harvesting EDA data is not available</h2>
        <p>{error}</p>
        <code>
          python scripts/export_harvest_eda_for_frontend.py
        </code>
      </section>
    );
  }

  const { summary } = dashboard;
  const target = summary.target;
  const features = summary.features;
  const grouped = summary.grouped_validation ?? {};

  return (
    <section className="harvesting-eda">
      <header className="harvesting-eda__header">
        <div>
          <span className="harvesting-eda__eyebrow">
            Time-Optimal Honey Harvesting
          </span>
          <h1>Reviewed Event Exploratory Analysis</h1>
          <p>
            Past-only feature analysis for the 72-hour probable
            harvest forecasting target.
          </p>
        </div>

        <div className="harvesting-eda__generated">
          <span>Last exported</span>
          <strong>
            {new Date(summary.generated_at).toLocaleString()}
          </strong>
        </div>
      </header>
{/* 
      <div className="harvesting-eda__notice">
        <strong>Research limitation</strong>
        <p>{summary.limitations.join(" ")}</p>
      </div> */}

      <div className="harvest-stat-grid">
        <StatCard
          label="Reviewed events"
          value={formatNumber(target.reviewed_event_count)}
          detail={`${formatNumber(
            target.reviewed_positive_hives,
          )} positive hives`}
        />
        <StatCard
          label="Modelling rows"
          value={formatNumber(target.final_modelling_rows)}
          detail={`${formatNumber(
            target.target_positive_rows,
          )} positive rows`}
        />
        <StatCard
          label="Target prevalence"
          value={formatPercent(target.target_positive_rate)}
          detail={`${target.horizon_hours}-hour horizon`}
        />
        <StatCard
          label="Engineered features"
          value={formatNumber(features.feature_count)}
          detail={`${features.minimum_history_hours} h history`}
        />
        <StatCard
          label="Training events"
          value={formatNumber(
            target.events_by_split.train ?? 0,
          )}
          detail="Official chronological split"
        />
        <StatCard
          label="Validation / test"
          value={`${formatNumber(
            target.events_by_split.validation ?? 0,
          )} / ${formatNumber(
            target.events_by_split.test ?? 0,
          )}`}
          detail="Independent reviewed events"
        />
      </div>

      <div className="harvesting-eda__two-column">
        <article className="harvest-panel">
          <div className="harvest-panel__header">
            <div>
              <span>Class balance</span>
              <h2>Target distribution by split</h2>
            </div>
          </div>

          <TargetDistribution rows={target.split_balance} />

        </article>

        <article className="harvest-panel">
          <div className="harvest-panel__header">
            <div>
              <span>Feature integrity</span>
              <h2>Quality and leakage checks</h2>
            </div>
          </div>

          <div className="harvest-status-list">
            <StatusItem
              ok={
                features.leakage_columns_present.length === 0
              }
              label="No prohibited leakage columns"
              detail={
                features.leakage_columns_present.length === 0
                  ? "Reviewed labels and event indicators are excluded."
                  : features.leakage_columns_present.join(", ")
              }
            />
            <StatusItem
              ok={features.positive_rows_removed === 0}
              label="Complete positive warning windows"
              detail={`${formatNumber(
                features.positive_rows_removed,
              )} positive rows removed during feature creation.`}
            />
            <StatusItem
              ok
              label="Past-only feature policy"
              detail={
                features.history_policy ??
                "Features use only current and historical sensor readings."
              }
            />
            <StatusItem
              ok={features.detected_non_hourly_gaps >= 0}
              label="Sensor timeline audited"
              detail={`${formatNumber(
                features.contiguous_segment_count,
              )} contiguous segments and ${formatNumber(
                features.detected_non_hourly_gaps,
              )} non-hourly gaps detected.`}
            />
          </div>
        </article>
      </div>

      <article className="harvest-panel harvest-panel--wide">
        <div className="harvest-panel__header harvest-panel__header--responsive">
          <div>
            <span>Matched event-control analysis</span>
            <h2>Feature differences by prediction lead</h2>
          </div>

          <div
            className="harvest-lead-selector"
            aria-label="Select prediction lead time"
          >
            {summary.eda.lead_hours.map((lead) => (
              <button
                className={
                  lead === selectedLead ? "is-active" : ""
                }
                key={lead}
                onClick={() => setSelectedLead(lead)}
                type="button"
              >
                {lead} h
              </button>
            ))}
          </div>
        </div>

        <div className="harvesting-eda__feature-grid">
          <div>
            <h3>
              Largest standardized differences at {selectedLead} h
            </h3>
            <FeatureBars rows={selectedFeatures} />
          </div>

          <div>
            {selectedFigure ? (
              <figure className="harvest-feature-figure">
                <img
                  alt={`Top harvesting features at ${selectedLead} hour lead`}
                  src={selectedFigure.url}
                />
                <figcaption>
                  Absolute standardized mean differences between
                  reviewed event samples and matched controls.
                </figcaption>
              </figure>
            ) : (
              <div className="harvest-empty-figure">
                No exported figure is available for this lead.
              </div>
            )}
          </div>
        </div>

        <FeatureTable rows={selectedFeatures} />
      </article>

      <div className="harvesting-eda__two-column">
        <article className="harvest-panel">
          <div className="harvest-panel__header">
            <div>
              <span>Sampling audit</span>
              <h2>Event and control coverage</h2>
            </div>
          </div>
          <CoveragePanel summary={summary} />
          <p className="harvest-panel__footnote">
            Controls are drawn from the same hive, split and
            hour of day, and are at least{" "}
            {formatNumber(
              summary.eda.control_exclusion_hours,
            )}{" "}
            hours from a reviewed event.
          </p>
        </article>

        <article className="harvest-panel">
          <div className="harvest-panel__header">
            <div>
              <span>Secondary evaluation</span>
              <h2>Grouped hive validation</h2>
            </div>
          </div>

          {grouped.fold_count ? (
            <div className="harvest-grouped-summary">
              <StatCard
                label="Grouped folds"
                value={formatNumber(grouped.fold_count)}
                detail="Leave-one-positive-hive-out"
              />
              <StatCard
                label="Positive train hives"
                value={formatNumber(
                  grouped.positive_training_hive_count,
                )}
                detail={`${formatNumber(
                  grouped.training_event_count,
                )} training events`}
              />
              <p>{grouped.warning}</p>
            </div>
          ) : (
            <div className="harvest-empty-figure">
              Run the grouped hive validation export to show this
              section.
            </div>
          )}
        </article>
      </div>
    </section>
  );
}
