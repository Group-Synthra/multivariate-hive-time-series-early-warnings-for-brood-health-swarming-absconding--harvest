import React from 'react';

const DEFAULT_LABELS = ['Critical', 'Poor', 'Good', 'Excellent'];

const HEALTH_COLORS = {
  Critical: '#dc2626',
  Poor: '#d97706',
  Good: '#2563eb',
  Excellent: '#0f766e',
};

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function formatPercent(value, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}%` : '—';
}

function readableLabel(value) {
  if (!value) return '';
  const text = String(value);
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function HealthLevelClassificationChart({
  matrix,
  labels = DEFAULT_LABELS,
}) {
  const safeLabels = labels?.length ? labels.map(readableLabel) : DEFAULT_LABELS;

  const safeMatrix = safeLabels.map((_, rowIndex) =>
    safeLabels.map((__, columnIndex) =>
      toNumber(matrix?.[rowIndex]?.[columnIndex]),
    ),
  );

  const totalCases = safeMatrix
    .flat()
    .reduce((sum, value) => sum + value, 0);

  const correctCases = safeMatrix.reduce(
    (sum, row, index) => sum + toNumber(row[index]),
    0,
  );

  const overallAccuracy = totalCases > 0
    ? (correctCases / totalCases) * 100
    : 0;

  const rows = safeLabels.map((actualLabel, rowIndex) => {
    const row = safeMatrix[rowIndex] || [];
    const rowTotal = row.reduce((sum, value) => sum + value, 0);

    const predictions = safeLabels.map((predictedLabel, columnIndex) => {
      const count = toNumber(row[columnIndex]);
      const percentage = rowTotal > 0 ? (count / rowTotal) * 100 : 0;

      return {
        predictedLabel,
        count,
        percentage,
        isCorrect: rowIndex === columnIndex,
      };
    });

    const correct = predictions[rowIndex] || {
      count: 0,
      percentage: 0,
    };

    const strongestMistake = predictions
      .filter((item) => !item.isCorrect)
      .sort((a, b) => b.percentage - a.percentage)[0];

    return {
      actualLabel,
      rowTotal,
      predictions,
      correctCount: correct.count,
      correctPercentage: correct.percentage,
      strongestMistake,
    };
  });

  const styles = {
    root: {
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
    },
    summary: {
      display: 'grid',
      gridTemplateColumns: 'minmax(0, 1fr) auto',
      alignItems: 'center',
      gap: '1rem',
      padding: '0.85rem 1rem',
      border: '1px solid #dbe5f0',
      borderRadius: '12px',
      background: 'linear-gradient(135deg, #f8fbff 0%, #ffffff 60%, #f0fdf4 100%)',
    },
    summaryText: {
      color: '#64748b',
      fontSize: '0.72rem',
      lineHeight: 1.5,
    },
    summaryAccuracy: {
      textAlign: 'right',
      whiteSpace: 'nowrap',
    },
    summaryAccuracyLabel: {
      display: 'block',
      color: '#64748b',
      fontSize: '0.64rem',
      fontWeight: 700,
    },
    summaryAccuracyValue: {
      display: 'block',
      marginTop: '0.12rem',
      color: '#0f172a',
      fontSize: '1.25rem',
      fontWeight: 900,
    },
    legend: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '0.45rem',
    },
    legendItem: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '0.35rem',
      padding: '0.32rem 0.5rem',
      border: '1px solid #e2e8f0',
      borderRadius: '999px',
      background: '#ffffff',
      color: '#475569',
      fontSize: '0.64rem',
      fontWeight: 700,
    },
    dot: {
      width: '9px',
      height: '9px',
      borderRadius: '50%',
      flex: '0 0 9px',
    },
    rows: {
      display: 'flex',
      flexDirection: 'column',
      gap: '0.8rem',
    },
    row: {
      padding: '0.78rem',
      border: '1px solid #e2e8f0',
      borderRadius: '11px',
      background: '#ffffff',
    },
    rowHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      gap: '0.75rem',
      marginBottom: '0.48rem',
    },
    actualLabel: {
      color: '#0f172a',
      fontSize: '0.73rem',
      fontWeight: 850,
    },
    correctText: {
      color: '#475569',
      fontSize: '0.66rem',
      fontWeight: 700,
      textAlign: 'right',
    },
    bar: {
      display: 'flex',
      width: '100%',
      minHeight: '34px',
      overflow: 'hidden',
      borderRadius: '8px',
      background: '#eef2f7',
      boxShadow: 'inset 0 0 0 1px rgba(148, 163, 184, .20)',
    },
    segment: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minWidth: 0,
      color: '#ffffff',
      fontSize: '0.62rem',
      fontWeight: 850,
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textShadow: '0 1px 2px rgba(15, 23, 42, .30)',
      transition: 'width .25s ease',
    },
    rowFooter: {
      display: 'flex',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '0.4rem 1rem',
      marginTop: '0.45rem',
      color: '#64748b',
      fontSize: '0.62rem',
      lineHeight: 1.45,
    },
    correctBadge: {
      color: '#047857',
      fontWeight: 800,
    },
    mistakeText: {
      color: '#64748b',
    },
  };

  if (!matrix?.length || totalCases <= 0) {
    return (
      <div
        style={{
          padding: '1rem',
          border: '1px dashed #cbd5e1',
          borderRadius: '10px',
          color: '#64748b',
          fontSize: '0.75rem',
          textAlign: 'center',
        }}
      >
        Health-level classification results are not available for this training run.
      </div>
    );
  }

  return (
    <div style={styles.root}>
      <div style={styles.summary}>
        <div style={styles.summaryText}>
          Each bar represents the <strong>actual health level</strong>. The coloured
          sections show what percentage the model predicted as Critical, Poor, Good,
          or Excellent.
        </div>

        <div style={styles.summaryAccuracy}>
          <span style={styles.summaryAccuracyLabel}>Overall Accuracy</span>
          <strong style={styles.summaryAccuracyValue}>
            {formatPercent(overallAccuracy, 1)}
          </strong>
        </div>
      </div>

      <div style={styles.legend}>
        {safeLabels.map((label) => (
          <span style={styles.legendItem} key={label}>
            <span
              style={{
                ...styles.dot,
                background: HEALTH_COLORS[label] || '#64748b',
              }}
            />
            Predicted {label}
          </span>
        ))}
      </div>

      <div style={styles.rows}>
        {rows.map((row) => (
          <div style={styles.row} key={row.actualLabel}>
            <div style={styles.rowHeader}>
              <span style={styles.actualLabel}>
                Actual {row.actualLabel}
              </span>

              <span style={styles.correctText}>
                Correctly predicted:{' '}
                <strong style={{ color: HEALTH_COLORS[row.actualLabel] || '#0f172a' }}>
                  {formatPercent(row.correctPercentage, 1)}
                </strong>
              </span>
            </div>

            <div style={styles.bar}>
              {row.predictions
                .filter((item) => item.percentage > 0)
                .map((item) => (
                  <div
                    key={`${row.actualLabel}-${item.predictedLabel}`}
                    title={`Actual ${row.actualLabel} → Predicted ${item.predictedLabel}: ${item.count.toLocaleString()} cases (${formatPercent(item.percentage, 2)})`}
                    style={{
                      ...styles.segment,
                      width: `${item.percentage}%`,
                      background: HEALTH_COLORS[item.predictedLabel] || '#64748b',
                      outline: item.isCorrect
                        ? '2px solid rgba(255,255,255,.92)'
                        : 'none',
                      outlineOffset: '-3px',
                    }}
                  >
                    {item.percentage >= 8
                      ? formatPercent(item.percentage, 1)
                      : ''}
                  </div>
                ))}
            </div>

            <div style={styles.rowFooter}>
              <span style={styles.correctBadge}>
                {row.correctCount.toLocaleString()} of {row.rowTotal.toLocaleString()} correctly classified
              </span>

              {row.strongestMistake?.percentage > 0 && (
                <span style={styles.mistakeText}>
                  Most common confusion: {row.strongestMistake.predictedLabel}{' '}
                  ({formatPercent(row.strongestMistake.percentage, 1)})
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
