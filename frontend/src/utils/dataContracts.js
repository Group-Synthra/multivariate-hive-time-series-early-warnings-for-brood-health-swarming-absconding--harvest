const SENSOR_ALIASES = {
  temperature: ['temperature', 'temp', 'temperature_c'],
  humidity: ['humidity', 'humidity_pct'],
  co2: ['co2', 'co2_ppm'],
  weight: ['weight', 'weight_kg'],
};

function firstDefined(object, keys, fallback = undefined) {
  for (const key of keys) {
    if (object?.[key] !== undefined && object?.[key] !== null) {
      return object[key];
    }
  }
  return fallback;
}

function normalizeSensorStatistics(raw = {}) {
  return Object.fromEntries(
    Object.entries(SENSOR_ALIASES).map(([canonical, aliases]) => {
      const value = firstDefined(raw, aliases, {});
      return [canonical, value || {}];
    }),
  );
}

function normalizeTargetBalance(raw = {}) {
  if (Array.isArray(raw)) return raw;

  return Object.entries(raw).map(([target, value]) => {
    if (typeof value === 'number') {
      return { target, positive: value, negative: 0 };
    }

    return {
      target,
      display_name: firstDefined(value, ['display_name', 'name'], target),
      positive: firstDefined(value, ['positive', 'ones', 'count_1'], 0),
      negative: firstDefined(value, ['negative', 'zeros', 'count_0'], 0),
      positive_percentage: firstDefined(
        value,
        ['positive_percentage', 'percentage', 'rate'],
        null,
      ),
      positive_per_10000: firstDefined(value, ['positive_per_10000'], null),
    };
  });
}

function normalizePatternRows(rows = []) {
  return rows.map((item) => ({
    ...item,
    temperature: firstDefined(item, ['temperature', 'temp', 'temperature_c']),
    humidity: firstDefined(item, ['humidity', 'humidity_pct']),
    co2: firstDefined(item, ['co2', 'co2_ppm']),
    weight: firstDefined(item, ['weight', 'weight_kg']),
  }));
}

export function normalizeEDAResponse(raw = {}) {
  return {
    summary: {
      total_records: firstDefined(raw.summary, ['total_records', 'rows'], 0),
      total_hives: firstDefined(raw.summary, ['total_hives', 'hives'], 0),
      analysis_start: firstDefined(raw.summary, ['analysis_start', 'start']),
      analysis_end: firstDefined(raw.summary, ['analysis_end', 'end']),
      duration_days: firstDefined(raw.summary, ['duration_days'], null),
      sampling_frequency: firstDefined(
        raw.summary,
        ['sampling_frequency', 'frequency'],
        '1 hour',
      ),
    },
    sensor_statistics: normalizeSensorStatistics(raw.sensor_statistics),
    outlier_analysis: raw.outlier_analysis || {},
    hive_stats: (raw.hive_stats || raw.hive_coverage || []).map((item) => ({
      ...item,
      hive: firstDefined(item, ['hive', 'hive_id']),
      records: firstDefined(item, ['records', 'row_count', 'count'], 0),
      coverage_percentage: firstDefined(item, ['coverage_percentage', 'coverage'], null),
    })),
    hourly_patterns: normalizePatternRows(raw.hourly_patterns || []),
    weekday_patterns: normalizePatternRows(raw.weekday_patterns || []),
    monthly_patterns: normalizePatternRows(raw.monthly_patterns || []),
    sensor_histograms: raw.sensor_histograms || {},
    target_balance: normalizeTargetBalance(raw.target_balance || {}),
    target_by_hive: raw.target_by_hive || [],
    monthly_target_counts: raw.monthly_target_counts || [],
    target_cooccurrence: raw.target_cooccurrence || [],
    target_sensor_effects: raw.target_sensor_effects || {},
    relationship_sample: normalizePatternRows(raw.relationship_sample || []),
    correlation: raw.correlation || raw.sensor_correlation || {},
    generated_images: raw.generated_images || {},
    data_quality: raw.data_quality || {},
  };
}
