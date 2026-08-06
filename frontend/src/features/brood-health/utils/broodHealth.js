export const SENSOR_META = {
  temperature_c: { label: 'Internal temperature', shortLabel: 'Temperature', unit: '°C' },
  humidity_pct: { label: 'Internal humidity', shortLabel: 'Humidity', unit: '% RH' },
  co2_ppm: { label: 'Internal CO₂', shortLabel: 'CO₂', unit: 'ppm' },
  weight_kg: { label: 'Hive weight', shortLabel: 'Weight', unit: 'kg' },
};

export const HEALTH_LEVEL_META = {
  Excellent: { label: 'Excellent', className: 'excellent', range: '80–100' },
  Good: { label: 'Good', className: 'good', range: '60–79.99' },
  Poor: { label: 'Poor', className: 'poor', range: '40–59.99' },
  Critical: { label: 'Critical', className: 'critical', range: '1–39.99' },
};

export const STABILITY_LEVEL_META = {
  High: { className: 'high', range: '70–100' },
  Moderate: { className: 'moderate', range: '40–69.99' },
  Low: { className: 'low', range: '0–39.99' },
};

export function asPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function percentValue(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}%`;
}

export function numberValue(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

export function signedNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const numeric = Number(value);
  return `${numeric > 0 ? '+' : ''}${numeric.toFixed(digits)}`;
}

export function timestampValue(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

export function healthClass(level) {
  return HEALTH_LEVEL_META[level]?.className || 'unknown';
}

export function stabilityClass(level) {
  return STABILITY_LEVEL_META[level]?.className || 'unknown';
}

export function freshnessLabel(minutes) {
  if (minutes === null || minutes === undefined || Number.isNaN(Number(minutes))) return 'Unknown';
  if (minutes < 20) return 'Fresh';
  if (minutes < 60) return 'Delayed';
  return 'Stale';
}
