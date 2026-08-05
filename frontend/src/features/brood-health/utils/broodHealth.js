export const SENSOR_META = {
  temperature_c: { label: 'Temperature', shortLabel: 'Temperature', unit: '°C' },
  humidity_pct: { label: 'Humidity', shortLabel: 'Humidity', unit: '%' },
  co2_ppm: { label: 'CO₂ concentration', shortLabel: 'CO₂', unit: 'ppm' },
  weight_kg: { label: 'Hive weight', shortLabel: 'Weight', unit: 'kg' },
};

export const HEALTH_LEVEL_META = {
  Excellent: { label: 'Excellent', className: 'excellent' },
  Good: { label: 'Good', className: 'good' },
  Poor: { label: 'Poor', className: 'poor' },
  Critical: { label: 'Critical', className: 'critical' },
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
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function timestampValue(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

export function healthClass(level) {
  return HEALTH_LEVEL_META[level]?.className || 'unknown';
}

export function freshnessLabel(minutes) {
  if (minutes === null || minutes === undefined || Number.isNaN(Number(minutes))) return 'Unknown';
  if (minutes < 20) return 'Fresh';
  if (minutes < 90) return 'Delayed';
  return 'Stale';
}
