export function formatNumber(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits,
  });
}

export function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString();
}

export function percentage(value) {
  return value === null || value === undefined ? '—' : `${formatNumber(value, 2)}%`;
}
