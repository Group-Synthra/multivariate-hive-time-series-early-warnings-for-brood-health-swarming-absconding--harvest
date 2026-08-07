import { apiGet, apiPost } from './apiClient';

export function getAbscondingSummary(options = {}) {
  return apiGet('/api/absconding/summary', options);
}

export function getAbscondingHive(hiveId, options = {}) {
  return apiGet(`/api/absconding/hives/${encodeURIComponent(hiveId)}`, options);
}

export function getAbscondingIotLive({ force = false, signal } = {}) {
  const query = force ? '?force=true' : '';
  return apiGet(`/api/absconding/iot/live${query}`, { signal });
}

export function getAbscondingIotMonitorStatus(options = {}) {
  return apiGet('/api/absconding/iot/monitor/status', options);
}

export function runAbscondingIotMonitor(options = {}) {
  return apiPost('/api/absconding/iot/monitor/run-now', {}, options);
}

export function startAbscondingIotMonitor(options = {}) {
  return apiPost('/api/absconding/iot/monitor/start', {}, options);
}

export function stopAbscondingIotMonitor(options = {}) {
  return apiPost('/api/absconding/iot/monitor/stop', {}, options);
}
