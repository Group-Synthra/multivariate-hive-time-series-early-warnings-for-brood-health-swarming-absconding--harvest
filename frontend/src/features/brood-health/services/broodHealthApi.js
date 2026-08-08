import { apiAssetUrl, apiGet, apiPost } from '../../../services/apiClient';

const ROOT = '/api/brood-health';

export const broodHealthApi = {
  getEDA: ({ force = false, signal } = {}) => apiGet(`${ROOT}/eda${force ? '?force=true' : ''}`, { signal }),
  getModel: ({ signal } = {}) => apiGet(`${ROOT}/model`, { signal }),
  startTraining: ({ horizonHours = 6, fastMode = false } = {}) =>
    apiPost(`${ROOT}/train`, { horizon_hours: horizonHours, fast_mode: fastMode }),
  getTrainingStatus: ({ signal } = {}) => apiGet(`${ROOT}/train/status`, { signal }),
  getIoTHealth: ({ signal } = {}) => apiGet(`${ROOT}/iot/health`, { signal }),
  getDevices: ({ signal } = {}) => apiGet(`${ROOT}/iot/devices`, { signal }),
  getDevicePrediction: (deviceId, { lookbackHours = 168, signal } = {}) =>
    apiGet(
      `${ROOT}/iot/predict?device_id=${encodeURIComponent(deviceId)}&lookback_hours=${lookbackHours}`,
      { signal },
    ),
  getValidationLog: (deviceId, { limit = 100, signal } = {}) =>
    apiGet(
      `${ROOT}/iot/validation-log?device_id=${encodeURIComponent(deviceId)}&limit=${limit}`,
      { signal },
    ),
  validationLogDownloadUrl: (deviceId = '') =>
    apiAssetUrl(
      `${ROOT}/iot/validation-log/download${
        deviceId ? `?device_id=${encodeURIComponent(deviceId)}` : ''
      }`,
    ),
  predictManual: (readings, { signal } = {}) => apiPost(`${ROOT}/predict/manual`, { readings }, { signal }),
  reportUrl: (filename) => apiAssetUrl(`${ROOT}/reports/${filename}`),
};
