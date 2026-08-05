const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

async function apiRequest(path, { method = 'GET', body, signal } = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers: {
      Accept: 'application/json',
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(
      payload?.error || payload?.message || `Request failed with HTTP ${response.status}`,
      response.status,
      payload,
    );
  }
  return payload;
}

export function apiGet(path, { signal } = {}) {
  return apiRequest(path, { method: 'GET', signal });
}

export function apiPost(path, body = {}, { signal } = {}) {
  return apiRequest(path, { method: 'POST', body, signal });
}

export function apiAssetUrl(path) {
  return `${API_BASE_URL}${path}`;
}
