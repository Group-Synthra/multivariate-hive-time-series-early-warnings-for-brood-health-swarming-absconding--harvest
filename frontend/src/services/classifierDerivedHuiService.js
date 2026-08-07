const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000"
).replace(/\/$/, "");

const HISTORICAL_DASHBOARD_URL =
  "/data/harvesting-research/classifier-derived-hui-viva-dashboard.json";

export class LiveHuiApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "LiveHuiApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function readJsonResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new LiveHuiApiError(
      payload?.message ??
        payload?.error ??
        `Request failed with HTTP ${response.status}.`,
      response.status,
      payload,
    );
  }

  return payload;
}

async function requestApi(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
    ...options,
  });

  return readJsonResponse(response);
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function extractHiveDiagnostics(payload) {
  const candidates = [
    payload?.hive_diagnostics,
    payload?.diagnostics,
    payload?.details?.hive_diagnostics,
    payload?.error?.hive_diagnostics,
    payload?.data?.hive_diagnostics,
  ];

  const arrayCandidate = candidates.find(Array.isArray);
  if (arrayCandidate) {
    return arrayCandidate;
  }

  const singleCandidate =
    payload?.diagnostic ??
    payload?.details?.diagnostic ??
    payload?.error?.diagnostic ??
    null;

  return singleCandidate ? [singleCandidate] : [];
}

function extractAvailableHives(payload, diagnostics) {
  return [
    ...new Set([
      ...asArray(payload?.available_hives),
      ...diagnostics
        .map((item) => item?.hive_id)
        .filter(Boolean),
    ]),
  ].sort();
}

function normalizeCollectingHistory(error) {
  const payload = error.payload ?? {};
  const diagnostics = extractHiveDiagnostics(payload);

  return {
    ...payload,
    status: "collecting_history",
    prediction_ready: false,
    message: error.message,
    http_status: error.status,
    available_hives: extractAvailableHives(payload, diagnostics),
    latest_by_hive: asArray(payload.latest_by_hive),
    hive_diagnostics: diagnostics,
  };
}

export async function loadClassifierDerivedHuiDashboard() {
  const response = await fetch(HISTORICAL_DASHBOARD_URL, {
    headers: { Accept: "application/json" },
  });
  return readJsonResponse(response);
}

export async function loadLiveHuiPrediction({
  hiveId = "",
  refresh = false,
} = {}) {
  const query = new URLSearchParams();
  if (hiveId) query.set("hive_id", hiveId);
  if (refresh) query.set("refresh", "true");
  const suffix = query.toString() ? `?${query.toString()}` : "";

  try {
    return await requestApi(`/api/harvesting/live-hui${suffix}`);
  } catch (error) {
    if (error instanceof LiveHuiApiError && error.status === 422) {
      return normalizeCollectingHistory(error);
    }
    throw error;
  }
}

export async function refreshLiveHuiPrediction(hiveId = "") {
  try {
    return await requestApi("/api/harvesting/live-hui/refresh", {
      method: "POST",
      body: JSON.stringify(hiveId ? { hive_id: hiveId } : {}),
    });
  } catch (error) {
    if (error instanceof LiveHuiApiError && error.status === 422) {
      return normalizeCollectingHistory(error);
    }
    throw error;
  }
}

export async function loadLiveHuiStatus() {
  return requestApi("/api/harvesting/live-hui/status");
}

export async function loadLiveSensorSnapshot(hiveId = "") {
  const suffix = hiveId
    ? `?hive_id=${encodeURIComponent(hiveId)}`
    : "";
  return requestApi(`/api/harvesting/live-sensors${suffix}`);
}

export const loadLiveHuiDashboard = loadLiveHuiPrediction;
export const refreshLiveHuiDashboard = refreshLiveHuiPrediction;
