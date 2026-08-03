import { apiGet } from './apiClient';
import { normalizeEDAResponse } from '../utils/dataContracts';

export async function fetchCommonEDA({ signal } = {}) {
  const response = await apiGet('/api/eda', { signal });
  return normalizeEDAResponse(response);
}

export async function fetchBackendHealth({ signal } = {}) {
  return apiGet('/api/health', { signal });
}
