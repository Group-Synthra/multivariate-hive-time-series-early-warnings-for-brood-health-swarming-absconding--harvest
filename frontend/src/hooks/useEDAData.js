import { useCallback } from 'react';
import { fetchCommonEDA } from '../services/commonApi';
import { useAsyncResource } from './useAsyncResource';

/**
 * Fetches aggregated common EDA data from GET /api/eda.
 * The frontend should not download all 311,044 raw rows.
 */
export function useEDAData() {
  const loader = useCallback(({ signal }) => fetchCommonEDA({ signal }), []);
  const { data, loading, error, refetch } = useAsyncResource(loader);

  return {
    edaData: data,
    loading,
    error,
    refetch,
  };
}
