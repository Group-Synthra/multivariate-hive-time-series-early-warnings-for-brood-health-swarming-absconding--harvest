import { useCallback, useEffect, useRef, useState } from 'react';
import {
  getAbscondingIotLive,
  getAbscondingSummary,
} from '../services/abscondingApi';

const REFRESH_MINUTES = 10;

export function useAbscondingData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [iotLiveData, setIotLiveData] = useState(null);
  const [iotLiveLoading, setIotLiveLoading] = useState(false);
  const [iotLiveError, setIotLiveError] = useState(null);
  const [lastIotFetchAt, setLastIotFetchAt] = useState(null);
  const [nextIotRefreshAt, setNextIotRefreshAt] = useState(null);
  const mounted = useRef(true);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getAbscondingSummary();
      if (mounted.current) setData(result);
      return result;
    } catch (requestError) {
      if (mounted.current) setError(requestError);
      throw requestError;
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  const refetchIotLive = useCallback(async (force = false) => {
    setIotLiveLoading(true);
    setIotLiveError(null);
    try {
      const result = await getAbscondingIotLive({ force });
      const fetchedAt = new Date();
      if (mounted.current) {
        setIotLiveData(result);
        setLastIotFetchAt(fetchedAt.toISOString());
        setNextIotRefreshAt(
          new Date(fetchedAt.getTime() + REFRESH_MINUTES * 60 * 1000).toISOString(),
        );
      }
      return result;
    } catch (requestError) {
      if (mounted.current) setIotLiveError(requestError?.message || String(requestError));
      throw requestError;
    } finally {
      if (mounted.current) setIotLiveLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    refetch().catch(() => undefined);
    refetchIotLive(false).catch(() => undefined);
    const timer = window.setInterval(
      () => refetchIotLive(false).catch(() => undefined),
      REFRESH_MINUTES * 60 * 1000,
    );
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
  }, [refetch, refetchIotLive]);

  return {
    data,
    loading,
    error,
    refetch,
    iotLiveData,
    iotLiveLoading,
    iotLiveError,
    refetchIotLive,
    dashboardRefreshIntervalMinutes: REFRESH_MINUTES,
    lastIotFetchAt,
    nextIotRefreshAt,
  };
}
