import { useCallback, useEffect, useRef, useState } from 'react';
import { broodHealthApi } from '../services/broodHealthApi';

function useRequest(loader, { enabled = true, initialData = null } = {}) {
  const mountedRef = useRef(false);
  const controllerRef = useRef(null);
  const [state, setState] = useState({ data: initialData, loading: enabled, error: null });

  const execute = useCallback(async () => {
    if (!enabled) return null;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const data = await loader({ signal: controller.signal });
      if (mountedRef.current && !controller.signal.aborted) {
        setState({ data, loading: false, error: null });
      }
      return data;
    } catch (error) {
      if (error.name !== 'AbortError' && mountedRef.current) {
        setState((current) => ({ ...current, loading: false, error }));
      }
      return null;
    }
  }, [enabled, loader]);

  useEffect(() => {
    mountedRef.current = true;
    if (enabled) execute();
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, [enabled, execute]);

  const updateData = useCallback((updater) => {
    setState((current) => ({
      ...current,
      data: typeof updater === 'function' ? updater(current.data) : updater,
    }));
  }, []);

  return { ...state, refetch: execute, setData: updateData };
}

export function useBroodEDA(enabled = true) {
  const loader = useCallback(({ signal }) => broodHealthApi.getEDA({ signal }), []);
  return useRequest(loader, { enabled });
}

export function useBroodTraining(enabled = true) {
  const loader = useCallback(({ signal }) => broodHealthApi.getModel({ signal }), []);
  const resource = useRequest(loader, { enabled });
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState(null);

  const startTraining = useCallback(async ({ fastMode = false, horizonHours = 6 } = {}) => {
    setStarting(true);
    setStartError(null);
    try {
      const status = await broodHealthApi.startTraining({ fastMode, horizonHours });
      resource.setData((current) => ({ ...(current || {}), training_status: status }));
      return status;
    } catch (error) {
      setStartError(error);
      return null;
    } finally {
      setStarting(false);
    }
  }, [resource.setData]);

  const running = Boolean(resource.data?.training_status?.running);
  useEffect(() => {
    if (!enabled || !running) return undefined;
    const timer = window.setInterval(async () => {
      const status = await broodHealthApi.getTrainingStatus().catch(() => null);
      if (!status) return;
      resource.setData((current) => ({ ...(current || {}), training_status: status }));
      if (!status.running) resource.refetch();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [enabled, running, resource.refetch, resource.setData]);

  return { ...resource, startTraining, starting, startError };
}

export function useBroodIoT(enabled = true) {
  const healthLoader = useCallback(({ signal }) => broodHealthApi.getIoTHealth({ signal }), []);
  const devicesLoader = useCallback(({ signal }) => broodHealthApi.getDevices({ signal }), []);
  const health = useRequest(healthLoader, { enabled });
  const devices = useRequest(devicesLoader, { enabled });
  const [selectedDevice, setSelectedDevice] = useState('');
  const [prediction, setPrediction] = useState({ data: null, loading: false, error: null });
  const predictionController = useRef(null);

  useEffect(() => {
    if (!selectedDevice && devices.data?.devices?.length) {
      setSelectedDevice(devices.data.devices[0].device_id);
    }
  }, [devices.data, selectedDevice]);

  const loadPrediction = useCallback(async (deviceId = selectedDevice) => {
    if (!deviceId) return null;
    predictionController.current?.abort();
    const controller = new AbortController();
    predictionController.current = controller;
    setPrediction((current) => ({ ...current, loading: true, error: null }));
    try {
      const data = await broodHealthApi.getDevicePrediction(deviceId, { signal: controller.signal });
      if (!controller.signal.aborted) setPrediction({ data, loading: false, error: null });
      return data;
    } catch (error) {
      if (error.name !== 'AbortError') setPrediction((current) => ({ ...current, loading: false, error }));
      return null;
    }
  }, [selectedDevice]);

  useEffect(() => {
    if (enabled && selectedDevice) loadPrediction(selectedDevice);
    return () => predictionController.current?.abort();
  }, [enabled, loadPrediction, selectedDevice]);

  const refresh = useCallback(async () => {
    await Promise.all([health.refetch(), devices.refetch()]);
    if (selectedDevice) await loadPrediction(selectedDevice);
  }, [devices.refetch, health.refetch, loadPrediction, selectedDevice]);

  return {
    health,
    devices,
    selectedDevice,
    setSelectedDevice,
    prediction,
    loadPrediction,
    refresh,
  };
}
