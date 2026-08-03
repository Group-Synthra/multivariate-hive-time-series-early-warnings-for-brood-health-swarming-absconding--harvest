import { useCallback, useEffect, useRef, useState } from 'react';

export function useAsyncResource(loader) {
  const controllerRef = useRef(null);
  const mountedRef = useRef(false);
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });

  const execute = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setState((current) => ({ ...current, loading: true, error: null }));

    try {
      const data = await loader({ signal: controller.signal });
      if (mountedRef.current && !controller.signal.aborted) {
        setState({ data, loading: false, error: null });
      }
    } catch (error) {
      if (error.name !== 'AbortError' && mountedRef.current) {
        setState((current) => ({ ...current, loading: false, error }));
      }
    }
  }, [loader]);

  useEffect(() => {
    mountedRef.current = true;
    execute();

    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, [execute]);

  return {
    ...state,
    refetch: execute,
  };
}
