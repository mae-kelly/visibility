import useSWR from 'swr';
import { apiClient } from '../lib/api';

export function useAssets(page = 1, size = 50) {
  const { data, error, isLoading } = useSWR(
    `/api/v1/assets?page=${page}&size=${size}`,
    () => apiClient.getAssets(page, size).then(res => res.data)
  );

  return {
    assets: data?.assets || [],
    total: data?.total || 0,
    isLoading,
    error
  };
}

export function useDashboard() {
  const { data, error, isLoading } = useSWR(
    '/api/v1/visibility/dashboard',
    () => apiClient.getDashboard().then(res => res.data),
    { refreshInterval: 30000 } // Refresh every 30 seconds
  );

  return {
    metrics: data,
    isLoading,
    error
  };
}

export function useGaps() {
  const { data, error, isLoading } = useSWR(
    '/api/v1/gaps',
    () => apiClient.getGaps().then(res => res.data)
  );

  return {
    gaps: data?.gaps || [],
    total: data?.total || 0,
    isLoading,
    error
  };
}

export function useAIStatus() {
  const { data, error, isLoading } = useSWR(
    '/api/v1/ai/status',
    () => apiClient.getAIStatus().then(res => res.data),
    { refreshInterval: 60000 } // Refresh every minute
  );

  return {
    aiStatus: data,
    isLoading,
    error
  };
}
