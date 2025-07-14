import axios from 'axios';
import { Asset, VisibilityMetrics, Gap, Correlation, AIStatus } from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for development
api.interceptors.request.use((config) => {
  if (process.env.NODE_ENV === 'development') {
    config.headers['x-development-mode'] = 'true';
  }
  return config;
});

export const apiClient = {
  // Health check
  health: () => api.get('/api/health'),

  // Assets
  getAssets: (page = 1, size = 50) => 
    api.get<{assets: Asset[], total: number}>(`/api/v1/assets?page=${page}&size=${size}`),
  
  getAsset: (id: string) => 
    api.get<Asset>(`/api/v1/assets/${id}`),

  // Visibility
  getDashboard: () => 
    api.get<VisibilityMetrics>(`/api/v1/visibility/dashboard`),

  // Gaps
  getGaps: () => 
    api.get<{gaps: Gap[], total: number}>(`/api/v1/gaps`),

  // Correlations
  getCorrelations: () => 
    api.get<{correlations: Correlation[], total: number}>(`/api/v1/correlations`),

  // AI Models
  getAIStatus: () => 
    api.get<AIStatus>(`/api/v1/ai/status`),
};
