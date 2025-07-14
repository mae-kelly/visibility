export interface Asset {
  id: string;
  hostname: string;
  ip: string;
  visibility: number;
  environment: string;
  asset_class?: string;
  last_seen: string;
  risk_score?: number;
  source_systems?: string[];
}

export interface VisibilityMetrics {
  total_assets: number;
  visibility_percentage: number;
  high_visibility: number;
  gaps: number;
  status: 'excellent' | 'good' | 'fair' | 'poor';
}

export interface Gap {
  asset_id: string;
  hostname: string;
  visibility_score: number;
  gap_severity: 'High' | 'Medium' | 'Low';
  missing_sources: string[];
}

export interface Correlation {
  correlation_id: string;
  asset1_id: string;
  asset2_id: string;
  similarity_score: number;
  correlation_type: string;
  created_at: string;
}

export interface AIModel {
  status: 'active' | 'training' | 'error';
  accuracy?: number;
  predictions_today?: number;
  last_trained?: string;
}

export interface AIStatus {
  models: Record<string, AIModel>;
  overall_health: 'good' | 'warning' | 'error';
  total_predictions_today: number;
}
