'use client';

import { Shield, Server, AlertTriangle, Brain, Target, Activity } from 'lucide-react';
import MetricsCard from './MetricsCard';
import VisibilityChart from '../charts/VisibilityChart';
import { useDashboard, useAIStatus } from '../../hooks/useApi';

export default function Dashboard() {
  const { metrics, isLoading: dashboardLoading } = useDashboard();
  const { aiStatus, isLoading: aiLoading } = useAIStatus();

  const getVisibilityStatus = (percentage: number) => {
    if (percentage >= 90) return { text: 'Excellent', color: 'green' as const };
    if (percentage >= 75) return { text: 'Good', color: 'blue' as const };
    if (percentage >= 50) return { text: 'Fair', color: 'yellow' as const };
    return { text: 'Poor', color: 'red' as const };
  };

  const visibilityStatus = metrics ? getVisibilityStatus(metrics.visibility_percentage) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Security Visibility Dashboard</h1>
            <p className="text-gray-600 mt-1">Real-time enterprise asset visibility and correlation</p>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Live</span>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricsCard
          title="Total Assets"
          value={metrics?.total_assets || 0}
          subtitle="Across all systems"
          icon={Server}
          color="blue"
          loading={dashboardLoading}
        />
        
        <MetricsCard
          title="Visibility Score"
          value={metrics ? `${metrics.visibility_percentage.toFixed(1)}%` : '0%'}
          subtitle={visibilityStatus?.text}
          icon={Target}
          color={visibilityStatus?.color}
          trend={{ value: 5.2, isPositive: true }}
          loading={dashboardLoading}
        />
        
        <MetricsCard
          title="High Visibility"
          value={metrics?.high_visibility || 0}
          subtitle="Well-monitored assets"
          icon={Shield}
          color="green"
          loading={dashboardLoading}
        />
        
        <MetricsCard
          title="Coverage Gaps"
          value={metrics?.gaps || 0}
          subtitle="Require attention"
          icon={AlertTriangle}
          color="red"
          loading={dashboardLoading}
        />
      </div>

      {/* AI Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricsCard
          title="AI Models Active"
          value={aiStatus ? Object.keys(aiStatus.models).length : 0}
          subtitle="Running models"
          icon={Brain}
          color="blue"
          loading={aiLoading}
        />
        
        <MetricsCard
          title="Daily Predictions"
          value={aiStatus?.total_predictions_today || 0}
          subtitle="AI predictions today"
          icon={Activity}
          color="green"
          loading={aiLoading}
        />
        
        <MetricsCard
          title="System Health"
          value={aiStatus?.overall_health || 'Unknown'}
          subtitle="AI system status"
          icon={Shield}
          color={aiStatus?.overall_health === 'good' ? 'green' : 'yellow'}
          loading={aiLoading}
        />
        
        <MetricsCard
          title="Auto Discovery"
          value="Active"
          subtitle="Continuous scanning"
          icon={Target}
          color="green"
          loading={false}
        />
      </div>

      {/* Charts */}
      {metrics && (
        <VisibilityChart data={metrics} />
      )}
    </div>
  );
}
