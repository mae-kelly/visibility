'use client';

import { useState, useEffect } from 'react';
import { 
  Shield, 
  Database, 
  Eye, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Server,
  Activity,
  BarChart3,
  PieChart
} from 'lucide-react';

interface RealData {
  metrics: any;
  assets: any[];
  gaps: any[];
  gapsBySource: any;
  rawSources: any;
}

export default function RealVisibilityDashboard() {
  const [data, setData] = useState<RealData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRealData = async () => {
      try {
        setLoading(true);
        
        const [metricsRes, assetsRes, gapsRes, gapsBySourceRes, rawRes] = await Promise.all([
          fetch('http://localhost:8000/api/v1/coverage/metrics'),
          fetch('http://localhost:8000/api/v1/assets/all'),
          fetch('http://localhost:8000/api/v1/gaps/list'),
          fetch('http://localhost:8000/api/v1/gaps/by-source'),
          fetch('http://localhost:8000/api/v1/sources/raw')
        ]);

        const [metrics, assets, gaps, gapsBySource, rawSources] = await Promise.all([
          metricsRes.json(),
          assetsRes.json(),
          gapsRes.json(),
          gapsBySourceRes.json(),
          rawRes.json()
        ]);

        setData({ metrics, assets: assets.assets || [], gaps: gaps.gaps || [], gapsBySource, rawSources });
        setError(null);
      } catch (err) {
        console.error('Real data fetch error:', err);
        setError('Failed to connect to your actual database');
      } finally {
        setLoading(false);
      }
    };

    fetchRealData();
    const interval = setInterval(fetchRealData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <LoadingScreen />;
  }

  if (error) {
    return <ErrorScreen error={error} />;
  }

  if (!data) {
    return <div>No data available</div>;
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="cinematic-bg"></div>
      
      {/* Header */}
      <div className="border-b border-gray-700 bg-black/80 backdrop-blur-sm relative z-10">
        <div className="px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-blue-600/20 border border-blue-500 rounded-lg">
                <Database size={32} className="text-blue-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">AO1 Real Visibility Platform</h1>
                <p className="text-gray-400">Live data from your Chronicle • Splunk • CrowdStrike • CMDB</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-400">{data.metrics.average_visibility}%</div>
              <div className="text-sm text-gray-400">Current Visibility</div>
            </div>
          </div>
        </div>
      </div>

      <div className="relative z-10 p-8 space-y-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <MetricCard
            title="Total Assets Found"
            value={data.metrics.total_assets}
            subtitle="Across all sources"
            icon={Server}
            color="blue"
          />
          <MetricCard
            title="Excellent Coverage"
            value={data.metrics.excellent_coverage}
            subtitle="4/4 sources monitoring"
            icon={CheckCircle}
            color="green"
          />
          <MetricCard
            title="Poor Coverage"
            value={data.metrics.poor_coverage}
            subtitle="1 source only"
            icon={AlertTriangle}
            color="red"
          />
          <MetricCard
            title="Action Items"
            value={data.gaps.length}
            subtitle="Gaps to fix"
            icon={Activity}
            color="yellow"
          />
        </div>

        {/* Source Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <SourceBreakdown rawSources={data.rawSources} />
          <CoverageChart metrics={data.metrics} />
        </div>

        {/* Gap Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <GapsBySource gapsBySource={data.gapsBySource} />
          <ActionableGaps gaps={data.gaps.slice(0, 10)} />
        </div>

        {/* Asset Details */}
        <AssetTable assets={data.assets.slice(0, 20)} />
      </div>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
      <div className="text-center">
        <Database size={64} className="mx-auto mb-4 text-blue-400 animate-pulse" />
        <h2 className="text-2xl font-bold mb-2">Querying Your Database</h2>
        <p className="text-gray-400">Analyzing Chronicle • Splunk • CrowdStrike • CMDB data...</p>
      </div>
    </div>
  );
}

function ErrorScreen({ error }: { error: string }) {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
      <div className="text-center max-w-md">
        <XCircle size={64} className="mx-auto mb-4 text-red-400" />
        <h2 className="text-2xl font-bold mb-4">Database Connection Failed</h2>
        <p className="text-gray-400 mb-4">{error}</p>
        <div className="text-left bg-gray-800 p-4 rounded text-sm">
          <p className="text-yellow-400 mb-2">To fix:</p>
          <p>1. Update DB_PATH in backend/app/main.py</p>
          <p>2. Ensure your DuckDB file is accessible</p>
          <p>3. Verify the database schema matches</p>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, subtitle, icon: Icon, color }: any) {
  const colorMap = {
    blue: 'border-blue-500 text-blue-400',
    green: 'border-green-500 text-green-400',
    red: 'border-red-500 text-red-400',
    yellow: 'border-yellow-500 text-yellow-400'
  };

  return (
    <div className={`bg-black/60 backdrop-blur border ${colorMap[color]} rounded-lg p-6`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm mb-1">{title}</p>
          <p className="text-3xl font-bold">{value}</p>
          <p className="text-gray-500 text-sm">{subtitle}</p>
        </div>
        <Icon size={32} className={colorMap[color].split(' ')[1]} />
      </div>
    </div>
  );
}

function SourceBreakdown({ rawSources }: any) {
  return (
    <div className="bg-black/60 backdrop-blur border border-gray-700 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4 flex items-center">
        <Database size={20} className="mr-2" />
        Data Sources
      </h3>
      <div className="space-y-4">
        {Object.entries(rawSources).map(([source, data]: [string, any]) => (
          <div key={source} className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
            <div>
              <div className="font-semibold capitalize">{source}</div>
              <div className="text-sm text-gray-400">Active data source</div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-400">{data.count}</div>
              <div className="text-sm text-gray-400">records</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CoverageChart({ metrics }: any) {
  const chartData = [
    { name: 'Excellent', value: metrics.excellent_coverage, color: '#22c55e' },
    { name: 'Good', value: metrics.good_coverage, color: '#3b82f6' },
    { name: 'Fair', value: metrics.fair_coverage, color: '#f59e0b' },
    { name: 'Poor', value: metrics.poor_coverage, color: '#ef4444' },
  ];

  return (
    <div className="bg-black/60 backdrop-blur border border-gray-700 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Coverage Distribution</h3>
      <div className="space-y-3">
        {chartData.map((item, index) => (
          <div key={index} className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: item.color }}></div>
              <span>{item.name}</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-32 bg-gray-800 rounded-full h-2">
                <div 
                  className="h-2 rounded-full" 
                  style={{ 
                    width: `${(item.value / metrics.total_assets) * 100}%`,
                    backgroundColor: item.color 
                  }}
                ></div>
              </div>
              <span className="w-8 text-right">{item.value}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GapsBySource({ gapsBySource }: any) {
  return (
    <div className="bg-black/60 backdrop-blur border border-gray-700 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Gaps by Source</h3>
      <div className="space-y-3">
        {Object.entries(gapsBySource).map(([source, hostnames]: [string, any]) => (
          <div key={source} className="p-3 bg-gray-800/50 rounded">
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold capitalize">{source.replace('needs_', '')}</span>
              <span className="text-red-400 font-bold">{hostnames.length} missing</span>
            </div>
            <div className="text-sm text-gray-400">
              {hostnames.slice(0, 3).join(', ')}{hostnames.length > 3 ? ` +${hostnames.length - 3} more` : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ActionableGaps({ gaps }: any) {
  return (
    <div className="bg-black/60 backdrop-blur border border-gray-700 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Action Items</h3>
      <div className="space-y-3">
        {gaps.map((gap: any, index: number) => (
          <div key={index} className="p-3 bg-gray-800/50 rounded">
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold">{gap.hostname}</span>
              <span className={`px-2 py-1 rounded text-xs ${
                gap.priority === 'HIGH' ? 'bg-red-600' : 'bg-yellow-600'
              }`}>
                {gap.priority}
              </span>
            </div>
            <div className="text-sm text-gray-400">
              Missing: {gap.missing_sources.join(', ')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AssetTable({ assets }: any) {
  return (
    <div className="bg-black/60 backdrop-blur border border-gray-700 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Asset Details</h3>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left p-3">Asset</th>
              <th className="text-left p-3">Sources</th>
              <th className="text-left p-3">Coverage</th>
              <th className="text-left p-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {assets.map((asset: any, index: number) => (
              <tr key={index} className="border-b border-gray-800">
                <td className="p-3">
                  <div className="font-semibold">{asset.hostname}</div>
                  <div className="text-sm text-gray-400">{asset.asset_id}</div>
                </td>
                <td className="p-3">
                  <div className="flex flex-wrap gap-1">
                    {asset.source_systems.map((source: string) => (
                      <span key={source} className="px-2 py-1 bg-blue-600/20 text-blue-400 rounded text-xs">
                        {source}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="p-3">
                  <div className="text-lg font-bold">{asset.visibility_score}%</div>
                </td>
                <td className="p-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    asset.coverage_status === 'EXCELLENT' ? 'bg-green-600' :
                    asset.coverage_status === 'GOOD' ? 'bg-blue-600' :
                    asset.coverage_status === 'FAIR' ? 'bg-yellow-600' : 'bg-red-600'
                  }`}>
                    {asset.coverage_status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
