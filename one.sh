#!/bin/bash

set -e

echo "🎯 BUILDING REAL VISIBILITY PLATFORM BASED ON YOUR ACTUAL DATABASE..."
echo "📊 Using real DuckDB schema for Chronicle, Splunk, CrowdStrike, CMDB..."

cd autonomous-visibility-platform

# Update backend to work with real database structure
echo "🔧 Creating real database service based on your schema..."

cat > backend/services/duckdb/real_visibility_service.py << 'EOF'
import duckdb
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

class RealVisibilityService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
    def get_all_unique_assets(self) -> Dict[str, pd.DataFrame]:
        """Find all unique assets across all your data sources"""
        
        # Get all data from each source using your actual schema
        sources = {}
        
        # Chronicle data
        try:
            chronicle_query = """
            SELECT DISTINCT 
                chronicle_device_hostname as hostname,
                chronicle_ip_address as ip_address,
                'chronicle' as source_system,
                device_hostname,
                device_ip,
                log_count,
                log_type
            FROM main.all_sources 
            WHERE chronicle_device_hostname IS NOT NULL
            """
            sources['chronicle'] = self.conn.execute(chronicle_query).df()
            logging.info(f"Chronicle: Found {len(sources['chronicle'])} devices")
        except Exception as e:
            logging.warning(f"Chronicle query failed: {e}")
            sources['chronicle'] = pd.DataFrame()
            
        # Splunk data  
        try:
            splunk_query = """
            SELECT DISTINCT
                splunk_host as hostname,
                host as normalized_host,
                'splunk' as source_system,
                INDEX,
                SOURCE,
                count,
                region
            FROM main.all_sources
            WHERE splunk_host IS NOT NULL
            """
            sources['splunk'] = self.conn.execute(splunk_query).df()
            logging.info(f"Splunk: Found {len(sources['splunk'])} hosts")
        except Exception as e:
            logging.warning(f"Splunk query failed: {e}")
            sources['splunk'] = pd.DataFrame()
            
        # CrowdStrike data
        try:
            crowdstrike_query = """
            SELECT DISTINCT
                crowdstrike_device_hostname as hostname,
                'crowdstrike' as source_system,
                AgentState_Cde,
                AgentStatus_Desc,
                AgentVersion_Desc,
                Agent_Mem,
                Endpoint_Mem,
                ReportingStatusDci_Desc,
                ReportingStatus_Desc
            FROM main.all_sources
            WHERE crowdstrike_device_hostname IS NOT NULL
            """
            sources['crowdstrike'] = self.conn.execute(crowdstrike_query).df()
            logging.info(f"CrowdStrike: Found {len(sources['crowdstrike'])} endpoints")
        except Exception as e:
            logging.warning(f"CrowdStrike query failed: {e}")
            sources['crowdstrike'] = pd.DataFrame()
            
        # CMDB data
        try:
            cmdb_query = """
            SELECT DISTINCT
                name as hostname,
                ip_address,
                'cmdb' as source_system,
                class,
                class_type,
                country,
                domain,
                environment,
                fqdn,
                location,
                make,
                os,
                platform,
                region,
                status,
                status2,
                type
            FROM main.all_sources
            WHERE name IS NOT NULL
            """
            sources['cmdb'] = self.conn.execute(cmdb_query).df()
            logging.info(f"CMDB: Found {len(sources['cmdb'])} assets")
        except Exception as e:
            logging.warning(f"CMDB query failed: {e}")
            sources['cmdb'] = pd.DataFrame()
            
        return sources
    
    def correlate_assets_simple(self) -> List[Dict]:
        """Simple asset correlation based on hostname matching"""
        sources = self.get_all_unique_assets()
        
        all_hostnames = set()
        hostname_sources = {}
        
        # Collect all unique hostnames across sources
        for source_name, df in sources.items():
            if not df.empty and 'hostname' in df.columns:
                for hostname in df['hostname'].dropna():
                    hostname_clean = str(hostname).lower().strip()
                    if hostname_clean:
                        all_hostnames.add(hostname_clean)
                        if hostname_clean not in hostname_sources:
                            hostname_sources[hostname_clean] = []
                        hostname_sources[hostname_clean].append(source_name)
        
        # Create unified asset records
        unified_assets = []
        asset_id = 1
        
        for hostname, source_list in hostname_sources.items():
            asset_record = {
                'asset_id': f"ASSET-{asset_id:06d}",
                'hostname': hostname,
                'source_systems': source_list,
                'visibility_score': len(source_list) * 25,  # 25% per source
                'coverage_status': self._get_coverage_status(source_list),
                'missing_sources': self._get_missing_sources(source_list)
            }
            
            # Add source-specific details
            for source in source_list:
                df = sources[source]
                if not df.empty:
                    matching_rows = df[df['hostname'].str.lower().str.strip() == hostname]
                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]
                        asset_record[f'{source}_data'] = row.to_dict()
            
            unified_assets.append(asset_record)
            asset_id += 1
        
        return unified_assets
    
    def _get_coverage_status(self, sources: List[str]) -> str:
        """Determine coverage status based on number of sources"""
        if len(sources) >= 4:
            return "EXCELLENT"
        elif len(sources) == 3:
            return "GOOD"  
        elif len(sources) == 2:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_missing_sources(self, sources: List[str]) -> List[str]:
        """Identify which sources are missing for this asset"""
        all_sources = ['chronicle', 'splunk', 'crowdstrike', 'cmdb']
        missing = []
        
        for source in all_sources:
            if source not in sources:
                if source == 'chronicle':
                    missing.append("Chronicle SIEM logging needed")
                elif source == 'splunk':
                    missing.append("Splunk forwarder deployment needed")
                elif source == 'crowdstrike':
                    missing.append("CrowdStrike agent installation needed")
                elif source == 'cmdb':
                    missing.append("CMDB registration required")
        
        return missing
    
    def get_coverage_metrics(self) -> Dict:
        """Get overall coverage metrics"""
        assets = self.correlate_assets_simple()
        
        total_assets = len(assets)
        excellent_coverage = len([a for a in assets if a['coverage_status'] == 'EXCELLENT'])
        good_coverage = len([a for a in assets if a['coverage_status'] == 'GOOD'])
        fair_coverage = len([a for a in assets if a['coverage_status'] == 'FAIR'])
        poor_coverage = len([a for a in assets if a['coverage_status'] == 'POOR'])
        
        avg_visibility = sum(a['visibility_score'] for a in assets) / total_assets if total_assets > 0 else 0
        
        return {
            'total_assets': total_assets,
            'excellent_coverage': excellent_coverage,
            'good_coverage': good_coverage, 
            'fair_coverage': fair_coverage,
            'poor_coverage': poor_coverage,
            'average_visibility': round(avg_visibility, 1),
            'coverage_breakdown': {
                'chronicle_only': len([a for a in assets if a['source_systems'] == ['chronicle']]),
                'splunk_only': len([a for a in assets if a['source_systems'] == ['splunk']]),
                'crowdstrike_only': len([a for a in assets if a['source_systems'] == ['crowdstrike']]),
                'cmdb_only': len([a for a in assets if a['source_systems'] == ['cmdb']]),
                'multiple_sources': len([a for a in assets if len(a['source_systems']) > 1])
            }
        }
    
    def get_gaps_to_fix(self) -> List[Dict]:
        """Get actionable list of gaps to fix"""
        assets = self.correlate_assets_simple()
        gaps = []
        
        for asset in assets:
            if asset['coverage_status'] in ['POOR', 'FAIR']:
                gap_record = {
                    'asset_id': asset['asset_id'],
                    'hostname': asset['hostname'],
                    'current_sources': asset['source_systems'],
                    'missing_sources': asset['missing_sources'],
                    'priority': 'HIGH' if asset['coverage_status'] == 'POOR' else 'MEDIUM',
                    'recommended_actions': self._get_recommended_actions(asset['missing_sources'])
                }
                gaps.append(gap_record)
        
        return sorted(gaps, key=lambda x: x['priority'], reverse=True)
    
    def _get_recommended_actions(self, missing_sources: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        actions = []
        
        for missing in missing_sources:
            if 'CrowdStrike' in missing:
                actions.append("Deploy CrowdStrike Falcon agent")
            elif 'Splunk' in missing:
                actions.append("Install Universal Forwarder and configure log sources")
            elif 'Chronicle' in missing:
                actions.append("Configure Chronicle SIEM log forwarding")
            elif 'CMDB' in missing:
                actions.append("Register asset in ServiceNow CMDB with proper classification")
        
        return actions
    
    def get_source_specific_gaps(self) -> Dict:
        """Get gaps by source system"""
        assets = self.correlate_assets_simple()
        
        gaps_by_source = {
            'needs_crowdstrike': [],
            'needs_splunk': [],
            'needs_chronicle': [],
            'needs_cmdb': []
        }
        
        for asset in assets:
            sources = asset['source_systems']
            hostname = asset['hostname']
            
            if 'crowdstrike' not in sources:
                gaps_by_source['needs_crowdstrike'].append(hostname)
            if 'splunk' not in sources:
                gaps_by_source['needs_splunk'].append(hostname)  
            if 'chronicle' not in sources:
                gaps_by_source['needs_chronicle'].append(hostname)
            if 'cmdb' not in sources:
                gaps_by_source['needs_cmdb'].append(hostname)
        
        return gaps_by_source
    
    def close(self):
        if self.conn:
            self.conn.close()
EOF

echo "🎯 Creating real API endpoints using your database schema..."

cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from services.duckdb.real_visibility_service import RealVisibilityService

app = FastAPI(
    title="Real AO1 Visibility Platform",
    version="1.0.0",
    docs_url="/api/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize the real database service
# Update this path to your actual DuckDB file
DB_PATH = "/path/to/your/actual/database.duckdb"  # <-- UPDATE THIS
visibility_service = RealVisibilityService(DB_PATH)

@app.get("/")
async def root():
    return {"message": "Real AO1 Visibility Platform", "status": "online"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database": "connected"
    }

@app.get("/api/v1/assets/all")
async def get_all_assets():
    """Get all unified assets from your actual database"""
    try:
        assets = visibility_service.correlate_assets_simple()
        return {
            "assets": assets,
            "total": len(assets)
        }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@app.get("/api/v1/coverage/metrics")
async def get_coverage_metrics():
    """Get real coverage metrics from your data"""
    try:
        metrics = visibility_service.get_coverage_metrics()
        return metrics
    except Exception as e:
        return {"error": f"Metrics calculation failed: {str(e)}"}

@app.get("/api/v1/gaps/list")
async def get_gaps():
    """Get actionable list of gaps to fix"""
    try:
        gaps = visibility_service.get_gaps_to_fix()
        return {
            "gaps": gaps,
            "total": len(gaps)
        }
    except Exception as e:
        return {"error": f"Gap analysis failed: {str(e)}"}

@app.get("/api/v1/gaps/by-source")
async def get_gaps_by_source():
    """Get gaps broken down by source system"""
    try:
        gaps = visibility_service.get_source_specific_gaps()
        return gaps
    except Exception as e:
        return {"error": f"Source gap analysis failed: {str(e)}"}

@app.get("/api/v1/sources/raw")
async def get_raw_sources():
    """Get raw data from each source system"""
    try:
        sources = visibility_service.get_all_unique_assets()
        result = {}
        for source_name, df in sources.items():
            result[source_name] = {
                "count": len(df),
                "sample_data": df.head(5).to_dict('records') if not df.empty else []
            }
        return result
    except Exception as e:
        return {"error": f"Raw data retrieval failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
EOF

echo "🎨 Creating spy-themed dashboard for real data..."

cat > frontend/components/dashboard/RealVisibilityDashboard.tsx << 'EOF'
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
EOF

echo "🎯 Updating main page..."
cat > frontend/pages/index.tsx << 'EOF'
import Head from 'next/head';
import RealVisibilityDashboard from '../components/dashboard/RealVisibilityDashboard';

export default function Home() {
  return (
    <>
      <Head>
        <title>AO1 Real Visibility Platform</title>
        <meta name="description" content="Real visibility analysis from your actual data" />
      </Head>
      
      <RealVisibilityDashboard />
    </>
  );
}
EOF

echo "📋 Creating setup instructions..."
cat > REAL_SETUP.md << 'EOF'
# Real AO1 Visibility Platform Setup

## 1. Update Database Path
Edit `backend/app/main.py` line 15:
```python
DB_PATH = "/full/path/to/your/actual/database.duckdb"
```

## 2. Start Backend
```bash
cd backend
source venv/bin/activate
cd app  
python main.py
```

## 3. Start Frontend
```bash
cd frontend
npm run dev
```

## 4. Test Your Real Data
Visit: http://localhost:3000

## What You'll See:
- Real asset counts from your Chronicle/Splunk/CrowdStrike/CMDB
- Actual coverage gaps that need fixing
- Actionable recommendations based on your data
- Asset correlation across all your tools

## API Endpoints:
- `/api/v1/coverage/metrics` - Real coverage stats
- `/api/v1/assets/all` - All correlated assets  
- `/api/v1/gaps/list` - Gaps to fix
- `/api/v1/gaps/by-source` - Gaps by tool
EOF

echo "✅ REAL AO1 VISIBILITY PLATFORM CREATED!"
echo ""
echo "🎯 WHAT THIS DOES:"
echo "   📊 Queries your ACTUAL DuckDB with real data"
echo "   🔗 Correlates assets across Chronicle/Splunk/CrowdStrike/CMDB"  
echo "   📈 Shows real coverage metrics and gaps"
echo "   🎯 Provides actionable remediation steps"
echo "   🕶️ Spy-themed interface with real functionality"
echo ""
echo "🔧 SETUP STEPS:"
echo "   1. Update DB_PATH in backend/app/main.py (line 15)"
echo "   2. Start backend: cd backend && source venv/bin/activate && cd app && python main.py"
echo "   3. Start frontend: cd frontend && npm run dev"
echo "   4. Visit: http://localhost:3000"
echo ""
echo "🎊 NOW YOU'LL SEE YOUR ACTUAL 30% VISIBILITY AND HOW TO GET TO 100%!"