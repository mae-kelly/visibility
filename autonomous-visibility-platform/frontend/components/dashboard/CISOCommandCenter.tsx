'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Shield, 
  Zap, 
  Eye, 
  AlertTriangle,
  Brain,
  Globe,
  Lock,
  Unlock,
  Server,
  Activity,
  TrendingUp,
  TrendingDown,
  Users,
  Clock,
  Target,
  Crosshair,
  Radar,
  Cpu,
  Database,
  Network,
  Bug,
  Key,
  ShieldCheck,
  AlertOctagon,
  BarChart3,
  PieChart,
  LineChart
} from 'lucide-react';

interface CISOData {
  health: any;
  assets: any;
  dashboard: any;
  gaps: any;
  aiStatus: any;
  riskMetrics: any;
}

export default function CISOCommandCenter() {
  const [data, setData] = useState<CISOData | null>(null);
  const [loading, setLoading] = useState(true);
  const [securityStatus, setSecurityStatus] = useState('INITIALIZING');
  const [threatLevel, setThreatLevel] = useState('ELEVATED');
  const [activeIncidents, setActiveIncidents] = useState(0);

  useEffect(() => {
    const fetchExecutiveData = async () => {
      try {
        setSecurityStatus('ANALYZING THREAT LANDSCAPE');
        
        const [healthRes, assetsRes, dashboardRes, gapsRes, aiRes] = await Promise.all([
          fetch('http://localhost:8000/api/health'),
          fetch('http://localhost:8000/api/v1/assets'),
          fetch('http://localhost:8000/api/v1/visibility/dashboard'),
          fetch('http://localhost:8000/api/v1/gaps'),
          fetch('http://localhost:8000/api/v1/ai/status')
        ]);

        const [health, assets, dashboard, gaps, aiStatus] = await Promise.all([
          healthRes.json(),
          assetsRes.json(),
          dashboardRes.json(),
          gapsRes.json(),
          aiRes.json()
        ]);

        // Calculate executive risk metrics
        const riskMetrics = calculateExecutiveRisk(dashboard, gaps, assets);
        setThreatLevel(riskMetrics.overallThreat);
        setActiveIncidents(gaps.total);
        
        setData({ health, assets, dashboard, gaps, aiStatus, riskMetrics });
        setSecurityStatus('COMMAND CENTER ONLINE');
        setLoading(false);
      } catch (err) {
        console.error('Executive briefing compromised:', err);
        setSecurityStatus('SECURITY BREACH - ISOLATING SYSTEMS');
        setThreatLevel('CRITICAL');
        setLoading(false);
      }
    };

    fetchExecutiveData();
    const interval = setInterval(fetchExecutiveData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <ExecutiveLoadingSequence status={securityStatus} />;
  }

  if (!data) {
    return <SecurityIncidentResponse />;
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white relative">
      {/* Mesh gradient background */}
      <div className="mesh-bg"></div>
      
      {/* Executive Command Header */}
      <ExecutiveHeader 
        status={securityStatus} 
        threatLevel={threatLevel}
        incidents={activeIncidents}
      />
      
      {/* Command Center Grid */}
      <div className="relative z-10 p-8 space-y-8">
        {/* Executive Summary Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <ExecutiveMetric
            title="ASSET VISIBILITY"
            value={`${data.dashboard.visibility_percentage.toFixed(1)}%`}
            subtitle="Enterprise Coverage"
            icon={Eye}
            trend={+2.3}
            status="secure"
            target={95}
            current={data.dashboard.visibility_percentage}
          />
          <ExecutiveMetric
            title="SECURITY POSTURE"
            value={data.riskMetrics.postureScore}
            subtitle="Risk Mitigation"
            icon={ShieldCheck}
            trend={+1.7}
            status="warning"
            target={100}
            current={85}
          />
          <ExecutiveMetric
            title="ACTIVE THREATS"
            value={data.gaps.total}
            subtitle="Critical Exposure"
            icon={AlertOctagon}
            trend={-5.2}
            status="critical"
            target={0}
            current={data.gaps.total}
          />
          <ExecutiveMetric
            title="AI PREDICTIONS"
            value={`${(data.aiStatus.total_predictions_today / 1000).toFixed(1)}K`}
            subtitle="Automated Analysis"
            icon={Brain}
            trend={+12.4}
            status="secure"
            target={10000}
            current={data.aiStatus.total_predictions_today}
          />
        </div>

        {/* Executive Intelligence Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Threat Landscape */}
          <div className="lg:col-span-2">
            <ThreatLandscapeMap data={data} />
          </div>
          
          {/* Executive Briefing */}
          <div className="space-y-8">
            <ExecutiveBriefing metrics={data.riskMetrics} />
            <RealTimeIntelligence threats={data.gaps.gaps.slice(0, 4)} />
          </div>
        </div>

        {/* Strategic Operations Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <AssetSecurityMatrix assets={data.assets.assets.slice(0, 6)} />
          <ThreatAnalysisPanel gaps={data.gaps.gaps.slice(0, 5)} />
        </div>
      </div>
    </div>
  );
}

function calculateExecutiveRisk(dashboard: any, gaps: any, assets: any) {
  const criticalGaps = gaps.gaps?.filter((g: any) => g.gap_severity === 'High')?.length || 0;
  const postureScore = Math.max(0, 100 - (criticalGaps * 5));
  
  let overallThreat = 'LOW';
  if (criticalGaps > 10) overallThreat = 'CRITICAL';
  else if (criticalGaps > 5) overallThreat = 'HIGH';
  else if (criticalGaps > 2) overallThreat = 'ELEVATED';

  return {
    postureScore: postureScore.toString(),
    overallThreat,
    criticalExposures: criticalGaps,
    businessImpact: criticalGaps > 5 ? 'HIGH' : 'MEDIUM'
  };
}

function ExecutiveLoadingSequence({ status }: { status: string }) {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
      <div className="mesh-bg"></div>
      <div className="relative z-10 text-center max-w-2xl mx-auto p-8">
        <div className="glass p-12 floating-element">
          <div className="mb-8">
            <Shield size={80} className="mx-auto mb-6 text-cyber-blue" />
            <h1 className="text-5xl font-bold mb-4 title-gradient font-executive">
              CISO COMMAND CENTER
            </h1>
            <p className="text-xl text-gray-300 font-cyber">EXECUTIVE SECURITY OPERATIONS</p>
          </div>
          
          <div className="space-y-6">
            <div className="text-xl font-cyber text-cyber-blue">{status}</div>
            
            <div className="relative">
              <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyber-blue via-cyber-purple to-cyber-pink animate-pulse"></div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6 text-sm font-cyber">
              <div className="status-secure">✓ NEURAL NETWORKS ACTIVE</div>
              <div className="status-secure">✓ THREAT INTELLIGENCE SYNCED</div>
              <div className="status-warning">◐ RISK ASSESSMENT RUNNING</div>
              <div className="status-warning">◐ EXECUTIVE BRIEFING PREP</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SecurityIncidentResponse() {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
      <div className="mesh-bg"></div>
      <div className="relative z-10 text-center glass p-12">
        <AlertTriangle size={80} className="mx-auto mb-6 status-critical" />
        <h1 className="text-4xl font-bold mb-4 status-critical font-executive">SECURITY INCIDENT</h1>
        <p className="text-xl text-gray-300 mb-6">Command center communication compromised</p>
        <button 
          onClick={() => window.location.reload()}
          className="glass px-8 py-4 hover:border-cyber-blue transition-all font-cyber micro-bounce"
        >
          REESTABLISH SECURE CONNECTION
        </button>
      </div>
    </div>
  );
}

function ExecutiveHeader({ status, threatLevel, incidents }: any) {
  const getThreatColor = (level: string) => {
    switch (level) {
      case 'CRITICAL': return 'status-critical';
      case 'HIGH': return 'status-critical';
      case 'ELEVATED': return 'status-warning';
      default: return 'status-secure';
    }
  };

  return (
    <div className="border-b border-gray-700 executive-panel relative z-10">
      <div className="px-8 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-4">
              <div className="p-3 glass">
                <Shield size={32} className="text-cyber-blue" />
              </div>
              <div>
                <h1 className="text-2xl font-bold font-executive">CISO COMMAND CENTER</h1>
                <p className="text-gray-400 font-cyber text-sm">Executive Security Operations</p>
              </div>
            </div>
            
            <div className="hidden lg:flex items-center space-x-8">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-cyber-green rounded-full animate-pulse"></div>
                <span className="font-cyber text-sm status-secure">SECURE CHANNEL</span>
              </div>
              <div className="flex items-center space-x-3">
                <Radar size={16} className="text-cyber-blue animate-spin" />
                <span className="font-cyber text-sm">CONTINUOUS MONITORING</span>
              </div>
              <div className="flex items-center space-x-3">
                <Brain size={16} className="text-cyber-purple" />
                <span className="font-cyber text-sm text-cyber-purple">AI ANALYSIS ACTIVE</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="text-right">
              <div className="font-cyber text-sm text-gray-400">THREAT LEVEL</div>
              <div className={`font-bold ${getThreatColor(threatLevel)}`}>{threatLevel}</div>
            </div>
            <div className="text-right">
              <div className="font-cyber text-sm text-gray-400">ACTIVE INCIDENTS</div>
              <div className="font-bold text-cyber-orange">{incidents}</div>
            </div>
            <div className="text-right">
              <div className="font-cyber text-sm text-gray-400">{new Date().toLocaleDateString()}</div>
              <div className="font-cyber text-sm text-cyber-blue">{new Date().toLocaleTimeString()}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ExecutiveMetric({ title, value, subtitle, icon: Icon, trend, status, target, current }: any) {
  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'critical': return 'threat-critical';
      case 'warning': return 'threat-medium';
      case 'secure': return 'threat-low';
      default: return 'threat-medium';
    }
  };

  const progressPercentage = (current / target) * 100;

  return (
    <div className={`metric-card p-8 ${getStatusStyle(status)}`}>
      <div className="flex items-start justify-between mb-6">
        <div className="flex-1">
          <p className="font-cyber text-xs text-gray-400 mb-3 tracking-wider">{title}</p>
          <p className="text-4xl font-bold font-executive mb-2">{value}</p>
          <p className="text-sm text-gray-400 font-cyber">{subtitle}</p>
        </div>
        <div className="p-4 glass">
          <Icon size={28} className="text-white" />
        </div>
      </div>
      
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {trend > 0 ? (
              <TrendingUp size={16} className="text-cyber-green" />
            ) : (
              <TrendingDown size={16} className="text-cyber-red" />
            )}
            <span className={`text-sm font-cyber ${trend > 0 ? 'text-cyber-green' : 'text-cyber-red'}`}>
              {trend > 0 ? '+' : ''}{trend}%
            </span>
            <span className="text-xs text-gray-500">vs last week</span>
          </div>
          <span className="text-xs font-cyber text-gray-400">
            Target: {target}{typeof target === 'number' && target > 100 ? '' : '%'}
          </span>
        </div>
        
        <div className="relative">
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-cyber-blue to-cyber-purple transition-all duration-1000"
              style={{ width: `${Math.min(progressPercentage, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ThreatLandscapeMap({ data }: any) {
  const threatNodes = [
    { id: 'DC-EAST', status: 'SECURE', risk: 'LOW', x: 25, y: 30, assets: 45 },
    { id: 'DC-WEST', status: 'MONITORING', risk: 'MEDIUM', x: 70, y: 25, assets: 38 },
    { id: 'CLOUD-1', status: 'THREAT', risk: 'HIGH', x: 80, y: 60, assets: 67 },
    { id: 'EDGE-NET', status: 'SECURE', risk: 'LOW', x: 40, y: 70, assets: 23 },
    { id: 'MOBILE', status: 'MONITORING', risk: 'MEDIUM', x: 55, y: 45, assets: 89 },
  ];

  return (
    <div className="glass p-8">
      <div className="flex items-center justify-between mb-8">
        <h3 className="text-2xl font-bold font-executive">Global Threat Landscape</h3>
        <div className="flex items-center space-x-4">
          <Globe size={20} className="text-cyber-blue animate-pulse" />
          <span className="font-cyber text-sm text-cyber-blue">REAL-TIME INTELLIGENCE</span>
        </div>
      </div>
      
      <div className="relative h-80 threat-map rounded-2xl p-6 neural-grid">
        {/* Threat nodes */}
        {threatNodes.map((node, index) => (
          <div
            key={index}
            className={`absolute w-6 h-6 rounded-full border-2 transition-all duration-500 cursor-pointer group ${
              node.status === 'THREAT' ? 'bg-cyber-red border-cyber-red animate-pulse' :
              node.status === 'MONITORING' ? 'bg-cyber-orange border-cyber-orange' :
              'bg-cyber-green border-cyber-green'
            }`}
            style={{ left: `${node.x}%`, top: `${node.y}%` }}
            title={`${node.id} - ${node.status}`}
          >
            <div className="absolute -top-8 -left-12 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 px-3 py-1 rounded text-xs font-cyber whitespace-nowrap">
              {node.id}: {node.assets} assets
            </div>
          </div>
        ))}
        
        {/* Connection lines */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <defs>
            <linearGradient id="connectionGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(0, 212, 255, 0.6)" />
              <stop offset="100%" stopColor="rgba(139, 92, 246, 0.6)" />
            </linearGradient>
          </defs>
          <line x1="25%" y1="30%" x2="70%" y2="25%" stroke="url(#connectionGrad)" strokeWidth="2" opacity="0.7" />
          <line x1="70%" y1="25%" x2="80%" y2="60%" stroke="url(#connectionGrad)" strokeWidth="2" opacity="0.7" />
          <line x1="40%" y1="70%" x2="55%" y2="45%" stroke="url(#connectionGrad)" strokeWidth="2" opacity="0.7" />
        </svg>
      </div>
      
      <div className="mt-6 grid grid-cols-3 gap-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-cyber-green">{threatNodes.filter(n => n.status === 'SECURE').length}</div>
          <div className="text-sm font-cyber text-gray-400">Secure Zones</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-cyber-orange">{threatNodes.filter(n => n.status === 'MONITORING').length}</div>
          <div className="text-sm font-cyber text-gray-400">Under Watch</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-cyber-red">{threatNodes.filter(n => n.status === 'THREAT').length}</div>
          <div className="text-sm font-cyber text-gray-400">Active Threats</div>
        </div>
      </div>
    </div>
  );
}

function ExecutiveBriefing({ metrics }: any) {
  return (
    <div className="glass p-6">
      <div className="flex items-center space-x-3 mb-6">
        <BarChart3 size={20} className="text-cyber-purple" />
        <span className="text-lg font-bold font-executive">Executive Brief</span>
      </div>
      <div className="space-y-6">
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="font-cyber text-sm">Security Posture</span>
            <span className="font-bold text-cyber-green">{metrics.postureScore}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="font-cyber text-sm">Business Impact</span>
            <span className={`font-bold ${metrics.businessImpact === 'HIGH' ? 'text-cyber-red' : 'text-cyber-orange'}`}>
              {metrics.businessImpact}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="font-cyber text-sm">Critical Exposures</span>
            <span className="font-bold text-cyber-red">{metrics.criticalExposures}</span>
          </div>
        </div>
        
        <div className="border-t border-gray-700 pt-4">
          <h4 className="font-cyber text-sm text-gray-400 mb-3">Recommended Actions</h4>
          <div className="space-y-2 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-cyber-red rounded-full"></div>
              <span>Deploy additional monitoring to high-risk assets</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-cyber-orange rounded-full"></div>
              <span>Review access controls for critical systems</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-cyber-blue rounded-full"></div>
              <span>Increase threat hunting frequency</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RealTimeIntelligence({ threats }: any) {
  return (
    <div className="glass p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Activity size={20} className="text-cyber-red animate-pulse" />
        <span className="text-lg font-bold font-executive">Live Intelligence</span>
      </div>
      <div className="space-y-3">
        {threats.map((threat: any, index: number) => (
          <div key={index} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                threat.gap_severity === 'High' ? 'bg-cyber-red animate-pulse' : 'bg-cyber-orange'
              }`}></div>
              <div>
                <div className="text-sm font-cyber font-bold">{threat.hostname}</div>
                <div className="text-xs text-gray-400">Risk Score: {threat.visibility_score}%</div>
              </div>
            </div>
            <div className={`px-3 py-1 rounded text-xs font-cyber ${
              threat.gap_severity === 'High' ? 'bg-cyber-red/20 text-cyber-red' : 'bg-cyber-orange/20 text-cyber-orange'
            }`}>
              {threat.gap_severity === 'High' ? 'CRITICAL' : 'ELEVATED'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AssetSecurityMatrix({ assets }: any) {
  return (
    <div className="glass p-8">
      <h3 className="text-2xl font-bold font-executive mb-6">Asset Security Matrix</h3>
      <div className="space-y-4">
        {assets.map((asset: any, index: number) => (
          <div key={index} className="flex items-center justify-between p-4 executive-panel rounded-xl hover:border-cyber-blue transition-all">
            <div className="flex items-center space-x-4">
              <div className="p-3 glass">
                <Server size={20} className="text-cyber-blue" />
              </div>
              <div>
                <div className="text-lg font-bold font-executive">{asset.hostname}</div>
                <div className="text-sm text-gray-400 font-cyber">{asset.ip} • {asset.environment}</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xl font-bold">{asset.visibility}%</div>
              <div className={`text-sm font-cyber ${
                asset.visibility > 90 ? 'status-secure' : 
                asset.visibility > 70 ? 'status-warning' : 'status-critical'
              }`}>
                {asset.visibility > 90 ? 'SECURE' : asset.visibility > 70 ? 'MONITOR' : 'CRITICAL'}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ThreatAnalysisPanel({ gaps }: any) {
  return (
    <div className="glass p-8">
      <h3 className="text-2xl font-bold font-executive mb-6">Threat Analysis</h3>
      <div className="space-y-4">
        {gaps.map((gap: any, index: number) => (
          <div key={index} className="p-4 threat-critical rounded-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                <AlertTriangle size={20} className="text-cyber-red" />
                <div>
                  <div className="font-bold font-executive">{gap.hostname}</div>
                  <div className="text-sm text-gray-400 font-cyber">Exposure Level: {gap.visibility_score}%</div>
                </div>
              </div>
              <div className="px-3 py-1 bg-cyber-red/20 text-cyber-red rounded text-sm font-cyber">
                {gap.gap_severity === 'High' ? 'CRITICAL' : 'ELEVATED'}
              </div>
            </div>
            <div className="text-sm text-gray-300">
              Multiple security gaps detected. Immediate remediation required.
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
