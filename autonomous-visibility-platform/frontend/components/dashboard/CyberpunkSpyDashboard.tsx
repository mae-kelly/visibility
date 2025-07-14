'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Shield, 
  Zap, 
  Eye, 
  Cpu, 
  Activity, 
  Target, 
  AlertTriangle,
  Brain,
  Wifi,
  Lock,
  Unlock,
  Server,
  Globe,
  Terminal,
  Search,
  Radar
} from 'lucide-react';

interface SpyData {
  health: any;
  assets: any;
  dashboard: any;
  gaps: any;
  aiStatus: any;
  timestamp: number;
}

export default function CyberpunkSpyDashboard() {
  const [data, setData] = useState<SpyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING');
  const [scanProgress, setScanProgress] = useState(0);
  const [terminalLines, setTerminalLines] = useState<string[]>([]);
  const matrixRef = useRef<HTMLDivElement>(null);

  // Matrix rain effect
  useEffect(() => {
    const createMatrixRain = () => {
      if (!matrixRef.current) return;
      
      const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
      const container = matrixRef.current;
      
      for (let i = 0; i < 50; i++) {
        const char = document.createElement('div');
        char.className = 'matrix-char';
        char.textContent = chars[Math.floor(Math.random() * chars.length)];
        char.style.left = Math.random() * 100 + '%';
        char.style.animationDelay = Math.random() * 3 + 's';
        char.style.animationDuration = (Math.random() * 3 + 2) + 's';
        container.appendChild(char);
        
        setTimeout(() => {
          if (container.contains(char)) {
            container.removeChild(char);
          }
        }, 5000);
      }
    };

    const interval = setInterval(createMatrixRain, 500);
    return () => clearInterval(interval);
  }, []);

  // Terminal simulation
  useEffect(() => {
    const terminalMessages = [
      '> NEURAL LINK ESTABLISHED...',
      '> QUANTUM ENCRYPTION ACTIVE',
      '> SCANNING THREAT VECTORS...',
      '> AI MODELS SYNCHRONIZED',
      '> SHADOW ASSETS DETECTED',
      '> CORRELATION ENGINE ONLINE',
      '> VISIBILITY MATRIX UPDATED',
      '> CYBER DEFENSE PROTOCOLS ACTIVE'
    ];

    let messageIndex = 0;
    const terminalInterval = setInterval(() => {
      if (messageIndex < terminalMessages.length) {
        setTerminalLines(prev => [...prev, terminalMessages[messageIndex]]);
        messageIndex++;
      } else {
        messageIndex = 0;
        setTerminalLines([]);
      }
    }, 2000);

    return () => clearInterval(terminalInterval);
  }, []);

  // Data fetching with spy-like status updates
  useEffect(() => {
    const fetchSpyData = async () => {
      try {
        setConnectionStatus('INFILTRATING');
        setScanProgress(20);
        
        const [healthRes, assetsRes, dashboardRes, gapsRes, aiRes] = await Promise.all([
          fetch('http://localhost:8000/api/health'),
          fetch('http://localhost:8000/api/v1/assets'),
          fetch('http://localhost:8000/api/v1/visibility/dashboard'),
          fetch('http://localhost:8000/api/v1/gaps'),
          fetch('http://localhost:8000/api/v1/ai/status')
        ]);

        setScanProgress(60);
        setConnectionStatus('DECRYPTING');

        const [health, assets, dashboard, gaps, aiStatus] = await Promise.all([
          healthRes.json(),
          assetsRes.json(),
          dashboardRes.json(),
          gapsRes.json(),
          aiRes.json()
        ]);

        setScanProgress(100);
        setConnectionStatus('NEURAL LINK ACTIVE');
        
        setData({ health, assets, dashboard, gaps, aiStatus, timestamp: Date.now() });
        setLoading(false);
      } catch (err) {
        console.error('SECURITY BREACH:', err);
        setConnectionStatus('CONNECTION COMPROMISED');
        setLoading(false);
      }
    };

    fetchSpyData();
    const interval = setInterval(fetchSpyData, 15000); // Update every 15 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <LoadingSequence status={connectionStatus} progress={scanProgress} />;
  }

  if (!data) {
    return <SecurityBreach />;
  }

  return (
    <div className="min-h-screen bg-cyber-dark text-neon-cyan grid-overlay relative">
      {/* Matrix rain background */}
      <div ref={matrixRef} className="matrix-rain"></div>
      
      {/* Main spy interface */}
      <div className="relative z-10">
        {/* Command Header */}
        <SpyHeader status={connectionStatus} />
        
        {/* Mission Control Grid */}
        <div className="p-6 space-y-6">
          {/* Primary Intel Dashboard */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <CyberMetric
              title="ASSETS UNDER SURVEILLANCE"
              value={data.assets.total}
              subtitle="ACTIVE TARGETS"
              icon={Target}
              color="cyan"
              threat="LOW"
            />
            <CyberMetric
              title="VISIBILITY PENETRATION"
              value={`${data.dashboard.visibility_percentage.toFixed(1)}%`}
              subtitle="NETWORK INFILTRATION"
              icon={Eye}
              color="green"
              threat="SECURE"
            />
            <CyberMetric
              title="THREAT VECTORS"
              value={data.gaps.total}
              subtitle="CRITICAL EXPOSURE"
              icon={AlertTriangle}
              color="pink"
              threat="HIGH"
            />
            <CyberMetric
              title="AI NEURAL ACTIVITY"
              value={data.aiStatus.total_predictions_today}
              subtitle="QUANTUM PREDICTIONS"
              icon={Brain}
              color="purple"
              threat="ACTIVE"
            />
          </div>

          {/* Intelligence Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Network Penetration Map */}
            <div className="lg:col-span-2">
              <NetworkPenetrationGrid data={data} />
            </div>
            
            {/* Mission Terminal */}
            <div className="space-y-6">
              <MissionTerminal lines={terminalLines} />
              <ThreatRadar threats={data.gaps.gaps.slice(0, 5)} />
            </div>
          </div>

          {/* Shadow Operations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ShadowAssetGrid assets={data.assets.assets.slice(0, 8)} />
            <IntelligenceMatrix gaps={data.gaps.gaps.slice(0, 6)} />
          </div>
        </div>
      </div>
    </div>
  );
}

function LoadingSequence({ status, progress }: { status: string; progress: number }) {
  return (
    <div className="min-h-screen bg-cyber-dark flex items-center justify-center text-neon-cyan">
      <div className="text-center max-w-2xl mx-auto p-8">
        <div className="mb-8">
          <Shield size={64} className="mx-auto mb-4 neon-text" />
          <h1 className="text-4xl font-bold mb-2 neon-text tracking-wider">
            QUANTUM SECURITY NEXUS
          </h1>
          <p className="text-neon-green font-mono">CLASSIFIED • LEVEL 9 CLEARANCE</p>
        </div>
        
        <div className="space-y-6">
          <div className="text-xl font-mono terminal-cursor">{status}</div>
          
          <div className="relative">
            <div className="w-full h-2 bg-black border border-neon-cyan">
              <div 
                className="h-full bg-gradient-to-r from-neon-cyan to-neon-pink transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <div className="absolute -top-6 right-0 text-sm font-mono">{progress}%</div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-xs font-mono">
            <div className="text-neon-green">✓ QUANTUM ENCRYPTION</div>
            <div className="text-neon-green">✓ NEURAL NETWORKS</div>
            <div className="text-neon-cyan">○ SHADOW PROTOCOLS</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SecurityBreach() {
  return (
    <div className="min-h-screen bg-cyber-dark flex items-center justify-center text-neon-pink">
      <div className="text-center">
        <AlertTriangle size={64} className="mx-auto mb-4 animate-pulse" />
        <h1 className="text-3xl font-bold mb-4 font-mono">SECURITY BREACH DETECTED</h1>
        <p className="text-neon-cyan mb-4">NEURAL LINK COMPROMISED</p>
        <button 
          onClick={() => window.location.reload()}
          className="px-6 py-3 border border-neon-pink hover:bg-neon-pink hover:text-black transition-all font-mono"
        >
          REESTABLISH CONNECTION
        </button>
      </div>
    </div>
  );
}

function SpyHeader({ status }: { status: string }) {
  return (
    <div className="border-b border-neon-cyan bg-cyber backdrop-blur-sm">
      <div className="px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 border border-neon-cyan bg-black">
                <Shield size={24} className="text-neon-cyan" />
              </div>
              <div>
                <h1 className="text-xl font-bold font-mono tracking-wider">NEXUS</h1>
                <p className="text-xs text-neon-green font-mono">QUANTUM • SECURITY • MATRIX</p>
              </div>
            </div>
            
            <div className="hidden lg:flex items-center space-x-6 font-mono text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-neon-green rounded-full animate-pulse"></div>
                <span className="text-neon-green">ENCRYPTED</span>
              </div>
              <div className="flex items-center space-x-2">
                <Radar size={16} className="text-neon-cyan animate-spin" />
                <span>SCANNING</span>
              </div>
              <div className="flex items-center space-x-2">
                <Brain size={16} className="text-neon-purple" />
                <span className="text-neon-purple">AI ACTIVE</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-right font-mono text-sm">
              <div className="text-neon-cyan">{status}</div>
              <div className="text-xs text-neon-green">
                {new Date().toLocaleTimeString()} UTC
              </div>
            </div>
            <div className="w-8 h-8 border border-neon-cyan bg-black flex items-center justify-center">
              <div className="w-4 h-4 bg-neon-cyan"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function CyberMetric({ title, value, subtitle, icon: Icon, color, threat }: any) {
  const colorMap = {
    cyan: 'text-neon-cyan border-neon-cyan',
    green: 'text-neon-green border-neon-green',
    pink: 'text-neon-pink border-neon-pink',
    purple: 'text-neon-purple border-neon-purple'
  };

  return (
    <div className="cyber-card p-6 holographic">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <p className="text-xs font-mono text-gray-400 mb-2 tracking-wider">{title}</p>
          <p className={`text-3xl font-bold font-mono mb-1 ${colorMap[color]}`}>{value}</p>
          <p className="text-xs text-gray-500 font-mono">{subtitle}</p>
        </div>
        <div className={`p-3 border ${colorMap[color]} bg-black`}>
          <Icon size={20} className={colorMap[color].split(' ')[0]} />
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className={`px-2 py-1 border text-xs font-mono ${colorMap[color]}`}>
          {threat}
        </div>
        <div className="flex space-x-1">
          {[...Array(5)].map((_, i) => (
            <div key={i} className={`w-1 h-4 ${i < 3 ? 'bg-neon-cyan' : 'bg-gray-800'}`}></div>
          ))}
        </div>
      </div>
    </div>
  );
}

function NetworkPenetrationGrid({ data }: any) {
  const networkNodes = [
    { id: 'NODE_001', status: 'INFILTRATED', threat: 'LOW', x: 20, y: 30 },
    { id: 'NODE_002', status: 'SECURED', threat: 'NONE', x: 60, y: 20 },
    { id: 'NODE_003', status: 'COMPROMISED', threat: 'HIGH', x: 80, y: 60 },
    { id: 'NODE_004', status: 'SCANNING', threat: 'MEDIUM', x: 40, y: 70 },
    { id: 'NODE_005', status: 'PROTECTED', threat: 'LOW', x: 30, y: 50 },
  ];

  return (
    <div className="cyber-card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-mono font-bold text-neon-cyan">NETWORK PENETRATION MAP</h3>
        <div className="flex items-center space-x-2">
          <Globe size={16} className="text-neon-green animate-pulse" />
          <span className="text-xs font-mono text-neon-green">LIVE FEED</span>
        </div>
      </div>
      
      <div className="relative h-64 bg-black border border-neon-cyan p-4">
        {/* Grid overlay */}
        <div className="absolute inset-0 grid-overlay opacity-30"></div>
        
        {/* Network nodes */}
        {networkNodes.map((node, index) => (
          <div
            key={index}
            className={`absolute w-4 h-4 border-2 ${
              node.status === 'COMPROMISED' ? 'border-neon-pink bg-neon-pink' :
              node.status === 'SECURED' ? 'border-neon-green bg-neon-green' :
              'border-neon-cyan bg-neon-cyan'
            } animate-pulse`}
            style={{ left: `${node.x}%`, top: `${node.y}%` }}
            title={`${node.id} - ${node.status}`}
          >
            <div className={`absolute -top-6 -left-8 text-xs font-mono whitespace-nowrap ${
              node.status === 'COMPROMISED' ? 'text-neon-pink' :
              node.status === 'SECURED' ? 'text-neon-green' :
              'text-neon-cyan'
            }`}>
              {node.id}
            </div>
          </div>
        ))}
        
        {/* Connection lines */}
        <svg className="absolute inset-0 w-full h-full">
          <line x1="20%" y1="30%" x2="60%" y2="20%" stroke="cyan" strokeWidth="1" opacity="0.5" />
          <line x1="60%" y1="20%" x2="80%" y2="60%" stroke="cyan" strokeWidth="1" opacity="0.5" />
          <line x1="40%" y1="70%" x2="30%" y2="50%" stroke="cyan" strokeWidth="1" opacity="0.5" />
        </svg>
      </div>
      
      <div className="mt-4 grid grid-cols-3 gap-4 text-xs font-mono">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-neon-green"></div>
          <span>SECURED: 2</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-neon-cyan"></div>
          <span>ACTIVE: 2</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-neon-pink"></div>
          <span>COMPROMISED: 1</span>
        </div>
      </div>
    </div>
  );
}

function MissionTerminal({ lines }: { lines: string[] }) {
  return (
    <div className="cyber-card p-4">
      <div className="flex items-center space-x-2 mb-4">
        <Terminal size={16} className="text-neon-green" />
        <span className="text-sm font-mono text-neon-green">MISSION_TERMINAL</span>
      </div>
      <div className="bg-black p-4 h-40 overflow-hidden">
        {lines.map((line, index) => (
          <div 
            key={index} 
            className="text-xs font-mono text-neon-green mb-1 terminal-cursor"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}

function ThreatRadar({ threats }: any) {
  return (
    <div className="cyber-card p-4">
      <div className="flex items-center space-x-2 mb-4">
        <Radar size={16} className="text-neon-pink animate-spin" />
        <span className="text-sm font-mono text-neon-pink">THREAT_RADAR</span>
      </div>
      <div className="space-y-2">
        {threats.map((threat: any, index: number) => (
          <div key={index} className="flex items-center justify-between text-xs font-mono">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 ${threat.gap_severity === 'High' ? 'bg-neon-pink' : 'bg-neon-orange'} animate-pulse`}></div>
              <span className="text-gray-300">{threat.hostname}</span>
            </div>
            <span className={threat.gap_severity === 'High' ? 'text-neon-pink' : 'text-neon-orange'}>
              {threat.gap_severity}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShadowAssetGrid({ assets }: any) {
  return (
    <div className="cyber-card p-6">
      <h3 className="text-lg font-mono font-bold text-neon-cyan mb-4">SHADOW ASSETS</h3>
      <div className="space-y-3">
        {assets.map((asset: any, index: number) => (
          <div key={index} className="flex items-center justify-between p-3 bg-black border border-gray-800 hover:border-neon-cyan transition-all">
            <div className="flex items-center space-x-3">
              <Server size={16} className="text-neon-cyan" />
              <div>
                <div className="text-sm font-mono font-bold">{asset.hostname}</div>
                <div className="text-xs text-gray-400 font-mono">{asset.ip}</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm font-mono font-bold">{asset.visibility}%</div>
              <div className={`text-xs font-mono ${
                asset.visibility > 90 ? 'text-neon-green' : 
                asset.visibility > 70 ? 'text-neon-orange' : 'text-neon-pink'
              }`}>
                {asset.visibility > 90 ? 'SECURE' : asset.visibility > 70 ? 'WATCH' : 'THREAT'}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function IntelligenceMatrix({ gaps }: any) {
  return (
    <div className="cyber-card p-6">
      <h3 className="text-lg font-mono font-bold text-neon-pink mb-4">INTELLIGENCE MATRIX</h3>
      <div className="space-y-3">
        {gaps.map((gap: any, index: number) => (
          <div key={index} className="flex items-center justify-between p-3 bg-red-950/20 border border-red-700/30">
            <div className="flex items-center space-x-3">
              <AlertTriangle size={16} className="text-neon-pink" />
              <div>
                <div className="text-sm font-mono font-bold">{gap.hostname}</div>
                <div className="text-xs text-gray-400 font-mono">EXPOSURE: {gap.visibility_score}%</div>
              </div>
            </div>
            <div className={`px-2 py-1 border text-xs font-mono ${
              gap.gap_severity === 'High' ? 'border-neon-pink text-neon-pink' : 'border-neon-orange text-neon-orange'
            }`}>
              {gap.gap_severity === 'High' ? 'CRITICAL' : 'MODERATE'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
