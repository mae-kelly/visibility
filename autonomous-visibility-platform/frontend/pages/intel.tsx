import { useState, useEffect } from 'react';

export default function IntelTest() {
  const [classified, setClassified] = useState(null);
  const [error, setError] = useState(null);
  const [accessLevel, setAccessLevel] = useState('RESTRICTED');

  useEffect(() => {
    const timer = setTimeout(() => setAccessLevel('LEVEL 9 CLEARANCE'), 1000);
    
    fetch('http://localhost:8000/api/health')
      .then(res => res.json())
      .then(data => {
        setClassified(data);
        setAccessLevel('ACCESS GRANTED');
      })
      .catch(err => {
        setError(err.message);
        setAccessLevel('ACCESS DENIED');
      });

    return () => clearTimeout(timer);
  }, []);

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: '#0a0a0a', 
      color: '#00ffff', 
      fontFamily: 'JetBrains Mono, monospace',
      padding: '20px'
    }}>
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ 
          fontSize: '2rem', 
          textShadow: '0 0 10px #00ffff',
          marginBottom: '10px'
        }}>
          🕶️ CLASSIFIED INTELLIGENCE TERMINAL
        </h1>
        <p style={{ color: '#39ff14' }}>STATUS: {accessLevel}</p>
      </div>
      
      {error && (
        <div style={{ 
          color: '#ff006e', 
          textAlign: 'center',
          border: '1px solid #ff006e',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <h2>🚨 SECURITY BREACH DETECTED</h2>
          <p>NEURAL LINK COMPROMISED: {error}</p>
          <p>RECOMMENDATION: VERIFY BACKEND QUANTUM ENCRYPTION</p>
        </div>
      )}

      {classified && (
        <div style={{ 
          border: '1px solid #00ffff',
          padding: '20px',
          backgroundColor: 'rgba(0, 255, 255, 0.05)'
        }}>
          <h2 style={{ color: '#39ff14' }}>✅ QUANTUM CHANNEL ESTABLISHED</h2>
          <pre style={{ 
            color: '#00ffff',
            fontSize: '14px',
            overflow: 'auto'
          }}>
            {JSON.stringify(classified, null, 2)}
          </pre>
        </div>
      )}

      <div style={{ 
        marginTop: '40px',
        padding: '20px',
        border: '1px solid #bf00ff',
        backgroundColor: 'rgba(191, 0, 255, 0.05)'
      }}>
        <h3 style={{ color: '#bf00ff' }}>🎯 MISSION OBJECTIVES</h3>
        <p>✓ Establish secure neural link</p>
        <p>✓ Verify quantum encryption protocols</p>
        <p>🎯 Navigate to <a href="/" style={{ color: '#00ffff' }}>MAIN NEXUS</a> for full interface</p>
      </div>
    </div>
  );
}
