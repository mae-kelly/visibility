import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import structlog
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "ml_engine"))

from models.graph.hgt_model import AssetCorrelationEngine
from models.entity.ditto_resolver import SecurityEntityResolver
from models.anomaly.deep_svdd import ShadowAssetDetector
from services.duckdb.visibility_analyzer import VisibilityAnalyzer

logger = structlog.get_logger()

class AutonomousCorrelationService:
    def __init__(self, db_path: str, model_dir: str):
        self.db = VisibilityAnalyzer(db_path)
        self.model_dir = Path(model_dir)
        
        self.correlation_engine = AssetCorrelationEngine()
        self.entity_resolver = SecurityEntityResolver()
        self.shadow_detector = ShadowAssetDetector()
        
        self._load_models()
    
    def _load_models(self):
        model_files = {
            'hgt': self.model_dir / "hgt_correlation.pth",
            'ditto': self.model_dir / "ditto_resolver.pth", 
            'svdd': self.model_dir / "shadow_detector.pth"
        }
        
        for model_name, model_path in model_files.items():
            if model_path.exists():
                logger.info(f"Loading {model_name} model from {model_path}")
                if model_name == 'hgt':
                    self.correlation_engine.load_model(str(model_path))
                elif model_name == 'ditto':
                    self.entity_resolver.load_model(str(model_path))
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    async def analyze_coverage_gaps(self) -> Dict:
        gaps_df = self.db.get_coverage_gaps()
        
        critical_gaps = gaps_df[gaps_df['gap_severity'] == 'critical']
        high_gaps = gaps_df[gaps_df['gap_severity'] == 'high']
        
        analysis = {
            'total_gaps': len(gaps_df),
            'critical_count': len(critical_gaps),
            'high_count': len(high_gaps),
            'gaps_by_environment': gaps_df.groupby('environment')['gap_severity'].value_counts().to_dict(),
            'gaps_by_class': gaps_df.groupby('asset_class')['gap_severity'].value_counts().to_dict(),
            'lowest_coverage_assets': gaps_df.head(10).to_dict('records')
        }
        
        logger.info(f"Analyzed {analysis['total_gaps']} coverage gaps")
        return analysis
    
    async def correlate_cross_tool_assets(self) -> List[Dict]:
        training_data = self.db.get_ml_training_data()
        
        crowdstrike_assets = training_data[training_data['source_system'] == 'crowdstrike']
        splunk_assets = training_data[training_data['source_system'] == 'splunk'] 
        chronicle_assets = training_data[training_data['source_system'] == 'chronicle']
        cmdb_assets = training_data[training_data['source_system'] == 'cmdb']
        
        correlations = []
        
        for _, cs_asset in crowdstrike_assets.iterrows():
            cs_dict = cs_asset.to_dict()
            
            for source_name, source_df in [
                ('splunk', splunk_assets),
                ('chronicle', chronicle_assets), 
                ('cmdb', cmdb_assets)
            ]:
                for _, source_asset in source_df.iterrows():
                    source_dict = source_asset.to_dict()
                    
                    similarity_score = self._calculate_similarity(cs_dict, source_dict)
                    
                    if similarity_score > 0.8:
                        correlations.append({
                            'asset1_id': cs_asset['asset_id'],
                            'asset1_source': 'crowdstrike',
                            'asset1_hostname': cs_asset.get('primary_hostname'),
                            'asset2_id': source_asset['asset_id'],
                            'asset2_source': source_name,
                            'asset2_hostname': source_asset.get('primary_hostname'),
                            'similarity_score': similarity_score,
                            'correlation_type': 'cross_tool',
                            'timestamp': datetime.now()
                        })
        
        correlations.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"Found {len(correlations)} high-confidence cross-tool correlations")
        
        return correlations[:100]
    
    def _calculate_similarity(self, asset1: Dict, asset2: Dict) -> float:
        hostname_sim = self._hostname_similarity(
            asset1.get('primary_hostname', ''),
            asset2.get('primary_hostname', '')
        )
        
        ip_sim = self._ip_similarity(
            asset1.get('primary_ip', ''),
            asset2.get('primary_ip', '')
        )
        
        location_sim = 1.0 if asset1.get('location') == asset2.get('location') else 0.0
        
        combined_score = (hostname_sim * 0.5 + ip_sim * 0.3 + location_sim * 0.2)
        
        return combined_score
    
    def _hostname_similarity(self, hostname1: str, hostname2: str) -> float:
        if not hostname1 or not hostname2:
            return 0.0
        
        hostname1 = hostname1.lower().strip()
        hostname2 = hostname2.lower().strip()
        
        if hostname1 == hostname2:
            return 1.0
        
        from difflib import SequenceMatcher
        return SequenceMatcher(None, hostname1, hostname2).ratio()
    
    def _ip_similarity(self, ip1: str, ip2: str) -> float:
        if not ip1 or not ip2:
            return 0.0
        
        if ip1 == ip2:
            return 1.0
        
        try:
            ip1_parts = [int(x) for x in ip1.split('.')]
            ip2_parts = [int(x) for x in ip2.split('.')]
            
            matches = sum(1 for a, b in zip(ip1_parts, ip2_parts) if a == b)
            return matches / 4.0
        except:
            return 0.0
    
    async def detect_shadow_assets(self) -> List[Dict]:
        discovery_stats = self.db.get_asset_discovery_stats()
        
        all_assets = self.db.get_ml_training_data()
        
        known_patterns = all_assets[all_assets['visibility_score'] > 80].to_dict('records')
        unknown_patterns = all_assets[all_assets['visibility_score'] < 50].to_dict('records')
        
        if len(known_patterns) > 10:
            self.shadow_detector.train(known_patterns)
            
            shadow_candidates = self.shadow_detector.detect_shadow_assets(unknown_patterns)
            
            shadow_assets = []
            for asset_data, anomaly_score in shadow_candidates[:20]:
                shadow_assets.append({
                    'asset_id': asset_data.get('asset_id'),
                    'hostname': asset_data.get('primary_hostname'),
                    'ip_address': asset_data.get('primary_ip'),
                    'anomaly_score': anomaly_score,
                    'detection_confidence': min(anomaly_score / 10.0, 1.0),
                    'discovery_recommendation': self._generate_discovery_action(asset_data),
                    'detected_at': datetime.now()
                })
            
            logger.info(f"Detected {len(shadow_assets)} potential shadow assets")
            return shadow_assets
        
        return []
    
    def _generate_discovery_action(self, asset_data: Dict) -> str:
        hostname = asset_data.get('primary_hostname', '')
        ip = asset_data.get('primary_ip', '')
        
        if hostname and not ip:
            return f"Perform DNS resolution for {hostname}"
        elif ip and not hostname:
            return f"Perform reverse DNS lookup for {ip}"
        elif not hostname and not ip:
            return "Network scan required to identify asset"
        else:
            return f"Deploy monitoring agent to {hostname} ({ip})"
    
    async def generate_visibility_insights(self) -> Dict:
        gaps_analysis = await self.analyze_coverage_gaps()
        correlations = await self.correlate_cross_tool_assets()
        shadow_assets = await self.detect_shadow_assets()
        
        coverage_stats = self.db.analyze_cross_tool_correlation()
        
        insights = {
            'summary': {
                'total_coverage_gaps': gaps_analysis['total_gaps'],
                'critical_gaps': gaps_analysis['critical_count'],
                'cross_tool_correlations': len(correlations),
                'shadow_assets_detected': len(shadow_assets),
                'overall_visibility_score': self._calculate_overall_visibility(coverage_stats)
            },
            'coverage_analysis': gaps_analysis,
            'asset_correlations': correlations[:20],
            'shadow_assets': shadow_assets,
            'tool_performance': coverage_stats,
            'recommendations': self._generate_recommendations(gaps_analysis, correlations, shadow_assets),
            'generated_at': datetime.now()
        }
        
        return insights
    
    def _calculate_overall_visibility(self, coverage_stats: Dict) -> float:
        if not coverage_stats:
            return 0.0
        
        total_coverage = sum(stats['avg_coverage'] for stats in coverage_stats.values())
        return min(total_coverage / len(coverage_stats), 100.0)
    
    def _generate_recommendations(
        self, 
        gaps: Dict, 
        correlations: List[Dict], 
        shadow_assets: List[Dict]
    ) -> List[Dict]:
        recommendations = []
        
        if gaps['critical_count'] > 0:
            recommendations.append({
                'priority': 'critical',
                'category': 'coverage_gap',
                'title': f"Address {gaps['critical_count']} critical coverage gaps",
                'description': "Deploy additional monitoring to assets with <50% visibility",
                'estimated_impact': '25% visibility improvement',
                'effort_level': 'high'
            })
        
        if len(correlations) > 10:
            recommendations.append({
                'priority': 'high', 
                'category': 'correlation',
                'title': f"Unify {len(correlations)} correlated assets",
                'description': "Merge duplicate asset records across security tools",
                'estimated_impact': '15% visibility improvement',
                'effort_level': 'medium'
            })
        
        if len(shadow_assets) > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'discovery',
                'title': f"Investigate {len(shadow_assets)} shadow assets",
                'description': "Deploy discovery agents to unmonitored network segments",
                'estimated_impact': '10% visibility improvement', 
                'effort_level': 'low'
            })
        
        return recommendations
