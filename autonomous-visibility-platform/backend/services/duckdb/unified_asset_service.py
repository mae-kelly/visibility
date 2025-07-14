import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

class UnifiedAssetService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._setup_schemas()
    
    def _setup_schemas(self):
        schema_sql = """
        CREATE SCHEMA IF NOT EXISTS unified;
        
        CREATE TABLE IF NOT EXISTS unified.master_assets (
            asset_id VARCHAR PRIMARY KEY,
            primary_hostname VARCHAR,
            all_hostnames VARCHAR[],
            primary_ip VARCHAR,
            all_ips VARCHAR[],
            asset_class VARCHAR,
            environment VARCHAR,
            location VARCHAR,
            region VARCHAR,
            country VARCHAR,
            source_systems VARCHAR[],
            visibility_score DECIMAL(5,2),
            risk_score INTEGER,
            last_seen TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS unified.correlation_mappings (
            mapping_id VARCHAR PRIMARY KEY,
            asset_id VARCHAR,
            source_system VARCHAR,
            source_id VARCHAR,
            confidence_score DECIMAL(5,2),
            correlation_method VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS unified.visibility_gaps (
            gap_id VARCHAR PRIMARY KEY,
            asset_id VARCHAR,
            gap_type VARCHAR,
            severity VARCHAR,
            description TEXT,
            recommended_action TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(schema_sql)
    
    def get_source_data(self) -> Dict[str, pd.DataFrame]:
        try:
            sources = {}
            
            tables = [
                ('crowdstrike', 'SELECT * FROM crowdstrike'),
                ('cmdb', 'SELECT * FROM cmdb'),
                ('splunk', 'SELECT * FROM splunk'),
                ('chronicle', 'SELECT * FROM chronicle'),
                ('all_sources', 'SELECT * FROM all_sources')
            ]
            
            for table_name, query in tables:
                try:
                    df = self.conn.execute(query).df()
                    sources[table_name] = df
                    logging.info(f"Loaded {len(df)} records from {table_name}")
                except Exception as e:
                    logging.warning(f"Could not load {table_name}: {e}")
                    sources[table_name] = pd.DataFrame()
            
            return sources
        except Exception as e:
            logging.error(f"Error loading source data: {e}")
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict[str, pd.DataFrame]:
        logging.info("Generating mock data for development...")
        
        mock_crowdstrike = pd.DataFrame({
            'crowdstrike_device_hostname': [
                'web-srv-01.corp.com', 'db-srv-02.corp.com', 'app-srv-03.corp.com',
                'file-srv-01.corp.com', 'mail-srv-01.corp.com'
            ],
            'ip_address': ['10.0.1.10', '10.0.1.20', '10.0.1.30', '10.0.1.40', '10.0.1.50'],
            'crowdstrike_agent_health': ['Healthy', 'Healthy', 'Warning', 'Healthy', 'Critical'],
            'last_seen': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        
        mock_cmdb = pd.DataFrame({
            'hostname': ['web-srv-01', 'db-srv-02', 'app-srv-03', 'file-srv-01', 'unknown-srv-01'],
            'ip_address': ['10.0.1.10', '10.0.1.20', '10.0.1.30', '10.0.1.40', '10.0.1.60'],
            'class': ['Server', 'Database', 'Application', 'FileServer', 'Unknown'],
            'environment': ['Production', 'Production', 'Staging', 'Production', 'Development'],
            'location': ['DC1', 'DC1', 'DC2', 'DC1', 'DC3']
        })
        
        mock_splunk = pd.DataFrame({
            'host': ['web-srv-01', 'db-srv-02', 'app-srv-03', 'proxy-srv-01', 'fw-01'],
            'splunk_host': ['10.0.1.10', '10.0.1.20', '10.0.1.30', '10.0.1.70', '10.0.1.1'],
            'source': ['webserver', 'database', 'application', 'proxy', 'firewall'],
            'log_count': [15420, 8934, 12034, 45231, 2341]
        })
        
        mock_chronicle = pd.DataFrame({
            'chronicle_device_hostname': [
                'web-srv-01.corp.com', 'db-srv-02.corp.com', 'workstation-01.corp.com',
                'laptop-user01.corp.com', 'mobile-device-01'
            ],
            'chronicle_ip_address': ['10.0.1.10', '10.0.1.20', '10.0.2.15', '192.168.1.100', '172.16.1.50'],
            'device_type': ['Server', 'Server', 'Workstation', 'Laptop', 'Mobile'],
            'log_type': ['Security', 'Security', 'Endpoint', 'Endpoint', 'Mobile']
        })
        
        return {
            'crowdstrike': mock_crowdstrike,
            'cmdb': mock_cmdb,
            'splunk': mock_splunk,
            'chronicle': mock_chronicle,
            'all_sources': pd.concat([mock_crowdstrike, mock_cmdb], ignore_index=True)
        }
    
    def correlate_assets(self, correlation_results: Dict) -> List[Dict]:
        unified_assets = []
        asset_id_counter = 1
        
        source_data = self.get_source_data()
        
        processed_indices = {system: set() for system in source_data.keys()}
        
        for correlation_type, matches in correlation_results.items():
            for match in matches:
                src_idx, dst_idx, confidence = match
                
                asset_id = f"ASSET-{asset_id_counter:06d}"
                asset_id_counter += 1
                
                systems = correlation_type.split('_')
                src_system, dst_system = systems[0], systems[1]
                
                if (src_idx not in processed_indices[src_system] or 
                    dst_idx not in processed_indices[dst_system]):
                    
                    unified_asset = self._merge_asset_data(
                        source_data[src_system].iloc[src_idx].to_dict(),
                        source_data[dst_system].iloc[dst_idx].to_dict(),
                        asset_id,
                        [src_system, dst_system],
                        confidence
                    )
                    
                    unified_assets.append(unified_asset)
                    processed_indices[src_system].add(src_idx)
                    processed_indices[dst_system].add(dst_idx)
        
        for system_name, df in source_data.items():
            unmatched_indices = set(range(len(df))) - processed_indices[system_name]
            
            for idx in unmatched_indices:
                asset_id = f"ASSET-{asset_id_counter:06d}"
                asset_id_counter += 1
                
                row = df.iloc[idx].to_dict()
                unified_asset = self._create_single_source_asset(row, asset_id, system_name)
                unified_assets.append(unified_asset)
        
        self._store_unified_assets(unified_assets)
        return unified_assets
    
    def _merge_asset_data(
        self, 
        asset1: Dict, 
        asset2: Dict, 
        asset_id: str, 
        source_systems: List[str], 
        confidence: float
    ) -> Dict:
        
        hostnames = []
        ips = []
        
        hostname_fields = ['hostname', 'crowdstrike_device_hostname', 'chronicle_device_hostname', 'host']
        ip_fields = ['ip_address', 'chronicle_ip_address', 'splunk_host']
        
        for asset in [asset1, asset2]:
            for field in hostname_fields:
                if field in asset and asset[field]:
                    hostname = str(asset[field]).replace('.corp.com', '').lower()
                    if hostname not in hostnames:
                        hostnames.append(hostname)
            
            for field in ip_fields:
                if field in asset and asset[field]:
                    ip = str(asset[field])
                    if ip not in ips and self._is_valid_ip(ip):
                        ips.append(ip)
        
        unified_asset = {
            'asset_id': asset_id,
            'primary_hostname': hostnames[0] if hostnames else 'unknown',
            'all_hostnames': hostnames,
            'primary_ip': ips[0] if ips else 'unknown',
            'all_ips': ips,
            'asset_class': asset1.get('class') or asset2.get('class') or asset1.get('device_type') or 'Unknown',
            'environment': asset1.get('environment') or asset2.get('environment') or 'Unknown',
            'location': asset1.get('location') or asset2.get('location') or 'Unknown',
            'region': asset1.get('region') or asset2.get('region') or 'Unknown',
            'country': asset1.get('country') or asset2.get('country') or 'US',
            'source_systems': source_systems,
            'visibility_score': self._calculate_visibility_score(source_systems, confidence),
            'risk_score': self._calculate_risk_score(asset1, asset2),
            'last_seen': datetime.now(),
            'correlation_confidence': confidence
        }
        
        return unified_asset
    
    def _create_single_source_asset(self, asset_data: Dict, asset_id: str, source_system: str) -> Dict:
        hostname_fields = ['hostname', 'crowdstrike_device_hostname', 'chronicle_device_hostname', 'host']
        ip_fields = ['ip_address', 'chronicle_ip_address', 'splunk_host']
        
        hostname = 'unknown'
        for field in hostname_fields:
            if field in asset_data and asset_data[field]:
                hostname = str(asset_data[field]).replace('.corp.com', '').lower()
                break
        
        ip = 'unknown'
        for field in ip_fields:
            if field in asset_data and asset_data[field]:
                potential_ip = str(asset_data[field])
                if self._is_valid_ip(potential_ip):
                    ip = potential_ip
                    break
        
        return {
            'asset_id': asset_id,
            'primary_hostname': hostname,
            'all_hostnames': [hostname] if hostname != 'unknown' else [],
            'primary_ip': ip,
            'all_ips': [ip] if ip != 'unknown' else [],
            'asset_class': asset_data.get('class') or asset_data.get('device_type') or 'Unknown',
            'environment': asset_data.get('environment') or 'Unknown',
            'location': asset_data.get('location') or 'Unknown',
            'region': asset_data.get('region') or 'Unknown',
            'country': asset_data.get('country') or 'US',
            'source_systems': [source_system],
            'visibility_score': 25.0,
            'risk_score': 70,
            'last_seen': datetime.now(),
            'correlation_confidence': 1.0
        }
    
    def _is_valid_ip(self, ip_str: str) -> bool:
        import re
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(pattern, ip_str):
            parts = ip_str.split('.')
            return all(0 <= int(part) <= 255 for part in parts)
        return False
    
    def _calculate_visibility_score(self, source_systems: List[str], confidence: float) -> float:
        base_score = len(source_systems) * 25.0
        confidence_bonus = confidence * 20.0
        return min(base_score + confidence_bonus, 100.0)
    
    def _calculate_risk_score(self, asset1: Dict, asset2: Dict) -> int:
        risk_factors = []
        
        if asset1.get('environment') == 'Production' or asset2.get('environment') == 'Production':
            risk_factors.append(30)
        
        if asset1.get('class') == 'Server' or asset2.get('class') == 'Server':
            risk_factors.append(25)
        
        health_status = asset1.get('crowdstrike_agent_health') or asset2.get('crowdstrike_agent_health')
        if health_status == 'Critical':
            risk_factors.append(40)
        elif health_status == 'Warning':
            risk_factors.append(20)
        
        return min(sum(risk_factors), 100) if risk_factors else 30
    
    def _store_unified_assets(self, unified_assets: List[Dict]):
        if not unified_assets:
            return
        
        self.conn.execute("DELETE FROM unified.master_assets")
        
        df = pd.DataFrame(unified_assets)
        df['all_hostnames'] = df['all_hostnames'].apply(lambda x: json.dumps(x) if x else '[]')
        df['all_ips'] = df['all_ips'].apply(lambda x: json.dumps(x) if x else '[]')
        df['source_systems'] = df['source_systems'].apply(lambda x: json.dumps(x) if x else '[]')
        
        self.conn.register('unified_assets_df', df)
        self.conn.execute("""
            INSERT INTO unified.master_assets 
            SELECT * FROM unified_assets_df
        """)
        
        logging.info(f"Stored {len(unified_assets)} unified assets")
    
    def get_coverage_analysis(self) -> Dict:
        query = """
        SELECT 
            COUNT(*) as total_assets,
            AVG(visibility_score) as avg_visibility,
            COUNT(CASE WHEN visibility_score >= 75 THEN 1 END) as high_visibility_assets,
            COUNT(CASE WHEN visibility_score < 50 THEN 1 END) as low_visibility_assets,
            COUNT(DISTINCT json_extract_string(source_systems, '$[0]')) as unique_sources
        FROM unified.master_assets
        """
        
        result = self.conn.execute(query).fetchone()
        
        return {
            'total_assets': result[0] if result[0] else 0,
            'average_visibility_score': round(result[1], 2) if result[1] else 0,
            'high_visibility_assets': result[2] if result[2] else 0,
            'low_visibility_assets': result[3] if result[3] else 0,
            'unique_source_systems': result[4] if result[4] else 0,
            'visibility_percentage': round((result[1] or 0), 2)
        }
    
    def get_gap_analysis(self) -> List[Dict]:
        gaps = []
        
        coverage_query = """
        SELECT 
            asset_id,
            primary_hostname,
            visibility_score,
            source_systems,
            asset_class,
            environment
        FROM unified.master_assets 
        WHERE visibility_score < 75
        ORDER BY visibility_score ASC
        """
        
        results = self.conn.execute(coverage_query).fetchall()
        
        for row in results:
            asset_id, hostname, vis_score, sources, asset_class, environment = row
            
            source_list = json.loads(sources) if sources else []
            missing_sources = []
            
            if 'crowdstrike' not in source_list:
                missing_sources.append('CrowdStrike agent deployment needed')
            if 'chronicle' not in source_list:
                missing_sources.append('Chronicle logging configuration required')
            if 'splunk' not in source_list:
                missing_sources.append('Splunk forwarder setup needed')
            if 'cmdb' not in source_list:
                missing_sources.append('CMDB registration required')
            
            gaps.append({
                'asset_id': asset_id,
                'hostname': hostname,
                'visibility_score': vis_score,
                'gap_severity': 'High' if vis_score < 50 else 'Medium',
                'missing_sources': missing_sources,
                'recommended_actions': self._generate_recommendations(missing_sources, asset_class, environment)
            })
        
        return gaps
    
    def _generate_recommendations(self, missing_sources: List[str], asset_class: str, environment: str) -> List[str]:
        recommendations = []
        
        if any('CrowdStrike' in source for source in missing_sources):
            if environment == 'Production':
                recommendations.append('HIGH PRIORITY: Deploy CrowdStrike agent immediately')
            else:
                recommendations.append('Deploy CrowdStrike agent for endpoint protection')
        
        if any('Chronicle' in source for source in missing_sources):
            recommendations.append('Configure Chronicle log forwarding for security analytics')
        
        if any('Splunk' in source for source in missing_sources):
            recommendations.append('Install Splunk Universal Forwarder for log aggregation')
        
        if any('CMDB' in source for source in missing_sources):
            recommendations.append('Register asset in CMDB with accurate metadata')
        
        if asset_class == 'Server' and environment == 'Production':
            recommendations.insert(0, 'CRITICAL: Production server requires full monitoring stack')
        
        return recommendations
    
    def close(self):
        if self.conn:
            self.conn.close()
