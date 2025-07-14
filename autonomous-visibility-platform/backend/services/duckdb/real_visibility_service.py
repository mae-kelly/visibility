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
