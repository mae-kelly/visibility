import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

class VisibilityAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_schema()
    
    def _initialize_schema(self):
        schema_sql = Path("data/schemas/security/visibility_schema.sql").read_text()
        self.conn.execute(schema_sql)
        logger.info("Database schema initialized")
    
    def get_coverage_gaps(self) -> pd.DataFrame:
        query = """
        WITH coverage_analysis AS (
            SELECT 
                a.asset_id,
                a.primary_hostname,
                a.asset_class,
                a.environment,
                COALESCE(c.coverage_percentage, 0) as total_coverage,
                CASE 
                    WHEN c.coverage_percentage < 50 THEN 'critical'
                    WHEN c.coverage_percentage < 75 THEN 'high'
                    WHEN c.coverage_percentage < 90 THEN 'medium'
                    ELSE 'low'
                END as gap_severity
            FROM security_visibility.unified_assets a
            LEFT JOIN security_visibility.coverage_metrics c ON a.asset_id = c.asset_id
        )
        SELECT * FROM coverage_analysis 
        WHERE total_coverage < 95
        ORDER BY total_coverage ASC, gap_severity DESC
        """
        
        return self.conn.execute(query).df()
    
    def analyze_cross_tool_correlation(self) -> Dict[str, float]:
        query = """
        SELECT 
            source_system,
            COUNT(DISTINCT asset_id) as unique_assets,
            AVG(coverage_percentage) as avg_coverage,
            AVG(confidence_score) as avg_confidence
        FROM security_visibility.coverage_metrics
        GROUP BY source_system
        """
        
        results = self.conn.execute(query).df()
        
        correlation_analysis = {}
        for _, row in results.iterrows():
            correlation_analysis[row['source_system']] = {
                'unique_assets': int(row['unique_assets']),
                'avg_coverage': float(row['avg_coverage']),
                'avg_confidence': float(row['avg_confidence'])
            }
        
        return correlation_analysis
    
    def get_asset_discovery_stats(self) -> Dict:
        queries = {
            'total_assets': "SELECT COUNT(*) as count FROM security_visibility.unified_assets",
            'assets_by_source': """
                SELECT 
                    UNNEST(discovery_sources) as source,
                    COUNT(*) as asset_count
                FROM security_visibility.unified_assets
                GROUP BY source
            """,
            'coverage_distribution': """
                SELECT 
                    CASE 
                        WHEN visibility_score >= 90 THEN '90-100%'
                        WHEN visibility_score >= 75 THEN '75-89%'
                        WHEN visibility_score >= 50 THEN '50-74%'
                        ELSE '<50%'
                    END as coverage_range,
                    COUNT(*) as asset_count
                FROM security_visibility.unified_assets
                GROUP BY coverage_range
                ORDER BY coverage_range DESC
            """
        }
        
        stats = {}
        for key, query in queries.items():
            if key == 'total_assets':
                stats[key] = self.conn.execute(query).fetchone()[0]
            else:
                stats[key] = self.conn.execute(query).df().to_dict('records')
        
        return stats
    
    def insert_unified_asset(self, asset_data: Dict) -> str:
        insert_query = """
        INSERT INTO security_visibility.unified_assets 
        (asset_id, primary_hostname, all_hostnames, primary_ip, all_ips, 
         asset_class, asset_type, environment, location, region, country,
         discovery_sources, visibility_score, risk_score, last_seen, 
         data_quality_score, correlation_confidence, ml_cluster_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (asset_id) DO UPDATE SET
            all_hostnames = EXCLUDED.all_hostnames,
            all_ips = EXCLUDED.all_ips,
            discovery_sources = EXCLUDED.discovery_sources,
            visibility_score = EXCLUDED.visibility_score,
            risk_score = EXCLUDED.risk_score,
            last_seen = EXCLUDED.last_seen,
            updated_at = CURRENT_TIMESTAMP
        """
        
        params = [
            asset_data['asset_id'],
            asset_data.get('primary_hostname'),
            asset_data.get('all_hostnames', []),
            asset_data.get('primary_ip'),
            asset_data.get('all_ips', []),
            asset_data.get('asset_class'),
            asset_data.get('asset_type'),
            asset_data.get('environment'),
            asset_data.get('location'),
            asset_data.get('region'),
            asset_data.get('country'),
            asset_data.get('discovery_sources', []),
            asset_data.get('visibility_score', 0.0),
            asset_data.get('risk_score', 0),
            asset_data.get('last_seen', datetime.now()),
            asset_data.get('data_quality_score', 0.0),
            asset_data.get('correlation_confidence', 0.0),
            asset_data.get('ml_cluster_id')
        ]
        
        self.conn.execute(insert_query, params)
        return asset_data['asset_id']
    
    def update_coverage_metrics(self, metrics: List[Dict]):
        for metric in metrics:
            insert_query = """
            INSERT INTO security_visibility.coverage_metrics
            (metric_id, asset_id, source_system, coverage_type, coverage_percentage,
             last_updated, confidence_score, gap_reasons, remediation_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (metric_id) DO UPDATE SET
                coverage_percentage = EXCLUDED.coverage_percentage,
                last_updated = EXCLUDED.last_updated,
                confidence_score = EXCLUDED.confidence_score,
                gap_reasons = EXCLUDED.gap_reasons,
                remediation_actions = EXCLUDED.remediation_actions
            """
            
            params = [
                metric['metric_id'],
                metric['asset_id'],
                metric['source_system'],
                metric['coverage_type'],
                metric['coverage_percentage'],
                metric.get('last_updated', datetime.now()),
                metric.get('confidence_score', 0.0),
                metric.get('gap_reasons', []),
                metric.get('remediation_actions', [])
            ]
            
            self.conn.execute(insert_query, params)
    
    def get_ml_training_data(self, days_back: int = 30) -> pd.DataFrame:
        query = f"""
        SELECT 
            a.*,
            c.coverage_percentage,
            c.confidence_score,
            c.source_system
        FROM security_visibility.unified_assets a
        JOIN security_visibility.coverage_metrics c ON a.asset_id = c.asset_id
        WHERE c.last_updated >= CURRENT_DATE - INTERVAL '{days_back}' DAY
        """
        
        return self.conn.execute(query).df()
    
    def close(self):
        self.conn.close()
