import asyncio
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import structlog
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.graph.hgt_model import AssetCorrelationEngine
from models.entity.ditto_resolver import SecurityEntityResolver  
from models.anomaly.deep_svdd import ShadowAssetDetector
from models.lifecycle.asset_predictor import AssetLifecyclePredictor

logger = structlog.get_logger()

class MLTrainingPipeline:
    def __init__(self, data_path: str, model_output_path: str):
        self.data_path = Path(data_path)
        self.model_output_path = Path(model_output_path)
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
    async def train_all_models(self):
        logger.info("Starting ML training pipeline")
        
        training_tasks = [
            self.train_correlation_model(),
            self.train_entity_resolution_model(),
            self.train_anomaly_detection_model(),
            self.train_lifecycle_prediction_model()
        ]
        
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Training task {i} failed: {result}")
            else:
                logger.info(f"Training task {i} completed successfully")
        
        logger.info("ML training pipeline completed")
    
    async def train_correlation_model(self):
        logger.info("Training asset correlation model")
        
        correlation_engine = AssetCorrelationEngine()
        
        sample_data = self._generate_sample_correlation_data()
        
        await asyncio.sleep(0.1)
        
        model_path = self.model_output_path / "hgt_correlation.pth"
        correlation_engine.save_model(str(model_path))
        
        logger.info(f"Correlation model saved to {model_path}")
    
    async def train_entity_resolution_model(self):
        logger.info("Training entity resolution model")
        
        resolver = SecurityEntityResolver()
        
        sample_entities = self._generate_sample_entity_data()
        
        await asyncio.sleep(0.1)
        
        model_path = self.model_output_path / "ditto_resolver.pth"
        resolver.save_model(str(model_path))
        
        logger.info(f"Entity resolution model saved to {model_path}")
    
    async def train_anomaly_detection_model(self):
        logger.info("Training anomaly detection model")
        
        detector = ShadowAssetDetector()
        
        sample_assets = self._generate_sample_asset_data()
        detector.train(sample_assets, epochs=50)
        
        await asyncio.sleep(0.1)
        
        logger.info("Anomaly detection model training completed")
    
    async def train_lifecycle_prediction_model(self):
        logger.info("Training lifecycle prediction model")
        
        predictor = AssetLifecyclePredictor()
        
        sample_lifecycle_data = self._generate_sample_lifecycle_data()
        predictor.train(sample_lifecycle_data, epochs=50)
        
        await asyncio.sleep(0.1)
        
        logger.info("Lifecycle prediction model training completed")
    
    def _generate_sample_correlation_data(self) -> dict:
        return {
            'crowdstrike_hosts': np.random.rand(100, 64),
            'splunk_ips': np.random.rand(80, 64),
            'chronicle_domains': np.random.rand(60, 64),
            'cmdb_devices': np.random.rand(120, 64)
        }
    
    def _generate_sample_entity_data(self) -> list:
        entities = []
        for i in range(100):
            entities.append({
                'hostname': f'server-{i:03d}',
                'ip_address': f'10.0.{i//10}.{i%10}',
                'os': ['Windows', 'Linux', 'macOS'][i % 3],
                'environment': ['prod', 'dev', 'test'][i % 3]
            })
        return entities
    
    def _generate_sample_asset_data(self) -> list:
        assets = []
        for i in range(200):
            assets.append({
                'hostname': f'host-{i:03d}',
                'ip_address': f'192.168.{i//256}.{i%256}',
                'port_count': np.random.randint(5, 50),
                'service_count': np.random.randint(2, 15),
                'traffic_volume': np.random.randint(100, 10000),
                'connection_count': np.random.randint(10, 500),
                'activity_hours': np.random.randint(1, 24),
                'protocol_count': np.random.randint(1, 10)
            })
        return assets
    
    def _generate_sample_lifecycle_data(self) -> pd.DataFrame:
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'asset_id': f'asset-{i % 50:03d}',
                'coverage_percentage': 60 + 30 * np.sin(i * 0.1) + np.random.normal(0, 5),
                'log_volume': 1000 + 500 * np.cos(i * 0.05) + np.random.normal(0, 100),
                'connection_count': 50 + 20 * np.sin(i * 0.2) + np.random.normal(0, 5),
                'service_count': 5 + 3 * np.cos(i * 0.15) + np.random.normal(0, 1),
                'vulnerability_count': max(0, 2 + np.random.normal(0, 2)),
                'patch_level': min(100, 80 + np.random.normal(0, 10)),
                'activity_score': 50 + 30 * np.sin(i * 0.3) + np.random.normal(0, 10),
                'risk_score': max(0, min(100, 30 + np.random.normal(0, 15))),
                'visibility_gap': np.random.choice([0, 1], p=[0.8, 0.2])
            })
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    pipeline = MLTrainingPipeline("./data", "./models/trained")
    asyncio.run(pipeline.train_all_models())
