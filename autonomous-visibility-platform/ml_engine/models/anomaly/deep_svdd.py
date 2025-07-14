import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler
import logging

class DeepSVDDNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], rep_dim: int = 32):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, rep_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DeepSVDD:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        rep_dim: int = 32,
        nu: float = 0.1,
        device: str = 'cpu'
    ):
        self.device = device
        self.nu = nu
        self.rep_dim = rep_dim
        
        self.network = DeepSVDDNetwork(input_dim, hidden_dims, rep_dim).to(device)
        self.center = None
        self.R = None
        self.scaler = StandardScaler()
        
    def train(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-6
    ) -> Dict[str, List[float]]:
        
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.network.train()
        optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        
        self._init_center(X_tensor)
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                
                outputs = self.network(batch)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                loss = torch.mean(dist)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            if epoch % 20 == 0:
                logging.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self._set_radius(X_tensor)
        self.network.eval()
        
        return history
    
    def _init_center(self, X: torch.Tensor):
        with torch.no_grad():
            outputs = self.network(X)
            self.center = torch.mean(outputs, dim=0)
    
    def _set_radius(self, X: torch.Tensor):
        with torch.no_grad():
            outputs = self.network(X)
            distances = torch.sum((outputs - self.center) ** 2, dim=1)
            self.R = torch.quantile(distances, 1 - self.nu)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(X_tensor)
            distances = torch.sum((outputs - self.center) ** 2, dim=1)
            
            scores = distances.cpu().numpy()
            predictions = (distances > self.R).cpu().numpy().astype(int)
        
        return predictions, scores

class ShadowAssetDetector:
    def __init__(self, model_path: str = None):
        self.deep_svdd = None
        self.feature_extractors = {
            'network_behavior': self._extract_network_features,
            'asset_metadata': self._extract_metadata_features,
            'temporal_patterns': self._extract_temporal_features
        }
        
        if model_path:
            self.load_model(model_path)
    
    def train_detector(
        self,
        known_assets: List[Dict],
        epochs: int = 100,
        batch_size: int = 128
    ) -> Dict[str, List[float]]:
        
        features = self._extract_all_features(known_assets)
        input_dim = features.shape[1]
        
        self.deep_svdd = DeepSVDD(input_dim=input_dim)
        history = self.deep_svdd.train(features, epochs=epochs, batch_size=batch_size)
        
        return history
    
    def detect_shadow_assets(
        self,
        candidate_assets: List[Dict],
        threshold: float = 0.5
    ) -> Tuple[List[int], List[float]]:
        
        if self.deep_svdd is None:
            raise ValueError("Model not trained. Call train_detector first.")
        
        features = self._extract_all_features(candidate_assets)
        predictions, scores = self.deep_svdd.predict(features)
        
        shadow_indices = []
        shadow_scores = []
        
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == 1:
                shadow_indices.append(i)
                shadow_scores.append(score)
        
        return shadow_indices, shadow_scores
    
    def _extract_all_features(self, assets: List[Dict]) -> np.ndarray:
        all_features = []
        
        for asset in assets:
            features = []
            
            for extractor_name, extractor_func in self.feature_extractors.items():
                extracted = extractor_func(asset)
                features.extend(extracted)
            
            all_features.append(features)
        
        return np.array(all_features, dtype=np.float32)
    
    def _extract_network_features(self, asset: Dict) -> List[float]:
        features = []
        
        ip = asset.get('ip_address', '')
        if ip:
            ip_parts = ip.split('.')
            if len(ip_parts) == 4:
                features.extend([float(part) / 255.0 for part in ip_parts])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        port_count = len(asset.get('open_ports', []))
        features.append(min(port_count / 100.0, 1.0))
        
        connection_count = asset.get('connection_count', 0)
        features.append(min(connection_count / 1000.0, 1.0))
        
        return features
    
    def _extract_metadata_features(self, asset: Dict) -> List[float]:
        features = []
        
        hostname = asset.get('hostname', '')
        features.append(len(hostname) / 50.0)
        
        has_domain = 1.0 if '.' in hostname else 0.0
        features.append(has_domain)
        
        environment_mapping = {'prod': 1.0, 'staging': 0.7, 'test': 0.5, 'dev': 0.3}
        env = asset.get('environment', '').lower()
        features.append(environment_mapping.get(env, 0.0))
        
        class_mapping = {'server': 1.0, 'workstation': 0.8, 'network': 0.6, 'mobile': 0.4}
        asset_class = asset.get('class', '').lower()
        features.append(class_mapping.get(asset_class, 0.0))
        
        return features
    
    def _extract_temporal_features(self, asset: Dict) -> List[float]:
        features = []
        
        last_seen = asset.get('last_seen_hours_ago', 0)
        features.append(min(last_seen / 168.0, 1.0))
        
        discovery_age = asset.get('discovery_age_days', 0)
        features.append(min(discovery_age / 365.0, 1.0))
        
        activity_score = asset.get('activity_score', 0.0)
        features.append(min(activity_score, 1.0))
        
        return features
    
    def save_model(self, path: str):
        if self.deep_svdd:
            torch.save({
                'model_state': self.deep_svdd.network.state_dict(),
                'center': self.deep_svdd.center,
                'radius': self.deep_svdd.R,
                'scaler': self.deep_svdd.scaler
            }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        input_dim = list(checkpoint['model_state'].values())[0].shape[1]
        
        self.deep_svdd = DeepSVDD(input_dim=input_dim)
        self.deep_svdd.network.load_state_dict(checkpoint['model_state'])
        self.deep_svdd.center = checkpoint['center']
        self.deep_svdd.R = checkpoint['radius']
        self.deep_svdd.scaler = checkpoint['scaler']
        self.deep_svdd.network.eval()
