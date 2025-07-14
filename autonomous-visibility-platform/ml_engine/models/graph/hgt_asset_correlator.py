import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Tuple, Optional

class HGTAssetCorrelator(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        
        self.node_embeddings = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim)
            for node_type in node_types
        })
        
        self.hgt_layers = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.correlation_head = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, 64),
            nn.ReLU(),
            Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: self.node_embeddings[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        for layer in self.hgt_layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            
        return x_dict
    
    def predict_correlation(self, node1_emb: torch.Tensor, node2_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([node1_emb, node2_emb], dim=-1)
        return self.correlation_head(combined)

class AssetGraphBuilder:
    def __init__(self):
        self.node_features = {
            'crowdstrike_asset': ['hostname', 'ip', 'os_type', 'agent_version'],
            'cmdb_asset': ['hostname', 'ip', 'class', 'environment'],
            'splunk_source': ['host', 'source', 'sourcetype', 'index'],
            'chronicle_device': ['hostname', 'ip', 'device_type', 'location']
        }
        
    def build_hetero_graph(self, asset_data: Dict) -> HeteroData:
        data = HeteroData()
        
        for node_type, features in self.node_features.items():
            if node_type in asset_data:
                node_data = asset_data[node_type]
                data[node_type].x = self._encode_features(node_data, features)
                data[node_type].num_nodes = len(node_data)
        
        edge_mappings = {
            ('crowdstrike_asset', 'correlates_with', 'cmdb_asset'): self._find_hostname_matches,
            ('cmdb_asset', 'maps_to', 'splunk_source'): self._find_ip_matches,
            ('chronicle_device', 'same_as', 'crowdstrike_asset'): self._find_device_matches,
        }
        
        for edge_type, match_func in edge_mappings.items():
            src_type, rel, dst_type = edge_type
            if src_type in asset_data and dst_type in asset_data:
                edge_index = match_func(asset_data[src_type], asset_data[dst_type])
                data[edge_type].edge_index = edge_index
                
        return data
    
    def _encode_features(self, node_data: List[Dict], features: List[str]) -> torch.Tensor:
        encoded = []
        for item in node_data:
            feature_vec = []
            for feat in features:
                if feat in item:
                    if isinstance(item[feat], str):
                        feature_vec.append(hash(item[feat]) % 10000 / 10000.0)
                    else:
                        feature_vec.append(float(item[feat]))
                else:
                    feature_vec.append(0.0)
            encoded.append(feature_vec)
        return torch.tensor(encoded, dtype=torch.float32)
    
    def _find_hostname_matches(self, src_data: List[Dict], dst_data: List[Dict]) -> torch.Tensor:
        edges = []
        for i, src in enumerate(src_data):
            for j, dst in enumerate(dst_data):
                if self._hostname_similarity(src.get('hostname', ''), dst.get('hostname', '')) > 0.8:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    
    def _find_ip_matches(self, src_data: List[Dict], dst_data: List[Dict]) -> torch.Tensor:
        edges = []
        for i, src in enumerate(src_data):
            for j, dst in enumerate(dst_data):
                if src.get('ip') == dst.get('host'):
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    
    def _find_device_matches(self, src_data: List[Dict], dst_data: List[Dict]) -> torch.Tensor:
        edges = []
        for i, src in enumerate(src_data):
            for j, dst in enumerate(dst_data):
                hostname_match = self._hostname_similarity(src.get('hostname', ''), dst.get('hostname', '')) > 0.9
                ip_match = src.get('ip') == dst.get('ip')
                if hostname_match or ip_match:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    
    def _hostname_similarity(self, h1: str, h2: str) -> float:
        if not h1 or not h2:
            return 0.0
        h1, h2 = h1.lower(), h2.lower()
        if h1 == h2:
            return 1.0
        
        from difflib import SequenceMatcher
        return SequenceMatcher(None, h1, h2).ratio()
