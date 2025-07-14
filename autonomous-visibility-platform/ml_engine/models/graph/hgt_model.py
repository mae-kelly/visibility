import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Tuple, Optional

class HeterogeneousGraphTransformer(nn.Module):
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[str], 
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_encoders = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) 
            for node_type in node_types
        })
        
        self.hgt_layers = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(node_types, edge_types),
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.correlation_head = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_dict, edge_index_dict, node_pairs):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_encoders[node_type](x)
        
        for layer in self.hgt_layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        correlations = []
        for (node1_type, node1_idx), (node2_type, node2_idx) in node_pairs:
            emb1 = x_dict[node1_type][node1_idx]
            emb2 = x_dict[node2_type][node2_idx]
            pair_emb = torch.cat([emb1, emb2], dim=-1)
            correlation = self.correlation_head(pair_emb)
            correlations.append(correlation)
        
        return torch.stack(correlations) if correlations else torch.tensor([])

class AssetCorrelationEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_types = ['crowdstrike_host', 'splunk_ip', 'chronicle_domain', 'cmdb_device']
        self.edge_types = ['communicates_with', 'belongs_to', 'manages', 'discovered_by']
        
        self.model = HeterogeneousGraphTransformer(
            node_types=self.node_types,
            edge_types=self.edge_types
        ).to(self.device)
        
        if model_path:
            self.load_model(model_path)
    
    def correlate_assets(self, asset_data: Dict, threshold: float = 0.8) -> List[Tuple]:
        self.model.eval()
        with torch.no_grad():
            x_dict, edge_index_dict, node_pairs = self._prepare_graph_data(asset_data)
            correlations = self.model(x_dict, edge_index_dict, node_pairs)
            
            high_confidence_pairs = []
            for i, correlation in enumerate(correlations):
                if correlation.item() > threshold:
                    high_confidence_pairs.append((node_pairs[i], correlation.item()))
            
            return high_confidence_pairs
    
    def _prepare_graph_data(self, asset_data: Dict):
        pass
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
