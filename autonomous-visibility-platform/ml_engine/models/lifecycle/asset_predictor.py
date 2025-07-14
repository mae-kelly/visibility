import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class AssetLifecycleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        final_hidden = attn_out[:, -1, :]
        
        output = self.classifier(final_hidden)
        
        return output

class AssetLifecyclePredictor:
    def __init__(self, sequence_length: int = 30):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
        self.feature_columns = [
            'coverage_percentage', 'log_volume', 'connection_count',
            'service_count', 'vulnerability_count', 'patch_level',
            'activity_score', 'risk_score'
        ]
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(data['visibility_gap'].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        historical_data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        X, y = self._prepare_sequences(historical_data)
        
        self.model = AssetLifecycleLSTM(
            input_size=len(self.feature_columns),
            hidden_size=128,
            num_layers=2
        ).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y).unsqueeze(1)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict_visibility_gaps(
        self,
        asset_data: pd.DataFrame,
        prediction_horizon: int = 7
    ) -> List[Dict]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for asset_id in asset_data['asset_id'].unique():
                asset_history = asset_data[asset_data['asset_id'] == asset_id].copy()
                asset_history = asset_history.sort_values('timestamp')
                
                if len(asset_history) < self.sequence_length:
                    continue
                
                recent_data = asset_history.tail(self.sequence_length)
                scaled_sequence = self.scaler.transform(recent_data[self.feature_columns])
                
                sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)
                
                gap_probability = self.model(sequence_tensor).cpu().item()
                
                predictions.append({
                    'asset_id': asset_id,
                    'gap_probability': gap_probability,
                    'risk_level': self._categorize_risk(gap_probability),
                    'prediction_date': datetime.now(),
                    'confidence': self._calculate_confidence(asset_history)
                })
        
        return sorted(predictions, key=lambda x: x['gap_probability'], reverse=True)
    
    def _categorize_risk(self, probability: float) -> str:
        if probability > 0.8:
            return 'critical'
        elif probability > 0.6:
            return 'high'
        elif probability > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, history: pd.DataFrame) -> float:
        data_completeness = history[self.feature_columns].notna().mean().mean()
        recency_score = min(1.0, len(history) / 100.0)
        return (data_completeness + recency_score) / 2
