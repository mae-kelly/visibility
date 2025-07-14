import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass

@dataclass
class EntityPair:
    entity1: Dict[str, str]
    entity2: Dict[str, str]
    label: Optional[int] = None

class DITTOEntityResolver(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def serialize_entity(self, entity: Dict[str, str]) -> str:
        serialized_parts = []
        for key, value in entity.items():
            if value and str(value).strip():
                clean_value = str(value).strip()
                serialized_parts.append(f"[COL] {key} [VAL] {clean_value}")
        return " ".join(serialized_parts)
    
    def forward(self, entity_pairs: List[EntityPair]) -> torch.Tensor:
        serialized_pairs = []
        for pair in entity_pairs:
            entity1_str = self.serialize_entity(pair.entity1)
            entity2_str = self.serialize_entity(pair.entity2)
            combined = f"{entity1_str} [SEP] {entity2_str}"
            serialized_pairs.append(combined)
        
        encoding = self.tokenizer(
            serialized_pairs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        outputs = self.bert(**encoding)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_match(self, entity1: Dict[str, str], entity2: Dict[str, str]) -> Tuple[float, bool]:
        pair = EntityPair(entity1, entity2)
        with torch.no_grad():
            logits = self.forward([pair])
            probabilities = torch.softmax(logits, dim=-1)
            match_probability = probabilities[0][1].item()
            is_match = match_probability > 0.5
        return match_probability, is_match

class AssetEntityProcessor:
    def __init__(self):
        self.hostname_patterns = [
            r'^([a-zA-Z0-9-]+)\..*',
            r'^([a-zA-Z0-9-]+)-\d+$',
            r'^([a-zA-Z]+)\d+$'
        ]
        
    def normalize_hostname(self, hostname: str) -> str:
        if not hostname:
            return ""
        
        hostname = hostname.lower().strip()
        hostname = re.sub(r'\.(local|corp|domain)$', '', hostname)
        
        for pattern in self.hostname_patterns:
            match = re.match(pattern, hostname)
            if match:
                return match.group(1)
        return hostname
    
    def normalize_ip(self, ip: str) -> str:
        if not ip:
            return ""
        ip = ip.strip()
        if re.match(r'^\d+\.\d+\.\d+\.\d+$', ip):
            return ip
        return ""
    
    def extract_features(self, asset: Dict) -> Dict[str, str]:
        features = {}
        
        hostname_fields = ['hostname', 'device_hostname', 'host', 'name']
        for field in hostname_fields:
            if field in asset and asset[field]:
                features['hostname'] = self.normalize_hostname(str(asset[field]))
                break
        
        ip_fields = ['ip_address', 'ip', 'device_ip']
        for field in ip_fields:
            if field in asset and asset[field]:
                features['ip'] = self.normalize_ip(str(asset[field]))
                break
        
        if 'class' in asset:
            features['class'] = str(asset['class']).lower()
        if 'environment' in asset:
            features['environment'] = str(asset['environment']).lower()
        if 'location' in asset:
            features['location'] = str(asset['location']).lower()
        if 'make' in asset:
            features['make'] = str(asset['make']).lower()
        if 'os' in asset:
            features['os'] = str(asset['os']).lower()
            
        return features
    
    def create_training_pairs(self, assets: List[Dict], labels: List[int]) -> List[EntityPair]:
        pairs = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                entity1 = self.extract_features(assets[i])
                entity2 = self.extract_features(assets[j])
                label = 1 if labels[i] == labels[j] else 0
                pairs.append(EntityPair(entity1, entity2, label))
        return pairs

class CrossSystemMatcher:
    def __init__(self, model_path: str = None):
        self.ditto_model = DITTOEntityResolver()
        self.entity_processor = AssetEntityProcessor()
        
        if model_path:
            self.ditto_model.load_state_dict(torch.load(model_path))
            self.ditto_model.eval()
    
    def match_assets_across_systems(
        self,
        crowdstrike_assets: List[Dict],
        cmdb_assets: List[Dict],
        splunk_sources: List[Dict],
        chronicle_devices: List[Dict]
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        
        matches = {
            'crowdstrike_cmdb': [],
            'crowdstrike_chronicle': [],
            'cmdb_splunk': [],
            'chronicle_splunk': []
        }
        
        system_pairs = [
            ('crowdstrike_cmdb', crowdstrike_assets, cmdb_assets),
            ('crowdstrike_chronicle', crowdstrike_assets, chronicle_devices),
            ('cmdb_splunk', cmdb_assets, splunk_sources),
            ('chronicle_splunk', chronicle_devices, splunk_sources)
        ]
        
        for pair_name, system1, system2 in system_pairs:
            for i, asset1 in enumerate(system1):
                for j, asset2 in enumerate(system2):
                    entity1 = self.entity_processor.extract_features(asset1)
                    entity2 = self.entity_processor.extract_features(asset2)
                    
                    probability, is_match = self.ditto_model.predict_match(entity1, entity2)
                    
                    if is_match and probability > 0.7:
                        matches[pair_name].append((i, j, probability))
        
        return matches
