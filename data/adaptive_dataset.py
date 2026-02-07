"""
Dataset loader for adaptive routing
Loads JSONL dataset and prepares for PyTorch training
"""

import json
import torch
from torch.utils.data import Dataset


NETWORK_MAP = {"wifi": 0, "4g": 1, "5g": 2}


class AdaptiveRoutingDataset(Dataset):
    """PyTorch dataset for adaptive routing"""
    
    def __init__(self, jsonl_path):
        """Load dataset from JSONL file"""
        self.samples = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract query text
        query_text = sample['query_text']
        
        # Extract device features: [battery, cpu, ram_normalized, network_encoded, privacy_risk]
        device = sample['device']
        device_features = torch.tensor([
            device['battery_level'],
            device['cpu_load'],
            device['ram_mb'] / 8192.0,  # Normalize RAM (max 8192)
            NETWORK_MAP[device['network']],  # Encode network type
            sample['privacy_risk']
        ], dtype=torch.float32)
        
        # Extract label (optimal route)
        optimal_route = sample['label']  # 0=local, 1=hybrid, 2=cloud
        
        # Extract costs and quality scores for each route
        routes = sample['routes']
        route_order = ['local', 'hybrid', 'cloud']
        privacy_costs = torch.tensor([routes[r]['privacy_leakage'] for r in route_order], dtype=torch.float32)
        energy_costs = torch.tensor([routes[r]['energy_cost'] for r in route_order], dtype=torch.float32)
        quality_scores = torch.tensor([routes[r]['task_quality'] for r in route_order], dtype=torch.float32)
        
        return {
            'query_text': query_text,
            'device_features': device_features,
            'optimal_route': optimal_route,
            'privacy_costs': privacy_costs,
            'energy_costs': energy_costs,
            'quality_scores': quality_scores
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    queries = [item['query_text'] for item in batch]
    device_features = torch.stack([item['device_features'] for item in batch])
    optimal_routes = torch.tensor([item['optimal_route'] for item in batch], dtype=torch.long)
    privacy_costs = torch.stack([item['privacy_costs'] for item in batch])
    energy_costs = torch.stack([item['energy_costs'] for item in batch])
    quality_scores = torch.stack([item['quality_scores'] for item in batch])
    
    return {
        'queries': queries,
        'device_features': device_features,
        'optimal_routes': optimal_routes,
        'privacy_costs': privacy_costs,
        'energy_costs': energy_costs,
        'quality_scores': quality_scores
    }
