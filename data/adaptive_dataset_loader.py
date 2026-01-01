"""
Dataset loader for adaptive routing training
Converts JSONL format to PyTorch Dataset
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class AdaptiveRoutingDataset(Dataset):
    """
    PyTorch Dataset for adaptive routing decisions
    
    Maps routes to action indices:
    - local = 0
    - hybrid = 1  
    - cloud = 2
    """
    
    def __init__(self, jsonl_path: str):
        self.data = []
        self.route_to_idx = {'local': 0, 'hybrid': 1, 'cloud': 2}
        
        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"✓ Loaded {len(self.data)} samples")
        
        # Print distribution
        route_counts = {}
        for item in self.data:
            route = item['optimal_route']
            route_counts[route] = route_counts.get(route, 0) + 1
        
        print(f"  Distribution: {route_counts}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract route costs (privacy, energy, quality)
        routes = item['routes']
        
        # Create cost vectors for each route [local, hybrid, cloud]
        privacy_costs = torch.tensor([
            routes['local']['privacy_leakage'],
            routes['hybrid']['privacy_leakage'],
            routes['cloud']['privacy_leakage']
        ], dtype=torch.float32)
        
        energy_costs = torch.tensor([
            routes['local']['energy_cost'],
            routes['hybrid']['energy_cost'],
            routes['cloud']['energy_cost']
        ], dtype=torch.float32)
        
        quality_scores = torch.tensor([
            routes['local']['task_quality'],
            routes['hybrid']['task_quality'],
            routes['cloud']['task_quality']
        ], dtype=torch.float32)
        
        # Device features
        device = item['device']
        device_features = torch.tensor([
            device['battery_level'],
            device['cpu_load'],
            device['ram_mb'] / 8192.0,  # Normalize to [0, 1]
            {'wifi': 0.0, '4g': 0.5, '5g': 1.0}[device['network']]
        ], dtype=torch.float32)
        
        # Optimal route as label
        optimal_route = self.route_to_idx[item['optimal_route']]
        
        return {
            'query': item['query_text'],
            'device_features': device_features,
            'privacy_costs': privacy_costs,
            'energy_costs': energy_costs,
            'quality_scores': quality_scores,
            'optimal_route': torch.tensor(optimal_route, dtype=torch.long),
            'privacy_risk': torch.tensor(item['privacy_risk'], dtype=torch.float32)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    return {
        'queries': [item['query'] for item in batch],
        'device_features': torch.stack([item['device_features'] for item in batch]),
        'privacy_costs': torch.stack([item['privacy_costs'] for item in batch]),
        'energy_costs': torch.stack([item['energy_costs'] for item in batch]),
        'quality_scores': torch.stack([item['quality_scores'] for item in batch]),
        'optimal_routes': torch.stack([item['optimal_route'] for item in batch]),
        'privacy_risks': torch.stack([item['privacy_risk'] for item in batch])
    }
