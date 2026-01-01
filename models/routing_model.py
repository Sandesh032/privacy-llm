"""
Adaptive Routing Model
Predicts optimal route (local/hybrid/cloud) based on query and device context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple


class AdaptiveRoutingModel(nn.Module):
    """
    Neural network for adaptive routing decisions
    
    Architecture:
    1. Text Encoder (BERT) - encodes query
    2. Device Feature Network - processes device state
    3. Fusion Layer - combines text + device features
    4. Routing Head - predicts optimal route
    """
    
    def __init__(self, 
                 pretrained_model: str = "bert-base-uncased",
                 device_feature_dim: int = 5,  # Added privacy_risk feature
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        # Text encoder
        print(f"Loading pretrained model: {pretrained_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        
        # Freeze BERT base layers, only fine-tune last 2 layers
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # Unfreeze last 2 encoder layers for task-specific adaptation
        for layer in self.text_encoder.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        text_embed_dim = self.text_encoder.config.hidden_size  # 768 for BERT
        
        # Device feature network - deeper for privacy+device context
        self.device_network = nn.Sequential(
            nn.Linear(device_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = text_embed_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Routing head (outputs 3 logits for local/hybrid/cloud)
        self.routing_head = nn.Linear(hidden_dim, 3)
        
        print(f"✓ Model initialized")
        print(f"  Text encoder dim: {text_embed_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output: 3 routes (local/hybrid/cloud)")
    
    def forward(self, queries: List[str], device_features: torch.Tensor):
        """
        Forward pass
        
        Args:
            queries: List of query strings
            device_features: [batch_size, 5] device state + privacy_risk
            
        Returns:
            route_logits: [batch_size, 3] logits for each route
        """
        device = next(self.text_encoder.parameters()).device
        
        # Encode text
        encoded = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        text_outputs = self.text_encoder(**encoded)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Encode device features
        device_features = device_features.to(device)
        device_encoded = self.device_network(device_features)
        
        # Fuse text + device
        combined = torch.cat([text_features, device_encoded], dim=1)
        fused = self.fusion(combined)
        
        # Predict route
        route_logits = self.routing_head(fused)
        
        return route_logits
    
    def predict_route(self, queries: List[str], device_features: torch.Tensor):
        """
        Predict optimal route
        
        Returns:
            routes: [batch_size] predicted route indices
            probs: [batch_size, 3] probabilities for each route
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(queries, device_features)
            probs = F.softmax(logits, dim=-1)
            routes = torch.argmax(probs, dim=-1)
        
        return routes, probs


# Count parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
