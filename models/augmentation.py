"""
Data augmentation techniques for routing model
"""

import torch
import numpy as np


def mixup_data(device_features, labels, alpha=0.2):
    """
    Apply mixup augmentation to device features
    
    Args:
        device_features: Batch of device feature tensors
        labels: Batch of labels
        alpha: Mixup interpolation strength
        
    Returns:
        mixed_features, labels_a, labels_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = device_features.size(0)
    index = torch.randperm(batch_size).to(device_features.device)

    mixed_features = lam * device_features + (1 - lam) * device_features[index, :]
    labels_a, labels_b = labels, labels[index]
    
    return mixed_features, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

