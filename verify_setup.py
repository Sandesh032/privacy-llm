"""
Test script to verify all improvements are correctly implemented
Run this before training to catch any issues early
"""

import sys
import torch
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("VERIFICATION: Testing Improved Model Setup")
print("="*80)

all_good = True

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    sys.path.append('data')
    sys.path.append('models')
    from data.adaptive_dataset_loader import AdaptiveRoutingDataset, collate_fn
    from models.routing_model import AdaptiveRoutingModel, count_parameters
    from models.augmentation import mixup_data, mixup_criterion
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    all_good = False

# Test 2: Load dataset and check features
print("\n2. Testing dataset...")
try:
    dataset = AdaptiveRoutingDataset('data/adaptive_dataset.jsonl')
    sample = dataset[0]
    
    # Check device features dimension
    if sample['device_features'].shape[0] == 5:
        print(f"   ✓ Device features: {sample['device_features'].shape[0]} dimensions (includes privacy)")
    else:
        print(f"   ❌ Device features: {sample['device_features'].shape[0]} dimensions (should be 5!)")
        all_good = False
    
    # Check privacy risk is included
    if sample['device_features'][4] > 0:
        print(f"   ✓ Privacy risk feature: {sample['device_features'][4]:.4f}")
    else:
        print(f"   ⚠️  Privacy risk might be 0 (check if this is expected)")
    
    print(f"   ✓ Dataset loaded: {len(dataset):,} samples")
    
except Exception as e:
    print(f"   ❌ Dataset test failed: {e}")
    all_good = False

# Test 3: Initialize model
print("\n3. Testing model architecture...")
try:
    model = AdaptiveRoutingModel()
    
    # Check input dimensions
    if model.device_network[0].in_features == 5:
        print(f"   ✓ Device network input: {model.device_network[0].in_features} features")
    else:
        print(f"   ❌ Device network input: {model.device_network[0].in_features} (should be 5!)")
        all_good = False
    
    # Check BERT freezing
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Frozen parameters: {frozen_params:,}")
    
    if trainable_params < 20_000_000:  # Should be around 12M
        print(f"   ✓ BERT base is frozen (good!)")
    else:
        print(f"   ⚠️  Too many trainable params - BERT might not be frozen correctly")
        all_good = False
    
    # Check device network depth
    device_net_layers = sum(1 for layer in model.device_network if isinstance(layer, torch.nn.Linear))
    if device_net_layers >= 3:
        print(f"   ✓ Device network depth: {device_net_layers} layers")
    else:
        print(f"   ⚠️  Device network: {device_net_layers} layers (expected 3+)")
        all_good = False
    
except Exception as e:
    print(f"   ❌ Model test failed: {e}")
    all_good = False

# Test 4: Test forward pass
print("\n4. Testing forward pass...")
try:
    queries = ["What is the weather today?", "Send email to john@example.com"]
    device_features = torch.tensor([
        [0.7, 0.4, 0.5, 0.0, 0.3],
        [0.5, 0.6, 0.5, 1.0, 0.8]
    ])  # batch_size=2 for BatchNorm
    
    logits = model(queries, device_features)
    
    if logits.shape == (2, 3):
        print(f"   ✓ Output shape: {logits.shape} (batch=2, classes=3)")
    else:
        print(f"   ❌ Output shape: {logits.shape} (expected [2, 3])")
        all_good = False
    
    routes, probs = model.predict_route(queries, device_features)
    route_names = ['local', 'hybrid', 'cloud']
    print(f"   ✓ Predicted routes: {[route_names[r] for r in routes]}")
    
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    all_good = False

# Test 5: Test mixup
print("\n5. Testing mixup augmentation...")
try:
    batch_features = torch.randn(4, 5)  # 4 samples, 5 features
    labels = torch.tensor([0, 1, 2, 1])
    
    mixed_features, labels_a, labels_b, lam = mixup_data(batch_features, labels, alpha=0.2)
    
    if mixed_features.shape == (4, 5):
        print(f"   ✓ Mixup output shape: {mixed_features.shape}")
    else:
        print(f"   ❌ Mixup output shape: {mixed_features.shape} (expected [4, 5])")
        all_good = False
    
    print(f"   ✓ Mixup lambda: {lam:.4f}")
    
except Exception as e:
    print(f"   ❌ Mixup test failed: {e}")
    all_good = False

# Test 6: Check training config
print("\n6. Checking training configuration...")
try:
    import training_improved
    config = training_improved.CONFIG
    
    checks = [
        ('learning_rate', 5e-5, "Learning rate should be 5e-5"),
        ('patience', 7, "Early stopping patience should be 7"),
        ('mixup_alpha', 0.2, "Mixup alpha should be 0.2"),
        ('label_smoothing', 0.1, "Label smoothing should be 0.1"),
        ('warmup_epochs', 3, "Warmup epochs should be 3"),
    ]
    
    for key, expected, msg in checks:
        if key in config and config[key] == expected:
            print(f"   ✓ {key}: {config[key]}")
        else:
            print(f"   ⚠️  {key}: {config.get(key, 'NOT FOUND')} (expected {expected})")
    
    # Check class weights
    if config['class_weights'] == [1.0, 1.5, 1.2]:
        print(f"   ✓ class_weights: {config['class_weights']}")
    else:
        print(f"   ⚠️  class_weights: {config['class_weights']} (expected [1.0, 1.5, 1.2])")
    
except Exception as e:
    print(f"   ❌ Config check failed: {e}")
    all_good = False

# Test 7: Check file structure
print("\n7. Checking file structure...")
files_to_check = [
    ('models/routing_model.py', 'Model architecture'),
    ('models/augmentation.py', 'Mixup augmentation'),
    ('data/adaptive_dataset_loader.py', 'Dataset loader'),
    ('data/oracle.py', 'Oracle (fixed)'),
    ('training.py', 'Original training script'),
    ('training_improved.py', 'Improved training script'),
    ('compare_models.py', 'Comparison utility'),
    ('IMPROVEMENTS.md', 'Detailed documentation'),
    ('QUICK_START_IMPROVEMENTS.md', 'Quick guide'),
    ('SUMMARY.md', 'Summary'),
]

for filepath, description in files_to_check:
    if Path(filepath).exists():
        print(f"   ✓ {filepath}")
    else:
        print(f"   ❌ {filepath} (missing!)")
        all_good = False

# Final summary
print("\n" + "="*80)
if all_good:
    print("✅ ALL CHECKS PASSED!")
    print("="*80)
    print("\nYou're ready to train the improved model:")
    print("  python training_improved.py")
    print("\nExpected results:")
    print("  - Overall accuracy: 85-90%")
    print("  - Cloud accuracy: 82-85% (up from 68%)")
    print("  - Trainable params: ~12M (down from 110M)")
else:
    print("⚠️  SOME CHECKS FAILED")
    print("="*80)
    print("\nPlease review the errors above before training.")
    print("Some warnings are OK, but errors should be fixed.")

print("\n" + "="*80)

