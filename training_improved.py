"""
IMPROVED Training script for adaptive routing model
Includes: privacy features, frozen BERT, mixup, early stopping, label smoothing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from pathlib import Path
import sys
import numpy as np

# Add paths
sys.path.append('data')
sys.path.append('models')

from data.adaptive_dataset_loader import AdaptiveRoutingDataset, collate_fn
from models.routing_model import AdaptiveRoutingModel, count_parameters
from models.augmentation import mixup_data, mixup_criterion


# Enhanced Configuration
CONFIG = {
    'dataset_path': 'data/adaptive_dataset.jsonl',
    'batch_size': 32,
    'learning_rate': 5e-5,
    'epochs': 30,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints',
    'log_interval': 100,
    'class_weights': [1.0, 1.5, 1.2],
    'patience': 7,
    'mixup_alpha': 0.2,
    'label_smoothing': 0.1,
    'warmup_epochs': 3,
}


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_mixup=True):
    """Train for one epoch with optional mixup"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        queries = batch['queries']
        device_features = batch['device_features'].to(device)
        labels = batch['optimal_routes'].to(device)

        # Apply mixup augmentation (only on device features, not text)
        if use_mixup and epoch > CONFIG['warmup_epochs']:
            mixed_features, labels_a, labels_b, lam = mixup_data(
                device_features, labels, alpha=CONFIG['mixup_alpha']
            )
            
            # Forward pass with mixed features
            logits = model(queries, mixed_features)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            # Standard forward pass
            logits = model(queries, device_features)
            loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics (use original labels for accuracy)
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-route metrics
    route_correct = {0: 0, 1: 0, 2: 0}
    route_total = {0: 0, 1: 0, 2: 0}

    for batch in tqdm(dataloader, desc="Evaluating"):
        queries = batch['queries']
        device_features = batch['device_features'].to(device)
        labels = batch['optimal_routes'].to(device)

        logits = model(queries, device_features)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Per-route accuracy
        for i in range(3):
            mask = labels == i
            if mask.sum() > 0:
                route_correct[i] += (predictions[mask] == labels[mask]).sum().item()
                route_total[i] += mask.sum().item()

    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'per_route_accuracy': {
            'local': route_correct[0] / route_total[0] if route_total[0] > 0 else 0,
            'hybrid': route_correct[1] / route_total[1] if route_total[1] > 0 else 0,
            'cloud': route_correct[2] / route_total[2] if route_total[2] > 0 else 0
        }
    }

    return metrics


def main():
    print("="*80)
    print("IMPROVED Adaptive Routing Model Training")
    print("="*80)

    # Set device
    device = torch.device(CONFIG['device'])
    print(f"\nUsing device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = AdaptiveRoutingDataset(CONFIG['dataset_path'])

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(CONFIG['train_split'] * total_size)
    val_size = int(CONFIG['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=collate_fn,
        num_workers=0
    )

    # Initialize model
    print("\nInitializing improved model...")
    print("  - Privacy risk as direct feature")
    print("  - Frozen BERT base (fine-tune last 2 layers)")
    print("  - Deeper device network")
    print("  - Mixup augmentation")
    print("  - Label smoothing")
    print("  - Early stopping")
    
    model = AdaptiveRoutingModel().to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # Loss with label smoothing
    class_weights = torch.tensor(CONFIG['class_weights']).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, 
        label_smoothing=CONFIG['label_smoothing']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=0.01
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return (epoch + 1) / CONFIG['warmup_epochs']
        return 0.5 * (1 + np.cos(np.pi * (epoch - CONFIG['warmup_epochs']) / 
                                   (CONFIG['epochs'] - CONFIG['warmup_epochs'])))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create save directory
    save_dir = Path(CONFIG['save_dir'])
    save_dir.mkdir(exist_ok=True)

    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Per-Route Accuracy:")
        print(f"    Local:  {val_metrics['per_route_accuracy']['local']:.4f}")
        print(f"    Hybrid: {val_metrics['per_route_accuracy']['hybrid']:.4f}")
        print(f"    Cloud:  {val_metrics['per_route_accuracy']['cloud']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Save best model with early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'config': CONFIG,
                'per_route_acc': val_metrics['per_route_accuracy']
            }, save_dir / 'best_model_improved.pt')
            print(f"  ✓ New best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['patience']})")
            
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break

    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)

    # Load best model
    checkpoint = torch.load(save_dir / 'best_model_improved.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Per-Route Accuracy:")
    print(f"    Local:  {test_metrics['per_route_accuracy']['local']:.4f}")
    print(f"    Hybrid: {test_metrics['per_route_accuracy']['hybrid']:.4f}")
    print(f"    Cloud:  {test_metrics['per_route_accuracy']['cloud']:.4f}")

    # Compare with baseline
    print(f"\n  Improvement over baseline (74.12%):")
    improvement = (test_metrics['accuracy'] - 0.7412) * 100
    print(f"  {improvement:+.2f} percentage points")

    # Save training history
    with open(save_dir / 'training_history_improved.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Best model: {save_dir / 'best_model_improved.pt'}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()

