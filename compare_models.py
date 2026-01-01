"""
Compare baseline vs improved model performance
"""

import torch
import json
from pathlib import Path


def compare_models():
    """Compare baseline and improved model checkpoints"""
    
    baseline_path = Path("checkpoints/best_model.pt")
    improved_path = Path("checkpoints/best_model_improved.pt")
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Check baseline
    if baseline_path.exists():
        baseline = torch.load(baseline_path, map_location='cpu')
        print("\n📊 BASELINE MODEL (Original)")
        print("-" * 80)
        print(f"  Validation Accuracy: {baseline['val_accuracy']:.4f} ({baseline['val_accuracy']*100:.2f}%)")
        print(f"  Epoch: {baseline['epoch']}")
        if 'per_route_acc' in baseline:
            print(f"  Per-Route Accuracy:")
            for route, acc in baseline['per_route_acc'].items():
                print(f"    {route.capitalize()}: {acc:.4f}")
    else:
        print("\n❌ Baseline model not found")
        print("   Run: python training.py")
    
    print()
    
    # Check improved
    if improved_path.exists():
        improved = torch.load(improved_path, map_location='cpu')
        print("\n🚀 IMPROVED MODEL")
        print("-" * 80)
        print(f"  Validation Accuracy: {improved['val_accuracy']:.4f} ({improved['val_accuracy']*100:.2f}%)")
        print(f"  Epoch: {improved['epoch']}")
        if 'per_route_acc' in improved:
            print(f"  Per-Route Accuracy:")
            for route, acc in improved['per_route_acc'].items():
                print(f"    {route.capitalize()}: {acc:.4f}")
        
        # Calculate improvement
        if baseline_path.exists():
            improvement = (improved['val_accuracy'] - baseline['val_accuracy']) * 100
            print(f"\n✨ IMPROVEMENT")
            print("-" * 80)
            print(f"  Absolute: {improvement:+.2f} percentage points")
            print(f"  Relative: {(improvement / (baseline['val_accuracy']*100)) * 100:+.2f}%")
            
            if 'per_route_acc' in baseline and 'per_route_acc' in improved:
                print(f"\n  Per-Route Improvements:")
                for route in ['local', 'hybrid', 'cloud']:
                    if route in baseline['per_route_acc'] and route in improved['per_route_acc']:
                        base_acc = baseline['per_route_acc'][route]
                        imp_acc = improved['per_route_acc'][route]
                        diff = (imp_acc - base_acc) * 100
                        print(f"    {route.capitalize()}: {diff:+.2f} pp")
    else:
        print("\n⚠️  Improved model not found")
        print("   Run: python training_improved.py")
    
    # Check training histories
    print("\n" + "="*80)
    print("TRAINING HISTORY")
    print("="*80)
    
    baseline_history = Path("checkpoints/training_history.json")
    improved_history = Path("checkpoints/training_history_improved.json")
    
    if baseline_history.exists():
        with open(baseline_history) as f:
            hist = json.load(f)
        print(f"\n📈 Baseline Training:")
        print(f"  Epochs completed: {len(hist['val_acc'])}")
        print(f"  Final train acc: {hist['train_acc'][-1]:.4f}")
        print(f"  Final val acc: {hist['val_acc'][-1]:.4f}")
        print(f"  Best val acc: {max(hist['val_acc']):.4f}")
    
    if improved_history.exists():
        with open(improved_history) as f:
            hist = json.load(f)
        print(f"\n🚀 Improved Training:")
        print(f"  Epochs completed: {len(hist['val_acc'])}")
        print(f"  Final train acc: {hist['train_acc'][-1]:.4f}")
        print(f"  Final val acc: {hist['val_acc'][-1]:.4f}")
        print(f"  Best val acc: {max(hist['val_acc']):.4f}")
        
        # Check if early stopping triggered
        if len(hist['val_acc']) < 30:
            print(f"  Early stopping: ✓ (stopped at epoch {len(hist['val_acc'])})")
    
    print("\n" + "="*80)
    

if __name__ == '__main__':
    compare_models()

