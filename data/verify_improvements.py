"""
Quick verification script to compare old vs new dataset quality
Run this after regenerating the dataset
"""

import json
import numpy as np
from collections import Counter

def analyze_dataset(filepath):
    """Analyze a dataset file and return key metrics"""
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Class distribution
    labels = [s['label'] for s in samples]
    counts = Counter(labels)
    
    # Calculate margins
    margins = {'local': [], 'hybrid': [], 'cloud': []}
    route_names = ['local', 'hybrid', 'cloud']
    
    for sample in samples:
        routes = sample['routes']
        alpha, beta = 0.5, 0.5
        
        scores = {
            r: routes[r]['task_quality'] - alpha * routes[r]['privacy_leakage'] - beta * routes[r]['energy_cost']
            for r in route_names
        }
        
        optimal = route_names[sample['label']]
        optimal_score = scores[optimal]
        other_scores = [scores[r] for r in route_names if r != optimal]
        margin = optimal_score - max(other_scores)
        
        margins[optimal].append(margin)
    
    # Calculate ambiguous percentage (margin < 0.02)
    ambiguous = {}
    for route in route_names:
        if margins[route]:
            ambig_count = sum(1 for m in margins[route] if m < 0.02)
            ambiguous[route] = (ambig_count / len(margins[route])) * 100
        else:
            ambiguous[route] = 0
    
    return {
        'total': len(samples),
        'counts': dict(counts),
        'margins': {r: {'mean': np.mean(margins[r]), 'std': np.std(margins[r])} 
                   for r in route_names if margins[r]},
        'ambiguous_pct': ambiguous
    }

# Analyze datasets
print("="*80)
print("DATASET QUALITY COMPARISON")
print("="*80)

try:
    print("\n📊 OLD DATASET (local_dataset.jsonl.backup):")
    old_stats = analyze_dataset('local_dataset.jsonl.backup')
    print(f"   Total samples: {old_stats['total']:,}")
    print(f"   Class distribution: {old_stats['counts']}")
    print(f"\n   Decision Margins:")
    for route in ['local', 'hybrid', 'cloud']:
        if route in old_stats['margins']:
            m = old_stats['margins'][route]
            print(f"      {route:6s}: mean={m['mean']:.4f}, std={m['std']:.4f}")
    print(f"\n   Ambiguous Samples (margin < 0.02):")
    for route in ['local', 'hybrid', 'cloud']:
        pct = old_stats['ambiguous_pct'][route]
        flag = "⚠️" if pct > 15 else "✓"
        print(f"      {route:6s}: {pct:5.1f}% {flag}")
except FileNotFoundError:
    print("   ⚠️  Backup file not found")

print("\n" + "-"*80)

try:
    print("\n📊 NEW DATASET (local_dataset.jsonl):")
    new_stats = analyze_dataset('local_dataset.jsonl')
    print(f"   Total samples: {new_stats['total']:,}")
    print(f"   Class distribution: {new_stats['counts']}")
    
    # Check balance
    if len(set(new_stats['counts'].values())) == 1:
        print(f"   ✓ Perfectly balanced!")
    else:
        print(f"   ⚠️  Imbalanced")
    
    print(f"\n   Decision Margins:")
    for route in ['local', 'hybrid', 'cloud']:
        if route in new_stats['margins']:
            m = new_stats['margins'][route]
            print(f"      {route:6s}: mean={m['mean']:.4f}, std={m['std']:.4f}")
    print(f"\n   Ambiguous Samples (margin < 0.02):")
    for route in ['local', 'hybrid', 'cloud']:
        pct = new_stats['ambiguous_pct'][route]
        flag = "⚠️" if pct > 15 else "✓"
        print(f"      {route:6s}: {pct:5.1f}% {flag}")
except FileNotFoundError:
    print("   ⚠️  New dataset not found. Run regenerate_dataset.py first.")

print("\n" + "="*80)
print("IMPROVEMENTS")
print("="*80)

try:
    print("\nMargin Improvements:")
    for route in ['local', 'hybrid', 'cloud']:
        if route in old_stats['margins'] and route in new_stats['margins']:
            old_margin = old_stats['margins'][route]['mean']
            new_margin = new_stats['margins'][route]['mean']
            improvement = ((new_margin - old_margin) / old_margin) * 100
            arrow = "↑" if improvement > 0 else "↓"
            print(f"   {route:6s}: {old_margin:.4f} → {new_margin:.4f} ({arrow}{abs(improvement):+.1f}%)")
    
    print("\nLabel Noise Reduction:")
    for route in ['local', 'hybrid', 'cloud']:
        old_ambig = old_stats['ambiguous_pct'][route]
        new_ambig = new_stats['ambiguous_pct'][route]
        reduction = old_ambig - new_ambig
        arrow = "✓" if reduction > 0 else "✗"
        print(f"   {route:6s}: {old_ambig:.1f}% → {new_ambig:.1f}% ({arrow} {reduction:+.1f}pp)")
    
    print("\n💡 Summary:")
    avg_old_margin = np.mean([old_stats['margins'][r]['mean'] for r in ['local', 'hybrid', 'cloud'] if r in old_stats['margins']])
    avg_new_margin = np.mean([new_stats['margins'][r]['mean'] for r in ['local', 'hybrid', 'cloud'] if r in new_stats['margins']])
    print(f"   Average margin: {avg_old_margin:.4f} → {avg_new_margin:.4f}")
    
    if avg_new_margin > avg_old_margin * 1.2:
        print(f"   ✓ Excellent! Margins increased by {((avg_new_margin/avg_old_margin - 1)*100):.1f}%")
    elif avg_new_margin > avg_old_margin:
        print(f"   ✓ Good improvement: {((avg_new_margin/avg_old_margin - 1)*100):.1f}%")
    else:
        print(f"   ⚠️  Margins decreased. Consider adjusting evaluate_routes()")
    
except Exception as e:
    print(f"\n⚠️  Could not compare: {e}")

print("="*80)

