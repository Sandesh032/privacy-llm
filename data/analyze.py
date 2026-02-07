import json
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Load dataset
samples = []
with open('local_dataset.jsonl', 'r') as f:
    for line in f:
        samples.append(json.loads(line))

print("="*80)
print("DATASET QUALITY ANALYSIS")
print("="*80)

# 1. Class distribution (you already know this)
labels = [s['label'] for s in samples]
print(f"\n1. Class Distribution:")
print(f"   {Counter(labels)}")
print(f"   ✓ Balanced")

# 2. Analyze routing scores - THE KEY INSIGHT
print(f"\n2. Routing Score Analysis (why each label was chosen):")
route_names = ['local', 'hybrid', 'cloud']
alpha, beta = 0.5, 0.5  # From choose_best()

score_margins = defaultdict(list)
score_differences = defaultdict(list)

for sample in samples:
    label = sample['label']
    routes = sample['routes']
    
    # Calculate scores for all routes
    scores = {}
    for route_name in route_names:
        r = routes[route_name]
        scores[route_name] = r['task_quality'] - alpha * r['privacy_leakage'] - beta * r['energy_cost']
    
    chosen_route = route_names[label]
    chosen_score = scores[chosen_route]
    
    # Calculate margin (how much better is chosen route vs second best)
    other_scores = [scores[r] for r in route_names if r != chosen_route]
    margin = chosen_score - max(other_scores)
    score_margins[chosen_route].append(margin)
    
    # Store score differences
    score_differences[chosen_route].append({
        'local': scores['local'],
        'hybrid': scores['hybrid'],
        'cloud': scores['cloud'],
        'margin': margin
    })

for route in route_names:
    margins = score_margins[route]
    print(f"\n   {route.upper()} (n={len(margins)}):")
    print(f"      Margin: mean={np.mean(margins):.4f}, std={np.std(margins):.4f}")
    print(f"      Min margin: {np.min(margins):.4f}")
    print(f"      10th percentile: {np.percentile(margins, 10):.4f}")
    
    if np.mean(margins) < 0.05:
        print(f"      ⚠️  WARNING: Very small margins! Label noise likely.")
    if np.percentile(margins, 10) < 0.01:
        print(f"      ⚠️  WARNING: 10% of samples have margin < 0.01 (ambiguous)")

# 3. Feature analysis by class
print(f"\n3. Feature Distributions by Class:")

features_by_class = defaultdict(lambda: defaultdict(list))
for sample in samples:
    label = sample['label']
    route = route_names[label]
    
    # Device features
    features_by_class[route]['battery'].append(sample['device']['battery_level'])
    features_by_class[route]['cpu'].append(sample['device']['cpu_load'])
    features_by_class[route]['privacy_risk'].append(sample['privacy_risk'])
    
    # Route-specific metrics
    r = sample['routes'][route]
    features_by_class[route]['privacy_leakage'].append(r['privacy_leakage'])
    features_by_class[route]['energy_cost'].append(r['energy_cost'])
    features_by_class[route]['task_quality'].append(r['task_quality'])

for route in route_names:
    print(f"\n   {route.upper()}:")
    for feat, values in features_by_class[route].items():
        print(f"      {feat:20s}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

# 4. Check for feature overlap (class separability)
print(f"\n4. Class Separability Analysis:")
print(f"   Checking if feature distributions overlap significantly...")

def overlap_coefficient(dist1, dist2):
    """Calculate overlap between two distributions"""
    min1, max1 = np.min(dist1), np.max(dist1)
    min2, max2 = np.min(dist2), np.max(dist2)
    
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    
    if overlap_max < overlap_min:
        return 0.0
    
    range1 = max1 - min1
    range2 = max2 - min2
    overlap_range = overlap_max - overlap_min
    
    return (overlap_range / range1 + overlap_range / range2) / 2

key_features = ['privacy_risk', 'privacy_leakage', 'energy_cost', 'task_quality']
for feat in key_features:
    local_dist = features_by_class['local'][feat]
    hybrid_dist = features_by_class['hybrid'][feat]
    cloud_dist = features_by_class['cloud'][feat]
    
    overlap_lh = overlap_coefficient(local_dist, hybrid_dist)
    overlap_hc = overlap_coefficient(hybrid_dist, cloud_dist)
    overlap_lc = overlap_coefficient(local_dist, cloud_dist)
    
    print(f"\n   {feat}:")
    print(f"      local-hybrid overlap: {overlap_lh:.2%}")
    print(f"      hybrid-cloud overlap: {overlap_hc:.2%}")
    print(f"      local-cloud overlap: {overlap_lc:.2%}")
    
    if overlap_lh > 0.8 or overlap_hc > 0.8:
        print(f"      ⚠️  HIGH OVERLAP! Classes not well separated on this feature.")

# 5. Check for label noise - samples very close to decision boundary
print(f"\n5. Label Noise Analysis (samples near decision boundary):")

ambiguous_threshold = 0.02
for route in route_names:
    ambiguous = [s for s in score_differences[route] if s['margin'] < ambiguous_threshold]
    pct = len(ambiguous) / len(score_differences[route]) * 100
    print(f"\n   {route}: {len(ambiguous)} ambiguous samples ({pct:.1f}%)")
    
    if pct > 15:
        print(f"      ⚠️  HIGH LABEL NOISE! {pct:.1f}% of {route} samples have margin < {ambiguous_threshold}")
        print(f"      This means the model needs to learn very fine-grained boundaries.")

# 6. Analyze hybrid's position in score space
print(f"\n6. Hybrid's Competitive Position:")
hybrid_samples = [s for s in samples if s['label'] == 1]

hybrid_beats_local = 0
hybrid_beats_cloud = 0
hybrid_beats_both = 0

for sample in hybrid_samples:
    routes = sample['routes']
    scores = {
        'local': routes['local']['task_quality'] - alpha * routes['local']['privacy_leakage'] - beta * routes['local']['energy_cost'],
        'hybrid': routes['hybrid']['task_quality'] - alpha * routes['hybrid']['privacy_leakage'] - beta * routes['hybrid']['energy_cost'],
        'cloud': routes['cloud']['task_quality'] - alpha * routes['cloud']['privacy_leakage'] - beta * routes['cloud']['energy_cost']
    }
    
    if scores['hybrid'] > scores['local']:
        hybrid_beats_local += 1
    if scores['hybrid'] > scores['cloud']:
        hybrid_beats_cloud += 1
    if scores['hybrid'] > scores['local'] and scores['hybrid'] > scores['cloud']:
        hybrid_beats_both += 1

print(f"   Hybrid beats local: {hybrid_beats_local}/{len(hybrid_samples)} ({hybrid_beats_local/len(hybrid_samples)*100:.1f}%)")
print(f"   Hybrid beats cloud: {hybrid_beats_cloud}/{len(hybrid_samples)} ({hybrid_beats_cloud/len(hybrid_samples)*100:.1f}%)")
print(f"   Hybrid beats both: {hybrid_beats_both}/{len(hybrid_samples)} ({hybrid_beats_both/len(hybrid_samples)*100:.1f}%)")

if hybrid_beats_both < len(hybrid_samples) * 0.95:
    print(f"   ⚠️  WARNING: Some hybrid labels may be incorrect!")

# 7. Compare task quality across routes
print(f"\n7. Task Quality Comparison (the main feature difference):")
for route in route_names:
    samples_route = [s for s in samples if s['label'] == route_names.index(route)]
    
    local_q = [s['routes']['local']['task_quality'] for s in samples_route]
    hybrid_q = [s['routes']['hybrid']['task_quality'] for s in samples_route]
    cloud_q = [s['routes']['cloud']['task_quality'] for s in samples_route]
    
    print(f"\n   Samples labeled as {route.upper()}:")
    print(f"      When they use local:  quality = {np.mean(local_q):.3f}")
    print(f"      When they use hybrid: quality = {np.mean(hybrid_q):.3f}")
    print(f"      When they use cloud:  quality = {np.mean(cloud_q):.3f}")

print("\n" + "="*80)
print("SUMMARY & DIAGNOSIS")
print("="*80)

# Calculate key metrics
hybrid_margin_mean = np.mean(score_margins['hybrid'])
local_margin_mean = np.mean(score_margins['local'])
cloud_margin_mean = np.mean(score_margins['cloud'])

print(f"\nDecision Margin Comparison:")
print(f"  Local:  {local_margin_mean:.4f}")
print(f"  Hybrid: {hybrid_margin_mean:.4f} {'⚠️ SMALLEST' if hybrid_margin_mean < min(local_margin_mean, cloud_margin_mean) else ''}")
print(f"  Cloud:  {cloud_margin_mean:.4f}")

if hybrid_margin_mean < 0.03:
    print(f"\n❌ PROBLEM IDENTIFIED:")
    print(f"   Hybrid samples have very small decision margins ({hybrid_margin_mean:.4f})")
    print(f"   This means they are very close to the decision boundary between local and cloud.")
    print(f"   Small variations in features or model uncertainty can flip the prediction.")
    print(f"\n💡 ROOT CAUSE:")
    print(f"   The dataset generation uses fixed quality values:")
    print(f"   - Local: ~0.65, Hybrid: ~0.88, Cloud: ~0.95")
    print(f"   - Hybrid's quality advantage (0.88 vs 0.65) is offset by its moderate")
    print(f"     privacy leakage and energy costs, creating a narrow optimal region.")
    print(f"\n🔧 SOLUTIONS:")
    print(f"   1. Increase hybrid's task_quality to 0.90-0.92 (bigger gap from local)")
    print(f"   2. Decrease hybrid's privacy_leakage coefficient")
    print(f"   3. Generate samples with more diverse alpha/beta weights in choose_best()")
    print(f"   4. Add explicit hybrid-favoring scenarios in dataset generation")
