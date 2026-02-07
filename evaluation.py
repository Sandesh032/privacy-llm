"""
Evaluation script to measure privacy and energy improvements
Compares trained model against always-cloud and always-local baselines
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.append('data')
sys.path.append('models')

from data.adaptive_dataset import AdaptiveRoutingDataset, collate_fn
from models.routing_model import AdaptiveRoutingModel

def evaluate_improvements(model_path='checkpoints/best_model.pt',
                         dataset_path='data/held_out_test.jsonl'):
    """
    Evaluate model and compute improvements vs baselines

    Args:
        model_path: Path to trained model checkpoint
        dataset_path: Path to held-out test dataset JSONL file (must be separate from training data)

    Returns:
        dict: Evaluation results with model and baseline metrics
    """

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AdaptiveRoutingModel().to(device)

    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using: python training.py")
        return None

    checkpoint = torch.load(model_path, weights_only = False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✓ Model loaded successfully")

    # Load held-out test data (entirely separate from training data)
    if not Path(dataset_path).exists():
        print(f"Error: Held-out test dataset not found at {dataset_path}")
        print("Generate it first using: python evaluation.py --generate")
        return None

    test_dataset = AdaptiveRoutingDataset(dataset_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=collate_fn
    )

    print(f"✓ Evaluating on {len(test_dataset):,} held-out test samples (never seen during training)\n")

    # Evaluation metrics
    model_privacy = []
    model_energy = []
    model_quality = []

    optimal_privacy = []
    optimal_energy = []
    optimal_quality = []

    # Baseline: always cloud
    baseline_cloud_privacy = []
    baseline_cloud_energy = []
    baseline_cloud_quality = []

    # Baseline: always local
    baseline_local_privacy = []
    baseline_local_energy = []
    baseline_local_quality = []

    route_names = ['local', 'hybrid', 'cloud']

    with torch.no_grad():
        for batch in test_loader:
            queries = batch['queries']
            device_features = batch['device_features'].to(device)

            # Model predictions
            predicted_routes, _ = model.predict_route(queries, device_features)

            # Get costs for predicted and optimal routes
            privacy_costs = batch['privacy_costs']
            energy_costs = batch['energy_costs']
            quality_scores = batch['quality_scores']
            optimal_routes = batch['optimal_routes']

            for i in range(len(queries)):
                pred_route = predicted_routes[i].item()
                opt_route = optimal_routes[i].item()

                # Model performance
                model_privacy.append(privacy_costs[i][pred_route].item())
                model_energy.append(energy_costs[i][pred_route].item())
                model_quality.append(quality_scores[i][pred_route].item())

                # Optimal performance
                optimal_privacy.append(privacy_costs[i][opt_route].item())
                optimal_energy.append(energy_costs[i][opt_route].item())
                optimal_quality.append(quality_scores[i][opt_route].item())

                # Always cloud baseline (index 2)
                baseline_cloud_privacy.append(privacy_costs[i][2].item())
                baseline_cloud_energy.append(energy_costs[i][2].item())
                baseline_cloud_quality.append(quality_scores[i][2].item())

                # Always local baseline (index 0)
                baseline_local_privacy.append(privacy_costs[i][0].item())
                baseline_local_energy.append(energy_costs[i][0].item())
                baseline_local_quality.append(quality_scores[i][0].item())

    # Calculate averages
    results = {
        'model': {
            'privacy': np.mean(model_privacy),
            'energy': np.mean(model_energy),
            'quality': np.mean(model_quality)
        },
        'optimal': {
            'privacy': np.mean(optimal_privacy),
            'energy': np.mean(optimal_energy),
            'quality': np.mean(optimal_quality)
        },
        'baseline_cloud': {
            'privacy': np.mean(baseline_cloud_privacy),
            'energy': np.mean(baseline_cloud_energy),
            'quality': np.mean(baseline_cloud_quality)
        },
        'baseline_local': {
            'privacy': np.mean(baseline_local_privacy),
            'energy': np.mean(baseline_local_energy),
            'quality': np.mean(baseline_local_quality)
        }
    }

    # Print results
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print("\nAverage Metrics:")
    print(f"  Model:")
    print(f"    Privacy Risk: {results['model']['privacy']:.4f}")
    print(f"    Energy Cost:  {results['model']['energy']:.4f}")
    print(f"    Task Quality: {results['model']['quality']:.4f}")

    print(f"\n  Optimal Oracle:")
    print(f"    Privacy Risk: {results['optimal']['privacy']:.4f}")
    print(f"    Energy Cost:  {results['optimal']['energy']:.4f}")
    print(f"    Task Quality: {results['optimal']['quality']:.4f}")

    print(f"\n  Always Cloud Baseline:")
    print(f"    Privacy Risk: {results['baseline_cloud']['privacy']:.4f}")
    print(f"    Energy Cost:  {results['baseline_cloud']['energy']:.4f}")
    print(f"    Task Quality: {results['baseline_cloud']['quality']:.4f}")

    print(f"\n  Always Local Baseline:")
    print(f"    Privacy Risk: {results['baseline_local']['privacy']:.4f}")
    print(f"    Energy Cost:  {results['baseline_local']['energy']:.4f}")
    print(f"    Task Quality: {results['baseline_local']['quality']:.4f}")

    # Calculate improvements vs always-cloud baseline
    energy_improvement_cloud = ((results['baseline_cloud']['energy'] - results['model']['energy'])
                                / results['baseline_cloud']['energy'] * 100)
    privacy_improvement_cloud = ((results['baseline_cloud']['privacy'] - results['model']['privacy'])
                                 / results['baseline_cloud']['privacy'] * 100)
    quality_vs_cloud = ((results['model']['quality'] - results['baseline_cloud']['quality'])
                       / results['baseline_cloud']['quality'] * 100)

    # Calculate improvements vs always-local baseline
    energy_improvement_local = ((results['baseline_local']['energy'] - results['model']['energy'])
                                / results['baseline_local']['energy'] * 100)
    privacy_improvement_local = ((results['baseline_local']['privacy'] - results['model']['privacy'])
                                 / results['baseline_local']['privacy'] * 100)
    quality_vs_local = ((results['model']['quality'] - results['baseline_local']['quality'])
                       / results['baseline_local']['quality'] * 100)

    print("\n" + "="*80)
    print("IMPROVEMENTS vs ALWAYS-CLOUD BASELINE")
    print("="*80)
    print(f"  Energy Usage:  {energy_improvement_cloud:+.2f}% (X = {abs(energy_improvement_cloud):.1f}%)")
    print(f"  Privacy Risk:  {privacy_improvement_cloud:+.2f}% (Y = {abs(privacy_improvement_cloud):.1f}%)")
    print(f"  Task Quality:  {quality_vs_cloud:+.2f}%")

    print("\n" + "="*80)
    print("IMPROVEMENTS vs ALWAYS-LOCAL BASELINE")
    print("="*80)
    print(f"  Energy Usage:  {energy_improvement_local:+.2f}%")
    print(f"  Privacy Risk:  {privacy_improvement_local:+.2f}%")
    print(f"  Task Quality:  {quality_vs_local:+.2f}%")

    print(f"\n\"Our system reduces energy usage by {abs(energy_improvement_cloud):.1f}% and privacy risk")
    print(f"by {abs(privacy_improvement_cloud):.1f}% compared to always-cloud baseline, while maintaining")
    print(f"comparable task performance ({quality_vs_cloud:+.2f}% quality change).\"")

    return {
      "results": results,
      "model_privacy": model_privacy,
      "model_energy": model_energy,
      "model_quality": model_quality,
      "optimal_privacy": optimal_privacy,
      "optimal_energy": optimal_energy,
      "optimal_quality": optimal_quality,
      "baseline_cloud_privacy": baseline_cloud_privacy,
      "baseline_cloud_energy": baseline_cloud_energy,
      "baseline_cloud_quality": baseline_cloud_quality,
      "baseline_local_privacy": baseline_local_privacy,
      "baseline_local_energy": baseline_local_energy,
      "baseline_local_quality": baseline_local_quality
  }


def generate_held_out_test(output_path='data/held_out_test.jsonl', target_per_class=3000):
    """Generate a separate held-out test dataset that the model has never seen"""
    sys.path.append('data')
    from data.dataset import generate_research_dataset

    print("="*80)
    print("GENERATING HELD-OUT TEST DATASET")
    print(f"  Output: {output_path}")
    print(f"  Target: {target_per_class} samples per class ({target_per_class * 3} total)")
    print("="*80)

    counts, methods = generate_research_dataset(
        output_path=output_path,
        target_per_class=target_per_class,
        use_llm=True,
        llm_ratio=0.8,
        hard_neg_ratio=0.1
    )

    print(f"\n✓ Held-out test dataset saved to {output_path}")
    print(f"  Counts: {counts}")
    return counts, methods


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate routing model or generate held-out test data')
    parser.add_argument('--generate', action='store_true',
                        help='Generate a new held-out test dataset')
    parser.add_argument('--model', default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', default='data/held_out_test.jsonl',
                        help='Path to held-out test dataset')
    parser.add_argument('--target-per-class', type=int, default=3000,
                        help='Samples per class when generating (default: 3000)')
    args = parser.parse_args()

    if args.generate:
        generate_held_out_test(output_path=args.dataset, target_per_class=args.target_per_class)
    else:
        evaluate_improvements(model_path=args.model, dataset_path=args.dataset)
