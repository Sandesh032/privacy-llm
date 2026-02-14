"""
Comprehensive plotting and metrics script for research paper
Generates all necessary visualizations and metrics tables
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import sys

# Add paths
sys.path.append('data')
sys.path.append('models')

from data.adaptive_dataset import AdaptiveRoutingDataset, collate_fn
from models.routing_model import AdaptiveRoutingModel
from evaluation import evaluate_improvements

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_all_plots(dataset_path, output_dir, dataset_label,
                       model_path='checkpoints/best_model.pt',
                       history_path='checkpoints/training_history.json'):
    """
    Generate all research plots and metrics for a given dataset.

    Args:
        dataset_path: Path to the JSONL dataset to evaluate on.
        output_dir: Directory to save plots and reports.
        dataset_label: Human-readable label for this dataset (used in titles/prints).
        model_path: Path to the model checkpoint.
        history_path: Path to the training history JSON.
    """
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"GENERATING RESEARCH PLOTS AND METRICS — {dataset_label}")
    print(f"  Dataset : {dataset_path}")
    print(f"  Output  : {OUTPUT_DIR.absolute()}")
    print("=" * 80)

    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load dataset
    print(f"\nLoading dataset ({dataset_label})...")
    test_dataset = AdaptiveRoutingDataset(dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # Load model
    print("Loading model...")
    model = AdaptiveRoutingModel().to(device)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)

    print("\nCollecting predictions...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            queries = batch['queries']
            device_features = batch['device_features'].to(device)
            labels = batch['optimal_routes']

            logits = model(queries, device_features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nAccuracy on {dataset_label}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    route_names = ['Local', 'Hybrid', 'Cloud']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # ========================================================================
    # PLOT 1: Training and Validation Curves
    # ========================================================================
    print("\n[1/7] Generating training curves...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PLOT 2: Confusion Matrix
    # ========================================================================
    print("[2/7] Generating confusion matrix...")

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=route_names, yticklabels=route_names, ax=ax1, linewidths=0.5)
    ax1.set_xlabel('Predicted Route', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Route', fontsize=12, fontweight='bold')
    ax1.set_title(f'Confusion Matrix — Counts ({dataset_label})', fontsize=14, fontweight='bold')

    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', cbar_kws={'label': 'Proportion'},
                xticklabels=route_names, yticklabels=route_names, ax=ax2, linewidths=0.5)
    ax2.set_xlabel('Predicted Route', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Route', fontsize=12, fontweight='bold')
    ax2.set_title(f'Confusion Matrix — Normalized ({dataset_label})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PLOT 3: Per-Class Accuracy Bar Chart
    # ========================================================================
    print("[3/7] Generating per-class accuracy chart...")

    route_accuracies = {}
    for i, route_name in enumerate(route_names):
        mask = all_labels == i
        route_accuracies[route_name] = (all_preds[mask] == all_labels[mask]).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(route_names, list(route_accuracies.values()), color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Route Type', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Route Classification Accuracy ({dataset_label})', fontsize=14, fontweight='bold')
    ax.set_ylim([0.95, 1.0])
    ax.axhline(y=accuracy, color='black', linestyle='--', linewidth=2,
               label=f'Overall: {accuracy:.4f}')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_route_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_route_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PLOT 4: Feature Distributions by Route
    # ========================================================================
    print("[4/7] Generating feature distributions...")

    features_data = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]

        def get_value(x):
            return x.item() if hasattr(x, 'item') else x

        features_data.append({
            'Battery': get_value(sample['device_features'][0]),
            'CPU Load': get_value(sample['device_features'][1]),
            'RAM (normalized)': get_value(sample['device_features'][2]),
            'Network Type': get_value(sample['device_features'][3]),
            'Privacy Risk': get_value(sample['device_features'][4]),
            'Route': get_value(sample['optimal_route'])
        })

    df = pd.DataFrame(features_data)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features = ['Battery', 'CPU Load', 'Privacy Risk', 'RAM (normalized)', 'Network Type']

    for idx, feature in enumerate(features):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        for route_id, (route_name, color) in enumerate(zip(route_names, colors)):
            data = df[df['Route'] == route_id][feature]
            ax.hist(data, bins=30, alpha=0.5, label=route_name, color=color, edgecolor='black')

        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} Distribution by Route', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[1, 2])

    plt.suptitle(f'Feature Distributions ({dataset_label})', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PLOT 5: Privacy-Energy-Quality Trade-off
    # ========================================================================
    print("[5/7] Generating privacy-energy trade-off plot...")

    try:
        results = evaluate_improvements(dataset_path=dataset_path)

        fig, ax = plt.subplots(figsize=(11, 7))

        systems = ['Always-Local', 'Always-Cloud', 'Our Model', 'Optimal']
        privacy_risks = [
            results['results']['baseline_local']['privacy'],
            results['results']['baseline_cloud']['privacy'],
            results['results']['model']['privacy'],
            results['results']['optimal']['privacy']
        ]
        energy_costs = [
            results['results']['baseline_local']['energy'],
            results['results']['baseline_cloud']['energy'],
            results['results']['model']['energy'],
            results['results']['optimal']['energy']
        ]
        quality_scores = [
            results['results']['baseline_local']['quality'],
            results['results']['baseline_cloud']['quality'],
            results['results']['model']['quality'],
            results['results']['optimal']['quality']
        ]

        colors_sys = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        sizes = [q * 2000 for q in quality_scores]

        for i, (system, color) in enumerate(zip(systems, colors_sys)):
            ax.scatter(privacy_risks[i], energy_costs[i], s=sizes[i],
                       c=color, alpha=0.7, edgecolors='black', linewidth=2, label=system)
            ax.annotate(system, (privacy_risks[i], energy_costs[i]),
                        fontsize=11, fontweight='bold',
                        xytext=(12, 12), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))

        ax.set_xlabel('Privacy Risk (lower is better)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Energy Cost (lower is better)', fontsize=13, fontweight='bold')
        ax.set_title(f'Privacy-Energy-Quality Trade-off ({dataset_label})\n(bubble size indicates task quality)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'privacy_energy_tradeoff.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / 'privacy_energy_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   Warning: Could not generate trade-off plot: {e}")

    # ========================================================================
    # PLOT 6: ROC Curves
    # ========================================================================
    print("[6/7] Generating ROC curves...")

    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (color, name) in enumerate(zip(colors, route_names)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=3,
                label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'ROC Curves for Multi-class Route Classification ({dataset_label})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # METRICS REPORT
    # ========================================================================
    print("[7/7] Generating metrics report...")

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2]
    )

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"DETAILED CLASSIFICATION REPORT — {dataset_label}")
    report_lines.append("=" * 80)
    report_lines.append("")

    metrics_df = pd.DataFrame({
        'Route': route_names,
        'Precision': [f'{p:.4f}' for p in precision],
        'Recall': [f'{r:.4f}' for r in recall],
        'F1-Score': [f'{f:.4f}' for f in f1],
        'Support': support.astype(int),
        'Accuracy': [f'{route_accuracies[name]:.4f}' for name in route_names]
    })

    report_lines.append(metrics_df.to_string(index=False))
    report_lines.append("")
    report_lines.append(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    report_lines.append(f"Macro Avg Precision: {np.mean(precision):.4f}")
    report_lines.append(f"Macro Avg Recall: {np.mean(recall):.4f}")
    report_lines.append(f"Macro Avg F1-Score: {np.mean(f1):.4f}")
    report_lines.append(f"Weighted Avg F1-Score: {np.average(f1, weights=support):.4f}")
    report_lines.append("")

    # Baseline comparison
    if 'results' in locals():
        report_lines.append("=" * 80)
        report_lines.append("COMPARISON WITH BASELINES")
        report_lines.append("=" * 80)
        report_lines.append("")

        comparison_df = pd.DataFrame({
            'System': ['Always-Local', 'Always-Cloud', 'Our Model', 'Optimal Oracle'],
            'Privacy Risk': [
                f"{results['results']['baseline_local']['privacy']:.4f}",
                f"{results['results']['baseline_cloud']['privacy']:.4f}",
                f"{results['results']['model']['privacy']:.4f}",
                f"{results['results']['optimal']['privacy']:.4f}"
            ],
            'Energy Cost': [
                f"{results['results']['baseline_local']['energy']:.4f}",
                f"{results['results']['baseline_cloud']['energy']:.4f}",
                f"{results['results']['model']['energy']:.4f}",
                f"{results['results']['optimal']['energy']:.4f}"
            ],
            'Task Quality': [
                f"{results['results']['baseline_local']['quality']:.4f}",
                f"{results['results']['baseline_cloud']['quality']:.4f}",
                f"{results['results']['model']['quality']:.4f}",
                f"{results['results']['optimal']['quality']:.4f}"
            ]
        })

        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")

        # Calculate improvements
        privacy_improvement = ((results['results']['baseline_cloud']['privacy'] -
                                results['results']['model']['privacy']) /
                               results['results']['baseline_cloud']['privacy'] * 100)
        energy_improvement = ((results['results']['baseline_cloud']['energy'] -
                               results['results']['model']['energy']) /
                              results['results']['baseline_cloud']['energy'] * 100)

        report_lines.append("IMPROVEMENTS vs Always-Cloud Baseline:")
        report_lines.append(f"  Privacy Risk: {privacy_improvement:+.2f}% (lower is better)")
        report_lines.append(f"  Energy Cost: {energy_improvement:+.2f}% (lower is better)")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Save report
    report_text = '\n'.join(report_lines)
    with open(OUTPUT_DIR / 'metrics_report.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)

    # Save LaTeX table for paper
    latex_table = metrics_df.to_latex(index=False, float_format="%.4f")
    with open(OUTPUT_DIR / 'metrics_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"\n{'=' * 80}")
    print(f"ALL PLOTS AND METRICS GENERATED FOR: {dataset_label}")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  - training_curves.pdf/png")
    print("  - confusion_matrix.pdf/png")
    print("  - per_route_accuracy.pdf/png")
    print("  - feature_distributions.pdf/png")
    print("  - privacy_energy_tradeoff.pdf/png")
    print("  - roc_curves.pdf/png")
    print("  - metrics_report.txt")
    print("  - metrics_table.tex (for LaTeX)")


# ============================================================================
# MAIN — Generate plots for both datasets
# ============================================================================
if __name__ == '__main__':
    # 1) Held-out test set (9K samples)
    generate_all_plots(
        dataset_path='data/held_out_test.jsonl',
        output_dir='research_plots',
        dataset_label='9K Held-Out Test',
    )

    print("\n\n")

    # 2) Full training dataset (75K samples)
    generate_all_plots(
        dataset_path='data/local_dataset.jsonl',
        output_dir='research_plots_75k',
        dataset_label='75K Training Samples',
    )

    print("\n" + "=" * 80)
    print("ALL DONE — Both plot sets generated!")
    print("  research_plots/      ← 9K held-out test samples")
    print("  research_plots_75k/  ← 75K training samples")
    print("=" * 80)
    print("\nReady for your research paper!")
