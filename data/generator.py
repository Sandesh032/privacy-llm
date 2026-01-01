"""
Dataset generator for adaptive routing training
Creates balanced dataset with local/hybrid/cloud routing decisions
"""

import json
import uuid
from collections import Counter

from prompt_generator import generate_prompt
from system_simulator import simulate_device, simulate_energy_latency
from oracle import compute_privacy_risk, evaluate_routes, choose_best


# Configuration
OUTPUT = "adaptive_dataset.jsonl"
TARGET_PER_CLASS = 17000  # ~51,000 total samples (balanced)
NETWORK_MAP = {"wifi": 0, "4g": 1, "5g": 2}


def generate_balanced_dataset(output_path=OUTPUT, target_per_class=TARGET_PER_CLASS):
    """
    Generate balanced dataset with equal samples per route
    
    Args:
        output_path: Path to save JSONL file
        target_per_class: Number of samples per route class
        
    Returns:
        dict: Final class distribution
    """
    counts = {"local": 0, "hybrid": 0, "cloud": 0}
    
    print("="*80)
    print("Starting Balanced Dataset Generation")
    print("="*80)
    print(f"Target per class: {target_per_class:,}")
    print(f"Total target samples: {target_per_class * 3:,}")
    print()

    with open(output_path, "w") as f:
        # Loop until all classes reach target
        while min(counts.values()) < target_per_class:
            prompt = generate_prompt()
            device = simulate_device()
            energy = simulate_energy_latency(device)

            privacy_risk = compute_privacy_risk(prompt["pii_types"])
            routes = evaluate_routes(privacy_risk, energy)
            optimal_route = choose_best(routes)

            # Only save if this class still needs samples
            if counts[optimal_route] < target_per_class:
                # Flatten device features for training
                device_feat_vector = [
                    device["battery_level"],
                    device["cpu_load"],
                    device["ram_mb"] / 8192.0,  # Normalize RAM
                    NETWORK_MAP[device["network"]]
                ]

                record = {
                    "id": str(uuid.uuid4()),
                    "query_text": prompt["query_text"],
                    "intent": prompt["intent"],
                    "pii_types": prompt["pii_types"],
                    "privacy_risk": round(privacy_risk, 3),
                    "device": device,
                    "device_vector": device_feat_vector,
                    "energy": energy,
                    "routes": routes,
                    "optimal_route": optimal_route,
                    "label": ["local", "hybrid", "cloud"].index(optimal_route)
                }

                f.write(json.dumps(record) + "\n")
                counts[optimal_route] += 1

                # Progress update
                total = sum(counts.values())
                if total % 5000 == 0:
                    print(f"Progress: {total:,}/{target_per_class * 3:,} samples")
                    print(f"  Distribution: {counts}")

    print("\n" + "="*80)
    print("Dataset Generation Complete!")
    print("="*80)
    print(f"Final distribution: {counts}")
    print(f"Total samples: {sum(counts.values()):,}")
    print(f"Saved to: {output_path}")
    print("="*80)
    
    return counts


if __name__ == "__main__":
    generate_balanced_dataset()
