"""
Oracle for computing optimal routing decisions
Evaluates privacy risk and selects best route based on multi-objective optimization
"""

import random

PII_SENSITIVITY = {
    "email": 0.5,
    "phone": 0.6,
    "location": 0.7,
    "medical": 0.95
}


def compute_privacy_risk(pii_types):
    """
    Compute privacy risk score based on PII types present
    
    Args:
        pii_types: List of PII type strings
        
    Returns:
        float: Privacy risk score [0, 1]
    """
    if not pii_types:
        return 0.05
    return sum(PII_SENSITIVITY[p] for p in pii_types) / len(pii_types)


def evaluate_routes(pii_risk, energy):
    """
    Evaluate all routing options with their costs
    
    Args:
        pii_risk: Privacy risk score
        energy: Dict with 'local_energy' and 'tx_energy'
        
    Returns:
        dict: Routes with privacy_leakage, energy_cost, task_quality
    """
    routes = {}

    routes["local"] = {
        "privacy_leakage": 0.1 + 0.1 * pii_risk,
        "energy_cost": energy["local_energy"],
        "task_quality": min(0.65 + random.uniform(-0.05, 0.05), 1.0)
    }

    routes["hybrid"] = {
        "privacy_leakage": 0.25 + 0.25 * pii_risk,
        "energy_cost": (energy["local_energy"] + energy["tx_energy"]) / 1.8,
        "task_quality": min(0.88 + random.uniform(-0.03, 0.03), 1.0)
    }

    routes["cloud"] = {
        "privacy_leakage": 0.6 + 0.4 * pii_risk,
        "energy_cost": energy["tx_energy"],
        "task_quality": min(0.95 + random.uniform(-0.03, 0.03), 1.0)  # Fixed: capped at 1.0
    }

    return routes


def choose_best(routes, alpha=None, beta=None):
    """
    Choose optimal route using balanced policy
    
    Args:
        routes: Dict of route evaluations
        alpha: Privacy importance weight (default: 0.5)
        beta: Energy importance weight (default: 0.5)
        
    Returns:
        str: Optimal route name ('local', 'hybrid', or 'cloud')
    """
    # Balanced system policy
    if alpha is None:
        alpha = 0.5
    if beta is None:
        beta = 0.5

    best, best_score = None, -1e9
    for r, v in routes.items():
        score = v["task_quality"] - alpha * v["privacy_leakage"] - beta * v["energy_cost"]
        if score > best_score:
            best, best_score = r, score

    return best
