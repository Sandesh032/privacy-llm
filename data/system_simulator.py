import random

NETWORKS = ["wifi", "4g", "5g"]

def simulate_device():
    return {
        "battery_level": round(random.uniform(0.15, 1.0), 2),
        "cpu_load": round(random.uniform(0.1, 0.9), 2),
        "ram_mb": random.choice([2048, 4096, 8192]),
        "network": random.choice(NETWORKS)
    }


def simulate_energy_latency(device):
    net = device["network"]

    latency = {
        "wifi": random.uniform(20, 60),
        "4g": random.uniform(60, 140),
        "5g": random.uniform(15, 40)
    }[net]

    tx_energy = 0.2 + latency / 200
    local_energy = 0.3 + device["cpu_load"] * 0.4

    return {
        "latency_ms": round(latency, 2),
        "tx_energy": round(tx_energy, 3),
        "local_energy": round(local_energy, 3)
    }
