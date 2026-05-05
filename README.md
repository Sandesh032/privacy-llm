# Privacy-Aware Adaptive LLM Routing

A neural network system that intelligently routes LLM queries to the optimal processing backend — **local**, **hybrid**, or **cloud** — by jointly optimizing for privacy, energy efficiency, and task quality.

---

## Overview

When a user sends a query to an AI assistant, the system must decide:

- **Local** — process entirely on-device (best privacy, limited quality)
- **Hybrid** — split processing between device and cloud
- **Cloud** — send to a remote LLM (best quality, highest privacy risk + energy cost)

This project trains a routing model that learns to make this decision based on:
- The **query text** (does it contain PII like email, phone, location, or medical data?)
- The **device state** (battery level, CPU load, RAM, network type)
- A computed **privacy risk score**

### Key Results (9K held-out test set)

| Metric | Value |
|--------|-------|
| Overall Accuracy | **99.34%** |
| Macro F1-Score | **0.9934** |
| Privacy Risk Reduction vs Always-Cloud | **−45.77%** |
| Energy Cost (vs Always-Cloud) | Balanced |

---

## Project Structure

```
privacy-llm/
├── data/
│   ├── dataset.py             # Dataset generator (Gemini API or local Gemma 2B)
│   ├── adaptive_dataset.py    # PyTorch Dataset & DataLoader
│   ├── generation.py          # Utility generation helpers
│   ├── analyze.py             # Dataset analysis tools
│   ├── verify_improvements.py # Verification helpers
│   ├── local_dataset.jsonl    # Training dataset (generated)
│   └── held_out_test.jsonl    # Held-out evaluation dataset
├── models/
│   └── routing_model.py       # AdaptiveRoutingModel (BERT + MLP fusion)
├── training.py                # Model training script
├── evaluation.py              # Evaluation vs baselines + held-out data generation
├── generate_plots.py          # Research visualizations
├── research_plots/            # Plots from held-out test set
├── research_plots_training/   # Plots from training dataset
└── research_plots_evaluation/ # Full evaluation plots with baselines
```

---

## Model Architecture

```
Query Text  ──► BERT Encoder (bert-base-uncased, last 2 layers fine-tuned)
                      │
                      ▼  [CLS] token (768-dim)
                  ┌───┴────────────────────┐
                  │       Fusion Layer      │
Device Features ──► Device MLP (5→128→256→128)
(battery, CPU,   │       └─ concat ─┘      │
 RAM, network,   │  Linear(896→256) + BN   │
 privacy_risk)   └───────────┬─────────────┘
                             │
                      Routing Head
                      Linear(256→3)
                             │
                   [Local | Hybrid | Cloud]
```

---

## Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.x | Model training & inference |
| Transformers (HuggingFace) | 4.x | BERT encoder, Gemma 2B |
| google-genai | Latest | Gemini API for dataset generation |
| Faker | Latest | Synthetic PII generation |
| scikit-learn | Latest | Evaluation metrics (F1, AUC, confusion matrix) |
| matplotlib | Latest | Plotting |
| seaborn | Latest | Heatmaps and styled plots |
| pandas | Latest | Data handling and report tables |
| tqdm | Latest | Progress bars |
| numpy | Latest | Numerical operations |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd privacy-llm

# Install dependencies
pip install torch transformers google-genai faker scikit-learn \
            matplotlib seaborn pandas tqdm numpy
```

---

## Instructions to Execute

### Step 1 — Generate the Training Dataset

The dataset is generated synthetically using either the **Google Gemini API** (fast, free tier) or a local **Gemma 2B** model as a fallback.

**Option A: Using Google Gemini API (recommended)**
```bash
export GOOGLE_API_KEY="your_api_key_here"
python data/dataset.py
```

**Option B: Using local Gemma 2B (no API key needed)**
```bash
# If GOOGLE_API_KEY is not set, the local model is used automatically
python data/dataset.py
```

This generates `data/local_dataset.jsonl` with balanced samples across three routing classes (local, hybrid, cloud). Each record includes the query text, device state, PII types, privacy/energy/quality costs per route, and the optimal route label.

---

### Step 2 — Train the Routing Model

```bash
python training.py
```

Training configuration (editable in `training.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Batch size |
| `learning_rate` | 5e-5 | AdamW learning rate |
| `epochs` | 15 | Max epochs (early stopping applies) |
| `train_split` | 0.70 | Training data fraction |
| `val_split` | 0.15 | Validation data fraction |
| `patience` | 5 | Early stopping patience |
| `device` | auto | `cuda` if available, else `cpu` |

The best model checkpoint is saved to `checkpoints/best_model.pt`. Training history is saved to `checkpoints/training_history.json`.

---

### Step 3 — Generate the Held-Out Test Dataset

This generates a **separate** dataset never seen during training for unbiased evaluation:

```bash
python evaluation.py --generate
```

Optional flags:
```bash
python evaluation.py --generate \
  --dataset data/held_out_test.jsonl \
  --target-per-class 3000        # 3000 samples × 3 classes = 9000 total
```

---

### Step 4 — Evaluate the Trained Model

```bash
python evaluation.py
```

Optional flags:
```bash
python evaluation.py \
  --model checkpoints/best_model.pt \
  --dataset data/held_out_test.jsonl
```

The evaluation compares the model against three baselines:
- **Always-Cloud**: always routes to cloud
- **Always-Local**: always routes to device
- **Optimal Oracle**: theoretical best routing

---

### Step 5 — Generate Research Plots

```bash
python generate_plots.py
```

This generates the following plots in `research_plots/` and `research_plots_training/`:

| Plot | Description |
|------|-------------|
| `training_curves.png` | Loss and accuracy over epochs |
| `confusion_matrix.png` | Predicted vs true route labels |
| `per_route_accuracy.png` | Per-class accuracy bar chart |
| `feature_distributions.png` | Input feature histograms by route |
| `privacy_energy_tradeoff.png` | Privacy vs energy scatter (bubble = quality) |
| `roc_curves.png` | Multi-class ROC curves with AUC |
| `metrics_report.txt` | Full classification report |
| `metrics_table.tex` | LaTeX table for papers |

---

## Dataset Format

Each line in the `.jsonl` files is a JSON record:

```json
{
  "id": "uuid",
  "query_text": "Send the report to john@example.com",
  "intent": "draft an email",
  "pii_types": ["email"],
  "privacy_risk": 0.5,
  "generation_method": "llm",
  "device": {
    "battery_level": 0.82,
    "cpu_load": 0.31,
    "ram_mb": 8192,
    "network": "wifi"
  },
  "energy": {
    "latency_ms": 45.2,
    "tx_energy": 0.423,
    "local_energy": 0.524
  },
  "routes": {
    "local":  { "privacy_leakage": 0.17, "energy_cost": 0.47, "task_quality": 0.69 },
    "hybrid": { "privacy_leakage": 0.43, "energy_cost": 0.52, "task_quality": 0.87 },
    "cloud":  { "privacy_leakage": 0.75, "energy_cost": 0.36, "task_quality": 0.96 }
  },
  "optimal_route": "local",
  "label": 0
}
```

**Labels**: `0` = Local, `1` = Hybrid, `2` = Cloud

---

## Routing Decision Logic

The optimal route is selected by maximizing a composite score:

```
score(r) = task_quality(r) − α × privacy_leakage(r) − β × energy_cost(r)
```

where `α` (privacy weight) and `β` (energy weight) are sampled uniformly from `[0.3, 0.7]` during dataset generation to produce a diverse, realistic label distribution.

---

## Baseline Comparison

| System | Privacy Risk | Energy Cost | Task Quality |
|--------|-------------|-------------|--------------|
| Always-Local | 0.1606 | 0.4600 | 0.7012 |
| Always-Cloud | 0.7123 | 0.3620 | 0.9598 |
| **Our Model** | **0.3863** | **0.4001** | **0.8512** |
| Optimal Oracle | 0.3862 | 0.4003 | 0.8512 |

Our model achieves **near-oracle performance** while reducing privacy risk by **45.77%** compared to the always-cloud baseline.
