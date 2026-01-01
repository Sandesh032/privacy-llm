# 🚀 Adaptive LLM Routing Project

## Overview

An adaptive decision framework that dynamically routes user queries between on-device (local), cloud-based, and hybrid LLM processing to jointly optimize **privacy exposure** and **energy consumption**.

---

## 📁 Project Structure

```
privacy_llm_model/
│
├── data/                           # Data generation & loading
│   ├── prompt_generator.py         # Generates queries with PII
│   ├── system_simulator.py         # Simulates device conditions
│   ├── oracle.py                   # Computes optimal routing decisions
│   ├── generator.py                # Main dataset generator
│   ├── adaptive_dataset_loader.py  # PyTorch dataset loader
│   └── adaptive_dataset.jsonl      # Generated dataset (51K samples)
│
├── models/                         # Neural network models
│   └── routing_model.py            # BERT-based routing classifier
│
├── training.py                     # Training script
├── evaluation.py                   # Evaluation & metrics
│
├── checkpoints/                    # Saved model weights
│   └── best_model.pt               # Best trained model
│
└── venv/                          # Virtual environment
```

---

## 🎯 Three Routing Options

| Route | Privacy | Energy | Quality | Best For |
|-------|---------|--------|---------|----------|
| **Local** | ✅ High (stays on device) | ❌ High consumption | ⚠️ Lower (limited compute) | Sensitive PII data |
| **Hybrid** | ⚖️ Medium (partial offload) | ⚖️ Medium | ⚖️ Good | Balanced scenarios |
| **Cloud** | ❌ Low (data transmitted) | ✅ Low | ✅ Best (powerful servers) | General queries |

---

## 🔄 Complete Workflow

### **Step 1: Generate Balanced Dataset**

```bash
cd /home/sandeshpandey/PyCharmMiscProject/privacy_llm_model
source venv/bin/activate
cd data
python generator.py
```

**Output**: `data/adaptive_dataset.jsonl` with ~51,000 samples (17K per class)

**What it generates:**
- Synthetic queries with PII (emails, phone, location, medical info)
- Device context (battery, CPU, network type)
- Energy and latency costs
- Optimal routing decision for each query

---

### **Step 2: Train the Model**

```bash
python training.py
```

**What happens:**
- Loads balanced dataset
- Splits: 70% train / 15% val / 15% test
- Initializes BERT-based routing model (~110M parameters)
- Trains for 20 epochs with early stopping
- Saves best model to `checkpoints/best_model.pt`

**Expected Results:**
- Training time: 1-3 hours (depends on GPU/CPU)
- Validation accuracy: 85-95%
- Per-route accuracy: 80-90% each

---

### **Step 3: Evaluate & Get Metrics**

```bash
python evaluation.py
```

**Outputs:**
- Overall accuracy
- Privacy risk reduction (Y%)
- Energy usage reduction (X%)
- Task quality maintenance
- Comparison vs baselines (always-cloud, always-local)

**Example Output:**
```
IMPROVEMENTS vs ALWAYS-CLOUD BASELINE
================================================================================
  Energy Usage:  +42.3% (X = 42.3%)
  Privacy Risk:  +58.7% (Y = 58.7%)
  Task Quality:  -3.2%

"Our system reduces energy usage by 42.3% and privacy risk
by 58.7% compared to always-cloud baseline, while maintaining
comparable task performance (-3.2% quality change)."
```

---

## 📊 Module Descriptions

### **1. Data Generation (`data/`)**

#### `prompt_generator.py`
- Generates realistic queries using Faker library
- 60% PII injection rate
- 5 intent types: information_retrieval, task_execution, personal_assistant, health_query, financial_query

#### `system_simulator.py`
- Simulates device conditions:
  - Battery level: 15-100%
  - CPU load: 10-90%
  - RAM: 2GB/4GB/8GB
  - Network: wifi/4g/5g
- Computes energy costs (local vs transmission)
- Calculates latency by network type

#### `oracle.py`
- Computes privacy risk based on PII sensitivity
- Evaluates all 3 routes with costs
- Selects optimal route: `argmin[Quality - α·Privacy - β·Energy]`

#### `generator.py`
- Orchestrates dataset generation
- Ensures balanced class distribution
- Outputs JSONL format with all features

#### `adaptive_dataset_loader.py`
- PyTorch Dataset wrapper
- Custom collate function for batching
- Converts to tensors for training

---

### **2. Model (`models/`)**

#### `routing_model.py`
- **Architecture**:
  ```
  Query Text → BERT Encoder → [CLS] token (768-dim)
                                      ↓
  Device Features → MLP (4→64→128)
                                      ↓
              Concatenate → Fusion Layer (896→256)
                                      ↓
                          Routing Head (256→3)
                                      ↓
                    [P(local), P(hybrid), P(cloud)]
  ```

- **Features**:
  - Pre-trained BERT for text understanding
  - Device context integration
  - Dropout & batch normalization
  - ~110M trainable parameters

---

### **3. Training (`training.py`)**

- **Loss**: Weighted Cross-Entropy (boost hybrid class)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing
- **Metrics**: Accuracy, per-route accuracy
- **Early Stopping**: Saves best validation model
- **Logging**: Progress bars, epoch summaries

---

### **4. Evaluation (`evaluation.py`)**

Computes comprehensive metrics:

1. **Model Performance**:
   - Average privacy risk
   - Average energy cost
   - Average task quality

2. **Baselines**:
   - Always-cloud strategy
   - Always-local strategy
   - Optimal oracle

3. **Improvements**:
   - Energy reduction (X%)
   - Privacy improvement (Y%)
   - Quality maintenance

---

## 🚀 Quick Start Guide

### **Installation**

```bash
cd /home/sandeshpandey/PyCharmMiscProject/privacy_llm_model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers faker tqdm numpy
```

### **Full Pipeline**

```bash
# 1. Generate dataset (if not already done)
cd data
python generator.py
cd ..

# 2. Train model
python training.py

# 3. Evaluate
python evaluation.py
```

---

## ⚙️ Configuration

### Training Parameters (in `training.py`)

```python
CONFIG = {
    'dataset_path': 'data/adaptive_dataset.jsonl',
    'batch_size': 32,
    'learning_rate': 2e-5,
    'epochs': 20,
    'train_split': 0.7,      # 70% training
    'val_split': 0.15,       # 15% validation
    'test_split': 0.15,      # 15% test
    'class_weights': [1.0, 3.0, 0.8]  # Boost hybrid
}
```

### Oracle Weights (in `data/oracle.py`)

```python
alpha = 0.5  # Privacy importance
beta = 0.5   # Energy importance
# Task quality is always maximized
```

**Tuning Guide:**
- **More privacy-focused**: Increase α (e.g., 0.7)
- **More energy-efficient**: Increase β (e.g., 0.7)
- **More quality-focused**: Decrease both (e.g., 0.3 each)

---

## 📈 Expected Performance

| Metric | Target | Excellent |
|--------|--------|-----------|
| **Overall Accuracy** | > 85% | > 90% |
| **Energy Reduction** | 30-50% | > 50% |
| **Privacy Improvement** | 40-60% | > 60% |
| **Quality Maintenance** | Within -5% | Within -3% |

---

## 🔍 Dataset Format

Each sample in `adaptive_dataset.jsonl`:

```json
{
  "id": "uuid",
  "query_text": "Send the report to john@example.com",
  "intent": "task_execution",
  "pii_types": ["email"],
  "privacy_risk": 0.500,
  "device": {
    "battery_level": 0.73,
    "cpu_load": 0.45,
    "ram_mb": 4096,
    "network": "wifi"
  },
  "device_vector": [0.73, 0.45, 0.5, 0.0],
  "energy": {
    "latency_ms": 35.42,
    "tx_energy": 0.377,
    "local_energy": 0.480
  },
  "routes": {
    "local": {"privacy_leakage": 0.100, "energy_cost": 0.480, "task_quality": 0.67},
    "hybrid": {"privacy_leakage": 0.375, "energy_cost": 0.428, "task_quality": 0.85},
    "cloud": {"privacy_leakage": 0.800, "energy_cost": 0.377, "task_quality": 0.95}
  },
  "optimal_route": "hybrid",
  "label": 1
}
```

---

## 🧪 Testing Individual Modules

### Test Data Generation
```bash
cd data
python prompt_generator.py  # Generates 5 example prompts
python system_simulator.py   # Simulates 5 device states
python oracle.py             # Tests routing selection
```

### Test Model
```python
from models.routing_model import AdaptiveRoutingModel
import torch

model = AdaptiveRoutingModel()
queries = ["What is AI?"]
device_features = torch.tensor([[0.7, 0.4, 0.5, 0.0]])
routes, probs = model.predict_route(queries, device_features)
print(f"Predicted route: {['local', 'hybrid', 'cloud'][routes[0]]}")
```

---

## 📊 Monitoring Training

Watch training progress:
```bash
# Real-time progress bars show:
# - Current epoch
# - Batch loss and accuracy
# - Per-route validation accuracy
# - Learning rate decay
```

Check saved results:
```bash
# View training history
cat checkpoints/training_history.json

# Load best model
python -c "import torch; print(torch.load('checkpoints/best_model.pt')['val_accuracy'])"
```

---

## 🎯 Paper Contributions

This project enables you to claim:

1. **Adaptive Routing Framework**: Dynamic query routing based on privacy-energy trade-offs
2. **Learned Decision Policy**: Neural network learns optimal routing from data
3. **Multi-Objective Optimization**: Balances privacy, energy, and task quality
4. **Quantitative Improvements**: X% energy reduction, Y% privacy improvement
5. **Device-Aware**: Considers battery, CPU, network conditions

---

## 🛠️ Troubleshooting

### Issue: Out of memory during training
```python
# Reduce batch size in training.py
CONFIG['batch_size'] = 16  # or 8
```

### Issue: Low accuracy on hybrid class
```python
# Increase class weight in training.py
CONFIG['class_weights'] = [1.0, 5.0, 0.8]
```

### Issue: Model overfits
- Add more dropout
- Increase weight decay
- Reduce model size
- Generate more training data

---

## 📚 Dependencies

```
torch>=2.0.0
transformers>=4.30.0
faker>=18.0.0
tqdm>=4.65.0
numpy>=1.24.0
```

---

## ✅ Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Dataset generated (~51K samples)
- [ ] Dataset is balanced (17K per class)
- [ ] Model training completed
- [ ] Best model saved to checkpoints/
- [ ] Evaluation run successfully
- [ ] Got X% and Y% improvements
- [ ] Results documented for paper

---

**Ready to train your adaptive routing model!** 🚀

For questions or issues, check individual module docstrings or reach out to the team.

