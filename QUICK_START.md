# 🚀 Quick Reference Card

## One-Command Pipeline

```bash
cd /home/sandeshpandey/PyCharmMiscProject/privacy_llm_model
source venv/bin/activate
./run_pipeline.sh
```

---

## Manual Steps

```bash
# 1. Generate dataset
cd data && python generator.py && cd ..

# 2. Train model
python training.py

# 3. Evaluate
python evaluation.py
```

---

## Project Structure

```
├── data/                      # Data generation
│   ├── prompt_generator.py    # Queries with PII
│   ├── system_simulator.py    # Device simulation
│   ├── oracle.py              # Optimal routing
│   ├── generator.py           # Dataset creation
│   └── adaptive_dataset_loader.py
├── models/
│   └── routing_model.py       # BERT classifier
├── training.py                # Train script
├── evaluation.py              # Eval & metrics
└── checkpoints/
    └── best_model.pt          # Trained model
```

---

## Key Commands

```bash
# Install dependencies
pip install torch transformers faker tqdm numpy

# Generate 51K balanced samples
cd data && python generator.py

# Train (1-3 hours)
python training.py

# Get X% and Y% metrics
python evaluation.py

# Test individual modules
python data/prompt_generator.py
python data/system_simulator.py
python data/oracle.py
```

---

## Expected Results

| Metric | Value |
|--------|-------|
| Accuracy | 85-95% |
| Energy Reduction (X) | 30-50% |
| Privacy Improvement (Y) | 40-60% |
| Quality Loss | < 5% |

---

## Module Functions

```python
# Data Generation
from data.prompt_generator import generate_prompt
from data.system_simulator import simulate_device
from data.oracle import choose_best

# Dataset
from data.adaptive_dataset_loader import AdaptiveRoutingDataset

# Model
from models.routing_model import AdaptiveRoutingModel

# Usage
model = AdaptiveRoutingModel()
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
```

---

## Configuration Tweaks

### More Privacy-Focused
```python
# data/oracle.py
alpha = 0.7  # Higher privacy weight
beta = 0.3
```

### Larger Dataset
```python
# data/generator.py
TARGET_PER_CLASS = 25000  # 75K total
```

### Longer Training
```python
# training.py
CONFIG['epochs'] = 30
```

---

## File Locations

| What | Where |
|------|-------|
| Dataset | `data/adaptive_dataset.jsonl` |
| Model | `checkpoints/best_model.pt` |
| History | `checkpoints/training_history.json` |
| Docs | `README_PROJECT.md` |

---

## Quick Checks

```bash
# Dataset size
wc -l data/adaptive_dataset.jsonl

# Model exists
ls -lh checkpoints/best_model.pt

# View accuracy
python -c "import torch; print(torch.load('checkpoints/best_model.pt')['val_accuracy'])"

# List all modules
find . -name "*.py" | grep -v venv | grep -v __pycache__
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size to 16 |
| Low accuracy | Increase epochs to 30 |
| Imbalanced results | Adjust class_weights |
| Slow training | Use GPU or reduce model size |

---

## Documentation

- **Full Guide**: `README_PROJECT.md`
- **Refactoring Summary**: `REFACTORING_SUMMARY.md`
- **This Card**: `QUICK_START.md`

---

**Ready to train!** 🚀

```bash
./run_pipeline.sh
```

