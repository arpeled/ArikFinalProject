# Class Imbalance Improvements & Training Feedback Enhancement

**Date:** 2025-12-29
**Objective:** Improve model sensitivity to rare classes and enhance AI-driven auto-tuning through better training feedback

---

## 📋 Overview

This document describes the implementation of dynamic class weight computation, training/validation loss tracking, and enhanced AI analysis for the chest X-ray classification pipeline.

### Key Problems Addressed

1. **Class Imbalance:** Rare diseases (e.g., Hernia: 0.17%, Pneumonia: 1.0%) were poorly detected
2. **Limited Training Feedback:** AI advisor lacked information about overfitting/underfitting
3. **Static Weights:** Class weights were computed once from config, not from actual training data
4. **Poor Generalization:** Model couldn't distinguish between overfitting and legitimate convergence

---

## 🔧 Implementation Details

### 1. Dynamic Class Weight Computation

**File:** `config_based_pipeline.py`

**Function Added:**
```python
def compute_class_weights(labels_tensor, epsilon=1e-6):
    """
    Dynamically compute class weights based on training label distribution.

    Uses inverse frequency weighting to give more importance to rare classes.
    """
    class_counts = labels_tensor.sum(dim=0)
    total = class_counts.sum()
    frequencies = class_counts / (total + epsilon)
    min_freq = frequencies[frequencies > 0].min()
    weights = min_freq / (frequencies + epsilon)
    return weights
```

**How It Works:**
- Counts positive samples for each of the 14 disease classes
- Computes frequency of each class relative to total positive samples
- Calculates inverse frequency weights (rare classes get higher weights)
- Returns tensor of shape (14,) with weights for each class

**Benefits:**
- Weights are computed from **actual training data**, not estimated frequencies
- Automatically adapts if training data distribution changes
- More accurate than hardcoded frequencies from config

**Usage:**
Enable in your config file:
```yaml
loss:
  type: FocalLoss
  gamma: 2.0
  use_dynamic_weights: true  # NEW: enables dynamic computation
```

---

### 2. Training/Validation Loss Tracking

**Files Modified:**
- `config_based_pipeline.py` - Training pipeline
- `auto_improvement_loop.py` - Iteration tracking

**Changes:**

#### A. Training Pipeline Returns Losses
```python
# config_based_pipeline.py (line ~511)
return model, final_train_loss, final_val_loss
```

**Tracking:**
- `final_train_loss`: Average training loss from the last epoch
- `final_val_loss`: Average validation loss from the last epoch

These values are logged at the end of training:
```
Final Train Loss: 0.0234
Final Val Loss: 0.0289
```

#### B. Auto-Improvement Loop Captures Losses
```python
# auto_improvement_loop.py (line ~299)
model, train_loss, val_loss = trainer.train()

# Stored in iteration summary (line ~359-360)
'train_loss': float(train_loss),
'val_loss': float(val_loss),
```

**Persistence:**
Losses are saved to `iteration_summary.json` for each iteration:
```json
{
  "iteration": 1,
  "avg_auc": 0.7895,
  "avg_f1": 0.1234,
  "train_loss": 0.0234,
  "val_loss": 0.0289,
  ...
}
```

---

### 3. AI Advisor Enhanced with Loss Analysis

**File Modified:** `ai_advisor.py`

**New Functionality:**

#### A. Loss-Based Overfitting/Underfitting Detection

The AI advisor now receives `train_loss` and `val_loss` and automatically analyzes them:

```python
loss_ratio = val_loss / train_loss

if loss_ratio > 1.2:
    # OVERFITTING DETECTED
    # Suggests: more dropout, weight decay, data augmentation

elif loss_ratio > 1.1:
    # Possible overfitting

elif train_loss > val_loss * 1.1:
    # UNDERFITTING
    # Suggests: more epochs, lower dropout, higher learning rate

else:
    # Balanced training/validation losses ✓
```

**Example Output in AI Prompt:**
```
TRAINING/VALIDATION LOSS:
  Final Training Loss: 0.0234
  Final Validation Loss: 0.0289
  ⚠️  Possible overfitting: Val loss is 23.5% higher than train loss
      Consider: more dropout, weight decay, or data augmentation
```

#### B. Adaptive Focal Loss Tuning

The AI advisor now includes rules for adaptive focal loss tuning based on F1-score:

**System Prompt Additions:**
```
ADAPTIVE FOCAL LOSS TUNING:
- If F1-score is very low (<0.05), model is not sensitive enough:
  * Decrease gamma (try max(1.5, current_gamma - 1))
  * Increase alpha (try min(0.95, current_alpha + 0.1))
  * Enable dynamic class weights with 'use_dynamic_weights: true'

- If validation loss >> training loss (overfitting):
  * Increase dropout rate
  * Add weight decay to optimizer

- If training loss >> validation loss (underfitting):
  * Decrease dropout rate
  * Increase epochs
```

**How It Works:**
1. AI advisor receives current F1-score and loss values
2. Automatically suggests gamma/alpha adjustments if F1 < 0.05
3. Balances between focusing on hard examples (high gamma) vs. rare classes (alpha weights)

#### C. Historical Loss Tracking

Previous iterations' losses are now shown in the AI prompt:

**Summary Table:**
```
| Iter | Avg AUC | Avg F1 | Train Loss | Val Loss | Config Changes |
|------|---------|--------|------------|----------|----------------|
|   1  | 0.7895  | 0.1234 | 0.0234     | 0.0289   | baseline       |
|   2  | 0.8012  | 0.1456 | 0.0198     | 0.0312   | gamma: 2→3     |
```

**Detailed Recent Iterations:**
```
Iteration 2:
  Metrics: AUC=0.8012, F1=0.1456, Recall=0.3456, Precision=0.0987
  Losses: Train=0.0198, Val=0.0312 (OVERFITTING)
  Changes applied: {...}
```

This helps the AI advisor:
- Track whether changes improved or degraded performance
- Identify patterns (e.g., "increasing gamma reduced overfitting")
- Avoid repeating failed experiments

---

## 📊 Expected Improvements

### Before These Changes:
- ❌ Static class weights from config (could be inaccurate)
- ❌ No visibility into overfitting/underfitting
- ❌ AI suggestions were based only on test metrics (AUC, F1)
- ❌ Hard to detect if model needed more regularization or capacity

### After These Changes:
- ✅ Dynamic class weights computed from actual training data
- ✅ Automatic detection of overfitting (val_loss > 1.2 × train_loss)
- ✅ Automatic detection of underfitting (train_loss > 1.1 × val_loss)
- ✅ AI advisor suggests dropout/epochs/LR adjustments based on loss patterns
- ✅ Adaptive focal loss tuning for very low F1-scores (<0.05)
- ✅ Historical tracking of all losses for trend analysis

---

## 🚀 Usage Guide

### Enabling Dynamic Class Weights

**Step 1:** Update your config file (e.g., `config_baseline.yaml`):
```yaml
loss:
  type: FocalLoss
  gamma: 2.0
  use_dynamic_weights: true  # Enable dynamic computation
  # use_class_weights: true  # Old static method (deprecated)
  # class_frequencies: [...]  # No longer needed with dynamic weights
```

**Step 2:** Run training as usual:
```bash
python auto_improvement_loop.py --config config_baseline.yaml --iterations 10
```

**Step 3:** Check logs for dynamic weight computation:
```
Computing dynamic class weights from training data...
Dynamic weights computed: min=1.0000, max=143.5294, mean=23.4567
```

### Interpreting Loss Feedback

**Example 1: Overfitting Detected**
```
Final Train Loss: 0.0198
Final Val Loss: 0.0289
⚠️  OVERFITTING DETECTED: Val loss is 46.0% higher than train loss
```
**Action:** AI will suggest increasing dropout, adding weight decay, or more augmentation

**Example 2: Underfitting Detected**
```
Final Train Loss: 0.0567
Final Val Loss: 0.0412
⚠️  UNDERFITTING: Train loss is 37.6% higher than val loss
```
**Action:** AI will suggest more epochs, lower dropout, or higher learning rate

**Example 3: Balanced Training**
```
Final Train Loss: 0.0234
Final Val Loss: 0.0251
✓ Balanced training/validation losses
```
**Action:** Focus on other improvements (augmentation, threshold tuning, etc.)

### Adaptive Focal Loss Tuning

If your model has very low F1-score (<0.05), the AI advisor will automatically suggest:

**Example AI Suggestion:**
```json
{
  "loss": {
    "gamma": 1.5,  // Reduced from 2.0 to focus less on hard examples
    "use_dynamic_weights": true  // Enable dynamic class weights
  },
  "reasoning": "F1-score of 0.032 indicates poor sensitivity to positive classes.
                Reducing gamma to 1.5 and enabling dynamic class weights will give
                more weight to rare classes and improve recall."
}
```

---

## 📈 Validation & Testing

### Manual Test
```python
# Test dynamic class weight computation
import torch
import numpy as np
from config_based_pipeline import compute_class_weights

# Simulate imbalanced labels (1000 samples, 14 classes)
# Class 0: 500 positives (common)
# Class 1: 5 positives (very rare)
labels = torch.zeros(1000, 14)
labels[:500, 0] = 1  # Common class
labels[:5, 1] = 1     # Rare class

weights = compute_class_weights(labels)
print(f"Weight for common class (500/505): {weights[0]:.4f}")
print(f"Weight for rare class (5/505): {weights[1]:.4f}")
# Expected: rare class weight should be ~100x higher
```

### Integration Test
```bash
# Run a single iteration and verify losses are logged
python auto_improvement_loop.py --config config_baseline.yaml --iterations 1

# Check iteration_summary.json
cat auto_improvement_runs/iteration_001/iteration_summary.json | jq '.train_loss, .val_loss'
```

---

## 🔍 Implementation Notes

### FocalLoss Already Supports Tensor Alpha
The existing `FocalLoss` class in both `chest_xray_train_pipeline.py` and `config_based_pipeline.py` already supports `alpha` as a tensor:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Can be a tensor of shape (num_classes,)
```

This means dynamic class weights are **fully compatible** with the existing loss function.

### Label Column Order
The 14 disease classes are expected in this order:
```python
['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
 'Pleural_Thickening', 'Hernia']
```

Ensure your CSV file has these columns in the correct order.

### Backward Compatibility
- If `use_dynamic_weights` is not set or `False`, the old static method is used
- If `train_labels` are not provided, static weights from config are used as fallback
- If `train_loss`/`val_loss` are not available (old iterations), AI advisor still works

---

## 📝 Summary of Files Changed

| File | Changes | Lines Modified |
|------|---------|----------------|
| `config_based_pipeline.py` | Added `compute_class_weights()`, dynamic weight support, loss tracking | ~25-54, 168-219, 351-364, 444, 470, 507-511 |
| `auto_improvement_loop.py` | Capture train/val loss, log in summary | 299, 303-304, 359-360, 386-387 |
| `ai_advisor.py` | Accept train/val loss, detect over/underfitting, adaptive tuning, historical tracking | 40-41, 52-53, 59, 159-171, 204-238, 260-282, 296-305 |

**Total Lines Added:** ~150
**Total Lines Modified:** ~50

---

## 🎯 Next Steps

1. **Test the implementation** with a small iteration run
2. **Monitor F1-score improvements** in rare classes (Hernia, Pneumonia, Fibrosis)
3. **Analyze AI suggestions** to ensure they adapt to overfitting/underfitting
4. **Compare dynamic vs. static weights** by running parallel experiments
5. **Document results** in final project report with before/after metrics

---

## 🙏 Acknowledgments

This implementation addresses critical issues in medical image classification:
- **Class imbalance** is a major challenge in ChestX-ray14 dataset
- **Dynamic weighting** adapts to actual training data distribution
- **Loss-based feedback** enables smarter AI-driven optimization
- **Adaptive focal loss tuning** specifically targets low F1-score problem

These improvements aim to reduce **false negatives** (missing diseases) by improving model sensitivity to rare classes, which is critical for medical diagnosis applications.

---

**Generated:** 2025-12-29
**Author:** AI Code Assistant
**Status:** ✅ Implementation Complete, Ready for Testing
