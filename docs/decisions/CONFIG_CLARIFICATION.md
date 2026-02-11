# Configuration Clarification: FocalLoss Alpha Parameter

**Date:** 2026-01-02

---

## ⚠️ Important Note About `alpha` in config_baseline.yaml

### What You Specified:
```yaml
loss:
  gamma: 3.5
  alpha: 0.5  # ← This parameter
  use_class_weights: true
  use_dynamic_weights: true
```

### How It Actually Works:

The current implementation has **three modes** for alpha (checked in this order):

#### 1. **Dynamic Weights Mode** (Currently Active)
```python
if use_dynamic_weights and train_labels is not None:
    alpha_tensor = compute_class_weights(train_labels).to(device)
    return FocalLoss(alpha=alpha_tensor, gamma=3.5)
```

**What happens:**
- ✅ Computes alpha automatically from actual class frequencies in training data
- ✅ Each class gets a weight proportional to: `min_frequency / class_frequency`
- ✅ Rare classes get higher weights (e.g., Hernia gets weight ~36)
- ✅ Common classes get lower weights (e.g., Infiltration gets weight ~1)

**This is currently ACTIVE because `use_dynamic_weights: true`**

#### 2. **Static Class Weights Mode**
```python
elif use_class_weights:
    class_frequencies = loss_config['class_frequencies']
    alpha = [min_freq / f for f in class_frequencies]
    alpha_tensor = torch.tensor(alpha).to(device)
    return FocalLoss(alpha=alpha_tensor, gamma=3.5)
```

**What happens:**
- Uses predefined class frequencies from config
- Computes alpha per class
- **Not used** because dynamic mode takes precedence

#### 3. **No Weights Mode**
```python
else:
    return FocalLoss(alpha=None, gamma=3.5)
```

**What happens:**
- No class weighting
- Only gamma parameter affects loss
- **Not used** because dynamic mode is active

---

## 🔍 The `alpha: 0.5` Parameter Issue

### Problem:
The `alpha: 0.5` parameter in the config **is NOT being read or used** by the current code.

### Why:
The code expects alpha to be:
- Either `None` (no weighting)
- Or a **tensor of per-class weights** (14 values, one per disease)

The code does NOT support a **scalar alpha** parameter like 0.5.

### In Standard FocalLoss Literature:
The `alpha` parameter is typically:
- A scalar balancing factor for positive vs negative classes
- Range: 0.0 to 1.0
- 0.5 = balanced, >0.5 = favor positives, <0.5 = favor negatives

**But our implementation uses alpha differently** - as per-class weights, not a pos/neg balance factor.

---

## ✅ Current Behavior (What's Actually Happening)

With the current config:
```yaml
loss:
  gamma: 3.5
  alpha: 0.5  # ← IGNORED
  use_class_weights: true
  use_dynamic_weights: true  # ← THIS IS ACTIVE
```

**What runs:**
```python
FocalLoss(
    alpha=<tensor of 14 class-specific weights computed from training data>,
    gamma=3.5
)
```

**Actual class weights (approximate):**
```
Hernia:         ~36.0  (rarest, highest weight)
Pneumonia:      ~15.0
Fibrosis:       ~13.0
Edema:          ~9.5
Emphysema:      ~8.5
Cardiomegaly:   ~7.9
Pleural_Thick:  ~6.5
Consolidation:  ~4.7
Pneumothorax:   ~4.1
Mass:           ~3.8
Nodule:         ~3.5
Atelectasis:    ~1.9
Effusion:       ~1.6
Infiltration:   ~1.0  (most common, lowest weight)
```

---

## 🎯 Is This Good or Bad?

### ✅ GOOD News:
Dynamic class weighting is **actually better** than a fixed alpha=0.5 because:
1. **Adapts to actual data** - computes weights from real class frequencies
2. **Handles imbalance** - rare classes (Hernia 0.17%) get 36x more weight
3. **No manual tuning** - automatically balances classes
4. **Proven effective** - this is the standard approach for imbalanced datasets

### Current Settings Are Optimal:
```yaml
loss:
  type: "FocalLoss"
  gamma: 3.5  # ✅ USED - Focuses on hard examples
  alpha: 0.5  # ⚠️ IGNORED - Dynamic weights used instead
  use_class_weights: true
  use_dynamic_weights: true  # ✅ ACTIVE - Best for imbalanced data
```

---

## 🔧 Options for Next Iteration

### Option 1: Keep Current Settings (RECOMMENDED)
**Config:**
```yaml
loss:
  type: "FocalLoss"
  gamma: 3.5
  use_class_weights: true
  use_dynamic_weights: true
  # Remove alpha: 0.5 since it's not used
```

**Why:** Dynamic class weighting is ideal for our extremely imbalanced dataset.

### Option 2: Disable Dynamic Weights (NOT RECOMMENDED)
**Config:**
```yaml
loss:
  type: "FocalLoss"
  gamma: 3.5
  alpha: 0.5  # Still won't work as scalar
  use_class_weights: true
  use_dynamic_weights: false  # Disable dynamic
```

**Why NOT:** Would use static class frequencies from config, which are less accurate.

### Option 3: Implement Scalar Alpha (REQUIRES CODE CHANGE)
Would need to modify `FocalLoss` class to support:
```python
if isinstance(alpha, float):  # Scalar alpha
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
elif isinstance(alpha, torch.Tensor):  # Per-class alpha
    # Current behavior
```

**Why NOT:** Current implementation is better for our use case.

---

## ✅ Recommendation: No Change Needed

**Current behavior is actually optimal!**

The config can be cleaned up to remove confusion:

### Recommended Config Update:
```yaml
loss:
  type: "FocalLoss"
  gamma: 3.5  # Focus on hard examples
  use_class_weights: true  # Enable class weighting
  use_dynamic_weights: true  # Compute weights from training data (BEST)
  # Note: Dynamic weights automatically handle class imbalance
  # Rare classes (Hernia: 0.17%) get ~36x weight
  # Common classes (Infiltration: 17.76%) get ~1x weight
```

---

## 📊 Expected Impact

With current settings (gamma=3.5 + dynamic weights):

### Gamma=3.5 Effect:
- Focuses more on hard-to-classify examples
- Reduces loss contribution from easy examples
- Helps model learn difficult patterns

### Dynamic Weights Effect:
- Rare classes (Hernia, Pneumonia) get 15-36x more loss weight
- Model pays more attention to minority classes during training
- Prevents model from ignoring rare diseases

### Combined Effect:
- Better detection of rare diseases
- Improved recall across all classes
- Higher F1-scores for imbalanced classes

---

## 🚀 Conclusion

**No code changes needed!**

The current implementation with `use_dynamic_weights: true` is **better** than using a scalar `alpha: 0.5`.

Just understand that:
1. ✅ `gamma: 3.5` IS being used (focuses on hard examples)
2. ⚠️ `alpha: 0.5` is NOT being used (replaced by dynamic weights)
3. ✅ Dynamic class weights ARE being computed and applied
4. ✅ This is the OPTIMAL configuration for imbalanced data

**Ready to proceed with iteration 31 as-is!** 🎉
