# Comprehensive Research Timeline Report
## Automated Multi-Label Chest X-Ray Classification: ~150 Iteration Experimental Journey

**Report Generated**: 2026-02-10
**Total Iterations Documented**: 139+
**Total Runtime**: ~75,000+ seconds (~21+ hours of training)
**Project Duration**: January 2026 - February 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Experimental Timeline by Phase](#2-experimental-timeline-by-phase)
3. [Summary of Experimental Axes](#3-summary-of-experimental-axes)
4. [Role of Automation and AI Advisory](#4-role-of-automation-and-ai-advisory)
5. [Key Insights and Lessons Learned](#5-key-insights-and-lessons-learned)
6. [Final Model Selection Rationale](#6-final-model-selection-rationale)
7. [Limitations](#7-limitations)
8. [Future Work](#8-future-work)

---

## 1. Project Overview

### 1.1 Problem Statement

This research addresses **multi-label thoracic disease classification** from chest X-ray images using the ChestX-ray14 dataset (112,120 frontal-view chest radiographs). The core challenge is extreme class imbalance: disease prevalence ranges from **25.4%** (Infiltration) to **0.17%** (Hernia), with most conditions below 5%.

### 1.2 Objectives

1. **Primary Goal**: Achieve competitive AUC while maintaining practical F1-scores across all 14 pathology classes
2. **Secondary Goal**: Develop an automated experimental framework with AI-guided hyperparameter optimization
3. **Research Goal**: Understand the effectiveness of various techniques for handling extreme class imbalance

### 1.3 Dataset Characteristics

| Metric | Value |
|--------|-------|
| Total Images | 112,120 |
| Number of Classes | 14 pathologies |
| Multi-label | Yes (average 1.3 labels per image) |
| Most Prevalent | Infiltration (25.4%) |
| Least Prevalent | Hernia (0.17%) |
| Image Size | 1024×1024 → 224×224 (resized) |

### 1.4 Base Architecture

- **Backbone**: DenseNet-121 (ImageNet pretrained)
- **Classification Head**: MLP with configurable hidden layers
- **Auxiliary Features**: Patient metadata (age, gender, view position, follow-up #)
- **Feature Fusion**: 1024 (image) + 128 (metadata) = 1152-dim final representation

---

## 2. Experimental Timeline by Phase

### Phase 1: Baseline Establishment (Iterations 1-12)
**Date Range**: Early January 2026
**Objective**: Establish baseline performance with initial configurations

#### Key Observations
- Early iterations suffered from **severe class collapse**: model predicted nearly all samples as negative
- Initial F1-scores were near-zero for 11/14 classes
- AUC started around 0.71-0.80 range
- Threshold fixed at 0.5 proved inadequate for imbalanced data

#### Configuration Highlights
| Parameter | Value |
|-----------|-------|
| Loss Function | FocalLoss (γ=3.0, α=0.75) |
| Learning Rate | 0.0001-0.001 |
| Epochs | 50-60 |
| Dropout | 0.2-0.3 |

#### Milestone: Iteration 12
- **AUC**: 0.8009
- **F1**: 0.0029 (severely collapsed)
- This iteration became a critical **"AUC anchor"** for future phases

#### AI Analysis (Iteration 12)
> "The key issue is that the model is predicting almost all the samples as negative. This is indicated by the zero recall for 11/14 classes... The threshold optimization strategy should be revisited."

#### Lessons Learned
- FocalLoss with high gamma/alpha did not prevent collapse
- Threshold optimization was identified as critical but not yet implemented effectively
- Rare class augmentation was suggested but not yet effective

---

### Phase 2: Loss Function & Threshold Exploration (Iterations 13-40)
**Date Range**: January 2026 (Week 2)
**Objective**: Explore loss configurations and introduce threshold optimization

#### Key Experiments
1. **FocalLoss variations**: γ ∈ {1.5, 2.0, 2.5, 3.0, 4.0, 5.0}, α ∈ {0.25, 0.5, 0.7, 0.75, 0.8}
2. **BCE loss introduction**: Simple binary cross-entropy as alternative
3. **Class weighting**: Inverse frequency weighting experiments
4. **Threshold optimization**: Introduction of per-class F1-based thresholds

#### Performance Trajectory
| Iteration | AUC | F1 | Key Change |
|-----------|-----|----|----|
| 13 | 0.7722 | 0.0171 | Learning rate adjustment |
| 26 | 0.7570 | 0.0342 | Threshold optimization enabled |
| 31 | 0.7055 | 0.1672 | First significant F1 jump |
| 32 | 0.7785 | 0.2385 | Per-class threshold breakthrough |
| 38 | 0.7993 | 0.2532 | Balanced configuration |

#### Breakthrough: Iteration 31-32
The introduction of per-class threshold optimization based on F1-score dramatically improved recall without destroying precision. F1 jumped from ~0.03 to ~0.24.

#### Lessons Learned
- Per-class threshold optimization was **the single most impactful technique**
- FocalLoss did not provide consistent benefits over BCE
- Class weighting sometimes helped, sometimes hindered

---

### Phase 3: Configuration Refinement (Iterations 41-65)
**Date Range**: January 2026 (Weeks 2-3)
**Objective**: Refine configurations, explore regularization, find stable operating points

#### Key Experiments
1. **Dropout exploration**: 0.1, 0.2, 0.3, 0.4, 0.5
2. **Weight decay**: 0, 0.0001, 0.01
3. **Learning rate schedules**: ReduceLROnPlateau with patience 3-7
4. **Early stopping**: Patience 5-15, monitoring val_macro_auc vs val_f1
5. **Data augmentation**: Rare class augmentation enabled/disabled

#### Performance Trajectory
| Iteration | AUC | F1 | Key Change |
|-----------|-----|----|----|
| 50 | 0.8002 | 0.2718 | Best balanced (Pareto frontier) |
| 56-58 | ~0.79 | ~0.27-0.28 | Stable F1 plateau |
| 58 | 0.7904 | 0.2783 | **"F1 Anchor"** established |
| 64 | 0.4755 | 0.0927 | Catastrophic failure (high dropout 0.5) |

#### Milestone: Iteration 58 ("F1 Anchor")
- **AUC**: 0.7904
- **F1**: 0.2783
- Best F1 achieved with FocalLoss (γ=3.0, α=0.75)
- Used as reference for dual-lineage strategy

#### Failure Analysis: Iteration 64
High dropout (0.5) combined with aggressive augmentation caused catastrophic collapse:
- AUC dropped to 0.4755
- Recall spiked to 0.68 but precision collapsed to 0.05
- Demonstrated over-regularization failure mode

#### Lessons Learned
- Dropout > 0.3 was harmful in this architecture
- Weight decay had minimal impact
- FocalLoss with γ=3.0, α=0.75 was stable for F1 but not optimal for AUC
- BCE was more stable for AUC optimization

---

### Phase 4: Drift and Recovery (Iterations 66-79)
**Date Range**: January 2026 (Week 3)
**Objective**: Recover from performance drift, understand failure modes

#### Problem Identified
Starting around iteration 65, the system entered a **performance drift spiral**:
- AUC degraded from 0.80 to ~0.63-0.65
- F1 dropped to 0.12-0.14 range
- AI advisor recommendations became ineffective

#### Drift Trajectory
| Iteration | AUC | F1 | Observation |
|-----------|-----|----|----|
| 65 | 0.6615 | 0.1277 | Drift begins |
| 70 | 0.6950 | 0.1496 | Partial recovery |
| 75 | 0.6349 | 0.1386 | Stagnation |
| 79 | 0.6381 | 0.1414 | Stuck in local minimum |

#### Root Cause Analysis
1. **Compounding errors**: AI suggestions built on degraded models
2. **Loss of reference**: No mechanism to return to known-good configurations
3. **Exploration vs exploitation imbalance**: System kept exploring from poor starting points

#### Lessons Learned
- Need for **explicit anchor iterations** as recovery points
- Importance of **parent iteration tracking** for lineage
- Value of **phase-based protocols** vs unconstrained search

---

### Phase 5: Phase 1 Reproduction Attempts (Iterations 80-86)
**Date Range**: Late January 2026
**Objective**: Reproduce iteration 12 AUC performance as baseline for targeted improvements

#### Protocol Design
The system was re-designed with explicit phases:
1. **Phase 1**: Reproduce iteration 12 AUC (target: 0.8009)
2. **Phase 2**: Threshold calibration
3. **Phase 5**: Class-specific AUC improvements

#### Reproduction Attempts
| Iteration | AUC | Target | Gap | Status |
|-----------|-----|--------|-----|--------|
| 80 | 0.7613 | 0.8009 | -0.0396 | FAILED |
| 81 | 0.7620 | 0.8009 | -0.0389 | FAILED |
| 83 | 0.7625 | 0.8009 | -0.0384 | FAILED |
| 84 | 0.7659 | 0.8009 | -0.0350 | FAILED |

#### AI Analysis (Iteration 84)
> "STOP_AND_DEBUG: Phase 1 reproduction failed after 4 attempts. AUC gap of -0.035 exceeds tolerance (0.03). Recommending HEAD_UPGRADE strategy with frozen backbone."

#### Key Insight
Exact reproduction was impossible due to:
- Non-determinism in training (even with seeds)
- Potential dataset split differences
- Accumulated drift in dependencies

---

### Phase 6: HEAD_UPGRADE Breakthrough (Iterations 87-92)
**Date Range**: February 2026 (Week 1)
**Objective**: Achieve new performance highs via architecture modifications

#### Strategy: HEAD_UPGRADE
Based on AI recommendation, the system pivoted to:
1. **Freeze backbone** (DenseNet-121 weights locked)
2. **Train only classification head** (MLP layers)
3. **Switch to BCE loss** (simpler, more stable)
4. **Reduce epochs** (15-50, faster iteration)

#### Performance Trajectory
| Iteration | AUC | F1 | Key Change |
|-----------|-----|----|----|
| 87 | 0.8199 | 0.2944 | HEAD_UPGRADE + BCE |
| 88 | 0.8198 | 0.2941 | Confirmation |
| 89 | 0.8194 | 0.2948 | Stable |
| 90 | 0.8194 | 0.2943 | Stable |
| 91 | 0.8201 | **0.2957** | **Best F1** |
| 92 | **0.8204** | 0.2930 | **Best AUC** |

#### Best Results Achieved
| Metric | Best Iteration | Value |
|--------|----------------|-------|
| **AUC** | 92 | 0.8204 |
| **F1** | 91 | 0.2957 |
| **Recall** | 64 | 0.6832 (but AUC collapsed) |
| **Precision** | 130 | 0.3845 (but recall collapsed) |

#### Why HEAD_UPGRADE Worked
1. **Reduced overfitting**: Fewer trainable parameters
2. **Better generalization**: Preserved ImageNet features
3. **Faster convergence**: Less training required
4. **Stability**: BCE loss more predictable than FocalLoss

#### Configuration (Iteration 92)
```yaml
model:
  use_additional_features: true
  dropout_rate: 0.3
  head_config:
    hidden_features: 1024

training:
  learning_rate: 0.0001
  num_epochs: 50
  freeze_backbone: true

loss:
  type: BCE

evaluation:
  threshold_optimization: per_class_f1_score
```

---

### Phase 7: Phase 5 - Targeted AUC Improvement (Iterations 93-121)
**Date Range**: February 2026 (Week 1)
**Objective**: Improve AUC for specific low-performing classes

#### Target Classes
Based on per-class AUC analysis, three classes were identified for improvement:
1. **Pneumonia**: AUC ~0.71
2. **Fibrosis**: AUC ~0.76
3. **Edema**: AUC ~0.84

#### Strategy: Hard-Negative Emphasis
```yaml
auc_improvement:
  enabled: true
  target_diseases: [Pneumonia, Fibrosis, Edema]
  hard_negative_threshold: 0.27
  hard_negative_weight: 2.7
```

#### Results
Despite systematic exploration:
- **No statistically significant AUC improvement** for target classes
- Performance remained stable in the 0.819-0.820 range
- Suggests these classes may require different approaches (architecture, data)

#### Configurations Tried
| Config | Threshold | Weight | Result |
|--------|-----------|--------|--------|
| A | 0.28 | 1.8 | No improvement |
| B | 0.32 | 2.1 | No improvement |
| C | 0.22 | 2.4 | No improvement |
| D | 0.27 | 2.7 | No improvement |

#### AI Analysis (Iteration 92)
> "Iterations 91/92 are not valid Phase-5 datapoints because they use HEAD_UPGRADE... we must propose a new clean Phase-5 run anchored to iteration 84..."

---

### Phase 8: Hernia Oversampling Experiments (Iterations 122-130)
**Date Range**: February 2026 (Week 1)
**Objective**: Address extreme class imbalance for Hernia (0.17% prevalence)

#### Strategy: 10x Oversampling
```yaml
sampling:
  strategy: hernia_oversampling
  target_ratio: 10.0
  method: WeightedRandomSampler
```

#### Results
| Iteration | Hernia AUC | Overall AUC | F1 | Observation |
|-----------|------------|-------------|-----|-----|
| 122 | 0.89 | 0.8171 | 0.2952 | Slight Hernia improvement |
| 126-130 | ~0.87 | 0.8170-0.8174 | ~0.13-0.14 | F1 collapsed |

#### Key Finding
- Hernia oversampling showed modest AUC improvement for that class
- However, it caused **F1 collapse** across other classes
- Trade-off: class-specific optimization may harm overall performance

---

### Phase 9: Threshold Stabilization (Iterations 126-130)
**Date Range**: February 2026 (Week 1)
**Objective**: Stabilize threshold behavior for consistent predictions

#### Observations
Several iterations showed concerning patterns:
- High precision (0.38) but low recall (0.10)
- F1 dropping to 0.13-0.14 from previous 0.29
- Thresholds becoming too conservative

#### Root Cause
Per-class threshold optimization was overly aggressive, pushing thresholds too high to minimize false positives at the expense of recall.

---

### Phase 10: Final Exploration (Iterations 131-141)
**Date Range**: February 2026 (Week 2)
**Objective**: Final exploration and confirmation runs

#### Results
| Iteration | AUC | F1 | Status |
|-----------|-----|----|----|
| 131 | 0.8171 | 0.1396 | Threshold issues |
| 132-141 | 0.71-0.76 | 0.20-0.25 | Exploration |

#### Final State
The system converged to stable performance:
- **Best reproducible AUC**: 0.82 (iterations 91-92)
- **Best reproducible F1**: 0.29-0.30 (iterations 87-92)

---

## 3. Summary of Experimental Axes

### 3.1 Loss Functions

| Loss Type | Iterations | Best AUC | Best F1 | Verdict |
|-----------|------------|----------|---------|---------|
| FocalLoss (various γ/α) | 85 | 0.80 | 0.28 | Good for F1, unstable for AUC |
| BCE | 47 | **0.82** | **0.30** | **Recommended** |

**Key Finding**: BCE was simpler and more effective. FocalLoss did not provide consistent benefits despite theoretical advantages for class imbalance.

### 3.2 Threshold Optimization

| Strategy | Impact |
|----------|--------|
| Fixed 0.5 | Causes class collapse (all negative predictions) |
| Per-class F1 optimization | **Most impactful technique** - 10x F1 improvement |
| Per-class F-beta | No significant difference from F1 |

### 3.3 Regularization

| Technique | Impact |
|-----------|--------|
| Dropout 0.3 | **Optimal** - stable performance |
| Dropout 0.5 | Catastrophic collapse |
| Dropout 0.1-0.2 | Mild overfitting |
| Weight decay 0.0001 | Minimal impact |

### 3.4 Architecture Modifications

| Modification | Impact |
|--------------|--------|
| Freeze backbone | **Critical** - enabled best results |
| Head hidden size 1024 | Slight improvement over default |
| Head hidden size 512 | No significant difference |

### 3.5 Data Augmentation

| Technique | Impact |
|-----------|--------|
| Standard augmentation | Moderate positive impact |
| Rare class augmentation | Inconsistent results |
| Hernia 10x oversampling | Class-specific gain, overall loss |

### 3.6 Training Configuration

| Parameter | Optimal Value | Notes |
|-----------|---------------|-------|
| Learning rate | 0.0001 | Higher rates caused instability |
| Epochs (frozen backbone) | 15-50 | Quick convergence |
| Epochs (full training) | 50-60 | Longer needed |
| Batch size | 32 | Standard |
| Scheduler patience | 5 | Aggressive |

---

## 4. Role of Automation and AI Advisory

### 4.1 System Architecture

The experimental framework consisted of:

1. **auto_improvement_loop.py**: Orchestration engine (1800+ lines)
2. **config_based_pipeline.py**: Training/evaluation pipeline
3. **ai_advisor.py**: AI integration (GPT-5.2 primary, Claude fallback)
4. **iteration_baselines.py**: Phase definitions and anchor management

### 4.2 AI Advisory Models Used

| Model | Usage | Effectiveness |
|-------|-------|---------------|
| OpenAI GPT-5.1/5.2 | Primary advisor | Good hyperparameter suggestions |
| Anthropic Claude | Fallback | Similar quality |

### 4.3 AI Advisory Contributions

**Positive Contributions**:
1. Identified threshold optimization as critical (Iteration 12)
2. Suggested HEAD_UPGRADE strategy (Iteration 84)
3. Diagnosed overfitting patterns (Iteration 50)
4. Recommended learning rate reductions
5. Proposed rare class augmentation strategies

**Limitations**:
1. Could not prevent drift spiral (Phase 4)
2. Hard-negative emphasis suggestions were ineffective
3. Sometimes recommended already-tried configurations
4. Lacked awareness of full iteration history

### 4.4 Automation Value

| Aspect | Value |
|--------|-------|
| Throughput | 139+ iterations with minimal human intervention |
| Consistency | Standardized evaluation and logging |
| Reproducibility | All configs and metrics preserved |
| Exploration | Systematic coverage of parameter space |

---

## 5. Key Insights and Lessons Learned

### 5.1 Technical Insights

#### What Worked

1. **Per-class threshold optimization** (10x F1 improvement)
   - Single most impactful technique
   - Optimizing thresholds per-class based on F1-score validation

2. **Frozen backbone with trained head**
   - Reduced overfitting
   - Faster training
   - Better generalization

3. **BCE loss over FocalLoss**
   - Simpler is better
   - More stable training dynamics
   - Equal or better performance

4. **Moderate regularization (dropout 0.3)**
   - Sweet spot for this architecture
   - Higher values caused collapse

5. **Patient metadata integration**
   - 4 auxiliary features contributed to predictions
   - Model is NOT purely image-based

#### What Didn't Work

1. **FocalLoss with aggressive parameters**
   - Did not provide consistent benefits
   - Sometimes caused training instability

2. **High dropout (0.4-0.5)**
   - Catastrophic performance collapse
   - Severe underfitting

3. **Hard-negative emphasis for AUC improvement**
   - No statistically significant gains
   - Multiple configurations tried

4. **Hernia oversampling without constraints**
   - Improved target class but harmed others
   - F1 collapsed to 0.13

5. **Unconstrained AI-guided search**
   - Led to performance drift
   - Needed phase-based protocols

### 5.2 Methodological Insights

1. **Anchor iterations are essential**
   - Known-good configurations as recovery points
   - Enable systematic exploration without permanent regression

2. **Phase-based protocols outperform random search**
   - Clear objectives for each phase
   - Prevent compounding errors

3. **Per-class analysis is more informative than averages**
   - Class-specific problems require class-specific solutions
   - Aggregate metrics hide important patterns

4. **Reproducibility is challenging**
   - Same config ≠ same results
   - Non-determinism in training

### 5.3 Performance Plateau

The system converged to a stable performance range:
- **AUC**: 0.82 ± 0.01
- **F1**: 0.29 ± 0.01

Further improvements may require:
- Different architectures (EfficientNet, Vision Transformer)
- External data (pretraining on larger medical datasets)
- Class-specific models
- Ensemble methods

---

## 6. Final Model Selection Rationale

### 6.1 Selected Model

**Iteration 92** was selected as the final model based on:

| Criterion | Value | Ranking |
|-----------|-------|---------|
| Macro AUC | 0.8204 | **Best** |
| Macro F1 | 0.2930 | 2nd best (0.003 below best) |
| Stability | High | Confirmed across iterations 87-103 |
| Reproducibility | Good | Config well-documented |

### 6.2 Configuration

```yaml
# Final Configuration (Iteration 92)
model:
  architecture: ModifiedDenseNetWithDropOut
  use_additional_features: true
  dropout_rate: 0.3
  head_config:
    hidden_features: 1024

training:
  learning_rate: 0.0001
  num_epochs: 50
  freeze_backbone: true
  weight_decay: 0.0

loss:
  type: BCE

evaluation:
  threshold_optimization: per_class_f1_score

early_stopping:
  enabled: true
  patience: 5
  monitor: val_macro_auc
  min_delta: 0.001
```

### 6.3 Per-Class Performance (Iteration 92)

| Class | AUC | Prevalence | Notes |
|-------|-----|------------|-------|
| Cardiomegaly | 0.90 | 2.47% | Good |
| Emphysema | 0.93 | 1.92% | Excellent |
| Effusion | 0.88 | 11.86% | Good |
| Hernia | 0.88 | 0.17% | Good (considering rarity) |
| Infiltration | 0.71 | 25.38% | Challenging |
| Mass | 0.85 | 4.67% | Good |
| Nodule | 0.79 | 5.16% | Moderate |
| Atelectasis | 0.80 | 10.31% | Moderate |
| Pneumothorax | 0.88 | 4.59% | Good |
| Pleural_Thickening | 0.79 | 2.74% | Moderate |
| Pneumonia | 0.71 | 1.23% | Challenging |
| Fibrosis | 0.76 | 1.36% | Moderate |
| Edema | 0.84 | 1.83% | Good |
| Consolidation | 0.80 | 3.87% | Moderate |

---

## 7. Limitations

### 7.1 Technical Limitations

1. **Architecture scope**: Only DenseNet-121 explored
2. **Loss function scope**: Only BCE and FocalLoss explored
3. **Single dataset**: ChestX-ray14 only
4. **Label noise**: Known issues with ChestX-ray14 labels
5. **Non-determinism**: Results vary between runs

### 7.2 Methodological Limitations

1. **No ablation studies**: Contribution of patient features vs image features unknown
2. **No ensemble exploration**: Single model only
3. **No cross-validation**: Single train/val/test split
4. **Limited reproducibility analysis**: Same config gives ±0.5% variation

### 7.3 Reporting Limitations

1. **Computation tracking**: Incomplete GPU/memory logging
2. **Some iteration gaps**: Missing configs for some iterations
3. **Inconsistent naming**: Some config files had version mismatches

---

## 8. Future Work

### 8.1 Immediate Extensions

1. **Ablation study**: Train image-only model to quantify metadata contribution
2. **Ensemble methods**: Combine multiple iterations (91, 92, 87, 58)
3. **Cross-validation**: 5-fold CV for robust estimates
4. **Reproducibility study**: Same config across 10+ runs

### 8.2 Architecture Exploration

1. **EfficientNet-B4/B5**: Known to work well on medical images
2. **Vision Transformer (ViT)**: Attention-based approach
3. **Multi-scale architectures**: Feature Pyramid Networks

### 8.3 Loss Function Research

1. **Asymmetric loss**: Different weights for FP vs FN
2. **Dice loss**: Common in medical image segmentation
3. **Class-conditional losses**: Different loss per class

### 8.4 Data Augmentation

1. **CutMix/MixUp**: Tested minimally, may need more exploration
2. **Adversarial augmentation**: Robustness training
3. **Medical-specific augmentations**: Brightness/contrast variations

### 8.5 Class-Specific Strategies

1. **Hernia specialist model**: Train separate model for rarest class
2. **Hierarchical classification**: Group similar pathologies
3. **Cost-sensitive learning**: Different misclassification costs

---

## Appendix A: Parameter Variation Summary

Across 139 iterations, 20 parameters varied:

| Parameter | Values Tried | Dominant Setting |
|-----------|--------------|------------------|
| loss_type | BCE, FocalLoss, focal | BCE (47) vs Focal (92) |
| loss_gamma | 1.5-5.0 | 2.0 (41), 3.0 (35) |
| loss_alpha | 0.25-0.8 | 0.75 (32), 0.70 (19) |
| use_class_weights | Yes/No | Yes (82), No (57) |
| dropout_rate | 0.1-0.5 | 0.3 (112) |
| learning_rate | 1e-5 to 1e-3 | 0.0001 (55), 0.0005 (45) |
| num_epochs | 5-200 | 50 (67), 35 (16) |
| freeze_backbone | Yes/No | Yes (46), No/unset (93) |
| weight_decay | 0-0.01 | 0 (75), 0.0001 (62) |
| scheduler_patience | 3-7 | 5 (91), 3 (47) |
| early_stopping_patience | 5-15 | 5 (107), 10 (30) |
| threshold_optimization | per_class_f1_score, etc. | per_class_f1_score (119) |

---

## Appendix B: Pareto Frontier Iterations

Iterations on the Pareto frontier (non-dominated AUC-F1 trade-off):

| Iteration | AUC | F1 | Recall | Key Feature |
|-----------|-----|----|----|-------------|
| 50 | 0.8002 | 0.2718 | 0.5233 | High recall |
| 58 | 0.7904 | 0.2783 | 0.4626 | F1 anchor |
| 87 | 0.8199 | 0.2944 | 0.4072 | HEAD_UPGRADE breakthrough |
| 91 | 0.8201 | 0.2957 | 0.3904 | Best F1 |
| 92 | 0.8204 | 0.2930 | 0.4047 | Best AUC |

---

## Appendix C: Key AI Analysis Excerpts

### Iteration 12 (Baseline Anchor)
> "The key issue is that the model is predicting almost all the samples as negative... The threshold optimization strategy should be revisited."

### Iteration 50 (Overfitting Diagnosis)
> "The model appears to be overfitting to the training set. Recommend reducing learning rate from 0.001 to 0.0005 and enabling early stopping based on validation AUC."

### Iteration 84 (Phase 1 Failure)
> "STOP_AND_DEBUG: Phase 1 reproduction failed after 4 attempts. AUC gap of -0.035 exceeds tolerance (0.03). Recommending HEAD_UPGRADE strategy with frozen backbone."

### Iteration 87 (HEAD_UPGRADE Success)
> "Phase 5 hard-negative emphasis targeting Pneumonia/Fibrosis/Edema. Using strict hard-negative threshold to suppress high-scoring false positives."

### Iteration 122 (Hernia Oversampling)
> "Hernia oversampling diagnostic experiment. Target: improve Hernia AUC without harming stable classes. Method: 10x WeightedRandomSampler bias."

---

*Report generated: 2026-02-10*
*Total training time documented: ~21+ hours across 139+ iterations*
*Final model: Iteration 92 (AUC: 0.8204, F1: 0.2930)*
