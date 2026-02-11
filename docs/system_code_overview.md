# System Code Overview: Automated Chest X-Ray Classification Pipeline

**MSc Thesis: "Chest X-ray Disease Identification using Deep Learning"**

This document provides a comprehensive technical description of the automated machine learning system developed for multi-label chest X-ray classification. The system implements an AI-guided iterative improvement loop that autonomously optimizes model performance across 14 disease classes.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Entry Point: auto_improvement_loop.py](#2-entry-point-auto_improvement_looppy)
3. [Execution Flow](#3-execution-flow)
4. [Core Components](#4-core-components)
   - 4.1 [Dataset Handling](#41-dataset-handling)
   - 4.2 [Training Pipeline](#42-training-pipeline)
   - 4.3 [Evaluation](#43-evaluation)
   - 4.4 [Threshold Optimization](#44-threshold-optimization)
   - 4.5 [Configuration Management](#45-configuration-management)
   - 4.6 [Iteration Control](#46-iteration-control)
5. [Configuration-Driven Design](#5-configuration-driven-design)
6. [Iterative Experimentation Logic](#6-iterative-experimentation-logic)
7. [External Libraries and Dependencies](#7-external-libraries-and-dependencies)
8. [Summary](#8-summary)

---

## 1. System Overview

The system is an automated machine learning pipeline designed for multi-label chest X-ray disease classification using the ChestX-ray14 dataset. It combines:

- **Deep Learning**: DenseNet121 backbone with configurable classification heads
- **Automated Experimentation**: AI-guided hyperparameter optimization
- **Phased Protocol**: Structured optimization phases with hard-coded constraints
- **Dual-Lineage Strategy**: Two anchor iterations (AUC-focused and F1-focused) for warm starts
- **Role-Based Iteration**: Each iteration has a specific optimization objective

### Key Features

| Feature | Description |
|---------|-------------|
| **14 Disease Classes** | Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis, Pneumothorax, Pleural_Thickening, Pneumonia, Fibrosis, Edema, Consolidation |
| **AI Advisor** | GPT-5.2 (primary) with Claude fallback for configuration suggestions |
| **Warm Start** | Resume training from checkpoint with intelligent parent selection |
| **Per-Class Thresholds** | Optimized decision thresholds for each disease class |
| **Class Imbalance Handling** | Focal Loss, class weights, and Hernia oversampling |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO IMPROVEMENT LOOP                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐│
│  │   Config    │──▶│   Train     │──▶│    Test & Evaluate      ││
│  │   Manager   │   │   Pipeline  │   │                         ││
│  └─────────────┘   └─────────────┘   └──────────┬──────────────┘│
│         ▲                                        │               │
│         │          ┌─────────────┐               ▼               │
│         └──────────│ AI Advisor  │◀───── Compare Baseline        │
│                    └─────────────┘               │               │
│                           │                      ▼               │
│                           └───────────── Generate New Config     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Entry Point: auto_improvement_loop.py

The main orchestration layer is implemented in `auto_improvement_loop.py` (~1965 lines). This file contains the `AutoImprovementLoop` class which coordinates all components.

### Class: AutoImprovementLoop

```python
class AutoImprovementLoop:
    """Automated training loop with AI-guided improvements"""

    def __init__(
        self,
        base_config_path: str = "config_baseline.yaml",
        max_iterations: int = 10,
        openai_api_key: str = None,
        output_dir: str = "auto_improvement_runs",
        resume: bool = False
    )
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ai_advisor` | `AIAdvisor` | Communicates with LLM for suggestions |
| `best_tracker` | `BestModelTracker` | Tracks best model for rollback |
| `best_iteration_tracker` | `BestIterationTracker` | Multi-metric best iteration tracking |
| `task_manager` | `TaskManager` | Manages optimization tasks |
| `iteration_history` | `List[Dict]` | Full history of all iterations |
| `telegram` | `TelegramNotifier` | Optional notification system |

### Data Class: IterationLog

Structured logging for thesis documentation:

```python
@dataclass
class IterationLog:
    iteration: int
    role: str                                    # TRAIN_AUC, RECOVER_F1, etc.
    parent_iteration: Optional[int]
    timestamp: str
    macro_auc: float
    macro_f1: float
    auc_vs_iter12: float                         # Delta from AUC anchor
    f1_vs_iter58: float                          # Delta from F1 anchor
    per_group_metrics: Dict[str, Dict[str, float]]
    rare_class_details: Dict[str, Dict[str, Any]]
    distribution_diagnostics: Dict[str, Dict[str, float]]
    decision: Dict[str, Any]
    training_metadata: Dict[str, Any]
    ai_corrections: List[str]                    # Corrections applied to AI suggestions
```

---

## 3. Execution Flow

### Main Run Method

The `run()` method implements the main iteration loop:

```
FOR each iteration from start_iteration to max_iterations:
    1. Determine current phase (PHASE_1, PHASE_2, PHASE_5, etc.)
    2. Determine iteration role (TRAIN_AUC, RECOVER_F1, TARGETED_AUC)
    3. Select parent iteration for warm start
    4. Get enforced configuration for phase
    5. Run training via ConfigBasedTrainer
    6. Run evaluation via test_model_pipeline
    7. Compare against baseline (Wang et al.)
    8. Get AI analysis and suggestions
    9. Generate next iteration's configuration
    10. Save all artifacts and update trackers
```

### Single Iteration Flow

The `run_single_iteration()` method executes a 4-phase process:

```
Phase 1: TRAINING
├── Load configuration
├── Initialize trainer with warm start (if applicable)
├── Train model with early stopping
└── Save model checkpoint

Phase 2: TESTING
├── Load trained model
├── Run inference on test set
├── Apply per-class thresholds
└── Calculate metrics

Phase 3: BASELINE COMPARISON
├── Compare against Wang et al. baseline
├── Calculate per-class deltas
└── Generate comparison report

Phase 4: AI ANALYSIS
├── Send results to AI advisor
├── Receive suggested changes
├── Validate and correct suggestions
└── Generate next configuration
```

### Resume Capability

The system supports resuming interrupted runs:

```python
def _detect_resume_point(self):
    """Detect the last completed iteration and resume from there"""
    # Find all iteration directories
    # Load previous iteration history
    # Determine starting configuration
```

---

## 4. Core Components

### 4.1 Dataset Handling

**File**: `dataset.py`

#### ChestXRayDataset Class

```python
class ChestXRayDataset(Dataset):
    """
    Chest X-Ray dataset with optional Hernia-specific augmentation.

    Label columns (14 diseases):
    ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
     'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
     'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    """
```

**Key Features**:
- Handles both training and validation splits
- Supports Hernia-specific augmentation
- Optional patient metadata features (age, gender, view position)
- Class-specific rare augmentation

#### Hernia Oversampling

Due to extreme class imbalance (Hernia: 0.17%), the system implements weighted random sampling:

```python
def create_hernia_oversampler(dataset, oversample_factor=10, hernia_idx=3, seed=42):
    """
    Create a WeightedRandomSampler that oversamples Hernia-positive samples.

    With 10x oversampling:
    - Original Hernia prevalence: 0.17%
    - Effective Hernia prevalence: ~1.7%
    """
```

#### HerniaAugmentTransform

Safe augmentations for chest X-ray images:

```python
class HerniaAugmentTransform:
    """
    Safe augmentations:
    - RandomResizedCrop (scale 0.9-1.0) - mild zoom
    - Small rotation (±5 degrees)
    - Mild brightness/contrast jitter
    - Mild Gaussian noise
    - Small random erasing (optional)
    """
```

### 4.2 Training Pipeline

**File**: `config_based_pipeline.py`

#### ConfigBasedTrainer Class

The main training orchestrator:

```python
class ConfigBasedTrainer:
    """
    Configuration-driven training pipeline.

    Supports:
    - Multiple loss functions (BCE, FocalLoss)
    - Warm start from checkpoints
    - Early stopping with configurable monitor
    - Learning rate scheduling
    - Class weights for imbalanced data
    """
```

#### Loss Functions

**FocalLoss** - Primary loss for handling class imbalance:

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters:
    - gamma: Focusing parameter (typically 2.0-3.0)
    - alpha: Class balance weight (typically 0.7-0.75)
    - use_class_weights: Optional per-class weighting
    """
```

**WeightedBCELoss** - Alternative with class weighting:

```python
class WeightedBCELoss(nn.Module):
    """Binary Cross-Entropy with per-class weights."""
```

#### EarlyStopping

```python
class EarlyStopping:
    """
    Early stopping with configurable monitoring.

    Parameters:
    - patience: Epochs to wait before stopping
    - min_delta: Minimum improvement threshold
    - monitor: Metric to monitor (val_loss, val_macro_auc, val_f1)
    - mode: 'min' or 'max' depending on metric
    - warmup_epochs: Epochs before early stopping activates
    """
```

### 4.3 Evaluation

**File**: `chest_xray_test_pipeline.py`

```python
def test_model_pipeline(model_file, results_file, logger=None, head_config=None):
    """
    Evaluation pipeline for trained models.

    Outputs per-class metrics:
    - AUC (Area Under ROC Curve)
    - Accuracy, Precision, Recall, Sensitivity, Specificity
    - F1 Score
    - Confusion matrix (TP, TN, FP, FN)
    """
```

### 4.4 Threshold Optimization

**File**: `threshold_optimizer.py`

Per-class threshold optimization is critical for handling class imbalance:

```python
def optimize_thresholds(y_true, y_prob, class_names, method='per_class_f1_score'):
    """
    Optimize decision thresholds per class.

    Methods:
    - per_class_f1_score: Maximize F1 for each class independently
    - per_class_fbeta_score: Weighted F1 (beta controls recall/precision trade-off)
    - global: Single threshold for all classes (not recommended)
    """
```

### 4.5 Configuration Management

**File**: `config_manager.py`

```python
class ConfigManager:
    """
    Manages configuration files for the training pipeline.

    Methods:
    - load_config(): Load YAML configuration
    - save_config(): Save configuration to file
    - create_new_config(): Create modified config based on suggestions
    - save_iteration_config(): Save config with iteration number
    - export_for_llm(): Export config for AI analysis
    """
```

#### Configuration Structure

```yaml
metadata:
  config_version: "1.0"
  iteration: 84
  description: "Phase 5: Class-specific AUC improvement"
  parent_iteration: 84
  phase: "PHASE_5_AUC_IMPROVEMENT"

model:
  architecture: "ModifiedDenseNetWithDropOut"
  backbone: "densenet121"
  dropout_rate: 0.3
  head:
    type: "mlp"
    hidden_features: 1024
    dropout: 0.2

training:
  batch_size: 64
  learning_rate: 0.0005
  num_epochs: 50
  freeze_backbone: false
  optimizer:
    type: "Adam"
    weight_decay: 0.0
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 5
    factor: 0.5
    mode: "min"
  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_macro_auc"
    min_delta: 0.01

loss:
  type: "FocalLoss"
  gamma: 3.0
  alpha: 0.75
  use_class_weights: true

augmentation:
  rare_class:
    enabled: true
    rotation_degrees: 10
    horizontal_flip_prob: 0.7

evaluation:
  threshold_optimization: "per_class_f1_score"

sampling:
  hernia_oversample:
    enabled: true
    factor: 10

auc_improvement:
  enabled: true
  strategy: "hard_negative_emphasis"
  target_classes: ["Pneumonia", "Fibrosis", "Edema"]
```

### 4.6 Iteration Control

**File**: `iteration_baselines.py`

This module defines the phased protocol and role-based iteration control.

#### Optimization Phases

```python
class OptimizationPhase(Enum):
    PHASE_5_AUC_IMPROVEMENT = "PHASE_5_AUC_IMPROVEMENT"  # Current: Class-specific AUC
    PHASE_1_REPRODUCE = "PHASE_1_REPRODUCE"              # Archived: Reproduce iter 12
    PHASE_2_CALIBRATE = "PHASE_2_CALIBRATE"              # Archived: Threshold only
    STOP_AND_DEBUG = "STOP_AND_DEBUG"                    # Stop for issues
    SUCCESS = "SUCCESS"                                   # Final criteria met
```

#### Iteration Roles

```python
class IterationRole(Enum):
    TARGETED_AUC = "TARGETED_AUC"          # Phase 5: Class-specific improvement
    TRAIN_AUC = "TRAIN_AUC"                # Focus on AUC (parent=iter12)
    RECOVER_F1 = "RECOVER_F1"              # Focus on F1 (parent=iter58)
    ADJUST_THRESHOLDS = "ADJUST_THRESHOLDS" # No training, threshold only
    ABORT = "ABORT"                         # Stop the loop
```

#### Reference Iterations (Anchors)

```python
# AUC Anchor - Best ranking quality
ITERATION_12_AUC_ANCHOR = ReferenceIteration(
    iteration=12,
    macro_auc=0.8009,
    macro_f1=0.0029,
    model_path="auto_improvement_runs/iteration_012/pipeline_model_*.pth",
    role="AUC_ANCHOR"
)

# F1 Anchor - Best decision quality
ITERATION_58_F1_ANCHOR = ReferenceIteration(
    iteration=58,
    macro_auc=0.7904,
    macro_f1=0.2783,
    model_path="auto_improvement_runs/iteration_058/pipeline_model_*.pth",
    role="F1_ANCHOR"
)
```

#### Phase 5 Configuration

```python
PHASE_5_BASELINE_ITERATION = 84
PHASE_5_BASELINE_MACRO_AUC = 0.7659

# Target diseases (lowest AUC, highest improvement potential)
PHASE_5_TARGET_DISEASES = ["Pneumonia", "Fibrosis", "Edema"]

# Stable diseases (should not regress)
PHASE_5_STABLE_DISEASES = ["Effusion", "Emphysema", "Cardiomegaly", "Hernia", "Pneumothorax"]
```

---

## 5. Configuration-Driven Design

The system is entirely configuration-driven, allowing experiments to be reproduced and modified without code changes.

### Configuration Inheritance

```
config_baseline.yaml (Base configuration)
       │
       ▼
config_iteration_001.yaml (First iteration)
       │
       ▼
config_iteration_002.yaml (Modified by AI)
       │
       ... (Chain of iterations)
       │
       ▼
config_iteration_139.yaml (Latest)
```

### Parameter Categories

Based on analysis of 139 iterations:

**Constant Parameters** (10 parameters):
| Parameter | Value |
|-----------|-------|
| architecture | ModifiedDenseNetWithDropOut |
| backbone | densenet121 |
| head_type | mlp |
| head_dropout | 0.2 |
| batch_size | 64 |
| scheduler_type | ReduceLROnPlateau |
| scheduler_factor | 0.5 |
| hernia_oversample_enabled | Yes |
| hernia_oversample_factor | 10 |

**Varying Parameters** (20 parameters explored):
- loss_type: FocalLoss (85), BCE (47), focal (7)
- loss_gamma: 2.0-5.0
- loss_alpha: 0.25-0.80
- learning_rate: 1e-5 to 1e-3
- num_epochs: 5-200
- dropout_rate: 0.1-0.5
- weight_decay: 0.0-0.01
- early_stopping_patience: 5-15
- scheduler_patience: 3-7
- And more...

---

## 6. Iterative Experimentation Logic

### AI Advisor Integration

**File**: `ai_advisor.py`

```python
class AIAdvisor:
    """
    AI-powered configuration advisor using LLM.

    Primary: GPT-5.2 via OpenAI API
    Fallback: Claude (if OpenAI unavailable)

    Methods:
    - analyze_results(): Analyze iteration results and suggest improvements
    - validate_suggestion(): Validate AI suggestions against hard rules
    """
```

### Suggestion Validation

AI suggestions are validated against hard-coded rules:

```python
def validate_ai_suggestion(suggestion: Dict, role: IterationRole) -> Tuple[Dict, List[str]]:
    """
    Validate and correct AI suggestions against hard-coded rules.

    Rules enforced:
    1. gamma cannot exceed gamma_max (2.5)
    2. use_class_weights prohibited for TRAIN_AUC role
    3. parent_iteration enforced by role (cannot be overridden)
    4. Only ONE major change per iteration
    """
```

### Best Iteration Tracking

**File**: `best_iteration_tracker.py`

```python
class BestIterationTracker:
    """
    Multi-metric best iteration tracking.

    Tracked metrics:
    - best_f1: Best F1-Score iteration
    - best_auc: Best AUC iteration
    - best_recall: Best Recall iteration
    - best_precision: Best Precision iteration
    - most_balanced: Lowest metric variance

    Also maintains:
    - Pareto frontier: Non-dominated iterations
    - Full history: All iterations for analysis
    """
```

### Phase Determination Logic

```python
def determine_current_phase(iteration_history: List[Dict]) -> Tuple[OptimizationPhase, Dict]:
    """
    Determine optimization phase based on history.

    Phase 5 Logic:
    1. Default to PHASE_5_AUC_IMPROVEMENT
    2. Check for stop conditions:
       - SUCCESS: 2+ target diseases improved by >= +0.01 AUC
       - STOP: Stable disease regressed by > 0.01 AUC
       - STAGNATION: 3 iterations without improvement
    """
```

### Warm Start Strategy

```python
def get_parent_for_role(role: IterationRole) -> Optional[int]:
    """
    Get parent iteration for warm start.

    TRAIN_AUC → Iteration 12 (AUC anchor)
    RECOVER_F1 → Iteration 58 (F1 anchor)
    TARGETED_AUC → Phase 5 baseline (Iteration 84)
    ADJUST_THRESHOLDS → Current iteration (no parent)
    """
```

---

## 7. External Libraries and Dependencies

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.x | Deep learning framework |
| torchvision | 0.x | Pre-trained models, transforms |
| pandas | 2.x | Data manipulation |
| numpy | 1.x | Numerical operations |
| scikit-learn | 1.x | Metrics, evaluation |
| PIL/Pillow | 10.x | Image loading |
| PyYAML | 6.x | Configuration files |
| openai | 1.x | GPT API access |

### Optional Dependencies

| Library | Purpose |
|---------|---------|
| anthropic | Claude API fallback |
| python-telegram-bot | Notification system |
| matplotlib | Visualization (thesis figures) |

### Hardware Requirements

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

Supports:
- NVIDIA CUDA GPUs (recommended)
- Apple Silicon (MPS)
- CPU fallback

---

## 8. Summary

### System Characteristics

| Aspect | Description |
|--------|-------------|
| **Scale** | 139 iterations, 14 disease classes, ~110,000 images |
| **Automation** | Fully automated with AI guidance |
| **Reproducibility** | Configuration-driven, version-controlled |
| **Monitoring** | Comprehensive logging, Telegram notifications |
| **Fault Tolerance** | Resume capability, checkpoint saving |

### Key Design Decisions

1. **Phased Protocol**: Structured optimization prevents conflicting objectives
2. **Dual-Lineage Strategy**: Two anchor points (AUC/F1) for warm starts
3. **Per-Class Thresholds**: Critical for imbalanced multi-label classification
4. **Hard Rules**: AI suggestions validated against immutable constraints
5. **Hernia Oversampling**: Addresses extreme class imbalance (0.17%)

### File Organization

```
ArikFinalProject/
├── auto_improvement_loop.py     # Main orchestration
├── config_based_pipeline.py     # Training pipeline
├── config_manager.py            # Configuration handling
├── dataset.py                   # Data loading, augmentation
├── iteration_baselines.py       # Phased protocol, roles
├── best_iteration_tracker.py    # Multi-metric tracking
├── ai_advisor.py                # LLM integration
├── chest_xray_test_pipeline.py  # Evaluation
├── threshold_optimizer.py       # Per-class thresholds
├── telegram_notifier.py         # Notifications
├── task_manager.py              # Task management
├── best_model_tracker.py        # Model checkpointing
├── config_baseline.yaml         # Base configuration
└── auto_improvement_runs/       # Iteration outputs
    ├── iteration_001/
    ├── iteration_002/
    ├── ...
    ├── iteration_139/
    ├── best_iterations_registry.json
    └── FINAL_REPORT.md
```

### Performance Achieved

| Metric | Best Value | Iteration |
|--------|------------|-----------|
| Macro AUC | 0.8009 | 12 (AUC Anchor) |
| Macro F1 | 0.2783 | 58 (F1 Anchor) |
| Phase 5 Baseline AUC | 0.7659 | 84 |

### Disease Group Performance

| Group | Diseases | Typical AUC Range |
|-------|----------|-------------------|
| Rare (<1%) | Hernia, Pneumonia, Fibrosis, Edema | 0.64-0.86 |
| Moderate (1-5%) | Emphysema, Cardiomegaly, Pleural_Thickening | 0.72-0.85 |
| Common (>5%) | Effusion, Infiltration, Mass, Nodule, Atelectasis, Pneumothorax, Consolidation | 0.65-0.85 |

---

*Document generated: 2026-02-08*
*System version: Phase 5 (Post-Iteration 139)*
