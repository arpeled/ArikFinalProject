"""
Iteration Baselines - Central Constants for Role-Based Auto-Improvement

PHASED PROTOCOL (POST-ITERATION 84):
================================================================================
PHASE 5: Class-Specific AUC Improvement (ACTIVE)
  - Goal: Improve AUC on target diseases (Pneumonia, Fibrosis, Edema)
  - Strategy: Hard-negative emphasis (noise suppression, not recall forcing)
  - Baseline: Iteration 84 (Macro AUC = 0.7659)
  - Success: ≥ +0.01 AUC improvement on 2+ target diseases
  - Forbidden: Loss changes, class re-weighting, data oversampling, threshold opt
  - Stop: Regression > 0.01 on any stable class

PREVIOUS PHASES (ARCHIVED):
  - PHASE 1: Exact reproduction of iteration 12 (CONCLUDED - not reproducible)
  - PHASE 2: Threshold calibration (SKIPPED)

KEY INSIGHT: AUC improvement is class-specific and achieved by noise suppression.
================================================================================

This module defines:
- Reference iterations (iteration 84 for Phase 5 baseline)
- Optimization phases (PHASE_5_AUC_IMPROVEMENT)
- Hard-coded rules that the AI advisor cannot override
- Disease groups and target classes
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class OptimizationPhase(Enum):
    """
    Optimization phases for the project.

    PHASE_5_AUC_IMPROVEMENT: Class-specific AUC improvement via hard-negative emphasis
    PHASE_1_REPRODUCE: (ARCHIVED) Exactly reproduce iteration 12's AUC
    PHASE_2_CALIBRATE: (ARCHIVED) Threshold calibration only
    STOP_AND_DEBUG: Stop and diagnose issues
    SUCCESS: Final criteria met
    """
    PHASE_5_AUC_IMPROVEMENT = "PHASE_5_AUC_IMPROVEMENT"
    PHASE_1_REPRODUCE = "PHASE_1_REPRODUCE"
    PHASE_2_CALIBRATE = "PHASE_2_CALIBRATE"
    STOP_AND_DEBUG = "STOP_AND_DEBUG"
    SUCCESS = "SUCCESS"


class IterationRole(Enum):
    """
    Roles that determine the behavior of each iteration.

    NOTE: These are subordinate to OptimizationPhase.
    The phase determines what role is allowed.

    TARGETED_AUC: Phase 5 - class-specific AUC improvement via hard-negative emphasis
    TRAIN_AUC: (ARCHIVED) Focus on AUC improvement, parent=iter12
    RECOVER_F1: (ARCHIVED) Focus on F1 recovery, parent=iter58
    ADJUST_THRESHOLDS: No training, only threshold optimization
    ABORT: Stop the loop (either due to failure or success)
    """
    TARGETED_AUC = "TARGETED_AUC"
    TRAIN_AUC = "TRAIN_AUC"
    RECOVER_F1 = "RECOVER_F1"
    ADJUST_THRESHOLDS = "ADJUST_THRESHOLDS"
    ABORT = "ABORT"


@dataclass
class ReferenceIteration:
    """Data class for reference iteration information"""
    iteration: int
    macro_auc: float
    macro_f1: float
    model_path: str
    config_path: str
    role: str  # "AUC_ANCHOR" or "F1_ANCHOR"
    description: str


# Reference Iterations (Verified from actual data)
ITERATION_12_AUC_ANCHOR = ReferenceIteration(
    iteration=12,
    macro_auc=0.8009,  # 0.8008795663804592
    macro_f1=0.0029,   # 0.0028660317373975287
    model_path="experiments/auto_improvement_runs/iteration_012/pipeline_model_20251229-101232.pth",
    config_path="experiments/auto_improvement_runs/iteration_012/config.yaml",
    role="AUC_ANCHOR",
    description="Best ranking quality (AUC-first anchor)"
)

ITERATION_58_F1_ANCHOR = ReferenceIteration(
    iteration=58,
    macro_auc=0.7904,  # 0.7903619466860595
    macro_f1=0.2783,   # 0.27831535758572
    model_path="experiments/auto_improvement_runs/iteration_058/pipeline_model_20260108-133418.pth",
    config_path="experiments/auto_improvement_runs/iteration_058/config.yaml",
    role="F1_ANCHOR",
    description="Best decision quality (F1-first anchor)"
)


# ================================================================================
# PHASE 5 BASELINE (ITERATION 84)
# ================================================================================
# This is the baseline for Phase 5 class-specific AUC improvement.
# Key insight: AUC improvement is class-specific, achieved by noise suppression.
# ================================================================================

PHASE_5_BASELINE_ITERATION = 84
PHASE_5_BASELINE_MACRO_AUC = 0.7659
PHASE_5_BASELINE_MODEL_PATH = "experiments/auto_improvement_runs/iteration_084/pipeline_model_20260124-003356.pth"
PHASE_5_BASELINE_CONFIG_PATH = "experiments/auto_improvement_runs/iteration_084/config.yaml"

# Per-class AUC baseline from iteration 84
PHASE_5_BASELINE_PER_CLASS_AUC = {
    "Cardiomegaly": 0.8230,
    "Emphysema": 0.8513,
    "Effusion": 0.8308,
    "Hernia": 0.8561,
    "Infiltration": 0.6503,
    "Mass": 0.7835,
    "Nodule": 0.6959,
    "Atelectasis": 0.7421,
    "Pneumothorax": 0.8469,
    "Pleural_Thickening": 0.7169,
    "Pneumonia": 0.6395,
    "Fibrosis": 0.7226,
    "Edema": 0.8367,
    "Consolidation": 0.7264,
}

# Target diseases for Phase 5 (lowest AUC, highest improvement potential)
PHASE_5_TARGET_DISEASES = ["Pneumonia", "Fibrosis", "Edema"]

# Stable diseases (should not regress)
PHASE_5_STABLE_DISEASES = ["Effusion", "Emphysema", "Cardiomegaly", "Hernia", "Pneumothorax"]


@dataclass
class Phase5Config:
    """Configuration for Phase 5 AUC improvement."""
    # Target diseases for improvement
    target_diseases: List[str] = None

    # Hard-negative emphasis settings
    strategy: str = "hard_negative_emphasis"
    hard_negative_threshold: float = 0.3
    hard_negative_weight: float = 1.5

    # Success/failure thresholds
    meaningful_improvement: float = 0.01  # +0.01 AUC is meaningful
    max_regression: float = 0.01  # -0.01 AUC triggers stop
    stagnation_iterations: int = 3  # Stop after 3 iterations without improvement

    # Stop condition: improvements needed
    required_improved_diseases: int = 2  # Need 2+ diseases with +0.01 AUC

    def __post_init__(self):
        if self.target_diseases is None:
            self.target_diseases = PHASE_5_TARGET_DISEASES.copy()


PHASE_5_CONFIG = Phase5Config()


# ================================================================================
# EXACT ITERATION 12 CONFIG (ARCHIVED - REPRODUCTION CONCLUDED)
# ================================================================================
# This config matches iteration 12's ACTUAL configuration exactly.
#
# ITERATION 12 RESULTS:
#   - Macro AUC: 0.8009 (best AUC achieved)
#   - Macro F1: 0.0029 (near-zero despite threshold_optimization in config)
#   - Macro Recall: 0.0728 (model was "afraid to predict positives")
#
# KEY INSIGHT: Iteration 12 had threshold_optimization in config but F1 was
# still near-zero. This suggests threshold optimization wasn't fully working
# at that time, OR the model's raw predictions were so conservative that
# even optimized thresholds couldn't help.
#
# Later iterations (80, 81) have working threshold optimization and achieve
# F1 ~0.26, which explains why we can't reproduce iteration 12's low F1.
# ================================================================================
ITERATION_12_EXACT_CONFIG = {
    "model": {
        "architecture": "ModifiedDenseNetWithDropOut",
        "backbone": "densenet121",
        "num_classes": 14,
        "use_additional_features": True,  # ACTUAL: Was enabled in iter12
        "dropout_rate": 0.3,
        "pretrained": True
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0005,
        "num_epochs": 50,
        "num_workers": 8,
        "use_mixed_precision": False,
        "optimizer": {
            "type": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.0
        },
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-05
        },
        "early_stopping": {
            "enabled": True,  # ACTUAL: Was enabled in iter12
            "patience": 5,
            "min_delta": 0.01,
            "warmup_epochs": 5
        }
    },
    "loss": {
        "type": "FocalLoss",
        "gamma": 3.0,
        "alpha": 0.75,
        "use_class_weights": True,
        "class_frequencies": [
            0.0342, 0.031, 0.1641, 0.0028, 0.2451, 0.0712, 0.078,
            0.1424, 0.0653, 0.0417, 0.0176, 0.0208, 0.0284, 0.0575
        ]
    },
    "augmentation": {
        "base": {
            "resize": [224, 224],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "rare_class": {
            "enabled": True,  # ACTUAL: Was enabled in iter12
            "rotation_degrees": 10,
            "horizontal_flip_prob": 0.7,
            "affine_translate": [0.05, 0.05],
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2
            },
            "thresholds": {
                "Hernia": 0.003,
                "Pneumonia": 0.01,
                "Fibrosis": 0.012,
                "Edema": 0.016,
                "Emphysema": 0.018,
                "Cardiomegaly": 0.02,
                "Pleural_Thickening": 0.025,
                "Consolidation": 0.035,
                "Pneumothorax": 0.045
            }
        }
    },
    "data_split": {
        "method": "GroupShuffleSplit",
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "random_state": 42
    },
    "data": {
        "train_csv": "./ChestX-ray14/train_data.csv",
        "test_csv": "./ChestX-ray14/test_data.csv",
        "images_dir": "./ChestX-ray14/images224"
    },
    "evaluation": {
        "threshold": "auto",  # ACTUAL: Was 'auto' in iter12
        "threshold_optimization": "per_class_f1_score",  # ACTUAL: Was enabled
        "threshold_optimization_metric": "f1_score",
        "metrics": ["AUC", "Accuracy", "Specificity", "Recall", "Precision", "Sensitivity", "F1_Score"]
    },
    "hardware": {
        "device": "auto",
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2
    }
}


@dataclass
class PhaseSuccessCriteria:
    """
    Success criteria for each optimization phase.

    PHASE 1 (EXACT REPRODUCTION):
    - AUC must match iteration 12 (>= 0.795)
    - F1 constraint REMOVED: iteration 12's low F1 was due to threshold
      optimization not working at that time, not a model characteristic.
      With working threshold optimization, F1 will be higher.

    PHASE 2 (THRESHOLD CALIBRATION):
    - AUC must be preserved
    - F1 should reach iteration 58 levels via threshold-only optimization
    """
    # PHASE 1: Exact reproduction success criteria (AUC-focused)
    phase1_auc_target: float = 0.8009  # Iteration 12 exact AUC
    phase1_auc_tolerance: float = 0.01  # Success if AUC >= 0.7909 (relaxed)
    phase1_f1_max: float = 1.0  # REMOVED: F1 constraint no longer applies
    phase1_max_attempts: int = 5  # Stop and debug after 5 failed attempts

    # PHASE 2: F1 injection success
    phase2_auc_tolerance: float = 0.01  # AUC must stay within 0.01 of Phase 1
    phase2_f1_target: float = 0.2783  # Iteration 58 F1


PHASE_CRITERIA = PhaseSuccessCriteria()


# Disease Groups by frequency
DISEASE_GROUPS = {
    'rare': ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema'],  # <1% positive samples
    'moderate': ['Emphysema', 'Cardiomegaly', 'Pleural_Thickening'],  # 1-5% positive samples
    'common': ['Effusion', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
               'Pneumothorax', 'Consolidation']  # >5% positive samples
}

# All disease labels in order
ALL_DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


@dataclass
class TrainAUCRules:
    """
    Hard-coded rules for TRAIN_AUC role.
    These rules CANNOT be overridden by AI suggestions.
    """
    # Loss configuration
    loss_type: str = "FocalLoss"
    gamma: float = 2.0
    gamma_max: float = 2.5  # AI cannot suggest gamma > 2.5
    alpha: float = 0.7
    use_class_weights: bool = False  # PROHIBITED for TRAIN_AUC

    # Early stopping
    early_stopping_monitor: str = "val_macro_auc"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_warmup_epochs: int = 15

    # Parent selection
    parent_iteration: int = 12  # ALWAYS iteration 12 for TRAIN_AUC


@dataclass
class RecoverF1Rules:
    """
    Hard-coded rules for RECOVER_F1 role.
    These rules CANNOT be overridden by AI suggestions.
    """
    # Threshold-focused, minimal training
    freeze_model: bool = False  # Can fine-tune with low LR
    max_learning_rate: float = 0.00001  # Very low LR to preserve AUC

    # Early stopping on F1
    early_stopping_monitor: str = "val_f1"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 5

    # Parent selection
    parent_iteration: int = 58  # ALWAYS iteration 58 for RECOVER_F1


@dataclass
class DecisionRules:
    """
    Rules for determining iteration role and abort conditions.
    """
    # AUC tolerance: how much AUC drop is acceptable
    auc_tolerance: float = 0.005  # 0.5% drop acceptable

    # F1 recovery threshold: target 95% of iter58's F1
    f1_recovery_threshold: float = 0.95
    f1_target: float = 0.2783 * 0.95  # ~0.264

    # Abort conditions
    max_consecutive_auc_drops: int = 3
    min_auc_threshold: float = 0.75  # Abort if AUC drops below this

    # All-negative prediction detection (F1 too low)
    min_f1_threshold: float = 0.01  # Abort if F1 drops below this


@dataclass
class FinalModelCriteria:
    """
    Stop conditions for thesis completion (success abort).
    """
    # Target metrics for thesis model
    target_auc: float = 0.80  # At least as good as iter12
    target_f1: float = 0.26   # At least 95% of iter58

    # Tolerance
    auc_tolerance: float = 0.005
    f1_tolerance: float = 0.01

    # Stability check
    required_stable_iterations: int = 2  # Need 2 consecutive good iterations


# Global instances
TRAIN_AUC_RULES = TrainAUCRules()
RECOVER_F1_RULES = RecoverF1Rules()
DECISION_RULES = DecisionRules()
FINAL_MODEL_CRITERIA = FinalModelCriteria()


def determine_current_phase(
    iteration_history: List[Dict[str, Any]]
) -> tuple:
    """
    Determine the current optimization phase.

    POST-ITERATION 84: We are now in PHASE 5 (class-specific AUC improvement).
    Phase 1 reproduction was concluded as not reproducible.

    PHASE 5 LOGIC:
    1. Default to PHASE_5_AUC_IMPROVEMENT
    2. Check for stop conditions (success, regression, stagnation)
    3. Return appropriate context for the iteration

    Args:
        iteration_history: Full history of iterations

    Returns:
        Tuple of (OptimizationPhase, Dict with phase context)
    """
    # Filter to Phase 5 iterations
    phase5_iterations = [
        h for h in iteration_history
        if h.get('phase') in ['PHASE_5_AUC_IMPROVEMENT',
                              OptimizationPhase.PHASE_5_AUC_IMPROVEMENT.value]
    ]

    # Calculate next iteration number
    all_iterations = [h.get('iteration', 0) for h in iteration_history if h.get('iteration')]
    next_iteration = max(all_iterations) + 1 if all_iterations else 85

    # If no Phase 5 iterations yet, start Phase 5
    if not phase5_iterations:
        return (
            OptimizationPhase.PHASE_5_AUC_IMPROVEMENT,
            {
                "reason": "Starting Phase 5: Class-specific AUC improvement",
                "baseline_iteration": PHASE_5_BASELINE_ITERATION,
                "baseline_macro_auc": PHASE_5_BASELINE_MACRO_AUC,
                "target_diseases": PHASE_5_TARGET_DISEASES,
                "stable_diseases": PHASE_5_STABLE_DISEASES,
                "iteration": next_iteration,
                "parent_iteration": PHASE_5_BASELINE_ITERATION,
                "attempts": 0,
                "note": "Phase 1 concluded - iteration 12 not reproducible. Moving to Phase 5."
            }
        )

    # We have Phase 5 iterations - analyze them for stop conditions
    latest_phase5 = phase5_iterations[-1]
    latest_iteration = latest_phase5.get('iteration', 0)

    # Get per-class AUC from latest iteration (if available)
    per_class_auc = latest_phase5.get('per_class_auc', {})

    # Count iterations since last improvement
    iterations_since_improvement = 0
    for h in reversed(phase5_iterations):
        h_per_class = h.get('per_class_auc', {})
        if h_per_class:
            deltas = compute_phase5_auc_deltas(h_per_class)
            improved = any(
                info['is_target'] and info['meaningful_improvement']
                for info in deltas.values()
            )
            if improved:
                break
            iterations_since_improvement += 1

    # Check stop conditions if we have per-class AUC data
    if per_class_auc:
        auc_deltas = compute_phase5_auc_deltas(per_class_auc)
        should_stop, stop_reason, decision = check_phase5_stop_conditions(
            auc_deltas, iterations_since_improvement
        )

        if should_stop:
            if "SUCCESS" in stop_reason:
                return (
                    OptimizationPhase.SUCCESS,
                    {
                        "reason": stop_reason,
                        "decision": decision,
                        "final_iteration": latest_iteration,
                        "auc_deltas": {d: info['delta'] for d, info in auc_deltas.items()}
                    }
                )
            else:
                return (
                    OptimizationPhase.STOP_AND_DEBUG,
                    {
                        "reason": stop_reason,
                        "decision": decision,
                        "iteration": latest_iteration,
                        "action": "Freeze model weights, review Phase 5 results"
                    }
                )

    # Get the last Phase 5 iteration's parent (for warm start)
    last_parent = latest_phase5.get('parent_iteration', PHASE_5_BASELINE_ITERATION)

    # Continue Phase 5
    return (
        OptimizationPhase.PHASE_5_AUC_IMPROVEMENT,
        {
            "reason": "Phase 5: Class-specific AUC improvement in progress",
            "baseline_iteration": PHASE_5_BASELINE_ITERATION,
            "baseline_macro_auc": PHASE_5_BASELINE_MACRO_AUC,
            "target_diseases": PHASE_5_TARGET_DISEASES,
            "stable_diseases": PHASE_5_STABLE_DISEASES,
            "iteration": next_iteration,
            "parent_iteration": last_parent,
            "attempts": len(phase5_iterations),
            "iterations_since_improvement": iterations_since_improvement
        }
    )


def determine_iteration_role(
    current_iteration: int,
    current_metrics: Dict[str, float],
    iteration_history: List[Dict[str, Any]],
    ai_suggestion: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Determine the role for the current iteration.

    POST-ITERATION 84: Default to PHASE 5 (class-specific AUC improvement).

    Args:
        current_iteration: Current iteration number
        current_metrics: Metrics from previous iteration (or None for first)
        iteration_history: Full history of iterations
        ai_suggestion: Optional AI suggestion (for logging only)

    Returns:
        Tuple of (IterationRole, parent_iteration, reasoning)
    """
    # First, determine the current phase
    phase, phase_context = determine_current_phase(iteration_history)

    # PHASE 5: Class-specific AUC improvement
    if phase == OptimizationPhase.PHASE_5_AUC_IMPROVEMENT:
        parent = phase_context.get('parent_iteration', PHASE_5_BASELINE_ITERATION)
        return (
            IterationRole.TARGETED_AUC,
            parent,
            f"PHASE 5: Class-specific AUC improvement - {phase_context.get('reason', '')}"
        )

    # PHASE 1: Reproduce iteration 12 (ARCHIVED)
    if phase == OptimizationPhase.PHASE_1_REPRODUCE:
        return (
            IterationRole.TRAIN_AUC,
            ITERATION_12_AUC_ANCHOR.iteration,
            f"PHASE 1: Reproducing iteration 12 - {phase_context.get('reason', '')}"
        )

    # PHASE 2: Threshold calibration only (ARCHIVED)
    if phase == OptimizationPhase.PHASE_2_CALIBRATE:
        return (
            IterationRole.ADJUST_THRESHOLDS,
            None,
            f"PHASE 2: Threshold calibration only - {phase_context.get('reason', '')}"
        )

    # STOP_AND_DEBUG
    if phase == OptimizationPhase.STOP_AND_DEBUG:
        return (
            IterationRole.ABORT,
            None,
            f"STOP_AND_DEBUG: {phase_context.get('reason', 'Issue detected')}"
        )

    # SUCCESS: Final criteria met
    if phase == OptimizationPhase.SUCCESS:
        return (
            IterationRole.ABORT,
            None,
            f"SUCCESS: {phase_context.get('reason', 'All criteria met')}"
        )

    # Fallback - default to Phase 5
    return (
        IterationRole.TARGETED_AUC,
        PHASE_5_BASELINE_ITERATION,
        f"Fallback: Defaulting to Phase 5 (unknown phase {phase})"
    )


def check_abort_conditions(iteration_history: List[Dict[str, Any]]) -> tuple:
    """
    Check if abort conditions are met.

    IMPORTANT: We do NOT abort just because AUC is low. Low AUC triggers
    TRAIN_AUC role to recover from iteration 12. Abort only happens when:
    1. We've had many consecutive TRAIN_AUC attempts from iter12 that all failed
    2. Or some catastrophic condition is detected

    Args:
        iteration_history: Full history of iterations

    Returns:
        Tuple of (should_abort: bool, reason: str)
    """
    if len(iteration_history) < 2:
        return (False, "")

    # Check for consecutive TRAIN_AUC attempts that all resulted in AUC < 0.75
    # This means we've tried to recover from iter12 multiple times and failed
    recent_roles = []
    for h in iteration_history[-5:]:  # Look at last 5 iterations
        role = h.get('role', None)
        auc = h.get('metrics', {}).get('avg_auc', 0.0)
        recent_roles.append({'role': role, 'auc': auc})

    # Only abort if we have 5+ consecutive TRAIN_AUC attempts all below threshold
    # This would indicate iter12 itself is problematic (unlikely)
    consecutive_failed_recoveries = 0
    for entry in reversed(recent_roles):
        if entry.get('role') == 'TRAIN_AUC' and entry.get('auc', 0) < DECISION_RULES.min_auc_threshold:
            consecutive_failed_recoveries += 1
        else:
            break

    if consecutive_failed_recoveries >= 5:
        return (True, f"ABORT: {consecutive_failed_recoveries} consecutive TRAIN_AUC recovery attempts failed")

    # Check for all-negative predictions (F1 too low) - only if AUC is also bad
    # This indicates a truly broken model
    latest_auc = iteration_history[-1].get('metrics', {}).get('avg_auc', 0.0)
    latest_f1 = iteration_history[-1].get('metrics', {}).get('avg_f1', 0.0)
    if latest_f1 < DECISION_RULES.min_f1_threshold and latest_auc < 0.5:
        return (True, f"ABORT: Catastrophic failure - AUC ({latest_auc:.4f}) and F1 ({latest_f1:.4f}) both critically low")

    return (False, "")


def check_final_model_criteria(iteration_history: List[Dict[str, Any]]) -> tuple:
    """
    Check if final model criteria are met (success condition).

    Args:
        iteration_history: Full history of iterations

    Returns:
        Tuple of (criteria_met: bool, reason: str)
    """
    if len(iteration_history) < FINAL_MODEL_CRITERIA.required_stable_iterations:
        return (False, "")

    # Check last N iterations for stability
    recent = iteration_history[-FINAL_MODEL_CRITERIA.required_stable_iterations:]

    all_good = True
    for hist in recent:
        metrics = hist.get('metrics', {})
        auc = metrics.get('avg_auc', 0.0)
        f1 = metrics.get('avg_f1', 0.0)

        auc_good = auc >= (FINAL_MODEL_CRITERIA.target_auc - FINAL_MODEL_CRITERIA.auc_tolerance)
        f1_good = f1 >= (FINAL_MODEL_CRITERIA.target_f1 - FINAL_MODEL_CRITERIA.f1_tolerance)

        if not (auc_good and f1_good):
            all_good = False
            break

    if all_good:
        latest = iteration_history[-1].get('metrics', {})
        return (
            True,
            f"Final model criteria met for {FINAL_MODEL_CRITERIA.required_stable_iterations} iterations "
            f"(AUC={latest.get('avg_auc', 0):.4f}, F1={latest.get('avg_f1', 0):.4f})"
        )

    return (False, "")


def get_phase1_exact_config(random_seed: Optional[int] = None, iteration: int = 0) -> Dict[str, Any]:
    """
    Get the EXACT iteration 12 configuration for Phase 1 reproduction.

    CRITICAL: This returns the frozen config. Only random_seed can be changed.
    All other parameters MUST match iteration 12 exactly.

    Args:
        random_seed: Optional new random seed (only allowed change)
        iteration: Current iteration number (for metadata)

    Returns:
        Exact iteration 12 configuration
    """
    import copy
    import datetime
    config = copy.deepcopy(ITERATION_12_EXACT_CONFIG)

    # Add metadata (required by config_based_pipeline)
    config['metadata'] = {
        'config_version': '1.0',
        'description': 'PHASE 1: Exact reproduction of iteration 12 (FROZEN CONFIG)',
        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'iteration': iteration,
        'parent_config': 'iteration_012/config.yaml',
        'phase': 'PHASE_1_REPRODUCE',
        'note': 'DO NOT MODIFY - This config must match iteration 12 exactly'
    }

    # Add logging config if missing
    if 'logging' not in config:
        config['logging'] = {
            'log_interval': 300,
            'save_model': True,
            'save_results': True
        }

    # Only random seed can be changed
    if random_seed is not None:
        config['data_split']['random_state'] = random_seed
        print(f"📌 Phase 1: Using seed {random_seed} (original was 42)")
    else:
        print(f"📌 Phase 1: Using original seed 42")

    return config


def get_phase5_config(
    iteration: int,
    parent_iteration: int = PHASE_5_BASELINE_ITERATION,
    target_diseases: Optional[List[str]] = None,
    hard_negative_threshold: float = 0.3,
    hard_negative_weight: float = 1.5
) -> Dict[str, Any]:
    """
    Get Phase 5 configuration for class-specific AUC improvement.

    Phase 5 uses the iteration 84 baseline config with added auc_improvement block.
    Only the auc_improvement settings can be modified between iterations.

    Args:
        iteration: Current iteration number
        parent_iteration: Parent iteration to warm-start from (default: 84)
        target_diseases: List of diseases to target (default: Pneumonia, Fibrosis, Edema)
        hard_negative_threshold: Threshold for hard negatives (default: 0.3)
        hard_negative_weight: Weight multiplier for hard negatives (default: 1.5)

    Returns:
        Phase 5 configuration
    """
    import copy
    import datetime

    # Start from iteration 12 config (which is what iteration 84 was based on)
    config = copy.deepcopy(ITERATION_12_EXACT_CONFIG)

    # Set target diseases
    if target_diseases is None:
        target_diseases = PHASE_5_TARGET_DISEASES.copy()

    # Add Phase 5 specific block - the ONLY thing that can change between iterations
    config['auc_improvement'] = {
        'enabled': True,
        'strategy': 'hard_negative_emphasis',
        'target_classes': target_diseases,
        'hard_negative_threshold': hard_negative_threshold,
        'hard_negative_weight': hard_negative_weight
    }

    # Add metadata
    config['metadata'] = {
        'config_version': '1.0',
        'description': f'PHASE 5: Class-specific AUC improvement (targets: {", ".join(target_diseases)})',
        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'iteration': iteration,
        'parent_iteration': parent_iteration,
        'parent_config': f'iteration_{parent_iteration:03d}/config.yaml',
        'phase': 'PHASE_5_AUC_IMPROVEMENT',
        'baseline_iteration': PHASE_5_BASELINE_ITERATION,
        'target_diseases': target_diseases
    }

    # Add logging config
    if 'logging' not in config:
        config['logging'] = {
            'log_interval': 300,
            'save_model': True,
            'save_results': True
        }

    print(f"📌 Phase 5: Targeting {', '.join(target_diseases)}")
    print(f"   Hard-negative threshold: {hard_negative_threshold}, weight: {hard_negative_weight}")

    return config


def compute_phase5_auc_deltas(
    current_per_class_auc: Dict[str, float],
    baseline_per_class_auc: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute AUC deltas for Phase 5 analysis.

    Args:
        current_per_class_auc: Current iteration's per-class AUC values
        baseline_per_class_auc: Baseline AUC values (default: Phase 5 baseline)

    Returns:
        Dict with per-class analysis including delta, status, and flags
    """
    if baseline_per_class_auc is None:
        baseline_per_class_auc = PHASE_5_BASELINE_PER_CLASS_AUC

    analysis = {}
    for disease, baseline_auc in baseline_per_class_auc.items():
        current_auc = current_per_class_auc.get(disease, 0.0)
        delta = current_auc - baseline_auc

        is_target = disease in PHASE_5_TARGET_DISEASES
        is_stable = disease in PHASE_5_STABLE_DISEASES

        # Determine status
        if delta >= PHASE_5_CONFIG.meaningful_improvement:
            status = "IMPROVED"
        elif delta <= -PHASE_5_CONFIG.max_regression:
            status = "REGRESSED"
        else:
            status = "STABLE"

        analysis[disease] = {
            'baseline_auc': baseline_auc,
            'current_auc': current_auc,
            'delta': delta,
            'status': status,
            'is_target': is_target,
            'is_stable': is_stable,
            'meaningful_improvement': delta >= PHASE_5_CONFIG.meaningful_improvement,
            'regression_warning': is_stable and delta <= -PHASE_5_CONFIG.max_regression
        }

    return analysis


def check_phase5_stop_conditions(
    auc_deltas: Dict[str, Dict[str, Any]],
    iterations_since_improvement: int = 0
) -> tuple:
    """
    Check if Phase 5 should stop based on the defined conditions.

    Stop conditions:
    1. Two or more target diseases show consistent AUC improvement >= +0.01
    2. Any stable disease regresses by more than -0.01 AUC
    3. Macro AUC stagnates for 3 consecutive iterations

    Args:
        auc_deltas: Output from compute_phase5_auc_deltas
        iterations_since_improvement: Count of iterations without meaningful improvement

    Returns:
        Tuple of (should_stop: bool, reason: str, decision: str)
    """
    # Count improved target diseases
    improved_targets = sum(
        1 for d, info in auc_deltas.items()
        if info['is_target'] and info['meaningful_improvement']
    )

    # Check for stable disease regression
    regressed_stable = [
        d for d, info in auc_deltas.items()
        if info['regression_warning']
    ]

    # Success condition: 2+ target diseases improved
    if improved_targets >= PHASE_5_CONFIG.required_improved_diseases:
        return (
            True,
            f"SUCCESS: {improved_targets} target diseases improved by >= +0.01 AUC",
            "STOP_PHASE_5_AND_FREEZE"
        )

    # Failure condition: stable disease regressed
    if regressed_stable:
        return (
            True,
            f"STOP: Stable disease(s) regressed: {', '.join(regressed_stable)}",
            "STOP_PHASE_5_AND_FREEZE"
        )

    # Stagnation condition
    if iterations_since_improvement >= PHASE_5_CONFIG.stagnation_iterations:
        return (
            True,
            f"STOP: No improvement for {iterations_since_improvement} consecutive iterations",
            "STOP_PHASE_5_AND_FREEZE"
        )

    # Continue
    return (False, "Continue Phase 5", "CONTINUE_TARGETED_AUC_IMPROVEMENT")


def get_enforced_config_for_phase(
    phase: OptimizationPhase,
    base_config: Dict[str, Any],
    phase_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enforce phase-specific configuration that cannot be overridden.

    PHASE 5: Uses baseline config with auc_improvement block (only auc_improvement can change)
    PHASE 1: (ARCHIVED) Returns EXACT iteration 12 config
    PHASE 2: (ARCHIVED) Returns base_config with training disabled

    Args:
        phase: The current optimization phase
        base_config: The base configuration
        phase_context: Context from phase determination

    Returns:
        Enforced configuration for the phase
    """
    import copy

    if phase == OptimizationPhase.PHASE_5_AUC_IMPROVEMENT:
        # PHASE 5: Class-specific AUC improvement
        # Only auc_improvement block can be modified
        iteration = phase_context.get('iteration', 85)
        target_diseases = phase_context.get('target_diseases', PHASE_5_TARGET_DISEASES)
        hard_negative_threshold = phase_context.get('hard_negative_threshold', 0.3)
        hard_negative_weight = phase_context.get('hard_negative_weight', 1.5)

        config = get_phase5_config(
            iteration=iteration,
            target_diseases=target_diseases,
            hard_negative_threshold=hard_negative_threshold,
            hard_negative_weight=hard_negative_weight
        )
        config['_phase'] = 'PHASE_5_AUC_IMPROVEMENT'
        config['_phase_note'] = 'Class-specific AUC improvement - ONLY auc_improvement block can change'
        config['_parent_iteration'] = phase_context.get('parent_iteration', PHASE_5_BASELINE_ITERATION)
        return config

    elif phase == OptimizationPhase.PHASE_1_REPRODUCE:
        # PHASE 1: Use EXACT iteration 12 config - ignore any suggestions
        config = get_phase1_exact_config()
        config['_phase'] = 'PHASE_1_REPRODUCE'
        config['_phase_note'] = 'EXACT reproduction - NO CONFIG CHANGES ALLOWED'
        return config

    elif phase == OptimizationPhase.PHASE_2_CALIBRATE:
        # PHASE 2: Disable training, only allow threshold optimization
        config = copy.deepcopy(base_config)
        config['_phase'] = 'PHASE_2_CALIBRATE'
        config['_phase_note'] = 'Threshold calibration ONLY - NO TRAINING'
        config['_skip_training'] = True  # Flag to skip training
        config['_load_model_from'] = phase_context.get('phase1_iteration', 12)
        return config

    else:
        # For STOP_AND_DEBUG or SUCCESS, return base config
        config = copy.deepcopy(base_config)
        config['_phase'] = phase.value if phase else 'UNKNOWN'
        return config


def get_enforced_config_for_role(role: IterationRole, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    LEGACY: Enforce role-specific configuration.

    NOTE: This function is deprecated. Use get_enforced_config_for_phase instead.
    Kept for backwards compatibility.

    Args:
        role: The iteration role
        base_config: The base configuration to modify

    Returns:
        Modified configuration with enforced rules
    """
    import copy
    config = copy.deepcopy(base_config)

    if role == IterationRole.TRAIN_AUC:
        # Enforce TRAIN_AUC rules
        if 'loss' not in config:
            config['loss'] = {}
        config['loss']['type'] = TRAIN_AUC_RULES.loss_type
        config['loss']['gamma'] = TRAIN_AUC_RULES.gamma
        config['loss']['use_class_weights'] = TRAIN_AUC_RULES.use_class_weights

        if 'training' not in config:
            config['training'] = {}
        if 'early_stopping' not in config['training']:
            config['training']['early_stopping'] = {}

        config['training']['early_stopping']['monitor'] = TRAIN_AUC_RULES.early_stopping_monitor
        config['training']['early_stopping']['mode'] = TRAIN_AUC_RULES.early_stopping_mode
        config['training']['early_stopping']['patience'] = TRAIN_AUC_RULES.early_stopping_patience
        config['training']['early_stopping']['min_delta'] = TRAIN_AUC_RULES.early_stopping_min_delta
        config['training']['early_stopping']['warmup_epochs'] = TRAIN_AUC_RULES.early_stopping_warmup_epochs

    elif role == IterationRole.RECOVER_F1:
        # Enforce RECOVER_F1 rules
        if 'training' not in config:
            config['training'] = {}

        # Cap learning rate
        current_lr = config['training'].get('learning_rate', 0.001)
        if current_lr > RECOVER_F1_RULES.max_learning_rate:
            config['training']['learning_rate'] = RECOVER_F1_RULES.max_learning_rate

        if 'early_stopping' not in config['training']:
            config['training']['early_stopping'] = {}

        config['training']['early_stopping']['monitor'] = RECOVER_F1_RULES.early_stopping_monitor
        config['training']['early_stopping']['mode'] = RECOVER_F1_RULES.early_stopping_mode
        config['training']['early_stopping']['patience'] = RECOVER_F1_RULES.early_stopping_patience

    return config


def validate_ai_suggestion(
    suggestion: Dict[str, Any],
    role: IterationRole
) -> tuple:
    """
    Validate and correct AI suggestions against hard-coded rules.

    Args:
        suggestion: AI-provided configuration suggestions
        role: The current iteration role

    Returns:
        Tuple of (corrected_suggestion, list_of_corrections)
    """
    import copy
    corrected = copy.deepcopy(suggestion)
    corrections = []

    # Rule 1: Check gamma limit
    if 'loss' in corrected and 'gamma' in corrected['loss']:
        gamma = corrected['loss']['gamma']
        if gamma > TRAIN_AUC_RULES.gamma_max:
            corrected['loss']['gamma'] = TRAIN_AUC_RULES.gamma
            corrections.append(f"gamma={gamma} > {TRAIN_AUC_RULES.gamma_max}, forced to {TRAIN_AUC_RULES.gamma}")

    # Rule 2: Check class weights for TRAIN_AUC
    if role == IterationRole.TRAIN_AUC:
        if 'loss' in corrected and corrected['loss'].get('use_class_weights', False):
            corrected['loss']['use_class_weights'] = False
            corrections.append("use_class_weights=true prohibited for TRAIN_AUC, forced to false")

    # Rule 3: Remove parent_iteration from suggestion (we enforce it)
    if 'parent_iteration' in corrected:
        del corrected['parent_iteration']
        corrections.append("parent_iteration removed (enforced by role)")

    # Rule 4: Limit to ONE major change
    major_changes = []
    if 'loss' in corrected:
        major_changes.append('loss')
    if 'training' in corrected and 'learning_rate' in corrected.get('training', {}):
        major_changes.append('learning_rate')
    if 'model' in corrected and 'dropout_rate' in corrected.get('model', {}):
        major_changes.append('dropout')

    if len(major_changes) > 1:
        # Keep only the first major change
        kept = major_changes[0]
        for change in major_changes[1:]:
            if change == 'learning_rate':
                del corrected['training']['learning_rate']
            elif change == 'dropout':
                del corrected['model']['dropout_rate']
            elif change == 'loss':
                del corrected['loss']
        corrections.append(f"Multiple major changes detected, keeping only: {kept}")

    return corrected, corrections


def get_parent_for_role(role: IterationRole) -> Optional[int]:
    """
    Get the correct parent iteration for a given role.

    This is the HARD-CODED parent selection that cannot be overridden.

    Args:
        role: The iteration role

    Returns:
        Parent iteration number, or None for ABORT
    """
    if role == IterationRole.TRAIN_AUC:
        return ITERATION_12_AUC_ANCHOR.iteration
    elif role == IterationRole.RECOVER_F1:
        return ITERATION_58_F1_ANCHOR.iteration
    elif role == IterationRole.ADJUST_THRESHOLDS:
        return None  # No parent, use current model
    else:
        return None  # ABORT


def compute_group_metrics(
    per_class_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate metrics for disease groups.

    Args:
        per_class_metrics: Dict mapping disease name to its metrics

    Returns:
        Dict mapping group name to aggregate metrics
    """
    import numpy as np

    group_metrics = {}

    for group_name, diseases in DISEASE_GROUPS.items():
        group_aucs = []
        group_f1s = []

        for disease in diseases:
            if disease in per_class_metrics:
                metrics = per_class_metrics[disease]
                if 'auc' in metrics:
                    group_aucs.append(metrics['auc'])
                if 'f1' in metrics:
                    group_f1s.append(metrics['f1'])

        group_metrics[group_name] = {
            'auc': float(np.mean(group_aucs)) if group_aucs else 0.0,
            'f1': float(np.mean(group_f1s)) if group_f1s else 0.0,
            'diseases': diseases
        }

    return group_metrics


def format_thesis_documentation_marker(
    iteration: int,
    role: IterationRole,
    parent_iteration: Optional[int],
    macro_auc: float,
    macro_f1: float,
    decision: str
) -> str:
    """
    Format a thesis documentation marker for logging.

    Args:
        iteration: Current iteration number
        role: Iteration role
        parent_iteration: Parent iteration number
        macro_auc: Macro AUC score
        macro_f1: Macro F1 score
        decision: Decision made for next iteration

    Returns:
        Formatted string for thesis documentation
    """
    auc_vs_12 = macro_auc - ITERATION_12_AUC_ANCHOR.macro_auc
    f1_vs_58 = macro_f1 - ITERATION_58_F1_ANCHOR.macro_f1

    marker = f"""
================================================================================
>>> THESIS DOCUMENTATION MARKER <<<
ITERATION {iteration} COMPLETE
ROLE: {role.value}
PARENT: Iteration {parent_iteration if parent_iteration else 'N/A'}
MACRO AUC: {macro_auc:.4f} (vs iter12: {auc_vs_12:+.4f})
MACRO F1:  {macro_f1:.4f} (vs iter58: {f1_vs_58:+.4f})
DECISION: {decision}
================================================================================
"""
    return marker


@dataclass
class AIAdvisoryDecision:
    """
    Structured AI advisory decision for each iteration.

    This is the expected output format from the AI advisor.
    """
    phase: str  # "PHASE_1" or "PHASE_2"
    decision: str  # "CONTINUE_REPRODUCTION" | "START_F1_INJECTION" | "ADJUST_THRESHOLDS_ONLY" | "STOP_AND_DEBUG"
    reasoning: str
    parent_iteration: Optional[int]
    next_config_action: str  # "NONE" | "THRESHOLDS_ONLY" | "RETRY_SAME_CONFIG"
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "parent_iteration": self.parent_iteration,
            "next_config_action": self.next_config_action,
            "success": self.success
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AIAdvisoryDecision':
        return cls(
            phase=d.get("phase", "PHASE_1"),
            decision=d.get("decision", "STOP_AND_DEBUG"),
            reasoning=d.get("reasoning", "Unknown"),
            parent_iteration=d.get("parent_iteration"),
            next_config_action=d.get("next_config_action", "NONE"),
            success=d.get("success", False)
        )


def generate_ai_advisory_decision(
    iteration_history: List[Dict[str, Any]],
    current_metrics: Dict[str, float]
) -> AIAdvisoryDecision:
    """
    Generate an AI advisory decision based on current state.

    This function implements the strict phased protocol logic.

    Args:
        iteration_history: Full iteration history
        current_metrics: Metrics from latest iteration

    Returns:
        AIAdvisoryDecision with the decision and reasoning
    """
    phase, phase_context = determine_current_phase(iteration_history)
    current_auc = current_metrics.get('avg_auc', 0.0)
    current_f1 = current_metrics.get('avg_f1', 0.0)

    if phase == OptimizationPhase.PHASE_1_REPRODUCE:
        # Check if Phase 1 is close to succeeding
        target_auc = PHASE_CRITERIA.phase1_auc_target - PHASE_CRITERIA.phase1_auc_tolerance
        auc_gap = target_auc - current_auc

        if current_auc >= target_auc:
            # Phase 1 succeeded!
            return AIAdvisoryDecision(
                phase="PHASE_1",
                decision="START_F1_INJECTION",
                reasoning=f"Phase 1 SUCCESS! AUC={current_auc:.4f} >= {target_auc:.4f}. Ready to enter Phase 2.",
                parent_iteration=12,
                next_config_action="THRESHOLDS_ONLY",
                success=True
            )
        else:
            # Phase 1 not yet achieved
            attempts = phase_context.get('attempts', 0)
            if attempts >= PHASE_CRITERIA.phase1_max_attempts - 1:
                return AIAdvisoryDecision(
                    phase="PHASE_1",
                    decision="STOP_AND_DEBUG",
                    reasoning=f"Phase 1 FAILED after {attempts + 1} attempts. AUC={current_auc:.4f}, gap={auc_gap:.4f}. Stop and debug.",
                    parent_iteration=12,
                    next_config_action="NONE",
                    success=False
                )
            return AIAdvisoryDecision(
                phase="PHASE_1",
                decision="CONTINUE_REPRODUCTION",
                reasoning=f"Phase 1 in progress. AUC={current_auc:.4f}, target={target_auc:.4f}, gap={auc_gap:.4f}. Retry with same config.",
                parent_iteration=12,
                next_config_action="RETRY_SAME_CONFIG",
                success=False
            )

    elif phase == OptimizationPhase.PHASE_2_CALIBRATE:
        phase1_auc = phase_context.get('phase1_auc', PHASE_CRITERIA.phase1_auc_target)
        auc_preserved = current_auc >= (phase1_auc - PHASE_CRITERIA.phase2_auc_tolerance)
        f1_target_met = current_f1 >= PHASE_CRITERIA.phase2_f1_target

        if auc_preserved and f1_target_met:
            return AIAdvisoryDecision(
                phase="PHASE_2",
                decision="ADJUST_THRESHOLDS_ONLY",
                reasoning=f"SUCCESS! AUC={current_auc:.4f} preserved, F1={current_f1:.4f} >= {PHASE_CRITERIA.phase2_f1_target:.4f}",
                parent_iteration=phase_context.get('phase1_iteration'),
                next_config_action="NONE",
                success=True
            )
        elif not auc_preserved:
            return AIAdvisoryDecision(
                phase="PHASE_2",
                decision="STOP_AND_DEBUG",
                reasoning=f"AUC DEGRADED in Phase 2! AUC={current_auc:.4f} < {phase1_auc - PHASE_CRITERIA.phase2_auc_tolerance:.4f}. Stop.",
                parent_iteration=phase_context.get('phase1_iteration'),
                next_config_action="NONE",
                success=False
            )
        else:
            return AIAdvisoryDecision(
                phase="PHASE_2",
                decision="ADJUST_THRESHOLDS_ONLY",
                reasoning=f"Phase 2 in progress. AUC={current_auc:.4f} preserved. F1={current_f1:.4f}, target={PHASE_CRITERIA.phase2_f1_target:.4f}",
                parent_iteration=phase_context.get('phase1_iteration'),
                next_config_action="THRESHOLDS_ONLY",
                success=False
            )

    elif phase == OptimizationPhase.STOP_AND_DEBUG:
        return AIAdvisoryDecision(
            phase="STOP",
            decision="STOP_AND_DEBUG",
            reasoning=phase_context.get('reason', 'Reproduction failed'),
            parent_iteration=None,
            next_config_action="NONE",
            success=False
        )

    elif phase == OptimizationPhase.SUCCESS:
        return AIAdvisoryDecision(
            phase="SUCCESS",
            decision="STOP_AND_DEBUG",  # Actually success, but loop should stop
            reasoning=phase_context.get('reason', 'All criteria met'),
            parent_iteration=None,
            next_config_action="NONE",
            success=True
        )

    # Fallback
    return AIAdvisoryDecision(
        phase="UNKNOWN",
        decision="STOP_AND_DEBUG",
        reasoning="Unknown phase - stopping for safety",
        parent_iteration=None,
        next_config_action="NONE",
        success=False
    )


if __name__ == "__main__":
    # Test the module
    print("Iteration Baselines Module - PHASED PROTOCOL")
    print("=" * 60)
    print("\n📌 STRICT PHASED PROTOCOL:")
    print("  PHASE 1: Exact reproduction of Iteration 12 (AUC anchor)")
    print("  PHASE 2: Threshold calibration only (F1 injection from iter58)")
    print("  RULE: First reproduce. Then calibrate. Never mix.")
    print()
    print("📊 Reference Iterations:")
    print(f"  AUC Anchor (Iteration 12): AUC={ITERATION_12_AUC_ANCHOR.macro_auc}, F1={ITERATION_12_AUC_ANCHOR.macro_f1}")
    print(f"  F1 Anchor (Iteration 58):  AUC={ITERATION_58_F1_ANCHOR.macro_auc}, F1={ITERATION_58_F1_ANCHOR.macro_f1}")
    print()
    print("📏 Phase Criteria:")
    print(f"  Phase 1 Success: AUC >= {PHASE_CRITERIA.phase1_auc_target - PHASE_CRITERIA.phase1_auc_tolerance:.4f}")
    print(f"  Phase 2 Success: AUC preserved AND F1 >= {PHASE_CRITERIA.phase2_f1_target:.4f}")
    print(f"  Max Phase 1 attempts: {PHASE_CRITERIA.phase1_max_attempts}")
    print()
    print("Disease Groups:")
    for group, diseases in DISEASE_GROUPS.items():
        print(f"  {group}: {', '.join(diseases)}")
    print()
    print("Optimization Phases:")
    for phase in OptimizationPhase:
        print(f"  {phase.value}")
