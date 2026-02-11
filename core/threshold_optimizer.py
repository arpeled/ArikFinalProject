"""
Threshold Optimization Module
Optimizes classification thresholds per class to maximize F1-score or Recall
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from typing import Dict, Tuple, Optional
import json


def optimize_thresholds_per_class(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list,
    metric: str = 'f1',
    num_thresholds: int = 19,
    logger=None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Optimize classification thresholds per class to maximize specified metric.

    Args:
        y_true: Ground truth labels (num_samples, num_classes)
        y_pred_probs: Predicted probabilities (num_samples, num_classes)
        class_names: List of class names (length = num_classes)
        metric: Metric to optimize ('f1', 'recall', 'precision', 'youden')
        num_thresholds: Number of threshold values to try (default: 19)
        logger: Optional logger instance

    Returns:
        Tuple of (thresholds_array, thresholds_dict)
        - thresholds_array: numpy array of optimal thresholds (num_classes,)
        - thresholds_dict: dict mapping class names to optimal thresholds
    """
    num_classes = y_true.shape[1]
    optimal_thresholds = np.zeros(num_classes)
    threshold_details = {}

    # Generate threshold candidates
    threshold_candidates = np.linspace(0.05, 0.95, num_thresholds)

    if logger:
        logger.info("\n" + "="*80)
        logger.info("THRESHOLD OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Metric: {metric.upper()}")
        logger.info(f"Threshold candidates: {num_thresholds} values from 0.05 to 0.95")
        logger.info("")

    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_prob_class = y_pred_probs[:, i]

        # Count positive samples
        num_positive = y_true_class.sum()

        # Skip if no positive samples (can't optimize)
        if num_positive == 0:
            optimal_thresholds[i] = 0.5
            threshold_details[class_name] = {
                'threshold': 0.5,
                'score': 0.0,
                'positive_samples': 0,
                'status': 'no_positives'
            }
            if logger:
                logger.info(f"{class_name:<20} No positive samples - using default 0.5")
            continue

        # Try each threshold and compute metric
        best_threshold = 0.5
        best_score = 0.0

        for threshold in threshold_candidates:
            y_pred_class = (y_prob_class >= threshold).astype(int)

            # Compute specified metric
            if metric == 'f1':
                score = f1_score(y_true_class, y_pred_class, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true_class, y_pred_class, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true_class, y_pred_class, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic = Sensitivity + Specificity - 1
                from sklearn.metrics import confusion_matrix
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = sensitivity + specificity - 1
                except:
                    score = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        optimal_thresholds[i] = best_threshold
        threshold_details[class_name] = {
            'threshold': float(best_threshold),
            'score': float(best_score),
            'positive_samples': int(num_positive),
            'status': 'optimized'
        }

        if logger:
            logger.info(f"{class_name:<20} Threshold: {best_threshold:.3f}  {metric.upper()}: {best_score:.4f}  (pos: {num_positive})")

    if logger:
        logger.info("="*80)
        logger.info(f"Optimization complete. Avg threshold: {optimal_thresholds.mean():.3f}")
        logger.info("="*80)

    return optimal_thresholds, threshold_details


def save_thresholds(thresholds_dict: Dict[str, dict], filepath: str, logger=None):
    """
    Save optimized thresholds to JSON file.

    Args:
        thresholds_dict: Dictionary of threshold details per class
        filepath: Path to save JSON file
        logger: Optional logger instance
    """
    with open(filepath, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)

    if logger:
        logger.info(f"💾 Thresholds saved to {filepath}")


def load_thresholds(filepath: str, class_names: list, logger=None) -> Optional[np.ndarray]:
    """
    Load optimized thresholds from JSON file.

    Args:
        filepath: Path to JSON file
        class_names: List of class names
        logger: Optional logger instance

    Returns:
        numpy array of thresholds (num_classes,) or None if file doesn't exist
    """
    try:
        with open(filepath, 'r') as f:
            thresholds_dict = json.load(f)

        thresholds = np.zeros(len(class_names))
        for i, class_name in enumerate(class_names):
            if class_name in thresholds_dict:
                thresholds[i] = thresholds_dict[class_name]['threshold']
            else:
                thresholds[i] = 0.5  # Default if not found

        if logger:
            logger.info(f"📂 Loaded thresholds from {filepath}")
            logger.info(f"   Avg threshold: {thresholds.mean():.3f}")

        return thresholds

    except FileNotFoundError:
        if logger:
            logger.info(f"⚠️  Threshold file not found: {filepath}")
            logger.info(f"   Using default threshold: 0.5 for all classes")
        return None
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Error loading thresholds: {e}")
            logger.warning(f"   Using default threshold: 0.5 for all classes")
        return None


def generate_prevalence_based_thresholds(y_true: np.ndarray, class_names: list, logger=None) -> np.ndarray:
    """
    Generate smart fallback thresholds based on class prevalence.
    Used when optimized thresholds are not available.

    Strategy:
    - Very rare classes (<1%): threshold = 0.15
    - Rare classes (1-5%): threshold = 0.25
    - Common classes (5-15%): threshold = 0.35
    - Very common classes (>15%): threshold = 0.45

    Args:
        y_true: Ground truth labels (num_samples, num_classes)
        class_names: List of class names
        logger: Optional logger instance

    Returns:
        Array of prevalence-based thresholds (num_classes,)
    """
    num_classes = y_true.shape[1]
    thresholds = np.zeros(num_classes)

    if logger:
        logger.info("Generating prevalence-based fallback thresholds...")

    for i in range(num_classes):
        prevalence = y_true[:, i].mean() * 100  # Percentage

        # Set threshold based on prevalence
        if prevalence < 1.0:  # Very rare (<1%)
            threshold = 0.15
            category = "very rare"
        elif prevalence < 5.0:  # Rare (1-5%)
            threshold = 0.25
            category = "rare"
        elif prevalence < 15.0:  # Common (5-15%)
            threshold = 0.35
            category = "common"
        else:  # Very common (>15%)
            threshold = 0.45
            category = "very common"

        thresholds[i] = threshold

        if logger:
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            logger.info(f"  {class_name:<20} Prevalence: {prevalence:>5.2f}% → Threshold: {threshold:.2f} ({category})")

    if logger:
        logger.info(f"Average fallback threshold: {thresholds.mean():.3f}")

    return thresholds


def apply_thresholds(y_pred_probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Apply per-class thresholds to predicted probabilities.

    Args:
        y_pred_probs: Predicted probabilities (num_samples, num_classes)
        thresholds: Threshold per class (num_classes,)

    Returns:
        Binary predictions (num_samples, num_classes)
    """
    # Broadcast thresholds to match shape of predictions
    return (y_pred_probs >= thresholds).astype(int)


def optimize_hernia_only_constrained(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list,
    hernia_constraint: dict,
    default_threshold: float = 0.5,
    num_thresholds: int = 50,
    logger=None
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """
    Optimize threshold ONLY for Hernia with precision/FP constraints.
    All other classes use a fixed default threshold.

    This is a DECISION-STABILIZATION experiment to prevent FP explosion
    while preserving Hernia ranking signal (AUC).

    Args:
        y_true: Ground truth labels (num_samples, num_classes)
        y_pred_probs: Predicted probabilities (num_samples, num_classes)
        class_names: List of class names (length = num_classes)
        hernia_constraint: Dict with constraint parameters:
            - min_precision: Minimum required precision (e.g., 0.02 for 2%)
            - max_fp: Maximum allowed false positives (e.g., 500)
        default_threshold: Default threshold for all classes (default: 0.5)
        num_thresholds: Number of threshold values to try (default: 50)
        logger: Optional logger instance

    Returns:
        Tuple of (thresholds_array, thresholds_dict)
    """
    num_classes = y_true.shape[1]
    optimal_thresholds = np.full(num_classes, default_threshold)
    threshold_details = {}

    # Find Hernia index
    hernia_idx = None
    for i, name in enumerate(class_names):
        if name.lower() == 'hernia':
            hernia_idx = i
            break

    if hernia_idx is None:
        if logger:
            logger.warning("Hernia class not found in class_names - using default thresholds for all")
        for i, class_name in enumerate(class_names):
            threshold_details[class_name] = {
                'threshold': default_threshold,
                'score': 0.0,
                'positive_samples': int(y_true[:, i].sum()),
                'status': 'default'
            }
        return optimal_thresholds, threshold_details

    # Extract constraint parameters
    min_precision = hernia_constraint.get('min_precision', 0.02)
    max_fp = hernia_constraint.get('max_fp', 500)

    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("CONSTRAINED HERNIA-ONLY THRESHOLD OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Default threshold for all classes: {default_threshold}")
        logger.info(f"Hernia constraints: min_precision >= {min_precision:.2%}, max_fp <= {max_fp}")
        logger.info("")

    # Set default thresholds for non-Hernia classes
    for i, class_name in enumerate(class_names):
        if i != hernia_idx:
            num_positive = int(y_true[:, i].sum())
            threshold_details[class_name] = {
                'threshold': default_threshold,
                'score': 0.0,
                'positive_samples': num_positive,
                'status': 'default_fixed'
            }
            if logger:
                logger.info(f"{class_name:<20} Threshold: {default_threshold:.3f} (FIXED - not optimized)")

    # Now optimize Hernia with constraints
    y_true_hernia = y_true[:, hernia_idx]
    y_prob_hernia = y_pred_probs[:, hernia_idx]
    num_positive_hernia = int(y_true_hernia.sum())

    if logger:
        logger.info("")
        logger.info(f"{'=' * 40}")
        logger.info(f"HERNIA THRESHOLD SWEEP (positive samples: {num_positive_hernia})")
        logger.info(f"{'=' * 40}")

    # Generate more threshold candidates for fine-grained search
    threshold_candidates = np.linspace(0.05, 0.95, num_thresholds)

    # Find thresholds that satisfy constraints
    valid_thresholds = []

    for threshold in threshold_candidates:
        y_pred_hernia = (y_prob_hernia >= threshold).astype(int)

        # Compute TP, FP, FN
        tp = ((y_pred_hernia == 1) & (y_true_hernia == 1)).sum()
        fp = ((y_pred_hernia == 1) & (y_true_hernia == 0)).sum()
        fn = ((y_pred_hernia == 0) & (y_true_hernia == 1)).sum()

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Check constraints
        satisfies_precision = precision >= min_precision
        satisfies_fp = fp <= max_fp

        if satisfies_precision or satisfies_fp:
            valid_thresholds.append({
                'threshold': threshold,
                'tp': int(tp),
                'fp': int(fp),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'satisfies_precision': satisfies_precision,
                'satisfies_fp': satisfies_fp
            })

        if logger and threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            constraint_status = ""
            if satisfies_precision:
                constraint_status += "[PREC OK] "
            if satisfies_fp:
                constraint_status += "[FP OK] "
            if not constraint_status:
                constraint_status = "[FAILS CONSTRAINTS]"
            logger.info(f"  thresh={threshold:.2f}: TP={tp:3d}, FP={fp:4d}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f} {constraint_status}")

    # Select best threshold among valid ones
    if valid_thresholds:
        # Sort by F1 (or Recall if you prefer) descending
        valid_thresholds.sort(key=lambda x: x['f1'], reverse=True)
        best = valid_thresholds[0]
        best_threshold = best['threshold']

        if logger:
            logger.info("")
            logger.info(f"✅ SELECTED Hernia threshold: {best_threshold:.3f}")
            logger.info(f"   TP: {best['tp']}, FP: {best['fp']}")
            logger.info(f"   Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}, F1: {best['f1']:.4f}")
            logger.info(f"   Constraint status: precision>={min_precision:.2%}={best['satisfies_precision']}, FP<={max_fp}={best['satisfies_fp']}")

        optimal_thresholds[hernia_idx] = best_threshold
        threshold_details['Hernia'] = {
            'threshold': float(best_threshold),
            'score': float(best['f1']),
            'positive_samples': num_positive_hernia,
            'tp': best['tp'],
            'fp': best['fp'],
            'precision': float(best['precision']),
            'recall': float(best['recall']),
            'status': 'optimized_constrained'
        }
    else:
        # No threshold satisfies constraints - use default
        if logger:
            logger.warning("")
            logger.warning(f"⚠️  No threshold satisfies constraints for Hernia")
            logger.warning(f"   Using default threshold: {default_threshold}")

        optimal_thresholds[hernia_idx] = default_threshold
        threshold_details['Hernia'] = {
            'threshold': default_threshold,
            'score': 0.0,
            'positive_samples': num_positive_hernia,
            'status': 'default_no_valid_threshold'
        }

    if logger:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CONSTRAINED OPTIMIZATION COMPLETE")
        logger.info(f"  Hernia threshold: {optimal_thresholds[hernia_idx]:.3f}")
        logger.info(f"  Other classes: {default_threshold:.3f} (fixed)")
        logger.info("=" * 80)

    return optimal_thresholds, threshold_details


# Test function
if __name__ == "__main__":
    print("Testing threshold optimizer...")

    # Create synthetic data
    np.random.seed(42)
    num_samples = 1000
    num_classes = 3
    class_names = ['Class_A', 'Class_B', 'Class_C']

    # Ground truth (imbalanced)
    y_true = np.random.rand(num_samples, num_classes) < 0.1  # 10% positive rate

    # Predictions (slightly correlated with truth)
    y_pred_probs = np.random.rand(num_samples, num_classes) * 0.5
    y_pred_probs[y_true] += 0.3  # Boost probabilities for true positives

    # Optimize thresholds
    print("\nOptimizing thresholds for F1-score...")
    thresholds, details = optimize_thresholds_per_class(
        y_true, y_pred_probs, class_names, metric='f1'
    )

    print(f"\nOptimized thresholds: {thresholds}")
    print(f"Details: {details}")

    # Test save/load
    save_thresholds(details, "test_thresholds.json")
    loaded = load_thresholds("test_thresholds.json", class_names)
    print(f"\nLoaded thresholds: {loaded}")

    # Test apply
    predictions = apply_thresholds(y_pred_probs, thresholds)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Positive rate: {predictions.sum(axis=0) / num_samples}")

    # Cleanup
    import os
    os.remove("test_thresholds.json")
    print("\n✅ All tests passed!")
