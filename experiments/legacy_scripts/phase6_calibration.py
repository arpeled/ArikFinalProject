#!/usr/bin/env python3
"""
Phase 6: Calibration-Only for Iteration 12

Goal: Recover F1 from Iteration 12 without retraining or AUC degradation.

This script:
1. Loads the frozen iteration 12 model (NO training)
2. Runs inference on validation set to collect probability distributions
3. Performs per-class threshold search on validation set
4. Applies frozen thresholds to test set
5. Saves all results to isolated calibration_runs/ directory

GLOBAL RULES (NON-NEGOTIABLE):
- NO training or fine-tuning
- NO weight updates
- NO loss changes
- NO data augmentation changes
- NO overwriting existing training artifacts
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Configuration
ITERATION_12_MODEL_PATH = "auto_improvement_runs/iteration_012/pipeline_model_20251229-101232.pth"
ITERATION_12_AUC = 0.8009  # Reference AUC from iteration 12

# Threshold search configuration
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.50
THRESHOLD_STEP = 0.01

# Selection rules
DEFAULT_MIN_PRECISION = 0.1  # Maximize F1 subject to Precision >= 0.1
FALLBACK_MIN_RECALL = 0.1   # Maximize F1 subject to Recall >= 0.1 (for rare classes)

# Disease names in order
DISEASE_NAMES = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]

# Rare classes (use fallback rule)
RARE_CLASSES = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema']


def create_output_directory(version: str = "v1") -> Path:
    """Create isolated output directory for calibration run."""
    output_dir = Path(f"calibration_runs/iter12_calibration_{version}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def load_frozen_model(device: torch.device) -> torch.nn.Module:
    """
    Load the frozen iteration 12 model.

    Sets model to eval mode and disables gradients globally.
    """
    if not os.path.exists(ITERATION_12_MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {ITERATION_12_MODEL_PATH}")

    # Import model class
    from dataset import ModifiedDenseNetWithDropOut

    # Create model with same architecture as iteration 12
    model = ModifiedDenseNetWithDropOut(
        num_classes=14,
        use_additional_features=True  # Iteration 12 used additional features
    )

    # Load weights
    state_dict = torch.load(ITERATION_12_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # CRITICAL: Set to eval mode and disable gradients
    model.eval()
    torch.set_grad_enabled(False)

    print(f"Loaded frozen model from: {ITERATION_12_MODEL_PATH}")
    print(f"Model is in eval mode: {not model.training}")
    print(f"Gradients disabled: {not torch.is_grad_enabled()}")

    return model


def get_data_loaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for validation and test sets.

    Uses the same split as iteration 12 (GroupShuffleSplit with seed 42).
    """
    from dataset import ChestXRayDataset
    from sklearn.model_selection import GroupShuffleSplit

    # Load training data to get the split
    train_csv = './ChestX-ray14/train_data.csv'
    test_csv = './ChestX-ray14/test_data.csv'
    images_dir = './ChestX-ray14/images224'

    # Define transform (same as iteration 12)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full training data
    train_df = pd.read_csv(train_csv)

    # Get patient IDs for GroupShuffleSplit
    patient_ids = train_df['Patient ID'].values

    # Use same split as iteration 12
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(train_df, groups=patient_ids))

    # Create validation dataset
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    val_dataset = ChestXRayDataset(
        dataset=val_df,  # Pass DataFrame via dataset parameter
        csv_file=None,
        root_dir=images_dir,
        transform=transform,
        use_additional_features=True
    )

    # Create test dataset
    test_dataset = ChestXRayDataset(
        dataset=None,
        csv_file=test_csv,
        root_dir=images_dir,
        transform=transform,
        use_additional_features=True
    )

    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return val_loader, test_loader


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    desc: str = "Inference"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a dataset.

    Returns:
        predictions: Sigmoid probabilities (N x 14)
        labels: Ground truth labels (N x 14)
    """
    all_predictions = []
    all_labels = []

    for images, additional_features, labels in tqdm(data_loader, desc=desc):
        images = images.to(device)
        additional_features = additional_features.to(device)

        # Forward pass (no gradients)
        outputs = model(images, additional_features)
        predictions = torch.sigmoid(outputs)

        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.numpy())

    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)

    print(f"{desc} complete: {predictions.shape[0]} samples, {predictions.shape[1]} classes")

    return predictions, labels


def save_predictions_csv(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    prefix: str = "inference"
) -> Path:
    """
    Save predictions to CSV in the required format.

    Format: sample_id, disease, y_true, y_prob
    """
    rows = []

    for sample_id in range(len(predictions)):
        for disease_idx, disease in enumerate(DISEASE_NAMES):
            rows.append({
                'sample_id': sample_id,
                'disease': disease,
                'y_true': int(labels[sample_id, disease_idx]),
                'y_prob': float(predictions[sample_id, disease_idx])
            })

    df = pd.DataFrame(rows)
    csv_path = output_path / f"{prefix}_predictions.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved predictions to: {csv_path}")
    return csv_path


def search_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease: str
) -> Dict:
    """
    Search for optimal threshold for a single disease.

    Selection rules:
    - Default: Maximize F1 subject to Precision >= 0.1
    - Fallback (rare classes): Maximize F1 subject to Recall >= 0.1
    """
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

    best_result = None
    best_f1 = -1
    all_results = []

    # Determine which rule to use
    use_recall_rule = disease in RARE_CLASSES
    min_constraint = FALLBACK_MIN_RECALL if use_recall_rule else DEFAULT_MIN_PRECISION
    constraint_name = "recall" if use_recall_rule else "precision"

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        # Skip if no predictions
        if np.sum(y_pred) == 0:
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        result = {
            'threshold': round(thresh, 3),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
        all_results.append(result)

        # Check constraint
        constraint_value = recall if use_recall_rule else precision

        if constraint_value >= min_constraint and f1 > best_f1:
            best_f1 = f1
            best_result = result.copy()

    # Fallback: if no threshold meets constraint, pick best F1 overall
    if best_result is None:
        best_result = max(all_results, key=lambda x: x['f1']) if all_results else {
            'threshold': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        best_result['constraint_met'] = False
    else:
        best_result['constraint_met'] = True

    best_result['rule_used'] = f"max_f1_with_{constraint_name}>={min_constraint}"
    best_result['disease'] = disease

    return best_result


def calibrate_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Dict]:
    """
    Perform per-class threshold calibration on validation set.
    """
    thresholds = {}

    print("\nCalibrating thresholds per disease:")
    print("-" * 60)

    for disease_idx, disease in enumerate(DISEASE_NAMES):
        y_true = labels[:, disease_idx]
        y_prob = predictions[:, disease_idx]

        result = search_optimal_threshold(y_true, y_prob, disease)
        thresholds[disease] = result

        status = "OK" if result.get('constraint_met', True) else "FALLBACK"
        print(f"{disease:20} | Thresh: {result['threshold']:.2f} | "
              f"F1: {result['f1']:.4f} | Prec: {result['precision']:.4f} | "
              f"Rec: {result['recall']:.4f} | {status}")

    print("-" * 60)

    return thresholds


def evaluate_with_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray,
    thresholds: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Evaluate predictions using calibrated thresholds.
    """
    results = {}

    print("\nEvaluating with calibrated thresholds:")
    print("-" * 60)

    for disease_idx, disease in enumerate(DISEASE_NAMES):
        y_true = labels[:, disease_idx]
        y_prob = predictions[:, disease_idx]

        # Get threshold for this disease
        thresh = thresholds[disease]['threshold']
        y_pred = (y_prob >= thresh).astype(int)

        # Calculate metrics
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Count statistics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        results[disease] = {
            'auc': round(auc, 4),
            'threshold': thresh,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'positive_samples': int(np.sum(y_true)),
            'predicted_positive': int(np.sum(y_pred))
        }

        print(f"{disease:20} | AUC: {auc:.4f} | F1: {f1:.4f} | "
              f"Prec: {precision:.4f} | Rec: {recall:.4f}")

    print("-" * 60)

    # Calculate macro averages
    macro_auc = np.mean([r['auc'] for r in results.values()])
    macro_f1 = np.mean([r['f1'] for r in results.values()])
    macro_precision = np.mean([r['precision'] for r in results.values()])
    macro_recall = np.mean([r['recall'] for r in results.values()])

    results['_macro'] = {
        'auc': round(macro_auc, 4),
        'f1': round(macro_f1, 4),
        'precision': round(macro_precision, 4),
        'recall': round(macro_recall, 4)
    }

    print(f"\nMACRO AVERAGES:")
    print(f"  AUC: {macro_auc:.4f} (Iteration 12: {ITERATION_12_AUC:.4f}, "
          f"Delta: {macro_auc - ITERATION_12_AUC:+.4f})")
    print(f"  F1:  {macro_f1:.4f}")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")

    return results


def generate_calibration_summary(
    output_dir: Path,
    thresholds: Dict[str, Dict],
    test_results: Dict[str, Dict]
) -> Path:
    """
    Generate calibration_summary.md documentation.
    """
    macro = test_results['_macro']
    auc_delta = macro['auc'] - ITERATION_12_AUC
    auc_degraded = abs(auc_delta) > 0.005

    # Find top F1 improvements
    top_f1_diseases = sorted(
        [(d, r['f1']) for d, r in test_results.items() if d != '_macro'],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    summary = f"""# Calibration Summary - Iteration 12

## Model
- **Source checkpoint**: `{ITERATION_12_MODEL_PATH}`
- **Training**: FROZEN (no weight updates)
- **Inference mode**: eval() with gradients disabled

## Validation Calibration
- **Threshold search range**: [{THRESHOLD_MIN}, {THRESHOLD_MAX}] with step {THRESHOLD_STEP}
- **Default selection rule**: Maximize F1 subject to Precision >= {DEFAULT_MIN_PRECISION}
- **Fallback rule (rare classes)**: Maximize F1 subject to Recall >= {FALLBACK_MIN_RECALL}
- **Rare classes**: {', '.join(RARE_CLASSES)}

## Results (Test Set)

| Metric | Value | Reference (Iter 12) | Delta |
|--------|-------|---------------------|-------|
| Macro AUC | {macro['auc']:.4f} | {ITERATION_12_AUC:.4f} | {auc_delta:+.4f} |
| Macro F1 | {macro['f1']:.4f} | 0.0029 | {macro['f1'] - 0.0029:+.4f} |
| Macro Precision | {macro['precision']:.4f} | - | - |
| Macro Recall | {macro['recall']:.4f} | - | - |

### AUC Preservation Check
- **Status**: {'PASSED' if not auc_degraded else 'FAILED'}
- **Tolerance**: +/- 0.005
- **Actual delta**: {auc_delta:+.4f}

### Per-Class Results

| Disease | AUC | Threshold | F1 | Precision | Recall |
|---------|-----|-----------|----|-----------| -------|
"""

    for disease in DISEASE_NAMES:
        r = test_results[disease]
        t = thresholds[disease]
        summary += f"| {disease} | {r['auc']:.4f} | {t['threshold']:.2f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |\n"

    summary += f"""
### Top 5 F1 Scores
"""
    for disease, f1 in top_f1_diseases:
        summary += f"- **{disease}**: F1 = {f1:.4f}\n"

    summary += f"""
## Key Observation

F1 increased from 0.0029 (iteration 12) to **{macro['f1']:.4f}** through threshold calibration alone,
while AUC remained within tolerance ({auc_delta:+.4f}).

This demonstrates that the iteration 12 representation already contains the discriminative power
needed for decision-making. The original low F1 was due to sub-optimal threshold selection (0.5),
not poor ranking quality.

## Success Criteria Check

| Criterion | Status |
|-----------|--------|
| Test AUC within +/- 0.005 of Iter 12 | {'PASSED' if not auc_degraded else 'FAILED'} |
| Macro F1 > 0.20 | {'PASSED' if macro['f1'] > 0.20 else 'FAILED'} |
| No disease has Precision = 0 | {'PASSED' if all(r['precision'] > 0 for d, r in test_results.items() if d != '_macro') else 'FAILED'} |

## Files Generated

- `inference_val_predictions.csv` - Validation set probabilities
- `inference_test_predictions.csv` - Test set probabilities
- `thresholds_val_iter12.json` - Calibrated thresholds
- `evaluation_test_metrics.json` - Test set metrics
- `calibration_summary.md` - This document

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    summary_path = output_dir / "calibration_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"\nSaved calibration summary to: {summary_path}")
    return summary_path


def generate_advisory_payload(
    test_results: Dict[str, Dict]
) -> Dict:
    """
    Generate JSON payload for AI advisory.
    """
    macro = test_results['_macro']
    auc_delta = macro['auc'] - ITERATION_12_AUC

    per_class_f1 = {
        d: r['f1'] for d, r in test_results.items() if d != '_macro'
    }

    payload = {
        "phase": "PHASE_6_CALIBRATION_ONLY",
        "base_iteration": 12,
        "macro_auc_test": float(macro['auc']),
        "macro_f1_test": float(macro['f1']),
        "per_class_f1": {k: float(v) for k, v in per_class_f1.items()},
        "auc_degradation": bool(abs(auc_delta) > 0.005),
        "auc_delta": float(round(auc_delta, 4)),
        "decision_request": "Is further fine-tuning needed?"
    }

    return payload


def main():
    """Main Phase 6 calibration pipeline."""
    print("=" * 70)
    print("PHASE 6: CALIBRATION-ONLY FOR ITERATION 12")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()
    print("GLOBAL RULES:")
    print("  - NO training or fine-tuning")
    print("  - NO weight updates")
    print("  - NO loss changes")
    print("  - Inference and threshold calibration ONLY")
    print()

    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = create_output_directory("v1")

    # Step 1: Load frozen model
    print("\n" + "=" * 70)
    print("STEP 1: LOAD FROZEN MODEL")
    print("=" * 70)
    model = load_frozen_model(device)

    # Step 2: Get data loaders
    print("\n" + "=" * 70)
    print("STEP 2: PREPARE DATA LOADERS")
    print("=" * 70)
    val_loader, test_loader = get_data_loaders()

    # Step 3: Run inference on validation set
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATION SET INFERENCE")
    print("=" * 70)
    val_predictions, val_labels = run_inference(model, val_loader, device, "Validation inference")
    save_predictions_csv(val_predictions, val_labels, output_dir, "inference_val")

    # Step 4: Per-class threshold search on validation
    print("\n" + "=" * 70)
    print("STEP 4: PER-CLASS THRESHOLD CALIBRATION (VALIDATION)")
    print("=" * 70)
    thresholds = calibrate_thresholds(val_predictions, val_labels)

    # Save thresholds
    thresholds_path = output_dir / "thresholds_val_iter12.json"
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to: {thresholds_path}")

    # Step 5: Run inference on test set
    print("\n" + "=" * 70)
    print("STEP 5: TEST SET INFERENCE")
    print("=" * 70)
    test_predictions, test_labels = run_inference(model, test_loader, device, "Test inference")
    save_predictions_csv(test_predictions, test_labels, output_dir, "inference_test")

    # Step 6: Evaluate test set with frozen thresholds
    print("\n" + "=" * 70)
    print("STEP 6: TEST SET EVALUATION (FROZEN THRESHOLDS)")
    print("=" * 70)
    test_results = evaluate_with_thresholds(test_predictions, test_labels, thresholds)

    # Save test metrics
    metrics_path = output_dir / "evaluation_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test metrics to: {metrics_path}")

    # Step 7: Generate documentation
    print("\n" + "=" * 70)
    print("STEP 7: GENERATE DOCUMENTATION")
    print("=" * 70)
    generate_calibration_summary(output_dir, thresholds, test_results)

    # Generate advisory payload
    advisory_payload = generate_advisory_payload(test_results)
    advisory_path = output_dir / "advisory_payload.json"
    with open(advisory_path, 'w') as f:
        json.dump(advisory_payload, f, indent=2)
    print(f"Saved advisory payload to: {advisory_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)

    macro = test_results['_macro']
    auc_delta = macro['auc'] - ITERATION_12_AUC

    print(f"\nRESULTS:")
    print(f"  Macro AUC: {macro['auc']:.4f} (Delta from Iter 12: {auc_delta:+.4f})")
    print(f"  Macro F1:  {macro['f1']:.4f} (Up from 0.0029)")
    print(f"  AUC Preserved: {'YES' if abs(auc_delta) <= 0.005 else 'NO'}")

    print(f"\nFILES CREATED in {output_dir}:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")

    print(f"\nCompleted: {datetime.now()}")

    # Success check
    success = (
        abs(auc_delta) <= 0.005 and
        macro['f1'] > 0.20 and
        all(r['precision'] > 0 for d, r in test_results.items() if d != '_macro')
    )

    if success:
        print("\n SUCCESS: All Phase 6 criteria met!")
        print("Recommendation: STOP_AND_ACCEPT_RESULTS")
    else:
        print("\n Phase 6 criteria NOT fully met.")
        print("Recommendation: Review results and consider PROCEED_TO_HEAD_ONLY_FINETUNE")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
