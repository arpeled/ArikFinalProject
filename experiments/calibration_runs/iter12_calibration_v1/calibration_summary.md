# Calibration Summary - Iteration 12

## Model
- **Source checkpoint**: `auto_improvement_runs/iteration_012/pipeline_model_20251229-101232.pth`
- **Training**: FROZEN (no weight updates)
- **Inference mode**: eval() with gradients disabled

## Validation Calibration
- **Threshold search range**: [0.01, 0.5] with step 0.01
- **Default selection rule**: Maximize F1 subject to Precision >= 0.1
- **Fallback rule (rare classes)**: Maximize F1 subject to Recall >= 0.1
- **Rare classes**: Hernia, Pneumonia, Fibrosis, Edema

## Results (Test Set)

| Metric | Value | Reference (Iter 12) | Delta |
|--------|-------|---------------------|-------|
| Macro AUC | 0.8009 | 0.8009 | +0.0000 |
| Macro F1 | 0.2912 | 0.0029 | +0.2883 |
| Macro Precision | 0.2371 | - | - |
| Macro Recall | 0.4810 | - | - |

### AUC Preservation Check
- **Status**: PASSED
- **Tolerance**: +/- 0.005
- **Actual delta**: +0.0000

### Per-Class Results

| Disease | AUC | Threshold | F1 | Precision | Recall |
|---------|-----|-----------|----|-----------| -------|
| Cardiomegaly | 0.9112 | 0.21 | 0.3508 | 0.3408 | 0.3615 |
| Emphysema | 0.9080 | 0.21 | 0.4368 | 0.4007 | 0.4802 |
| Effusion | 0.8694 | 0.17 | 0.5161 | 0.4437 | 0.6166 |
| Hernia | 0.5319 | 0.01 | 0.0035 | 0.0017 | 1.0000 |
| Infiltration | 0.7031 | 0.17 | 0.4051 | 0.3108 | 0.5815 |
| Mass | 0.8537 | 0.18 | 0.4010 | 0.3367 | 0.4956 |
| Nodule | 0.7433 | 0.18 | 0.2765 | 0.2528 | 0.3052 |
| Atelectasis | 0.7997 | 0.19 | 0.3726 | 0.3117 | 0.4631 |
| Pneumothorax | 0.8812 | 0.20 | 0.3890 | 0.2979 | 0.5604 |
| Pleural_Thickening | 0.7807 | 0.20 | 0.2099 | 0.1489 | 0.3557 |
| Pneumonia | 0.7400 | 0.21 | 0.0951 | 0.0560 | 0.3160 |
| Fibrosis | 0.7976 | 0.21 | 0.1413 | 0.0962 | 0.2658 |
| Edema | 0.8951 | 0.25 | 0.2523 | 0.1691 | 0.4967 |
| Consolidation | 0.7973 | 0.20 | 0.2261 | 0.1527 | 0.4352 |

### Top 5 F1 Scores
- **Effusion**: F1 = 0.5161
- **Emphysema**: F1 = 0.4368
- **Infiltration**: F1 = 0.4051
- **Mass**: F1 = 0.4010
- **Pneumothorax**: F1 = 0.3890

## Key Observation

F1 increased from 0.0029 (iteration 12) to **0.2912** through threshold calibration alone,
while AUC remained within tolerance (+0.0000).

This demonstrates that the iteration 12 representation already contains the discriminative power
needed for decision-making. The original low F1 was due to sub-optimal threshold selection (0.5),
not poor ranking quality.

## Success Criteria Check

| Criterion | Status |
|-----------|--------|
| Test AUC within +/- 0.005 of Iter 12 | PASSED |
| Macro F1 > 0.20 | PASSED |
| No disease has Precision = 0 | PASSED |

## Files Generated

- `inference_val_predictions.csv` - Validation set probabilities
- `inference_test_predictions.csv` - Test set probabilities
- `thresholds_val_iter12.json` - Calibrated thresholds
- `evaluation_test_metrics.json` - Test set metrics
- `calibration_summary.md` - This document

---
*Generated: 2026-01-24 22:21:01*
