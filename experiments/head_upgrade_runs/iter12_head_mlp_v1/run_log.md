## iter12_head_mlp_v1
- Base checkpoint: iteration_012
- Change: replace linear head with MLP (512 hidden, relu, dropout=0.2)
- Backbone: frozen
- Trainable params: 597,518

### Validation
- Best val_macro_auc: 0.8148
- Epoch of best model: 14
- Total epochs run: 15

### Test
- Macro AUC: 0.8198
- Delta Macro AUC vs Iter12: +0.0189

### Per-class AUC deltas:
- Cardiomegaly: +0.0018
- Emphysema: +0.0002
- Effusion: +0.0062
- Hernia: +0.2349
- Infiltration: -0.0020
- Mass: +0.0038
- Nodule: +0.0032
- Atelectasis: +0.0018
- Pneumothorax: +0.0040
- Pleural_Thickening: +0.0035
- Pneumonia: +0.0073
- Fibrosis: -0.0020
- Edema: -0.0031
- Consolidation: +0.0048

### Summary
- Classes improved (>= +0.005): 3 (Effusion, Hernia, Pneumonia)
- Classes regressed (< -0.01): 0 (none)

### Notes
- Training time: 4008.6s (66.8 min)
- Early stopping: No


## Phase 7A-B - Full Metric Evaluation
- Calibration: validation-based, per-class thresholds
- Metrics computed: AUC, Precision, Recall, F1
- Confusion matrices generated for all classes
- Comparison baseline: Iteration 12

### Summary
- Macro AUC: 0.8198 (change: +0.0189)
- Macro Precision: 0.2387 (change: +0.1136)
- Macro Recall: 0.4071 (change: +0.3344)
- Macro F1: 0.2915 (change: +0.2886)

### Notable Per-Class Improvements
- Cardiomegaly: F1 0.0000 -> 0.3497 (+0.3497)
- Emphysema: F1 0.0349 -> 0.4413 (+0.4064)
- Effusion: F1 0.0000 -> 0.5220 (+0.5220)
- Hernia: F1 0.0035 -> 0.0190 (+0.0156)
- Infiltration: F1 0.0000 -> 0.4040 (+0.4040)
- Mass: F1 0.0018 -> 0.3947 (+0.3929)
- Nodule: F1 0.0000 -> 0.2783 (+0.2783)
- Atelectasis: F1 0.0000 -> 0.3775 (+0.3775)
- Pneumothorax: F1 0.0000 -> 0.4174 (+0.4174)
- Pleural_Thickening: F1 0.0000 -> 0.2086 (+0.2086)
- Pneumonia: F1 0.0000 -> 0.0510 (+0.0510)
- Fibrosis: F1 0.0000 -> 0.1353 (+0.1353)
- Edema: F1 0.0000 -> 0.2660 (+0.2660)
- Consolidation: F1 0.0000 -> 0.2157 (+0.2157)
