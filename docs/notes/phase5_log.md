# Phase 5: Class-Specific AUC Improvement Log

This log documents each iteration in Phase 5.
**Key Insight**: AUC improvement is class-specific and achieved by noise suppression, not recall forcing.

## Baseline (Iteration 84)

- **Date**: 2026-01-24
- **Macro AUC**: 0.7659
- **Model**: `auto_improvement_runs/iteration_084/pipeline_model_20260124-003356.pth`

### Per-Class AUC Baseline:

| Disease | AUC | Category |
|---------|-----|----------|
| Cardiomegaly | 0.8230 | Stable |
| Emphysema | 0.8513 | Stable |
| Effusion | 0.8308 | Stable |
| Hernia | 0.8561 | Stable |
| Pneumothorax | 0.8469 | Stable |
| Edema | 0.8367 | Target |
| Mass | 0.7835 | - |
| Atelectasis | 0.7421 | - |
| Consolidation | 0.7264 | - |
| Fibrosis | 0.7226 | Target |
| Pleural_Thickening | 0.7169 | - |
| Nodule | 0.6959 | - |
| Infiltration | 0.6503 | - |
| Pneumonia | 0.6395 | Target |

### Target Diseases (Lowest AUC):
- Pneumonia (0.6395)
- Fibrosis (0.7226)
- Edema (0.8367)

### Stable Diseases (Must Not Regress):
- Effusion, Emphysema, Cardiomegaly, Hernia, Pneumothorax

---

## Phase 5 Iterations

<!-- Iterations will be appended below this line -->


## Iteration 87
- **Date**: 2026-02-03 20:10
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8199 (baseline: 0.7659, delta: +0.0540)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8042 | +0.0621 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9123 | +0.0893 | IMPROVED |
| Consolidation | 0.7264 | 0.8040 | +0.0776 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8911 | +0.0544 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8755 | +0.0447 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9099 | +0.0586 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7953 | +0.0727 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7603 | -0.0958 | REGRESSED |
| Infiltration | 0.6503 | 0.7033 | +0.0530 | IMPROVED |
| Mass | 0.7835 | 0.8602 | +0.0767 | IMPROVED |
| Nodule | 0.6959 | 0.7513 | +0.0554 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7825 | +0.0656 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7443 | +0.1048 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8840 | +0.0371 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 89
- **Date**: 2026-02-03 22:46
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8194 (baseline: 0.7659, delta: +0.0535)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8040 | +0.0619 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9119 | +0.0889 | IMPROVED |
| Consolidation | 0.7264 | 0.8038 | +0.0774 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8901 | +0.0534 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8756 | +0.0448 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9104 | +0.0591 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7948 | +0.0722 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7528 | -0.1033 | REGRESSED |
| Infiltration | 0.6503 | 0.7028 | +0.0525 | IMPROVED |
| Mass | 0.7835 | 0.8599 | +0.0764 | IMPROVED |
| Nodule | 0.6959 | 0.7511 | +0.0552 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7838 | +0.0669 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7453 | +0.1058 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8847 | +0.0378 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 91
- **Date**: 2026-02-04 09:50
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8201 (baseline: 0.7659, delta: +0.0542)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8043 | +0.0622 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9120 | +0.0890 | IMPROVED |
| Consolidation | 0.7264 | 0.8039 | +0.0775 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8911 | +0.0544 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8759 | +0.0451 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9097 | +0.0584 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7956 | +0.0730 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7619 | -0.0942 | REGRESSED |
| Infiltration | 0.6503 | 0.7034 | +0.0531 | IMPROVED |
| Mass | 0.7835 | 0.8601 | +0.0766 | IMPROVED |
| Nodule | 0.6959 | 0.7512 | +0.0553 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7842 | +0.0673 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7433 | +0.1038 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8849 | +0.0380 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 93
- **Date**: 2026-02-04 12:10
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8201 (baseline: 0.7659, delta: +0.0542)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8044 | +0.0623 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9123 | +0.0893 | IMPROVED |
| Consolidation | 0.7264 | 0.8040 | +0.0776 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8913 | +0.0546 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8756 | +0.0448 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9095 | +0.0582 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7957 | +0.0731 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7631 | -0.0930 | REGRESSED |
| Infiltration | 0.6503 | 0.7026 | +0.0523 | IMPROVED |
| Mass | 0.7835 | 0.8601 | +0.0766 | IMPROVED |
| Nodule | 0.6959 | 0.7513 | +0.0554 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7821 | +0.0652 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7445 | +0.1050 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8846 | +0.0377 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 95
- **Date**: 2026-02-04 20:47
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8192 (baseline: 0.7659, delta: +0.0533)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8015 | +0.0594 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9111 | +0.0881 | IMPROVED |
| Consolidation | 0.7264 | 0.8022 | +0.0758 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8892 | +0.0525 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8724 | +0.0416 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9091 | +0.0578 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7944 | +0.0718 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7777 | -0.0784 | REGRESSED |
| Infiltration | 0.6503 | 0.6985 | +0.0482 | IMPROVED |
| Mass | 0.7835 | 0.8582 | +0.0747 | IMPROVED |
| Nodule | 0.6959 | 0.7506 | +0.0547 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7817 | +0.0648 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7423 | +0.1028 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8795 | +0.0326 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 97
- **Date**: 2026-02-04 21:05
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8184 (baseline: 0.7659, delta: +0.0525)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8025 | +0.0604 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9126 | +0.0896 | IMPROVED |
| Consolidation | 0.7264 | 0.8024 | +0.0760 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8847 | +0.0480 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8710 | +0.0402 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9093 | +0.0580 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7938 | +0.0712 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7686 | -0.0875 | REGRESSED |
| Infiltration | 0.6503 | 0.6976 | +0.0473 | IMPROVED |
| Mass | 0.7835 | 0.8599 | +0.0764 | IMPROVED |
| Nodule | 0.6959 | 0.7512 | +0.0553 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7821 | +0.0652 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7420 | +0.1025 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8800 | +0.0331 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 98
- **Date**: 2026-02-04 21:09
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8190 (baseline: 0.7659, delta: +0.0531)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8031 | +0.0610 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9123 | +0.0893 | IMPROVED |
| Consolidation | 0.7264 | 0.8039 | +0.0775 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8884 | +0.0517 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8730 | +0.0422 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9074 | +0.0561 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7962 | +0.0736 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7648 | -0.0913 | REGRESSED |
| Infiltration | 0.6503 | 0.7008 | +0.0505 | IMPROVED |
| Mass | 0.7835 | 0.8598 | +0.0763 | IMPROVED |
| Nodule | 0.6959 | 0.7504 | +0.0545 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7807 | +0.0638 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7449 | +0.1054 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8802 | +0.0333 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 100
- **Date**: 2026-02-04 21:32
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8189 (baseline: 0.7659, delta: +0.0530)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8022 | +0.0601 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9099 | +0.0869 | IMPROVED |
| Consolidation | 0.7264 | 0.8023 | +0.0759 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8854 | +0.0487 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8727 | +0.0419 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9094 | +0.0581 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7960 | +0.0734 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7782 | -0.0779 | REGRESSED |
| Infiltration | 0.6503 | 0.6982 | +0.0479 | IMPROVED |
| Mass | 0.7835 | 0.8592 | +0.0757 | IMPROVED |
| Nodule | 0.6959 | 0.7497 | +0.0538 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7820 | +0.0651 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7384 | +0.0989 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8815 | +0.0346 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 101
- **Date**: 2026-02-04 21:36
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8185 (baseline: 0.7659, delta: +0.0526)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8011 | +0.0590 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9127 | +0.0897 | IMPROVED |
| Consolidation | 0.7264 | 0.8049 | +0.0785 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8899 | +0.0532 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8735 | +0.0427 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9074 | +0.0561 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7959 | +0.0733 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7654 | -0.0907 | REGRESSED |
| Infiltration | 0.6503 | 0.7003 | +0.0500 | IMPROVED |
| Mass | 0.7835 | 0.8566 | +0.0731 | IMPROVED |
| Nodule | 0.6959 | 0.7513 | +0.0554 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7785 | +0.0616 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7413 | +0.1018 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8802 | +0.0333 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 102
- **Date**: 2026-02-04 21:40
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8192 (baseline: 0.7659, delta: +0.0533)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8032 | +0.0611 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9105 | +0.0875 | IMPROVED |
| Consolidation | 0.7264 | 0.8029 | +0.0765 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8887 | +0.0520 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8727 | +0.0419 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9089 | +0.0576 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7950 | +0.0724 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7797 | -0.0764 | REGRESSED |
| Infiltration | 0.6503 | 0.6921 | +0.0418 | IMPROVED |
| Mass | 0.7835 | 0.8602 | +0.0767 | IMPROVED |
| Nodule | 0.6959 | 0.7509 | +0.0550 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7819 | +0.0650 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7411 | +0.1016 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8815 | +0.0346 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 103
- **Date**: 2026-02-04 21:45
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8201 (baseline: 0.7659, delta: +0.0542)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8029 | +0.0608 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9123 | +0.0893 | IMPROVED |
| Consolidation | 0.7264 | 0.8038 | +0.0774 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8901 | +0.0534 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8735 | +0.0427 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9095 | +0.0582 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7938 | +0.0712 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7823 | -0.0738 | REGRESSED |
| Infiltration | 0.6503 | 0.7000 | +0.0497 | IMPROVED |
| Mass | 0.7835 | 0.8596 | +0.0761 | IMPROVED |
| Nodule | 0.6959 | 0.7513 | +0.0554 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7819 | +0.0650 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7379 | +0.0984 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8819 | +0.0350 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 104
- **Date**: 2026-02-04 21:50
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8179 (baseline: 0.7659, delta: +0.0520)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8015 | +0.0594 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9091 | +0.0861 | IMPROVED |
| Consolidation | 0.7264 | 0.8036 | +0.0772 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8898 | +0.0531 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8736 | +0.0428 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9088 | +0.0575 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7955 | +0.0729 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7612 | -0.0949 | REGRESSED |
| Infiltration | 0.6503 | 0.6987 | +0.0484 | IMPROVED |
| Mass | 0.7835 | 0.8566 | +0.0731 | IMPROVED |
| Nodule | 0.6959 | 0.7503 | +0.0544 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7807 | +0.0638 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7402 | +0.1007 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8807 | +0.0338 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 105
- **Date**: 2026-02-04 21:56
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8190 (baseline: 0.7659, delta: +0.0531)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8026 | +0.0605 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9113 | +0.0883 | IMPROVED |
| Consolidation | 0.7264 | 0.8017 | +0.0753 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8890 | +0.0523 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8739 | +0.0431 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9082 | +0.0569 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7949 | +0.0723 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7765 | -0.0796 | REGRESSED |
| Infiltration | 0.6503 | 0.6975 | +0.0472 | IMPROVED |
| Mass | 0.7835 | 0.8589 | +0.0754 | IMPROVED |
| Nodule | 0.6959 | 0.7502 | +0.0543 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7825 | +0.0656 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7367 | +0.0972 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8815 | +0.0346 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 106
- **Date**: 2026-02-04 22:01
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8177 (baseline: 0.7659, delta: +0.0518)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7993 | +0.0572 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9124 | +0.0894 | IMPROVED |
| Consolidation | 0.7264 | 0.8017 | +0.0753 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8903 | +0.0536 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8728 | +0.0420 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9081 | +0.0568 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7937 | +0.0711 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7615 | -0.0946 | REGRESSED |
| Infiltration | 0.6503 | 0.7014 | +0.0511 | IMPROVED |
| Mass | 0.7835 | 0.8571 | +0.0736 | IMPROVED |
| Nodule | 0.6959 | 0.7504 | +0.0545 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7784 | +0.0615 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7389 | +0.0994 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8816 | +0.0347 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 107
- **Date**: 2026-02-04 22:07
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8191 (baseline: 0.7659, delta: +0.0532)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8022 | +0.0601 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9101 | +0.0871 | IMPROVED |
| Consolidation | 0.7264 | 0.8043 | +0.0779 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8870 | +0.0503 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8732 | +0.0424 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9092 | +0.0579 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7985 | +0.0759 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7698 | -0.0863 | REGRESSED |
| Infiltration | 0.6503 | 0.7022 | +0.0519 | IMPROVED |
| Mass | 0.7835 | 0.8577 | +0.0742 | IMPROVED |
| Nodule | 0.6959 | 0.7511 | +0.0552 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7808 | +0.0639 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7418 | +0.1023 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8801 | +0.0332 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 108
- **Date**: 2026-02-04 22:13
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8200 (baseline: 0.7659, delta: +0.0541)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8025 | +0.0604 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9132 | +0.0902 | IMPROVED |
| Consolidation | 0.7264 | 0.8031 | +0.0767 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8900 | +0.0533 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8736 | +0.0428 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9098 | +0.0585 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7957 | +0.0731 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7812 | -0.0749 | REGRESSED |
| Infiltration | 0.6503 | 0.7013 | +0.0510 | IMPROVED |
| Mass | 0.7835 | 0.8602 | +0.0767 | IMPROVED |
| Nodule | 0.6959 | 0.7507 | +0.0548 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7809 | +0.0640 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7358 | +0.0963 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8823 | +0.0354 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 111
- **Date**: 2026-02-04 23:11
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8146 (baseline: 0.7659, delta: +0.0487)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7973 | +0.0552 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9063 | +0.0833 | IMPROVED |
| Consolidation | 0.7264 | 0.8029 | +0.0765 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8875 | +0.0508 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8681 | +0.0373 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9037 | +0.0524 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7968 | +0.0742 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7527 | -0.1034 | REGRESSED |
| Infiltration | 0.6503 | 0.6851 | +0.0348 | IMPROVED |
| Mass | 0.7835 | 0.8576 | +0.0741 | IMPROVED |
| Nodule | 0.6959 | 0.7469 | +0.0510 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7805 | +0.0636 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7413 | +0.1018 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8777 | +0.0308 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP
- **Status**: CONTINUE

---

## Iteration 117
- **Date**: 2026-02-05 16:01
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8175 (baseline: 0.7659, delta: +0.0516)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8022 | +0.0601 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9084 | +0.0854 | IMPROVED |
| Consolidation | 0.7264 | 0.8022 | +0.0758 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8846 | +0.0479 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8740 | +0.0432 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9072 | +0.0559 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7936 | +0.0710 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7507 | -0.1054 | REGRESSED |
| Infiltration | 0.6503 | 0.7000 | +0.0497 | IMPROVED |
| Mass | 0.7835 | 0.8602 | +0.0767 | IMPROVED |
| Nodule | 0.6959 | 0.7501 | +0.0542 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7825 | +0.0656 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7469 | +0.1074 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8820 | +0.0351 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: PROBE_AGAIN
- **Status**: CONTINUE

---

## Iteration 119
- **Date**: 2026-02-05 17:09
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8181 (baseline: 0.7659, delta: +0.0522)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8009 | +0.0588 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9096 | +0.0866 | IMPROVED |
| Consolidation | 0.7264 | 0.8024 | +0.0760 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8857 | +0.0490 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8741 | +0.0433 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9092 | +0.0579 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7938 | +0.0712 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7592 | -0.0969 | REGRESSED |
| Infiltration | 0.6503 | 0.6974 | +0.0471 | IMPROVED |
| Mass | 0.7835 | 0.8606 | +0.0771 | IMPROVED |
| Nodule | 0.6959 | 0.7494 | +0.0535 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7843 | +0.0674 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7465 | +0.1070 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8808 | +0.0339 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: PROBE_AGAIN
- **Status**: CONTINUE

---

## Iteration 123
- **Date**: 2026-02-07 00:57
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7556 (baseline: 0.7659, delta: -0.0103)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7257 | -0.0164 | REGRESSED |
| Cardiomegaly (Stable) | 0.8230 | 0.8734 | +0.0504 | IMPROVED |
| Consolidation | 0.7264 | 0.7080 | -0.0184 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8456 | +0.0089 | stable |
| Effusion (Stable) | 0.8308 | 0.8277 | -0.0031 | stable |
| Emphysema (Stable) | 0.8513 | 0.8555 | +0.0042 | stable |
| Fibrosis (Target) | 0.7226 | 0.7154 | -0.0072 | stable |
| Hernia (Stable) | 0.8561 | 0.6737 | -0.1824 | REGRESSED |
| Infiltration | 0.6503 | 0.6622 | +0.0119 | IMPROVED |
| Mass | 0.7835 | 0.7903 | +0.0068 | stable |
| Nodule | 0.6959 | 0.6755 | -0.0204 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.7117 | -0.0052 | stable |
| Pneumonia (Target) | 0.6395 | 0.6700 | +0.0305 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8434 | -0.0035 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 124
- **Date**: 2026-02-07 03:07
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7830 (baseline: 0.7659, delta: +0.0171)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7821 | +0.0400 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9102 | +0.0872 | IMPROVED |
| Consolidation | 0.7264 | 0.7893 | +0.0629 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8857 | +0.0490 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8640 | +0.0332 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.8745 | +0.0232 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7600 | +0.0374 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.5377 | -0.3184 | REGRESSED |
| Infiltration | 0.6503 | 0.7009 | +0.0506 | IMPROVED |
| Mass | 0.7835 | 0.8205 | +0.0370 | IMPROVED |
| Nodule | 0.6959 | 0.7044 | +0.0085 | stable |
| Pleural_Thickening | 0.7169 | 0.7589 | +0.0420 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7156 | +0.0761 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8588 | +0.0119 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 1
- **Date**: 2026-02-07 10:19
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8172 (baseline: 0.7659, delta: +0.0513)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8026 | +0.0605 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9106 | +0.0876 | IMPROVED |
| Consolidation | 0.7264 | 0.8025 | +0.0761 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8862 | +0.0495 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8748 | +0.0440 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9088 | +0.0575 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7918 | +0.0692 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7351 | -0.1210 | REGRESSED |
| Infiltration | 0.6503 | 0.7008 | +0.0505 | IMPROVED |
| Mass | 0.7835 | 0.8603 | +0.0768 | IMPROVED |
| Nodule | 0.6959 | 0.7510 | +0.0551 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7846 | +0.0677 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7475 | +0.1080 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8839 | +0.0370 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 2
- **Date**: 2026-02-07 11:07
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8174 (baseline: 0.7659, delta: +0.0515)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8024 | +0.0603 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9104 | +0.0874 | IMPROVED |
| Consolidation | 0.7264 | 0.8026 | +0.0762 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8876 | +0.0509 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8748 | +0.0440 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9087 | +0.0574 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7935 | +0.0709 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7362 | -0.1199 | REGRESSED |
| Infiltration | 0.6503 | 0.7011 | +0.0508 | IMPROVED |
| Mass | 0.7835 | 0.8595 | +0.0760 | IMPROVED |
| Nodule | 0.6959 | 0.7510 | +0.0551 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7842 | +0.0673 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7470 | +0.1075 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8841 | +0.0372 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 3
- **Date**: 2026-02-07 11:55
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8179 (baseline: 0.7659, delta: +0.0520)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8023 | +0.0602 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9105 | +0.0875 | IMPROVED |
| Consolidation | 0.7264 | 0.8029 | +0.0765 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8885 | +0.0518 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8752 | +0.0444 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9095 | +0.0582 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7934 | +0.0708 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7411 | -0.1150 | REGRESSED |
| Infiltration | 0.6503 | 0.7015 | +0.0512 | IMPROVED |
| Mass | 0.7835 | 0.8599 | +0.0764 | IMPROVED |
| Nodule | 0.6959 | 0.7506 | +0.0547 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7836 | +0.0667 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7469 | +0.1074 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8842 | +0.0373 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 4
- **Date**: 2026-02-07 12:42
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8176 (baseline: 0.7659, delta: +0.0517)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8021 | +0.0600 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9115 | +0.0885 | IMPROVED |
| Consolidation | 0.7264 | 0.8020 | +0.0756 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8881 | +0.0514 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8750 | +0.0442 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9092 | +0.0579 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7929 | +0.0703 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7356 | -0.1205 | REGRESSED |
| Infiltration | 0.6503 | 0.7009 | +0.0506 | IMPROVED |
| Mass | 0.7835 | 0.8598 | +0.0763 | IMPROVED |
| Nodule | 0.6959 | 0.7515 | +0.0556 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7841 | +0.0672 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7491 | +0.1096 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8842 | +0.0373 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 5
- **Date**: 2026-02-07 13:29
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.8170 (baseline: 0.7659, delta: +0.0511)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.8021 | +0.0600 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.9109 | +0.0879 | IMPROVED |
| Consolidation | 0.7264 | 0.8021 | +0.0757 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8877 | +0.0510 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8750 | +0.0442 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.9088 | +0.0575 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7913 | +0.0687 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7322 | -0.1239 | REGRESSED |
| Infiltration | 0.6503 | 0.7010 | +0.0507 | IMPROVED |
| Mass | 0.7835 | 0.8600 | +0.0765 | IMPROVED |
| Nodule | 0.6959 | 0.7508 | +0.0549 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7832 | +0.0663 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.7486 | +0.1091 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8839 | +0.0370 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---

## Iteration 132
- **Date**: 2026-02-07 23:37
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7617 (baseline: 0.7659, delta: -0.0042)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7505 | +0.0084 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8121 | -0.0109 | REGRESSED |
| Consolidation | 0.7264 | 0.7106 | -0.0158 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8347 | -0.0020 | stable |
| Effusion (Stable) | 0.8308 | 0.8378 | +0.0070 | stable |
| Emphysema (Stable) | 0.8513 | 0.8470 | -0.0043 | stable |
| Fibrosis (Target) | 0.7226 | 0.7550 | +0.0324 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.7249 | -0.1312 | REGRESSED |
| Infiltration | 0.6503 | 0.6577 | +0.0074 | stable |
| Mass | 0.7835 | 0.8030 | +0.0195 | IMPROVED |
| Nodule | 0.6959 | 0.7029 | +0.0070 | stable |
| Pleural_Thickening | 0.7169 | 0.7096 | -0.0073 | stable |
| Pneumonia (Target) | 0.6395 | 0.6700 | +0.0305 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8478 | +0.0009 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 133
- **Date**: 2026-02-08 01:37
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7543 (baseline: 0.7659, delta: -0.0116)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7442 | +0.0021 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8427 | +0.0197 | IMPROVED |
| Consolidation | 0.7264 | 0.7214 | -0.0050 | stable |
| Edema (Target) | 0.8367 | 0.8362 | -0.0005 | stable |
| Effusion (Stable) | 0.8308 | 0.8381 | +0.0073 | stable |
| Emphysema (Stable) | 0.8513 | 0.8541 | +0.0028 | stable |
| Fibrosis (Target) | 0.7226 | 0.7356 | +0.0130 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.6705 | -0.1856 | REGRESSED |
| Infiltration | 0.6503 | 0.6514 | +0.0011 | stable |
| Mass | 0.7835 | 0.7991 | +0.0156 | IMPROVED |
| Nodule | 0.6959 | 0.6810 | -0.0149 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.6966 | -0.0203 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6462 | +0.0067 | stable |
| Pneumothorax (Stable) | 0.8469 | 0.8430 | -0.0039 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 134
- **Date**: 2026-02-08 03:37
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7248 (baseline: 0.7659, delta: -0.0411)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7373 | -0.0048 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8076 | -0.0154 | REGRESSED |
| Consolidation | 0.7264 | 0.7129 | -0.0135 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8159 | -0.0208 | REGRESSED |
| Effusion (Stable) | 0.8308 | 0.8117 | -0.0191 | REGRESSED |
| Emphysema (Stable) | 0.8513 | 0.8222 | -0.0291 | REGRESSED |
| Fibrosis (Target) | 0.7226 | 0.6906 | -0.0320 | REGRESSED |
| Hernia (Stable) | 0.8561 | 0.5810 | -0.2751 | REGRESSED |
| Infiltration | 0.6503 | 0.6355 | -0.0148 | REGRESSED |
| Mass | 0.7835 | 0.7640 | -0.0195 | REGRESSED |
| Nodule | 0.6959 | 0.6640 | -0.0319 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.6773 | -0.0396 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6588 | +0.0193 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.7682 | -0.0787 | REGRESSED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 135
- **Date**: 2026-02-08 05:38
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7097 (baseline: 0.7659, delta: -0.0562)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7152 | -0.0269 | REGRESSED |
| Cardiomegaly (Stable) | 0.8230 | 0.8004 | -0.0226 | REGRESSED |
| Consolidation | 0.7264 | 0.6729 | -0.0535 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8082 | -0.0285 | REGRESSED |
| Effusion (Stable) | 0.8308 | 0.7968 | -0.0340 | REGRESSED |
| Emphysema (Stable) | 0.8513 | 0.7791 | -0.0722 | REGRESSED |
| Fibrosis (Target) | 0.7226 | 0.6341 | -0.0885 | REGRESSED |
| Hernia (Stable) | 0.8561 | 0.6288 | -0.2273 | REGRESSED |
| Infiltration | 0.6503 | 0.6398 | -0.0105 | REGRESSED |
| Mass | 0.7835 | 0.7633 | -0.0202 | REGRESSED |
| Nodule | 0.6959 | 0.6290 | -0.0669 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.6668 | -0.0501 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6131 | -0.0264 | REGRESSED |
| Pneumothorax (Stable) | 0.8469 | 0.7889 | -0.0580 | REGRESSED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 136
- **Date**: 2026-02-08 07:39
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7436 (baseline: 0.7659, delta: -0.0223)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7538 | +0.0117 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8326 | +0.0096 | stable |
| Consolidation | 0.7264 | 0.7054 | -0.0210 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8197 | -0.0170 | REGRESSED |
| Effusion (Stable) | 0.8308 | 0.8327 | +0.0019 | stable |
| Emphysema (Stable) | 0.8513 | 0.8395 | -0.0118 | REGRESSED |
| Fibrosis (Target) | 0.7226 | 0.7149 | -0.0077 | stable |
| Hernia (Stable) | 0.8561 | 0.6587 | -0.1974 | REGRESSED |
| Infiltration | 0.6503 | 0.6445 | -0.0058 | stable |
| Mass | 0.7835 | 0.7634 | -0.0201 | REGRESSED |
| Nodule | 0.6959 | 0.6795 | -0.0164 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.7053 | -0.0116 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6263 | -0.0132 | REGRESSED |
| Pneumothorax (Stable) | 0.8469 | 0.8339 | -0.0130 | REGRESSED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 137
- **Date**: 2026-02-08 10:01
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7501 (baseline: 0.7659, delta: -0.0158)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7405 | -0.0016 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8590 | +0.0360 | IMPROVED |
| Consolidation | 0.7264 | 0.7153 | -0.0111 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8455 | +0.0088 | stable |
| Effusion (Stable) | 0.8308 | 0.8313 | +0.0005 | stable |
| Emphysema (Stable) | 0.8513 | 0.8586 | +0.0073 | stable |
| Fibrosis (Target) | 0.7226 | 0.7152 | -0.0074 | stable |
| Hernia (Stable) | 0.8561 | 0.6830 | -0.1731 | REGRESSED |
| Infiltration | 0.6503 | 0.6537 | +0.0034 | stable |
| Mass | 0.7835 | 0.7823 | -0.0012 | stable |
| Nodule | 0.6959 | 0.6691 | -0.0268 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.7054 | -0.0115 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6141 | -0.0254 | REGRESSED |
| Pneumothorax (Stable) | 0.8469 | 0.8281 | -0.0188 | REGRESSED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 138
- **Date**: 2026-02-08 12:04
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7647 (baseline: 0.7659, delta: -0.0012)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7624 | +0.0203 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8676 | +0.0446 | IMPROVED |
| Consolidation | 0.7264 | 0.7689 | +0.0425 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8643 | +0.0276 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8533 | +0.0225 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.8739 | +0.0226 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7316 | +0.0090 | stable |
| Hernia (Stable) | 0.8561 | 0.5200 | -0.3361 | REGRESSED |
| Infiltration | 0.6503 | 0.6718 | +0.0215 | IMPROVED |
| Mass | 0.7835 | 0.8134 | +0.0299 | IMPROVED |
| Nodule | 0.6959 | 0.6887 | -0.0072 | stable |
| Pleural_Thickening | 0.7169 | 0.7533 | +0.0364 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.6913 | +0.0518 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8449 | -0.0020 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 139
- **Date**: 2026-02-08 14:07
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7489 (baseline: 0.7659, delta: -0.0170)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7330 | -0.0091 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8337 | +0.0107 | IMPROVED |
| Consolidation | 0.7264 | 0.6996 | -0.0268 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8476 | +0.0109 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8305 | -0.0003 | stable |
| Emphysema (Stable) | 0.8513 | 0.8080 | -0.0433 | REGRESSED |
| Fibrosis (Target) | 0.7226 | 0.7062 | -0.0164 | REGRESSED |
| Hernia (Stable) | 0.8561 | 0.6460 | -0.2101 | REGRESSED |
| Infiltration | 0.6503 | 0.6380 | -0.0123 | REGRESSED |
| Mass | 0.7835 | 0.7827 | -0.0008 | stable |
| Nodule | 0.6959 | 0.7087 | +0.0128 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7251 | +0.0082 | stable |
| Pneumonia (Target) | 0.6395 | 0.6729 | +0.0334 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8528 | +0.0059 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 140
- **Date**: 2026-02-08 16:11
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7550 (baseline: 0.7659, delta: -0.0109)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7394 | -0.0027 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8208 | -0.0022 | stable |
| Consolidation | 0.7264 | 0.7386 | +0.0122 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8374 | +0.0007 | stable |
| Effusion (Stable) | 0.8308 | 0.8249 | -0.0059 | stable |
| Emphysema (Stable) | 0.8513 | 0.8663 | +0.0150 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7275 | +0.0049 | stable |
| Hernia (Stable) | 0.8561 | 0.6738 | -0.1823 | REGRESSED |
| Infiltration | 0.6503 | 0.6487 | -0.0016 | stable |
| Mass | 0.7835 | 0.7956 | +0.0121 | IMPROVED |
| Nodule | 0.6959 | 0.6877 | -0.0082 | stable |
| Pleural_Thickening | 0.7169 | 0.7084 | -0.0085 | stable |
| Pneumonia (Target) | 0.6395 | 0.6584 | +0.0189 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8427 | -0.0042 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 142
- **Date**: 2026-02-08 21:53
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7442 (baseline: 0.7659, delta: -0.0217)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7322 | -0.0099 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8566 | +0.0336 | IMPROVED |
| Consolidation | 0.7264 | 0.6967 | -0.0297 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8592 | +0.0225 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8194 | -0.0114 | REGRESSED |
| Emphysema (Stable) | 0.8513 | 0.8310 | -0.0203 | REGRESSED |
| Fibrosis (Target) | 0.7226 | 0.7243 | +0.0017 | stable |
| Hernia (Stable) | 0.8561 | 0.6154 | -0.2407 | REGRESSED |
| Infiltration | 0.6503 | 0.6504 | +0.0001 | stable |
| Mass | 0.7835 | 0.7778 | -0.0057 | stable |
| Nodule | 0.6959 | 0.6809 | -0.0150 | REGRESSED |
| Pleural_Thickening | 0.7169 | 0.6859 | -0.0310 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6481 | +0.0086 | stable |
| Pneumothorax (Stable) | 0.8469 | 0.8411 | -0.0058 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 143
- **Date**: 2026-02-08 23:51
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7628 (baseline: 0.7659, delta: -0.0031)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7623 | +0.0202 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8561 | +0.0331 | IMPROVED |
| Consolidation | 0.7264 | 0.7109 | -0.0155 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8516 | +0.0149 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8540 | +0.0232 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.8753 | +0.0240 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7394 | +0.0168 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.6176 | -0.2385 | REGRESSED |
| Infiltration | 0.6503 | 0.6730 | +0.0227 | IMPROVED |
| Mass | 0.7835 | 0.8059 | +0.0224 | IMPROVED |
| Nodule | 0.6959 | 0.7016 | +0.0057 | stable |
| Pleural_Thickening | 0.7169 | 0.7238 | +0.0069 | stable |
| Pneumonia (Target) | 0.6395 | 0.6523 | +0.0128 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8550 | +0.0081 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 144
- **Date**: 2026-02-09 01:51
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7534 (baseline: 0.7659, delta: -0.0125)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7370 | -0.0051 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8046 | -0.0184 | REGRESSED |
| Consolidation | 0.7264 | 0.7064 | -0.0200 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8417 | +0.0050 | stable |
| Effusion (Stable) | 0.8308 | 0.8396 | +0.0088 | stable |
| Emphysema (Stable) | 0.8513 | 0.8466 | -0.0047 | stable |
| Fibrosis (Target) | 0.7226 | 0.7155 | -0.0071 | stable |
| Hernia (Stable) | 0.8561 | 0.6834 | -0.1727 | REGRESSED |
| Infiltration | 0.6503 | 0.6502 | -0.0001 | stable |
| Mass | 0.7835 | 0.7915 | +0.0080 | stable |
| Nodule | 0.6959 | 0.7040 | +0.0081 | stable |
| Pleural_Thickening | 0.7169 | 0.7080 | -0.0089 | stable |
| Pneumonia (Target) | 0.6395 | 0.6770 | +0.0375 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8424 | -0.0045 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 145
- **Date**: 2026-02-09 03:50
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7575 (baseline: 0.7659, delta: -0.0084)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7402 | -0.0019 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8524 | +0.0294 | IMPROVED |
| Consolidation | 0.7264 | 0.7387 | +0.0123 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8556 | +0.0189 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8403 | +0.0095 | stable |
| Emphysema (Stable) | 0.8513 | 0.8436 | -0.0077 | stable |
| Fibrosis (Target) | 0.7226 | 0.7361 | +0.0135 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.6276 | -0.2285 | REGRESSED |
| Infiltration | 0.6503 | 0.6394 | -0.0109 | REGRESSED |
| Mass | 0.7835 | 0.7948 | +0.0113 | IMPROVED |
| Nodule | 0.6959 | 0.6968 | +0.0009 | stable |
| Pleural_Thickening | 0.7169 | 0.7287 | +0.0118 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.6786 | +0.0391 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8324 | -0.0145 | REGRESSED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 146
- **Date**: 2026-02-09 05:49
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7563 (baseline: 0.7659, delta: -0.0096)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7542 | +0.0121 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8405 | +0.0175 | IMPROVED |
| Consolidation | 0.7264 | 0.7365 | +0.0101 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8257 | -0.0110 | REGRESSED |
| Effusion (Stable) | 0.8308 | 0.8380 | +0.0072 | stable |
| Emphysema (Stable) | 0.8513 | 0.8506 | -0.0007 | stable |
| Fibrosis (Target) | 0.7226 | 0.7099 | -0.0127 | REGRESSED |
| Hernia (Stable) | 0.8561 | 0.6567 | -0.1994 | REGRESSED |
| Infiltration | 0.6503 | 0.6353 | -0.0150 | REGRESSED |
| Mass | 0.7835 | 0.8112 | +0.0277 | IMPROVED |
| Nodule | 0.6959 | 0.7001 | +0.0042 | stable |
| Pleural_Thickening | 0.7169 | 0.7210 | +0.0041 | stable |
| Pneumonia (Target) | 0.6395 | 0.6669 | +0.0274 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8417 | -0.0052 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 147
- **Date**: 2026-02-09 07:48
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7555 (baseline: 0.7659, delta: -0.0104)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7426 | +0.0005 | stable |
| Cardiomegaly (Stable) | 0.8230 | 0.8449 | +0.0219 | IMPROVED |
| Consolidation | 0.7264 | 0.7001 | -0.0263 | REGRESSED |
| Edema (Target) | 0.8367 | 0.8210 | -0.0157 | REGRESSED |
| Effusion (Stable) | 0.8308 | 0.8423 | +0.0115 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.8626 | +0.0113 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7216 | -0.0010 | stable |
| Hernia (Stable) | 0.8561 | 0.6527 | -0.2034 | REGRESSED |
| Infiltration | 0.6503 | 0.6546 | +0.0043 | stable |
| Mass | 0.7835 | 0.8015 | +0.0180 | IMPROVED |
| Nodule | 0.6959 | 0.7090 | +0.0131 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7146 | -0.0023 | stable |
| Pneumonia (Target) | 0.6395 | 0.6633 | +0.0238 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8466 | -0.0003 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 148
- **Date**: 2026-02-09 10:16
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7546 (baseline: 0.7659, delta: -0.0113)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7569 | +0.0148 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8524 | +0.0294 | IMPROVED |
| Consolidation | 0.7264 | 0.7291 | +0.0027 | stable |
| Edema (Target) | 0.8367 | 0.8406 | +0.0039 | stable |
| Effusion (Stable) | 0.8308 | 0.8416 | +0.0108 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.8635 | +0.0122 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7276 | +0.0050 | stable |
| Hernia (Stable) | 0.8561 | 0.5908 | -0.2653 | REGRESSED |
| Infiltration | 0.6503 | 0.6634 | +0.0131 | IMPROVED |
| Mass | 0.7835 | 0.8011 | +0.0176 | IMPROVED |
| Nodule | 0.6959 | 0.6949 | -0.0010 | stable |
| Pleural_Thickening | 0.7169 | 0.7014 | -0.0155 | REGRESSED |
| Pneumonia (Target) | 0.6395 | 0.6794 | +0.0399 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8216 | -0.0253 | REGRESSED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 149
- **Date**: 2026-02-09 12:32
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7766 (baseline: 0.7659, delta: +0.0107)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7740 | +0.0319 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8600 | +0.0370 | IMPROVED |
| Consolidation | 0.7264 | 0.7589 | +0.0325 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8592 | +0.0225 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8528 | +0.0220 | IMPROVED |
| Emphysema (Stable) | 0.8513 | 0.8726 | +0.0213 | IMPROVED |
| Fibrosis (Target) | 0.7226 | 0.7368 | +0.0142 | IMPROVED |
| Hernia (Stable) | 0.8561 | 0.6485 | -0.2076 | REGRESSED |
| Infiltration | 0.6503 | 0.6833 | +0.0330 | IMPROVED |
| Mass | 0.7835 | 0.8264 | +0.0429 | IMPROVED |
| Nodule | 0.6959 | 0.7121 | +0.0162 | IMPROVED |
| Pleural_Thickening | 0.7169 | 0.7283 | +0.0114 | IMPROVED |
| Pneumonia (Target) | 0.6395 | 0.6994 | +0.0599 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8598 | +0.0129 | IMPROVED |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: STOP_PHASE_5_AND_FREEZE
- **Status**: STOP

---

## Iteration 150
- **Date**: 2026-02-09 15:30
- **Parent iteration**: 84
- **Target diseases**: Pneumonia, Fibrosis, Edema
- **Macro AUC**: 0.7610 (baseline: 0.7659, delta: -0.0049)

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
| Atelectasis | 0.7421 | 0.7549 | +0.0128 | IMPROVED |
| Cardiomegaly (Stable) | 0.8230 | 0.8608 | +0.0378 | IMPROVED |
| Consolidation | 0.7264 | 0.7425 | +0.0161 | IMPROVED |
| Edema (Target) | 0.8367 | 0.8492 | +0.0125 | IMPROVED |
| Effusion (Stable) | 0.8308 | 0.8255 | -0.0053 | stable |
| Emphysema (Stable) | 0.8513 | 0.8471 | -0.0042 | stable |
| Fibrosis (Target) | 0.7226 | 0.7172 | -0.0054 | stable |
| Hernia (Stable) | 0.8561 | 0.6752 | -0.1809 | REGRESSED |
| Infiltration | 0.6503 | 0.6402 | -0.0101 | REGRESSED |
| Mass | 0.7835 | 0.7996 | +0.0161 | IMPROVED |
| Nodule | 0.6959 | 0.7022 | +0.0063 | stable |
| Pleural_Thickening | 0.7169 | 0.7156 | -0.0013 | stable |
| Pneumonia (Target) | 0.6395 | 0.6748 | +0.0353 | IMPROVED |
| Pneumothorax (Stable) | 0.8469 | 0.8491 | +0.0022 | stable |

- **Stable diseases regression check**: FAILED
- **Decision from AI advisory**: CONTINUE_TARGETED_AUC_IMPROVEMENT
- **Status**: CONTINUE

---
