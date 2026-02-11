# Graph Organization Summary

## Overview
The graph generator creates **19 graphs** organized by metric and disease prevalence groups.

## Disease Groups (by Prevalence)

### Group 1: Common Diseases (>8% prevalence)
📊 **4 diseases** - Most frequent in dataset
- **Infiltration** (24.51%)
- **Effusion** (16.41%)
- **Atelectasis** (14.24%)
- **Nodule** (7.80%)

### Group 2: Moderate Diseases (3-8% prevalence)
📊 **6 diseases** - Mid-range frequency
- **Mass** (7.12%)
- **Pneumothorax** (6.53%)
- **Consolidation** (5.75%)
- **Pleural_Thickening** (4.17%)
- **Cardiomegaly** (3.42%)
- **Emphysema** (3.10%)

### Group 3: Rare Diseases (<3% prevalence)
📊 **4 diseases** - Least frequent (minority classes)
- **Edema** (2.84%)
- **Fibrosis** (2.08%)
- **Pneumonia** (1.76%)
- **Hernia** (0.28%)

---

## Generated Graph Files (19 total)

### AUC Metric (3 graphs)
```
✓ AUC_Common_8pct_progression.png          → Infiltration, Effusion, Atelectasis, Nodule
✓ AUC_Moderate_3-8pct_progression.png      → Mass, Pneumothorax, Consolidation, Pleural_Thickening, Cardiomegaly, Emphysema
✓ AUC_Rare_3pct_progression.png            → Edema, Fibrosis, Pneumonia, Hernia
```

### Specificity Metric (3 graphs)
```
✓ Specificity_Common_8pct_progression.png
✓ Specificity_Moderate_3-8pct_progression.png
✓ Specificity_Rare_3pct_progression.png
```

### Recall Metric (3 graphs)
```
✓ Recall_Common_8pct_progression.png
✓ Recall_Moderate_3-8pct_progression.png
✓ Recall_Rare_3pct_progression.png
```

### Precision Metric (3 graphs)
```
✓ Precision_Common_8pct_progression.png
✓ Precision_Moderate_3-8pct_progression.png
✓ Precision_Rare_3pct_progression.png
```

### Sensitivity Metric (3 graphs)
```
✓ Sensitivity_Common_8pct_progression.png
✓ Sensitivity_Moderate_3-8pct_progression.png
✓ Sensitivity_Rare_3pct_progression.png
```

### F1-Score Metric (3 graphs)
```
✓ F1_Score_Common_8pct_progression.png
✓ F1_Score_Moderate_3-8pct_progression.png
✓ F1_Score_Rare_3pct_progression.png
```

### Summary Graph (1 graph)
```
✓ Average_Metrics_progression.png          → All 6 metrics averaged across all 14 diseases
```

---

## Why This Organization?

### Problem with Original Approach
- **14 diseases on 1 graph** = Very cluttered, hard to read
- Lines overlap and obscure each other
- Difficult to track individual disease progression
- Legend takes up too much space

### Benefits of Grouped Approach
✅ **Better Readability**: 4-6 diseases per graph (vs 14)
✅ **Clear Visualization**: Less line overlap, easier tracking
✅ **Logical Grouping**: Diseases grouped by similar characteristics
✅ **Faster Analysis**: Focus on specific disease groups of interest
✅ **Clinical Relevance**: Compare diseases with similar prevalence

### Use Cases

**Analyzing Common Diseases**
- Open: `*_Common_8pct_progression.png` files
- Focus on high-prevalence diseases
- These contribute most to overall performance

**Debugging Rare Disease Performance**
- Open: `*_Rare_3pct_progression.png` files
- Track minority class handling
- Identify class imbalance issues

**Comparing Similar Prevalence**
- Compare within-group performance
- Identify if prevalence correlates with performance

**Overall Trends**
- Open: `Average_Metrics_progression.png`
- Get high-level view of all metrics

---

## File Naming Convention

```
{Metric}_{GroupName}_{Prevalence}_progression.png
```

**Examples:**
- `AUC_Common_8pct_progression.png`
  - Metric: AUC
  - Group: Common diseases
  - Prevalence: >8%

- `F1_Score_Rare_3pct_progression.png`
  - Metric: F1-Score
  - Group: Rare diseases
  - Prevalence: <3%

---

## Quick Access Guide

### Want to see AUC trends?
```bash
# All AUC graphs
ls graphs/AUC_*.png

# Just rare diseases AUC
open graphs/AUC_Rare_3pct_progression.png
```

### Want to see all metrics for rare diseases?
```bash
# All rare disease graphs
ls graphs/*Rare*.png

# Open them all
open graphs/*Rare*.png
```

### Want to see overall performance?
```bash
# Summary graph
open graphs/Average_Metrics_progression.png
```

### Want to see specific disease (e.g., Infiltration)?
**Infiltration is in Common group**, so check:
```bash
open graphs/*_Common_8pct_progression.png
```

---

## Graph Properties

### Size & Quality
- **Resolution**: 150 DPI (publication quality)
- **Dimensions**: 16×8 inches (landscape)
- **Format**: PNG with transparency support
- **File Size**: 60-170 KB per graph

### Visual Elements
- **Lines**: 2.5px width, 85% opacity
- **Markers**: 5px circles at each iteration
- **Grid**: Light gray, 30% opacity, dashed
- **Legend**: Auto-positioned, 95% opacity background
- **Baseline**: Gray dotted line at y=0.5

### Axis Details
- **X-axis**: Iteration numbers (automatic range)
- **Y-axis**: Metric values (fixed 0-1.05 range)
- **Title**: Format: "{Metric} Progression: {Group}"
- **Labels**: Bold, 14pt font

---

## Tracking Specific Diseases

| Disease | Group | Prevalence | Check These Graphs |
|---------|-------|------------|-------------------|
| Infiltration | Common | 24.51% | `*_Common_8pct_*.png` |
| Effusion | Common | 16.41% | `*_Common_8pct_*.png` |
| Atelectasis | Common | 14.24% | `*_Common_8pct_*.png` |
| Nodule | Common | 7.80% | `*_Common_8pct_*.png` |
| Mass | Moderate | 7.12% | `*_Moderate_3-8pct_*.png` |
| Pneumothorax | Moderate | 6.53% | `*_Moderate_3-8pct_*.png` |
| Consolidation | Moderate | 5.75% | `*_Moderate_3-8pct_*.png` |
| Pleural_Thickening | Moderate | 4.17% | `*_Moderate_3-8pct_*.png` |
| Cardiomegaly | Moderate | 3.42% | `*_Moderate_3-8pct_*.png` |
| Emphysema | Moderate | 3.10% | `*_Moderate_3-8pct_*.png` |
| Edema | Rare | 2.84% | `*_Rare_3pct_*.png` |
| Fibrosis | Rare | 2.08% | `*_Rare_3pct_*.png` |
| Pneumonia | Rare | 1.76% | `*_Rare_3pct_*.png` |
| Hernia | Rare | 0.28% | `*_Rare_3pct_*.png` |

---

Generated: 2026-01-08
Total Graphs: 19 (18 grouped + 1 summary)
