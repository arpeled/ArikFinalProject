# Patient Metadata Usage Investigation Report

**Investigation Date**: 2026-02-08
**Scope**: Full codebase analysis for patient metadata usage
**Objective**: Determine whether and how patient clinical information from `Data_Entry_2017_v2020.csv` was used in the experimental pipeline

---

## 1. Files That Load or Reference `Data_Entry_2017_v2020.csv`

### Direct References Found

| File | Purpose | How Data Is Used |
|------|---------|------------------|
| `add_additional_info.py` | Data preparation script | **Merges** patient metadata into training data |
| `thesis_dataset_analysis.py` | Dataset analysis | References `Data_Entry_2017.csv` for statistics |
| `print_stats.py` | Statistics generation | Loads for data exploration |
| `create_files_for_test_train.py` | Train/test split | Uses for splitting data |
| `data_analize.ipynb` | Exploratory analysis | Data exploration notebook |
| `chest-x-ray-classifier-densenet121_modified.ipynb` | Development notebook | Early experimentation |

### Key Evidence: `add_additional_info.py`

```python
# Merges patient metadata into training data
additional_df = pd.read_csv('./ChestX-ray14/Data_Entry_2017_v2020.csv')
merged_df = pd.merge(existing_df, additional_df[[
    'Image Index', 'Follow-up #', 'Patient Age', 'Patient Gender', 'View Position'
]], on='Image Index', how='left')
```

This script was used to create the training CSV files with embedded patient metadata.

---

## 2. Actual Usage of Patient Metadata in Training Pipeline

### **Answer: YES - Patient metadata IS used in training**

### Evidence from `dataset.py`

The `ChestXRayDataset` class has a parameter `use_additional_features` that when enabled:

```python
# Lines 251-252: Define additional feature columns
self.additional_columns = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']

# Lines 312-320: Extract features in __getitem__
if self.use_additional_features:
    follow_up = torch.tensor(self.data.iloc[idx]["Follow-up #"], dtype=torch.float32)
    patient_age = torch.tensor(self.data.iloc[idx]["Patient Age"], dtype=torch.float32)
    patient_gender = 1.0 if self.data.iloc[idx]["Patient Gender"] == "M" else 0.0
    view_position = 1.0 if self.data.iloc[idx]["View Position"] == "PA" else 0.0
    additional_features = torch.tensor([follow_up, patient_age, patient_gender, view_position],
                                       dtype=torch.float32)
    return image, additional_features, labels
```

### Evidence from Model Architecture (`dataset.py`)

Both model classes (`ModifiedDenseNet` and `ModifiedDenseNetWithDropOut`) include dedicated layers for patient features:

```python
# Lines 336-338: Additional feature processing
if self.use_additional_features:
    self.additional_fc = nn.Linear(4, 128)  # Four features → 128 dimensions
    self.final_fc = nn.Linear(1024 + 128, num_classes)  # Concatenated features
```

### Evidence from Configuration Files

**Baseline configuration (`config_baseline.yaml`, line 20)**:
```yaml
model:
  use_additional_features: true
```

**All iteration configs (139 iterations)** inherit this setting:
```yaml
use_additional_features: true  # Present in all configs
```

**Reference configuration in `iteration_baselines.py` (line 191)**:
```python
"use_additional_features": True,  # ACTUAL: Was enabled in iter12
```

### Evidence from Training Pipeline (`config_based_pipeline.py`)

The training loop explicitly handles additional features:

```python
# Line 742: Get setting from config
use_additional_features = model_config['use_additional_features']

# Lines 1201-1204: Training loop unpacks 3 values when enabled
if use_additional_features:
    images, additional_features, labels = batch
    outputs = model(images, additional_features)
```

---

## 3. Actual Usage in Evaluation or Analysis

### **Answer: YES - Patient metadata IS used in evaluation**

### Evidence from `chest_xray_test_pipeline.py`

```python
# Line 38: HARDCODED to True
use_additional_features = True

# Lines 56-58: Test dataset uses additional features
test_dataset = ChestXRayDataset(
    dataset=None, csv_file=test_csv_file,
    use_additional_features=use_additional_features
)

# Lines 63-66: Model initialized with additional features
model = ModifiedDenseNetWithDropOut(
    num_classes=num_classes,
    use_additional_features=use_additional_features,
    head_config=head_config
)

# Lines 135-138: Inference uses additional features
if use_additional_features:
    images, additional_features, labels = batch
    outputs = model(images, additional_features)
```

### Training Data CSV Structure

The actual training data (`ChestX-ray14/train_data.csv`) contains patient metadata columns:

```
Image Index,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,Cardiomegaly,...
00029763_001.png,1,29763,32,M,PA,0,...
```

---

## 4. Summary Conclusion

### Was Patient Metadata Used?

**YES.** Patient metadata was integrated into the entire training and evaluation pipeline.

### At What Stage?

| Stage | Metadata Used? | Details |
|-------|----------------|---------|
| **Data Preparation** | Yes | `add_additional_info.py` merged metadata into CSVs |
| **Data Loading** | Yes | `ChestXRayDataset` extracts 4 features per sample |
| **Model Architecture** | Yes | Dedicated FC layer (4→128) + concatenation (1024+128) |
| **Training** | Yes | All 139 iterations used `use_additional_features: true` |
| **Validation** | Yes | Same pipeline as training |
| **Testing/Inference** | Yes | Hardcoded `use_additional_features = True` |
| **Thesis Analysis** | Yes | Analysis scripts read from metadata-enriched CSVs |

### Features Used

| Feature | Encoding | Description |
|---------|----------|-------------|
| Follow-up # | Numeric (as-is) | Sequential exam number for patient |
| Patient Age | Numeric (as-is) | Patient age in years |
| Patient Gender | Binary (M=1.0, F=0.0) | Patient sex |
| View Position | Binary (PA=1.0, other=0.0) | X-ray view direction |

### Why This Is Important for Interpreting Results

1. **The model is NOT purely image-based**:
   - Final feature vector is 1152-dimensional (1024 from DenseNet + 128 from patient features)
   - Patient metadata contributes to all predictions

2. **Potential confounding factors**:
   - Age and gender correlations with disease prevalence may influence predictions
   - View position (PA vs AP) affects image quality and disease visibility
   - Follow-up number may correlate with disease severity

3. **Reproducibility considerations**:
   - Any reproduction attempt must include patient metadata
   - Using image-only models would produce different results

4. **Clinical deployment implications**:
   - The model requires patient demographic information at inference time
   - Missing metadata would require either imputation or model modification

5. **Comparison with other work**:
   - Studies using image-only approaches are not directly comparable
   - The contribution of patient features vs. image features is not explicitly ablated in this work

### Classification of Usage

Based on the evidence:

- [ ] (a) Explored but abandoned
- [ ] (b) Used only for data analysis / statistics
- [ ] (c) Used only for dataset splitting or filtering
- [ ] (d) Not used at all
- [x] **(e) Fully integrated into training and inference pipelines**

---

## Appendix: File-by-File Evidence Summary

| File | Line(s) | Evidence |
|------|---------|----------|
| `add_additional_info.py` | 6-14 | Loads and merges `Data_Entry_2017_v2020.csv` |
| `dataset.py` | 220-221 | `use_additional_features` constructor parameter |
| `dataset.py` | 252 | Defines `additional_columns` list |
| `dataset.py` | 312-320 | Extracts 4 features in `__getitem__` |
| `dataset.py` | 336-338 | Model FC layer for additional features |
| `dataset.py` | 364-368 | `ModifiedDenseNetWithDropOut` uses features |
| `config_baseline.yaml` | 20 | `use_additional_features: true` |
| `iteration_baselines.py` | 191 | Reference config has features enabled |
| `config_based_pipeline.py` | 742 | Reads setting from config |
| `config_based_pipeline.py` | 1201-1204 | Training loop handles features |
| `chest_xray_test_pipeline.py` | 38 | Hardcoded `use_additional_features = True` |
| `chest_xray_test_pipeline.py` | 135-138 | Inference uses features |
| `ChestX-ray14/train_data.csv` | Header | Contains patient metadata columns |

---

*Report generated: 2026-02-08*
