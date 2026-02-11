# Testing Scripts Documentation

This document describes the testing/evaluation scripts for the Chest X-Ray classifier.

## Available Test Scripts

### 1. `chest-x-ray-classifier-densenet121-test_v05.py`
**Basic evaluation script with classification report**

#### Features:
- Loads trained model from timestamp
- Evaluates on test dataset
- Generates sklearn classification report
- Basic logging functionality

#### Usage:
```bash
python chest-x-ray-classifier-densenet121-test_v05.py
```

#### Configuration:
```python
run_timestamp = "20250402-230126"  # Model timestamp
use_additional_features = True     # Match training config
batch_size = 64
```

#### Output:
- Console classification report
- Log file: `run_log_{timestamp}.txt`

---

### 2. `chest-x-ray-classifier-densenet121-test_v05_updated.py`
**Comprehensive evaluation with detailed metrics**

#### Features:
- Multi-label evaluation metrics
- Per-class AUC-ROC scores
- Sensitivity/Specificity analysis
- Precision/Recall/F1-Score per condition
- Confusion matrix analysis
- Probability threshold evaluation

#### Usage:
```bash
python chest-x-ray-classifier-densenet121-test_v05_updated.py
```

#### Metrics Calculated:
- **AUC**: Area Under ROC Curve
- **Sensitivity**: True Positive Rate (Recall)
- **Specificity**: True Negative Rate
- **Accuracy**: Overall correctness
- **Precision**: Positive Predictive Value
- **Recall**: Sensitivity (same as above)
- **F1-Score**: Harmonic mean of Precision/Recall

#### Output Format:
```
Label                     AUC      Threshold Sensitivity  Specificity  Accuracy   Precision  Recall     F1-Score  
Cardiomegaly             0.8234   0.50      0.75         0.82         0.81       0.68       0.75       0.71      
Emphysema                0.9012   0.50      0.83         0.89         0.88       0.72       0.83       0.77      
...
```

---

## Test Data Requirements

### File Structure:
```
ChestX-ray14/
├── test_data.csv          # Test metadata
└── images224/             # Test images (224x224)
    ├── image1.png
    ├── image2.png
    └── ...
```

### CSV Format:
Required columns in `test_data.csv`:
- **Image Index**: Filename
- **Disease Labels**: 14 binary columns (0/1)
- **Additional Features** (if enabled):
  - Follow-up #
  - Patient Age  
  - Patient Gender
  - View Position

## Model Loading

Both scripts load models using the timestamp pattern:
```python
model_save_path = f'run_info_model_{run_timestamp}.pth'
```

Ensure the model file exists before running tests.

## Configuration Matching

**Critical**: Test configuration must match training:
```python
use_additional_features = True  # Must match training
num_classes = 14               # Fixed for ChestX-ray14
```

## Device Support

Scripts automatically detect and use:
- CUDA GPUs (if available)
- Apple Metal Performance Shaders (MPS)
- CPU fallback

## Troubleshooting

### Common Issues:

1. **Model Not Found**
   ```
   FileNotFoundError: run_info_model_TIMESTAMP.pth
   ```
   - Check timestamp matches trained model
   - Verify model file exists in project directory

2. **Feature Mismatch**
   ```
   RuntimeError: size mismatch
   ```
   - Ensure `use_additional_features` matches training
   - Check CSV has required additional feature columns

3. **Missing Images**
   ```
   Warning: File not found: ./ChestX-ray14/images224/image.png
   ```
   - Verify image directory path
   - Check image files exist and are accessible

### Performance Tips:

- Use GPU for faster evaluation
- Increase `batch_size` if memory allows
- Ensure images are pre-resized to 224x224

## Output Interpretation

### AUC Scores:
- **0.9-1.0**: Excellent performance
- **0.8-0.9**: Good performance  
- **0.7-0.8**: Fair performance
- **0.6-0.7**: Poor performance
- **0.5-0.6**: Very poor performance

### Sensitivity vs Specificity:
- **High Sensitivity**: Good at detecting disease (fewer false negatives)
- **High Specificity**: Good at ruling out disease (fewer false positives)
- **Balance**: Depends on clinical use case

### F1-Score:
- Balances Precision and Recall
- Useful for imbalanced classes
- Range: 0.0 (worst) to 1.0 (perfect)

## Logging

Both scripts generate detailed logs:
- Evaluation parameters
- Per-class metrics
- Timestamp and configuration
- Error messages and warnings

Log files: `run_log_{timestamp}.txt`