# Network Architecture Changelog

This document tracks all changes made to the chest X-ray classification network architecture, training parameters, and methodology.

## Version 1.0 - Initial Implementation
**Date**: December 2024  
**Status**: Current Implementation

### Base Architecture: Modified DenseNet-121

#### Core Network Structure
- **Base Model**: DenseNet-121 with ImageNet pre-trained weights
- **Input Size**: 224×224×3 RGB images
- **Feature Extractor**: DenseNet-121 backbone (1024 features)
- **Classifier**: Custom multi-label head

#### Model Variants

##### 1. ModifiedDenseNet (Basic)
```python
class ModifiedDenseNet(nn.Module):
    - Base: DenseNet-121 (ImageNet pretrained)
    - Classifier: nn.Linear(1024, 14)
    - Additional Features: Optional 4-feature input (128-dim FC layer)
    - Final Output: 14 classes (multi-label)
```

##### 2. ModifiedDenseNetWithDropOut (Current)
```python
class ModifiedDenseNetWithDropOut(nn.Module):
    - Base: DenseNet-121 (ImageNet pretrained)
    - Dropout: 0.3 after feature extraction
    - Classifier: nn.Linear(1024, 14) or nn.Linear(1152, 14) with additional features
    - Additional Features: Optional 4-feature input (128-dim FC layer)
    - Final Output: 14 classes (multi-label)
```

**Why Dropout was Added:**
- **Problem**: Overfitting on training data
- **Solution**: 30% dropout after DenseNet backbone
- **Benefit**: Better generalization and reduced overfitting

### Multi-Label Classification Setup

#### Target Classes (14 conditions):
1. Cardiomegaly
2. Emphysema  
3. Effusion
4. Hernia
5. Infiltration
6. Mass
7. Nodule
8. Atelectasis
9. Pneumothorax
10. Pleural_Thickening
11. Pneumonia
12. Fibrosis
13. Edema
14. Consolidation

#### Additional Features Integration
**Feature Vector (4 dimensions):**
- Follow-up # (continuous)
- Patient Age (continuous)
- Patient Gender (binary: M=1, F=0)
- View Position (binary: PA=1, AP=0)

**Architecture Flow:**
```
Image (224×224×3) → DenseNet-121 → Features (1024)
                                        ↓
Additional Features (4) → FC(128) → ReLU → Concat → FC(1152, 14)
```

### Training Configuration

#### Data Preprocessing
- **Image Normalization**: ImageNet statistics `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`
- **Resize**: 224×224 pixels
- **Data Augmentation**: 
  - Base: Resize + Normalize
  - Rare Classes: RandomRotation(10°) + RandomHorizontalFlip

#### Class Imbalance Handling

##### Weighted Loss Function
```python
# Class weights (calculated from training data frequency)
class_weights = [
    2.456, 2.710, 0.512, 30.0, 0.343, 1.180,
    1.077, 0.590, 1.286, 2.014, 4.773,
    4.038, 2.958, 1.461
]
```

**Why Weighted Loss:**
- **Problem**: Severe class imbalance (e.g., Hernia: 30x weight)
- **Solution**: Inverse frequency weighting
- **Benefit**: Forces model to learn rare conditions

##### Rare Class Augmentation
```python
# Augmentation thresholds (class frequency)
thresholds = {
    'Hernia': 0.003,           # Most rare
    'Pneumonia': 0.01,
    'Fibrosis': 0.012,
    'Edema': 0.016,
    'Emphysema': 0.018,
    'Cardiomegaly': 0.02,
    'Pleural_Thickening': 0.025,
    'Consolidation': 0.035,
    'Pneumothorax': 0.045      # Least rare in this group
}
```

**Why Rare Class Augmentation:**
- **Problem**: Some classes have <1% representation
- **Solution**: Oversample rare classes with augmentation
- **Benefit**: Balanced learning across all conditions

#### Training Parameters
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Weighted Binary Cross-Entropy with Logits
- **Batch Size**: 64 (32 for testing, 64 for full training)
- **Epochs**: 20 (5 for testing, 50 for full training)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=7)
- **Early Stopping**: 7 epochs patience after 25 warmup epochs

#### Data Splitting Strategy
- **Method**: Patient-level GroupShuffleSplit
- **Split**: 80% train, 20% validation
- **Random State**: 42 (reproducible)

**Why Patient-Level Splitting:**
- **Problem**: Data leakage if same patient in train/val
- **Solution**: Group by Patient ID before splitting
- **Benefit**: True generalization to unseen patients

### Hardware Optimizations

#### Apple Silicon (M1/M2/M3/M4)
- **Device**: MPS (Metal Performance Shaders)
- **Batch Size**: 64 (optimized for unified memory)
- **Workers**: 8 (performance cores)
- **Optimizations**: Persistent workers, prefetch factor=2

#### NVIDIA CUDA
- **Device**: CUDA
- **Multi-GPU**: DataParallel support
- **Memory**: Pin memory enabled

#### CPU Fallback
- **Device**: CPU
- **Performance**: Significantly slower but functional

### Evaluation Metrics

#### Comprehensive Metrics (8 total):
1. **AUC-ROC**: Area Under ROC Curve
2. **Threshold**: Classification threshold (0.5)
3. **Accuracy**: Overall correctness
4. **Specificity**: True Negative Rate
5. **Recall**: True Positive Rate
6. **Precision**: Positive Predictive Value
7. **Sensitivity**: Same as Recall
8. **F1-Score**: Harmonic mean of Precision/Recall

#### Baseline Comparison
- **Reference**: Wang et al. ChestX-ray8 paper results
- **Format**: Side-by-side comparison with improvements
- **Output**: Baseline → Our → Improvement → Better (Yes/No/Equal)

### Dataset Information

#### ChestX-ray14 Dataset
- **Total Images**: 112,120 frontal-view X-rays
- **Unique Patients**: 30,805
- **Image Size**: Originally variable, resized to 224×224
- **Labels**: Multi-label (patient can have multiple conditions)
- **Format**: CSV metadata + image files

#### Data Files
- **Training**: `train_data.csv` (~90K samples)
- **Testing**: `test_data.csv` (~22K samples)
- **Small Training**: `train_data_small.csv` (500 samples)
- **Small Testing**: `test_data_small.csv` (200 samples)

### Current Limitations & Future Considerations

#### Known Issues
1. **Small Dataset Performance**: Limited epochs cause poor precision/recall
2. **Class Imbalance**: Still challenging despite mitigation strategies
3. **Memory Usage**: Large dataset requires significant RAM

#### Potential Improvements
1. **Architecture**: Consider ResNet, EfficientNet, or Vision Transformers
2. **Loss Functions**: Focal Loss, LDAM Loss for imbalanced data
3. **Augmentation**: More sophisticated medical image augmentations
4. **Ensemble Methods**: Multiple model voting
5. **Transfer Learning**: Medical domain-specific pre-training

---

## Version 1.1 - Training Pipeline Optimizations
**Date**: December 2024  
**Status**: Completed

#### Changes Made
- **Early Stopping Fix**: Reduced warmup epochs from 25 to 5 for 20-epoch training
- **Early Stopping Patience**: Reduced from 7 to 5 epochs for faster convergence detection
- **Loss Function**: Replaced Weighted BCE with FocalLoss for better class imbalance handling
- **Hardware Optimization**: Added M4 Pro specific optimizations (batch size 64, 8 workers)
- **Data Loading**: Added persistent workers and prefetch factor for better performance
- **Epoch Timing**: Added per-epoch time tracking for performance monitoring

#### Rationale
- **Problem**: Early stopping never activated due to 25-epoch warmup vs 20 total epochs
- **Solution**: Reduced warmup to 5 epochs, allowing early stopping from epoch 6 onwards
- **Loss Function Problem**: Weighted BCE still struggled with extreme class imbalance (Hernia: 30x weight)
- **FocalLoss Solution**: Focuses learning on hard examples, better handles class imbalance
- **Expected Benefit**: Prevents overfitting and improves rare class detection

#### Performance Impact
- **Before**: Training always completed full 20 epochs regardless of overfitting
- **After**: Can stop early when validation loss increases (e.g., should stop around epoch 14-15)
- **Analysis**: Validation loss increased from 0.2375 (epoch 9) to 0.3072 (epoch 20), indicating overfitting

#### Code Changes
```python
# Early stopping parameters updated
early_stopping = EarlyStopping(patience=5, min_delta=0.01, warmup_epochs=5)

# FocalLoss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss

# Alpha weights calculation from class frequencies
class_frequencies = [0.0342, 0.0310, 0.1641, 0.0028, 0.2451, 0.0712, 0.0780,
                     0.1424, 0.0653, 0.0417, 0.0176, 0.0208, 0.0284, 0.0575]
min_freq = min(class_frequencies)
alpha = [min_freq / f for f in class_frequencies]
alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(device)
criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0, reduction='mean')

# M4 Pro optimizations
batch_size = 64  # Increased from 32
num_workers = 8  # Increased from 4
persistent_workers=True, prefetch_factor=2  # Added to DataLoader

# Epoch timing
epoch_start_time = time.time()
# ... training code ...
epoch_time = time.time() - epoch_start_time
print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
```

---

## Version 2.0 - Vision-Language Model (VLLM) Integration (Proposed)
**Date**: October 2025 (Planned)
**Status**: Proposal / Research Phase

### Concept: From Classification to Understanding
The current DenseNet-121 approach treats the X-ray as a generic image. Modern **Visual Large Language Models (VLLMs)** and **Foundation Models** (like CLIP, BioMedCLIP, LLaVA-Med) understand images in the context of medical language.

### Proposed Approaches

#### 1. Domain-Specific Foundation Models (BioMedCLIP)
Instead of using ImageNet weights (natural images like cats/dogs), we use a vision encoder pre-trained on medical literature (PubMed).
- **Model**: BioMedCLIP (Microsoft) or PubMedCLIP.
- **Mechanism**: Replace DenseNet-121 backbone with BioMedCLIP's Vision Transformer (ViT).
- **Benefit**: The model already "knows" what medical pathologies look like from millions of image-text pairs.
- **Implementation**:
  ```python
  # Concept
  model = BioMedCLIP_Vision_Encoder()
  features = model(image) # 512 or 768 dim
  output = Classifier(features)
  ```

#### 2. Zero-Shot / Few-Shot Classification via Prompting
Using the text-encoder capability to match images to class descriptions.
- **Method**: Compare image embeddings with text embeddings of prompts like "An X-ray showing pneumonia", "A normal chest X-ray".
- **Benefit**: Can handle rare classes (like Hernia) better if the model has seen them in literature, even if our training set is small.

#### 3. Generative Diagnosis (LLaVA-Med / CheXagent)
Instead of just outputting probabilities, use a VLLM to generate a textual report.
- **Input**: Image + Prompt "Describe the findings in this chest X-ray."
- **Output**: "The image shows opacity in the right lower lobe consistent with pneumonia..."
- **Parsing**: Extract labels from the generated text.
- **Pros**: Explainability.
- **Cons**: High computational cost, slower inference.

### Recommended Roadmap
1.  **Phase 1 (Immediate)**: Replace DenseNet-121 with **BioMedCLIP Vision Encoder** (Linear Probe or Fine-tuning). This keeps the current pipeline but upgrades the "eyes" of the model.
2.  **Phase 2**: Experiment with **Zero-Shot Ensembling**. Combine DenseNet predictions with BioMedCLIP zero-shot predictions to improve robustness on rare classes.
3.  **Phase 3**: Explore **CheXagent** for report generation if explainability becomes a requirement.

### Why this matters for this project?
- **Data Efficiency**: Foundation models require fewer examples to learn (good for our small dataset issues).
- **Class Imbalance**: Semantic understanding of "Hernia" from text helps when image examples are scarce.

---

## Change Log Template

### Version X.X - [Change Description]
**Date**: [Date]  
**Status**: [Planned/In Progress/Completed]

#### Changes Made
- [Specific change 1]
- [Specific change 2]

#### Rationale
- **Problem**: [What issue was being addressed]
- **Solution**: [How the change addresses it]
- **Expected Benefit**: [What improvement is expected]

#### Performance Impact
- **Before**: [Baseline metrics]
- **After**: [New metrics]
- **Analysis**: [Interpretation of results]

#### Code Changes
```python
# Code snippets showing the changes
```

---

*This document should be updated with each architectural change to maintain a clear history of model evolution and decision rationale.*