"""
Modified testing script for pipeline integration
Based on chest-x-ray-classifier-densenet121-test_v05_updated.py
"""

import os
import sys

# Add core/ directory to path for imports
_core_dir = os.path.dirname(os.path.abspath(__file__))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from dataset import ChestXRayDataset, ModifiedDenseNetWithDropOut
from threshold_optimizer import load_thresholds, apply_thresholds, generate_prevalence_based_thresholds


def test_model_pipeline(model_file, results_file, logger=None, head_config=None):
    """
    Testing function that can be called from pipeline

    Args:
        model_file: Path to model weights
        results_file: Path to save results
        logger: Logger instance
        head_config: Optional head configuration dict (for MLP head models)
    """
    # Use provided logger or create default
    if logger is None:
        import logging
        logger = logging.getLogger('testing')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
    # Test parameters
    batch_size = 32
    num_classes = 14
    use_additional_features = True
    test_csv_file = './ChestX-ray14/test_data.csv'
    root_dir = './ChestX-ray14/images224'

    label_columns = [
        'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass',
        'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
        'Fibrosis', 'Edema', 'Consolidation'
    ]

    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load test data
    test_dataset = ChestXRayDataset(dataset=None, csv_file=test_csv_file,
                                   root_dir=root_dir, transform=test_transform,
                                   use_additional_features=use_additional_features)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model (config-driven head selection)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ModifiedDenseNetWithDropOut(
        num_classes=num_classes,
        use_additional_features=use_additional_features,
        head_config=head_config
    ).to(device)

    # Upgrade to multi-layer MLP head if 'layers' is specified in head_config
    if head_config and head_config.get('layers'):
        import torch.nn as nn
        layers = head_config.get('layers')
        dropout = head_config.get('dropout', 0.2)
        activation_name = head_config.get('activation', 'relu')
        in_features = 1024 + 128 if use_additional_features else 1024

        activation_fn = nn.ReLU if activation_name == 'relu' else nn.GELU

        modules = []
        prev_dim = in_features
        for dim in layers:
            modules.append(nn.Linear(prev_dim, dim))
            modules.append(activation_fn())
            modules.append(nn.Dropout(dropout))
            prev_dim = dim
        modules.append(nn.Linear(prev_dim, num_classes))

        model.mlp_head = nn.Sequential(*modules).to(device)
        model.head_type = 'mlp_multi'
        logger.info(f"   Multi-layer head: {in_features} -> {' -> '.join(map(str, layers))} -> {num_classes}")

        # Monkey-patch the forward method to use multi-layer head
        def patched_forward(x, additional_features=None):
            """Patched forward that uses multi-layer MLP head."""
            base_out = model.base_model(x)
            base_out = model.dropout(base_out)

            if model.use_additional_features and additional_features is not None:
                additional_out = nn.functional.relu(model.additional_fc(additional_features))
                combined = torch.cat([base_out, additional_out], dim=1)
            else:
                combined = base_out

            out = model.mlp_head(combined)
            return out

        model.forward = patched_forward

    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()

    # Try to load optimized thresholds
    threshold_file = model_file.replace('pipeline_model_', 'thresholds_').replace('.pth', '.json')
    optimized_thresholds = load_thresholds(threshold_file, label_columns, logger=logger)

    use_optimized = (optimized_thresholds is not None)
    use_prevalence_fallback = False

    if optimized_thresholds is not None:
        logger.info(f"✅ Using optimized per-class thresholds (avg: {optimized_thresholds.mean():.3f})")
    else:
        logger.info(f"⚠️  No optimized thresholds found")
        logger.info(f"   Will generate prevalence-based fallback thresholds after collecting predictions")
        use_prevalence_fallback = True

    logger.info(f"Running evaluation on {len(test_dataset)} test samples...")

    # Run evaluation
    predictions = []
    true_labels = []
    probs_all = []

    with torch.no_grad():
        for batch in test_loader:
            if use_additional_features:
                images, additional_features, labels = batch
                images, additional_features = images.to(device), additional_features.to(device)
                outputs = model(images, additional_features)
            else:
                images, labels = batch
                images = images.to(device)
                outputs = model(images)

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            probs_all.extend(probs)
            true_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    probs_all = np.array(probs_all)
    true_labels = np.array(true_labels)

    # Generate prevalence-based fallback thresholds if needed
    if use_prevalence_fallback:
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING PREVALENCE-BASED FALLBACK THRESHOLDS")
        logger.info("="*80)
        optimized_thresholds = generate_prevalence_based_thresholds(
            y_true=true_labels,
            class_names=label_columns,
            logger=logger
        )
        logger.info("="*80)
        logger.info(f"✅ Using prevalence-based thresholds (avg: {optimized_thresholds.mean():.3f})")
        logger.info("")

    # Apply per-class thresholds to all predictions at once
    predictions = apply_thresholds(probs_all, optimized_thresholds)
    
    # Calculate metrics
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    probs_all = np.array(probs_all)
    
    results = []
    for i, label in enumerate(label_columns):
        y_true = true_labels[:, i]
        y_pred = predictions[:, i]
        y_prob = probs_all[:, i]
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = float('nan')
        
        # Calculate confusion matrix metrics
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class present
                if y_true[0] == 0:  # Only negatives
                    tn, fp, fn, tp = cm[0,0], 0, 0, 0
                else:  # Only positives
                    tn, fp, fn, tp = 0, 0, 0, cm[0,0]
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        except Exception as e:
            print(f"Warning: Confusion matrix error for {label}: {e}")
            sensitivity = 0
            specificity = 0
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Debug info
        pos_count = np.sum(y_true)
        pred_pos_count = np.sum(y_pred)
        logger.info(f"{label}: True positives: {pos_count}, Predicted positives: {pred_pos_count}, AUC: {auc:.4f}")
        
        results.append({
            'Label': label,
            'AUC': auc,
            'Threshold': float(optimized_thresholds[i]),  # Use actual threshold for this class
            'Accuracy': accuracy,
            'Specificity': specificity,
            'Recall': recall,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'F1_Score': f1,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")

    # Save confusion matrix to JSON
    import json
    confusion_file = results_file.replace('pipeline_results_', 'confusion_matrix_').replace('.csv', '.json')
    confusion_dict = {}
    for result in results:
        label = result['Label']
        confusion_dict[label] = {
            'TP': result['TP'],
            'FP': result['FP'],
            'TN': result['TN'],
            'FN': result['FN'],
            'Total_Positive': result['TP'] + result['FN'],
            'Total_Negative': result['TN'] + result['FP'],
            'Predicted_Positive': result['TP'] + result['FP'],
            'Predicted_Negative': result['TN'] + result['FN']
        }

    with open(confusion_file, 'w') as f:
        json.dump(confusion_dict, f, indent=2)
    logger.info(f"Confusion matrix saved to {confusion_file}")

    # Print summary
    logger.info("\nTest Results Summary:")
    logger.info(f"\n{results_df.to_string(index=False)}")

    # Print confusion matrix summary
    logger.info("\nConfusion Matrix Summary:")
    logger.info("="*80)
    logger.info(f"{'Disease':<20} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'TPR':>8} {'FPR':>8}")
    logger.info("="*80)
    for result in results:
        label = result['Label']
        tp, fp, tn, fn = result['TP'], result['FP'], result['TN'], result['FN']
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        logger.info(f"{label:<20} {tp:>6} {fp:>6} {tn:>6} {fn:>6} {tpr:>8.4f} {fpr:>8.4f}")
    logger.info("="*80)

    return results_df