#!/usr/bin/env python3
"""
Simple model evaluation test for iteration_012.
Loads the trained model and evaluates on test dataset, saving results per disease.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

def load_model():
    """Load the trained model"""
    model_path = "auto_improvement_runs/iteration_012/pipeline_model_20251229-101232.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import and create model
    from dataset import ModifiedDenseNetWithDropOut
    
    model = ModifiedDenseNetWithDropOut(
        num_classes=14,
        use_additional_features=True
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, device

def evaluate_model(model, device):
    """Evaluate model on test dataset"""
    
    # Check test data
    # test_csv = './ChestX-ray14/test_data_small.csv'
    test_csv = './ChestX-ray14/test_data.csv'
    if not os.path.exists(test_csv):
        print(f"❌ Test data not found: {test_csv}")
        return None
    
    # Import dataset
    from dataset import ChestXRayDataset
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    test_dataset = ChestXRayDataset(
        dataset=None,
        csv_file=test_csv,
        root_dir='./ChestX-ray14/images224',
        transform=transform,
        use_additional_features=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Disease names
    disease_names = [
        'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 
        'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 
        'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
    ]
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, additional_features, labels) in enumerate(test_loader):
            images = images.to(device)
            additional_features = additional_features.to(device)
            
            # Forward pass
            outputs = model(images, additional_features)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
    
    # Combine predictions
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    
    print(f"✅ Evaluation completed")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return predictions, labels, disease_names

def calculate_metrics(predictions, labels, disease_names):
    """Calculate metrics per disease"""
    
    results = []
    
    for i, disease in enumerate(disease_names):
        y_true = labels[:, i]
        y_pred = predictions[:, i]
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.5  # Default for classes with no positive samples
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        try:
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # Calculate specificity
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Count statistics
        total_samples = len(y_true)
        positive_samples = np.sum(y_true == 1)
        negative_samples = np.sum(y_true == 0)
        predicted_positive = np.sum(y_pred_binary == 1)
        
        results.append({
            'Disease': disease,
            'AUC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1_Score': f1,
            'Threshold': 0.5,
            'Total_Samples': total_samples,
            'Positive_Samples': positive_samples,
            'Negative_Samples': negative_samples,
            'Predicted_Positive': predicted_positive,
            'Mean_Prediction': np.mean(y_pred),
            'Std_Prediction': np.std(y_pred),
            'Min_Prediction': np.min(y_pred),
            'Max_Prediction': np.max(y_pred)
        })
        
        print(f"{disease:20} | AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}")
    
    return results

def save_results(results, predictions, labels, disease_names):
    """Save results to CSV files"""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    summary_file = f"iteration_012_test_evaluation_summary_{timestamp}.csv"
    results_df.to_csv(summary_file, index=False)
    print(f"✅ Summary saved to: {summary_file}")
    
    # Save detailed predictions per disease
    detailed_results = []
    
    for i, disease in enumerate(disease_names):
        y_true = labels[:, i]
        y_pred = predictions[:, i]
        
        for j in range(len(y_true)):
            detailed_results.append({
                'Sample_ID': j,
                'Disease': disease,
                'True_Label': int(y_true[j]),
                'Prediction': float(y_pred[j]),
                'Predicted_Label': int(y_pred[j] > 0.5)
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = f"iteration_012_test_detailed_predictions_{timestamp}.csv"
    detailed_df.to_csv(detailed_file, index=False)
    print(f"✅ Detailed predictions saved to: {detailed_file}")
    
    return summary_file, detailed_file

def main():
    """Main evaluation function"""
    
    print("ITERATION 012 MODEL EVALUATION")
    print("=" * 40)
    print(f"Started: {datetime.now()}")
    
    # Load model
    model_result = load_model()
    if model_result is None:
        return
    
    model, device = model_result
    
    # Evaluate model
    eval_result = evaluate_model(model, device)
    if eval_result is None:
        return
    
    predictions, labels, disease_names = eval_result
    
    # Calculate metrics
    print("\nCalculating metrics per disease...")
    results = calculate_metrics(predictions, labels, disease_names)
    
    # Save results
    print("\nSaving results...")
    summary_file, detailed_file = save_results(results, predictions, labels, disease_names)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    results_df = pd.DataFrame(results)
    print(f"Average AUC: {results_df['AUC'].mean():.4f}")
    print(f"Average F1: {results_df['F1_Score'].mean():.4f}")
    print(f"Average Accuracy: {results_df['Accuracy'].mean():.4f}")
    
    print("\n=== FILES CREATED ===")
    print(f"- {summary_file}")
    print(f"- {detailed_file}")
    
    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    main()