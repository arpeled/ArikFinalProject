#!/usr/bin/env python3
"""
Simple repeatability test for iteration_012.
Tests if the trained model produces consistent results across multiple inference runs.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_model_repeatability():
    """Test if model produces same results on multiple runs"""
    
    # Load the trained model from iteration_012
    model_path = "auto_improvement_runs/iteration_012/pipeline_model_20251229-101232.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print("Loading trained model...")
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import model class and create model
    from dataset import ModifiedDenseNetWithDropOut
    
    model = ModifiedDenseNetWithDropOut(
        num_classes=14,
        use_additional_features=True
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load test data (small dataset for speed)
    test_csv = './ChestX-ray14/test_data_small.csv'
    if not os.path.exists(test_csv):
        print(f"❌ Test data not found: {test_csv}")
        return None
    
    # Import dataset class
    from dataset import ChestXRayDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = ChestXRayDataset(
        dataset=None,
        csv_file=test_csv,
        root_dir='./ChestX-ray14/images224',
        transform=transform,
        use_additional_features=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = []
    num_runs = 5
    
    print(f"Running {num_runs} inference tests...")
    
    for run_id in range(num_runs):
        print(f"Run {run_id + 1}/{num_runs}")
        
        # Set same seed for each run
        set_seeds(42)
        
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
        
        # Combine all predictions
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Calculate metrics for each class
        from sklearn.metrics import roc_auc_score
        
        class_aucs = []
        for i in range(14):  # 14 classes
            try:
                auc = roc_auc_score(labels[:, i], predictions[:, i])
                class_aucs.append(auc)
            except:
                class_aucs.append(0.5)  # Default for classes with no positive samples
        
        run_results = {
            'run_id': run_id + 1,
            'avg_auc': np.mean(class_aucs),
            'std_auc': np.std(class_aucs),
            'predictions_sum': np.sum(predictions),
            'predictions_mean': np.mean(predictions)
        }
        
        results.append(run_results)
        print(f"  Avg AUC: {run_results['avg_auc']:.6f}")
    
    return results

def analyze_repeatability(results):
    """Analyze repeatability of results"""
    
    if not results:
        print("❌ No results to analyze")
        return None
    
    df = pd.DataFrame(results)
    
    print("\n=== REPEATABILITY ANALYSIS ===")
    print("\nResults Summary:")
    print(df.round(8))
    
    # Check consistency
    metrics = ['avg_auc', 'predictions_sum', 'predictions_mean']
    
    print("\nConsistency Check:")
    for metric in metrics:
        values = df[metric].values
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        print(f"{metric}:")
        print(f"  Values: {values}")
        print(f"  Std Dev: {std_dev:.10f}")
        print(f"  Range: {np.min(values):.8f} - {np.max(values):.8f}")
        
        # Check if identical (very small std dev)
        if std_dev < 1e-6:
            print(f"  ✅ IDENTICAL across runs")
        elif std_dev < 1e-3:
            print(f"  ✅ HIGHLY CONSISTENT")
        else:
            print(f"  ❌ INCONSISTENT")
    
    # Overall verdict
    auc_std = np.std(df['avg_auc'])
    pred_std = np.std(df['predictions_sum'])
    
    print(f"\n=== REPEATABILITY VERDICT ===")
    if auc_std < 1e-6 and pred_std < 1e-6:
        print("✅ PERFECTLY REPEATABLE")
        print("   Model produces identical results across runs")
    elif auc_std < 1e-3 and pred_std < 1e-3:
        print("✅ HIGHLY REPEATABLE")
        print("   Model produces very consistent results")
    else:
        print("❌ NOT REPEATABLE")
        print("   Model produces different results across runs")
    
    return df

if __name__ == "__main__":
    print("Testing repeatability of iteration_012 model...")
    
    # Check prerequisites
    if not os.path.exists("auto_improvement_runs/iteration_012/pipeline_model_20251229-101232.pth"):
        print("❌ Model file not found")
        exit(1)
    
    if not os.path.exists("./ChestX-ray14/test_data_small.csv"):
        print("❌ Test data not found. Run create_small_datasets.py first")
        exit(1)
    
    # Run repeatability test
    results = test_model_repeatability()
    
    if results:
        # Analyze results
        df = analyze_repeatability(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"iteration_012_test_repeatability_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("❌ Test failed")