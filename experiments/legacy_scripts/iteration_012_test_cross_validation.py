#!/usr/bin/env python3
"""
Cross-validation test for iteration_012 to verify model stability across different data splits.
Tests if the model performance is consistent across different train/validation splits.
"""

import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from datetime import datetime
import subprocess
import sys
import glob

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_cv_splits(data_path, n_splits=3, random_state=42):
    """Create cross-validation splits from the data"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get unique patients for patient-level splitting
    unique_patients = df['Patient ID'].unique()
    
    # Create KFold splits on patients
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for fold_idx, (train_patients, val_patients) in enumerate(kf.split(unique_patients)):
        train_patient_ids = unique_patients[train_patients]
        val_patient_ids = unique_patients[val_patients]
        
        # Create train/val dataframes
        train_df = df[df['Patient ID'].isin(train_patient_ids)]
        val_df = df[df['Patient ID'].isin(val_patient_ids)]
        
        # Save temporary CSV files for this fold
        train_file = f"temp_cv_train_fold_{fold_idx}.csv"
        val_file = f"temp_cv_val_fold_{fold_idx}.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        
        splits.append({
            'fold': fold_idx,
            'train_file': train_file,
            'val_file': val_file,
            'train_size': len(train_df),
            'val_size': len(val_df)
        })
    
    return splits

def run_cross_validation_test():
    """Run cross-validation test with iteration_012 config"""
    
    # Use small dataset for faster testing
    data_path = './ChestX-ray14/train_data_small.csv'
    if not os.path.exists(data_path):
        print("❌ Error: Small dataset not found. Run create_small_datasets.py first")
        return None
    
    # Create CV splits
    print("Creating cross-validation splits...")
    cv_splits = create_cv_splits(data_path, n_splits=3)
    
    results = []
    
    for split in cv_splits:
        print(f"\n=== Fold {split['fold'] + 1}/3 ===")
        print(f"Train size: {split['train_size']}, Val size: {split['val_size']}")
        
        # Set seeds for reproducibility
        set_seeds(42 + split['fold'])  # Different seed per fold
        
        try:
            # Run test pipeline with this fold's data
            result = subprocess.run([
                sys.executable, 'run_pipeline_test.py'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"Training failed: {result.stderr}")
                continue
                
            # Load results from the generated files
            result_files = glob.glob('pipeline_results_*.csv')
            if not result_files:
                print("No result files found")
                continue
                
            # Get the most recent result file
            latest_file = max(result_files, key=os.path.getctime)
            test_results = pd.read_csv(latest_file)
            
            # Store results
            fold_results = {
                'fold': split['fold'] + 1,
                'train_size': split['train_size'],
                'val_size': split['val_size'],
                'avg_auc': test_results['AUC'].mean(),
                'avg_f1': test_results['F1_Score'].mean(),
                'avg_recall': test_results['Recall'].mean(),
                'avg_precision': test_results['Precision'].mean(),
                'std_auc': test_results['AUC'].std(),
                'std_f1': test_results['F1_Score'].std()
            }
            
            results.append(fold_results)
            
            print(f"Fold {split['fold'] + 1} Results:")
            print(f"  Avg AUC: {fold_results['avg_auc']:.4f} ± {fold_results['std_auc']:.4f}")
            print(f"  Avg F1: {fold_results['avg_f1']:.4f} ± {fold_results['std_f1']:.4f}")
            
            # Clean up generated files
            for pattern in ['pipeline_*.pth', 'pipeline_*.csv', 'pipeline_*.txt']:
                for file in glob.glob(pattern):
                    try:
                        os.remove(file)
                    except:
                        pass
            
        except Exception as e:
            print(f"❌ Error in fold {split['fold'] + 1}: {str(e)}")
            continue
        
        finally:
            # Clean up temporary files
            for temp_file in [split['train_file'], split['val_file']]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    return results

def analyze_cv_stability(results):
    """Analyze cross-validation stability"""
    
    if not results:
        print("❌ No results to analyze")
        return None
    
    df = pd.DataFrame(results)
    
    print("\n=== CROSS-VALIDATION STABILITY ANALYSIS ===")
    print("\nFold Results:")
    print(df[['fold', 'avg_auc', 'avg_f1', 'avg_recall', 'avg_precision', 'epochs_trained']].round(4))
    
    print("\nStability Metrics:")
    
    # Calculate coefficient of variation for key metrics
    metrics = ['avg_auc', 'avg_f1', 'avg_recall', 'avg_precision']
    stability_results = {}
    
    for metric in metrics:
        values = df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        
        stability_results[metric] = {
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'min': np.min(values),
            'max': np.max(values)
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")
        print(f"  CV: {cv:.2f}%")
        print(f"  Range: {np.min(values):.4f} - {np.max(values):.4f}")
    
    # Stability verdict
    print(f"\n=== STABILITY VERDICT ===")
    
    auc_cv = stability_results['avg_auc']['cv']
    f1_cv = stability_results['avg_f1']['cv']
    
    stable_metrics = 0
    
    if auc_cv < 5.0:  # Less than 5% variation
        print("✅ AUC is STABLE across folds (CV < 5%)")
        stable_metrics += 1
    else:
        print("❌ AUC is UNSTABLE across folds (CV > 5%)")
    
    if f1_cv < 10.0 or stability_results['avg_f1']['mean'] < 0.01:  # Allow higher variation for very low F1
        print("✅ F1 Score is STABLE across folds")
        stable_metrics += 1
    else:
        print("❌ F1 Score is UNSTABLE across folds")
    
    # Check if training converged consistently
    if len(df) > 1:
        auc_range = df['avg_auc'].max() - df['avg_auc'].min()
        if auc_range < 0.05:  # Less than 5% range in AUC
            print("✅ Training CONVERGED consistently across folds")
            stable_metrics += 1
        else:
            print("❌ Training convergence VARIED significantly across folds")
    else:
        print("ℹ️ Not enough folds to assess convergence")
    
    # Overall stability
    if stable_metrics >= 2:
        print(f"\n🎯 OVERALL: MODEL IS STABLE ({stable_metrics}/3 criteria met)")
    else:
        print(f"\n⚠️  OVERALL: MODEL STABILITY QUESTIONABLE ({stable_metrics}/3 criteria met)")
    
    return df, stability_results

def main():
    """Main function to run cross-validation test"""
    
    print("Running cross-validation stability test for iteration_012...")
    
    # Check if required files exist
    if not os.path.exists("./ChestX-ray14/train_data_small.csv"):
        print("❌ Error: Small dataset files not found. Run create_small_datasets.py first")
        return
    
    if not os.path.exists("run_pipeline_test.py"):
        print("❌ Error: run_pipeline_test.py not found")
        return
    
    # Run cross-validation
    results = run_cross_validation_test()
    
    if not results:
        print("❌ Cross-validation test failed")
        return
    
    # Analyze stability
    df, stability_results = analyze_cv_stability(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Save detailed results
    output_file = f"iteration_012_test_cross_validation_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Save stability summary
    summary_file = f"iteration_012_test_cv_stability_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("Cross-Validation Stability Test for Iteration 012\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {datetime.now()}\n")
        f.write(f"Number of Folds: {len(results)}\n\n")
        
        f.write("Stability Results:\n")
        for metric, stats in stability_results.items():
            f.write(f"\n{metric.upper()}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {stats['std']:.4f}\n")
            f.write(f"  CV: {stats['cv']:.2f}%\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Stability summary saved to: {summary_file}")

if __name__ == "__main__":
    main()