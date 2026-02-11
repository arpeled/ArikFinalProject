#!/usr/bin/env python3
"""
Test reproducibility of iteration_012 results by running multiple tests with same config.
Verifies that results are not random by checking consistency across runs.
"""

import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime
def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_reproducibility_test():
    """Run multiple tests with same config to check reproducibility"""
    
    results = []
    num_runs = 3
    
    print(f"Running {num_runs} reproducibility tests...")
    
    for run_id in range(num_runs):
        print(f"\n=== Run {run_id + 1}/{num_runs} ===")
        
        # Set same seed for each run
        set_seeds(42)
        
        # Run training script directly
        import subprocess
        import sys
        
        try:
            # Run training with small dataset
            result = subprocess.run([
                sys.executable, 'run_pipeline_test.py'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"Training failed: {result.stderr}")
                continue
                
            # Load results from the generated files
            import glob
            result_files = glob.glob('pipeline_results_*.csv')
            if not result_files:
                print("No result files found")
                continue
                
            # Get the most recent result file
            latest_file = max(result_files, key=os.path.getctime)
            test_results = pd.read_csv(latest_file)
            
            # Store key metrics
            run_results = {
                'run_id': run_id + 1,
                'avg_auc': test_results['AUC'].mean(),
                'avg_f1': test_results['F1_Score'].mean(),
                'avg_recall': test_results['Recall'].mean(),
                'avg_precision': test_results['Precision'].mean()
            }
            
            results.append(run_results)
            print(f"Run {run_id + 1} - Avg AUC: {run_results['avg_auc']:.4f}, Avg F1: {run_results['avg_f1']:.4f}")
            
            # Clean up generated files
            for pattern in ['pipeline_*.pth', 'pipeline_*.csv', 'pipeline_*.txt']:
                for file in glob.glob(pattern):
                    try:
                        os.remove(file)
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error in run {run_id + 1}: {str(e)}")
            continue
    
    return results

def analyze_reproducibility(results):
    """Analyze reproducibility of results"""
    
    if not results:
        print("❌ No results to analyze")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    print("\n=== REPRODUCIBILITY ANALYSIS ===")
    print("\nResults Summary:")
    print(df.round(6))
    
    print("\nStatistical Analysis:")
    for metric in ['avg_auc', 'avg_f1', 'avg_recall', 'avg_precision']:
        values = df[metric].values
        std_dev = np.std(values)
        mean_val = np.mean(values)
        cv = (std_dev / mean_val) * 100 if mean_val != 0 else 0
        
        print(f"{metric}:")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std Dev: {std_dev:.6f}")
        print(f"  Coefficient of Variation: {cv:.2f}%")
        print(f"  Range: {np.min(values):.6f} - {np.max(values):.6f}")
    
    # Check if results are consistent (low variation)
    auc_cv = (np.std(df['avg_auc']) / np.mean(df['avg_auc'])) * 100
    f1_cv = (np.std(df['avg_f1']) / np.mean(df['avg_f1'])) * 100 if np.mean(df['avg_f1']) != 0 else 0
    
    print(f"\n=== REPRODUCIBILITY VERDICT ===")
    if auc_cv < 1.0:  # Less than 1% variation
        print("✅ REPRODUCIBLE: AUC results show low variation (<1%)")
    else:
        print("❌ NOT REPRODUCIBLE: AUC results show high variation (>1%)")
    
    if f1_cv < 5.0 or np.mean(df['avg_f1']) < 0.01:  # Allow higher variation for very low F1 scores
        print("✅ REPRODUCIBLE: F1 results are consistent")
    else:
        print("❌ NOT REPRODUCIBLE: F1 results show high variation")
    
    return df

if __name__ == "__main__":
    print("Testing reproducibility of iteration_012 results...")
    
    # Check if required files exist
    if not os.path.exists("./ChestX-ray14/train_data_small.csv"):
        print("❌ Error: Small dataset files not found. Run create_small_datasets.py first")
        exit(1)
    
    if not os.path.exists("run_pipeline_test.py"):
        print("❌ Error: run_pipeline_test.py not found")
        exit(1)
    
    # Run reproducibility test
    results = run_reproducibility_test()
    
    # Analyze results
    df = analyze_reproducibility(results)
    
    # Save results if we have any
    if not df.empty:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"iteration_012_test_reproducibility_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\n❌ No results to save")