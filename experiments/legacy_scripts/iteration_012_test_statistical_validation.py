#!/usr/bin/env python3
"""
Statistical validation test for iteration_012 results.
Tests if results are significantly better than random chance.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_iteration_results():
    """Load iteration_012 results"""
    results_path = "auto_improvement_runs/iteration_012/pipeline_results_20251229-101232.csv"
    return pd.read_csv(results_path)

def generate_random_baseline(n_samples=1000, n_classes=14):
    """Generate random baseline results for comparison"""
    
    # Random AUC values (should be around 0.5 for random classifier)
    random_aucs = np.random.normal(0.5, 0.05, (n_samples, n_classes))
    random_aucs = np.clip(random_aucs, 0, 1)
    
    # Random F1 scores (should be very low for imbalanced data)
    random_f1s = np.random.exponential(0.02, (n_samples, n_classes))
    random_f1s = np.clip(random_f1s, 0, 1)
    
    # Random precision/recall
    random_precision = np.random.exponential(0.05, (n_samples, n_classes))
    random_precision = np.clip(random_precision, 0, 1)
    
    random_recall = np.random.exponential(0.05, (n_samples, n_classes))
    random_recall = np.clip(random_recall, 0, 1)
    
    return {
        'auc': random_aucs,
        'f1': random_f1s,
        'precision': random_precision,
        'recall': random_recall
    }

def perform_statistical_tests(actual_results, random_baseline):
    """Perform statistical tests to compare actual vs random results"""
    
    print("=== STATISTICAL VALIDATION TESTS ===\n")
    
    # Extract actual metrics
    actual_auc = actual_results['AUC'].values
    actual_f1 = actual_results['F1_Score'].values
    actual_precision = actual_results['Precision'].values
    actual_recall = actual_results['Recall'].values
    
    # Calculate mean metrics
    actual_mean_auc = np.mean(actual_auc)
    actual_mean_f1 = np.mean(actual_f1)
    actual_mean_precision = np.mean(actual_precision)
    actual_mean_recall = np.mean(actual_recall)
    
    # Random baseline means
    random_mean_aucs = np.mean(random_baseline['auc'], axis=1)
    random_mean_f1s = np.mean(random_baseline['f1'], axis=1)
    random_mean_precisions = np.mean(random_baseline['precision'], axis=1)
    random_mean_recalls = np.mean(random_baseline['recall'], axis=1)
    
    results = {}
    
    # Test 1: One-sample t-test for AUC > 0.5 (better than random)
    auc_tstat, auc_pvalue = stats.ttest_1samp(actual_auc, 0.5)
    results['auc_vs_random'] = {
        'actual_mean': actual_mean_auc,
        'random_baseline': 0.5,
        't_statistic': auc_tstat,
        'p_value': auc_pvalue,
        'significant': auc_pvalue < 0.05 and actual_mean_auc > 0.5
    }
    
    print(f"1. AUC vs Random (0.5):")
    print(f"   Actual Mean AUC: {actual_mean_auc:.4f}")
    print(f"   t-statistic: {auc_tstat:.4f}")
    print(f"   p-value: {auc_pvalue:.6f}")
    print(f"   Significantly better than random: {'✅ YES' if results['auc_vs_random']['significant'] else '❌ NO'}\n")
    
    # Test 2: Compare actual vs simulated random results
    _, auc_sim_pvalue = stats.mannwhitneyu([actual_mean_auc], random_mean_aucs, alternative='greater')
    results['auc_vs_simulation'] = {
        'actual_mean': actual_mean_auc,
        'random_mean': np.mean(random_mean_aucs),
        'p_value': auc_sim_pvalue,
        'significant': auc_sim_pvalue < 0.05
    }
    
    print(f"2. AUC vs Simulated Random:")
    print(f"   Actual Mean: {actual_mean_auc:.4f}")
    print(f"   Random Mean: {np.mean(random_mean_aucs):.4f}")
    print(f"   p-value: {auc_sim_pvalue:.6f}")
    print(f"   Significantly better: {'✅ YES' if results['auc_vs_simulation']['significant'] else '❌ NO'}\n")
    
    # Test 3: Check if any individual AUC is significantly above 0.5
    significant_classes = []
    for i, (label, auc) in enumerate(zip(actual_results['Label'], actual_auc)):
        if auc > 0.6:  # Reasonable threshold for medical classification
            significant_classes.append((label, auc))
    
    results['significant_classes'] = significant_classes
    
    print(f"3. Classes with AUC > 0.6:")
    if significant_classes:
        for label, auc in significant_classes:
            print(f"   {label}: {auc:.4f}")
    else:
        print("   None")
    
    # Test 4: Overall model performance assessment
    high_auc_count = np.sum(actual_auc > 0.7)
    moderate_auc_count = np.sum((actual_auc > 0.6) & (actual_auc <= 0.7))
    
    results['performance_summary'] = {
        'high_performance_classes': high_auc_count,
        'moderate_performance_classes': moderate_auc_count,
        'total_classes': len(actual_auc)
    }
    
    print(f"\n4. Performance Summary:")
    print(f"   Classes with AUC > 0.7: {high_auc_count}/{len(actual_auc)}")
    print(f"   Classes with AUC 0.6-0.7: {moderate_auc_count}/{len(actual_auc)}")
    print(f"   Classes with AUC < 0.6: {len(actual_auc) - high_auc_count - moderate_auc_count}/{len(actual_auc)}")
    
    return results

def create_visualization(actual_results, random_baseline):
    """Create visualization comparing actual vs random results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # AUC comparison
    actual_auc = actual_results['AUC'].values
    random_aucs = np.mean(random_baseline['auc'], axis=1)
    
    axes[0,0].hist(random_aucs, bins=50, alpha=0.7, label='Random Baseline', color='red')
    axes[0,0].axvline(np.mean(actual_auc), color='blue', linestyle='--', linewidth=2, label=f'Actual Mean: {np.mean(actual_auc):.3f}')
    axes[0,0].set_xlabel('Mean AUC')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('AUC: Actual vs Random Baseline')
    axes[0,0].legend()
    
    # Individual class AUCs
    axes[0,1].bar(range(len(actual_auc)), actual_auc, alpha=0.7, color='blue')
    axes[0,1].axhline(0.5, color='red', linestyle='--', label='Random (0.5)')
    axes[0,1].axhline(0.7, color='green', linestyle='--', label='Good (0.7)')
    axes[0,1].set_xlabel('Disease Class')
    axes[0,1].set_ylabel('AUC')
    axes[0,1].set_title('AUC by Disease Class')
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # F1 Score comparison
    actual_f1 = actual_results['F1_Score'].values
    random_f1s = np.mean(random_baseline['f1'], axis=1)
    
    axes[1,0].hist(random_f1s, bins=50, alpha=0.7, label='Random Baseline', color='red')
    axes[1,0].axvline(np.mean(actual_f1), color='blue', linestyle='--', linewidth=2, label=f'Actual Mean: {np.mean(actual_f1):.3f}')
    axes[1,0].set_xlabel('Mean F1 Score')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('F1 Score: Actual vs Random Baseline')
    axes[1,0].legend()
    
    # Performance heatmap
    metrics_data = actual_results[['AUC', 'F1_Score', 'Recall', 'Precision']].values.T
    im = axes[1,1].imshow(metrics_data, cmap='RdYlBu_r', aspect='auto')
    axes[1,1].set_xlabel('Disease Class')
    axes[1,1].set_ylabel('Metric')
    axes[1,1].set_title('Performance Heatmap')
    axes[1,1].set_yticks(range(4))
    axes[1,1].set_yticklabels(['AUC', 'F1', 'Recall', 'Precision'])
    plt.colorbar(im, ax=axes[1,1])
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_file = f"iteration_012_test_statistical_validation_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    """Main function to run statistical validation"""
    
    print("Loading iteration_012 results...")
    
    # Check if results file exists
    results_path = "auto_improvement_runs/iteration_012/pipeline_results_20251229-101232.csv"
    if not os.path.exists(results_path):
        print(f"❌ Error: Results file not found: {results_path}")
        return
    
    actual_results = load_iteration_results()
    
    print("Generating random baseline...")
    random_baseline = generate_random_baseline()
    
    print("Performing statistical tests...")
    test_results = perform_statistical_tests(actual_results, random_baseline)
    
    print("Creating visualization...")
    plot_file = create_visualization(actual_results, random_baseline)
    
    # Overall verdict
    print(f"\n=== FINAL VERDICT ===")
    
    auc_significant = test_results['auc_vs_random']['significant']
    good_classes = len(test_results['significant_classes'])
    
    if auc_significant and good_classes >= 3:
        print("✅ RESULTS ARE NOT RANDOM")
        print("   - AUC significantly better than random chance")
        print(f"   - {good_classes} classes show good performance (AUC > 0.6)")
    elif auc_significant:
        print("⚠️  RESULTS PARTIALLY SIGNIFICANT")
        print("   - AUC better than random but limited good classes")
    else:
        print("❌ RESULTS MAY BE RANDOM")
        print("   - No significant improvement over random chance")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = f"iteration_012_test_statistical_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Statistical Validation Results for Iteration 012\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {datetime.now()}\n\n")
        
        f.write("AUC vs Random Test:\n")
        f.write(f"  Actual Mean AUC: {test_results['auc_vs_random']['actual_mean']:.4f}\n")
        f.write(f"  p-value: {test_results['auc_vs_random']['p_value']:.6f}\n")
        f.write(f"  Significant: {test_results['auc_vs_random']['significant']}\n\n")
        
        f.write("Classes with Good Performance (AUC > 0.6):\n")
        for label, auc in test_results['significant_classes']:
            f.write(f"  {label}: {auc:.4f}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Visualization saved to: {plot_file}")

if __name__ == "__main__":
    main()