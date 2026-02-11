#!/usr/bin/env python3
"""
Complete Iteration 30 Recovery
Generates baseline comparison and iteration summary.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def generate_baseline_comparison(results_file, comparison_file, baseline_file='baseline_results.csv'):
    """Generate baseline comparison"""

    # Load results
    our_results = pd.read_csv(results_file)

    # Load baseline
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
    else:
        print(f"Warning: Baseline file not found: {baseline_file}")
        baseline_df = None

    comparison_data = []

    for _, row in our_results.iterrows():
        label = row['Label']

        # Get baseline metrics
        if baseline_df is not None and label in baseline_df['Label'].values:
            baseline_row = baseline_df[baseline_df['Label'] == label].iloc[0]
            baseline_metrics = {
                'AUC': baseline_row.get('AUC', np.nan),
                'F1_Score': baseline_row.get('F1_Score', np.nan),
                'Recall': baseline_row.get('Recall', np.nan),
                'Accuracy': baseline_row.get('Accuracy', np.nan),
                'Specificity': baseline_row.get('Specificity', np.nan),
                'Precision': baseline_row.get('Precision', np.nan),
                'Sensitivity': baseline_row.get('Sensitivity', np.nan),
                'Threshold': baseline_row.get('Threshold', 0.5)
            }
        else:
            baseline_metrics = {k: np.nan for k in ['AUC', 'F1_Score', 'Recall', 'Accuracy', 'Specificity', 'Precision', 'Sensitivity', 'Threshold']}

        def calc_improvement(ours, baseline):
            if pd.isna(baseline) or pd.isna(ours) or baseline == 0:
                return 0.0, ''
            improvement = ((ours - baseline) / abs(baseline)) * 100
            better = '✓' if improvement > 0 else ('✗' if improvement < 0 else '=')
            return improvement, better

        # Calculate improvements
        metrics = ['AUC', 'F1_Score', 'Recall', 'Accuracy', 'Specificity', 'Precision', 'Sensitivity', 'Threshold']
        comparison_row = {'Label': label}

        for metric in metrics:
            our_value = row.get(metric, np.nan)
            baseline_value = baseline_metrics.get(metric, np.nan)
            improvement, better = calc_improvement(our_value, baseline_value)

            comparison_row[f'{metric}_Baseline'] = baseline_value
            comparison_row[f'{metric}_Ours'] = our_value
            comparison_row[f'{metric}_Improvement'] = improvement
            comparison_row[f'{metric}_Better'] = better

        comparison_data.append(comparison_row)

    # Save comparison
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ Baseline comparison saved: {comparison_file}")

    return comparison_df


def generate_iteration_summary(iteration_dir, iteration=30):
    """Generate iteration summary JSON"""

    results_file = os.path.join(iteration_dir, 'pipeline_results_20260102-082519.csv')

    # Load results
    results_df = pd.read_csv(results_file)

    summary = {
        'iteration': iteration,
        'timestamp': '20260102-082519',
        'status': 'recovered',
        'avg_auc': float(results_df['AUC'].mean()),
        'avg_f1': float(results_df['F1_Score'].mean()),
        'avg_recall': float(results_df['Recall'].mean()),
        'avg_precision': float(results_df['Precision'].mean()),
        'avg_accuracy': float(results_df['Accuracy'].mean()),
        'avg_specificity': float(results_df['Specificity'].mean()),
        'avg_sensitivity': float(results_df['Sensitivity'].mean()),
        'recovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'recovery_note': 'Iteration failed during threshold optimization. Model and test results recovered manually.',
        'ai_analysis_status': 'not_available',
        'threshold_optimization': False,
        'files': {
            'model': 'pipeline_model_20260102-082519.pth',
            'results': 'pipeline_results_20260102-082519.csv',
            'confusion_matrix': 'confusion_matrix_20260102-082519.json',
            'baseline_comparison': 'baseline_comparison_20260102-082519.csv',
            'config': 'config.yaml'
        }
    }

    # Save summary
    summary_file = os.path.join(iteration_dir, 'iteration_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Iteration summary saved: {summary_file}")
    return summary


def main():
    iteration_dir = 'auto_improvement_runs/iteration_030'
    results_file = os.path.join(iteration_dir, 'pipeline_results_20260102-082519.csv')
    comparison_file = os.path.join(iteration_dir, 'baseline_comparison_20260102-082519.csv')

    print("="*80)
    print("COMPLETING ITERATION 30 RECOVERY")
    print("="*80)
    print()

    # Generate baseline comparison
    print("Generating baseline comparison...")
    comparison_df = generate_baseline_comparison(results_file, comparison_file)
    print()

    # Generate iteration summary
    print("Generating iteration summary...")
    summary = generate_iteration_summary(iteration_dir)
    print()

    print("="*80)
    print("✅ ITERATION 30 FULLY RECOVERED")
    print("="*80)
    print()
    print(f"📁 Location: {iteration_dir}/")
    print()
    print("📊 Summary:")
    print(f"   Average AUC:         {summary['avg_auc']:.4f}")
    print(f"   Average F1-Score:    {summary['avg_f1']:.4f}")
    print(f"   Average Recall:      {summary['avg_recall']:.4f}")
    print(f"   Average Precision:   {summary['avg_precision']:.4f}")
    print(f"   Average Accuracy:    {summary['avg_accuracy']:.4f}")
    print()
    print("📋 Files:")
    for file in os.listdir(iteration_dir):
        if not file.startswith('.'):
            size = os.path.getsize(os.path.join(iteration_dir, file))
            print(f"   ✓ {file} ({size:,} bytes)")
    print()
    print("="*80)
    print()
    print("⚠️  Note: Threshold optimization was not applied (iteration failed before that step).")
    print("   Results use default 0.5 threshold for all classes.")
    print("   This explains the poor F1/Recall scores (Hernia issue shows threshold problem).")
    print()

if __name__ == "__main__":
    main()
