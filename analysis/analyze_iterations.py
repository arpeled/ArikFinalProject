"""
Analyze iteration performance trends
"""
import os
import pandas as pd
import json
from pathlib import Path

output_dir = "experiments/auto_improvement_runs"

print("="*80)
print("ITERATION PERFORMANCE ANALYSIS")
print("="*80)
print()

iterations_data = []

for i in range(1, 11):
    iter_dir = os.path.join(output_dir, f"iteration_{i:03d}")

    if not os.path.exists(iter_dir):
        continue

    print(f"Iteration {i}:")
    print("-" * 40)

    # Try to load from JSON summary first
    summary_file = os.path.join(iter_dir, "iteration_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        avg_auc = summary.get('avg_auc', 0)
        avg_f1 = summary.get('avg_f1', 0)
        avg_recall = summary.get('avg_recall', 0)
        avg_precision = summary.get('avg_precision', 0)
        ai_status = summary.get('ai_analysis_status', 'unknown')

        print(f"  AUC:       {avg_auc:.4f}")
        print(f"  F1:        {avg_f1:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  AI Status: {ai_status}")

        iterations_data.append({
            'iteration': i,
            'avg_auc': avg_auc,
            'avg_f1': avg_f1,
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'ai_status': ai_status
        })
    else:
        # Fallback to CSV
        csv_files = list(Path(iter_dir).glob("pipeline_results_*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            avg_auc = df['AUC'].mean()
            avg_f1 = df.get('F1_Score', pd.Series([0])).mean()
            avg_recall = df.get('Recall', pd.Series([0])).mean()
            avg_precision = df.get('Precision', pd.Series([0])).mean()

            print(f"  AUC:       {avg_auc:.4f}")
            print(f"  F1:        {avg_f1:.4f}")
            print(f"  Recall:    {avg_recall:.4f}")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  AI Status: reconstructed from CSV")

            iterations_data.append({
                'iteration': i,
                'avg_auc': avg_auc,
                'avg_f1': avg_f1,
                'avg_recall': avg_recall,
                'avg_precision': avg_precision,
                'ai_status': 'unknown'
            })
        else:
            print("  No data available")

    print()

# Analyze trends
if len(iterations_data) > 1:
    print("="*80)
    print("TREND ANALYSIS")
    print("="*80)
    print()

    best_iter = max(iterations_data, key=lambda x: x['avg_auc'])
    worst_iter = min(iterations_data, key=lambda x: x['avg_auc'])

    print(f"Best Iteration:  #{best_iter['iteration']} - AUC: {best_iter['avg_auc']:.4f}")
    print(f"Worst Iteration: #{worst_iter['iteration']} - AUC: {worst_iter['avg_auc']:.4f}")
    print()

    # Check if performance is degrading
    first = iterations_data[0]
    last = iterations_data[-1]

    auc_change = last['avg_auc'] - first['avg_auc']
    f1_change = last['avg_f1'] - first['avg_f1']
    recall_change = last['avg_recall'] - first['avg_recall']

    print(f"Change from first to last iteration:")
    print(f"  AUC:    {auc_change:+.4f} ({auc_change/first['avg_auc']*100:+.2f}%)")
    print(f"  F1:     {f1_change:+.4f}")
    print(f"  Recall: {recall_change:+.4f}")
    print()

    # Check monotonicity
    auc_values = [x['avg_auc'] for x in iterations_data]
    degrading_count = sum(1 for i in range(1, len(auc_values)) if auc_values[i] < auc_values[i-1])
    improving_count = sum(1 for i in range(1, len(auc_values)) if auc_values[i] > auc_values[i-1])

    print(f"Performance transitions:")
    print(f"  Improved: {improving_count} times")
    print(f"  Degraded: {degrading_count} times")
    print()

    if degrading_count > improving_count:
        print("⚠️  WARNING: Performance is mostly degrading!")
        print()
        print("Possible causes:")
        print("  1. Overfitting to validation set")
        print("  2. AI making too aggressive changes")
        print("  3. Training instability (need to check loss curves)")
        print("  4. Hyperparameters moving away from good regions")
        print("  5. Random seed changes causing variance")
