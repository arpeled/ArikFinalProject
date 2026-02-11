#!/usr/bin/env python3
"""
View Confusion Matrix from Iteration Results
Reads confusion_matrix_*.json files and displays them in a readable format
"""

import json
import argparse
import os
from pathlib import Path


def find_confusion_matrix_file(iteration_num):
    """Find the confusion matrix JSON file for a given iteration."""
    # Try iteration directory first
    iter_dir = Path(f"experiments/auto_improvement_runs/iteration_{iteration_num:03d}")
    if iter_dir.exists():
        json_files = list(iter_dir.glob("confusion_matrix_*.json"))
        if json_files:
            return json_files[0]

    # Try current directory
    json_files = list(Path(".").glob(f"confusion_matrix_*{iteration_num:03d}*.json"))
    if json_files:
        return json_files[0]

    return None


def print_confusion_matrix(confusion_dict, show_rates=True):
    """Print confusion matrix in a formatted table."""

    print("\n" + "="*110)
    print("CONFUSION MATRIX - Per Disease")
    print("="*110)

    # Header
    if show_rates:
        print(f"{'Disease':<20} {'Prev%':>6} | {'TP':>6} {'FP':>6} {'TN':>7} {'FN':>6} | {'TPR':>8} {'FPR':>8} {'PPV':>8} {'NPV':>8}")
    else:
        print(f"{'Disease':<20} {'Prev%':>6} | {'TP':>6} {'FP':>6} {'TN':>7} {'FN':>6} | {'Total+':>7} {'Total-':>7} {'Pred+':>7} {'Pred-':>7}")
    print("-"*110)

    # Sort by disease name
    sorted_diseases = sorted(confusion_dict.keys())

    total_tp = total_fp = total_tn = total_fn = 0

    for disease in sorted_diseases:
        data = confusion_dict[disease]
        tp = data['TP']
        fp = data['FP']
        tn = data['TN']
        fn = data['FN']

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        # Calculate prevalence percentage
        total_samples = tp + fp + tn + fn
        prevalence = ((tp + fn) / total_samples * 100) if total_samples > 0 else 0

        if show_rates:
            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity/Recall)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

            print(f"{disease:<20} {prevalence:>6.2f} | {tp:>6} {fp:>6} {tn:>7} {fn:>6} | {tpr:>8.4f} {fpr:>8.4f} {ppv:>8.4f} {npv:>8.4f}")
        else:
            total_pos = data['Total_Positive']
            total_neg = data['Total_Negative']
            pred_pos = data['Predicted_Positive']
            pred_neg = data['Predicted_Negative']

            print(f"{disease:<20} {prevalence:>6.2f} | {tp:>6} {fp:>6} {tn:>7} {fn:>6} | {total_pos:>7} {total_neg:>7} {pred_pos:>7} {pred_neg:>7}")

    print("-"*110)

    # Calculate overall rates and prevalence
    total_samples = total_tp + total_fp + total_tn + total_fn
    overall_prevalence = ((total_tp + total_fn) / total_samples * 100) if total_samples > 0 else 0
    overall_tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    overall_ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_npv = total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else 0

    if show_rates:
        print(f"{'OVERALL':<20} {overall_prevalence:>6.2f} | {total_tp:>6} {total_fp:>6} {total_tn:>7} {total_fn:>6} | {overall_tpr:>8.4f} {overall_fpr:>8.4f} {overall_ppv:>8.4f} {overall_npv:>8.4f}")
    else:
        print(f"{'OVERALL':<20} {overall_prevalence:>6.2f} | {total_tp:>6} {total_fp:>6} {total_tn:>7} {total_fn:>6}")

    print("="*110)

    # Print legend
    print("\nLegend:")
    print("  Prev% = Prevalence (percentage of positive cases in dataset)")
    print("  TP    = True Positives   (Correctly predicted as positive)")
    print("  FP    = False Positives  (Incorrectly predicted as positive)")
    print("  TN    = True Negatives   (Correctly predicted as negative)")
    print("  FN    = False Negatives  (Incorrectly predicted as negative)")
    if show_rates:
        print("\n  TPR = True Positive Rate  (Sensitivity/Recall) = TP / (TP + FN)")
        print("  FPR = False Positive Rate = FP / (FP + TN)")
        print("  PPV = Positive Predictive Value (Precision) = TP / (TP + FP)")
        print("  NPV = Negative Predictive Value = TN / (TN + FN)")
    else:
        print("\n  Total+ = Total actual positives (TP + FN)")
        print("  Total- = Total actual negatives (TN + FP)")
        print("  Pred+  = Total predicted positives (TP + FP)")
        print("  Pred-  = Total predicted negatives (TN + FN)")
    print()


def print_summary_stats(confusion_dict):
    """Print summary statistics."""
    print("\n" + "="*110)
    print("SUMMARY STATISTICS")
    print("="*110)

    total_tp = total_fp = total_tn = total_fn = 0

    for disease, data in confusion_dict.items():
        total_tp += data['TP']
        total_fp += data['FP']
        total_tn += data['TN']
        total_fn += data['FN']

    total_samples = (total_tp + total_fn)  # Total actual positives
    total_predictions = total_tp + total_fp + total_tn + total_fn

    accuracy = (total_tp + total_tn) / total_predictions if total_predictions > 0 else 0
    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"Total Predictions:  {total_predictions:,}")
    print(f"Total Positives:    {total_tp + total_fn:,}")
    print(f"Total Negatives:    {total_tn + total_fp:,}")
    print()
    print(f"Overall Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Overall Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Overall Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Overall Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"Overall F1-Score:    {f1_score:.4f} ({f1_score*100:.2f}%)")
    print("="*110)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View confusion matrix from iteration results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View confusion matrix from iteration 27
  python view_confusion_matrix.py 27

  # View with counts instead of rates
  python view_confusion_matrix.py 27 --no-rates

  # Load from specific file
  python view_confusion_matrix.py --file confusion_matrix_20260101-021417.json

  # Show summary statistics only
  python view_confusion_matrix.py 27 --summary-only
        """
    )

    parser.add_argument('iteration', type=int, nargs='?',
                       help='Iteration number (e.g., 27 for iteration_027)')
    parser.add_argument('--file', type=str,
                       help='Path to specific confusion_matrix_*.json file')
    parser.add_argument('--no-rates', action='store_true',
                       help='Show counts instead of rates (TPR, FPR, etc.)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary statistics, not full matrix')

    args = parser.parse_args()

    # Find the confusion matrix file
    if args.file:
        json_file = Path(args.file)
        if not json_file.exists():
            print(f"Error: File not found: {json_file}")
            return 1
    elif args.iteration:
        json_file = find_confusion_matrix_file(args.iteration)
        if not json_file:
            print(f"Error: No confusion matrix found for iteration {args.iteration}")
            print(f"Looked in: experiments/auto_improvement_runs/iteration_{args.iteration:03d}/")
            return 1
    else:
        print("Error: Must provide either iteration number or --file")
        parser.print_help()
        return 1

    # Load the JSON file
    print(f"\nLoading: {json_file}")

    try:
        with open(json_file, 'r') as f:
            confusion_dict = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return 1

    # Print results
    if not args.summary_only:
        print_confusion_matrix(confusion_dict, show_rates=not args.no_rates)

    print_summary_stats(confusion_dict)

    return 0


if __name__ == "__main__":
    exit(main())
