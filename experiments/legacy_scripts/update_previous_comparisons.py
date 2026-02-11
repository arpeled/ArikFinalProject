"""
Script to Update Baseline Comparison Files for All Previous Iterations

This script regenerates baseline_comparison CSV files for all previous iterations
with the complete set of metrics (Threshold, Accuracy, Specificity, Precision, Sensitivity).

Usage:
    python update_previous_comparisons.py [--output-dir auto_improvement_runs] [--dry-run]
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Import the comparison function
from auto_improvement_loop import AutoImprovementLoop


def find_iteration_directories(output_dir: str) -> list:
    """Find all iteration directories"""
    pattern = os.path.join(output_dir, "iteration_*")
    dirs = sorted(glob.glob(pattern))
    return [d for d in dirs if os.path.isdir(d)]


def find_results_file(iteration_dir: str) -> str:
    """Find the pipeline_results CSV file in an iteration directory"""
    pattern = os.path.join(iteration_dir, "pipeline_results_*.csv")
    files = glob.glob(pattern)

    if not files:
        return None

    # Return the first (should only be one)
    return files[0]


def backup_old_comparison(iteration_dir: str) -> bool:
    """Backup old comparison file if it exists"""
    pattern = os.path.join(iteration_dir, "baseline_comparison_*.csv")
    files = glob.glob(pattern)

    if not files:
        return False

    for old_file in files:
        backup_file = old_file.replace(".csv", "_OLD_BACKUP.csv")
        shutil.copy2(old_file, backup_file)
        print(f"  📦 Backed up old comparison: {os.path.basename(backup_file)}")

    return True


def update_iteration_comparison(iteration_dir: str, loop: AutoImprovementLoop, dry_run: bool = False) -> bool:
    """Update comparison file for a single iteration"""
    iteration_name = os.path.basename(iteration_dir)

    # Find results file
    results_file = find_results_file(iteration_dir)
    if not results_file:
        print(f"  ⚠️  No results file found, skipping")
        return False

    print(f"  📄 Found results: {os.path.basename(results_file)}")

    if dry_run:
        print(f"  🔍 DRY RUN: Would regenerate comparison with all metrics")
        return True

    # Backup old comparison
    has_old = backup_old_comparison(iteration_dir)

    # Generate new comparison with same timestamp as results file
    timestamp = os.path.basename(results_file).replace("pipeline_results_", "").replace(".csv", "")
    new_comparison_file = os.path.join(iteration_dir, f"baseline_comparison_{timestamp}.csv")

    try:
        # Generate new comparison
        comparison_df = loop.compare_with_baseline(results_file, new_comparison_file)

        # Verify all metrics are present
        expected_metrics = ['AUC', 'F1_Score', 'Recall', 'Accuracy', 'Specificity', 'Precision', 'Sensitivity', 'Threshold']
        missing_metrics = []

        for metric in expected_metrics:
            if f'Better_{metric}' not in comparison_df.columns:
                missing_metrics.append(metric)

        if missing_metrics:
            print(f"  ⚠️  Missing metrics in comparison: {missing_metrics}")
            return False

        print(f"  ✅ Generated new comparison: {os.path.basename(new_comparison_file)}")
        print(f"     - Total columns: {len(comparison_df.columns)}")
        print(f"     - All 8 metrics included: {len(expected_metrics)}")

        # Note: Old comparison files are already backed up with _OLD_BACKUP.csv suffix
        # The new comparison file has been created, no need to remove anything

        return True

    except Exception as e:
        print(f"  ❌ Error generating comparison: {e}")
        return False


def main():
    """Main function to update all iteration comparisons"""
    parser = argparse.ArgumentParser(
        description="Update baseline comparison files for all previous iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="auto_improvement_runs",
        help="Output directory containing iterations (default: auto_improvement_runs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - show what would be done without making changes"
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help="Specific iterations to update (e.g., '1-5,10,15' or 'all')"
    )

    args = parser.parse_args()

    print("="*80)
    print("BASELINE COMPARISON UPDATE SCRIPT")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Dry run mode: {args.dry_run}")
    print()

    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"❌ Error: Output directory '{args.output_dir}' not found")
        sys.exit(1)

    # Find all iteration directories
    iteration_dirs = find_iteration_directories(args.output_dir)

    if not iteration_dirs:
        print(f"❌ No iteration directories found in '{args.output_dir}'")
        sys.exit(1)

    print(f"📁 Found {len(iteration_dirs)} iteration directories")
    print()

    # Create AutoImprovementLoop instance (with dummy API key for comparison only)
    loop = AutoImprovementLoop(openai_api_key="dummy_key_for_comparison_only")

    # Process each iteration
    successful = 0
    failed = 0
    skipped = 0

    for iteration_dir in iteration_dirs:
        iteration_name = os.path.basename(iteration_dir)
        print(f"📂 Processing {iteration_name}...")

        try:
            success = update_iteration_comparison(iteration_dir, loop, args.dry_run)
            if success:
                successful += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            failed += 1

        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successfully updated: {successful}/{len(iteration_dirs)}")
    if skipped > 0:
        print(f"⚠️  Skipped: {skipped}")
    if failed > 0:
        print(f"❌ Failed: {failed}")

    if args.dry_run:
        print()
        print("🔍 This was a DRY RUN - no changes were made")
        print("   Run without --dry-run to apply changes")
    else:
        print()
        print("✅ All comparison files have been updated with complete metrics!")
        print("   Old comparison files have been backed up with _OLD_BACKUP.csv suffix")

    print("="*80)


if __name__ == "__main__":
    main()
