"""
Retry AI Advisory for a specific iteration
Re-runs only the AI analysis part without retraining the model
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from ai_advisor import AIAdvisor

def retry_ai_advisory(iteration: int, output_dir: str = "auto_improvement_runs"):
    """
    Retry AI advisory for a specific iteration

    Args:
        iteration: Iteration number to retry
        output_dir: Base output directory
    """

    iteration_dir = Path(output_dir) / f"iteration_{iteration:03d}"

    print("=" * 70)
    print(f"RETRYING AI ADVISORY FOR ITERATION {iteration}")
    print("=" * 70)

    # 1. Check if iteration directory exists
    if not iteration_dir.exists():
        print(f"❌ Error: Iteration directory not found: {iteration_dir}")
        return False

    print(f"✓ Found iteration directory: {iteration_dir}")

    # 2. Load iteration summary
    summary_file = iteration_dir / "iteration_summary.json"
    if not summary_file.exists():
        print(f"❌ Error: iteration_summary.json not found")
        return False

    with open(summary_file, 'r') as f:
        iteration_data = json.load(f)

    print(f"✓ Loaded iteration summary")

    # 3. Load results
    results_files = list(iteration_dir.glob("pipeline_results_*.csv"))
    if not results_files:
        print(f"❌ Error: No results CSV file found")
        return False

    results_df = pd.read_csv(results_files[0])
    print(f"✓ Loaded results: {results_files[0].name}")

    # 4. Load baseline comparison
    comparison_files = list(iteration_dir.glob("baseline_comparison_*.csv"))
    if not comparison_files:
        print(f"❌ Error: No baseline comparison CSV found")
        return False

    comparison_df = pd.read_csv(comparison_files[0])
    print(f"✓ Loaded comparison: {comparison_files[0].name}")

    # 5. Load config
    config_file = iteration_dir / "config.yaml"
    if not config_file.exists():
        print(f"❌ Error: config.yaml not found")
        return False

    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Loaded config")

    # 6. Load iteration history (from parent directory)
    history_file = Path(output_dir) / "iteration_history.json"
    iteration_history = []

    if history_file.exists():
        with open(history_file, 'r') as f:
            all_history = json.load(f)
            # Get history up to (but not including) current iteration
            iteration_history = [h for h in all_history if h['iteration'] < iteration]

    print(f"✓ Loaded iteration history: {len(iteration_history)} previous iterations")

    # 7. Get train/val loss from summary
    train_loss = iteration_data.get('train_loss')
    val_loss = iteration_data.get('val_loss')

    print(f"✓ Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    # 8. Load task info if available
    task_registry_file = Path(output_dir) / "tasks_registry.json"
    task_info = None

    if task_registry_file.exists():
        with open(task_registry_file, 'r') as f:
            task_registry = json.load(f)

        # Get current task
        current_task = None
        for task in task_registry.get('tasks', []):
            if task['status'] == 'active':
                current_task = task
                break

        task_info = {
            'current_task': current_task,
            'completed_tasks': [t for t in task_registry.get('tasks', []) if t['status'] in ['succeeded', 'failed']]
        }

        print(f"✓ Loaded task registry")

    # 9. Create iteration analysis (last 3 iterations)
    iteration_analysis = None
    if len(iteration_history) >= 1:
        recent_history = iteration_history[-3:]  # Last 3

        iteration_analysis = {
            'f1_comparison': {
                'values': [h.get('avg_f1', 0) for h in recent_history],
                'best_value': max([h.get('avg_f1', 0) for h in iteration_history]),
                'best_iteration': max(iteration_history, key=lambda x: x.get('avg_f1', 0))['iteration'],
                'trend': 'improving' if len(recent_history) >= 2 and recent_history[-1]['avg_f1'] > recent_history[0]['avg_f1'] else 'declining',
                'improvement': recent_history[-1]['avg_f1'] - recent_history[0]['avg_f1'] if len(recent_history) >= 2 else 0
            },
            'auc_comparison': {
                'values': [h.get('avg_auc', 0) for h in recent_history],
                'best_value': max([h.get('avg_auc', 0) for h in iteration_history]),
                'best_iteration': max(iteration_history, key=lambda x: x.get('avg_auc', 0))['iteration'],
                'trend': 'improving' if len(recent_history) >= 2 and recent_history[-1]['avg_auc'] > recent_history[0]['avg_auc'] else 'declining',
                'improvement': recent_history[-1]['avg_auc'] - recent_history[0]['avg_auc'] if len(recent_history) >= 2 else 0
            },
            'recall_comparison': {
                'values': [h.get('avg_recall', 0) for h in recent_history],
                'best_value': max([h.get('avg_recall', 0) for h in iteration_history]),
                'best_iteration': max(iteration_history, key=lambda x: x.get('avg_recall', 0))['iteration'],
                'trend': 'improving' if len(recent_history) >= 2 and recent_history[-1]['avg_recall'] > recent_history[0]['avg_recall'] else 'declining',
                'improvement': recent_history[-1]['avg_recall'] - recent_history[0]['avg_recall'] if len(recent_history) >= 2 else 0
            }
        }

        print(f"✓ Created iteration trend analysis")

    # 10. Initialize AI Advisor with Claude fallback
    print("\n" + "-" * 70)
    print("Initializing AI Advisor...")
    print("-" * 70)

    advisor = AIAdvisor(model="gpt-5.1", use_claude_fallback=True)

    if advisor.claude_client:
        print("✅ AI Advisor initialized with Claude fallback enabled")
    else:
        print("⚠️  AI Advisor initialized (Claude fallback not available)")

    # 11. Run AI analysis
    print("\n" + "-" * 70)
    print("Running AI Analysis...")
    print("-" * 70)
    print("(This may take 30-60 seconds...)")
    print()

    try:
        analysis, suggested_changes = advisor.analyze_results(
            config=config,
            results_df=results_df,
            comparison_df=comparison_df,
            iteration=iteration,
            iteration_history=iteration_history,
            train_loss=train_loss,
            val_loss=val_loss,
            task_info=task_info,
            iteration_analysis=iteration_analysis
        )

        print("\n" + "=" * 70)
        print("✅ AI ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        # 12. Save analysis
        analysis_file = iteration_dir / f"ai_analysis_iteration_{iteration:03d}.txt"
        with open(analysis_file, 'w') as f:
            f.write(analysis)

        print(f"\n✓ Saved analysis to: {analysis_file}")

        # 13. Save suggested changes
        changes_file = iteration_dir / f"suggested_changes_{iteration:03d}.json"
        with open(changes_file, 'w') as f:
            json.dump(suggested_changes, indent=2, fp=f)

        print(f"✓ Saved changes to: {changes_file}")

        # 14. Update iteration summary
        iteration_data['ai_analysis_completed'] = True
        iteration_data['suggested_changes'] = suggested_changes

        with open(summary_file, 'w') as f:
            json.dump(iteration_data, indent=2, fp=f)

        print(f"✓ Updated iteration summary")

        # 15. Show preview of analysis
        print("\n" + "-" * 70)
        print("ANALYSIS PREVIEW:")
        print("-" * 70)
        print(analysis[:500] + "...\n")

        print("-" * 70)
        print("SUGGESTED CHANGES:")
        print("-" * 70)
        print(json.dumps(suggested_changes, indent=2))
        print()

        return True

    except Exception as e:
        print(f"\n❌ AI Analysis failed again: {e}")

        # Save error
        error_file = iteration_dir / f"ai_analysis_error_{iteration:03d}_retry.txt"
        with open(error_file, 'w') as f:
            f.write(f"AI Analysis Retry Failed\n")
            f.write(f"Error: {e}\n\n")
            import traceback
            f.write(traceback.format_exc())

        print(f"✓ Saved error to: {error_file}")

        return False

def main():
    parser = argparse.ArgumentParser(description="Retry AI advisory for a specific iteration")
    parser.add_argument("iteration", type=int, help="Iteration number to retry")
    parser.add_argument("--output-dir", default="auto_improvement_runs", help="Output directory")

    args = parser.parse_args()

    success = retry_ai_advisory(args.iteration, args.output_dir)

    if success:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print("=" * 70)
        print(f"\nIteration {args.iteration} AI advisory completed.")
        print(f"You can now continue with new iterations:")
        print(f"  uv run python auto_improvement_loop.py --resume --iterations 5")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ FAILED")
        print("=" * 70)
        print(f"\nAI advisory retry failed for iteration {args.iteration}")
        print("Check the error file in the iteration directory for details.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())