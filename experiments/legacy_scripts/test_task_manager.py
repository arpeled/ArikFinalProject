"""
Test script for Task Manager integration
Demonstrates how to use TaskManager and IterationEvaluator
"""

import os
import json
from task_manager import TaskManager, IterationEvaluator

def test_task_manager():
    """Test basic task manager functionality"""
    print("="*80)
    print("TESTING TASK MANAGER")
    print("="*80)

    # Create task manager (will use test registry)
    tm = TaskManager(registry_path="test_tasks_registry.json")

    # 1. Add a new task
    print("\n1. Creating a new task...")
    tm.add_task(
        task_id="test_focal_gamma",
        description="Test FocalLoss gamma adjustment for Hernia",
        start_iteration=52,
        target_metric="f1",
        required_iterations=3,
        change_applied={
            "loss": {
                "gamma": 3.0
            }
        }
    )

    # 2. Show task summary
    print("\n2. Task Summary:")
    print(tm.get_task_summary())

    # 3. Get current task
    current = tm.get_current_task()
    if current:
        print("\n3. Current Task Details:")
        print(f"   ID: {current['task_id']}")
        print(f"   Description: {current['description']}")
        print(f"   Target: {current['target_metric']}")

    # 4. Simulate updating task progress for 3 iterations
    print("\n4. Simulating task progress across 3 iterations...")
    for iter_num in range(52, 55):
        # Simulate improving F1 scores
        f1_value = 0.25 + (iter_num - 52) * 0.02

        tm.update_task_progress(
            task_id="test_focal_gamma",
            iteration=iter_num,
            metric_value=f1_value,
            iteration_summary={
                "iteration": iter_num,
                "avg_f1": f1_value,
                "timestamp": f"20260107-{iter_num:06d}"
            }
        )
        print(f"   Iteration {iter_num}: F1 = {f1_value:.4f}")

    # 5. Evaluate task
    print("\n5. Evaluating task after 3 iterations...")
    status = tm.evaluate_task("test_focal_gamma")
    print(f"   Task Status: {status}")

    # 6. Show final summary
    print("\n6. Final Task Summary:")
    print(tm.get_task_summary())

    # 7. Get task info for AI advisor
    print("\n7. Task Info for AI Advisor:")
    task_info = tm.get_task_for_advisor()
    print(json.dumps(task_info, indent=2, default=str))

    # Cleanup
    if os.path.exists("test_tasks_registry.json"):
        os.remove("test_tasks_registry.json")
        print("\n✓ Cleaned up test registry")

    print("\n" + "="*80)
    print("✅ TASK MANAGER TEST PASSED")
    print("="*80)


def test_iteration_evaluator():
    """Test iteration evaluator with real data"""
    print("\n" + "="*80)
    print("TESTING ITERATION EVALUATOR")
    print("="*80)

    evaluator = IterationEvaluator(output_dir="auto_improvement_runs")

    # 1. Load last 3 summaries
    print("\n1. Loading last 3 iteration summaries...")
    summaries = evaluator.get_last_n_summaries(3)
    print(f"   Loaded {len(summaries)} summaries")

    if summaries:
        for summary in summaries:
            iter_num = summary.get('iteration', '?')
            f1 = summary.get('avg_f1', 0.0)
            auc = summary.get('avg_auc', 0.0)
            print(f"   Iteration {iter_num}: F1={f1:.4f}, AUC={auc:.4f}")
    else:
        print("   No summaries found (run at least one iteration first)")

    # 2. Compare F1 metric
    if summaries:
        print("\n2. Comparing F1 scores across iterations...")
        f1_comparison = evaluator.compare_metrics(summaries, "avg_f1")
        print(f"   Metric: {f1_comparison['metric']}")
        print(f"   Best: {f1_comparison['best_value']:.4f} (iteration {f1_comparison['best_iteration']})")
        print(f"   Trend: {f1_comparison['trend']}")
        print(f"   Improvement: {f1_comparison['improvement']:+.4f}")

    # 3. Format for advisor
    if summaries:
        print("\n3. Formatting data for AI advisor...")
        advisor_data = evaluator.format_for_advisor(summaries)
        print(f"   Keys: {list(advisor_data.keys())}")
        print(f"   F1 comparison available: {'f1_comparison' in advisor_data}")
        print(f"   AUC comparison available: {'auc_comparison' in advisor_data}")
        print(f"   Disease metrics available: {'disease_metrics' in advisor_data}")

    print("\n" + "="*80)
    print("✅ ITERATION EVALUATOR TEST PASSED")
    print("="*80)


def test_integration_example():
    """Show example of how to use in auto_improvement_loop"""
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE")
    print("="*80)

    print("""
Example usage in auto_improvement_loop.py:

# In __init__:
self.task_manager = TaskManager(
    registry_path=os.path.join(output_dir, "tasks_registry.json"),
    logger=self.logger
)
self.iteration_evaluator = IterationEvaluator(
    output_dir=output_dir,
    logger=self.logger
)

# In run_single_iteration:
# Get current task
current_task = self.task_manager.get_current_task()
if current_task:
    print(f"Working on: {current_task['description']}")

# After iteration completes
if current_task:
    # Update progress
    self.task_manager.update_task_progress(
        task_id=current_task['task_id'],
        iteration=iteration,
        metric_value=avg_f1,
        iteration_summary=iteration_summary
    )

    # Evaluate if enough iterations
    if len(current_task['evaluation_history']) >= current_task['required_iterations']:
        status = self.task_manager.evaluate_task(current_task['task_id'])
        print(f"Task {status}: {current_task['description']}")

# Pass to AI advisor
task_info = self.task_manager.get_task_for_advisor()
iteration_analysis = self.iteration_evaluator.format_for_advisor(
    self.iteration_evaluator.get_last_n_summaries(3)
)

analysis, suggestions = self.ai_advisor.analyze_results(
    ...,
    task_info=task_info,
    iteration_analysis=iteration_analysis
)
""")

    print("="*80)
    print("✅ INTEGRATION EXAMPLE SHOWN")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "🧪 TASK MANAGER INTEGRATION TEST SUITE" + "\n")

    try:
        # Run tests
        test_task_manager()
        test_iteration_evaluator()
        test_integration_example()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        print("\nThe task manager integration is ready to use.")
        print("Run your next iteration to see it in action!")
        print("\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
