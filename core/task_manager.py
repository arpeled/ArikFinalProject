"""
Task Management System for Auto-Improvement Pipeline
Tracks tasks across iterations, evaluates their success, and manages task lifecycle
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path


class TaskManager:
    """
    Manages tasks in the auto-improvement pipeline.

    Tasks are stored in tasks_registry.json and track:
    - What change is being tested
    - Which iteration started it
    - How many iterations to evaluate
    - Target metric to optimize
    - Success/failure status
    """

    def __init__(self, registry_path: str = "tasks_registry.json", logger: Optional[logging.Logger] = None):
        """
        Initialize task manager

        Args:
            registry_path: Path to task registry JSON file
            logger: Optional logger instance
        """
        self.registry_path = registry_path
        self.logger = logger or self._create_logger()
        self.tasks = self._load_registry()

    def _create_logger(self):
        """Create default logger"""
        logger = logging.getLogger('task_manager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def _load_registry(self) -> Dict[str, Any]:
        """Load task registry from file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to load task registry: {e}")
                return {"tasks": [], "completed_tasks": []}
        else:
            return {"tasks": [], "completed_tasks": []}

    def _save_registry(self):
        """Save task registry to file"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.tasks, f, indent=2)
        self.logger.debug(f"Task registry saved to {self.registry_path}")

    def add_task(
        self,
        task_id: str,
        description: str,
        start_iteration: int,
        target_metric: str = "f1",
        required_iterations: int = 3,
        change_applied: Optional[Dict[str, Any]] = None
    ):
        """
        Add a new task to the registry

        Args:
            task_id: Unique identifier for the task
            description: Human-readable description
            start_iteration: Iteration number when task starts
            target_metric: Metric to optimize (f1, auc, recall, etc.)
            required_iterations: Number of iterations to evaluate
            change_applied: Config changes for this task
        """
        task = {
            "task_id": task_id,
            "description": description,
            "start_iteration": start_iteration,
            "status": "pending",
            "target_metric": target_metric,
            "required_iterations": required_iterations,
            "change_applied": change_applied or {},
            "evaluation_history": []
        }

        self.tasks["tasks"].append(task)
        self._save_registry()
        self.logger.info(f"Added task: {task_id} - {description}")

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active task

        Returns:
            Current task dictionary or None if no active task
        """
        for task in self.tasks["tasks"]:
            if task["status"] == "pending":
                return task
        return None

    def update_task_progress(
        self,
        task_id: str,
        iteration: int,
        metric_value: float,
        iteration_summary: Dict[str, Any]
    ):
        """
        Update task progress with iteration results

        Args:
            task_id: Task identifier
            iteration: Current iteration number
            metric_value: Value of target metric for this iteration
            iteration_summary: Full iteration summary data
        """
        for task in self.tasks["tasks"]:
            if task["task_id"] == task_id:
                task["evaluation_history"].append({
                    "iteration": iteration,
                    "metric_value": metric_value,
                    "timestamp": iteration_summary.get("timestamp", ""),
                })
                self._save_registry()
                self.logger.info(
                    f"Task {task_id}: Iteration {iteration}, "
                    f"{task['target_metric']}={metric_value:.4f}"
                )
                break

    def evaluate_task(self, task_id: str) -> Optional[str]:
        """
        Evaluate if a task succeeded or failed based on evaluation history

        Args:
            task_id: Task identifier

        Returns:
            "succeeded", "failed", or None if not enough data
        """
        for task in self.tasks["tasks"]:
            if task["task_id"] == task_id:
                history = task["evaluation_history"]
                required = task["required_iterations"]

                # Need at least required_iterations to evaluate
                if len(history) < required:
                    return None

                # Get baseline (first iteration)
                baseline = history[0]["metric_value"]

                # Check if there's significant improvement
                improvements = [
                    h["metric_value"] - baseline
                    for h in history[1:required+1]
                ]

                # Consider improved if average improvement > 1% of baseline
                avg_improvement = sum(improvements) / len(improvements)
                threshold = 0.01 * abs(baseline) if baseline != 0 else 0.001

                if avg_improvement > threshold:
                    status = "succeeded"
                else:
                    status = "failed"

                task["status"] = status

                # Move to completed tasks
                if status in ["succeeded", "failed"]:
                    self.tasks["completed_tasks"].append(task)
                    self.tasks["tasks"] = [t for t in self.tasks["tasks"] if t["task_id"] != task_id]

                self._save_registry()
                self.logger.info(
                    f"Task {task_id} evaluated: {status} "
                    f"(avg improvement: {avg_improvement:.4f})"
                )

                return status

        return None

    def get_task_for_advisor(self) -> Dict[str, Any]:
        """
        Get task information formatted for AI advisor

        Returns:
            Dictionary with current task info and history
        """
        current_task = self.get_current_task()

        return {
            "current_task": current_task,
            "completed_tasks": self.tasks.get("completed_tasks", []),
            "all_tasks": self.tasks.get("tasks", [])
        }

    def get_task_summary(self) -> str:
        """
        Get human-readable task summary

        Returns:
            Formatted string with task status
        """
        lines = ["=" * 60]
        lines.append("TASK REGISTRY SUMMARY")
        lines.append("=" * 60)

        current_task = self.get_current_task()
        if current_task:
            lines.append(f"\n📋 CURRENT TASK:")
            lines.append(f"   ID: {current_task['task_id']}")
            lines.append(f"   Description: {current_task['description']}")
            lines.append(f"   Started: Iteration {current_task['start_iteration']}")
            lines.append(f"   Target: {current_task['target_metric']}")
            lines.append(f"   Progress: {len(current_task['evaluation_history'])}/{current_task['required_iterations']} iterations")
        else:
            lines.append("\n📋 No active task")

        completed = self.tasks.get("completed_tasks", [])
        if completed:
            lines.append(f"\n✅ COMPLETED TASKS: {len(completed)}")
            for task in completed[-3:]:  # Show last 3
                lines.append(f"   - {task['task_id']}: {task['status']} ({task['description'][:50]}...)")

        lines.append("=" * 60)
        return "\n".join(lines)


class IterationEvaluator:
    """
    Evaluates iteration results by analyzing last N iterations
    """

    def __init__(self, output_dir: str = "auto_improvement_runs", logger: Optional[logging.Logger] = None):
        """
        Initialize iteration evaluator

        Args:
            output_dir: Directory containing iteration results
            logger: Optional logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or self._create_logger()

    def _create_logger(self):
        """Create default logger"""
        logger = logging.getLogger('iteration_evaluator')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def load_iteration_summaries(self, iterations: List[int]) -> List[Dict[str, Any]]:
        """
        Load iteration summaries for specified iterations

        Args:
            iterations: List of iteration numbers to load

        Returns:
            List of iteration summary dictionaries
        """
        summaries = []

        for iter_num in iterations:
            iter_dir = os.path.join(self.output_dir, f"iteration_{iter_num:03d}")
            summary_file = os.path.join(iter_dir, "iteration_summary.json")

            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        summaries.append(summary)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to load iteration {iter_num}: {e}")
            else:
                self.logger.warning(f"Summary not found for iteration {iter_num}")

        return summaries

    def get_last_n_summaries(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get summaries for the last N iterations

        Args:
            n: Number of recent iterations to retrieve

        Returns:
            List of iteration summaries
        """
        # Find all iteration directories
        iteration_dirs = []
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                if item.startswith("iteration_") and os.path.isdir(os.path.join(self.output_dir, item)):
                    try:
                        iter_num = int(item.split("_")[1])
                        iteration_dirs.append(iter_num)
                    except (ValueError, IndexError):
                        continue

        # Get last N iterations
        iteration_dirs.sort()
        last_n = iteration_dirs[-n:] if len(iteration_dirs) >= n else iteration_dirs

        return self.load_iteration_summaries(last_n)

    def compare_metrics(
        self,
        summaries: List[Dict[str, Any]],
        metric: str = "avg_f1"
    ) -> Dict[str, Any]:
        """
        Compare a specific metric across iterations

        Args:
            summaries: List of iteration summaries
            metric: Metric to compare (e.g., "avg_f1", "avg_auc")

        Returns:
            Dictionary with comparison results
        """
        if not summaries:
            return {"error": "No summaries provided"}

        values = [s.get(metric, 0.0) for s in summaries]
        iterations = [s.get("iteration", 0) for s in summaries]

        return {
            "metric": metric,
            "values": values,
            "iterations": iterations,
            "best_value": max(values),
            "best_iteration": iterations[values.index(max(values))],
            "worst_value": min(values),
            "worst_iteration": iterations[values.index(min(values))],
            "trend": "improving" if values[-1] > values[0] else "degrading",
            "improvement": values[-1] - values[0]
        }

    def format_for_advisor(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format iteration data for AI advisor

        Args:
            summaries: List of iteration summaries

        Returns:
            Dictionary formatted for AI advisor consumption
        """
        if not summaries:
            return {}

        # Per-disease metrics breakdown for each iteration
        disease_metrics = {}

        for summary in summaries:
            iter_num = summary.get("iteration", 0)
            thresholds = summary.get("thresholds", {})

            # Extract per-disease metrics from thresholds
            for disease, threshold_data in thresholds.items():
                if disease not in disease_metrics:
                    disease_metrics[disease] = []

                disease_metrics[disease].append({
                    "iteration": iter_num,
                    "threshold": threshold_data.get("threshold", 0.5),
                    "f1_score": threshold_data.get("score", 0.0),
                    "positive_samples": threshold_data.get("positive_samples", 0),
                    "status": threshold_data.get("status", "unknown")
                })

        return {
            "iteration_summaries": summaries,
            "disease_metrics": disease_metrics,
            "f1_comparison": self.compare_metrics(summaries, "avg_f1"),
            "auc_comparison": self.compare_metrics(summaries, "avg_auc"),
            "recall_comparison": self.compare_metrics(summaries, "avg_recall"),
        }


if __name__ == "__main__":
    # Example usage
    print("Task Manager module loaded successfully")

    # Create task manager
    tm = TaskManager()
    print(tm.get_task_summary())

    # Create iteration evaluator
    evaluator = IterationEvaluator()
    summaries = evaluator.get_last_n_summaries(3)
    print(f"\nLoaded {len(summaries)} iteration summaries")
