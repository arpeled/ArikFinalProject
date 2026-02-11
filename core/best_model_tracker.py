"""
Track best performing iteration and enable rollback
"""
import os
import json
import shutil
from typing import Dict, Any, Optional


class BestModelTracker:
    """Tracks best iteration and enables rollback"""

    def __init__(self, output_dir: str = "auto_improvement_runs", metric: str = "avg_auc"):
        """
        Initialize tracker

        Args:
            output_dir: Directory containing iterations
            metric: Metric to optimize (avg_auc, avg_f1, etc.)
        """
        self.output_dir = output_dir
        self.metric = metric
        self.tracker_file = os.path.join(output_dir, "best_model_tracker.json")
        self.best_iteration = None
        self.best_metric_value = -float('inf')

        # Load existing tracker if available
        self.load()

    def load(self):
        """Load best model tracker from file"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                data = json.load(f)
                self.best_iteration = data.get('best_iteration')
                self.best_metric_value = data.get('best_metric_value', -float('inf'))

    def save(self):
        """Save best model tracker to file"""
        data = {
            'best_iteration': self.best_iteration,
            'best_metric_value': self.best_metric_value,
            'metric': self.metric
        }
        with open(self.tracker_file, 'w') as f:
            json.dump(data, f, indent=2)

    def update(self, iteration_summary: Dict[str, Any]) -> bool:
        """
        Update tracker with new iteration

        Args:
            iteration_summary: Summary of current iteration

        Returns:
            True if this is a new best, False otherwise
        """
        current_value = iteration_summary.get(self.metric, -float('inf'))

        if current_value > self.best_metric_value:
            self.best_iteration = iteration_summary['iteration']
            self.best_metric_value = current_value
            self.save()
            return True

        return False

    def should_rollback(self, iterations_without_improvement: int, threshold: int = 3) -> bool:
        """
        Determine if we should rollback to best model

        Args:
            iterations_without_improvement: Number of iterations since last improvement
            threshold: Number of iterations to tolerate before rollback

        Returns:
            True if should rollback, False otherwise
        """
        return iterations_without_improvement >= threshold

    def get_best_config_path(self) -> Optional[str]:
        """Get path to best config file"""
        if self.best_iteration is None:
            return None

        # Try multiple locations
        candidates = [
            f"config_iteration_{self.best_iteration:03d}.yaml",
            os.path.join(self.output_dir, f"iteration_{self.best_iteration:03d}", "config.yaml")
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return None

    def get_best_summary(self) -> Optional[Dict[str, Any]]:
        """Get full summary of best iteration"""
        if self.best_iteration is None:
            return None

        summary_file = os.path.join(
            self.output_dir,
            f"iteration_{self.best_iteration:03d}",
            "iteration_summary.json"
        )

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)

        return None

    def rollback_to_best(self, next_iteration: int) -> Optional[str]:
        """
        Create config for next iteration by copying best config

        Args:
            next_iteration: Next iteration number

        Returns:
            Path to new config file, or None if rollback failed
        """
        best_config = self.get_best_config_path()

        if best_config is None:
            return None

        # Create new config path
        new_config_path = f"config_iteration_{next_iteration:03d}.yaml"

        # Copy best config to new iteration
        shutil.copy(best_config, new_config_path)

        return new_config_path

    def get_status_message(self, current_iteration: int) -> str:
        """Get status message for logging"""
        if self.best_iteration is None:
            return "No best iteration tracked yet"

        iterations_since_best = current_iteration - self.best_iteration

        return (f"Best: Iteration #{self.best_iteration} "
                f"({self.metric}={self.best_metric_value:.4f}) "
                f"[{iterations_since_best} iterations ago]")
