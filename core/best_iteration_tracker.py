"""
Best Iteration Tracker - Multi-Metric Tracking System
Tracks multiple "best" iterations across different metrics and objectives

Extended to support role-based iteration tracking for dual-lineage strategy:
- TRAIN_AUC: Focus on AUC improvement (parent=iter12)
- RECOVER_F1: Focus on F1 recovery (parent=iter58)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

# Import role-based constants
try:
    from iteration_baselines import (
        IterationRole,
        ITERATION_12_AUC_ANCHOR,
        ITERATION_58_F1_ANCHOR,
        get_parent_for_role as baselines_get_parent
    )
    ROLE_BASED_TRACKING = True
except ImportError:
    ROLE_BASED_TRACKING = False
    print("Warning: iteration_baselines not available, role-based tracking disabled")


class BestIterationTracker:
    """
    Tracks multiple "best" iterations across different metrics

    Maintains a registry of:
    - Best F1, AUC, Recall, Precision
    - Best rare/common disease performance
    - Pareto optimal iterations
    - Most balanced iteration
    """

    # Metrics to track (can be extended)
    TRACKED_METRICS = {
        'best_f1': {
            'metric': 'avg_f1',
            'mode': 'max',
            'display_name': 'Best F1-Score'
        },
        'best_auc': {
            'metric': 'avg_auc',
            'mode': 'max',
            'display_name': 'Best AUC'
        },
        'best_recall': {
            'metric': 'avg_recall',
            'mode': 'max',
            'display_name': 'Best Recall'
        },
        'best_precision': {
            'metric': 'avg_precision',
            'mode': 'max',
            'display_name': 'Best Precision'
        },
        'most_balanced': {
            'metric': 'metric_std',  # Lower is better (less variance)
            'mode': 'min',
            'display_name': 'Most Balanced'
        }
    }

    def __init__(self, registry_path: str = "best_iterations_registry.json", logger: Optional[logging.Logger] = None):
        """
        Initialize best iteration tracker

        Args:
            registry_path: Path to JSON file storing best iterations
            logger: Optional logger instance
        """
        self.registry_path = registry_path
        self.logger = logger or self._create_logger()
        self.best_iterations = self._load_registry()

    def _create_logger(self):
        """Create default logger"""
        logger = logging.getLogger('best_tracker')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def _load_registry(self) -> Dict:
        """Load best iterations registry from file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                self.logger.info(f"Loaded best iterations registry: {len(registry.get('tracked', {}))} metrics tracked")
                return registry
            except Exception as e:
                self.logger.warning(f"Failed to load registry: {e}. Starting fresh.")
                return self._init_registry()
        else:
            return self._init_registry()

    def _init_registry(self) -> Dict:
        """Initialize empty registry"""
        return {
            'tracked': {},  # {metric_name: {iteration: X, value: Y, ...}}
            'pareto_frontier': [],  # List of Pareto optimal iterations
            'history': [],  # Full history for analysis
            'last_updated': None
        }

    def _save_registry(self):
        """Save registry to file"""
        self.best_iterations['last_updated'] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.best_iterations, f, indent=2)
        self.logger.debug(f"Saved registry to {self.registry_path}")

    def backfill_from_previous_iterations(self, output_dir: str = "auto_improvement_runs"):
        """
        Backfill tracker with data from all previous iterations

        Args:
            output_dir: Directory containing iteration folders
        """
        import glob

        self.logger.info("🔄 Backfilling best iteration tracker from previous iterations...")

        # Find all iteration directories
        iteration_dirs = sorted(glob.glob(f"{output_dir}/iteration_*"))

        if not iteration_dirs:
            self.logger.info("   No previous iterations found")
            return

        backfilled_count = 0
        for iter_dir in iteration_dirs:
            # Extract iteration number
            iter_name = os.path.basename(iter_dir)
            try:
                iter_num = int(iter_name.split('_')[1])
            except (IndexError, ValueError):
                continue

            # Check if already in history
            if any(h.get('iteration') == iter_num for h in self.best_iterations['history']):
                continue

            # Load iteration summary
            summary_file = f"{iter_dir}/iteration_summary.json"
            if not os.path.exists(summary_file):
                continue

            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)

                # Extract metrics
                metrics = {
                    'avg_f1': summary.get('avg_f1', 0.0),
                    'avg_auc': summary.get('avg_auc', 0.0),
                    'avg_recall': summary.get('avg_recall', 0.0),
                    'avg_precision': summary.get('avg_precision', 0.0),
                    'avg_accuracy': summary.get('avg_accuracy', 0.0)
                }

                # Load config if available
                config = {}
                config_file = f"{iter_dir}/config.yaml"
                if os.path.exists(config_file):
                    import yaml
                    with open(config_file, 'r') as cf:
                        config = yaml.safe_load(cf)

                # Find model file
                model_files = glob.glob(f"{iter_dir}/pipeline_model_*.pth")
                model_path = model_files[0] if model_files else f"{iter_dir}/model.pth"

                # Update tracker (silent mode - no logging, no saving each time)
                self.update(
                    iteration=iter_num,
                    metrics=metrics,
                    config=config,
                    model_path=model_path,
                    task_info=None,  # Historical iterations don't have task info
                    silent=True  # Suppress logging during batch backfill
                )

                backfilled_count += 1

            except Exception as e:
                self.logger.warning(f"   Failed to backfill iteration {iter_num}: {e}")
                continue

        self.logger.info(f"✅ Backfilled {backfilled_count} iterations into tracker")
        self.logger.info(f"   Total tracked: {len(self.best_iterations.get('tracked', {}))} metrics")

        # Save once after all backfilling
        self._save_registry()

    def update(
        self,
        iteration: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        model_path: str,
        task_info: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        role: Optional[str] = None,
        parent_iteration: Optional[int] = None
    ):
        """
        Update best iteration tracking

        Args:
            iteration: Current iteration number
            metrics: Dictionary of metrics from this iteration
            config: Configuration used for this iteration
            model_path: Path to saved model file
            task_info: Optional task information (target metric, constraints, etc.)
            silent: If True, suppress logging (for batch backfill)
            role: Optional iteration role (TRAIN_AUC, RECOVER_F1, etc.)
            parent_iteration: Optional parent iteration number (for lineage tracking)
        """
        if not silent:
            role_info = f" [Role: {role}, Parent: {parent_iteration}]" if role else ""
            self.logger.info(f"Updating best iteration tracker for iteration {iteration}{role_info}")

        # Track task-specific best if task info provided
        if task_info:
            self._update_task_specific_best(iteration, metrics, config, model_path, task_info)

        # Add to history (including role and parent for lineage tracking)
        history_entry = {
            'iteration': iteration,
            'metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
        if role:
            history_entry['role'] = role
        if parent_iteration is not None:
            history_entry['parent_iteration'] = parent_iteration
        self.best_iterations['history'].append(history_entry)

        # Check each tracked metric
        updates = []
        for metric_name, tracker_config in self.TRACKED_METRICS.items():
            metric_key = tracker_config['metric']
            mode = tracker_config['mode']

            # Calculate metric value
            if metric_key == 'metric_std':
                # For "most balanced", calculate standard deviation of main metrics
                values = [
                    metrics.get('avg_f1', 0),
                    metrics.get('avg_auc', 0),
                    metrics.get('avg_recall', 0),
                    metrics.get('avg_precision', 0)
                ]
                metric_value = np.std(values)
            else:
                metric_value = metrics.get(metric_key)

            if metric_value is None:
                continue

            # Check if this is best
            if self._is_better(metric_value, metric_name, mode):
                old_best = self.best_iterations['tracked'].get(metric_name, {})
                old_iter = old_best.get('iteration', 'N/A')
                old_value = old_best.get('value', 'N/A')

                self.best_iterations['tracked'][metric_name] = {
                    'iteration': iteration,
                    'value': float(metric_value),
                    'metrics': metrics.copy(),
                    'config_summary': self._summarize_config(config),
                    'model_path': model_path,
                    'timestamp': datetime.now().isoformat()
                }

                updates.append(f"  ✨ New {tracker_config['display_name']}: Iteration {iteration} "
                              f"({metric_key}={metric_value:.4f}) [was: Iter {old_iter}, {old_value}]")

        # Update Pareto frontier
        pareto_updates = self._update_pareto_frontier(iteration, metrics, config, model_path)
        if pareto_updates:
            updates.append(f"  🏆 Added to Pareto frontier: Iteration {iteration}")

        # Log updates (unless silent mode)
        if not silent:
            if updates:
                self.logger.info(f"📊 Best iteration updates:")
                for update in updates:
                    self.logger.info(update)
            else:
                self.logger.info(f"   No new bests for iteration {iteration}")

        # Save registry (unless silent - backfill saves once at end)
        if not silent:
            self._save_registry()

    def _is_better(self, value: float, metric_name: str, mode: str) -> bool:
        """Check if value is better than current best"""
        current_best = self.best_iterations['tracked'].get(metric_name)

        if current_best is None:
            return True  # First time, always better

        current_value = current_best['value']

        if mode == 'max':
            return value > current_value
        else:  # mode == 'min'
            return value < current_value

    def _summarize_config(self, config: Dict) -> Dict:
        """Create a compact summary of important config parameters"""
        summary = {}

        # Training params
        if 'training' in config:
            summary['learning_rate'] = config['training'].get('learning_rate')
            summary['num_epochs'] = config['training'].get('num_epochs')

        # Loss params
        if 'loss' in config:
            summary['loss_type'] = config['loss'].get('type')
            summary['gamma'] = config['loss'].get('gamma')
            summary['alpha'] = config['loss'].get('alpha')

        # Model params
        if 'model' in config:
            summary['dropout'] = config['model'].get('dropout_rate')
            summary['architecture'] = config['model'].get('architecture')

        return summary

    def _update_pareto_frontier(
        self,
        iteration: int,
        metrics: Dict[str, float],
        config: Dict,
        model_path: str
    ) -> bool:
        """
        Update Pareto frontier

        An iteration is Pareto optimal if no other iteration dominates it
        (i.e., no other iteration is better in ALL metrics)

        Returns:
            True if iteration was added to frontier
        """
        # Key metrics for Pareto analysis
        key_metrics = ['avg_f1', 'avg_auc', 'avg_recall', 'avg_precision']

        # Get current metrics
        current_values = [metrics.get(m, 0) for m in key_metrics]

        # Check if current iteration is dominated by any existing frontier member
        frontier = self.best_iterations['pareto_frontier']
        is_dominated = False

        for frontier_item in frontier:
            frontier_values = [frontier_item['metrics'].get(m, 0) for m in key_metrics]

            # Check if frontier_item dominates current
            if all(f >= c for f, c in zip(frontier_values, current_values)) and \
               any(f > c for f, c in zip(frontier_values, current_values)):
                is_dominated = True
                break

        if is_dominated:
            return False

        # Current iteration is not dominated - add to frontier
        # Remove any frontier members dominated by current
        new_frontier = []
        for frontier_item in frontier:
            frontier_values = [frontier_item['metrics'].get(m, 0) for m in key_metrics]

            # Check if current dominates frontier_item
            dominates = all(c >= f for c, f in zip(current_values, frontier_values)) and \
                       any(c > f for c, f in zip(current_values, frontier_values))

            if not dominates:
                new_frontier.append(frontier_item)

        # Add current iteration
        new_frontier.append({
            'iteration': iteration,
            'metrics': metrics.copy(),
            'config_summary': self._summarize_config(config),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        })

        self.best_iterations['pareto_frontier'] = new_frontier
        return True

    def get_best_for_metric(self, metric: str) -> Optional[Dict]:
        """
        Get best iteration for specific metric

        Args:
            metric: Metric name (e.g., 'best_f1', 'best_auc')

        Returns:
            Dictionary with iteration info or None
        """
        return self.best_iterations['tracked'].get(metric)

    def get_all_best(self) -> Dict[str, Dict]:
        """Get all tracked best iterations"""
        return self.best_iterations['tracked']

    def get_pareto_frontier(self) -> List[Dict]:
        """Get all Pareto optimal iterations"""
        return self.best_iterations['pareto_frontier']

    def get_parent_for_role(self, role: 'IterationRole') -> Tuple[Optional[int], str]:
        """
        Get the correct parent iteration for a given role.

        This implements the dual-lineage strategy:
        - TRAIN_AUC: Always use iteration 12 (AUC anchor)
        - RECOVER_F1: Always use iteration 58 (F1 anchor)
        - ADJUST_THRESHOLDS: Use current iteration (no parent)

        Args:
            role: The IterationRole enum value

        Returns:
            Tuple of (parent_iteration, model_path)
        """
        if not ROLE_BASED_TRACKING:
            self.logger.warning("Role-based tracking not available, returning None")
            return (None, "")

        if role == IterationRole.TRAIN_AUC:
            return (
                ITERATION_12_AUC_ANCHOR.iteration,
                ITERATION_12_AUC_ANCHOR.model_path
            )
        elif role == IterationRole.RECOVER_F1:
            return (
                ITERATION_58_F1_ANCHOR.iteration,
                ITERATION_58_F1_ANCHOR.model_path
            )
        elif role == IterationRole.ADJUST_THRESHOLDS:
            # No parent - use current model without training
            return (None, "")
        else:
            # ABORT or unknown
            return (None, "")

    def get_anchor_iterations(self) -> Dict[str, Dict]:
        """
        Get information about the anchor iterations.

        Returns:
            Dict with AUC and F1 anchor information
        """
        if not ROLE_BASED_TRACKING:
            return {}

        return {
            'auc_anchor': {
                'iteration': ITERATION_12_AUC_ANCHOR.iteration,
                'macro_auc': ITERATION_12_AUC_ANCHOR.macro_auc,
                'macro_f1': ITERATION_12_AUC_ANCHOR.macro_f1,
                'model_path': ITERATION_12_AUC_ANCHOR.model_path,
                'description': ITERATION_12_AUC_ANCHOR.description
            },
            'f1_anchor': {
                'iteration': ITERATION_58_F1_ANCHOR.iteration,
                'macro_auc': ITERATION_58_F1_ANCHOR.macro_auc,
                'macro_f1': ITERATION_58_F1_ANCHOR.macro_f1,
                'model_path': ITERATION_58_F1_ANCHOR.model_path,
                'description': ITERATION_58_F1_ANCHOR.description
            }
        }

    def get_comparison_data(self, current_iteration: int, task_info: Optional[Dict] = None) -> Dict:
        """
        Get comprehensive comparison data for AI advisor

        Args:
            current_iteration: Current iteration number
            task_info: Optional current task information

        Returns:
            Dictionary with all comparison data
        """
        data = {
            'current_iteration': current_iteration,
            'best_iterations': self.get_all_best(),
            'pareto_frontier': self.get_pareto_frontier(),
            'total_iterations_tracked': len(self.best_iterations['history']),
            'iterations_since_best_f1': self._iterations_since('best_f1', current_iteration),
            'iterations_since_best_auc': self._iterations_since('best_auc', current_iteration),
            'recent_trend': self._analyze_recent_trend()
        }

        # Add anchor iterations for dual-lineage strategy
        anchor_info = self.get_anchor_iterations()
        if anchor_info:
            data['anchor_iterations'] = anchor_info

        # Add task-specific data if available
        if task_info:
            data['current_task'] = task_info
            data['task_specific_best'] = self.best_iterations.get('task_specific', {})

        return data

    def _update_task_specific_best(
        self,
        iteration: int,
        metrics: Dict[str, float],
        config: Dict,
        model_path: str,
        task_info: Dict
    ):
        """
        Track best iteration for specific task objectives

        Example task_info:
        {
            'target_metric': 'avg_auc',
            'target_value': 0.81,
            'constraints': {
                'avg_f1': {'min': 0.25, 'operation': 'maintain'}  # Don't harm F1
            }
        }
        """
        if 'task_specific' not in self.best_iterations:
            self.best_iterations['task_specific'] = {}

        target_metric = task_info.get('target_metric')
        constraints = task_info.get('constraints', {})

        if not target_metric:
            return

        target_value = metrics.get(target_metric)
        if target_value is None:
            return

        # Check if constraints are satisfied
        constraints_met = True
        for constraint_metric, constraint_spec in constraints.items():
            current_value = metrics.get(constraint_metric)
            if current_value is None:
                continue

            if constraint_spec.get('operation') == 'maintain':
                min_value = constraint_spec.get('min', 0)
                if current_value < min_value:
                    constraints_met = False
                    self.logger.info(f"   ⚠️  Constraint violated: {constraint_metric}={current_value:.4f} < {min_value:.4f}")
                    break

        if not constraints_met:
            self.logger.info(f"   ⚠️  Iteration {iteration} violates task constraints, not recorded as task-best")
            return

        # Check if this is best for task
        task_key = f"task_{target_metric}"
        current_best = self.best_iterations['task_specific'].get(task_key)

        is_new_best = False
        if current_best is None:
            is_new_best = True
        elif target_value > current_best['value']:
            is_new_best = True

        if is_new_best:
            self.best_iterations['task_specific'][task_key] = {
                'iteration': iteration,
                'value': float(target_value),
                'metrics': metrics.copy(),
                'config_summary': self._summarize_config(config),
                'model_path': model_path,
                'task_info': task_info.copy(),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.info(f"   ✨ New task-specific best: {target_metric}={target_value:.4f} "
                           f"(task: improve {target_metric} without harming {list(constraints.keys())})")

    def get_task_specific_best(self, target_metric: str) -> Optional[Dict]:
        """Get best iteration for specific task"""
        task_key = f"task_{target_metric}"
        return self.best_iterations.get('task_specific', {}).get(task_key)

    def _iterations_since(self, metric: str, current_iteration: int) -> Optional[int]:
        """Calculate iterations since best for a metric"""
        best = self.get_best_for_metric(metric)
        if best:
            return current_iteration - best['iteration']
        return None

    def _analyze_recent_trend(self, window: int = 5) -> Dict:
        """Analyze trend in recent iterations"""
        history = self.best_iterations['history'][-window:]

        if len(history) < 2:
            return {'trend': 'insufficient_data'}

        # Calculate trends for key metrics
        trends = {}
        for metric in ['avg_f1', 'avg_auc', 'avg_recall']:
            values = [h['metrics'].get(metric, 0) for h in history]
            if len(values) >= 2:
                # Simple linear trend
                if values[-1] > values[0]:
                    trends[metric] = 'improving'
                elif values[-1] < values[0]:
                    trends[metric] = 'declining'
                else:
                    trends[metric] = 'stable'

        return trends

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        lines = []
        lines.append("="*80)
        lines.append("BEST ITERATIONS SUMMARY")
        lines.append("="*80)

        # Best iterations
        for metric_name, tracker_config in self.TRACKED_METRICS.items():
            best = self.get_best_for_metric(metric_name)
            if best:
                lines.append(f"\n{tracker_config['display_name']}:")
                lines.append(f"  Iteration: {best['iteration']}")
                lines.append(f"  Value: {best['value']:.4f}")
                lines.append(f"  Timestamp: {best['timestamp']}")

        # Pareto frontier
        frontier = self.get_pareto_frontier()
        lines.append(f"\nPareto Frontier:")
        lines.append(f"  {len(frontier)} iteration(s): {[item['iteration'] for item in frontier]}")

        # Recent trend
        trend = self._analyze_recent_trend()
        lines.append(f"\nRecent Trend (last 5 iterations):")
        for metric, direction in trend.items():
            if metric != 'trend':
                lines.append(f"  {metric}: {direction}")

        lines.append("="*80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test the tracker
    tracker = BestIterationTracker()

    # Simulate some iterations
    test_iterations = [
        {'iteration': 1, 'avg_f1': 0.25, 'avg_auc': 0.75, 'avg_recall': 0.40, 'avg_precision': 0.22},
        {'iteration': 2, 'avg_f1': 0.27, 'avg_auc': 0.76, 'avg_recall': 0.42, 'avg_precision': 0.23},
        {'iteration': 3, 'avg_f1': 0.26, 'avg_auc': 0.78, 'avg_recall': 0.38, 'avg_precision': 0.25},
        {'iteration': 4, 'avg_f1': 0.28, 'avg_auc': 0.77, 'avg_recall': 0.45, 'avg_precision': 0.24},
    ]

    for test in test_iterations:
        iteration = test.pop('iteration')
        tracker.update(
            iteration=iteration,
            metrics=test,
            config={'training': {'learning_rate': 0.001}},
            model_path=f"model_{iteration}.pth"
        )

    print(tracker.get_summary_report())
