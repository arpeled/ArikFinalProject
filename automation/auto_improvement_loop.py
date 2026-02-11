"""
Auto-Improvement Loop for Chest X-Ray Classification Pipeline
Automatically runs training iterations with AI-suggested improvements

Extended with role-based auto-improvement with dual-lineage strategy:
- TRAIN_AUC: Focus on AUC improvement (parent=iter12)
- RECOVER_F1: Focus on F1 recovery (parent=iter58)
- ADJUST_THRESHOLDS: No training, only threshold optimization
"""

import os
import sys

# Add automation/ and core/ directories to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
_core_dir = os.path.join(_project_root, 'core')
for _d in [_this_dir, _core_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)
import time
import datetime
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from config_manager import ConfigManager
from config_based_pipeline import ConfigBasedTrainer
from chest_xray_test_pipeline import test_model_pipeline
from ai_advisor import AIAdvisor
from telegram_notifier import TelegramNotifier
from best_model_tracker import BestModelTracker
from best_iteration_tracker import BestIterationTracker
from task_manager import TaskManager, IterationEvaluator

# Import role-based constants and phased protocol
try:
    from iteration_baselines import (
        IterationRole,
        OptimizationPhase,
        ITERATION_12_AUC_ANCHOR,
        ITERATION_58_F1_ANCHOR,
        ITERATION_12_EXACT_CONFIG,
        PHASE_CRITERIA,
        TRAIN_AUC_RULES,
        DECISION_RULES,
        FINAL_MODEL_CRITERIA,
        DISEASE_GROUPS,
        determine_iteration_role,
        determine_current_phase,
        generate_ai_advisory_decision,
        get_phase1_exact_config,
        get_enforced_config_for_phase,
        check_abort_conditions,
        check_final_model_criteria,
        get_parent_for_role,
        format_thesis_documentation_marker,
        compute_group_metrics,
        AIAdvisoryDecision,
        # Phase 5 imports
        PHASE_5_BASELINE_ITERATION,
        PHASE_5_BASELINE_MACRO_AUC,
        PHASE_5_BASELINE_PER_CLASS_AUC,
        PHASE_5_TARGET_DISEASES,
        PHASE_5_STABLE_DISEASES,
        PHASE_5_CONFIG,
        get_phase5_config,
        compute_phase5_auc_deltas,
        check_phase5_stop_conditions
    )
    ROLE_BASED_LOOP = True
    PHASED_PROTOCOL_ENABLED = True
except ImportError:
    ROLE_BASED_LOOP = False
    PHASED_PROTOCOL_ENABLED = False
    print("Warning: iteration_baselines not available, role-based loop disabled")


@dataclass
class IterationLog:
    """Structured log for thesis documentation"""
    iteration: int
    role: str
    parent_iteration: Optional[int]
    timestamp: str

    # Core metrics
    macro_auc: float
    macro_f1: float
    auc_vs_iter12: float
    f1_vs_iter58: float

    # Per-group metrics
    per_group_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Rare class details
    rare_class_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Distribution diagnostics
    distribution_diagnostics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Decision
    decision: Dict[str, Any] = field(default_factory=dict)

    # Training metadata
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    # Corrections applied to AI suggestions
    ai_corrections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# Wang et al. baseline results
BASELINE_RESULTS = {
    'Cardiomegaly': {'AUC': 0.919101, 'Threshold': 0.489469, 'Sensitivity': 0.865, 'Specificity': 0.819083, 'Accuracy': 0.820225, 'Precision': 0.112312, 'Recall': 0.863322, 'F1_Score': 0.198765},
    'Emphysema': {'AUC': 0.916227, 'Threshold': 0.445399, 'Sensitivity': 0.866, 'Specificity': 0.819995, 'Accuracy': 0.82103, 'Precision': 0.103055, 'Recall': 0.864245, 'F1_Score': 0.184152},
    'Effusion': {'AUC': 0.888239, 'Threshold': 0.484694, 'Sensitivity': 0.839, 'Specificity': 0.794271, 'Accuracy': 0.799491, 'Precision': 0.354027, 'Recall': 0.838296, 'F1_Score': 0.497818},
    'Hernia': {'AUC': 0.82202, 'Threshold': 0.581222, 'Sensitivity': 0.717, 'Specificity': 0.802346, 'Accuracy': 0.802127, 'Precision': 0.00719748, 'Recall': 0.695652, 'F1_Score': 0.0142476},
    'Infiltration': {'AUC': 0.715871, 'Threshold': 0.476786, 'Sensitivity': 0.666, 'Specificity': 0.645026, 'Accuracy': 0.648673, 'Precision': 0.288003, 'Recall': 0.665575, 'F1_Score': 0.402838},
    'Mass': {'AUC': 0.869582, 'Threshold': 0.585565, 'Sensitivity': 0.758, 'Specificity': 0.825244, 'Accuracy': 0.821655, 'Precision': 0.193864, 'Recall': 0.757089, 'F1_Score': 0.308678},
    'Nodule': {'AUC': 0.792307, 'Threshold': 0.485297, 'Sensitivity': 0.700, 'Specificity': 0.737614, 'Accuracy': 0.735723, 'Precision': 0.136427, 'Recall': 0.703614, 'F1_Score': 0.228502},
    'Atelectasis': {'AUC': 0.818337, 'Threshold': 0.458412, 'Sensitivity': 0.8, 'Specificity': 0.682717, 'Accuracy': 0.694521, 'Precision': 0.2282, 'Recall': 0.799911, 'F1_Score': 0.345336},
    'Pneumothorax': {'AUC': 0.886252, 'Threshold': 0.465026, 'Sensitivity': 0.831, 'Specificity': 0.787684, 'Accuracy': 0.789749, 'Precision': 0.166667, 'Recall': 0.830119, 'F1_Score': 0.277599},
    'Pleural_Thickening': {'AUC': 0.812256, 'Threshold': 0.441709, 'Sensitivity': 0.83, 'Specificity': 0.657618, 'Accuracy': 0.663107, 'Precision': 0.0742729, 'Recall': 0.828691, 'F1_Score': 0.136327},
    'Pneumonia': {'AUC': 0.765119, 'Threshold': 0.51558, 'Sensitivity': 0.73, 'Specificity': 0.696425, 'Accuracy': 0.6968, 'Precision': 0.0292288, 'Recall': 0.726619, 'F1_Score': 0.056197},
    'Fibrosis': {'AUC': 0.826327, 'Threshold': 0.585557, 'Sensitivity': 0.798, 'Specificity': 0.722608, 'Accuracy': 0.723782, 'Precision': 0.0419931, 'Recall': 0.795252, 'F1_Score': 0.0797738},
    'Edema': {'AUC': 0.908537, 'Threshold': 0.413506, 'Sensitivity': 0.911, 'Specificity': 0.746759, 'Accuracy': 0.750156, 'Precision': 0.0714644, 'Recall': 0.908511, 'F1_Score': 0.132596},
    'Consolidation': {'AUC': 0.812462, 'Threshold': 0.546624, 'Sensitivity': 0.774, 'Specificity': 0.722326, 'Accuracy': 0.724462, 'Precision': 0.109814, 'Recall': 0.772632, 'F1_Score': 0.192298}
}


class AutoImprovementLoop:
    """Automated training loop with AI-guided improvements"""

    def __init__(
        self,
        base_config_path: str = "experiments/configs/config_baseline.yaml",
        max_iterations: int = 10,
        openai_api_key: str = None,
        output_dir: str = "experiments/auto_improvement_runs",
        resume: bool = False
    ):
        """
        Initialize auto-improvement loop

        Args:
            base_config_path: Path to baseline configuration
            max_iterations: Maximum number of iterations to run
            openai_api_key: OpenAI API key
            output_dir: Directory to store run outputs
            resume: If True, resume from the last iteration in output_dir
        """
        self.base_config_path = base_config_path
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.openai_api_key = openai_api_key
        self.resume = resume

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize AI advisor
        self.ai_advisor = AIAdvisor(api_key=openai_api_key, model="gpt-5.2")

        # Initialize Telegram notifier (optional)
        self.telegram = TelegramNotifier()

        # Setup logging (before resume detection so we can log resume info)
        self.logger = self._setup_logging()

        # Track iterations
        self.iteration_history = []

        # Track best model for rollback
        self.best_tracker = BestModelTracker(output_dir=output_dir, metric="avg_auc")
        self.iterations_without_improvement = 0

        # Track best iterations for intelligent warm start
        self.best_iteration_tracker = BestIterationTracker(
            registry_path=os.path.join(output_dir, "best_iterations_registry.json"),
            logger=self.logger
        )

        # Initialize task management system
        self.task_manager = TaskManager(
            registry_path=os.path.join(output_dir, "tasks_registry.json"),
            logger=self.logger
        )
        self.iteration_evaluator = IterationEvaluator(
            output_dir=output_dir,
            logger=self.logger
        )

        # Resume detection
        self.start_iteration = 1
        self.resume_config_path = None
        if resume:
            self._detect_resume_point()

        # Backfill best iteration tracker with historical data
        # This ensures warm start has access to all previous iterations
        if resume or os.path.exists(output_dir):
            self.best_iteration_tracker.backfill_from_previous_iterations(output_dir)

    def _setup_logging(self):
        """Setup centralized logging"""
        log_file = os.path.join(self.output_dir, f"auto_improvement_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

        logger = logging.getLogger('auto_improvement')
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        # Add Telegram logging handler to automatically capture important logs
        if self.telegram.enabled:
            telegram_handler = self.telegram.create_logging_handler(level=logging.INFO)
            telegram_handler.setFormatter(formatter)
            logger.addHandler(telegram_handler)
            logger.info("📱 Telegram logging handler attached")

        return logger

    def _detect_resume_point(self):
        """Detect the last completed iteration and resume from there"""
        self.logger.info("Resume mode enabled - searching for previous iterations...")

        # Find all iteration directories
        iteration_dirs = []
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                if item.startswith("iteration_") and os.path.isdir(os.path.join(self.output_dir, item)):
                    try:
                        # Extract iteration number from directory name (e.g., iteration_003 -> 3)
                        iter_num = int(item.split("_")[1])
                        iteration_dirs.append(iter_num)
                    except (ValueError, IndexError):
                        continue

        if not iteration_dirs:
            self.logger.warning("No previous iterations found in output directory.")
            self.logger.info("Starting from iteration 1 with baseline config.")
            self.start_iteration = 1
            self.resume_config_path = self.base_config_path
            return

        # Find the latest iteration
        last_iteration = max(iteration_dirs)
        self.start_iteration = last_iteration + 1

        # Load previous iteration history for AI context
        self._load_previous_iteration_history(sorted(iteration_dirs))

        # Look for the config file for the next iteration
        next_config_path = os.path.join("experiments", "configs", f"config_iteration_{self.start_iteration:03d}.yaml")

        # If next config already exists (from previous run's AI suggestions), use it
        if os.path.exists(next_config_path):
            self.resume_config_path = next_config_path
            self.logger.info(f"✅ Found existing config for iteration {self.start_iteration}: {next_config_path}")
        else:
            # Otherwise, use the config from the last completed iteration
            last_config_path = os.path.join("experiments", "configs", f"config_iteration_{last_iteration:03d}.yaml")
            if os.path.exists(last_config_path):
                self.resume_config_path = last_config_path
                self.logger.info(f"✅ Resuming from iteration {last_iteration}'s config: {last_config_path}")
            else:
                # Fallback: check in the iteration directory
                last_iter_dir = os.path.join(self.output_dir, f"iteration_{last_iteration:03d}")
                config_in_dir = os.path.join(last_iter_dir, "config.yaml")
                if os.path.exists(config_in_dir):
                    self.resume_config_path = config_in_dir
                    self.logger.info(f"✅ Found config in iteration directory: {config_in_dir}")
                else:
                    self.logger.warning(f"Could not find config for iteration {last_iteration}")
                    self.logger.info("Falling back to baseline config")
                    self.resume_config_path = self.base_config_path
                    self.start_iteration = last_iteration + 1

        self.logger.info("="*80)
        self.logger.info(f"RESUME SUMMARY:")
        self.logger.info(f"  Last completed iteration: {last_iteration}")
        self.logger.info(f"  Resuming from iteration: {self.start_iteration}")
        self.logger.info(f"  Using config: {self.resume_config_path}")
        self.logger.info(f"  Loaded {len(self.iteration_history)} previous iteration(s) for AI context")
        self.logger.info(f"  Will run iterations {self.start_iteration} to {self.start_iteration + self.max_iterations - 1}")
        self.logger.info("="*80)

    def _load_previous_iteration_history(self, iteration_nums: List[int]):
        """
        Load previous iteration history from completed iterations

        Args:
            iteration_nums: Sorted list of iteration numbers to load
        """
        self.logger.info(f"Loading history for {len(iteration_nums)} previous iteration(s)...")

        for iter_num in iteration_nums:
            iter_dir = os.path.join(self.output_dir, f"iteration_{iter_num:03d}")

            try:
                # Check if iteration failed
                failed_file = os.path.join(iter_dir, f"ITERATION_FAILED_{iter_num:03d}.txt")
                is_failed = os.path.exists(failed_file)

                # First, try to load from the saved JSON summary (preferred method)
                summary_file = os.path.join(iter_dir, "iteration_summary.json")

                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        iteration_summary = json.load(f)

                    # Mark as failed if error file exists
                    if is_failed:
                        iteration_summary['status'] = 'failed'
                        iteration_summary['note'] = 'Iteration failed but partial results saved'

                    # Extract key metrics for logging
                    avg_auc = iteration_summary.get('avg_auc', 0.0)
                    avg_f1 = iteration_summary.get('avg_f1', 0.0)
                    avg_recall = iteration_summary.get('avg_recall', 0.0)
                    ai_status = iteration_summary.get('ai_analysis_status', 'unknown')
                    status_marker = " ⚠️ FAILED" if is_failed else ""

                    self.iteration_history.append(iteration_summary)
                    self.logger.info(f"  Loaded iteration {iter_num}: AUC={avg_auc:.4f}, F1={avg_f1:.4f}, Recall={avg_recall:.4f} (AI: {ai_status}){status_marker}")

                else:
                    # Fallback: reconstruct from CSV files (for backward compatibility)
                    self.logger.info(f"  No summary file found for iteration {iter_num}, reconstructing from CSV...")

                    results_files = [f for f in os.listdir(iter_dir) if f.startswith("pipeline_results_") and f.endswith(".csv")]

                    if not results_files:
                        self.logger.warning(f"  No results file found for iteration {iter_num}, skipping")
                        continue

                    # Load the most recent results file
                    results_file = os.path.join(iter_dir, sorted(results_files)[-1])
                    results_df = pd.read_csv(results_file)

                    # Extract metrics
                    avg_auc = results_df['AUC'].mean()
                    avg_f1 = results_df['F1_Score'].mean() if 'F1_Score' in results_df else 0.0
                    avg_recall = results_df['Recall'].mean() if 'Recall' in results_df else 0.0
                    avg_precision = results_df['Precision'].mean() if 'Precision' in results_df else 0.0
                    avg_accuracy = results_df['Accuracy'].mean() if 'Accuracy' in results_df else 0.0

                    # Try to load AI analysis to get suggested changes
                    suggested_changes = {}
                    analysis_files = [f for f in os.listdir(iter_dir) if f.startswith("ai_analysis_") and f.endswith(".txt")]
                    if analysis_files:
                        suggested_changes = {"reasoning": f"Analysis available in {analysis_files[0]}"}

                    # Create iteration summary
                    iteration_summary = {
                        'iteration': iter_num,
                        'avg_auc': avg_auc,
                        'avg_f1': avg_f1,
                        'avg_recall': avg_recall,
                        'avg_precision': avg_precision,
                        'avg_accuracy': avg_accuracy,
                        'suggested_changes': suggested_changes,
                        'ai_analysis_status': 'completed' if analysis_files else 'unknown'
                    }

                    # Mark as failed if error file exists
                    if is_failed:
                        iteration_summary['status'] = 'failed'
                        iteration_summary['note'] = 'Iteration failed but partial results reconstructed'

                    status_marker = " ⚠️ FAILED (reconstructed)" if is_failed else " (reconstructed)"
                    self.iteration_history.append(iteration_summary)
                    self.logger.info(f"  Loaded iteration {iter_num}: AUC={avg_auc:.4f}, F1={avg_f1:.4f}, Recall={avg_recall:.4f}{status_marker}")

            except Exception as e:
                self.logger.warning(f"  Failed to load iteration {iter_num}: {e}")
                continue

        self.logger.info(f"Successfully loaded {len(self.iteration_history)} iteration(s) history")

    def _determine_iteration_role(
        self,
        iteration: int,
        ai_suggestion: Optional[Dict[str, Any]] = None
    ) -> Tuple['IterationRole', Optional[int], str]:
        """
        Determine the role for this iteration based on STRICT PHASED PROTOCOL.

        PHASED PROTOCOL:
        - PHASE 1: Reproduce iteration 12 exactly (no config changes)
        - PHASE 2: Threshold calibration only (no training)

        Args:
            iteration: Current iteration number
            ai_suggestion: Optional AI suggestion (for logging only - cannot override)

        Returns:
            Tuple of (role, parent_iteration, reasoning)
        """
        if not ROLE_BASED_LOOP:
            self.logger.warning("Role-based loop not available, using default behavior")
            return (None, None, "Role-based loop disabled")

        # Build iteration history in format expected by determine_iteration_role
        history_for_role = []
        for hist in self.iteration_history:
            history_for_role.append({
                'iteration': hist.get('iteration', 0),
                'metrics': {
                    'avg_auc': hist.get('avg_auc', 0.0),
                    'avg_f1': hist.get('avg_f1', 0.0),
                    'avg_recall': hist.get('avg_recall', 0.0),
                    'avg_precision': hist.get('avg_precision', 0.0)
                },
                'phase': hist.get('phase'),
                'role': hist.get('role')
            })

        # Current metrics from last iteration
        current_metrics = {}
        if history_for_role:
            current_metrics = history_for_role[-1].get('metrics', {})

        # Determine role using phased protocol
        role, parent, reasoning = determine_iteration_role(
            current_iteration=iteration,
            current_metrics=current_metrics,
            iteration_history=history_for_role,
            ai_suggestion=ai_suggestion
        )

        return role, parent, reasoning

    def _get_current_phase(self) -> Tuple['OptimizationPhase', Dict[str, Any]]:
        """
        Get the current optimization phase.

        Returns:
            Tuple of (phase, phase_context)
        """
        if not PHASED_PROTOCOL_ENABLED:
            return (None, {"reason": "Phased protocol not enabled"})

        # Build iteration history
        history_for_phase = []
        for hist in self.iteration_history:
            history_for_phase.append({
                'iteration': hist.get('iteration', 0),
                'metrics': {
                    'avg_auc': hist.get('avg_auc', 0.0),
                    'avg_f1': hist.get('avg_f1', 0.0),
                    'avg_recall': hist.get('avg_recall', 0.0),
                    'avg_precision': hist.get('avg_precision', 0.0)
                },
                'phase': hist.get('phase'),
                'role': hist.get('role')
            })

        return determine_current_phase(history_for_phase)

    def _get_phase1_config(self, iteration: int = 0) -> Dict[str, Any]:
        """
        Get the EXACT iteration 12 config for Phase 1 reproduction.

        This returns the frozen config - NO CHANGES ALLOWED except random seed.

        Args:
            iteration: Current iteration number (for metadata)
        """
        if not PHASED_PROTOCOL_ENABLED:
            self.logger.warning("Phased protocol not enabled")
            return {}

        return get_phase1_exact_config(iteration=iteration)

    def _log_phase_status(self, phase: 'OptimizationPhase', phase_context: Dict[str, Any]):
        """Log the current phase status prominently."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 OPTIMIZATION PHASE: {phase.value if phase else 'UNKNOWN'}")
        self.logger.info("=" * 80)

        if phase == OptimizationPhase.PHASE_5_AUC_IMPROVEMENT:
            self.logger.info(f"   🎯 CLASS-SPECIFIC AUC IMPROVEMENT")
            self.logger.info(f"   Baseline: Iteration {phase_context.get('baseline_iteration', PHASE_5_BASELINE_ITERATION)}")
            self.logger.info(f"   Baseline Macro AUC: {phase_context.get('baseline_macro_auc', PHASE_5_BASELINE_MACRO_AUC):.4f}")
            self.logger.info(f"   Target Diseases: {', '.join(phase_context.get('target_diseases', PHASE_5_TARGET_DISEASES))}")
            self.logger.info(f"   Stable Diseases: {', '.join(phase_context.get('stable_diseases', PHASE_5_STABLE_DISEASES))}")
            self.logger.info(f"   Attempts: {phase_context.get('attempts', 0)}")
            if 'iterations_since_improvement' in phase_context:
                self.logger.info(f"   Iterations since improvement: {phase_context['iterations_since_improvement']}")
            self.logger.info("   ALLOWED: auc_improvement block changes ONLY")
            self.logger.info("   FORBIDDEN: Loss changes, class weights, threshold optimization")
            self.logger.info("   FOCUS: Ranking improvement (F1 is OUT OF SCOPE)")

        elif phase == OptimizationPhase.PHASE_1_REPRODUCE:
            target_auc = PHASE_CRITERIA.phase1_auc_target - PHASE_CRITERIA.phase1_auc_tolerance
            self.logger.info(f"   🔬 EXACT REPRODUCTION MODE (ARCHIVED)")
            self.logger.info(f"   Goal: Reproduce Iteration 12 AUC ({PHASE_CRITERIA.phase1_auc_target:.4f})")
            self.logger.info(f"   Success Criteria: AUC >= {target_auc:.4f}")
            self.logger.info(f"   Attempts: {phase_context.get('attempts', 0)}/{PHASE_CRITERIA.phase1_max_attempts}")
            if 'best_auc_so_far' in phase_context:
                self.logger.info(f"   Best AUC so far: {phase_context['best_auc_so_far']:.4f}")
            if 'best_f1_so_far' in phase_context:
                self.logger.info(f"   Best F1 so far: {phase_context['best_f1_so_far']:.4f}")
            if phase_context.get('last_failure'):
                self.logger.warning(f"   ⚠️  Last failure: {phase_context['last_failure']}")
            self.logger.info("   CONFIG (EXACT MATCH TO ITERATION 12):")
            self.logger.info("     - use_additional_features: TRUE")
            self.logger.info("     - early_stopping: ENABLED (patience=5)")
            self.logger.info("     - rare_class.enabled: TRUE")
            self.logger.info("     - threshold_optimization: per_class_f1_score")

        elif phase == OptimizationPhase.PHASE_2_CALIBRATE:
            self.logger.info(f"   Goal: Improve F1 via threshold calibration")
            self.logger.info(f"   Phase 1 AUC: {phase_context.get('phase1_auc', 'N/A')}")
            self.logger.info(f"   Current F1: {phase_context.get('current_f1', 'N/A')}")
            self.logger.info(f"   Target F1: {PHASE_CRITERIA.phase2_f1_target:.4f}")
            self.logger.info("   ALLOWED: Threshold optimization ONLY")
            self.logger.info("   FORBIDDEN: ANY retraining, loss changes, LR changes")

        elif phase == OptimizationPhase.STOP_AND_DEBUG:
            self.logger.warning(f"   ⚠️  STOP AND DEBUG: {phase_context.get('reason', 'Unknown')}")
            if 'action' in phase_context:
                self.logger.warning(f"   ACTION: {phase_context['action']}")

        elif phase == OptimizationPhase.SUCCESS:
            self.logger.info("   🎉 SUCCESS! All criteria met!")
            self.logger.info(f"   Final AUC: {phase_context.get('final_auc', 'N/A')}")
            self.logger.info(f"   Final F1: {phase_context.get('final_f1', 'N/A')}")

        self.logger.info("=" * 80)
        self.logger.info("")

    def _compute_per_class_metrics(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Extract per-class metrics from results DataFrame.

        Args:
            results_df: Results DataFrame with per-class metrics

        Returns:
            Dict mapping disease name to its metrics
        """
        per_class_metrics = {}

        for _, row in results_df.iterrows():
            disease = row.get('Label', 'Unknown')
            per_class_metrics[disease] = {
                'auc': float(row.get('AUC', 0.0)),
                'f1': float(row.get('F1_Score', 0.0)),
                'precision': float(row.get('Precision', 0.0)),
                'recall': float(row.get('Recall', 0.0)),
                'threshold': float(row.get('Threshold', 0.5)),
                'tp': int(row.get('TP', 0)),
                'fp': int(row.get('FP', 0)),
                'fn': int(row.get('FN', 0)),
                'tn': int(row.get('TN', 0))
            }

        return per_class_metrics

    def _compute_group_metrics_from_results(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute per-group (rare/moderate/common) metrics.

        Args:
            results_df: Results DataFrame

        Returns:
            Dict mapping group name to aggregate metrics
        """
        if not ROLE_BASED_LOOP:
            return {}

        per_class_metrics = self._compute_per_class_metrics(results_df)
        return compute_group_metrics(per_class_metrics)

    def _log_phase5_iteration(self, iteration_summary: Dict[str, Any]):
        """
        Log a Phase 5 iteration to phase5_log.md.

        Args:
            iteration_summary: Summary of the completed iteration
        """
        import datetime

        iteration = iteration_summary.get('iteration', 0)
        parent_iteration = iteration_summary.get('parent_iteration', PHASE_5_BASELINE_ITERATION)
        avg_auc = iteration_summary.get('avg_auc', 0.0)
        suggested_changes = iteration_summary.get('suggested_changes', {})

        # Get per-class AUC if available
        per_class_auc = iteration_summary.get('per_class_auc', {})

        # Compute AUC deltas vs baseline
        auc_deltas = {}
        if per_class_auc:
            delta_analysis = compute_phase5_auc_deltas(per_class_auc)
            for disease, info in delta_analysis.items():
                auc_deltas[disease] = info['delta']

        # Determine decision from AI
        decision = suggested_changes.get('decision', 'CONTINUE_TARGETED_AUC_IMPROVEMENT')

        # Check stable diseases
        stable_check = "PASSED"
        for disease in PHASE_5_STABLE_DISEASES:
            if disease in auc_deltas and auc_deltas[disease] <= -PHASE_5_CONFIG.max_regression:
                stable_check = "FAILED"
                break

        # Format log entry
        log_entry = f"""
## Iteration {iteration}
- **Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Parent iteration**: {parent_iteration}
- **Target diseases**: {', '.join(PHASE_5_TARGET_DISEASES)}
- **Macro AUC**: {avg_auc:.4f} (baseline: {PHASE_5_BASELINE_MACRO_AUC:.4f}, delta: {avg_auc - PHASE_5_BASELINE_MACRO_AUC:+.4f})

### AUC Change Per Disease:
| Disease | Baseline | Current | Delta | Status |
|---------|----------|---------|-------|--------|
"""
        # Add per-class AUC rows
        for disease in sorted(PHASE_5_BASELINE_PER_CLASS_AUC.keys()):
            baseline = PHASE_5_BASELINE_PER_CLASS_AUC[disease]
            current = per_class_auc.get(disease, baseline)
            delta = current - baseline
            is_target = disease in PHASE_5_TARGET_DISEASES
            is_stable = disease in PHASE_5_STABLE_DISEASES

            if delta >= PHASE_5_CONFIG.meaningful_improvement:
                status = "IMPROVED"
            elif delta <= -PHASE_5_CONFIG.max_regression:
                status = "REGRESSED"
            else:
                status = "stable"

            category = ""
            if is_target:
                category = " (Target)"
            elif is_stable:
                category = " (Stable)"

            log_entry += f"| {disease}{category} | {baseline:.4f} | {current:.4f} | {delta:+.4f} | {status} |\n"

        log_entry += f"""
- **Stable diseases regression check**: {stable_check}
- **Decision from AI advisory**: {decision}
- **Status**: {'STOP' if decision == 'STOP_PHASE_5_AND_FREEZE' else 'CONTINUE'}

---
"""
        # Append to phase5_log.md
        log_path = os.path.join(os.path.dirname(self.output_dir), "phase5_log.md")
        try:
            with open(log_path, 'a') as f:
                f.write(log_entry)
            self.logger.info(f"📝 Phase 5 log updated: {log_path}")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to update phase5_log.md: {e}")

    def _extract_rare_class_details(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Extract detailed metrics for rare classes.

        Args:
            results_df: Results DataFrame

        Returns:
            Dict with rare class details
        """
        if not ROLE_BASED_LOOP:
            return {}

        rare_classes = DISEASE_GROUPS.get('rare', [])
        rare_details = {}

        for _, row in results_df.iterrows():
            disease = row.get('Label', '')
            if disease in rare_classes:
                rare_details[disease] = {
                    'auc': float(row.get('AUC', 0.0)),
                    'f1': float(row.get('F1_Score', 0.0)),
                    'recall': float(row.get('Recall', 0.0)),
                    'precision': float(row.get('Precision', 0.0)),
                    'threshold': float(row.get('Threshold', 0.5)),
                    'tp': int(row.get('TP', 0)),
                    'fp': int(row.get('FP', 0)),
                    'fn': int(row.get('FN', 0))
                }

        return rare_details

    def _create_iteration_log(
        self,
        iteration: int,
        role: 'IterationRole',
        parent_iteration: Optional[int],
        results_df: pd.DataFrame,
        training_metadata: Dict[str, Any],
        decision: Dict[str, Any],
        ai_corrections: List[str] = None
    ) -> IterationLog:
        """
        Create a structured iteration log for thesis documentation.

        Args:
            iteration: Iteration number
            role: The iteration role
            parent_iteration: Parent iteration number
            results_df: Results DataFrame
            training_metadata: Training metadata
            decision: Decision made for next iteration
            ai_corrections: List of corrections applied to AI suggestions

        Returns:
            IterationLog instance
        """
        # Compute metrics
        macro_auc = float(results_df['AUC'].mean())
        macro_f1 = float(results_df['F1_Score'].mean())

        # Compute deltas vs reference iterations
        if ROLE_BASED_LOOP:
            auc_vs_iter12 = macro_auc - ITERATION_12_AUC_ANCHOR.macro_auc
            f1_vs_iter58 = macro_f1 - ITERATION_58_F1_ANCHOR.macro_f1
        else:
            auc_vs_iter12 = 0.0
            f1_vs_iter58 = 0.0

        # Create the log
        log = IterationLog(
            iteration=iteration,
            role=role.value if role else "UNKNOWN",
            parent_iteration=parent_iteration,
            timestamp=datetime.datetime.now().isoformat(),
            macro_auc=macro_auc,
            macro_f1=macro_f1,
            auc_vs_iter12=auc_vs_iter12,
            f1_vs_iter58=f1_vs_iter58,
            per_group_metrics=self._compute_group_metrics_from_results(results_df),
            rare_class_details=self._extract_rare_class_details(results_df),
            distribution_diagnostics={},  # Would need prediction data
            decision=decision,
            training_metadata=training_metadata,
            ai_corrections=ai_corrections or []
        )

        return log

    def _save_iteration_log(self, log: IterationLog, iteration_dir: str) -> str:
        """
        Save iteration log to JSON file.

        Args:
            log: IterationLog instance
            iteration_dir: Directory to save the log

        Returns:
            Path to saved log file
        """
        log_file = os.path.join(iteration_dir, "iteration_log.json")

        with open(log_file, 'w') as f:
            json.dump(log.to_dict(), f, indent=2)

        self.logger.info(f"📝 Iteration log saved to {log_file}")
        return log_file

    def _log_thesis_documentation_marker(
        self,
        iteration: int,
        role: 'IterationRole',
        parent_iteration: Optional[int],
        macro_auc: float,
        macro_f1: float,
        decision: str
    ):
        """
        Log the thesis documentation marker.

        Args:
            iteration: Iteration number
            role: Iteration role
            parent_iteration: Parent iteration number
            macro_auc: Macro AUC score
            macro_f1: Macro F1 score
            decision: Decision made
        """
        if ROLE_BASED_LOOP:
            marker = format_thesis_documentation_marker(
                iteration, role, parent_iteration, macro_auc, macro_f1, decision
            )
            self.logger.info(marker)
        else:
            self.logger.info(f"Iteration {iteration} complete: AUC={macro_auc:.4f}, F1={macro_f1:.4f}")

    def run_single_iteration(self, config_path: str, iteration: int) -> Dict[str, Any]:
        """
        Run a single training iteration

        Args:
            config_path: Path to configuration file
            iteration: Iteration number

        Returns:
            Dictionary with iteration results
        """
        iteration_start_time = time.time()

        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"🚀 ITERATION {iteration} STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)

        # Determine iteration role (ROLE-BASED STRATEGY)
        iteration_role = None
        parent_iteration = None
        role_reasoning = ""
        ai_corrections = []

        # Check if config specifies HEAD_UPGRADE, SMOTE_HEAD, or REPRESENTATION_FINETUNE - skip role-based logic entirely
        head_upgrade_mode = False
        smote_head_mode = False
        representation_finetune_mode = False
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            config_phase = config_data.get('metadata', {}).get('phase', '')
            config_metadata = config_data.get('metadata', {})

            if config_phase == 'HEAD_UPGRADE':
                head_upgrade_mode = True
                parent_iteration = config_data.get('metadata', {}).get('parent_iteration', 12)
                self.logger.info("")
                self.logger.info("="*80)
                self.logger.info("🔧 HEAD_UPGRADE MODE - PHASE 5 / TARGETED_AUC DISABLED")
                self.logger.info("="*80)
                self.logger.info(f"   Config phase: HEAD_UPGRADE")
                self.logger.info(f"   Parent iteration: {parent_iteration}")
                self.logger.info(f"   Role-based loop: BYPASSED")
                self.logger.info(f"   Phase 5 heuristics: DISABLED")
                self.logger.info("="*80)

            elif config_phase == 'SMOTE_HEAD':
                smote_head_mode = True
                parent_iteration = config_data.get('metadata', {}).get('parent_iteration', 91)
                smote_config = config_data.get('smote', {})
                self.logger.info("")
                self.logger.info("="*80)
                self.logger.info("🧬 SMOTE_HEAD MODE - FEATURE-SPACE SMOTE EXPERIMENT")
                self.logger.info("="*80)
                self.logger.info(f"   Config phase: SMOTE_HEAD")
                self.logger.info(f"   Parent iteration: {parent_iteration}")
                self.logger.info(f"   SMOTE target class: {smote_config.get('target_class', 'Hernia')}")
                self.logger.info(f"   SMOTE sampling ratio: {smote_config.get('sampling_ratio', 4)}x")
                self.logger.info(f"   Role-based loop: BYPASSED")
                self.logger.info(f"   Phase 5 heuristics: DISABLED")
                self.logger.info("="*80)

            elif config_phase == 'REPRESENTATION_FINETUNE' or config_metadata.get('role_override', False):
                # REPRESENTATION_FINETUNE: Controlled experiment with explicit anchor override
                representation_finetune_mode = True
                anchor_iteration = config_metadata.get('anchor_iteration', config_metadata.get('parent_iteration', 89))
                parent_iteration = anchor_iteration
                unfreeze_blocks = config_data.get('model', {}).get('unfreeze_last_blocks', 1)
                self.logger.info("")
                self.logger.info("="*80)
                self.logger.info("🔬 REPRESENTATION_FINETUNE MODE - CONTROLLED BACKBONE EXPERIMENT")
                self.logger.info("="*80)
                self.logger.info(f"   Config phase: {config_phase}")
                self.logger.info(f"   Role override: ACTIVE (bypasses all phase logic)")
                self.logger.info(f"   Anchor iteration: {anchor_iteration}")
                self.logger.info(f"   Parent iteration: {parent_iteration}")
                self.logger.info(f"   Unfreeze last blocks: {unfreeze_blocks}")
                self.logger.info(f"   Role-based loop: BYPASSED")
                self.logger.info(f"   Phase 5 heuristics: DISABLED")
                self.logger.info(f"   TARGETED_AUC: DISABLED")
                self.logger.info("="*80)

        # Skip role-based logic for HEAD_UPGRADE, SMOTE_HEAD, or REPRESENTATION_FINETUNE modes
        bypass_role_based = head_upgrade_mode or smote_head_mode or representation_finetune_mode

        if ROLE_BASED_LOOP and not bypass_role_based:
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("🎯 DETERMINING ITERATION ROLE")
            self.logger.info("="*80)

            iteration_role, parent_iteration, role_reasoning = self._determine_iteration_role(iteration)

            self.logger.info(f"   Role: {iteration_role.value if iteration_role else 'N/A'}")
            self.logger.info(f"   Parent: Iteration {parent_iteration if parent_iteration else 'N/A'}")
            self.logger.info(f"   Reasoning: {role_reasoning}")
            self.logger.info("="*80)

            # Check for ABORT condition
            if iteration_role == IterationRole.ABORT:
                self.logger.info("")
                self.logger.info("="*80)
                self.logger.info("🛑 ABORT CONDITION DETECTED")
                self.logger.info("="*80)
                self.logger.info(f"   Reason: {role_reasoning}")
                self.logger.info("="*80)

                # Return a special summary indicating abort
                return {
                    'iteration': iteration,
                    'status': 'aborted',
                    'abort_reason': role_reasoning,
                    'avg_auc': 0.0,
                    'avg_f1': 0.0,
                    'ai_analysis_status': 'skipped'
                }

        # Get current task from task manager
        current_task = self.task_manager.get_current_task()
        if current_task:
            self.logger.info("")
            self.logger.info("📋 CURRENT TASK:")
            self.logger.info(f"   ID: {current_task['task_id']}")
            self.logger.info(f"   Description: {current_task['description']}")
            self.logger.info(f"   Target Metric: {current_task['target_metric']}")
            self.logger.info(f"   Progress: {len(current_task['evaluation_history'])}/{current_task['required_iterations']} iterations")
            self.logger.info("="*80)
        else:
            self.logger.info("")
            self.logger.info("📋 No active task - AI will suggest new task")
            self.logger.info("="*80)

        # Send Telegram notification - iteration start
        end_iteration = self.start_iteration + self.max_iterations - 1
        self.telegram.send_iteration_start(iteration, end_iteration, config_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration:03d}")
        os.makedirs(iteration_dir, exist_ok=True)

        try:
            # Phase 1: Training
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"📚 [Iteration {iteration}] PHASE 1/4: TRAINING")
            self.logger.info("="*80)
            phase_start = time.time()

            trainer = ConfigBasedTrainer(config_path, timestamp, self.logger, self.telegram, iteration)

            # Pass best tracker and AI advisor for intelligent warm start
            trainer.best_iteration_tracker = self.best_iteration_tracker
            trainer.ai_advisor = self.ai_advisor
            trainer.task_manager = self.task_manager

            # CRITICAL: Pass role-enforced parent iteration for warm start
            # This OVERRIDES AI warm start recommendation when set
            # SKIP for HEAD_UPGRADE or SMOTE_HEAD mode - backbone loading handled separately
            if ROLE_BASED_LOOP and iteration_role and parent_iteration and not bypass_role_based:
                trainer.role_enforced_parent = parent_iteration
                trainer.iteration_role = iteration_role
                self.logger.info(f"   [ROLE-BASED] Warm start will use iteration {parent_iteration} (enforced by {iteration_role.value} role)")

            # Enforce role-specific configuration (ROLE-BASED STRATEGY)
            # SKIP for HEAD_UPGRADE or SMOTE_HEAD mode
            if ROLE_BASED_LOOP and iteration_role and not bypass_role_based:
                trainer.enforce_role_config(iteration_role)

            # ================================================================
            # PHASE 2: Pass Phase 1 model path for threshold-only mode
            # ================================================================
            if PHASED_PROTOCOL_ENABLED and iteration_role == IterationRole.ADJUST_THRESHOLDS:
                # Get the Phase 1 model path from phase context
                _, phase_context = self._get_current_phase()
                phase1_model_path = phase_context.get('phase1_model_path')
                phase1_iteration = phase_context.get('phase1_iteration')

                if phase1_model_path:
                    trainer.phase1_model_path = phase1_model_path
                    self.logger.info(f"   [PHASE 2] Using Phase 1 model: {phase1_model_path}")
                elif phase1_iteration:
                    # Try to find the model path from the iteration directory
                    phase1_dir = os.path.join(self.output_dir, f"iteration_{phase1_iteration:03d}")
                    if os.path.exists(phase1_dir):
                        # Find the model file in the directory
                        for f in os.listdir(phase1_dir):
                            if f.startswith('pipeline_model_') and f.endswith('.pth'):
                                trainer.phase1_model_path = os.path.join(phase1_dir, f)
                                self.logger.info(f"   [PHASE 2] Found Phase 1 model: {trainer.phase1_model_path}")
                                break
                    if not trainer.phase1_model_path:
                        self.logger.error(f"   [PHASE 2] Could not find Phase 1 model in iteration {phase1_iteration}")
                else:
                    self.logger.error("   [PHASE 2] No Phase 1 iteration info available!")

            model, train_loss, val_loss, training_metadata = trainer.train()

            phase_time = time.time() - phase_start
            self.logger.info(f"✅ Training phase completed in {phase_time/60:.1f} minutes")
            self.logger.info(f"   Final Train Loss: {train_loss:.4f}")
            self.logger.info(f"   Final Val Loss: {val_loss:.4f}")
            self.logger.info(f"   Actual Epochs: {training_metadata.get('actual_epochs', 'N/A')}")

            model_file = trainer.model_file
            results_file = f"pipeline_results_{timestamp}.csv"
            comparison_file = f"baseline_comparison_{timestamp}.csv"

            # Phase 2: Testing
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"🧪 [Iteration {iteration}] PHASE 2/4: TESTING")
            self.logger.info("="*80)
            phase_start = time.time()

            self.telegram.add_log(f"Testing phase started for iteration {iteration}")
            # Pass head_config for MLP head models
            head_config = trainer.config.get('model', {}).get('head')
            test_model_pipeline(
                model_file=model_file,
                results_file=results_file,
                logger=self.logger,
                head_config=head_config
            )

            phase_time = time.time() - phase_start
            self.logger.info(f"✅ Testing phase completed in {phase_time/60:.1f} minutes")
            self.telegram.add_log(f"Testing completed in {phase_time/60:.1f} minutes")

            # Phase 3: Baseline Comparison
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"📊 [Iteration {iteration}] PHASE 3/4: BASELINE COMPARISON")
            self.logger.info("="*80)
            phase_start = time.time()

            comparison_df = self.compare_with_baseline(results_file, comparison_file)

            phase_time = time.time() - phase_start
            self.logger.info(f"✅ Comparison phase completed in {phase_time:.1f} seconds")

            # Load results
            results_df = pd.read_csv(results_file)

            # Find confusion matrix and threshold files
            confusion_file = results_file.replace('pipeline_results_', 'confusion_matrix_').replace('.csv', '.json')
            threshold_file = model_file.replace('pipeline_model_', 'thresholds_').replace('.pth', '.json')

            # Move files to iteration directory BEFORE AI analysis
            # This ensures results are saved even if AI advisor fails
            self._move_iteration_files(iteration_dir, model_file, results_file, comparison_file, config_path, confusion_file, threshold_file)

            # Collect iteration summary BEFORE AI analysis
            # This ensures we have the data even if AI fails
            # Extract per-class AUC for Phase 5 tracking
            per_class_auc = {row['Label']: row['AUC'] for _, row in results_df.iterrows()}

            iteration_summary = {
                'iteration': iteration,
                'timestamp': timestamp,
                'config_path': config_path,
                'model_file': model_file,
                'results_file': results_file,
                'comparison_file': comparison_file,
                'analysis_file': None,  # Will be updated if AI analysis succeeds
                'avg_auc': results_df['AUC'].mean(),
                'avg_f1': results_df['F1_Score'].mean(),
                'avg_recall': results_df['Recall'].mean(),
                'avg_precision': results_df['Precision'].mean(),
                'avg_accuracy': results_df['Accuracy'].mean(),
                'per_class_auc': per_class_auc,  # For Phase 5 delta tracking
                'train_loss': float(train_loss),  # Final training loss
                'val_loss': float(val_loss),  # Final validation loss
                'suggested_changes': None,  # Will be updated if AI analysis succeeds
                'directory': iteration_dir,
                'total_time': time.time() - iteration_start_time,
                'ai_analysis_status': 'pending',
                # Training metadata from config_based_pipeline
                'actual_epochs': training_metadata.get('actual_epochs', 0),
                'thresholds': training_metadata.get('thresholds', {}),
                'description': training_metadata.get('description', ''),
                'early_stopping_monitor': training_metadata.get('early_stopping_monitor'),
                'early_stopping_mode': training_metadata.get('early_stopping_mode'),
                'min_delta': training_metadata.get('min_delta'),
                'patience': training_metadata.get('patience'),
                # Model path for Phase 2 tracking
                'model_path': os.path.join(iteration_dir, model_file) if model_file else None,
            }

            # Add phase info to summary if phased protocol is enabled
            if PHASED_PROTOCOL_ENABLED:
                current_phase, _ = self._get_current_phase()
                iteration_summary['phase'] = current_phase.value if current_phase else 'UNKNOWN'
                iteration_summary['role'] = iteration_role.value if iteration_role else 'UNKNOWN'

            # Save iteration summary to file immediately (before AI call)
            self._save_iteration_summary(iteration_summary, iteration_dir)

            # Update task progress if there's an active task
            if current_task:
                target_metric = current_task['target_metric']
                # Map target_metric to iteration_summary key
                metric_key_map = {
                    'f1': 'avg_f1',
                    'auc': 'avg_auc',
                    'recall': 'avg_recall',
                    'precision': 'avg_precision',
                    'accuracy': 'avg_accuracy'
                }
                metric_key = metric_key_map.get(target_metric, 'avg_f1')
                metric_value = iteration_summary.get(metric_key, 0.0)

                self.task_manager.update_task_progress(
                    task_id=current_task['task_id'],
                    iteration=iteration,
                    metric_value=metric_value,
                    iteration_summary=iteration_summary
                )

                # Evaluate task if enough iterations completed
                if len(current_task['evaluation_history']) >= current_task['required_iterations']:
                    task_status = self.task_manager.evaluate_task(current_task['task_id'])
                    self.logger.info("")
                    self.logger.info("="*80)
                    self.logger.info(f"📊 TASK EVALUATION")
                    self.logger.info("="*80)
                    self.logger.info(f"   Task: {current_task['description']}")
                    self.logger.info(f"   Status: {task_status.upper()}")
                    self.logger.info(f"   Iterations: {current_task['required_iterations']}")
                    self.logger.info("="*80)

                    if self.telegram:
                        self.telegram.add_log(f"Task '{current_task['task_id']}' {task_status}")

            # Update best iteration tracker with task-aware tracking
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"📊 UPDATING BEST ITERATION TRACKER")
            self.logger.info("="*80)

            # Prepare task info for best tracker
            task_info_for_tracker = None
            if current_task:
                task_info_for_tracker = {
                    'target_metric': current_task.get('target_metric', 'avg_f1'),
                    'target_value': current_task.get('target_value'),
                    'constraints': current_task.get('constraints', {})
                }

            # Update tracker with iteration results (including role info)
            self.best_iteration_tracker.update(
                iteration=iteration,
                metrics={
                    'avg_f1': iteration_summary['avg_f1'],
                    'avg_auc': iteration_summary['avg_auc'],
                    'avg_recall': iteration_summary['avg_recall'],
                    'avg_precision': iteration_summary['avg_precision'],
                    'avg_accuracy': iteration_summary['avg_accuracy']
                },
                config=trainer.config,
                model_path=model_file,
                task_info=task_info_for_tracker,
                role=iteration_role.value if iteration_role else None,
                parent_iteration=parent_iteration
            )

            # Log summary
            self.logger.info(self.best_iteration_tracker.get_summary_report())

            # Phase 4: AI Analysis (this can fail, but iteration data is already saved)
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"🤖 [Iteration {iteration}] PHASE 4/4: AI ANALYSIS")
            self.logger.info("="*80)
            phase_start = time.time()

            self.telegram.add_log(f"AI analysis started for iteration {iteration}")

            # Try to get AI analysis, but don't fail the iteration if it doesn't work
            try:
                # Get task information for AI advisor
                task_info = self.task_manager.get_task_for_advisor()

                # Get formatted iteration data for AI advisor
                last_3_summaries = self.iteration_evaluator.get_last_n_summaries(3)
                iteration_analysis = self.iteration_evaluator.format_for_advisor(last_3_summaries)

                # Pass previous iteration history to AI advisor (excluding current iteration)
                # Also pass role info for role-aware analysis
                analysis, suggested_changes = self.ai_advisor.analyze_results(
                    config=trainer.config,
                    results_df=results_df,
                    comparison_df=comparison_df,
                    iteration=iteration,
                    iteration_history=self.iteration_history,  # Pass history of completed iterations
                    iteration_role=iteration_role,  # Pass current role for role-aware analysis
                    parent_iteration=parent_iteration,  # Pass parent iteration for context
                    train_loss=train_loss,  # Pass final training loss
                    val_loss=val_loss,  # Pass final validation loss
                    task_info=task_info,  # Pass task registry information
                    iteration_analysis=iteration_analysis  # Pass detailed iteration analysis
                )

                # Save analysis
                analysis_file = os.path.join(iteration_dir, f"ai_analysis_{iteration:03d}.txt")
                with open(analysis_file, 'w') as f:
                    f.write(analysis)

                # Update iteration summary with AI results
                iteration_summary['analysis_file'] = analysis_file
                iteration_summary['suggested_changes'] = suggested_changes
                iteration_summary['ai_analysis_status'] = 'completed'

                phase_time = time.time() - phase_start
                self.logger.info(f"✅ AI analysis phase completed in {phase_time:.1f} seconds")

            except Exception as e:
                self.logger.error(f"⚠️ AI analysis failed: {str(e)}")
                self.logger.error("Iteration results are saved, but no AI suggestions available")

                # Mark AI analysis as failed but continue
                iteration_summary['ai_analysis_status'] = 'failed'
                iteration_summary['ai_error'] = str(e)

                # Save error info to file
                error_file = os.path.join(iteration_dir, f"ai_analysis_error_{iteration:03d}.txt")
                with open(error_file, 'w') as f:
                    f.write(f"AI Analysis Failed\n")
                    f.write(f"Error: {str(e)}\n\n")
                    import traceback
                    f.write(traceback.format_exc())

            # Update saved iteration summary with final status
            iteration_summary['total_time'] = time.time() - iteration_start_time
            self._save_iteration_summary(iteration_summary, iteration_dir)

            # Create and save structured iteration log (ROLE-BASED STRATEGY)
            if ROLE_BASED_LOOP and iteration_role:
                # Determine next action for decision field
                decision_action = "CONTINUE"
                if iteration_role == IterationRole.TRAIN_AUC:
                    decision_action = "CONTINUE_AUC_TRAINING"
                elif iteration_role == IterationRole.RECOVER_F1:
                    decision_action = "CONTINUE_F1_RECOVERY"

                decision = {
                    'action': decision_action,
                    'reasoning': role_reasoning,
                    'stop_condition_met': False
                }

                iteration_log = self._create_iteration_log(
                    iteration=iteration,
                    role=iteration_role,
                    parent_iteration=parent_iteration,
                    results_df=results_df,
                    training_metadata=training_metadata,
                    decision=decision,
                    ai_corrections=ai_corrections
                )

                self._save_iteration_log(iteration_log, iteration_dir)

                # Add role info to iteration summary
                iteration_summary['role'] = iteration_role.value
                iteration_summary['parent_iteration'] = parent_iteration
                iteration_summary['role_reasoning'] = role_reasoning
                iteration_summary['ai_corrections'] = ai_corrections

            # Iteration completion summary
            iteration_time = time.time() - iteration_start_time
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"✅ ITERATION {iteration} COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"📊 RESULTS SUMMARY:")
            self.logger.info(f"   Average AUC:       {iteration_summary['avg_auc']:.4f}")
            self.logger.info(f"   Average F1:        {iteration_summary['avg_f1']:.4f}")
            self.logger.info(f"   Average Recall:    {iteration_summary['avg_recall']:.4f}")
            self.logger.info(f"   Average Precision: {iteration_summary['avg_precision']:.4f}")
            self.logger.info(f"   Average Accuracy:  {iteration_summary['avg_accuracy']:.4f}")
            self.logger.info(f"   AI Analysis:       {iteration_summary.get('ai_analysis_status', 'unknown')}")
            self.logger.info("")
            self.logger.info(f"⏱️  TIMING:")
            self.logger.info(f"   Total iteration time: {iteration_time/60:.1f} minutes")
            self.logger.info("")
            self.logger.info(f"📁 OUTPUT:")
            self.logger.info(f"   Results directory: {iteration_dir}")

            # Only show AI analysis file if it exists
            if iteration_summary.get('analysis_file'):
                self.logger.info(f"   AI analysis: {os.path.basename(iteration_summary['analysis_file'])}")
            elif iteration_summary.get('ai_analysis_status') == 'failed':
                self.logger.info(f"   AI analysis: FAILED (see ai_analysis_error_{iteration:03d}.txt)")

            # Show role info if available
            if ROLE_BASED_LOOP and iteration_role:
                self.logger.info(f"   Role: {iteration_role.value}")
                self.logger.info(f"   Parent: Iteration {parent_iteration if parent_iteration else 'N/A'}")

            self.logger.info("="*80)

            # Log thesis documentation marker (ROLE-BASED STRATEGY)
            if ROLE_BASED_LOOP and iteration_role:
                self._log_thesis_documentation_marker(
                    iteration=iteration,
                    role=iteration_role,
                    parent_iteration=parent_iteration,
                    macro_auc=iteration_summary['avg_auc'],
                    macro_f1=iteration_summary['avg_f1'],
                    decision=role_reasoning
                )

            # Show AI suggestions preview
            if iteration_summary.get('suggested_changes') and 'reasoning' in iteration_summary.get('suggested_changes', {}):
                self.logger.info("")
                self.logger.info("🤖 AI SUGGESTIONS PREVIEW:")
                self.logger.info(f"   {iteration_summary['suggested_changes'].get('reasoning', 'N/A')[:200]}...")
                if iteration_summary.get('analysis_file'):
                    self.logger.info(f"   (Full analysis in {os.path.basename(iteration_summary['analysis_file'])})")
                self.logger.info("="*80)

            # Send Telegram notification - iteration complete
            self.telegram.send_iteration_complete(
                iteration=iteration,
                total_iterations=end_iteration,
                avg_auc=iteration_summary['avg_auc'],
                avg_f1=iteration_summary['avg_f1'],
                avg_recall=iteration_summary['avg_recall'],
                avg_precision=iteration_summary['avg_precision'],
                time_minutes=iteration_time/60,
                train_loss=train_loss,
                val_loss=val_loss
            )

            return iteration_summary

        except Exception as e:
            self.logger.error(f"[Iteration {iteration}] Failed with error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Save whatever files exist to preserve partial results
            self.logger.info("⚠️  Attempting to save partial results from failed iteration...")

            try:
                import shutil
                import glob

                # Try to save any files that were created
                files_to_save = [
                    f"pipeline_model_{timestamp}.pth",
                    f"pipeline_results_{timestamp}.csv",
                    f"baseline_comparison_{timestamp}.csv",
                    f"confusion_matrix_{timestamp}.json",
                    f"thresholds_{timestamp}.json",
                    f"pipeline_log_{timestamp}.txt"
                ]

                saved_files = []
                for filename in files_to_save:
                    if os.path.exists(filename):
                        dest = os.path.join(iteration_dir, filename)
                        shutil.copy2(filename, dest)
                        saved_files.append(filename)
                        self.logger.info(f"   ✓ Saved {filename}")

                # Also copy the config
                if os.path.exists(config_path):
                    shutil.copy2(config_path, os.path.join(iteration_dir, "config.yaml"))
                    saved_files.append("config.yaml")

                # Save error information
                error_file = os.path.join(iteration_dir, f"ITERATION_FAILED_{iteration:03d}.txt")
                with open(error_file, 'w') as f:
                    f.write(f"Iteration {iteration} Failed\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"Error: {str(e)}\n\n")
                    f.write(f"Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Saved files:\n")
                    for fname in saved_files:
                        f.write(f"  - {fname}\n")

                self.logger.info(f"✓ Saved {len(saved_files)} partial file(s) to {iteration_dir}")
                self.logger.info(f"✓ Error details saved to {os.path.basename(error_file)}")

            except Exception as save_error:
                self.logger.error(f"⚠️  Failed to save partial results: {save_error}")

            # Send Telegram notification - iteration failed
            self.telegram.send_iteration_failed(iteration, str(e))

            # Re-raise to stop the loop
            raise

    def compare_with_baseline(self, results_file: str, comparison_file: str) -> pd.DataFrame:
        """Compare results with baseline"""
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file {results_file} not found")

        # Load our results
        our_results = pd.read_csv(results_file)

        # Create comparison
        comparison_data = []
        for _, row in our_results.iterrows():
            label = row['Label']
            baseline_metrics = BASELINE_RESULTS.get(label, {})

            def calc_improvement(our_val, baseline_val):
                if pd.isna(our_val) or pd.isna(baseline_val):
                    return np.nan, 'Equal'
                improvement = our_val - baseline_val
                better = 'Yes' if improvement > 0 else 'No' if improvement < 0 else 'Equal'
                return improvement, better

            # Calculate improvements for all metrics
            auc_imp, auc_better = calc_improvement(row.get('AUC', np.nan), baseline_metrics.get('AUC', np.nan))
            f1_imp, f1_better = calc_improvement(row.get('F1_Score', np.nan), baseline_metrics.get('F1_Score', np.nan))
            recall_imp, recall_better = calc_improvement(row.get('Recall', np.nan), baseline_metrics.get('Recall', np.nan))
            accuracy_imp, accuracy_better = calc_improvement(row.get('Accuracy', np.nan), baseline_metrics.get('Accuracy', np.nan))
            specificity_imp, specificity_better = calc_improvement(row.get('Specificity', np.nan), baseline_metrics.get('Specificity', np.nan))
            precision_imp, precision_better = calc_improvement(row.get('Precision', np.nan), baseline_metrics.get('Precision', np.nan))
            sensitivity_imp, sensitivity_better = calc_improvement(row.get('Sensitivity', np.nan), baseline_metrics.get('Sensitivity', np.nan))
            threshold_imp, threshold_better = calc_improvement(row.get('Threshold', np.nan), baseline_metrics.get('Threshold', np.nan))

            comparison_data.append({
                'Label': label,
                # AUC
                'Baseline_AUC': baseline_metrics.get('AUC', np.nan),
                'Our_AUC': row.get('AUC', np.nan),
                'AUC_Improvement': auc_imp,
                'Better_AUC': auc_better,
                # F1 Score
                'Baseline_F1_Score': baseline_metrics.get('F1_Score', np.nan),
                'Our_F1_Score': row.get('F1_Score', np.nan),
                'F1_Score_Improvement': f1_imp,
                'Better_F1_Score': f1_better,
                # Recall
                'Baseline_Recall': baseline_metrics.get('Recall', np.nan),
                'Our_Recall': row.get('Recall', np.nan),
                'Recall_Improvement': recall_imp,
                'Better_Recall': recall_better,
                # Accuracy
                'Baseline_Accuracy': baseline_metrics.get('Accuracy', np.nan),
                'Our_Accuracy': row.get('Accuracy', np.nan),
                'Accuracy_Improvement': accuracy_imp,
                'Better_Accuracy': accuracy_better,
                # Specificity
                'Baseline_Specificity': baseline_metrics.get('Specificity', np.nan),
                'Our_Specificity': row.get('Specificity', np.nan),
                'Specificity_Improvement': specificity_imp,
                'Better_Specificity': specificity_better,
                # Precision
                'Baseline_Precision': baseline_metrics.get('Precision', np.nan),
                'Our_Precision': row.get('Precision', np.nan),
                'Precision_Improvement': precision_imp,
                'Better_Precision': precision_better,
                # Sensitivity
                'Baseline_Sensitivity': baseline_metrics.get('Sensitivity', np.nan),
                'Our_Sensitivity': row.get('Sensitivity', np.nan),
                'Sensitivity_Improvement': sensitivity_imp,
                'Better_Sensitivity': sensitivity_better,
                # Threshold
                'Baseline_Threshold': baseline_metrics.get('Threshold', np.nan),
                'Our_Threshold': row.get('Threshold', np.nan),
                'Threshold_Improvement': threshold_imp,
                'Better_Threshold': threshold_better
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_file, index=False)

        return comparison_df

    def _move_iteration_files(self, iteration_dir, model_file, results_file, comparison_file, config_path, confusion_file=None, threshold_file=None):
        """Move iteration files to iteration directory"""
        import shutil

        # Move files to iteration directory (remove originals from CWD)
        for src in [model_file, results_file, comparison_file]:
            if os.path.exists(src):
                shutil.move(src, os.path.join(iteration_dir, os.path.basename(src)))
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(iteration_dir, "config.yaml"))
        if confusion_file and os.path.exists(confusion_file):
            shutil.move(confusion_file, os.path.join(iteration_dir, os.path.basename(confusion_file)))
            self.logger.info(f"📊 Confusion matrix moved to iteration directory")
        if threshold_file and os.path.exists(threshold_file):
            shutil.move(threshold_file, os.path.join(iteration_dir, os.path.basename(threshold_file)))
            self.logger.info(f"🎯 Optimized thresholds moved to iteration directory")

    def _save_iteration_summary(self, iteration_summary: Dict[str, Any], iteration_dir: str):
        """
        Save iteration summary to JSON file

        Args:
            iteration_summary: Dictionary containing iteration results
            iteration_dir: Directory to save summary
        """
        summary_file = os.path.join(iteration_dir, "iteration_summary.json")

        # Convert summary to JSON-serializable format
        json_summary = {}
        for key, value in iteration_summary.items():
            # Convert numpy/pandas types to Python native types
            if hasattr(value, 'item'):
                json_summary[key] = value.item()
            elif isinstance(value, (np.integer, np.floating)):
                json_summary[key] = value.item()
            else:
                json_summary[key] = value

        with open(summary_file, 'w') as f:
            json.dump(json_summary, f, indent=2)

        self.logger.debug(f"Saved iteration summary to {summary_file}")

    def run(self):
        """Run the auto-improvement loop with STRICT PHASED PROTOCOL"""
        self.logger.info("="*80)
        self.logger.info("AUTO-IMPROVEMENT LOOP STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Base config: {self.base_config_path}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Resume mode: {self.resume}")

        # Log phased protocol status
        if PHASED_PROTOCOL_ENABLED:
            self.logger.info("")
            self.logger.info("🎯 STRICT PHASED PROTOCOL ENABLED")
            self.logger.info("   PHASE 1: Reproduce iteration 12 exactly (AUC target)")
            self.logger.info("   PHASE 2: Threshold calibration only (F1 injection)")
            self.logger.info("   RULE: First reproduce. Then calibrate. Never mix.")
        self.logger.info("="*80)

        # Start periodic Telegram updates (every 30 minutes)
        self.telegram.start_periodic_updates()
        self.telegram.add_log(f"Auto-improvement loop started with {self.max_iterations} iterations")

        start_time = time.time()

        # Use resume config if available, otherwise use baseline
        current_config_path = self.resume_config_path if self.resume else self.base_config_path

        # Calculate iteration range based on resume
        end_iteration = self.start_iteration + self.max_iterations - 1

        for iteration in range(self.start_iteration, end_iteration + 1):
            try:
                # ========================================================
                # PHASED PROTOCOL: Determine phase and apply enforcement
                # ========================================================
                if PHASED_PROTOCOL_ENABLED:
                    # Check if config specifies HEAD_UPGRADE phase (bypass phase enforcement)
                    config_phase = None
                    if os.path.exists(current_config_path):
                        import yaml
                        with open(current_config_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        config_phase = config_data.get('metadata', {}).get('phase')

                    if config_phase == 'HEAD_UPGRADE':
                        self.logger.info("")
                        self.logger.info("=" * 80)
                        self.logger.info("🔧 HEAD_UPGRADE PHASE (Config-Driven)")
                        self.logger.info("=" * 80)
                        self.logger.info(f"   Config: {current_config_path}")
                        self.logger.info(f"   Parent: {config_data.get('metadata', {}).get('parent_iteration', 12)}")
                        self.logger.info(f"   Head: {config_data.get('model', {}).get('head', {}).get('type', 'linear')}")
                        self.logger.info("   Skipping phase enforcement - using config as-is")
                        self.logger.info("=" * 80)
                        phase = None  # Skip phase enforcement
                        phase_context = {'phase': 'HEAD_UPGRADE', 'config_driven': True}

                    elif config_phase == 'SMOTE_HEAD':
                        smote_cfg = config_data.get('smote', {})
                        self.logger.info("")
                        self.logger.info("=" * 80)
                        self.logger.info("🧬 SMOTE_HEAD PHASE (Config-Driven)")
                        self.logger.info("=" * 80)
                        self.logger.info(f"   Config: {current_config_path}")
                        self.logger.info(f"   Parent: {config_data.get('metadata', {}).get('parent_iteration', 91)}")
                        self.logger.info(f"   Target: {smote_cfg.get('target_class', 'Hernia')}")
                        self.logger.info(f"   Ratio: {smote_cfg.get('sampling_ratio', 4)}x")
                        self.logger.info("   Skipping phase enforcement - using config as-is")
                        self.logger.info("=" * 80)
                        phase = None  # Skip phase enforcement
                        phase_context = {'phase': 'SMOTE_HEAD', 'config_driven': True}

                    elif config_phase == 'REPRESENTATION_FINETUNE' or config_data.get('metadata', {}).get('role_override', False):
                        # REPRESENTATION_FINETUNE: Controlled experiment bypassing all phase logic
                        config_metadata = config_data.get('metadata', {})
                        anchor_iteration = config_metadata.get('anchor_iteration', config_metadata.get('parent_iteration', 89))
                        self.logger.info("")
                        self.logger.info("=" * 80)
                        self.logger.info("🔬 REPRESENTATION_FINETUNE PHASE (Config-Driven)")
                        self.logger.info("=" * 80)
                        self.logger.info(f"   Config: {current_config_path}")
                        self.logger.info(f"   Anchor iteration: {anchor_iteration}")
                        self.logger.info(f"   Unfreeze blocks: {config_data.get('model', {}).get('unfreeze_last_blocks', 0)}")
                        self.logger.info(f"   Learning rate: {config_data.get('training', {}).get('learning_rate', 0)}")
                        self.logger.info("   Role override: ACTIVE (bypasses all phase logic)")
                        self.logger.info("   Phase 5 heuristics: DISABLED")
                        self.logger.info("=" * 80)
                        phase = None  # Skip phase enforcement
                        phase_context = {'phase': 'REPRESENTATION_FINETUNE', 'config_driven': True, 'anchor_iteration': anchor_iteration}

                    else:
                        phase, phase_context = self._get_current_phase()
                        self._log_phase_status(phase, phase_context)

                    # Check for termination conditions (only if phase was determined)
                    if phase == OptimizationPhase.STOP_AND_DEBUG:
                        self.logger.error("")
                        self.logger.error("=" * 80)
                        self.logger.error("🛑 STOPPING: Phase 1 reproduction failed")
                        self.logger.error(f"   Reason: {phase_context.get('reason', 'Unknown')}")
                        self.logger.error("   Action: Diagnose reproducibility issues")
                        self.logger.error("=" * 80)
                        self.telegram.send_message(f"🛑 STOP_AND_DEBUG: {phase_context.get('reason', 'Reproduction failed')}")
                        break

                    if phase == OptimizationPhase.SUCCESS:
                        self.logger.info("")
                        self.logger.info("=" * 80)
                        self.logger.info("🎉 SUCCESS! All optimization criteria met!")
                        self.logger.info(f"   Final AUC: {phase_context.get('final_auc', 'N/A')}")
                        self.logger.info(f"   Final F1: {phase_context.get('final_f1', 'N/A')}")
                        self.logger.info("=" * 80)
                        self.telegram.send_message(f"🎉 SUCCESS: AUC={phase_context.get('final_auc', 0):.4f}, F1={phase_context.get('final_f1', 0):.4f}")
                        break

                    # PHASE 5: Class-specific AUC improvement
                    if phase == OptimizationPhase.PHASE_5_AUC_IMPROVEMENT:
                        self.logger.info("📌 PHASE 5: Class-specific AUC improvement")
                        # Get Phase 5 config with auc_improvement block
                        parent_iteration = phase_context.get('parent_iteration', PHASE_5_BASELINE_ITERATION)
                        target_diseases = phase_context.get('target_diseases', PHASE_5_TARGET_DISEASES)

                        phase5_config = get_phase5_config(
                            iteration=iteration,
                            parent_iteration=parent_iteration,
                            target_diseases=target_diseases
                        )
                        phase5_config_path = os.path.join("experiments", "configs", f"config_iteration_{iteration:03d}.yaml")
                        import yaml
                        with open(phase5_config_path, 'w') as f:
                            yaml.dump(phase5_config, f, default_flow_style=False)
                        current_config_path = phase5_config_path
                        self.logger.info(f"   Config saved to: {phase5_config_path}")
                        self.logger.info(f"   Parent iteration: {parent_iteration}")
                        self.logger.info(f"   Target diseases: {', '.join(target_diseases)}")

                    # PHASE 1: Use EXACT iteration 12 config (ARCHIVED)
                    elif phase == OptimizationPhase.PHASE_1_REPRODUCE:
                        self.logger.info("📌 PHASE 1: Using EXACT iteration 12 configuration")
                        # Save the exact config for this iteration
                        phase1_config = self._get_phase1_config(iteration=iteration)
                        phase1_config_path = os.path.join("experiments", "configs", f"config_iteration_{iteration:03d}.yaml")
                        import yaml
                        with open(phase1_config_path, 'w') as f:
                            yaml.dump(phase1_config, f, default_flow_style=False)
                        current_config_path = phase1_config_path
                        self.logger.info(f"   Config saved to: {phase1_config_path}")

                    # PHASE 2: Skip training, threshold optimization only (ARCHIVED)
                    elif phase == OptimizationPhase.PHASE_2_CALIBRATE:
                        self.logger.info("📌 PHASE 2: Threshold calibration ONLY (no training)")
                        # TODO: Implement threshold-only optimization
                        # For now, continue with the loop but mark as Phase 2
                        pass

                # Run iteration
                iteration_summary = self.run_single_iteration(current_config_path, iteration)

                # Add phase info to summary
                if PHASED_PROTOCOL_ENABLED:
                    iteration_summary['phase'] = phase.value if phase else 'UNKNOWN'
                    iteration_summary['phase_context'] = phase_context
                self.iteration_history.append(iteration_summary)

                # Update best model tracker
                is_new_best = self.best_tracker.update(iteration_summary)

                if is_new_best:
                    self.iterations_without_improvement = 0
                    self.logger.info("")
                    self.logger.info("🏆 NEW BEST ITERATION!")
                    self.logger.info(f"   {self.best_tracker.get_status_message(iteration)}")
                    self.logger.info("")
                else:
                    self.iterations_without_improvement += 1
                    self.logger.info("")
                    self.logger.info(f"📊 {self.best_tracker.get_status_message(iteration)}")
                    self.logger.info(f"   ({self.iterations_without_improvement} iterations without improvement)")
                    self.logger.info("")

                # Generate next config if not last iteration
                if iteration < end_iteration:
                    suggested_changes = iteration_summary.get('suggested_changes')
                    ai_status = iteration_summary.get('ai_analysis_status', 'unknown')

                    # ========================================================
                    # PHASED PROTOCOL: Phase-aware config generation
                    # ========================================================
                    if PHASED_PROTOCOL_ENABLED:
                        next_phase, next_phase_context = self._get_current_phase()

                        # PHASE 5: Use Phase 5 config with AI-suggested auc_improvement changes
                        if next_phase == OptimizationPhase.PHASE_5_AUC_IMPROVEMENT:
                            self.logger.info("")
                            self.logger.info(f"[PHASE 5] Next iteration: class-specific AUC improvement")
                            self.logger.info(f"         Only auc_improvement block can change")

                            # Log Phase 5 entry to phase5_log.md
                            self._log_phase5_iteration(iteration_summary)

                            # Config will be set at the start of the next iteration
                            continue

                        # PHASE 1: Always use exact iteration 12 config - NO AI suggestions
                        if next_phase == OptimizationPhase.PHASE_1_REPRODUCE:
                            self.logger.info("")
                            self.logger.info(f"[PHASE 1] Next iteration will use EXACT iteration 12 config")
                            self.logger.info(f"         AI suggestions IGNORED (reproduction phase)")
                            # Config will be set at the start of the next iteration
                            continue

                        # PHASE 2: Only threshold changes allowed
                        if next_phase == OptimizationPhase.PHASE_2_CALIBRATE:
                            self.logger.info("")
                            self.logger.info(f"[PHASE 2] Next iteration: threshold calibration ONLY")
                            self.logger.info(f"         Training config LOCKED")
                            # TODO: Extract threshold suggestions from AI and apply only those
                            continue

                        # STOP_AND_DEBUG or SUCCESS - loop will break at start of next iteration
                        if next_phase in [OptimizationPhase.STOP_AND_DEBUG, OptimizationPhase.SUCCESS]:
                            self.logger.info(f"[{next_phase.value}] Loop will terminate at next iteration")
                            continue

                    # LEGACY: Non-phased protocol behavior
                    # Check if we should rollback to best config
                    if self.best_tracker.should_rollback(self.iterations_without_improvement, threshold=3):
                        self.logger.warning("")
                        self.logger.warning("="*80)
                        self.logger.warning("⚠️  ROLLBACK TRIGGERED")
                        self.logger.warning("="*80)
                        self.logger.warning(f"No improvement for {self.iterations_without_improvement} iterations")
                        self.logger.warning(f"Rolling back to best iteration #{self.best_tracker.best_iteration}")
                        self.logger.warning("="*80)
                        self.logger.warning("")

                        # Rollback to best config
                        rolled_back_config = self.best_tracker.rollback_to_best(iteration + 1)
                        if rolled_back_config:
                            current_config_path = rolled_back_config
                            self.logger.info(f"✅ Rolled back to: {current_config_path}")
                            self.iterations_without_improvement = 0  # Reset counter
                        else:
                            self.logger.error("❌ Rollback failed, continuing with current config")

                    elif suggested_changes and 'reasoning' in suggested_changes and ai_status == 'completed':
                        self.logger.info(f"\n[Iteration {iteration}] Applying AI suggestions for next iteration:")
                        self.logger.info(f"  Reasoning: {suggested_changes.get('reasoning', 'N/A')}")

                        # Validate and correct AI suggestions (ROLE-BASED STRATEGY)
                        if ROLE_BASED_LOOP and hasattr(self.ai_advisor, 'validate_and_correct_suggestion'):
                            # Determine role for NEXT iteration
                            next_role, next_parent, _ = self._determine_iteration_role(iteration + 1)
                            if next_role and next_role != IterationRole.ABORT:
                                corrected_changes, corrections = self.ai_advisor.validate_and_correct_suggestion(
                                    suggested_changes, next_role
                                )
                                if corrections:
                                    self.logger.info(f"  [ROLE-BASED] AI suggestions corrected:")
                                    for correction in corrections:
                                        self.logger.info(f"    - {correction}")
                                    suggested_changes = corrected_changes

                        # Create new config
                        config_manager = ConfigManager(current_config_path)
                        new_config = config_manager.create_new_config(suggested_changes, iteration + 1)

                        # Save new config
                        next_config_path = os.path.join("experiments", "configs", f"config_iteration_{iteration + 1:03d}.yaml")
                        config_manager.save_config(new_config, next_config_path)

                        self.logger.info(f"  New config saved: {next_config_path}")
                        current_config_path = next_config_path
                    else:
                        if ai_status == 'failed':
                            self.logger.warning(f"[Iteration {iteration}] AI analysis failed, will retry in next iteration")
                            self.logger.warning(f"  Continuing with same config: {current_config_path}")
                        else:
                            self.logger.warning(f"[Iteration {iteration}] No valid suggestions from AI, using same config")

            except Exception as e:
                self.logger.error(f"Iteration {iteration} failed: {str(e)}")
                break

        # Generate final report
        total_time = time.time() - start_time
        self.generate_final_report(total_time)

        self.logger.info("="*80)
        self.logger.info("AUTO-IMPROVEMENT LOOP COMPLETED")
        self.logger.info(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        self.logger.info(f"Iterations completed: {len(self.iteration_history)}")
        self.logger.info("="*80)

        # Send final Telegram notification
        if self.iteration_history:
            best_iteration = max(self.iteration_history, key=lambda x: x.get('avg_auc', 0.0))
            self.telegram.send_pipeline_complete(
                total_iterations=len(self.iteration_history),
                total_time_hours=total_time / 3600,
                best_auc=best_iteration.get('avg_auc', 0.0),
                best_iteration=best_iteration.get('iteration', 0)
            )

    def generate_final_report(self, total_time: float):
        """Generate final summary report"""
        report_file = os.path.join(self.output_dir, "FINAL_REPORT.md")

        with open(report_file, 'w') as f:
            f.write("# Auto-Improvement Loop - Final Report\n\n")
            f.write(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Runtime**: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n\n")
            f.write(f"**Iterations Completed**: {len(self.iteration_history)}/{self.max_iterations}\n\n")

            f.write("## Iteration Summary\n\n")
            f.write("| Iteration | Avg AUC | Avg F1 | Avg Recall | Avg Precision | Avg Accuracy | AI Status | Config |\n")
            f.write("|-----------|---------|--------|------------|---------------|--------------|-----------|--------|\n")

            for summary in self.iteration_history:
                iter_num = summary.get('iteration', 'N/A')
                avg_auc = summary.get('avg_auc', 0.0)
                avg_f1 = summary.get('avg_f1', 0.0)
                avg_recall = summary.get('avg_recall', 0.0)
                avg_precision = summary.get('avg_precision', 0.0)
                avg_accuracy = summary.get('avg_accuracy', 0.0)
                ai_status = summary.get('ai_analysis_status', 'unknown')
                config_path = summary.get('config_path', 'N/A')

                f.write(f"| {iter_num} | {avg_auc:.4f} | "
                       f"{avg_f1:.4f} | {avg_recall:.4f} | {avg_precision:.4f} | "
                       f"{avg_accuracy:.4f} | {ai_status} | "
                       f"{config_path} |\n")

            f.write("\n## Best Performing Iteration\n\n")
            if self.iteration_history:
                best_iteration = max(self.iteration_history, key=lambda x: x.get('avg_auc', 0.0))
                f.write(f"**Iteration**: {best_iteration.get('iteration', 'N/A')}\n\n")
                f.write(f"**Avg AUC**: {best_iteration.get('avg_auc', 0.0):.4f}\n\n")
                f.write(f"**Avg F1**: {best_iteration.get('avg_f1', 0.0):.4f}\n\n")
                f.write(f"**Avg Recall**: {best_iteration.get('avg_recall', 0.0):.4f}\n\n")
                f.write(f"**Avg Precision**: {best_iteration.get('avg_precision', 0.0):.4f}\n\n")
                f.write(f"**Avg Accuracy**: {best_iteration.get('avg_accuracy', 0.0):.4f}\n\n")
                f.write(f"**Train Loss**: {best_iteration.get('train_loss', 0.0):.4f}\n\n")
                f.write(f"**Val Loss**: {best_iteration.get('val_loss', 0.0):.4f}\n\n")
                f.write(f"**AI Analysis Status**: {best_iteration.get('ai_analysis_status', 'unknown')}\n\n")

                if 'config_path' in best_iteration:
                    f.write(f"**Config**: {best_iteration['config_path']}\n\n")
                if 'model_file' in best_iteration:
                    f.write(f"**Model**: {best_iteration['model_file']}\n\n")
                if 'directory' in best_iteration:
                    f.write(f"**Directory**: {best_iteration['directory']}\n\n")

            f.write("\n## Improvement Over Iterations\n\n")
            if len(self.iteration_history) > 1:
                first = self.iteration_history[0]
                last = self.iteration_history[-1]

                auc_improvement = last.get('avg_auc', 0.0) - first.get('avg_auc', 0.0)
                f1_improvement = last.get('avg_f1', 0.0) - first.get('avg_f1', 0.0)
                recall_improvement = last.get('avg_recall', 0.0) - first.get('avg_recall', 0.0)
                precision_improvement = last.get('avg_precision', 0.0) - first.get('avg_precision', 0.0)
                accuracy_improvement = last.get('avg_accuracy', 0.0) - first.get('avg_accuracy', 0.0)

                first_auc = first.get('avg_auc', 0.0)
                if first_auc > 0:
                    f.write(f"**AUC Change**: {auc_improvement:+.4f} ({auc_improvement/first_auc*100:+.2f}%)\n\n")
                else:
                    f.write(f"**AUC Change**: {auc_improvement:+.4f}\n\n")

                f.write(f"**F1 Change**: {f1_improvement:+.4f}\n\n")
                f.write(f"**Recall Change**: {recall_improvement:+.4f}\n\n")
                f.write(f"**Precision Change**: {precision_improvement:+.4f}\n\n")
                f.write(f"**Accuracy Change**: {accuracy_improvement:+.4f}\n\n")

        self.logger.info(f"Final report saved to {report_file}")


def main():
    """Main entry point for the auto-improvement loop"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-improvement loop for chest X-ray classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training from iteration 1
  python main.py --config experiments/configs/config_baseline.yaml --iterations 10

  # Resume from last completed iteration (auto-detects config)
  python main.py --resume --iterations 10

  # Resume with explicit iteration count
  python main.py --resume
        """
    )
    parser.add_argument("--config", type=str, default="experiments/configs/config_baseline.yaml",
                       help="Base configuration file (default: experiments/configs/config_baseline.yaml). "
                            "Only used for NEW runs or as fallback when resuming.")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations to run (default: 10)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default="experiments/auto_improvement_runs",
                       help="Output directory (default: experiments/auto_improvement_runs)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from the last completed iteration. "
                            "Automatically finds and uses the correct config from previous runs.")

    args = parser.parse_args()

    loop = AutoImprovementLoop(
        base_config_path=args.config,
        max_iterations=args.iterations,
        openai_api_key=args.api_key,
        output_dir=args.output_dir,
        resume=args.resume
    )

    loop.run()


if __name__ == "__main__":
    main()
