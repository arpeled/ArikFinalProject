#!/usr/bin/env python3
"""
Chest X-Ray Auto-Improvement Pipeline — Main Entry Point
=========================================================

Runs the automated training loop for multi-label chest X-ray classification
using DenseNet-121 with AI-guided hyperparameter optimization.

Usage:
    # Start a new training run (10 iterations by default)
    python main.py --config experiments/configs/config_baseline.yaml --iterations 10

    # Resume from the last completed iteration
    python main.py --resume --iterations 10

    # Run a single training iteration with a specific config
    python main.py --single --config experiments/configs/config_baseline.yaml

See README.md for full documentation.
"""

import os
import sys

# Ensure project root is on the path so that core/ and automation/ are importable
_project_root = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(_project_root, 'core')
_automation_dir = os.path.join(_project_root, 'automation')
for _d in [_project_root, _core_dir, _automation_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


def run_auto_improvement():
    """Run the full auto-improvement loop (primary use case)."""
    from auto_improvement_loop import main as loop_main
    loop_main()


def run_single_training():
    """Run a single training iteration from a config file."""
    import argparse
    import datetime
    from config_based_pipeline import ConfigBasedTrainer

    parser = argparse.ArgumentParser(description="Run a single training iteration")
    parser.add_argument("--config", type=str, default="experiments/configs/config_baseline.yaml",
                        help="Path to YAML configuration file")
    args, _ = parser.parse_known_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = ConfigBasedTrainer(args.config, timestamp)
    trainer.train()


if __name__ == "__main__":
    # Check if --single flag is passed for single-iteration mode
    if "--single" in sys.argv:
        sys.argv.remove("--single")
        run_single_training()
    else:
        run_auto_improvement()
