"""
Complete AI Analysis for Failed Iteration 66
This script completes the missing AI analysis that was skipped due to the crash.
"""

import os
import json
import pandas as pd
import logging
from ai_advisor import AIAdvisor
from config_manager import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def complete_iteration_66_analysis():
    """Complete the missing AI analysis for iteration 66"""

    iteration = 66
    iter_dir = f"auto_improvement_runs/iteration_{iteration:03d}"

    logger.info("="*80)
    logger.info(f"COMPLETING AI ANALYSIS FOR ITERATION {iteration}")
    logger.info("="*80)

    # Check if directory exists
    if not os.path.exists(iter_dir):
        logger.error(f"Iteration directory not found: {iter_dir}")
        return False

    # Load results
    try:
        import glob
        results_files = glob.glob(f"{iter_dir}/pipeline_results_*.csv")
        comparison_files = glob.glob(f"{iter_dir}/baseline_comparison_*.csv")

        if not results_files or not comparison_files:
            logger.error("Missing results or comparison files")
            return False

        results_df = pd.read_csv(results_files[0])
        comparison_df = pd.read_csv(comparison_files[0])

        logger.info(f"✓ Loaded results: {os.path.basename(results_files[0])}")
        logger.info(f"✓ Loaded comparison: {os.path.basename(comparison_files[0])}")

        # Load summary if exists
        summary_file = f"{iter_dir}/iteration_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            train_loss = summary.get('train_loss', 0.0)
            val_loss = summary.get('val_loss', 0.0)
            logger.info(f"✓ Loaded summary: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        else:
            train_loss = 0.0
            val_loss = 0.0
            logger.warning("No summary file found, using default losses")

        # Load config
        config_file = f"{iter_dir}/config.yaml"
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            return False

        config_manager = ConfigManager(config_file)
        config = config_manager.config
        logger.info(f"✓ Loaded config: {config_file}")

        # Initialize AI advisor
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set")
            return False

        ai_advisor = AIAdvisor(api_key=api_key, model="gpt-5.2")
        logger.info("✓ Initialized AI advisor with gpt-5.2")

        # Load previous iteration history (last 3)
        iteration_history = []
        for prev_iter in range(max(1, iteration - 3), iteration):
            prev_dir = f"auto_improvement_runs/iteration_{prev_iter:03d}"
            prev_summary_file = f"{prev_dir}/iteration_summary.json"

            if os.path.exists(prev_summary_file):
                with open(prev_summary_file, 'r') as f:
                    prev_summary = json.load(f)
                iteration_history.append(prev_summary)
                logger.info(f"✓ Loaded previous iteration {prev_iter}")

        # Run AI analysis
        logger.info("")
        logger.info("Running AI analysis...")

        analysis, suggested_changes = ai_advisor.analyze_results(
            config=config,
            results_df=results_df,
            comparison_df=comparison_df,
            iteration=iteration,
            iteration_history=iteration_history,
            train_loss=train_loss,
            val_loss=val_loss,
            task_info=None,  # No task info for historical iteration
            iteration_analysis=None
        )

        # Save analysis
        analysis_file = f"{iter_dir}/ai_analysis_{iteration:03d}.txt"
        with open(analysis_file, 'w') as f:
            f.write(analysis)

        logger.info("")
        logger.info(f"✓ AI analysis saved: {analysis_file}")
        logger.info(f"   Analysis length: {len(analysis)} characters")

        # Update iteration summary
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            summary['analysis_file'] = analysis_file
            summary['suggested_changes'] = suggested_changes
            summary['ai_analysis_status'] = 'completed'
            summary['status'] = 'completed_with_recovery'
            summary['note'] = 'AI analysis completed via recovery script'

            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"✓ Updated iteration summary")

        # Show suggested changes
        if suggested_changes and 'reasoning' in suggested_changes:
            logger.info("")
            logger.info("🤖 AI Suggestions:")
            logger.info(f"   {suggested_changes['reasoning'][:200]}...")

        logger.info("")
        logger.info("="*80)
        logger.info("✅ AI ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error(f"Error completing AI analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = complete_iteration_66_analysis()
    exit(0 if success else 1)
