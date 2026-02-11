#!/usr/bin/env python3
"""
Recover Iteration 30 Results
Runs testing on the saved model from iteration 30 that failed during threshold optimization.
"""

import os
import sys
import logging
from chest_xray_test_pipeline import test_model_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recover_30')

def main():
    iteration_dir = 'auto_improvement_runs/iteration_030'
    model_file = os.path.join(iteration_dir, 'pipeline_model_20260102-082519.pth')
    results_file = os.path.join(iteration_dir, 'pipeline_results_20260102-082519.csv')

    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return 1

    logger.info("="*80)
    logger.info("RECOVERING ITERATION 30 RESULTS")
    logger.info("="*80)
    logger.info(f"Model: {model_file}")
    logger.info(f"Output: {results_file}")
    logger.info("")

    try:
        # Run testing
        logger.info("Running test pipeline...")
        test_model_pipeline(
            model_file=model_file,
            results_file=results_file,
            logger=logger
        )

        logger.info("")
        logger.info("="*80)
        logger.info("✅ RECOVERY COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {iteration_dir}/")
        logger.info("")

        # List generated files
        logger.info("Generated files:")
        for f in os.listdir(iteration_dir):
            if f.endswith(('.csv', '.json')):
                size = os.path.getsize(os.path.join(iteration_dir, f))
                logger.info(f"  ✓ {f} ({size:,} bytes)")

        return 0

    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
