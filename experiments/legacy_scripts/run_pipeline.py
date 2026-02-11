#!/usr/bin/env python3
"""
Simple runner for the Chest X-Ray Classification Pipeline
"""

import logging
import datetime
from chest_xray_pipeline import ChestXRayPipeline

def setup_logger():
    """Setup centralized logger for the entire pipeline"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"pipeline_log_{timestamp}.txt"
    
    # Create logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, timestamp

if __name__ == "__main__":
    logger, timestamp = setup_logger()
    
    logger.info("🚀 Starting Chest X-Ray Classification Pipeline")
    logger.info("This will run training, testing, and baseline comparison")
    logger.info("Estimated time: 1-3 hours depending on hardware")
    logger.info("-" * 60)

    pipeline = ChestXRayPipeline(timestamp=timestamp, logger=logger)
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
    else:
        logger.info("❌ Pipeline failed. Check logs for details.")