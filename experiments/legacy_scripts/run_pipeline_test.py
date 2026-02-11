#!/usr/bin/env python3
"""
Test runner for the Chest X-Ray Classification Pipeline with small datasets
"""

import os
from create_small_datasets import create_small_datasets
from chest_xray_pipeline import ChestXRayPipeline

if __name__ == "__main__":
    print("🧪 Starting Chest X-Ray Pipeline Test")
    print("Creating small datasets for testing...")
    print("-" * 60)
    
    # Create small datasets first
    if not os.path.exists('./ChestX-ray14/train_data_small.csv'):
        create_small_datasets(train_samples=500, test_samples=200)
    else:
        print("Small datasets already exist, using existing files")
    
    print("\n🚀 Running pipeline with small datasets")
    print("Estimated time: 5-15 minutes")
    print("-" * 60)
    
    pipeline = ChestXRayPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("✅ Pipeline test completed successfully!")
        print("You can now run the full pipeline with: python run_pipeline.py")
    else:
        print("❌ Pipeline test failed. Check logs for details.")