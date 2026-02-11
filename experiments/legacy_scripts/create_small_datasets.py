#!/usr/bin/env python3
"""
Create small train and test datasets for pipeline testing
"""

import pandas as pd
import os

def create_small_datasets(train_samples=5000, test_samples=2000):
    """Create small datasets for testing"""
    
    # Read original datasets
    train_df = pd.read_csv('./ChestX-ray14/train_data.csv')
    test_df = pd.read_csv('./ChestX-ray14/test_data.csv')
    
    print(f"Original train size: {len(train_df)}")
    print(f"Original test size: {len(test_df)}")
    
    # Sample small datasets
    train_small = train_df.sample(n=min(train_samples, len(train_df)), random_state=42)
    test_small = test_df.sample(n=min(test_samples, len(test_df)), random_state=42)
    
    # Save small datasets
    train_small.to_csv('./ChestX-ray14/train_data_small.csv', index=False)
    test_small.to_csv('./ChestX-ray14/test_data_small.csv', index=False)
    
    print(f"Created train_data_small.csv with {len(train_small)} samples")
    print(f"Created test_data_small.csv with {len(test_small)} samples")
    
    return train_small, test_small

if __name__ == "__main__":
    create_small_datasets()