"""
Test script to verify loss type validation accepts both formats
"""

import torch
import yaml
from config_based_pipeline import ConfigBasedTrainer
import tempfile
import os

def create_test_config(loss_type):
    """Create a minimal test config with specified loss type"""
    config = {
        'metadata': {'iteration': 1, 'description': 'test'},
        'model': {
            'architecture': 'ModifiedDenseNetWithDropOut',
            'backbone': 'densenet121',
            'num_classes': 14,
            'dropout_rate': 0.3,
            'use_additional_features': False
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 1,
            'num_workers': 2,
            'optimizer': {
                'type': 'Adam',
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.0
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'mode': 'min',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-5
            },
            'early_stopping': {
                'enabled': True,
                'patience': 5,
                'min_delta': 0.01,
                'warmup_epochs': 5
            }
        },
        'loss': {
            'type': loss_type,
            'gamma': 2.0,
            'use_class_weights': False,
            'use_dynamic_weights': False
        },
        'augmentation': {
            'base': {
                'resize': [224, 224],
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'rare_class': {
                'enabled': False,
                'thresholds': {}
            }
        },
        'data': {
            'train_csv': './ChestX-ray14/train_data.csv',
            'test_csv': './ChestX-ray14/test_data.csv',
            'images_dir': './ChestX-ray14/images224'
        },
        'data_split': {
            'val_ratio': 0.2,
            'random_state': 42
        },
        'hardware': {
            'device': 'auto',
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2
        },
        'logging': {
            'log_interval': 50
        },
        'evaluation': {
            'threshold_optimization': 'per_class_f1_score'
        }
    }
    return config

def test_loss_type(loss_type):
    """Test if a loss type works"""
    print(f"Testing loss type: '{loss_type}'...")

    try:
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(create_test_config(loss_type), f)
            config_path = f.name

        # Create trainer
        trainer = ConfigBasedTrainer(config_path, "test", None, None, 1)

        # Create a dummy device
        device = torch.device('cpu')

        # Try to create loss function
        criterion = trainer.create_loss_function(device)

        print(f"  ✅ SUCCESS: '{loss_type}' was accepted")
        print(f"     Created: {criterion.__class__.__name__}")

        # Cleanup
        os.unlink(config_path)
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        # Cleanup
        if 'config_path' in locals():
            os.unlink(config_path)
        return False

def main():
    print("="*60)
    print("Testing Loss Type Validation Fix")
    print("="*60)
    print()

    test_cases = [
        # Should all work now
        ("FocalLoss", True),
        ("focal", True),
        ("Focal", True),
        ("FOCAL", True),
        ("WeightedBCE", True),
        ("bce", True),
        ("BCE", True),
        ("weightedbce", True),
        # Should fail
        ("invalid", False),
        ("CrossEntropy", False),
    ]

    passed = 0
    failed = 0

    for loss_type, should_pass in test_cases:
        result = test_loss_type(loss_type)

        if result == should_pass:
            passed += 1
        else:
            failed += 1
            if should_pass:
                print(f"    ⚠️  Expected to pass but failed!")
            else:
                print(f"    ⚠️  Expected to fail but passed!")
        print()

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    print("="*60)

    if failed == 0:
        print("✅ All tests passed! Loss type validation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
