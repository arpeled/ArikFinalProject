"""
Configuration-based Pipeline Wrapper
Loads configuration from YAML file and runs the training pipeline
"""

import os
import sys

# Add core/ directory to path for imports
_core_dir = os.path.dirname(os.path.abspath(__file__))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)
import time
import datetime
import logging
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms
from dataset import (
    ChestXRayDataset, ModifiedDenseNetWithDropOut, ModifiedDenseNet,
    create_hernia_oversampler, HerniaAugmentTransform
)
from sklearn.model_selection import GroupShuffleSplit
import torch.nn.functional as F

from config_manager import ConfigManager

# Import role-based configuration
try:
    from iteration_baselines import (
        IterationRole,
        TRAIN_AUC_RULES,
        RECOVER_F1_RULES,
        get_enforced_config_for_role
    )
    ROLE_BASED_CONFIG = True
except ImportError:
    ROLE_BASED_CONFIG = False
    print("Warning: iteration_baselines not available, role-based config enforcement disabled")

# Import SMOTE and oversampling techniques
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not installed. SMOTE and oversampling features disabled.")

# Import Feature-Space SMOTE module
try:
    from feature_smote import (
        FeatureExtractor, FeatureSMOTE, EmbeddingHeadTrainer,
        compute_hernia_metrics, cache_embeddings, load_cached_embeddings,
        get_embedding_cache_path, CLASS_NAMES, CLASS_TO_IDX
    )
    FEATURE_SMOTE_AVAILABLE = True
except ImportError:
    FEATURE_SMOTE_AVAILABLE = False
    print("⚠️  feature_smote module not available. Feature-space SMOTE disabled.")


def compute_class_weights(labels_tensor, epsilon=1e-6):
    """
    Dynamically compute class weights based on training label distribution.

    This function addresses class imbalance by computing inverse frequency weights,
    giving higher weight to rare classes to improve model sensitivity.

    Args:
        labels_tensor: Tensor of shape (num_samples, num_classes) containing binary labels
        epsilon: Small value to avoid division by zero

    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    # Count positive samples for each class
    class_counts = labels_tensor.sum(dim=0)

    # Calculate total positive samples across all classes
    total = class_counts.sum()

    # Calculate class frequencies
    frequencies = class_counts / (total + epsilon)

    # Find minimum frequency (excluding zero frequencies)
    min_freq = frequencies[frequencies > 0].min()

    # Compute inverse frequency weights (min_freq / frequency)
    weights = min_freq / (frequencies + epsilon)

    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.weights is not None:
            loss = loss * self.weights
        return loss.mean()


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.

    Supported monitors:
    - val_loss: Validation loss (mode='min')
    - val_f1: Validation F1 score (mode='max')
    - val_auc: Validation AUC (mode='max')
    - val_macro_auc: Validation macro AUC (mode='max') - for TRAIN_AUC role

    Args:
        monitor: Metric to monitor (e.g., 'val_loss', 'val_f1', 'val_auc', 'val_macro_auc')
        mode: 'min' for metrics to minimize (loss), 'max' for metrics to maximize (f1, auc)
        patience: Number of epochs with no improvement after which training stops
        min_delta: Minimum change to qualify as improvement
        warmup_epochs: Number of epochs to wait before monitoring starts
    """
    # Supported monitors and their default modes
    SUPPORTED_MONITORS = {
        'val_loss': 'min',
        'val_f1': 'max',
        'val_auc': 'max',
        'val_macro_auc': 'max'
    }

    def __init__(self, monitor='val_loss', mode='min', patience=7, min_delta=0.01, warmup_epochs=10):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_score = None
        self.counter = 0
        self.epoch_count = 0

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, current_score):
        """
        Check if training should stop.

        Args:
            current_score: Current value of monitored metric

        Returns:
            True if training should stop, False otherwise
        """
        self.epoch_count += 1

        # Don't stop during warmup period
        if self.epoch_count < self.warmup_epochs:
            return False

        # Initialize best score on first call after warmup
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
            return False

        # Check for improvement
        if self.mode == 'min':
            # For loss: improvement = current < best - min_delta
            improved = current_score < self.best_score - self.min_delta
        else:  # mode == 'max'
            # For F1/AUC: improvement = current > best + min_delta
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training

        return False

    def get_status(self):
        """Get current early stopping status for logging."""
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'best_score': float(self.best_score) if self.best_score is not None else None,
            'counter': self.counter,
            'patience': self.patience,
            'epochs_since_warmup': max(0, self.epoch_count - self.warmup_epochs)
        }


class ConfigBasedTrainer:
    """Trainer that uses configuration files"""

    def __init__(self, config_path: str, timestamp: str, logger=None, telegram_notifier=None, iteration: int = 1):
        """
        Initialize trainer with configuration

        Args:
            config_path: Path to configuration file
            timestamp: Timestamp for this run
            logger: Logger instance
            telegram_notifier: TelegramNotifier instance (optional)
            iteration: Current iteration number (for notifications)
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config_dict()
        self.timestamp = timestamp
        self.logger = logger or self._create_logger()
        self.telegram = telegram_notifier
        self.iteration = iteration

        self.model_file = f"pipeline_model_{self.timestamp}.pth"
        self.log_file = f"pipeline_log_{self.timestamp}.txt"

        # Phase 2 flag - when True, training is skipped entirely
        self.skip_training = False
        self.phase1_model_path = None  # Path to Phase 1 model for Phase 2

    def _create_logger(self):
        """Create a default logger"""
        logger = logging.getLogger('config_trainer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def enforce_role_config(self, role: 'IterationRole') -> None:
        """
        Enforce role-specific configuration that cannot be overridden.

        This is called at the start of training to ensure that hard-coded rules
        for each role are applied regardless of what the AI suggested.

        Args:
            role: The IterationRole for this iteration
        """
        if not ROLE_BASED_CONFIG:
            self.logger.warning("Role-based config not available, skipping enforcement")
            return

        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info(f"ENFORCING ROLE-BASED CONFIGURATION: {role.value}")
        self.logger.info("="*60)

        if role == IterationRole.TRAIN_AUC:
            self._enforce_train_auc_config()
        elif role == IterationRole.RECOVER_F1:
            self._enforce_recover_f1_config()
        elif role == IterationRole.ADJUST_THRESHOLDS:
            self._enforce_threshold_only_config()

        self.logger.info("="*60)

    def _enforce_train_auc_config(self) -> None:
        """
        Enforce TRAIN_AUC configuration rules.

        Forces:
        - FocalLoss with gamma=2.0
        - use_class_weights=False
        - Early stopping on val_macro_auc
        """
        original_loss = self.config.get('loss', {}).copy()
        original_es = self.config.get('training', {}).get('early_stopping', {}).copy()

        # Enforce loss configuration
        if 'loss' not in self.config:
            self.config['loss'] = {}

        self.config['loss']['type'] = TRAIN_AUC_RULES.loss_type
        self.config['loss']['gamma'] = TRAIN_AUC_RULES.gamma
        self.config['loss']['use_class_weights'] = TRAIN_AUC_RULES.use_class_weights

        # Log loss changes
        if original_loss.get('gamma') != TRAIN_AUC_RULES.gamma:
            self.logger.info(f"  [ENFORCED] loss.gamma: {original_loss.get('gamma')} -> {TRAIN_AUC_RULES.gamma}")
        if original_loss.get('use_class_weights') != TRAIN_AUC_RULES.use_class_weights:
            self.logger.info(f"  [ENFORCED] loss.use_class_weights: {original_loss.get('use_class_weights')} -> {TRAIN_AUC_RULES.use_class_weights}")

        # Enforce early stopping configuration
        if 'training' not in self.config:
            self.config['training'] = {}
        if 'early_stopping' not in self.config['training']:
            self.config['training']['early_stopping'] = {}

        es = self.config['training']['early_stopping']
        es['monitor'] = TRAIN_AUC_RULES.early_stopping_monitor
        es['mode'] = TRAIN_AUC_RULES.early_stopping_mode
        es['patience'] = TRAIN_AUC_RULES.early_stopping_patience
        es['min_delta'] = TRAIN_AUC_RULES.early_stopping_min_delta
        es['warmup_epochs'] = TRAIN_AUC_RULES.early_stopping_warmup_epochs

        # Log early stopping changes
        if original_es.get('monitor') != TRAIN_AUC_RULES.early_stopping_monitor:
            self.logger.info(f"  [ENFORCED] early_stopping.monitor: {original_es.get('monitor')} -> {TRAIN_AUC_RULES.early_stopping_monitor}")
        if original_es.get('mode') != TRAIN_AUC_RULES.early_stopping_mode:
            self.logger.info(f"  [ENFORCED] early_stopping.mode: {original_es.get('mode')} -> {TRAIN_AUC_RULES.early_stopping_mode}")

        self.logger.info("  TRAIN_AUC configuration enforced successfully")

    def _enforce_recover_f1_config(self) -> None:
        """
        Enforce RECOVER_F1 configuration rules.

        Forces:
        - Very low learning rate (max 0.00001)
        - Early stopping on val_f1
        """
        if 'training' not in self.config:
            self.config['training'] = {}

        # Cap learning rate
        current_lr = self.config['training'].get('learning_rate', 0.001)
        if current_lr > RECOVER_F1_RULES.max_learning_rate:
            self.logger.info(f"  [ENFORCED] learning_rate: {current_lr} -> {RECOVER_F1_RULES.max_learning_rate}")
            self.config['training']['learning_rate'] = RECOVER_F1_RULES.max_learning_rate

        # Enforce early stopping on val_f1
        if 'early_stopping' not in self.config['training']:
            self.config['training']['early_stopping'] = {}

        es = self.config['training']['early_stopping']
        original_monitor = es.get('monitor')
        es['monitor'] = RECOVER_F1_RULES.early_stopping_monitor
        es['mode'] = RECOVER_F1_RULES.early_stopping_mode
        es['patience'] = RECOVER_F1_RULES.early_stopping_patience

        if original_monitor != RECOVER_F1_RULES.early_stopping_monitor:
            self.logger.info(f"  [ENFORCED] early_stopping.monitor: {original_monitor} -> {RECOVER_F1_RULES.early_stopping_monitor}")

        self.logger.info("  RECOVER_F1 configuration enforced successfully")

    def _enforce_threshold_only_config(self) -> None:
        """
        Enforce ADJUST_THRESHOLDS configuration.

        In this mode, we skip training and only optimize thresholds.
        Sets self.skip_training = True which is checked in train().
        """
        self.skip_training = True
        self.logger.info("  ADJUST_THRESHOLDS mode: Training will be SKIPPED")
        self.logger.info("  Only threshold optimization will be performed")
        self.logger.info("  ⚠️  self.skip_training = True")

    def get_rare_class_augmentation(self, train_df, root_dir, rare_transform, use_additional_features):
        """Create rare class augmentation datasets"""
        thresholds = self.config['augmentation']['rare_class']['thresholds']
        rare_datasets = []

        for label, threshold in thresholds.items():
            positive_samples = train_df[train_df[label] == 1]
            class_ratio = len(positive_samples) / len(train_df)
            if class_ratio < threshold:
                repeat_factor = int(threshold / class_ratio)
                if repeat_factor > 1:
                    rare_aug_df = pd.concat([positive_samples] * repeat_factor, ignore_index=True)
                    rare_dataset = ChestXRayDataset(
                        dataset=rare_aug_df, csv_file=None,
                        root_dir=root_dir, transform=rare_transform,
                        use_additional_features=use_additional_features
                    )
                    rare_datasets.append(rare_dataset)

        return rare_datasets

    def apply_class_balancing(self, train_df, logger=None):
        """
        Apply class balancing strategy (oversampling or SMOTE)

        Args:
            train_df: Training dataframe
            logger: Logger instance

        Returns:
            Balanced dataframe
        """
        if logger is None:
            logger = self.logger

        balancing_config = self.config.get('data_balancing', {})
        strategy = balancing_config.get('strategy', 'none')

        if strategy == 'none':
            logger.info("No class balancing applied")
            return train_df

        if strategy not in ['oversample', 'smote', 'smote_tomek']:
            logger.warning(f"Unknown balancing strategy: {strategy}. Skipping.")
            return train_df

        logger.info(f"📊 Applying class balancing: {strategy}")

        # Define label columns
        label_columns = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                        'Pleural_Thickening', 'Hernia']

        # Get rare classes (frequency < threshold)
        target_frequency = balancing_config.get('target_frequency', 0.05)
        rare_classes = []

        for label in label_columns:
            class_ratio = train_df[label].sum() / len(train_df)
            if class_ratio < target_frequency:
                rare_classes.append(label)

        if not rare_classes:
            logger.info("No rare classes found below target frequency")
            return train_df

        logger.info(f"Found {len(rare_classes)} rare classes: {rare_classes}")

        # Apply oversampling
        if strategy == 'oversample':
            balanced_df = self._apply_random_oversampling(
                train_df, rare_classes, target_frequency, logger
            )
        elif strategy in ['smote', 'smote_tomek']:
            if not SMOTE_AVAILABLE:
                logger.error("SMOTE requested but imbalanced-learn not installed!")
                logger.info("Install with: pip install imbalanced-learn")
                logger.info("Falling back to random oversampling...")
                balanced_df = self._apply_random_oversampling(
                    train_df, rare_classes, target_frequency, logger
                )
            else:
                # For images, we still use oversampling with augmentation
                # True SMOTE on raw images doesn't make sense
                logger.info("Note: For image data, using augmented oversampling instead of SMOTE")
                balanced_df = self._apply_random_oversampling(
                    train_df, rare_classes, target_frequency, logger
                )
        else:
            balanced_df = train_df

        logger.info(f"Dataset size: {len(train_df)} → {len(balanced_df)} (+{len(balanced_df)-len(train_df)} samples)")

        return balanced_df

    def _apply_random_oversampling(self, train_df, rare_classes, target_frequency, logger):
        """
        Apply random oversampling to rare classes

        Args:
            train_df: Training dataframe
            rare_classes: List of rare class labels
            target_frequency: Target frequency for each class
            logger: Logger instance

        Returns:
            Balanced dataframe
        """
        balanced_frames = [train_df]

        for label in rare_classes:
            positive_samples = train_df[train_df[label] == 1]
            current_ratio = len(positive_samples) / len(train_df)

            if current_ratio < target_frequency:
                # Calculate how many times to repeat
                target_count = int(len(train_df) * target_frequency)
                current_count = len(positive_samples)
                additional_needed = max(0, target_count - current_count)

                if additional_needed > 0:
                    # Randomly sample with replacement
                    oversampled = positive_samples.sample(
                        n=additional_needed,
                        replace=True,
                        random_state=42
                    ).reset_index(drop=True)

                    balanced_frames.append(oversampled)
                    logger.info(f"  {label}: {current_count} → {current_count + additional_needed} (+{additional_needed})")

        # Combine all frames
        balanced_df = pd.concat(balanced_frames, ignore_index=True)

        # Shuffle the combined dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return balanced_df

    def create_loss_function(self, device, train_labels=None):
        """
        Create loss function from config with optional dynamic class weights.

        Args:
            device: Device to place tensors on
            train_labels: Optional tensor of training labels for dynamic weight computation

        Returns:
            Loss function instance
        """
        loss_config = self.config['loss']
        loss_type = loss_config['type']

        # Normalize loss type (accept both "focal" and "FocalLoss", case-insensitive)
        loss_type_normalized = loss_type.lower().replace('_', '').replace('-', '')

        # Determine if we should use dynamic class weights
        use_dynamic_weights = loss_config.get('use_dynamic_weights', False)

        if loss_type_normalized in ["focalloss", "focal"]:
            if use_dynamic_weights and train_labels is not None:
                # Compute dynamic class weights from actual training data
                self.logger.info("Computing dynamic class weights from training data...")
                alpha_tensor = compute_class_weights(train_labels).to(device)
                self.logger.info(f"Dynamic weights computed: min={alpha_tensor.min():.4f}, max={alpha_tensor.max():.4f}, mean={alpha_tensor.mean():.4f}")
                return FocalLoss(alpha=alpha_tensor, gamma=loss_config['gamma'], reduction='mean')
            elif loss_config['use_class_weights']:
                # Use static class weights from config
                class_frequencies = loss_config['class_frequencies']
                min_freq = min(class_frequencies)
                alpha = [min_freq / f for f in class_frequencies]
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(device)
                return FocalLoss(alpha=alpha_tensor, gamma=loss_config['gamma'], reduction='mean')
            else:
                return FocalLoss(alpha=None, gamma=loss_config['gamma'], reduction='mean')

        elif loss_type_normalized in ["weightedbce", "weightedbceloss", "bce", "bceloss"]:
            if use_dynamic_weights and train_labels is not None:
                # Compute dynamic class weights from actual training data
                self.logger.info("Computing dynamic class weights from training data...")
                weights_tensor = compute_class_weights(train_labels).to(device)
                self.logger.info(f"Dynamic weights computed: min={weights_tensor.min():.4f}, max={weights_tensor.max():.4f}, mean={weights_tensor.mean():.4f}")
                return WeightedBCELoss(weights=weights_tensor)
            elif loss_config['use_class_weights']:
                class_frequencies = loss_config['class_frequencies']
                min_freq = min(class_frequencies)
                weights = [min_freq / f for f in class_frequencies]
                weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                return WeightedBCELoss(weights=weights_tensor)
            else:
                return WeightedBCELoss(weights=None)

        else:
            raise ValueError(
                f"Unknown loss type: '{loss_type}'. "
                f"Supported types: 'FocalLoss'/'focal', 'WeightedBCE'/'bce'"
            )

    def create_transforms(self):
        """Create data transforms from config"""
        aug_config = self.config['augmentation']

        # Base transform
        base_transform = transforms.Compose([
            transforms.Resize(tuple(aug_config['base']['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(
                aug_config['base']['normalize']['mean'],
                aug_config['base']['normalize']['std']
            )
        ])

        # Rare class transform
        rare_config = aug_config['rare_class']
        rare_transform_list = []

        if rare_config['enabled']:
            rare_transform_list.extend([
                transforms.RandomRotation(rare_config['rotation_degrees']),
                transforms.RandomHorizontalFlip(p=rare_config['horizontal_flip_prob']),
            ])

        rare_transform_list.append(base_transform)
        rare_transform = transforms.Compose(rare_transform_list)

        return base_transform, rare_transform

    def train(self):
        """Execute training with configuration"""
        self.logger.info("="*60)
        self.logger.info("CONFIG-BASED TRAINING STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Config: {self.config_manager.config_path}")
        # Handle missing metadata (e.g., when using exact Phase 1 config)
        iteration_num = self.config.get('metadata', {}).get('iteration', self.iteration)
        self.logger.info(f"Iteration: {iteration_num}")

        # ================================================================
        # SMOTE_HEAD PHASE GUARD (MANDATORY)
        # ================================================================
        metadata = self.config.get('metadata', {})
        config_phase = metadata.get('phase', '')
        smote_config = self.config.get('smote', {})

        if config_phase == 'SMOTE_HEAD':
            # GUARD: If phase is SMOTE_HEAD, smote.enabled MUST be true
            if not smote_config.get('enabled', False):
                self.logger.error("=" * 80)
                self.logger.error("FATAL: SMOTE_HEAD phase requires smote.enabled=true")
                self.logger.error("=" * 80)
                raise RuntimeError(
                    "SMOTE_HEAD phase detected but smote.enabled is not true. "
                    "This is a configuration error - SMOTE_HEAD requires smote.enabled=true."
                )

            # GUARD: Ensure auc_improvement is disabled (cannot coexist with SMOTE_HEAD)
            auc_improvement_config = self.config.get('auc_improvement', {})
            if auc_improvement_config.get('enabled', False):
                self.logger.error("=" * 80)
                self.logger.error("FATAL: SMOTE_HEAD phase cannot coexist with auc_improvement.enabled=true")
                self.logger.error("=" * 80)
                raise RuntimeError(
                    "SMOTE_HEAD phase detected but auc_improvement.enabled is true. "
                    "SMOTE_HEAD and Phase 5 (auc_improvement) are mutually exclusive."
                )

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("SMOTE_HEAD PHASE VALIDATED")
            self.logger.info("=" * 80)
            self.logger.info(f"   smote.enabled: {smote_config.get('enabled', False)}")
            self.logger.info(f"   smote.target_class: {smote_config.get('target_class', 'Hernia')}")
            self.logger.info(f"   smote.sampling_ratio: {smote_config.get('sampling_ratio', 4)}x")
            self.logger.info(f"   auc_improvement.enabled: {auc_improvement_config.get('enabled', False)}")
            self.logger.info("=" * 80)

        # ================================================================
        # PHASE 2 CHECK: Skip training if in threshold-only mode
        # ================================================================
        if self.skip_training:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("🎯 PHASE 2: THRESHOLD-ONLY MODE")
            self.logger.info("=" * 60)
            self.logger.info("  Training is SKIPPED in Phase 2")
            self.logger.info("  Only threshold optimization will be performed")

            # Load the Phase 1 model if path is provided
            if self.phase1_model_path and os.path.exists(self.phase1_model_path):
                self.logger.info(f"  Loading Phase 1 model from: {self.phase1_model_path}")
                # Copy the Phase 1 model to be used as this iteration's model
                import shutil
                shutil.copy(self.phase1_model_path, self.model_file)
                self.logger.info(f"  Model copied to: {self.model_file}")
            else:
                self.logger.error(f"  ❌ Phase 1 model path not set or doesn't exist: {self.phase1_model_path}")
                raise ValueError("Phase 2 requires a Phase 1 model path, but none was provided")

            self.logger.info("=" * 60)

            # Return minimal metadata for Phase 2
            training_metadata = {
                'actual_epochs': 0,
                'early_stopping_triggered': False,
                'thresholds': {},  # Will be optimized during testing
                'description': 'Phase 2: Threshold-only mode, training skipped',
                'phase': 'PHASE_2_CALIBRATE',
                'skip_training': True,
                'phase1_model_path': self.phase1_model_path
            }

            # Load model to return (for testing phase)
            device = self._get_device()
            model_config = self.config['model']
            if model_config['architecture'] == "ModifiedDenseNetWithDropOut":
                model = ModifiedDenseNetWithDropOut(
                    num_classes=model_config['num_classes'],
                    use_additional_features=model_config['use_additional_features'],
                    head_config=model_config.get('head')
                )
            else:
                model = ModifiedDenseNet(
                    num_classes=model_config['num_classes'],
                    use_additional_features=model_config['use_additional_features']
                )
            model.load_state_dict(torch.load(self.model_file, map_location=device))
            model = model.to(device)

            return model, 0.0, 0.0, training_metadata

        # Extract config values
        model_config = self.config['model']
        training_config = self.config['training']
        data_config = self.config['data']
        split_config = self.config['data_split']

        # Device setup
        device = self._get_device()
        self.logger.info(f"Using device: {device}")

        # Load data
        df = pd.read_csv(data_config['train_csv'])
        if "Patient ID" not in df.columns:
            raise ValueError("Patient ID column not found in CSV")

        # Split data
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=split_config['val_ratio'],
            random_state=split_config['random_state']
        )
        train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Apply class balancing (oversampling/SMOTE) if configured
        train_df = self.apply_class_balancing(train_df)

        # Create transforms
        base_transform, rare_transform = self.create_transforms()

        # Create datasets
        use_additional_features = model_config['use_additional_features']
        root_dir = data_config['images_dir']

        # ============================================================
        # HERNIA OVERSAMPLING AND AUGMENTATION SETUP
        # ============================================================
        sampling_config = self.config.get('sampling', {})
        hernia_oversample_config = sampling_config.get('hernia_oversample', {})
        hernia_oversample_enabled = hernia_oversample_config.get('enabled', False)
        hernia_oversample_factor = hernia_oversample_config.get('factor', 10)
        hernia_sampler = None
        hernia_augment_transform = None

        if hernia_oversample_enabled:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("HERNIA OVERSAMPLING AND AUGMENTATION ENABLED")
            self.logger.info("=" * 60)
            self.logger.info(f"   Oversample factor: {hernia_oversample_factor}x")
            self.logger.info(f"   Hernia-specific augmentation: ENABLED")

            # Create Hernia-specific augmentation transform
            random_seed = self.config.get('data_split', {}).get('random_state', 42)
            hernia_augment_transform = HerniaAugmentTransform(
                base_size=(224, 224),
                seed=random_seed
            )
            self.logger.info(f"   Augmentations: RandomResizedCrop, Rotation(±5°), ColorJitter, GaussianNoise, RandomErasing")
            self.logger.info("=" * 60)

        # Create training dataset with optional Hernia augmentation
        train_dataset = ChestXRayDataset(
            dataset=train_df, csv_file=None, root_dir=root_dir,
            transform=base_transform, use_additional_features=use_additional_features,
            hernia_augment_enabled=hernia_oversample_enabled,
            hernia_augment_transform=hernia_augment_transform
        )

        # Create Hernia oversampler AFTER dataset is created (needs labels)
        if hernia_oversample_enabled:
            hernia_sampler, hernia_positive_indices = create_hernia_oversampler(
                dataset=train_dataset,
                oversample_factor=hernia_oversample_factor,
                hernia_idx=ChestXRayDataset.HERNIA_IDX,
                seed=self.config.get('data_split', {}).get('random_state', 42),
                logger=self.logger
            )
            self.logger.info(f"   Hernia-positive samples in training set: {len(hernia_positive_indices)}")

        if self.config['augmentation']['rare_class']['enabled']:
            rare_datasets = self.get_rare_class_augmentation(
                train_df, root_dir, rare_transform, use_additional_features
            )
            if rare_datasets:
                train_dataset = ConcatDataset([train_dataset] + rare_datasets)
                self.logger.info(f"Added {len(rare_datasets)} rare class augmentation datasets")
                # Note: If ConcatDataset is used, Hernia sampler won't work correctly
                if hernia_oversample_enabled:
                    self.logger.warning("⚠️  Hernia oversampling disabled: incompatible with ConcatDataset rare_class augmentation")
                    hernia_sampler = None

        # Validation dataset - NO oversampling or Hernia augmentation
        val_dataset = ChestXRayDataset(
            dataset=val_df, csv_file=None, root_dir=root_dir,
            transform=base_transform, use_additional_features=use_additional_features,
            hernia_augment_enabled=False,  # Never augment validation
            hernia_augment_transform=None
        )

        # Create data loaders
        hw_config = self.config['hardware']

        # Use sampler if Hernia oversampling is enabled, otherwise use shuffle
        if hernia_sampler is not None:
            self.logger.info("📊 Using Hernia weighted sampler (shuffle disabled)")
            dataloader_train = DataLoader(
                train_dataset,
                batch_size=training_config['batch_size'],
                sampler=hernia_sampler,  # Use sampler instead of shuffle
                num_workers=training_config['num_workers'],
                pin_memory=hw_config['pin_memory'],
                persistent_workers=hw_config['persistent_workers'],
                prefetch_factor=hw_config['prefetch_factor']
            )
        else:
            dataloader_train = DataLoader(
                train_dataset,
                batch_size=training_config['batch_size'],
                shuffle=True,
                num_workers=training_config['num_workers'],
                pin_memory=hw_config['pin_memory'],
                persistent_workers=hw_config['persistent_workers'],
                prefetch_factor=hw_config['prefetch_factor']
            )

        dataloader_val = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config['num_workers'],
            pin_memory=hw_config['pin_memory'],
            persistent_workers=hw_config['persistent_workers'],
            prefetch_factor=hw_config['prefetch_factor']
        )

        self.logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # Create model
        if model_config['architecture'] == "ModifiedDenseNetWithDropOut":
            model = ModifiedDenseNetWithDropOut(
                num_classes=model_config['num_classes'],
                use_additional_features=use_additional_features,
                head_config=model_config.get('head')
            )
        else:
            model = ModifiedDenseNet(
                num_classes=model_config['num_classes'],
                use_additional_features=use_additional_features
            )

        model = model.to(device)

        # Upgrade to multi-layer MLP head if 'layers' is specified
        head_config = model_config.get('head', {})
        if head_config and head_config.get('layers'):
            model = self._upgrade_to_multilayer_head(model, head_config, use_additional_features, device)

        # Handle frozen backbone configuration (for head-only training)
        if training_config.get('freeze_backbone', False):
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("FROZEN BACKBONE MODE")
            self.logger.info("=" * 80)

            # Load backbone from backbone_source (or parent_iteration as fallback)
            metadata = self.config.get('metadata', {})
            backbone_source = metadata.get('backbone_source', metadata.get('parent_iteration', 12))
            import glob
            pattern = f"experiments/auto_improvement_runs/iteration_{backbone_source:03d}/pipeline_model_*.pth"
            model_files = glob.glob(pattern)

            if model_files:
                checkpoint_path = model_files[0]
                self.logger.info(f"   Backbone source: iteration {backbone_source:03d}")
                self.logger.info(f"   Loading backbone from: {os.path.basename(checkpoint_path)}")
                missing, unexpected = model.load_backbone_from_checkpoint(checkpoint_path, device)
                self.logger.info(f"   Missing keys (expected - new head): {len(missing)}")

            # Freeze backbone
            model.freeze_backbone()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            self.logger.info(f"   Backbone: FROZEN")
            self.logger.info(f"   Trainable parameters: {trainable:,} / {total:,}")
            self.logger.info("=" * 80)

        # Handle partial backbone unfreezing (fine-tuning last block only)
        elif model_config.get('unfreeze_last_blocks') == 1:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("PARTIAL BACKBONE FINE-TUNING MODE")
            self.logger.info("=" * 80)

            # Load weights from anchor_iteration or parent_iteration
            metadata = self.config.get('metadata', {})
            backbone_source = metadata.get('anchor_iteration', metadata.get('parent_iteration', 89))
            import glob
            pattern = f"experiments/auto_improvement_runs/iteration_{backbone_source:03d}/pipeline_model_*.pth"
            model_files = glob.glob(pattern)

            if model_files:
                checkpoint_path = model_files[0]
                self.logger.info(f"   Backbone source: iteration {backbone_source:03d}")
                self.logger.info(f"   Loading weights from: {os.path.basename(checkpoint_path)}")

                # Load full model weights (including head if compatible)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    # Try to load as much as possible, ignore mismatched keys
                    model_state = model.state_dict()
                    compatible_state = {}
                    for key, value in checkpoint.items():
                        if key in model_state and model_state[key].shape == value.shape:
                            compatible_state[key] = value
                    model.load_state_dict(compatible_state, strict=False)
                    self.logger.info(f"   Loaded {len(compatible_state)} / {len(checkpoint)} weight tensors")
                except Exception as e:
                    self.logger.warning(f"   Failed to load checkpoint: {e}")
                    self.logger.warning(f"   Starting with ImageNet pretrained weights")
            else:
                self.logger.warning(f"   No checkpoint found for iteration {backbone_source}")
                self.logger.warning(f"   Starting with ImageNet pretrained weights")

            # First, freeze ALL backbone parameters
            for param in model.base_model.parameters():
                param.requires_grad = False

            # Then, selectively unfreeze ONLY the last block (denseblock4 + norm5)
            unfrozen_layers = []
            unfrozen_params = 0

            # Unfreeze denseblock4
            if hasattr(model.base_model, 'features') and hasattr(model.base_model.features, 'denseblock4'):
                for param in model.base_model.features.denseblock4.parameters():
                    param.requires_grad = True
                block_params = sum(p.numel() for p in model.base_model.features.denseblock4.parameters())
                unfrozen_layers.append('denseblock4')
                unfrozen_params += block_params
                self.logger.info(f"   Unfrozen: denseblock4 ({block_params:,} parameters)")

            # Unfreeze norm5 (final batch norm after denseblock4)
            if hasattr(model.base_model, 'features') and hasattr(model.base_model.features, 'norm5'):
                for param in model.base_model.features.norm5.parameters():
                    param.requires_grad = True
                norm_params = sum(p.numel() for p in model.base_model.features.norm5.parameters())
                unfrozen_layers.append('norm5')
                unfrozen_params += norm_params
                self.logger.info(f"   Unfrozen: norm5 ({norm_params:,} parameters)")

            # Log summary
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            frozen_backbone = sum(p.numel() for p in model.base_model.parameters()) - unfrozen_params

            self.logger.info("")
            self.logger.info(f"   Fine-tuning enabled: unfrozen backbone blocks = {', '.join(unfrozen_layers)}")
            self.logger.info(f"   Unfrozen backbone parameters: {unfrozen_params:,}")
            self.logger.info(f"   Frozen backbone parameters: {frozen_backbone:,}")
            self.logger.info(f"   Total trainable parameters: {trainable:,} / {total:,}")
            self.logger.info("=" * 80)

        # Intelligent warm start (if enabled and not first iteration)
        # SKIP warm start if freeze_backbone is true (backbone already loaded above)
        if training_config.get('freeze_backbone', False):
            self.logger.info("   Skipping intelligent warm start (frozen backbone mode)")
        elif self.iteration > 1 and hasattr(self, 'ai_advisor') and hasattr(self, 'best_iteration_tracker'):
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("🤖 INTELLIGENT WARM START")
            self.logger.info("="*80)

            try:
                # ============================================================
                # ROLE-ENFORCED PARENT (OVERRIDES AI RECOMMENDATION)
                # ============================================================
                # If role_enforced_parent is set by auto_improvement_loop, use it directly
                # This ensures TRAIN_AUC uses iteration 12 and RECOVER_F1 uses iteration 58
                if hasattr(self, 'role_enforced_parent') and self.role_enforced_parent is not None:
                    source_iter = self.role_enforced_parent
                    role_name = getattr(self, 'iteration_role', 'UNKNOWN')

                    self.logger.info(f"   🎯 ROLE-ENFORCED WARM START")
                    self.logger.info(f"   Role: {role_name}")
                    self.logger.info(f"   Parent iteration: {source_iter} (HARD-CODED - AI CANNOT OVERRIDE)")

                    # Find model file for the role-enforced parent
                    import glob
                    pattern = f"experiments/auto_improvement_runs/iteration_{source_iter:03d}/pipeline_model_*.pth"
                    model_files = glob.glob(pattern)

                    if model_files:
                        model_path = model_files[0]
                        self.logger.info(f"   📦 Loading weights from: {os.path.basename(model_path)}")

                        try:
                            state_dict = torch.load(model_path, map_location=device)
                            model.load_state_dict(state_dict)
                            self.logger.info(f"   ✅ Successfully loaded weights from iteration {source_iter}")
                            self.logger.info(f"   ✅ This ensures we start from the best {role_name} anchor")
                        except Exception as e:
                            self.logger.error(f"   ❌ CRITICAL: Failed to load role-enforced parent weights: {e}")
                            self.logger.error(f"   ❌ This is a critical error - role-based training requires parent weights!")
                            raise RuntimeError(f"Cannot load required parent iteration {source_iter} weights")
                    else:
                        self.logger.error(f"   ❌ CRITICAL: Model file not found for role-enforced parent iteration {source_iter}")
                        self.logger.error(f"   ❌ Expected pattern: {pattern}")
                        raise RuntimeError(f"Cannot find required parent iteration {source_iter} model file")

                    self.logger.info("="*80)

                # ============================================================
                # AI-BASED WARM START (FALLBACK WHEN NO ROLE ENFORCEMENT)
                # ============================================================
                else:
                    # Get previous iteration data
                    previous_iter_num = self.iteration - 1
                    previous_iter_dir = f"experiments/auto_improvement_runs/iteration_{previous_iter_num:03d}"
                    previous_summary_file = f"{previous_iter_dir}/iteration_summary.json"

                    previous_iteration_data = {}
                    if os.path.exists(previous_summary_file):
                        with open(previous_summary_file, 'r') as f:
                            previous_iteration_data = json.load(f)
                    else:
                        self.logger.warning(f"   ⚠️  Previous iteration summary not found, using defaults")
                        previous_iteration_data = {'iteration': previous_iter_num}

                    # Get best iterations data with task awareness
                    current_task = None
                    if hasattr(self, 'task_manager'):
                        current_task = self.task_manager.get_current_task()

                    best_iterations_data = self.best_iteration_tracker.get_comparison_data(
                        self.iteration,
                        task_info=current_task
                    )

                    # Add current task to best iterations data for AI advisor
                    if current_task:
                        best_iterations_data['current_task'] = current_task

                    # Get AI recommendation
                    self.logger.info(f"   Requesting AI warm start recommendation...")

                    recommendation = self.ai_advisor.recommend_warm_start(
                        current_iteration=self.iteration,
                        previous_iteration_data=previous_iteration_data,
                        best_iterations_data=best_iterations_data,
                        proposed_config=self.config,
                        previous_config=previous_iteration_data.get('config', self.config)
                    )

                    self.logger.info("")
                    self.logger.info(f"   🎯 AI DECISION: {recommendation['warm_start_from']}")
                    self.logger.info(f"   Reasoning: {recommendation['reasoning']}")
                    self.logger.info(f"   Confidence: {recommendation['confidence']:.0%}")
                    self.logger.info(f"   Expected benefit: {recommendation.get('expected_benefit', 'N/A')}")

                    # Apply recommendation
                    if recommendation['warm_start_from'] != 'cold_start':
                        source_iter_str = recommendation['warm_start_from']

                        if source_iter_str.startswith('iteration_'):
                            source_iter = int(source_iter_str.split('_')[1])

                            # Find model file
                            import glob
                            pattern = f"experiments/auto_improvement_runs/iteration_{source_iter:03d}/pipeline_model_*.pth"
                            model_files = glob.glob(pattern)

                            if model_files:
                                model_path = model_files[0]
                                self.logger.info(f"   📦 Loading weights from: {os.path.basename(model_path)}")

                                try:
                                    state_dict = torch.load(model_path, map_location=device)
                                    model.load_state_dict(state_dict)
                                    self.logger.info(f"   ✅ Successfully loaded weights from iteration {source_iter}")
                                except Exception as e:
                                    self.logger.warning(f"   ⚠️  Failed to load weights: {e}")
                                    self.logger.warning(f"   ⚠️  Falling back to ImageNet weights")
                            else:
                                self.logger.warning(f"   ⚠️  Model file not found for iteration {source_iter}")
                                self.logger.warning(f"   ⚠️  Falling back to ImageNet weights")
                    else:
                        self.logger.info(f"   🆕 Using ImageNet pretrained weights (cold start)")

                    self.logger.info("="*80)

            except Exception as e:
                self.logger.error(f"   ❌ Error in warm start: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.info(f"   🆕 Defaulting to ImageNet weights")
                self.logger.info("="*80)

        # Multi-GPU support
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        # Extract training labels for dynamic weight computation
        train_labels_list = []
        for idx in range(len(train_df)):
            label_columns = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                           'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                           'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                           'Pleural_Thickening', 'Hernia']
            labels = train_df.iloc[idx][label_columns].values.astype('float32')
            train_labels_list.append(labels)
        train_labels_tensor = torch.tensor(np.array(train_labels_list), dtype=torch.float32)

        # Loss function (with optional dynamic class weights)
        criterion = self.create_loss_function(device, train_labels_tensor)
        loss_type_display = self.config['loss']['type']
        if loss_type_display.lower() in ['focal', 'focalloss']:
            loss_type_display = 'FocalLoss'
        elif loss_type_display.lower() in ['bce', 'weightedbce', 'bceloss']:
            loss_type_display = 'WeightedBCE'
        self.logger.info(f"Loss function: {loss_type_display}")

        # Optimizer
        optimizer = self._create_optimizer(model, training_config['optimizer'], training_config['learning_rate'])

        # Scheduler
        scheduler = self._create_scheduler(optimizer, training_config['scheduler'])

        # Early stopping
        es_config = training_config['early_stopping']
        if es_config['enabled']:
            # Support new monitor/mode parameters or fallback to defaults
            monitor = es_config.get('monitor', 'val_loss')
            mode = es_config.get('mode', 'min')
            early_stopping = EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=es_config['patience'],
                min_delta=es_config['min_delta'],
                warmup_epochs=es_config['warmup_epochs']
            )
            self.logger.info(f"Early stopping enabled: monitor={monitor}, mode={mode}, patience={es_config['patience']}")
        else:
            early_stopping = None

        # ================================================================
        # FEATURE-SPACE SMOTE MODE (if enabled in config)
        # ================================================================
        smote_config = self.config.get('smote', {})
        if smote_config.get('enabled', False) and FEATURE_SMOTE_AVAILABLE:
            return self._run_feature_smote_training(
                model=model,
                train_df=train_df,
                val_df=val_df,
                dataloader_train=dataloader_train,
                dataloader_val=dataloader_val,
                device=device,
                use_additional_features=use_additional_features,
                training_config=training_config,
                smote_config=smote_config,
                criterion=criterion
            )

        # Training loop
        self.logger.info("="*60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Total epochs: {training_config['num_epochs']}")
        self.logger.info(f"Batches per epoch: {len(dataloader_train)}")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info("="*60)

        start_time = time.time()
        total_batches = len(dataloader_train) * training_config['num_epochs']
        batches_processed = 0

        # Track final losses for iteration analysis
        final_train_loss = 0.0
        final_val_loss = 0.0
        actual_epochs = 0  # Track how many epochs were actually completed

        for epoch in range(training_config['num_epochs']):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0

            self.logger.info(f"\n📊 EPOCH {epoch + 1}/{training_config['num_epochs']}")

            for batch_idx, batch in enumerate(dataloader_train, start=1):
                if use_additional_features:
                    images, additional_features, labels = batch
                    images = images.to(device)
                    additional_features = additional_features.to(device)
                    labels = labels.to(device)
                    outputs = model(images, additional_features)
                else:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batches_processed += 1

                if batch_idx % self.config['logging']['log_interval'] == 0:
                    # Calculate progress
                    batch_progress = (batch_idx / len(dataloader_train)) * 100
                    overall_progress = (batches_processed / total_batches) * 100
                    elapsed_time = time.time() - start_time
                    batches_per_sec = batches_processed / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (total_batches - batches_processed) / batches_per_sec if batches_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60

                    self.logger.info(
                        f"  Batch {batch_idx:4d}/{len(dataloader_train)} ({batch_progress:5.1f}%) | "
                        f"Loss: {loss.item():.4f} | "
                        f"Overall: {overall_progress:5.1f}% | "
                        f"ETA: {eta_minutes:.1f}min"
                    )

            epoch_loss = running_loss / len(dataloader_train)
            final_train_loss = epoch_loss  # Store final training loss

            # Validation
            self.logger.info("  🔍 Running validation...")
            model.eval()
            val_loss = 0.0
            val_preds_all = []
            val_labels_all = []
            val_probs_all = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_val, start=1):
                    if use_additional_features:
                        images, additional_features, labels = batch
                        images = images.to(device)
                        additional_features = additional_features.to(device)
                        labels = labels.to(device)
                        outputs = model(images, additional_features)
                    else:
                        images, labels = batch
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                    val_loss += criterion(outputs, labels).item()

                    # Collect predictions for metrics computation
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()  # Use 0.5 for validation monitoring

                    val_probs_all.append(probs.cpu())
                    val_preds_all.append(preds.cpu())
                    val_labels_all.append(labels.cpu())

                    # Show validation progress occasionally
                    if batch_idx % 50 == 0 or batch_idx == len(dataloader_val):
                        val_progress = (batch_idx / len(dataloader_val)) * 100
                        self.logger.info(f"     Validation batch {batch_idx}/{len(dataloader_val)} ({val_progress:.1f}%)")

            val_loss /= len(dataloader_val)
            final_val_loss = val_loss  # Store final validation loss

            # Compute validation metrics for early stopping
            val_preds_all = torch.cat(val_preds_all, dim=0).numpy()
            val_labels_all = torch.cat(val_labels_all, dim=0).numpy()
            val_probs_all = torch.cat(val_probs_all, dim=0).numpy()

            # Compute average F1 score across all classes
            from sklearn.metrics import f1_score, roc_auc_score
            val_f1 = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)

            # Compute average AUC score across all classes
            try:
                val_auc = roc_auc_score(val_labels_all, val_probs_all, average='macro')
            except:
                val_auc = 0.0  # If AUC computation fails

            epoch_time = time.time() - epoch_start_time

            # Calculate overall progress
            epoch_progress = ((epoch + 1) / training_config['num_epochs']) * 100
            elapsed_total = time.time() - start_time
            avg_epoch_time = elapsed_total / (epoch + 1)
            remaining_epochs = training_config['num_epochs'] - (epoch + 1)
            eta_total_seconds = remaining_epochs * avg_epoch_time
            eta_total_minutes = eta_total_seconds / 60

            self.logger.info("")
            self.logger.info("="*60)
            self.logger.info(f"✅ EPOCH {epoch + 1}/{training_config['num_epochs']} COMPLETED ({epoch_progress:.1f}% of training)")
            self.logger.info(f"   Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.logger.info(f"   Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            self.logger.info(f"   Epoch Time: {epoch_time:.1f}s | Total Elapsed: {elapsed_total/60:.1f}min")
            if remaining_epochs > 0:
                self.logger.info(f"   ETA for remaining {remaining_epochs} epochs: {eta_total_minutes:.1f}min")
            self.logger.info("="*60)

            # Send Telegram notification for epoch completion
            if self.telegram:
                self.telegram.add_log(f"Epoch {epoch + 1}/{training_config['num_epochs']}: Train={epoch_loss:.4f}, Val={val_loss:.4f}")
                self.telegram.send_epoch_complete(
                    epoch=epoch + 1,
                    total_epochs=training_config['num_epochs'],
                    train_loss=epoch_loss,
                    val_loss=val_loss,
                    iteration=self.iteration,
                    time_seconds=epoch_time
                )

            # Track completed epoch
            actual_epochs = epoch + 1

            # Scheduler step
            if scheduler is not None:
                scheduler.step(val_loss)

            # Early stopping - use the monitored metric
            if early_stopping is not None:
                # Get the metric value based on what's being monitored
                # Note: val_macro_auc is the same as val_auc (both are macro-averaged)
                metric_map = {
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_auc': val_auc,
                    'val_macro_auc': val_auc  # Alias for clarity in TRAIN_AUC role
                }
                monitored_value = metric_map.get(early_stopping.monitor, val_loss)

                if early_stopping(monitored_value):
                    es_status = early_stopping.get_status()
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    self.logger.info(f"  Monitored metric: {early_stopping.monitor}")
                    self.logger.info(f"  Best {early_stopping.monitor}: {early_stopping.best_score:.4f}")
                    self.logger.info(f"  Patience counter: {es_status['counter']}/{es_status['patience']}")
                    break

        # Save model
        self.logger.info("Training complete. Saving model...")
        torch.save(model.state_dict(), self.model_file)
        self.logger.info(f"Model saved as {self.model_file}")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        # Log final losses for AI analysis
        self.logger.info(f"Final Train Loss: {final_train_loss:.4f}")
        self.logger.info(f"Final Val Loss: {final_val_loss:.4f}")

        # Optimize thresholds on validation set
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("OPTIMIZING CLASSIFICATION THRESHOLDS")
        self.logger.info("="*60)
        threshold_file = self._optimize_thresholds(model, dataloader_val, device, use_additional_features)

        # Load optimized thresholds
        thresholds_dict = {}
        if threshold_file and os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                thresholds_dict = json.load(f)

        # Collect training metadata for iteration_summary.json
        training_metadata = {
            'actual_epochs': actual_epochs,
            'thresholds': thresholds_dict,
            'description': self.config.get('metadata', {}).get('description', ''),
        }

        # Add early stopping metadata if enabled
        if early_stopping is not None:
            es_status = early_stopping.get_status()
            training_metadata.update({
                'early_stopping_monitor': early_stopping.monitor,
                'early_stopping_mode': early_stopping.mode,
                'min_delta': early_stopping.min_delta,
                'patience': early_stopping.patience,
                'best_score': es_status.get('best_score'),
            })

        return model, final_train_loss, final_val_loss, training_metadata

    def _run_feature_smote_training(
        self,
        model,
        train_df,
        val_df,
        dataloader_train,
        dataloader_val,
        device,
        use_additional_features,
        training_config,
        smote_config,
        criterion
    ):
        """
        Run feature-space SMOTE training (head-only on embeddings).

        This method:
        1. Extracts embeddings from frozen backbone
        2. Applies SMOTE on training embeddings for target class
        3. Trains only the classification head on real + synthetic embeddings
        4. Uses standard evaluation pipeline

        Args:
            model: Model with frozen backbone
            train_df: Training dataframe
            val_df: Validation dataframe
            dataloader_train: Training dataloader
            dataloader_val: Validation dataloader
            device: torch device
            use_additional_features: Whether model uses additional features
            training_config: Training configuration
            smote_config: SMOTE configuration
            criterion: Loss function

        Returns:
            Tuple of (model, train_loss, val_loss, training_metadata)
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("FEATURE-SPACE SMOTE MODE")
        self.logger.info("=" * 80)

        target_class = smote_config.get('target_class', 'Hernia')
        sampling_ratio = smote_config.get('sampling_ratio', 4)
        k_neighbors = smote_config.get('k_neighbors', 3)
        seed = smote_config.get('seed', 42)

        self.logger.info(f"   Target class: {target_class}")
        self.logger.info(f"   Sampling ratio: {sampling_ratio}x")
        self.logger.info(f"   K neighbors: {k_neighbors}")
        self.logger.info(f"   Seed: {seed}")
        self.logger.info("=" * 80)

        # Step 1: Extract embeddings
        self.logger.info("")
        self.logger.info("STEP 1: Extracting embeddings from frozen backbone")
        self.logger.info("-" * 60)

        feature_extractor = FeatureExtractor(model, device, use_additional_features)

        # Check for cached embeddings
        metadata = self.config.get('metadata', {})
        backbone_source = metadata.get('backbone_source', metadata.get('parent_iteration', 12))
        train_cache_path = get_embedding_cache_path(backbone_source, 'train')
        val_cache_path = get_embedding_cache_path(backbone_source, 'val')

        # Extract or load training embeddings
        if os.path.exists(train_cache_path):
            self.logger.info(f"   Loading cached train embeddings from {train_cache_path}")
            train_embeddings, train_labels, _ = load_cached_embeddings(train_cache_path)
        else:
            self.logger.info("   Extracting training embeddings...")
            train_embeddings, train_labels = feature_extractor.extract_embeddings(dataloader_train, self.logger)
            cache_embeddings(train_embeddings, train_labels, train_cache_path, {'backbone_source': backbone_source})
            self.logger.info(f"   Cached train embeddings to {train_cache_path}")

        # Extract or load validation embeddings
        if os.path.exists(val_cache_path):
            self.logger.info(f"   Loading cached val embeddings from {val_cache_path}")
            val_embeddings, val_labels, _ = load_cached_embeddings(val_cache_path)
        else:
            self.logger.info("   Extracting validation embeddings...")
            val_embeddings, val_labels = feature_extractor.extract_embeddings(dataloader_val, self.logger)
            cache_embeddings(val_embeddings, val_labels, val_cache_path, {'backbone_source': backbone_source})
            self.logger.info(f"   Cached val embeddings to {val_cache_path}")

        # Step 2: Apply Feature-Space SMOTE
        self.logger.info("")
        self.logger.info("STEP 2: Applying Feature-Space SMOTE")
        self.logger.info("-" * 60)

        smote_augmenter = FeatureSMOTE(
            target_class=target_class,
            sampling_ratio=sampling_ratio,
            k_neighbors=k_neighbors,
            seed=seed,
            logger=self.logger
        )

        synthetic_embeddings, synthetic_labels, combined_embeddings, combined_labels = smote_augmenter.apply(
            train_embeddings, train_labels
        )

        # Step 3: Train head on embeddings
        self.logger.info("")
        self.logger.info("STEP 3: Training classification head on embeddings")
        self.logger.info("-" * 60)

        head_trainer = EmbeddingHeadTrainer(
            model=model,
            device=device,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['optimizer'].get('weight_decay', 0.0001),
            logger=self.logger
        )

        # Training parameters
        num_epochs = training_config['num_epochs']
        batch_size = training_config['batch_size']
        es_config = training_config['early_stopping']

        # Early stopping setup
        best_val_auc = 0.0
        patience_counter = 0
        patience = es_config.get('patience', 5)
        min_delta = es_config.get('min_delta', 0.001)
        warmup_epochs = es_config.get('warmup_epochs', 3)

        final_train_loss = 0.0
        final_val_loss = 0.0
        actual_epochs = 0

        self.logger.info(f"   Training for up to {num_epochs} epochs")
        self.logger.info(f"   Early stopping: patience={patience}, min_delta={min_delta}, warmup={warmup_epochs}")

        start_time = time.time()

        for epoch in range(num_epochs):
            # Train epoch on combined embeddings
            train_loss = head_trainer.train_epoch_on_embeddings(
                combined_embeddings, combined_labels, batch_size, shuffle=True
            )
            final_train_loss = train_loss

            # Validate
            val_loss, val_auc, val_f1, val_preds, val_probs = self._validate_on_embeddings(
                model, val_embeddings, val_labels, device, criterion, batch_size
            )
            final_val_loss = val_loss
            actual_epochs = epoch + 1

            # Log progress
            self.logger.info(f"   Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.4f}, "
                           f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}")

            # Early stopping check (after warmup)
            if epoch >= warmup_epochs:
                if val_auc > best_val_auc + min_delta:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.model_file)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"   Early stopping triggered at epoch {epoch + 1}")
                        break
            else:
                # During warmup, still track best
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), self.model_file)

        total_time = time.time() - start_time
        self.logger.info(f"   Training completed in {total_time:.1f}s ({actual_epochs} epochs)")

        # Ensure model is saved
        if not os.path.exists(self.model_file):
            torch.save(model.state_dict(), self.model_file)

        # Load best model
        model.load_state_dict(torch.load(self.model_file, map_location=device))

        # Compute Hernia-specific metrics for AI advisory
        _, _, _, final_preds, final_probs = self._validate_on_embeddings(
            model, val_embeddings, val_labels, device, criterion, batch_size
        )
        hernia_metrics = compute_hernia_metrics(val_labels, final_preds, final_probs)

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("HERNIA METRICS (for AI Advisory)")
        self.logger.info("=" * 60)
        for key, val in hernia_metrics.items():
            self.logger.info(f"   {key}: {val}")
        self.logger.info("=" * 60)

        # Optimize thresholds
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZING CLASSIFICATION THRESHOLDS")
        self.logger.info("=" * 60)
        threshold_file = self._optimize_thresholds(model, dataloader_val, device, use_additional_features)

        # Load optimized thresholds
        thresholds_dict = {}
        if threshold_file and os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                thresholds_dict = json.load(f)

        # Collect training metadata
        training_metadata = {
            'actual_epochs': actual_epochs,
            'thresholds': thresholds_dict,
            'description': self.config.get('metadata', {}).get('description', ''),
            'smote': {
                'enabled': True,
                'mode': 'feature_space',
                'target_class': target_class,
                'sampling_ratio': sampling_ratio,
                'k_neighbors': k_neighbors,
                'original_positive_count': smote_augmenter.original_positive_count,
                'synthetic_count': smote_augmenter.synthetic_count,
            },
            'hernia_metrics': hernia_metrics,
            'best_val_auc': best_val_auc,
        }

        return model, final_train_loss, final_val_loss, training_metadata

    def _validate_on_embeddings(self, model, embeddings, labels, device, criterion, batch_size):
        """
        Validate model on pre-computed embeddings.

        Returns:
            Tuple of (loss, auc, f1, predictions, probabilities)
        """
        from sklearn.metrics import roc_auc_score, f1_score

        model.eval()
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        dataset = TensorDataset(emb_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_emb, batch_labels in dataloader:
                batch_emb = batch_emb.to(device)
                batch_labels = batch_labels.to(device)

                # Forward through head only
                if model.head_type == 'mlp_multi':
                    # Multi-layer MLP head (Sequential)
                    out = model.mlp_head(batch_emb)
                elif model.head_type == 'mlp':
                    # Single-layer MLP head (backward compatible)
                    out = model.head_fc1(batch_emb)
                    out = model.head_activation(out)
                    out = model.head_dropout(out)
                    out = model.final_fc(out)
                else:
                    # Linear head
                    out = model.final_fc(batch_emb)

                loss = criterion(out, batch_labels)
                total_loss += loss.item()

                probs = torch.sigmoid(out)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        all_preds = (all_probs >= 0.5).astype(float)

        avg_loss = total_loss / len(dataloader)

        try:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
        except:
            auc = 0.0

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, auc, f1, all_preds, all_probs

    def _optimize_thresholds(self, model, dataloader_val, device, use_additional_features):
        """
        Optimize classification thresholds on validation set.

        Supports three modes via config['thresholds']:
        1. Default: Optimize all classes for F1-score (legacy behavior)
        2. Disabled: Use fixed default threshold (0.5) for all classes
        3. Hernia-only constrained: Optimize only Hernia with precision/FP constraints

        Args:
            model: Trained model
            dataloader_val: Validation data loader
            device: Device to run on
            use_additional_features: Whether model uses additional features

        Returns:
            Path to saved threshold file
        """
        from threshold_optimizer import optimize_thresholds_per_class, save_thresholds, optimize_hernia_only_constrained
        import numpy as np

        # Get threshold config
        threshold_config = self.config.get('thresholds', {})
        optimize_disabled = threshold_config.get('disable_optimization', False)
        optimize_hernia_only = threshold_config.get('optimize_hernia_only', False)
        hernia_constraint_with_legacy = threshold_config.get('hernia_constraint_with_legacy', False)
        hernia_constraint = threshold_config.get('hernia_constraint', {})
        default_threshold = threshold_config.get('default_threshold', 0.5)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("THRESHOLD OPTIMIZATION CONFIGURATION")
        self.logger.info("=" * 60)
        self.logger.info(f"  disable_optimization: {optimize_disabled}")
        self.logger.info(f"  optimize_hernia_only: {optimize_hernia_only}")
        self.logger.info(f"  hernia_constraint_with_legacy: {hernia_constraint_with_legacy}")
        self.logger.info(f"  default_threshold: {default_threshold}")
        if hernia_constraint:
            self.logger.info(f"  hernia_constraint: {hernia_constraint}")
        self.logger.info("=" * 60)

        # Define class names (same order as in dataset)
        class_names = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass',
                      'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
                      'Fibrosis', 'Edema', 'Consolidation']

        # Mode 1: Optimization disabled - use fixed default threshold
        if optimize_disabled:
            self.logger.info("\n⚠️  THRESHOLD OPTIMIZATION DISABLED - using fixed thresholds")
            threshold_details = {}
            for class_name in class_names:
                threshold_details[class_name] = {
                    'threshold': default_threshold,
                    'score': 0.0,
                    'positive_samples': 0,
                    'status': 'fixed_disabled'
                }
            self.logger.info(f"   All classes using threshold: {default_threshold}")

            # Save thresholds to JSON file
            threshold_file = self.model_file.replace('pipeline_model_', 'thresholds_').replace('.pth', '.json')
            save_thresholds(threshold_details, threshold_file, logger=self.logger)
            return threshold_file

        # Collect validation predictions (needed for both optimization modes)
        self.logger.info("Collecting validation predictions...")

        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader_val:
                if use_additional_features:
                    images, additional_features, labels = batch
                    images = images.to(device)
                    additional_features = additional_features.to(device)
                    labels = labels.to(device)
                    outputs = model(images, additional_features)
                else:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        self.logger.info(f"Validation set size: {all_probs.shape[0]} samples")

        # Check for hybrid mode: legacy optimization + Hernia constraint
        hernia_constraint_with_legacy = threshold_config.get('hernia_constraint_with_legacy', False)

        # Mode 2: Hernia-only constrained optimization (others fixed at default)
        if optimize_hernia_only and not hernia_constraint_with_legacy:
            self.logger.info("\n🎯 HERNIA-ONLY CONSTRAINED OPTIMIZATION MODE")
            optimal_thresholds, threshold_details = optimize_hernia_only_constrained(
                y_true=all_labels,
                y_pred_probs=all_probs,
                class_names=class_names,
                hernia_constraint=hernia_constraint,
                default_threshold=default_threshold,
                num_thresholds=50,
                logger=self.logger
            )
        elif hernia_constraint_with_legacy and hernia_constraint:
            # Mode 2b: HYBRID - Legacy F1 optimization for all classes + Hernia constraint
            self.logger.info("\n🔀 HYBRID MODE: Legacy F1 optimization + Hernia constraint")

            # Step 1: Run legacy per-class F1 optimization for all classes
            optimal_thresholds, threshold_details = optimize_thresholds_per_class(
                y_true=all_labels,
                y_pred_probs=all_probs,
                class_names=class_names,
                metric='f1',
                num_thresholds=19,
                logger=self.logger
            )

            # Step 2: Apply Hernia constraint on top
            if 'Hernia' in class_names:
                hernia_idx = class_names.index('Hernia')
                min_precision = hernia_constraint.get('min_precision', 0.02)
                max_fp = hernia_constraint.get('max_fp', 500)

                self.logger.info(f"\n🎯 Applying Hernia constraint (min_precision={min_precision}, max_fp={max_fp})")

                # Get current Hernia threshold from legacy optimization
                current_hernia_threshold = optimal_thresholds[hernia_idx]
                y_true_hernia = all_labels[:, hernia_idx]
                y_pred_hernia = all_probs[:, hernia_idx]

                # Check if current threshold satisfies constraints
                preds = (y_pred_hernia >= current_hernia_threshold).astype(int)
                tp = np.sum((preds == 1) & (y_true_hernia == 1))
                fp = np.sum((preds == 1) & (y_true_hernia == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0

                self.logger.info(f"  Legacy threshold: {current_hernia_threshold:.4f} -> TP={tp}, FP={fp}, Precision={precision:.4f}")

                # If constraint not satisfied, search for better threshold
                if precision < min_precision and fp > max_fp:
                    self.logger.info("  Constraint NOT satisfied, searching for constrained threshold...")

                    best_threshold = current_hernia_threshold
                    best_f1 = 0

                    for t in np.linspace(0.01, 0.99, 50):
                        preds_t = (y_pred_hernia >= t).astype(int)
                        tp_t = np.sum((preds_t == 1) & (y_true_hernia == 1))
                        fp_t = np.sum((preds_t == 1) & (y_true_hernia == 0))
                        fn_t = np.sum((preds_t == 0) & (y_true_hernia == 1))

                        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
                        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0

                        # Check constraint satisfaction
                        if precision_t >= min_precision or fp_t <= max_fp:
                            if f1_t > best_f1:
                                best_f1 = f1_t
                                best_threshold = t

                    optimal_thresholds[hernia_idx] = best_threshold

                    # Update threshold details
                    preds_final = (y_pred_hernia >= best_threshold).astype(int)
                    tp_final = np.sum((preds_final == 1) & (y_true_hernia == 1))
                    fp_final = np.sum((preds_final == 1) & (y_true_hernia == 0))
                    precision_final = tp_final / (tp_final + fp_final) if (tp_final + fp_final) > 0 else 0

                    threshold_details['Hernia'] = {
                        'threshold': float(best_threshold),
                        'tp': int(tp_final),
                        'fp': int(fp_final),
                        'precision': float(precision_final),
                        'status': 'hybrid_constrained'
                    }
                    self.logger.info(f"  Constrained threshold: {best_threshold:.4f} -> TP={tp_final}, FP={fp_final}")
                else:
                    self.logger.info("  Constraint already satisfied with legacy threshold")
                    threshold_details['Hernia']['status'] = 'legacy_satisfied'
        else:
            # Mode 3: Default - optimize all classes for F1
            self.logger.info("\n📊 STANDARD PER-CLASS F1 OPTIMIZATION")
            optimal_thresholds, threshold_details = optimize_thresholds_per_class(
                y_true=all_labels,
                y_pred_probs=all_probs,
                class_names=class_names,
                metric='f1',  # Optimize for F1-score
                num_thresholds=19,
                logger=self.logger
            )

        # Save thresholds to JSON file
        threshold_file = self.model_file.replace('pipeline_model_', 'thresholds_').replace('.pth', '.json')
        save_thresholds(threshold_details, threshold_file, logger=self.logger)

        return threshold_file

    def _upgrade_to_multilayer_head(self, model, head_config, use_additional_features, device):
        """
        Upgrade model's MLP head to support multiple hidden layers.

        This method replaces the single-layer MLP head with a multi-layer Sequential head
        when 'layers' is specified in head_config.

        Args:
            model: The model to upgrade
            head_config: Head configuration dict containing 'layers', 'dropout', 'activation'
            use_additional_features: Whether model uses additional features
            device: torch device

        Returns:
            Model with upgraded head
        """
        layers = head_config.get('layers')
        if not layers:
            return model

        # Get configuration
        dropout = head_config.get('dropout', 0.2)
        activation_name = head_config.get('activation', 'relu')
        num_classes = self.config['model']['num_classes']

        # Determine input dimension (1024 from DenseNet + 128 if additional features)
        in_features = 1024 + 128 if use_additional_features else 1024

        # Select activation function
        if activation_name == 'relu':
            activation_fn = nn.ReLU
        elif activation_name == 'gelu':
            activation_fn = nn.GELU
        else:
            activation_fn = nn.ReLU

        # Build multi-layer head
        modules = []
        prev_dim = in_features

        for i, dim in enumerate(layers):
            modules.append(nn.Linear(prev_dim, dim))
            modules.append(activation_fn())
            modules.append(nn.Dropout(dropout))
            prev_dim = dim

        # Final classification layer
        modules.append(nn.Linear(prev_dim, num_classes))

        # Create Sequential head
        mlp_head = nn.Sequential(*modules).to(device)

        # Store the new head on the model
        model.mlp_head = mlp_head
        model.head_type = 'mlp_multi'

        # Monkey-patch the forward method to use multi-layer head
        original_forward = model.forward

        def patched_forward(x, additional_features=None):
            """Patched forward that uses multi-layer MLP head."""
            # Get base features from backbone
            base_out = model.base_model(x)
            base_out = model.dropout(base_out)

            if model.use_additional_features and additional_features is not None:
                additional_out = nn.functional.relu(model.additional_fc(additional_features))
                combined = torch.cat([base_out, additional_out], dim=1)
            else:
                combined = base_out

            # Use multi-layer head
            out = model.mlp_head(combined)
            return out

        model.forward = patched_forward

        # Log the upgrade
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("MULTI-LAYER MLP HEAD UPGRADE")
        self.logger.info("=" * 60)
        self.logger.info(f"   Layers: {layers}")
        self.logger.info(f"   Activation: {activation_name}")
        self.logger.info(f"   Dropout: {dropout}")
        self.logger.info(f"   Input dim: {in_features}")
        self.logger.info(f"   Architecture: {in_features} -> {' -> '.join(map(str, layers))} -> {num_classes}")

        # Count parameters
        head_params = sum(p.numel() for p in mlp_head.parameters())
        self.logger.info(f"   Head parameters: {head_params:,}")
        self.logger.info("=" * 60)

        return model

    def _get_device(self):
        """Get device based on config"""
        device_config = self.config['hardware']['device']

        if device_config == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_config)

    def _create_optimizer(self, model, optimizer_config, learning_rate):
        """Create optimizer from config"""
        optimizer_type = optimizer_config['type']

        # For frozen/partially frozen backbone, only optimize trainable parameters
        freeze_backbone = self.config.get('training', {}).get('freeze_backbone', False)
        unfreeze_last_blocks = self.config.get('model', {}).get('unfreeze_last_blocks')

        if freeze_backbone or unfreeze_last_blocks:
            params = filter(lambda p: p.requires_grad, model.parameters())
        else:
            params = model.parameters()

        if optimizer_type == "Adam":
            return optim.Adam(
                params,
                lr=learning_rate,
                betas=tuple(optimizer_config['betas']),
                eps=optimizer_config['eps'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _create_scheduler(self, optimizer, scheduler_config):
        """Create learning rate scheduler from config"""
        scheduler_type = scheduler_config['type']

        if scheduler_type == "NONE" or scheduler_type is None:
            # No scheduler - return None
            return None
        elif scheduler_type == "ReduceLROnPlateau":
            # Note: 'verbose' parameter not supported in all PyTorch versions
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config['mode'],
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Example usage
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = ConfigBasedTrainer("experiments/configs/config_baseline.yaml", timestamp)
    trainer.train()