"""
Feature-Space SMOTE Module for Chest X-Ray Classification

This module implements DeepSMOTE-style feature-space oversampling:
- Extract embeddings from frozen backbone
- Apply SMOTE only on feature embeddings (not images)
- Target specific rare classes (e.g., Hernia)

Critical constraints:
- Does NOT modify dataset.py, auto_improvement_loop.py, or chest_xray_pipeline.py
- Does NOT activate Phase 5 logic or class weights
- Backbone remains frozen throughout
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, Optional, List
import logging

# SMOTE import
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Feature-SMOTE disabled.")


# Class name to index mapping
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class FeatureExtractor:
    """
    Extract deep embeddings from a frozen model backbone.

    The embeddings are the features right before the classification head,
    after the backbone and dropout layers.
    """

    def __init__(self, model: nn.Module, device: torch.device, use_additional_features: bool = True):
        """
        Initialize the feature extractor.

        Args:
            model: The full model (ModifiedDenseNetWithDropOut)
            device: torch device
            use_additional_features: Whether model uses additional features
        """
        self.model = model
        self.device = device
        self.use_additional_features = use_additional_features
        self.model.eval()

        # Determine feature dimension
        self.feature_dim = 1024 + 128 if use_additional_features else 1024

    def extract_embeddings(self, dataloader: DataLoader, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for all samples in a dataloader.

        Args:
            dataloader: DataLoader with images and labels
            logger: Optional logger

        Returns:
            Tuple of (embeddings, labels) as numpy arrays
        """
        all_embeddings = []
        all_labels = []

        if logger:
            logger.info(f"   Extracting embeddings from {len(dataloader)} batches...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if self.use_additional_features:
                    images, additional_features, labels = batch
                    images = images.to(self.device)
                    additional_features = additional_features.to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    additional_features = None

                # Extract embeddings (features before classification head)
                embeddings = self._get_embeddings(images, additional_features)

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())

                if logger and (batch_idx + 1) % 50 == 0:
                    logger.info(f"      Processed {batch_idx + 1}/{len(dataloader)} batches")

        embeddings = np.vstack(all_embeddings)
        labels = np.vstack(all_labels)

        if logger:
            logger.info(f"   Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

        return embeddings, labels

    def _get_embeddings(self, images: torch.Tensor, additional_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings from the model (features before classification head).

        This extracts the 'combined' tensor from ModifiedDenseNetWithDropOut.forward()
        """
        # Get base features from backbone
        base_out = self.model.base_model(images)
        base_out = self.model.dropout(base_out)

        if self.use_additional_features and additional_features is not None:
            additional_out = nn.ReLU()(self.model.additional_fc(additional_features))
            combined = torch.cat([base_out, additional_out], dim=1)
        else:
            combined = base_out

        return combined


class FeatureSMOTE:
    """
    Apply SMOTE in feature space for rare class augmentation.

    This is a DeepSMOTE-style approach:
    1. Extract embeddings from frozen backbone
    2. Apply SMOTE only on embeddings of target class
    3. Synthetic samples get target class label = 1, all others = 0
    """

    def __init__(
        self,
        target_class: str,
        sampling_ratio: int = 4,
        k_neighbors: int = 3,
        seed: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FeatureSMOTE.

        Args:
            target_class: Name of the class to oversample (e.g., 'Hernia')
            sampling_ratio: How many times to multiply the minority class
            k_neighbors: Number of neighbors for SMOTE
            seed: Random seed for reproducibility
            logger: Optional logger
        """
        if not SMOTE_AVAILABLE:
            raise RuntimeError("imbalanced-learn is required for FeatureSMOTE. Install with: pip install imbalanced-learn")

        self.target_class = target_class
        self.target_idx = CLASS_TO_IDX.get(target_class)
        if self.target_idx is None:
            raise ValueError(f"Unknown target class: {target_class}. Must be one of {CLASS_NAMES}")

        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.logger = logger

        # Statistics
        self.original_positive_count = 0
        self.synthetic_count = 0

    def apply(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply feature-space SMOTE to generate synthetic embeddings.

        Args:
            embeddings: Original embeddings (N, D)
            labels: Original multi-label matrix (N, 14)

        Returns:
            Tuple of:
                - synthetic_embeddings: Generated embeddings
                - synthetic_labels: Labels for synthetic samples (only target class = 1)
                - combined_embeddings: Original + synthetic
                - combined_labels: Original + synthetic labels
        """
        # Get indices of positive samples for target class
        positive_mask = labels[:, self.target_idx] == 1
        self.original_positive_count = positive_mask.sum()

        if self.logger:
            self.logger.info(f"   Target class: {self.target_class} (index {self.target_idx})")
            self.logger.info(f"   Original positive samples: {self.original_positive_count}")
            self.logger.info(f"   Sampling ratio: {self.sampling_ratio}x")

        # Check if we have enough samples for SMOTE
        if self.original_positive_count < self.k_neighbors + 1:
            if self.logger:
                self.logger.warning(f"   Not enough positive samples ({self.original_positive_count}) for k_neighbors={self.k_neighbors}")
                self.logger.warning(f"   Skipping SMOTE for {self.target_class}")
            return np.array([]), np.array([]), embeddings, labels

        # Create binary labels for SMOTE (1 = target class positive, 0 = otherwise)
        binary_labels = positive_mask.astype(int)

        # Calculate desired count
        negative_count = (~positive_mask).sum()
        desired_positive = int(self.original_positive_count * self.sampling_ratio)

        # Don't oversample to full balance - just multiply by ratio
        # The sampling_strategy tells SMOTE the desired final count for minority class
        smote = SMOTE(
            sampling_strategy={1: desired_positive},
            k_neighbors=min(self.k_neighbors, self.original_positive_count - 1),
            random_state=self.seed
        )

        try:
            resampled_embeddings, resampled_binary = smote.fit_resample(embeddings, binary_labels)
        except Exception as e:
            if self.logger:
                self.logger.error(f"   SMOTE failed: {e}")
            return np.array([]), np.array([]), embeddings, labels

        # Extract only the synthetic samples (new ones added by SMOTE)
        # SMOTE appends synthetic samples at the end
        n_original = embeddings.shape[0]
        synthetic_embeddings = resampled_embeddings[n_original:]
        self.synthetic_count = synthetic_embeddings.shape[0]

        if self.logger:
            if self.synthetic_count == 0:
                self.logger.warning("=" * 60)
                self.logger.warning(f"   WARNING: SMOTE generated 0 synthetic embeddings!")
                self.logger.warning(f"   This may indicate an issue with the SMOTE configuration.")
                self.logger.warning(f"   Training will continue with original data only.")
                self.logger.warning("=" * 60)
            else:
                self.logger.info(f"   Generated {self.synthetic_count} synthetic embeddings")

        # Create labels for synthetic samples
        # Only target class = 1, all others = 0
        synthetic_labels = np.zeros((self.synthetic_count, labels.shape[1]), dtype=np.float32)
        synthetic_labels[:, self.target_idx] = 1.0

        # Combine original and synthetic
        combined_embeddings = np.vstack([embeddings, synthetic_embeddings])
        combined_labels = np.vstack([labels, synthetic_labels])

        if self.logger:
            self.logger.info(f"   Total training samples: {combined_embeddings.shape[0]} ({n_original} original + {self.synthetic_count} synthetic)")
            new_positive_count = (combined_labels[:, self.target_idx] == 1).sum()
            self.logger.info(f"   {self.target_class} positives: {self.original_positive_count} -> {new_positive_count}")
            # Mandatory summary log line
            self.logger.info("")
            self.logger.info(f"   SMOTE({self.target_class}): positives before={self.original_positive_count}, generated={self.synthetic_count}, total after={combined_embeddings.shape[0]}")

        return synthetic_embeddings, synthetic_labels, combined_embeddings, combined_labels


class EmbeddingHeadTrainer:
    """
    Train only the classification head on embeddings (real + synthetic).

    This allows training on pre-computed embeddings without re-running the backbone.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the head trainer.

        Args:
            model: Full model with frozen backbone
            device: torch device
            learning_rate: Learning rate for head training
            weight_decay: Weight decay for optimizer
            logger: Optional logger
        """
        self.model = model
        self.device = device
        self.logger = logger

        # Ensure backbone is frozen
        self.model.freeze_backbone()

        # Get only head parameters
        head_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                head_params.append(param)
                if logger:
                    logger.info(f"   Trainable: {name}")

        # Optimizer for head only
        self.optimizer = torch.optim.Adam(head_params, lr=learning_rate, weight_decay=weight_decay)

        # Loss function (neutral BCE, no class weights)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch_on_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> float:
        """
        Train one epoch on pre-computed embeddings.

        Args:
            embeddings: Feature embeddings (N, D)
            labels: Multi-label matrix (N, 14)
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            Average loss for the epoch
        """
        self.model.train()

        # Create TensorDataset
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        dataset = TensorDataset(emb_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        total_loss = 0.0
        for batch_emb, batch_labels in dataloader:
            batch_emb = batch_emb.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward through head only
            outputs = self._forward_head(batch_emb)

            loss = self.criterion(outputs, batch_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _forward_head(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head only.

        Args:
            embeddings: Pre-computed embeddings (B, D)

        Returns:
            Logits (B, num_classes)
        """
        if self.model.head_type == 'mlp_multi':
            # Multi-layer MLP head (Sequential)
            out = self.model.mlp_head(embeddings)
        elif self.model.head_type == 'mlp':
            # Single-layer MLP head
            out = self.model.head_fc1(embeddings)
            out = self.model.head_activation(out)
            out = self.model.head_dropout(out)
            out = self.model.final_fc(out)
        else:
            # Linear head
            out = self.model.final_fc(embeddings)

        return out


def compute_hernia_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> Dict[str, float]:
    """
    Compute detailed metrics for Hernia class.

    Args:
        labels: Ground truth labels (N, 14)
        predictions: Binary predictions (N, 14)
        probabilities: Prediction probabilities (N, 14)

    Returns:
        Dict with TP, FP, FN, TN, Precision, Recall, F1, AUC for Hernia
    """
    from sklearn.metrics import roc_auc_score

    hernia_idx = CLASS_TO_IDX['Hernia']

    y_true = labels[:, hernia_idx]
    y_pred = predictions[:, hernia_idx]
    y_prob = probabilities[:, hernia_idx]

    # Confusion matrix elements
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    return {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'AUC': float(auc)
    }


def cache_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cache_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Cache embeddings to disk for reuse.

    Args:
        embeddings: Feature embeddings
        labels: Labels
        cache_path: Path to save cache
        metadata: Optional metadata to save
    """
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    np.savez(
        cache_path,
        embeddings=embeddings,
        labels=labels,
        metadata=json.dumps(metadata or {})
    )


def load_cached_embeddings(cache_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load cached embeddings from disk.

    Args:
        cache_path: Path to cache file

    Returns:
        Tuple of (embeddings, labels, metadata)
    """
    data = np.load(cache_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    metadata = json.loads(str(data['metadata']))
    return embeddings, labels, metadata


def get_embedding_cache_path(iteration: int, split: str = 'train') -> str:
    """
    Get the cache path for embeddings.

    Args:
        iteration: Backbone source iteration
        split: 'train' or 'val'

    Returns:
        Cache file path
    """
    cache_dir = f"auto_improvement_runs/embedding_cache"
    return f"{cache_dir}/iter{iteration:03d}_{split}_embeddings.npz"
