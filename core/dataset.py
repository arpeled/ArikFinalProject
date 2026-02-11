import random
import logging

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import models, transforms

import torch.nn as nn
from torchvision import models

rare_classes = ['Hernia', 'Fibrosis', 'Emphysema', 'Pneumonia']
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Legacy rare transform (for backward compatibility)
rare_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    base_transform
])


# ==============================================================================
# HERNIA-SPECIFIC OVERSAMPLING AND AUGMENTATION
# ==============================================================================

class AddGaussianNoise:
    """Picklable Gaussian noise transform for multiprocessing DataLoader."""

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self):
        return f'{self.__class__.__name__}(std={self.std})'


class ClampTensor:
    """Picklable clamp transform for multiprocessing DataLoader."""

    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return torch.clamp(tensor, self.min_val, self.max_val)

    def __repr__(self):
        return f'{self.__class__.__name__}(min={self.min_val}, max={self.max_val})'


class HerniaAugmentTransform:
    """
    Hernia-specific augmentation transform.
    Applied ONLY to Hernia-positive samples when enabled.

    Safe augmentations for chest X-ray:
    - RandomResizedCrop (scale 0.9-1.0) - mild zoom
    - Small rotation (±5 degrees)
    - Mild brightness/contrast jitter
    - Mild Gaussian noise
    - Small random erasing (optional)
    """

    def __init__(self, base_size=(224, 224), seed=None):
        self.base_size = base_size
        self.seed = seed

        # Hernia-specific augmentation pipeline
        # Note: Using picklable classes instead of lambdas for multiprocessing compatibility
        self.hernia_augment = transforms.Compose([
            # Mild random crop (90-100% of image)
            transforms.RandomResizedCrop(
                base_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05)
            ),
            # Small rotation (±5 degrees - safe for chest X-ray)
            transforms.RandomRotation(5),
            # Mild brightness/contrast jitter
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            ),
            # Convert to tensor
            transforms.ToTensor(),
            # Add mild Gaussian noise (picklable class)
            AddGaussianNoise(std=0.01),
            # Clamp values to valid range (picklable class)
            ClampTensor(min_val=0, max_val=1),
            # Normalize
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Optional: small random erasing (helps with occlusion robustness)
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.02), ratio=(0.5, 2.0))
        ])

        # Standard transform (no augmentation)
        self.standard_transform = transforms.Compose([
            transforms.Resize(base_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, image, is_hernia_positive=False):
        """
        Apply transform based on whether sample is Hernia-positive.

        Args:
            image: PIL Image
            is_hernia_positive: bool, whether sample has Hernia label

        Returns:
            Transformed tensor
        """
        if is_hernia_positive:
            return self.hernia_augment(image)
        else:
            return self.standard_transform(image)


def create_hernia_oversampler(dataset, oversample_factor=10, hernia_idx=3, seed=42, logger=None):
    """
    Create a WeightedRandomSampler that oversamples Hernia-positive samples.

    Args:
        dataset: ChestXRayDataset instance
        oversample_factor: How many times more likely to sample Hernia-positive (default 10)
        hernia_idx: Index of Hernia in label columns (default 3)
        seed: Random seed for reproducibility
        logger: Optional logger instance

    Returns:
        WeightedRandomSampler instance, hernia_positive_indices list
    """
    if logger is None:
        logger = logging.getLogger('hernia_sampler')

    # Get labels array
    labels = dataset.labels if hasattr(dataset, 'labels') else dataset.data[dataset.label_columns].values

    # Find Hernia-positive indices
    hernia_positive_mask = labels[:, hernia_idx] == 1
    hernia_positive_indices = np.where(hernia_positive_mask)[0].tolist()
    hernia_negative_indices = np.where(~hernia_positive_mask)[0].tolist()

    num_hernia_positive = len(hernia_positive_indices)
    num_hernia_negative = len(hernia_negative_indices)
    total_samples = len(labels)

    logger.info("=" * 60)
    logger.info("HERNIA OVERSAMPLING ENABLED")
    logger.info("=" * 60)
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Hernia-positive: {num_hernia_positive} ({100*num_hernia_positive/total_samples:.3f}%)")
    logger.info(f"   Hernia-negative: {num_hernia_negative}")
    logger.info(f"   Oversample factor: {oversample_factor}x")

    # Create sample weights
    # Hernia-positive samples get weight = oversample_factor
    # Hernia-negative samples get weight = 1
    sample_weights = np.ones(total_samples, dtype=np.float64)
    sample_weights[hernia_positive_mask] = float(oversample_factor)

    # Calculate effective sampling rate
    # Expected Hernia samples per epoch = (num_hernia_positive * factor) / (num_neg + num_pos * factor) * total
    effective_hernia_weight = num_hernia_positive * oversample_factor
    total_weight = num_hernia_negative + effective_hernia_weight
    expected_hernia_per_epoch = (effective_hernia_weight / total_weight) * total_samples

    logger.info(f"   Expected Hernia samples per epoch: ~{expected_hernia_per_epoch:.0f}")
    logger.info(f"   Effective Hernia prevalence: {100*effective_hernia_weight/total_weight:.2f}%")
    logger.info("=" * 60)

    # Set seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create weighted sampler
    # num_samples = total_samples means we sample the same number as original dataset
    # replacement=True is required for weighted sampling
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True,
        generator=generator
    )

    return sampler, hernia_positive_indices

class ChestXRayDataset(Dataset):
    """
    Chest X-Ray dataset with optional Hernia-specific augmentation.

    Args:
        dataset: DataFrame or None (if None, reads from csv_file)
        csv_file: Path to CSV file
        root_dir: Directory containing images
        transform: Standard transform to apply
        use_additional_features: Whether to include patient metadata
        hernia_augment_enabled: Enable Hernia-specific augmentation (default False)
        hernia_augment_transform: HerniaAugmentTransform instance (optional)
    """

    # Index of Hernia in label_columns
    HERNIA_IDX = 3

    def __init__(self, dataset, csv_file, root_dir, transform=None, use_additional_features=False,
                 hernia_augment_enabled=False, hernia_augment_transform=None):
        if dataset is not None:
            self.data = dataset.copy() if hasattr(dataset, 'copy') else dataset
        else:
            self.data = pd.read_csv(csv_file)

        # Reset index to ensure iloc works correctly
        self.data = self.data.reset_index(drop=True)

        # Drop unnecessary columns
        for col in ["PatientId", "Patient ID", "FilePath", "No Finding"]:
            if col in self.data.columns:
                self.data.drop(columns=[col], inplace=True)

        # Define label columns explicitly
        self.label_columns = [
            'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
            'Pneumothorax',
            'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
        ]

        # Add class_names property for compatibility
        self.class_names = self.label_columns

        # Ensure all label columns are numeric
        self.data[self.label_columns] = self.data[self.label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Create labels property for compatibility
        self.labels = self.data[self.label_columns].values

        # Define additional feature columns (if any)
        self.additional_columns = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']

        self.root_dir = root_dir
        self.transform = transform
        self.use_additional_features = use_additional_features

        # Hernia-specific augmentation
        self.hernia_augment_enabled = hernia_augment_enabled
        self.hernia_augment_transform = hernia_augment_transform

        # Pre-compute Hernia-positive indices for efficiency
        if self.hernia_augment_enabled:
            self._hernia_positive_set = set(
                np.where(self.labels[:, self.HERNIA_IDX] == 1)[0].tolist()
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.root_dir}/{self.data.iloc[idx, 0]}"
        try:
            # Get current labels
            current_labels = self.labels[idx]

            # Check if this is a Hernia-positive sample
            is_hernia_positive = (
                self.hernia_augment_enabled and
                idx in self._hernia_positive_set
            )

            # Attempt to open the image
            image = Image.open(img_name).convert("RGB")

            # Apply appropriate transform
            if is_hernia_positive and self.hernia_augment_transform is not None:
                # Apply Hernia-specific augmentation
                image = self.hernia_augment_transform(image, is_hernia_positive=True)
            elif self.transform:
                # Check if current sample has rare classes (legacy behavior)
                is_rare = any(
                    self.class_names[i] in rare_classes and current_labels[i] == 1
                    for i in range(len(self.class_names))
                )
                # Apply standard transform (legacy rare transform disabled by default)
                if is_rare and random.random() < 0.5 and hasattr(self, 'use_rare_transform') and self.use_rare_transform:
                    image = rare_transform(image)
                else:
                    image = self.transform(image)

        except FileNotFoundError:
            print(f"Warning: File not found: {img_name}")
            if self.use_additional_features:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        # Extract labels
        labels = torch.tensor(current_labels.astype(float), dtype=torch.float32)

        if self.use_additional_features:
            # Extract additional features
            follow_up = torch.tensor(self.data.iloc[idx]["Follow-up #"], dtype=torch.float32)
            patient_age = torch.tensor(self.data.iloc[idx]["Patient Age"], dtype=torch.float32)
            patient_gender = 1.0 if self.data.iloc[idx]["Patient Gender"] == "M" else 0.0
            view_position = 1.0 if self.data.iloc[idx]["View Position"] == "PA" else 0.0
            additional_features = torch.tensor([follow_up, patient_age, patient_gender, view_position],
                                               dtype=torch.float32)
            return image, additional_features, labels

        return image, labels


# Define a modified DenseNet model
class ModifiedDenseNet(nn.Module):
    def __init__(self, num_classes=14, use_additional_features=False):
        super(ModifiedDenseNet, self).__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
        self.base_model = model
        # self.base_model = models.densenet121(pretrained=True)
        self.base_model.classifier = nn.Identity()  # Remove DenseNet's classifier
        self.use_additional_features = use_additional_features

        if self.use_additional_features:
            self.additional_fc = nn.Linear(4, 128)  # Four additional features
            self.final_fc = nn.Linear(1024 + 128, num_classes)  # Combine DenseNet and additional features
        else:
            self.final_fc = nn.Linear(1024, num_classes)

    def forward(self, x, additional_features=None):
        base_out = self.base_model(x)

        if self.use_additional_features:
            additional_out = nn.ReLU()(self.additional_fc(additional_features))
            combined = torch.cat([base_out, additional_out], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(base_out)

        return out


class ModifiedDenseNetWithDropOut(nn.Module):
    def __init__(self, num_classes=14, use_additional_features=False, head_config=None):
        super(ModifiedDenseNetWithDropOut, self).__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
        self.base_model = model
        self.base_model.classifier = nn.Identity()  # Remove DenseNet's classifier
        self.dropout = nn.Dropout(0.3)

        self.use_additional_features = use_additional_features
        in_features = 1024 + 128 if self.use_additional_features else 1024

        if self.use_additional_features:
            self.additional_fc = nn.Linear(4, 128)

        # Config-driven head selection
        self.head_type = 'linear'
        if head_config and head_config.get('type') == 'mlp':
            self.head_type = 'mlp'
            hidden = head_config.get('hidden_features', 512)
            head_dropout = head_config.get('dropout', 0.2)
            activation = head_config.get('activation', 'relu')

            self.head_fc1 = nn.Linear(in_features, hidden)
            self.head_activation = nn.ReLU() if activation == 'relu' else nn.GELU()
            self.head_dropout = nn.Dropout(head_dropout)
            self.final_fc = nn.Linear(hidden, num_classes)
        else:
            self.final_fc = nn.Linear(in_features, num_classes)

    def forward(self, x, additional_features=None):
        base_out = self.base_model(x)
        base_out = self.dropout(base_out)

        if self.use_additional_features:
            additional_out = nn.ReLU()(self.additional_fc(additional_features))
            combined = torch.cat([base_out, additional_out], dim=1)
        else:
            combined = base_out

        if self.head_type == 'mlp':
            out = self.head_fc1(combined)
            out = self.head_activation(out)
            out = self.head_dropout(out)
            out = self.final_fc(out)
        else:
            out = self.final_fc(combined)

        return out

    def freeze_backbone(self):
        """Freeze backbone - only head trainable."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.dropout.parameters():
            param.requires_grad = False
        if self.use_additional_features:
            for param in self.additional_fc.parameters():
                param.requires_grad = False

    def load_backbone_from_checkpoint(self, checkpoint_path, device):
        """Load backbone weights from checkpoint, skip head layers."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        new_state_dict = {}
        for key, value in checkpoint.items():
            if 'final_fc' in key or 'head_' in key:
                continue
            new_state_dict[key] = value
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        return missing, unexpected