"""
Modified training script for pipeline integration
Based on chest-x-ray-classifier-densenet121-train_v05.py
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from dataset import ChestXRayDataset, ModifiedDenseNetWithDropOut
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
import datetime
import logging

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape (num_classes,) or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits of shape (batch_size, num_classes)
        targets: same shape, 0 or 1
        """
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


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.01, warmup_epochs=25):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_loss = None
        self.counter = 0
        self.epoch_count = 0

    def __call__(self, val_loss):
        self.epoch_count += 1
        if self.epoch_count < self.warmup_epochs:
            return False
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_rare_class_augmentation(train_df, root_dir, rare_transform, use_additional_features):
    thresholds = {
        'Hernia': 0.003, 'Pneumonia': 0.01, 'Fibrosis': 0.012,
        'Edema': 0.016, 'Emphysema': 0.018, 'Cardiomegaly': 0.02,
        'Pleural_Thickening': 0.025, 'Consolidation': 0.035, 'Pneumothorax': 0.045
    }
    rare_datasets = []
    for label, threshold in thresholds.items():
        positive_samples = train_df[train_df[label] == 1]
        class_ratio = len(positive_samples) / len(train_df)
        if class_ratio < threshold:
            repeat_factor = int(threshold / class_ratio)
            if repeat_factor > 1:
                rare_aug_df = pd.concat([positive_samples] * repeat_factor, ignore_index=True)
                rare_dataset = ChestXRayDataset(dataset=rare_aug_df, csv_file=None, 
                                              root_dir=root_dir, transform=rare_transform, 
                                              use_additional_features=use_additional_features)
                rare_datasets.append(rare_dataset)
    return rare_datasets

def train_model_pipeline(timestamp, model_file, log_file, logger=None):
    """
    Training function that can be called from pipeline
    """
    # Use provided logger or create default
    if logger is None:
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
    # Training parameters - optimized for M4 Pro
    num_classes = 14
    batch_size = 64  # Increased for M4 Pro
    learning_rate = 0.001
    num_epochs = 20
    use_additional_features = True
    num_workers = 8  # Increased for M4 Pro's performance cores
    
    # Data paths - use small dataset for testing
    csv_file = './ChestX-ray14/train_data.csv'
    root_dir = './ChestX-ray14/images224'
    
    # Transforms
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    rare_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        base_transform
    ])
    
    # Device setup - optimized for M4 Pro
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"✅ Using Apple M4 Pro GPU (20 cores) with MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        logger.info(f"✅ Using {num_gpus} CUDA GPU(s)")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ Using CPU - GPU acceleration not available")
    
    # Load and split data
    df = pd.read_csv(csv_file)
    if "Patient ID" not in df.columns:
        raise ValueError("Patient ID column not found in CSV")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Create datasets with rare class augmentation
    rare_datasets = get_rare_class_augmentation(train_df, root_dir, rare_transform, use_additional_features)
    
    train_dataset = ChestXRayDataset(dataset=train_df, csv_file=None, root_dir=root_dir, 
                                    transform=base_transform, use_additional_features=True)
    if rare_datasets:
        train_dataset = ConcatDataset([train_dataset] + rare_datasets)
    
    val_dataset = ChestXRayDataset(dataset=val_df, csv_file=None, root_dir=root_dir, 
                                  transform=base_transform, use_additional_features=True)
    
    # Data loaders - optimized for M4 Pro
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True, 
                                 persistent_workers=True, prefetch_factor=2)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Model setup - optimized for M4 Pro
    model = ModifiedDenseNetWithDropOut(num_classes=num_classes, use_additional_features=use_additional_features)
    model = model.to(device)
    
    # Enable optimizations for MPS
    if device.type == 'mps':
        logger.info("🚀 MPS optimizations enabled for M4 Pro")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Using parallel GPU training")
        model = torch.nn.DataParallel(model)

    # Example: Replace with your dataset stats
    class_frequencies = [0.0342, 0.0310, 0.1641, 0.0028, 0.2451, 0.0712, 0.0780,
                         0.1424, 0.0653, 0.0417, 0.0176, 0.0208, 0.0284, 0.0575]

    min_freq = min(class_frequencies)
    alpha = [min_freq / f for f in class_frequencies]

    # Convert to tensor
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0, reduction='mean')
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, warmup_epochs=5)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader_train, start=1):
            if use_additional_features:
                images, additional_features, labels = batch
                images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(device)
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
            
            # Log progress every 50 batches
            if batch_idx % 300 == 0 or batch_idx == len(dataloader_train):
                logger.info(f"Batch {batch_idx}/{len(dataloader_train)} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader_train)
        
        # Validation
        logger.info("Running validation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader_val:
                if use_additional_features:
                    images, additional_features, labels = batch
                    images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(device)
                    outputs = model(images, additional_features)
                else:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(dataloader_val)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Learning rate scheduling and early stopping
        scheduler.step(val_loss)
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Save model
    logger.info("Training complete. Saving model...")
    torch.save(model.state_dict(), model_file)
    logger.info(f"Model saved as {model_file}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    return model