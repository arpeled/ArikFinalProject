import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms
from dataset import ChestXRayDataset, ModifiedDenseNet, ModifiedDenseNetWithDropOut
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.model_selection import GroupShuffleSplit
import os
import datetime

# Parameters
num_classes = 14
batch_size = 64
learning_rate = 0.0001
num_epochs = 50
use_additional_features = True
num_workers = 8
model_name = "ModifiedDenseNetWithDropOut"
augmentation_description = "RandomHorizontalFlip, RandomRotation(10) + rare class oversampling"

# Load data
csv_file = './ChestX-ray14/train_data.csv'
root_dir = './ChestX-ray14/images224'

# Transforms

RARE_CLASS_THRESHOLD = 2000  # פחות מ-2000 מופעים נחשב נדיר
RARE_CLASS_AUG_MULTIPLIER = 2  # כפול דוגמאות לקלאסים נדירים

# הגדרת קלאסים נדירים (תעדכן לפי הסטטיסטיקה שלך)

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def weighted_bce_with_logits_loss(output, target, weights):
    bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    weighted_loss = bce_loss * weights
    return weighted_loss.mean()

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
        print(f"Epoch: {self.epoch_count}, Validation Loss: {val_loss:.4f}")
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

def get_rare_class_augmentation(train_df, label_names, log_file):
    thresholds = {
        'Hernia': 0.003,
        'Pneumonia': 0.01,
        'Fibrosis': 0.012,
        'Edema': 0.016,
        'Emphysema': 0.018,
        'Cardiomegaly': 0.02,
        'Pleural_Thickening': 0.025,
        'Consolidation': 0.035,
        'Pneumothorax': 0.045
    }
    rare_datasets = []
    with open(log_file, 'a') as f:
        f.write("Rare class augmentations performed:\n")
        for label, threshold in thresholds.items():
            positive_samples = train_df[train_df[label] == 1]
            class_ratio = len(positive_samples) / len(train_df)
            if class_ratio < threshold:
                repeat_factor = int(threshold / class_ratio)
                if repeat_factor > 1:
                    rare_aug_df = pd.concat([positive_samples] * repeat_factor, ignore_index=True)
                    rare_dataset = ChestXRayDataset(dataset=rare_aug_df, csv_file=None, root_dir=root_dir, transform=rare_transform, use_additional_features=use_additional_features)
                    rare_datasets.append(rare_dataset)
                    f.write(f"{label}: repeated x{repeat_factor} ({len(rare_aug_df)} samples added)\n")
    return rare_datasets

def train_model(log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"✅ Using {num_gpus} GPU(s) on {device}")

    df = pd.read_csv(csv_file)
    if "Patient ID" not in df.columns:
        raise ValueError("עמודת 'patient_id' לא קיימת ב-CSV. צריך לוודא שהיא קיימת.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Rare data augmentation
    rare_datasets = get_rare_class_augmentation(train_df, df.columns[4:], log_file)

    train_dataset = ChestXRayDataset(dataset=train_df, csv_file=None, root_dir=root_dir, transform=base_transform, use_additional_features=True)
    if rare_datasets:
        train_dataset = ConcatDataset([train_dataset] + rare_datasets)

    val_dataset = ChestXRayDataset(dataset=val_df, csv_file=None, root_dir=root_dir, transform=base_transform, use_additional_features=True)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"מספר הדוגמאות ב-Train: {len(train_dataset)}, מספר הדוגמאות ב-Validation: {len(val_dataset)}")

    model = ModifiedDenseNetWithDropOut(num_classes=num_classes, use_additional_features=use_additional_features).to(device)
    if torch.cuda.is_available() and num_gpus > 1:
        print("parallel gpu training")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    class_weights = torch.tensor([
        2.456140350877193, 2.709677419354839, 0.5118829981718466, 30.0, 0.34271725826193394, 1.1797752808988766,
        1.076923076923077, 0.5898876404494383, 1.2863705972434918, 2.0143884892086334, 4.7727272727272725,
        4.038461538461539, 2.95774647887324, 1.4608695652173913
    ], dtype=torch.float32).to(device)
    log_weights(log_file, class_weights)
    criterion = lambda output, target: weighted_bce_with_logits_loss(output, target, class_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-5)
    early_stopping = EarlyStopping()
    log_parameters(log_file, model, augmentation_description, device)

    loss_history = []
    print("Starting training... at: ", time.time())
    for epoch in range(num_epochs):
        epoch_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"Starting Epoch [{epoch + 1}/{num_epochs}]")

        for batch_idx, batch in enumerate(dataloader_train, start=1):
            batch_start_time = time.time()
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

            if batch_idx % 100 == 0 or batch_idx == len(dataloader_train):
                print(f"{batch_idx}/{len(dataloader_train)}: Loss={loss.item():.4f} - Time={time.time() - batch_start_time:.2f}s")

        epoch_loss = running_loss / len(dataloader_train)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Complete. Average Loss: {epoch_loss:.4f}")

        val_loss = 0.0
        model.eval()
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
        print(f"Validation Loss: {val_loss:.4f}")
        loss_history.append(val_loss)
        scheduler.step(val_loss)
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            log_early_stopping(log_file, epoch + 1)
            break
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr}")
        log_epoch_results(log_file, epoch + 1, val_loss, current_lr)

    print("Training complete.")
    print("Loss history: ", loss_history)
    log_loss_history(log_file, loss_history)
    return model

# (Log functions stay unchanged...)



def log_weights(log_file, class_weights ):
    """Logs the initial parameters of the run."""
    with open(log_file, 'w') as f:  # 'w' to create/overwrite the file
        f.write(f"Class Weights used: {class_weights}\n")
        f.write("-" * 30 + "\n")

def log_parameters(log_file, model, augmentation_description, device):
    """Logs the initial parameters of the run."""
    with open(log_file, 'w') as f:  # 'w' to create/overwrite the file
        f.write(f"Run started at: {datetime.datetime.now()}\n")
        f.write(f"Using {device} GPU(s) ")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Architecture: {model}\n")
        f.write(f"Data Augmentation: {augmentation_description}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Use Additional Features: {use_additional_features}\n")
        f.write(f"Number of Workers: {num_workers}\n")
        f.write("-" * 30 + "\n")


def log_epoch_results(log_file, epoch, val_loss, current_lr):
    """Logs the results of each epoch."""
    with open(log_file, 'a') as f:  # 'a' to append to the file
        f.write(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, Current LR: {current_lr:.6f}\n")


def log_early_stopping(log_file, epoch):
    """Logs when early stopping occurred."""
    with open(log_file, 'a') as f:
        f.write(f"Early stopping triggered at epoch {epoch}\n")
        f.write("-" * 30 + "\n")


def log_loss_history(log_file, loss_history):
    """Logs the complete loss history."""
    with open(log_file, 'a') as f:
        f.write("Loss History:\n")
        for i, loss in enumerate(loss_history):
            f.write(f"Epoch {i + 1}: {loss:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Run ended at: {datetime.datetime.now()}\n")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"run_log_{timestamp}.txt"
    start_time = time.time()
    model = train_model(log_file)
    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    torch.save(model.state_dict(), f'run_info_model_{timestamp}.pth')
    print("Model saved successfully.")
