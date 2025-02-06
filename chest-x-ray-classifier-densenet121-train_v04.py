import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ChestXRayDataset, ModifiedDenseNetWithDropOut
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

# Parameters
num_classes = 14
batch_size = 64
learning_rate = 0.0001
num_epochs = 50
use_additional_features = True

# Data paths
csv_file = './ChestX-ray14/train_data.csv'
root_dir = './ChestX-ray14/images224'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"✅ Using {num_gpus} GPUs")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU found. Using CPU.")
    return device


def train_model():
    device = select_device()
    start_time = time.time()

    df = pd.read_csv(csv_file)
    if "Patient ID" not in df.columns:
        raise ValueError("Missing 'Patient ID' column in CSV.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = ChestXRayDataset(train_df, None, root_dir, transform, use_additional_features)
    val_dataset = ChestXRayDataset(val_df, None, root_dir, transform, use_additional_features)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ModifiedDenseNetWithDropOut(num_classes=num_classes, use_additional_features=use_additional_features)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    class_weights = torch.tensor([
        1.42, 2.12, 2.44, 4.47, 4.89, 5.33, 6.06,
        8.36, 10.20, 11.23, 12.27, 16.80, 19.80, 50.00
    ], dtype=torch.float32).to(device)

    criterion = lambda output, target: F.binary_cross_entropy_with_logits(output, target,
                                                                          reduction='none') * class_weights
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-5)

    print(f"🚀 Starting training at {time.strftime('%H:%M:%S', time.localtime())}")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader_train, start=1):
            images, labels = batch if not use_additional_features else batch[:2]
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) if not use_additional_features else model(images, batch[1].to(device))
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader_train)}], Loss: {loss.item():.4f}")

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in dataloader_val:
                images, labels = batch if not use_additional_features else batch[:2]
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) if not use_additional_features else model(images, batch[1].to(device))
                val_loss += criterion(outputs, labels).mean().item()
        val_loss /= len(dataloader_val)
        scheduler.step(val_loss)

        print(
            f"✅ Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - epoch_start:.2f} sec | Train Loss: {running_loss / len(dataloader_train):.4f} | Val Loss: {val_loss:.4f}")

    print(f"🎉 Training completed in {time.time() - start_time:.2f} seconds")
    return model


if __name__ == "__main__":
    # Ensure your main training code is inside this block
    start_time = time.time()
    model = train_model()
    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    model_save_path = '_v4_batch64_epoch50_config01.pth'
    torch.save(model.state_dict(),
               'model_with_features'+model_save_path if use_additional_features else 'model_without_features'+model_save_path)
    print("Model saved successfully.")
