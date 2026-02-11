import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from dataset import ChestXRayDataset, ModifiedDenseNet
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# Parameters
num_classes = 14
batch_size = 64
learning_rate = 0.0001
num_epochs = 50
use_additional_features = True  # Toggle for using additional features

# Load data
csv_file = './ChestX-ray14/train_data.csv'  # Use the fixed CSV
root_dir = './ChestX-ray14/images224'

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def weighted_bce_with_logits_loss(output, target, weights):
    bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    weighted_loss = bce_loss * weights
    return weighted_loss.mean()




def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = ChestXRayDataset(csv_file, root_dir, transform, use_additional_features=True)

    train_size = int(0.8 * len(dataset))  # 80% לאימון
    val_size = len(dataset) - train_size  # 20% ל-Validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    model = ModifiedDenseNet(num_classes=num_classes, use_additional_features=use_additional_features).to(device)
    class_weights = torch.tensor([
        14.22, 21.26, 24.46, 44.74, 48.90, 53.34, 60.60,
        83.68, 102.04, 112.36, 122.70, 168.06, 198.02, 1250.00
    ], dtype=torch.float32).to(device)

    # Define a custom weighted BCEWithLogitsLoss

    # During training
    criterion = lambda output, target: weighted_bce_with_logits_loss(output, target, class_weights)

    # Loss and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    loss_history = []
    # Training loop
    print("Starting training... at: ", time.time())

    for epoch in range(num_epochs):
        epoch_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"Starting Epoch [{epoch + 1}/{num_epochs}]")

        for batch_idx, batch in enumerate(dataloader, start=1):
            if use_additional_features:
                images, additional_features, labels = batch
                images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(
                    device)
                outputs = model(images, additional_features)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0 or batch_idx == len(dataloader):  # Print every 100 batches or the last batch
                current_time = time.time()
                elapsed_time = current_time - epoch_time
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f} seconds")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Complete. Average Loss: {epoch_loss:.4f}")

        # Validation loop (if you have validation data)
        val_loss = 0.0
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if use_additional_features:
                    images, additional_features, labels = batch
                    images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(
                        device)
                    outputs = model(images, additional_features)
                else:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        loss_history.append(val_loss)
        # Step the scheduler with the validation loss
        scheduler.step(val_loss)
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        loss_history.append(val_loss)

        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
    print("Training complete.")
    print("Loss history: ", loss_history)
    return model


if __name__ == "__main__":
    # Ensure your main training code is inside this block
    start_time = time.time()
    model = train_model()
    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    torch.save(model.state_dict(),
               'model_with_features_v2_batch64_epoch50_scheduler_lr0001_images224_weighted_bce_with_logits_loss.pth' if use_additional_features else 'model_without_features.pth')
    print("Model saved successfully.")
