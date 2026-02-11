import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import ChestXRayDataset, ModifiedDenseNet
import time

# Parameters
num_classes = 14
batch_size = 64
learning_rate = 0.001
num_epochs = 5
use_additional_features = True  # Toggle for using additional features

# Device configuration






# Load data
# Load data
# Load data
csv_file = './ChestX-ray14/train_data.csv'  # Use the fixed CSV
root_dir = './ChestX-ray14/images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = ChestXRayDataset(csv_file, root_dir, transform, use_additional_features=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8, pin_memory=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialize model
    model = ModifiedDenseNet(num_classes=num_classes, use_additional_features=use_additional_features).to(device)

# Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
    print("Starting training...")
    # Training loop
    for epoch in range(num_epochs):
        epoch_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"Starting Epoch [{epoch + 1}/{num_epochs}]")

        for batch_idx, batch in enumerate(dataloader, start=1):
            if use_additional_features:
                images, additional_features, labels = batch

                # Filter out empty tensors
                valid_indices = [i for i in range(len(images)) if images[i].numel() > 0]
                images = images[valid_indices]
                additional_features = additional_features[valid_indices]
                labels = labels[valid_indices]

                if len(images) == 0:  # Skip batch if all are invalid
                    continue

                images = images.to(device)
                additional_features = additional_features.to(device)
                labels = labels.to(device)
                outputs = model(images, additional_features)
            else:
                images, labels = batch

                # Filter out empty tensors
                valid_indices = [i for i in range(len(images)) if images[i].numel() > 0]
                images = images[valid_indices]
                labels = labels[valid_indices]

                if len(images) == 0:  # Skip batch if all are invalid
                    continue

                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0 or batch_idx == len(dataloader):  # Print every 10 batches or the last batch
                current_time = time.time()
                elapsed_time = current_time - epoch_time
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f} seconds")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Complete. Average Loss: {epoch_loss:.4f}\n")
        return model;

if __name__ == "__main__":
    # Ensure your main training code is inside this block
    start_time = time.time()
    model  = train_model()
    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    torch.save(model.state_dict(), 'model_with_features_v2_batch64_epoch5.pth' if use_additional_features else 'model_without_features.pth')
    print("Model saved successfully.")

