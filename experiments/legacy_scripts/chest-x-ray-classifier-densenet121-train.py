import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset import ChestXRayDataset  # Ensure this is correct
from torchvision import models
import time

# Initialize parameters
num_classes = 14
batch_size = 16
learning_rate = 0.001
num_epochs = 4

# Build PyTorch DenseNet121 model
pytorch_model = models.densenet121(weights=None)
pytorch_model.classifier = nn.Linear(pytorch_model.classifier.in_features, num_classes)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
pytorch_model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
optimizer = optim.Adam(pytorch_model.parameters(), lr=learning_rate)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_csv = './ChestX-ray14/train-small.csv'
root_dir = './ChestX-ray14/images'  # Update this to your image folder path
print("Loading training dataset...")
try:
    train_dataset = ChestXRayDataset(csv_file=train_csv, root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training dataset loaded. Number of samples: {len(train_dataset)}")
except Exception as e:
    print(f"Error loading training dataset: {e}")
    exit()

# Training loop
print("Starting training...")
start_time = time.time()

pytorch_model.train()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = pytorch_model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Track predictions for accuracy (optional)
        predictions = torch.sigmoid(outputs) > 0.5
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.numel()

        # Debug: Print loss for first few batches
        if batch_idx < 3:  # Print for the first 3 batches only
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    print(f"Time taken for epoch {epoch + 1}: {time.time() - epoch_start_time:.2f} seconds")

# Save the trained model
model_save_path = "trained_model.pth"
torch.save(pytorch_model.state_dict(), model_save_path)
print(f"Training complete. Model saved as '{model_save_path}'.")
print(f"Total training time: {time.time() - start_time:.2f} seconds")


# base_model = DenseNet121(weights='./ChestX-ray14/densenet.hdf5', include_top=False)
# x = base_model.output
# add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)
# and a logistic layer
# predictions = Dense(len(labels), activation="sigmoid")(x)
# model = Model(inputs=base_model.input, outputs=predictions)
# model.load_weights("./ChestX-ray14/pretrained_model.h5")
