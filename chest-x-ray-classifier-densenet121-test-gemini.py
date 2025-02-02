import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ChestXRayDataset  # Make sure this is correct
from torchvision import models
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd

# Initialize parameters
num_classes = 14
batch_size = 16  # Adjust as needed

# Build PyTorch DenseNet121 model
pytorch_model = models.densenet121(weights=None)
pytorch_model.classifier = nn.Linear(pytorch_model.classifier.in_features, num_classes)

# Load the trained model
model_save_path = "trained_model.pth"  # Path where the trained model is saved
try:
    pytorch_model.load_state_dict(torch.load(model_save_path))
    print(f"Model loaded successfully from '{model_save_path}'")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_save_path}'")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
pytorch_model.to(device)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
test_csv = './ChestX-ray14/test.csv'  # Update this to your test CSV path
root_dir = './ChestX-ray14/images'  # Update this to your image folder path
# Get label names from CSV
df = pd.read_csv(test_csv)
label_names = df.columns[1:]  # Assuming the first column is the image path

print("Loading test dataset...")
try:
    test_dataset = ChestXRayDataset(csv_file=test_csv, root_dir=root_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data
    print(f"Test dataset loaded. Number of samples: {len(test_dataset)}")
except Exception as e:
    print(f"Error loading test dataset: {e}")
    exit()

# Evaluation loop
print("Starting evaluation...")
pytorch_model.eval()  # Set the model to evaluation mode

all_labels = []
all_predictions = []
all_predicted_probs = []  # Initialize list to store predicted probabilities

with torch.no_grad():  # No need to calculate gradients during evaluation
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = pytorch_model(inputs)
        loss = criterion(outputs, labels)

        # Track loss
        running_loss += loss.item()

        # Correctly apply sigmoid and threshold for multi-label predictions
        predicted_probs = torch.sigmoid(outputs)  # Get probabilities  # Convert to 0/1
        predictions = (predicted_probs > 0.5).int()

        # Track predictions for accuracy (this might need adjustment for multi-label)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.numel()

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        # Store probabilities
        all_predicted_probs.append(predicted_probs.cpu().numpy())

        if batch_idx < 3:  # Print for the first 3 batches only
            print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Calculate metrics (corrected)
all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)
all_predicted_probs = np.concatenate(all_predicted_probs)  # Get predicted probabilities

# Print shapes
print("Shape of all_labels:", all_labels.shape)
print("Shape of all_predicted_probs:", all_predicted_probs.shape)

# Summary report
print("Summary Report:")
print("-" * 90)
# Table header
print("Label|AUC|Threshold|Sensitivity|Specificity|Accuracy|Precision|Recall|F1-score")
print("-" * 90)
for i in range(all_labels.shape[1]):
    # Calculate metrics for the current label
    auc = roc_auc_score(all_labels[:, i], all_predicted_probs[:, i])
    fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_predicted_probs[:, i])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Apply optimal threshold to get binary predictions
    binary_predictions = (all_predicted_probs[:, i] > optimal_threshold).astype(int)

    # Calculate confusion matrix using binary predictions
    tn, fp, fn, tp = confusion_matrix(all_labels[:, i], binary_predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Handle zero division
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Handle zero division

    # Use original predictions for accuracy
    accuracy = accuracy_score(all_labels[:, i], all_predictions[:, i])

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Handle zero division
    recall = sensitivity  # Recall is the same as sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Print the metrics
    print(f"{label_names[i+1]}|{auc:.4f}|{optimal_threshold:.4f}|{sensitivity:.4f}|"
          f"{specificity:.4f}|{accuracy:.4f}|{precision:.4f}|{recall:.4f}|{f1:.4f}")

print("-" * 90)

print("Evaluation complete.")

print('*'*100)
print("evaluation by chatgpt code")
# Summary report
print("Summary Report:")
print("-" * 90)
# Table header
print("Label|AUC|Optimal Threshold|Sensitivity|Specificity|Accuracy|Precision|Recall|F1-score")
print("-" * 90)

for i in range(all_labels.shape[1]):
    # Calculate metrics for the current label
    auc = roc_auc_score(all_labels[:, i], all_predicted_probs[:, i])
    fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_predicted_probs[:, i])
    optimal_idx = np.argmax(tpr - fpr)  # Maximize TPR - FPR
    optimal_threshold = thresholds[optimal_idx]

    # Apply optimal threshold to calculate metrics
    binary_predictions = (all_predicted_probs[:, i] > optimal_threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels[:, i], binary_predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity  # Recall is the same as sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate accuracy
    accuracy = accuracy_score(all_labels[:, i], binary_predictions)

    # Print metrics for the current label
    print(f"{label_names[i+1]}|{auc:.4f}|{optimal_threshold:.4f}|{sensitivity:.4f}|"
          f"{specificity:.4f}|{accuracy:.4f}|{precision:.4f}|{recall:.4f}|{f1:.4f}")

print("-" * 90)
print("Evaluation complete.")
