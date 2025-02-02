import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ChestXRayDataset  # Ensure this matches your dataset class
from torchvision import models
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np

# Initialize parameters
num_classes = 14
batch_size = 16

# Build PyTorch DenseNet121 model
pytorch_model = models.densenet121(weights=None)
pytorch_model.classifier = nn.Linear(pytorch_model.classifier.in_features, num_classes)

# Load the trained model
model_save_path = "trained_model.pth"
pytorch_model.load_state_dict(torch.load(model_save_path))
print(f"Model loaded successfully from '{model_save_path}'")

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
# Load test data
test_csv = './ChestX-ray14/test.csv'
root_dir = './ChestX-ray14/images'
df = pd.read_csv(test_csv)

# Extract label names from the CSV (excluding the image path column)
label_names = df.columns[1:1+num_classes]  # Ensure only num_classes columns are used

# Debugging: Print label names and confirm count
print(f"Extracted label names ({len(label_names)}): {label_names.tolist()}")

test_dataset = ChestXRayDataset(csv_file=test_csv, root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Test dataset loaded. Number of samples: {len(test_dataset)}")

# Evaluation
pytorch_model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = pytorch_model(inputs)
        predicted_probs = torch.sigmoid(outputs)
        predictions = (predicted_probs > 0.5).int()

        # Append predictions and labels
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

        # Debugging first 2 batches
        if batch_idx < 2:
            print(f"Batch {batch_idx + 1}, True Labels:\n{labels.cpu().numpy()}")
            print(f"Batch {batch_idx + 1}, Predictions:\n{predictions.cpu().numpy()}")

# Concatenate results
all_labels = np.vstack(all_labels)
all_predictions = np.vstack(all_predictions)

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=label_names, zero_division=1))

# Calculate F1 Score and Accuracy
f1_scores = f1_score(all_labels, all_predictions, average=None)
overall_accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"F1 Scores per class: {f1_scores}")
