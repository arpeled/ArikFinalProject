import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, roc_curve
import pandas as pd
import numpy as np
from dataset import ChestXRayDataset
from dataset import ModifiedDenseNet
# Parameters
batch_size = 64
use_additional_features = True  # Set to match training
num_classes = 14
label_columns = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
    'Fibrosis', 'Edema', 'Consolidation'
]
model_save_path = 'model_with_features_v2_batch64_epoch20_scheduler_lr0001_images224_weighted_bce_with_logits_loss.pth'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define the test data
test_csv_file = './ChestX-ray14/test_data.csv'  # Path to the test data
root_dir = './ChestX-ray14/images224'

# Transformations
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = ChestXRayDataset(test_csv_file, root_dir, transform, use_additional_features)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = ModifiedDenseNet(num_classes=num_classes, use_additional_features=use_additional_features)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Evaluation loop
print("Starting evaluation...")
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if use_additional_features:
            images, additional_features, labels = batch

            # Filter out invalid data
            valid_indices = [i for i in range(len(images)) if images[i].numel() > 0]
            if len(valid_indices) == 0:  # Skip batch if all data is invalid
                continue

            images = images[valid_indices].to(device)
            additional_features = additional_features[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            outputs = model(images, additional_features)
        else:
            images, labels = batch

            # Filter out invalid data
            valid_indices = [i for i in range(len(images)) if images[i].numel() > 0]
            if len(valid_indices) == 0:  # Skip batch if all data is invalid
                continue

            images = images[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            outputs = model(images)

        predictions = torch.sigmoid(outputs).cpu()
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0 or batch_idx == len(test_loader) - 1:
            print(f"Processed batch [{batch_idx + 1}/{len(test_loader)}]")

print("Evaluation complete.")

# Convert predictions and labels to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Compute metrics
metrics = []
for i, label in enumerate(label_columns):
    # Compute ROC AUC
    auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])

    # Compute optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_predictions[:, i])
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]

    # Apply threshold to get binary predictions
    binary_predictions = (all_predictions[:, i] >= optimal_threshold).astype(int)

    # Compute other metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels[:, i], binary_predictions, average='binary')
    accuracy = accuracy_score(all_labels[:, i], binary_predictions)
    sensitivity = recall
    specificity = sum((binary_predictions == 0) & (all_labels[:, i] == 0)) / sum(all_labels[:, i] == 0)

    metrics.append({
        'Label': label,
        'AUC': auc,
        'Threshold': optimal_threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Save metrics to a CSV file
metrics_df = pd.DataFrame(metrics)
metrics_csv_path = './ChestX-ray14/test_metrics_model_v2_batch64_epoch20_scheduler_lr0001_images224_weighted_bce_with_logits_loss.csv'
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved to {metrics_csv_path}")

# Print metrics in table format
print(metrics_df)
