import torch
import os
import sys
import time
import logging
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from dataset import ChestXRayDataset, ModifiedDenseNetWithDropOut

# ======== קבלת timestamp ========
run_timestamp = "20250402-230126"
use_additional_features = True  # Set to match training

# ======== קונפיגורציה ========
data_dir = './data/chest_xray/test'
batch_size = 64
model_save_path = f'run_info_model_{run_timestamp}.pth'
log_file_path = f'./run_log_{run_timestamp}.txt'
test_csv_file = './ChestX-ray14/test_data.csv'
root_dir = './ChestX-ray14/images224'
num_classes = 14
label_columns = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
    'Fibrosis', 'Edema', 'Consolidation'
]
# ======== התחלת לוג ========
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s: %(message)s'
)
logging.info("Start Evaluation")
logging.info(f"Timestamp: {run_timestamp}")

# ======== הגדרת טרנספורמים ========
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======== טעינת דאטה ========
test_dataset = ChestXRayDataset(dataset=None, csv_file=test_csv_file,
                                root_dir=root_dir, transform=test_transform,
                                use_additional_features=use_additional_features)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======== טעינת המודל המאומן ========
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = ModifiedDenseNetWithDropOut(num_classes=num_classes, use_additional_features=use_additional_features).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# ======== הרצת המודל על הדאטה ========
predictions = []
true_labels = []
probs_all = []

with torch.no_grad():
    for batch in test_loader:
        if use_additional_features:
            inputs, additional_features, labels = batch
            inputs = inputs.to(device)
            additional_features = additional_features.to(device)
            outputs = model(inputs, additional_features)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        probs_all.extend(probs)
        predictions.extend((probs > 0.5).astype(int))
        true_labels.extend(labels.cpu().numpy())

# ======== חישוב מטריקות ========
true_labels = np.array(true_labels)
predictions = np.array(predictions)
probs_all = np.array(probs_all)

headers = ["Label", "AUC", "Threshold", "Sensitivity", "Specificity", "Accuracy", "Precision", "Recall", "F1-Score"]
header_str = "{:<25} {:<8} {:<9} {:<12} {:<12} {:<10} {:<10} {:<10} {:<10}".format(*headers)
logging.info(header_str)
print(header_str)

for i in range(num_classes):
    y_true = true_labels[:, i]
    y_pred = predictions[:, i]
    y_prob = probs_all[:, i]

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')

    threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    label_name = label_columns[i]
    result_str = "{:<25} {:.4f}   {:<9.2f} {:<12.2f} {:<12.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
        label_name, auc, threshold, sensitivity, specificity, accuracy, precision, recall, f1)

    logging.info(result_str)
    print(result_str)

logging.info("Evaluation finished.")
print(f"Log saved to: {log_file_path}")
