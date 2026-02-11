
# Get timestamp from command line or define manually

import torch
import os
import sys
import time
import logging
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from dataset import ChestXRayDataset, ModifiedDenseNetWithDropOut

# ======== קבלת timestamp מהמשתמש ========
# if len(sys.argv) < 2:
#     print("Usage: python chest-x-ray-classifier-densenet121-test_v05.py <timestamp>")
#     sys.exit(1)

run_timestamp = "20250402-230126"
use_additional_features = True  # Set to match training

# ======== קונפיגורציה ========
data_dir = './data/chest_xray/test'
batch_size = 64
model_save_path = f'run_info_model_{run_timestamp}.pth'
log_file_path = f'./run_log_{run_timestamp}.txt'
test_csv_file = './ChestX-ray14/test_data.csv'  # Path to the test data
root_dir = './ChestX-ray14/images224'
num_classes = 14
# ======== התחלת לוג ========
# os.makedirs("./logs", exist_ok=True)
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
                                root_dir= root_dir,transform= test_transform,use_additional_features=use_additional_features)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======== טעינת המודל המאומן ========
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# model = torch.load(model_save_path, map_location=device)

model = ModifiedDenseNetWithDropOut(num_classes=num_classes, use_additional_features=use_additional_features).to(device)

# טען את המשקלים
model.load_state_dict(torch.load(model_save_path, map_location=device))

# העבר למצב הערכה
model.eval()
# ======== הרצת המודל על הדאטה ========
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.numpy())

# ======== חישוב מטריקות ========
report = classification_report(true_labels, predictions, target_names=test_dataset.class_names)
print(report)
logging.info("Evaluation Results:")
logging.info("\n" + report)

logging.info("Evaluation finished.")
print(f"Log saved to: {log_file_path}")
