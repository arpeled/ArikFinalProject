import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import time
from torchvision.models import densenet121, DenseNet121_Weights

import torch.nn as nn
from torchvision import models


class ChestXRayDataset(Dataset):
    def __init__(self,dataset, csv_file, root_dir, transform=None, use_additional_features=False):
        if dataset is not None:
            self.data = dataset
        else:
            self.data = pd.read_csv(csv_file)

        # Drop unnecessary columns
        for col in ["PatientId", "Patient ID", "FilePath", "No Finding"]:
            if col in self.data.columns:
                self.data.drop(columns=[col], inplace=True)

        # Define label columns explicitly
        self.label_columns = [
            'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
            'Pneumothorax',
            'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
        ]

        # Ensure all label columns are numeric
        self.data[self.label_columns] = self.data[self.label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Define additional feature columns (if any)
        self.additional_columns = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']

        self.root_dir = root_dir
        self.transform = transform
        self.use_additional_features = use_additional_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.root_dir}/{self.data.iloc[idx, 0]}"
        try:
            # Attempt to open the image
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: File not found: {img_name}")
            if self.use_additional_features:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)
        # print(type(self.data.loc[idx, self.label_columns].values))
        # print(self.data.loc[idx, self.label_columns].values)

        # Extract labels
        labels = torch.tensor(self.data.loc[idx, self.label_columns].astype(float).values, dtype=torch.float32)

        if self.use_additional_features:
            # Extract additional features
            follow_up = torch.tensor(self.data.iloc[idx]["Follow-up #"], dtype=torch.float32)
            patient_age = torch.tensor(self.data.iloc[idx]["Patient Age"], dtype=torch.float32)
            patient_gender = 1.0 if self.data.iloc[idx]["Patient Gender"] == "M" else 0.0
            view_position = 1.0 if self.data.iloc[idx]["View Position"] == "PA" else 0.0
            additional_features = torch.tensor([follow_up, patient_age, patient_gender, view_position],
                                               dtype=torch.float32)
            return image, additional_features, labels

        return image, labels


# Define a modified DenseNet model
class ModifiedDenseNet(nn.Module):
    def __init__(self, num_classes=14, use_additional_features=False):
        super(ModifiedDenseNet, self).__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
        self.base_model = model
        # self.base_model = models.densenet121(pretrained=True)
        self.base_model.classifier = nn.Identity()  # Remove DenseNet's classifier
        self.use_additional_features = use_additional_features

        if self.use_additional_features:
            self.additional_fc = nn.Linear(4, 128)  # Four additional features
            self.final_fc = nn.Linear(1024 + 128, num_classes)  # Combine DenseNet and additional features
        else:
            self.final_fc = nn.Linear(1024, num_classes)

    def forward(self, x, additional_features=None):
        base_out = self.base_model(x)

        if self.use_additional_features:
            additional_out = nn.ReLU()(self.additional_fc(additional_features))
            combined = torch.cat([base_out, additional_out], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(base_out)

        return out


class ModifiedDenseNetWithDropOut(nn.Module):
    def __init__(self, num_classes=14, use_additional_features=False):
        super(ModifiedDenseNetWithDropOut, self).__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
        self.base_model = model
        self.base_model.classifier = nn.Identity()  # Remove DenseNet's classifier
        self.dropout = nn.Dropout(0.3)  # Dropout חדש!

        self.use_additional_features = use_additional_features
        if self.use_additional_features:
            self.additional_fc = nn.Linear(4, 128)
            self.final_fc = nn.Linear(1024 + 128, num_classes)
        else:
            self.final_fc = nn.Linear(1024, num_classes)

    def forward(self, x, additional_features=None):
        base_out = self.base_model(x)
        base_out = self.dropout(base_out)  # הוספת Dropout כאן

        if self.use_additional_features:
            additional_out = nn.ReLU()(self.additional_fc(additional_features))
            combined = torch.cat([base_out, additional_out], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(base_out)

        return out