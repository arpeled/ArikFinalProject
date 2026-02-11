import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import ChestXRayDataset, ModifiedDenseNet, ModifiedDenseNetWithDropOut
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.model_selection import GroupShuffleSplit

#  fix performace issue, validation loss is not decreasing even when training loss is decreasing
# change weights - normalize them
# change learning rate
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5, verbose=True)
# add dropout using ModifiedDenseNetWithDropOut
# add data augmentation
# GroupShuffleSplit - split the data by patient id



# Parameters
num_classes = 14
batch_size = 64
learning_rate = 0.0001
num_epochs = 50
use_additional_features = True  # Toggle for using additional features
num_workers =8

# Load data
csv_file = './ChestX-ray14/train_data.csv'  # Use the fixed CSV
root_dir = './ChestX-ray14/images224'

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# for augmentation on train only
transform_train = transforms.Compose([
                        # transforms.Resize((224, 224)),
                        # transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])


def weighted_bce_with_logits_loss(output, target, weights):
    bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    weighted_loss = bce_loss * weights
    return weighted_loss.mean()


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.01, warmup_epochs=25):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_loss = None
        self.counter = 0
        self.epoch_count = 0

    def __call__(self, val_loss):
        self.epoch_count += 1
        print(f"Epoch: {self.epoch_count}, Validation Loss: {val_loss:.4f}")
        if self.epoch_count < self.warmup_epochs:
            return False

        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # עצור את האימון
        return False


def train_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"✅ Using {num_gpus} GPU(s) on {device}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        # dist.init_process_group(backend=backend)
        print(f"✅ Using backend for distributed training.")

    # Dataset and DataLoader

    df = pd.read_csv(csv_file)

    # בדוק אם יש עמודת patient_id
    if "Patient ID" not in df.columns:
        raise ValueError("עמודת 'patient_id' לא קיימת ב-CSV. צריך לוודא שהיא קיימת.")

    # יישום GroupShuffleSplit לחלוקה לפי חולים
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))

    # יצירת DataFrames נפרדים
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_patients = set(train_df["Patient ID"].unique())
    val_patients = set(val_df["Patient ID"].unique())

    common_patients = train_patients.intersection(val_patients)
    print(f"חולים חופפים בין הסטים: {len(common_patients)}")  # אמור להיות 0!

    # יצירת הדאטהסטים תוך שימוש ב-DataFrame הספציפי לכל סט
    train_dataset = ChestXRayDataset(dataset=train_df, csv_file=None, root_dir= root_dir,transform= transform_train, use_additional_features=True)
    val_dataset = ChestXRayDataset(dataset=val_df, csv_file=None, root_dir= root_dir,transform=transform, use_additional_features=True)

    # יצירת ה-DataLoaders
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"מספר הדוגמאות ב-Train: {len(train_dataset)}, מספר הדוגמאות ב-Validation: {len(val_dataset)}")
    # Initialize model
    model = ModifiedDenseNetWithDropOut(num_classes=num_classes, use_additional_features=use_additional_features).to(device)
    if torch.cuda.is_available() and num_gpus > 1:
        print("parallel gpu training")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    class_weights = torch.tensor([
        2.456140350877193,2.709677419354839, 0.5118829981718466,30.0, 0.34271725826193394, 1.1797752808988766, 1.076923076923077,
        0.5898876404494383, 1.2863705972434918, 2.0143884892086334, 4.7727272727272725, 4.038461538461539, 2.95774647887324,  1.4608695652173913
    ], dtype=torch.float32).to(device)
    #         self.label_columns = [
    #             'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    #             'Atelectasis', 'Pneumothorax','Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
    #         ]
    # New Class Weights: {'Infiltration': 0.34271725826193394, 'Effusion': 0.5118829981718466,
    #                       'Atelectasis': 0.5898876404494383, 'Nodule': 1.076923076923077,
    #                       'Mass': 1.1797752808988766, 'Pneumothorax': 1.2863705972434918,
    #                       'Consolidation': 1.4608695652173913,
    #                       'Pleural_Thickening': 2.0143884892086334,
    #                       'Cardiomegaly': 2.456140350877193, 'Emphysema': 2.709677419354839,
    #                       'Edema': 2.95774647887324, 'Fibrosis': 4.038461538461539,
    #                       'Pneumonia': 4.7727272727272725, 'Hernia': 30.0}

    # Define a custom weighted BCEWithLogitsLoss

    # During training
    criterion = lambda output, target: weighted_bce_with_logits_loss(output, target, class_weights)

    # Loss and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-5)

    loss_history = []
    # Training loop
    print("Starting training... at: ", time.time())
    early_stopping = EarlyStopping()

    for epoch in range(num_epochs):
        epoch_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"Starting Epoch [{epoch + 1}/{num_epochs}]")

        for batch_idx, batch in enumerate(dataloader_train, start=1):
            batch_start_time = time.time()

            if use_additional_features:
                images, additional_features, labels = batch
                images, additional_features, labels = images.to(device), additional_features.to(device), labels.to(
                    device)
                outputs = model(images, additional_features)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            # print(outputs.shape)
            if isinstance(outputs, tuple):  # ✅ Handle DataParallel outputs
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0 or batch_idx == len(dataloader_train):  # Print every 100 batches or the last batch
                current_time = time.time()
                elapsed_time = current_time - epoch_time
                print(
                    f"🟢 Batch {batch_idx}/{len(dataloader_train)}: Loss={loss.item():.4f} - Time={time.time() - batch_start_time:.2f}s")

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader_train)}], Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f} seconds")

        epoch_loss = running_loss / len(dataloader_train)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Complete. Average Loss: {epoch_loss:.4f}")

        # Validation loop (if you have validation data)
        val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for batch in dataloader_val:
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
        val_loss /= len(dataloader_val)
        print(f"Validation Loss: {val_loss:.4f}")
        loss_history.append(val_loss)
        # Step the scheduler with the validation loss
        scheduler.step(val_loss)


        # Early Stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr}")

    print("Training complete.")
    print("Loss history: ", loss_history)
    return model


if __name__ == "__main__":
    # Ensure your main training code is inside this block
    start_time = time.time()
    model = train_model()
    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    torch.save(model.state_dict(),
               'run_info_5_model_with_features_v4_change_weights3_plus_augmentation.pth' if use_additional_features else 'model_without_features.pth')
    print("Model saved successfully.")
