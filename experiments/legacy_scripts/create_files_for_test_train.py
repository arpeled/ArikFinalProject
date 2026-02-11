import pandas as pd
from sklearn.model_selection import train_test_split

# Paths to input and output files
input_csv = './ChestX-ray14/Data_Entry_2017_v2020.csv'  # Full data file
train_csv = './ChestX-ray14/train_data.csv'  # Train data file
test_csv = './ChestX-ray14/test_data.csv'  # Test data file

# Label columns
label_columns = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]

# Load the full dataset
data = pd.read_csv(input_csv)

# Split the "Finding Labels" column into multiple binary label columns
for label in label_columns:
    data[label] = data['Finding Labels'].str.contains(label).astype(int)

# Drop unnecessary columns
columns_to_drop = ['Finding Labels', 'OriginalImage[Width','Height]', 'OriginalImagePixelSpacing[x','y]']
data = data.drop(columns=columns_to_drop)

# Split into train and test datasets (80/20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test datasets
train_data.to_csv(train_csv, index=False)
test_data.to_csv(test_csv, index=False)

print(f"Train dataset saved to {train_csv}.")
print(f"Test dataset saved to {test_csv}.")
