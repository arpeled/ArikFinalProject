import pandas as pd

# Load the existing dataset
existing_data_path = './ChestX-ray14/train_df.csv'  # Path to your current dataset
updated_data_path = './ChestX-ray14/updated_train_df.csv'  # Save path for the updated dataset
additional_info_path = './ChestX-ray14/Data_Entry_2017_v2020.csv'  # Path to the additional info file

# Read the existing and additional data
existing_df = pd.read_csv(existing_data_path)
additional_df = pd.read_csv(additional_info_path)

# Merge the datasets on 'Image Index'
merged_df = pd.merge(existing_df, additional_df[['Image Index', 'Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']],
                     on='Image Index', how='left')

# Save the updated dataset
merged_df.to_csv(updated_data_path, index=False)
print(f"Updated dataset saved to {updated_data_path}.")
