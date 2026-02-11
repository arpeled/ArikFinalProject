# # This is a sample Python script.
#
# # Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
# import tensorflow as tf
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#     print(tf.config.list_physical_devices('GPU'))
#     all_xray_df = pd.read_csv("./ChestX-ray14/train_df.csv")
#     # all_xray_df.drop(['No Finding'], axis = 1, inplace = True)
#     print(all_xray_df.head())
#     for col in all_xray_df.columns[2:]:
#         all_xray_df[col] = pd.to_numeric(all_xray_df[col], errors='coerce')
#
#     # Calculating statistics for each column
#     stats = all_xray_df.iloc[:, 2:].sum().sort_values(ascending=False).rename("Count").to_frame()
#     stats["Percentage"] = (stats["Count"] / len(all_xray_df)) * 100
#     plt.figure(figsize=(10, 6))
#     stats["Count"].plot(kind='bar')
#     plt.title("Number of Patients for Each Condition")
#     plt.xlabel("Condition")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
#
#     # Pie Chart Visualization (for percentage)
#     plt.figure(figsize=(8, 8))
#     stats["Percentage"].plot(kind='pie', autopct='%1.1f%%', startangle=140, legend=True)
#     plt.title("Percentage of Patients for Each Condition")
#     plt.ylabel("")  # Hide the y-axis label
#     plt.show()
#     # Display the statistics
#
#     stats_no_findings = stats.drop("No Finding", errors='ignore')
#
#     # Bar Chart Visualization
#     plt.figure(figsize=(10, 6))
#     stats_no_findings["Count"].plot(kind='bar')
#     plt.title("Number of Patients for Each Condition (Excluding 'No Finding')")
#     plt.xlabel("Condition")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
#
#     # Pie Chart Visualization (for percentage)
#     plt.figure(figsize=(8, 8))
#     stats_no_findings["Percentage"].plot(kind='pie', autopct='%1.1f%%', startangle=140, legend=True)
#     plt.title("Percentage of Patients for Each Condition (Excluding 'No Finding')")
#     plt.ylabel("")  # Hide the y-axis label
#     plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import pandas as pd
#
# csv_file = './ChestX-ray14/updated_train_df.csv'
# data = pd.read_csv(csv_file)
#
# # Select only the first 14 label columns
# num_classes = 14
# label_columns = data.columns[1:1+num_classes]  # Adjust based on the dataset structure
# data = data[['Image Index'] + list(label_columns)]  # Retain the image index and label columns
#
# # Save the updated CSV
# data.to_csv('./ChestX-ray14/updated_train_df_fixed.csv', index=False)
import pandas as pd

# Paths to the files
input_csv = './ChestX-ray14/updated_train_df.csv'  # Input CSV file with additional features
output_csv = './ChestX-ray14/updated_train_df_fixed.csv'  # Processed CSV file

# Define the 14 label columns explicitly
label_columns = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]

# Additional feature columns to retain
additional_columns = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']

# Load the CSV
data = pd.read_csv(input_csv)

# Ensure all label columns exist in the dataset
missing_columns = [col for col in label_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"The following label columns are missing from the input CSV: {missing_columns}")

# Retain only the required columns
columns_to_keep = ['Image Index'] + label_columns + additional_columns
existing_columns_to_keep = [col for col in columns_to_keep if col in data.columns]  # Ensure columns exist in the dataset
data = data[existing_columns_to_keep]

# Save the processed CSV
data.to_csv(output_csv, index=False)
print(f"Processed CSV saved to {output_csv}")
#
# import pandas as pd
#
# csv_file = './ChestX-ray14/updated_train_df_fixed.csv'
# data = pd.read_csv(csv_file)
#
# # Check for non-numeric values in the label columns
# num_classes = 14
# label_columns = data.columns[1:1 + num_classes]
# non_numeric_labels = data[label_columns].apply(pd.to_numeric, errors='coerce').isnull()
#
# # Print rows with non-numeric labels
# if non_numeric_labels.any().any():
#     print("Rows with non-numeric labels:")
#     print(data[non_numeric_labels.any(axis=1)])
# else:
#     print("All label columns are numeric.")
