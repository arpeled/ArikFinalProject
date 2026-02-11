import pandas as pd

# Load the two Excel files
file1_path = '/users/arikpeled/PycharmProjects/ArikFinalProject/ChestX-ray14/test_metrics_model_with_features_v4_batch64_epoch50_scheduler_lr0001_images224_weighted_bce_with_logits_loss_early_stop_warmup25_pat7_ModifiedDenseNetWithDropOut_fix_val_test_split_change_weights.csv'
# file2_path = '/users/arikpeled/Downloads/Wang_performance_table.xlsx'
file2_path = '/users/arikpeled/PycharmProjects/ArikFinalProject/ChestX-ray14/test_metrics_model_with_features_v4_batch64_epoch50_scheduler_lr0001_images224_weighted_bce_with_logits_loss_early_stop_warmup25_pat7_ModifiedDenseNetWithDropOut_fix_val_test_split_change_weights2.csv'

# Read both Excel files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the files
# wang_df = pd.read_excel(file2_path, engine='calamine')
my_df1 = pd.read_csv(file2_path, engine='python')
my_df = pd.read_csv(file1_path, engine='python')
#

# Create a longer format for easier plotting
my_df1['Source'] = 'My Metrics 1'
my_df['Source'] = 'My Metrics 2'

# Combine the dataframes
combined_df = pd.concat([my_df1, my_df])

# Create a figure with subplots for different metrics
# metrics = ['AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'F1-Score']
metrics = ['AUC',  'Accuracy',  'F1-Score']
plt.figure(figsize=(20, 12))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.barplot(data=combined_df, x='Label', y=metric, hue='Source')
    plt.xticks(rotation=45)
    plt.title(f'Comparison of {metric}')
    plt.legend(title='')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\
Summary of differences (My Metrics - Wang):")
for metric in metrics:
    diff = my_df[metric].sum() - my_df1[metric].sum()
    print(f"{metric} average difference: {diff:.4f}")

import pandas as pd


# Read the Excel file
# df1 = pd.read_excel(file2_path, engine='openpyxl')
df1 = pd.read_csv(file2_path)

# Read the CSV file
df2 = pd.read_csv(file1_path)

# Finding common columns excluding 'Label'
common_columns = list(set(df1.columns).intersection(set(df2.columns)) - {'Label'})

# Renaming columns to indicate source file
df1_selected = df1[['Label'] + common_columns].add_suffix('_File1').rename(columns={'Label_File1': 'Label'})
df2_selected = df2[['Label'] + common_columns].add_suffix('_File2').rename(columns={'Label_File2': 'Label'})

# Merging both datasets based on the 'Label' column
comparison_df = pd.merge(df1_selected, df2_selected, on='Label', how='inner')

# Save the comparison results
comparison_df.to_csv("comparison_all_metrics2.csv", index=False)

print("Comparison file saved as 'comparison_all_metrics2.csv'")
