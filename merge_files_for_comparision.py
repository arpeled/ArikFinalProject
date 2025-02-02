import pandas as pd

# Load the two Excel files
file1_path = '/users/arikpeled/PycharmProjects/ArikFinalProject/ChestX-ray14/test_metrics_model_with_features_v2_batch32_epoch30_scheduler_lr0001_images224_weighted_bce_with_logits_loss.csv'
file2_path = '/users/arikpeled/Downloads/Wang_performance_table.xlsx'

# Read both Excel files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the files
wang_df = pd.read_excel(file2_path, engine='calamine')
my_df = pd.read_csv(file1_path, engine='python')
#

# Create a longer format for easier plotting
wang_df['Source'] = 'Wang'
my_df['Source'] = 'My Metrics'

# Combine the dataframes
combined_df = pd.concat([wang_df, my_df])

# Create a figure with subplots for different metrics
metrics = ['AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'F1-Score']
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
    diff = my_df[metric].sum() - wang_df[metric].sum()
    print(f"{metric} average difference: {diff:.4f}")
