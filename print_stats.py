import pandas as pd

# Load the dataset
df = pd.read_csv('./ChestX-ray14/Data_Entry_2017_v2020.csv')  # Replace with your actual CSV file path
print(df.shape)
#(112120, 11)Data_Entry_2017_v2020
#(111863, 19)updated_train_df_fixed
# Split finding labels
all_finding_labels = []
labels_without_no_finding = []
for _, row in df.iterrows():
    finding_labels = row['Finding Labels'].split('|')
    for label in finding_labels:
        all_finding_labels.append(label)
        if label != 'No Finding':
            labels_without_no_finding.append(label)


# Calculate the frequency of each finding label
finding_counts = pd.Series(all_finding_labels).value_counts()
class_percentages = (finding_counts / len(all_finding_labels)) * 100

finding_counts_without_no_finding = pd.Series(labels_without_no_finding).value_counts()
class_percentages_without_no_finding = (finding_counts_without_no_finding / len(labels_without_no_finding)) * 100


# Print the counts
print("finding_counts_without_no_finding")
for finding, percentage in class_percentages_without_no_finding.items():
    print(f'{finding}:{percentage:.2f}%')
print("finding_counts")
# Print the percentages
for finding, percentage in class_percentages.items():
    print(f'{finding}:{percentage:.2f}%')
class_weights = 1 - (finding_counts / len(all_finding_labels))
print(class_weights)
for finding, weight in class_weights.items():
    print(f'{finding}:{weight:.4f}')

#
# # Print weights
# for finding, weight in class_weights.items():
#     print(f'{finding}:{weight:.4f}')