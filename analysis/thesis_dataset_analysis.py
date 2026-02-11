"""
Thesis Dataset Analysis: ChestX-ray14
=====================================
MSc Thesis: "Chest X-ray Disease Identification using Deep Learning"
Chapter 3: Dataset and Problem Definition

This script generates statistical reports and publication-quality figures
for the NIH ChestX-ray14 dataset analysis.

Author: [Your Name]
Date: 2026-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths - adjust as needed
DATA_DIR = Path("./ChestX-ray14")
TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = DATA_DIR / "test_data.csv"
FULL_CSV = DATA_DIR / "Data_Entry_2017.csv"  # Original NIH metadata if available

# Output directories
OUTPUT_DIR = Path("./thesis_figures")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Disease classes (canonical order for consistency across all plots)
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

# =============================================================================
# FIGURE STYLE CONFIGURATION (Thesis-Ready)
# =============================================================================

def setup_thesis_style():
    """Configure matplotlib for thesis-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        # Font sizes
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,

        # Font family
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],

        # Figure settings
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Axes
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',

        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        # Lines
        'lines.linewidth': 1.5,
    })

# Grayscale-friendly color palette
THESIS_COLORS = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#7F8C8D',    # Medium gray
    'accent': '#34495E',       # Slate
    'highlight': '#1A5276',    # Deep blue
    'light': '#BDC3C7',        # Light gray
    'rare': '#922B21',         # Dark red for rare classes
}

# Grayscale gradient for bar charts
def get_grayscale_palette(n_colors):
    """Generate a grayscale palette from dark to light."""
    return [plt.cm.Greys(0.3 + 0.5 * i / n_colors) for i in range(n_colors)]


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data(csv_path):
    """
    Load dataset and prepare for analysis.

    Parameters:
    -----------
    csv_path : Path
        Path to CSV file with binary disease columns

    Returns:
    --------
    pd.DataFrame with additional processed columns
    """
    print(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'Image Index': 'Image',
        'Patient ID': 'PatientID',
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # The CSV already has binary columns for each disease
    # Build LabelList from binary columns
    def get_label_list(row):
        labels = []
        for disease in DISEASE_CLASSES:
            if disease in row.index and row[disease] == 1:
                labels.append(disease)
        return labels

    df['LabelList'] = df.apply(get_label_list, axis=1)

    # Count labels per image
    df['LabelCount'] = df['LabelList'].apply(len)

    # Check for "No Finding" - if all disease columns are 0
    df['HasDisease'] = df['LabelCount'] > 0

    print(f"  Loaded {len(df):,} images from {df['PatientID'].nunique():,} patients")

    return df


def load_split_data():
    """Load train and test splits."""
    train_df = load_and_prepare_data(TRAIN_CSV)
    test_df = load_and_prepare_data(TEST_CSV)

    # Add split indicator
    train_df['Split'] = 'train'
    test_df['Split'] = 'test'

    # Combine for full analysis
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    return train_df, test_df, full_df


# =============================================================================
# ANALYSIS 1: DATASET OVERVIEW STATISTICS
# =============================================================================

def generate_dataset_overview(df, output_name='dataset_overview'):
    """
    Generate dataset overview statistics table.

    Outputs:
    - CSV table with summary statistics
    """
    print("\n" + "="*60)
    print("ANALYSIS 1: Dataset Overview Statistics")
    print("="*60)

    # Calculate statistics
    stats = {
        'Metric': [
            'Total Images',
            'Total Unique Patients',
            'Average Images per Patient',
            'Median Images per Patient',
            'Min Images per Patient',
            'Max Images per Patient',
            'Images with Disease',
            'Images without Disease (No Finding)',
            'Disease Prevalence Rate'
        ],
        'Value': [
            f"{len(df):,}",
            f"{df['PatientID'].nunique():,}",
            f"{len(df) / df['PatientID'].nunique():.2f}",
            f"{df.groupby('PatientID').size().median():.1f}",
            f"{df.groupby('PatientID').size().min()}",
            f"{df.groupby('PatientID').size().max()}",
            f"{df['HasDisease'].sum():,}",
            f"{(~df['HasDisease']).sum():,}",
            f"{df['HasDisease'].mean()*100:.1f}%"
        ]
    }

    stats_df = pd.DataFrame(stats)

    # Save to CSV
    output_path = TABLES_DIR / f"{output_name}.csv"
    stats_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Print summary
    print("\n  Dataset Overview:")
    print(stats_df.to_string(index=False))

    return stats_df


# =============================================================================
# ANALYSIS 2: DISEASE FREQUENCY ANALYSIS
# =============================================================================

def generate_disease_frequency_analysis(df, output_name='disease_frequency'):
    """
    Generate disease frequency analysis with bar chart and table.

    Outputs:
    - Bar chart (log scale) of image count per disease
    - CSV table with counts and percentages
    """
    print("\n" + "="*60)
    print("ANALYSIS 2: Disease Frequency Analysis")
    print("="*60)

    # Calculate frequencies
    freq_data = []
    for disease in DISEASE_CLASSES:
        image_count = df[disease].sum()
        patient_count = df[df[disease] == 1]['PatientID'].nunique()
        freq_data.append({
            'Disease': disease,
            'Image_Count': image_count,
            'Patient_Count': patient_count,
            'Image_Percentage': image_count / len(df) * 100,
            'Patient_Percentage': patient_count / df['PatientID'].nunique() * 100
        })

    freq_df = pd.DataFrame(freq_data)
    freq_df = freq_df.sort_values('Image_Count', ascending=False).reset_index(drop=True)

    # Save table
    table_path = TABLES_DIR / f"{output_name}_table.csv"
    freq_df.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")

    # Create bar chart (log scale)
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort by frequency for visualization
    sorted_df = freq_df.sort_values('Image_Count', ascending=True)

    # Create horizontal bar chart
    colors = [THESIS_COLORS['rare'] if c < 1000 else THESIS_COLORS['primary']
              for c in sorted_df['Image_Count']]

    bars = ax.barh(sorted_df['Disease'], sorted_df['Image_Count'],
                   color=colors, edgecolor='white', linewidth=0.5)

    # Log scale
    ax.set_xscale('log')

    # Labels and formatting
    ax.set_xlabel('Number of Images (log scale)', fontweight='bold')
    ax.set_ylabel('Disease Class', fontweight='bold')
    ax.set_title('Disease Frequency Distribution in ChestX-ray14 Dataset',
                 fontweight='bold', pad=20)

    # Add count labels on bars
    for bar, count in zip(bars, sorted_df['Image_Count']):
        ax.text(count * 1.1, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)

    # Add legend for rare classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THESIS_COLORS['primary'], label='Common (>1,000 images)'),
        Patch(facecolor=THESIS_COLORS['rare'], label='Rare (<1,000 images)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    # Grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    fig_path = FIGURES_DIR / f"{output_name}_barchart.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Print summary
    print("\n  Disease Frequency Summary:")
    print(freq_df[['Disease', 'Image_Count', 'Image_Percentage']].to_string(index=False))

    return freq_df


# =============================================================================
# ANALYSIS 3: MULTI-LABEL CHARACTERISTICS
# =============================================================================

def generate_multilabel_analysis(df, output_name='multilabel_distribution'):
    """
    Analyze multi-label characteristics of the dataset.

    Outputs:
    - Histogram of label count per image
    - Summary statistics table
    """
    print("\n" + "="*60)
    print("ANALYSIS 3: Multi-Label Characteristics")
    print("="*60)

    # Count distribution
    label_counts = df['LabelCount'].value_counts().sort_index()

    # For images with "No Finding", set label count to 0
    df_analysis = df.copy()
    df_analysis.loc[~df_analysis['HasDisease'], 'DiseaseCount'] = 0
    df_analysis.loc[df_analysis['HasDisease'], 'DiseaseCount'] = df_analysis.loc[
        df_analysis['HasDisease'], DISEASE_CLASSES
    ].sum(axis=1)

    disease_counts = df_analysis['DiseaseCount'].value_counts().sort_index()

    # Create summary table
    summary_data = []
    for count in range(int(df_analysis['DiseaseCount'].max()) + 1):
        n_images = disease_counts.get(count, 0)
        summary_data.append({
            'Label_Count': count,
            'Number_of_Images': n_images,
            'Percentage': n_images / len(df) * 100,
            'Cumulative_Percentage': disease_counts[disease_counts.index <= count].sum() / len(df) * 100
        })

    summary_df = pd.DataFrame(summary_data)

    # Save table
    table_path = TABLES_DIR / f"{output_name}_table.csv"
    summary_df.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")

    # Create histogram
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    x_positions = summary_df['Label_Count'].values
    heights = summary_df['Number_of_Images'].values

    bars = ax.bar(x_positions, heights, color=THESIS_COLORS['primary'],
                  edgecolor='white', linewidth=0.5, width=0.8)

    # Labels and formatting
    ax.set_xlabel('Number of Disease Labels per Image', fontweight='bold')
    ax.set_ylabel('Number of Images', fontweight='bold')
    ax.set_title('Distribution of Multi-Label Annotations in ChestX-ray14',
                 fontweight='bold', pad=20)

    # Add percentage labels on top of bars
    for bar, pct in zip(bars, summary_df['Percentage']):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{int(x)}' for x in x_positions])

    # Format y-axis with comma separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add annotation for multi-label percentage
    multi_label_pct = summary_df[summary_df['Label_Count'] > 1]['Percentage'].sum()
    ax.annotate(f'Multi-label images: {multi_label_pct:.1f}%',
                xy=(0.7, 0.85), xycoords='axes fraction',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = FIGURES_DIR / f"{output_name}_histogram.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Print summary
    print("\n  Multi-Label Distribution:")
    print(summary_df.to_string(index=False))

    return summary_df


# =============================================================================
# ANALYSIS 4: CLASS IMBALANCE VISUALIZATION
# =============================================================================

def generate_imbalance_analysis(df, output_name='class_imbalance'):
    """
    Visualize class imbalance severity.

    Outputs:
    - Imbalance ratio chart (max_count / class_count)
    - Lorenz-style cumulative distribution
    """
    print("\n" + "="*60)
    print("ANALYSIS 4: Class Imbalance Visualization")
    print("="*60)

    # Calculate class counts
    class_counts = {disease: df[disease].sum() for disease in DISEASE_CLASSES}

    # Sort by count (ascending for imbalance visualization)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    diseases = [x[0] for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]

    max_count = max(counts)
    imbalance_ratios = [max_count / c for c in counts]

    # Create imbalance summary table
    imbalance_df = pd.DataFrame({
        'Disease': diseases,
        'Image_Count': counts,
        'Imbalance_Ratio': imbalance_ratios,
        'Percentage_of_Max': [c / max_count * 100 for c in counts]
    })

    # Save table
    table_path = TABLES_DIR / f"{output_name}_table.csv"
    imbalance_df.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")

    # Create figure with two subplots
    setup_thesis_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Subplot 1: Imbalance Ratio ----
    colors = [THESIS_COLORS['rare'] if r > 10 else THESIS_COLORS['primary']
              for r in imbalance_ratios]

    bars = ax1.barh(diseases, imbalance_ratios, color=colors,
                    edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Imbalance Ratio (Max Count / Class Count)', fontweight='bold')
    ax1.set_ylabel('Disease Class', fontweight='bold')
    ax1.set_title('Class Imbalance Severity', fontweight='bold', pad=15)

    # Add ratio labels
    for bar, ratio in zip(bars, imbalance_ratios):
        ax1.text(ratio + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{ratio:.1f}x', va='center', fontsize=9)

    ax1.axvline(x=10, color=THESIS_COLORS['secondary'], linestyle='--',
                alpha=0.7, label='10x threshold')
    ax1.legend(loc='lower right')
    ax1.xaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # ---- Subplot 2: Lorenz-style Cumulative Distribution ----
    # Sort by count for Lorenz curve
    sorted_counts = sorted(counts)
    cumulative_proportion = np.cumsum(sorted_counts) / sum(sorted_counts)
    class_proportion = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    ax2.plot([0] + list(class_proportion), [0] + list(cumulative_proportion),
             'o-', color=THESIS_COLORS['primary'], linewidth=2, markersize=6,
             label='Actual Distribution')
    ax2.plot([0, 1], [0, 1], '--', color=THESIS_COLORS['secondary'],
             linewidth=1.5, label='Perfect Equality')

    ax2.fill_between([0] + list(class_proportion), [0] + list(cumulative_proportion),
                     [0] + list(class_proportion), alpha=0.2, color=THESIS_COLORS['primary'])

    ax2.set_xlabel('Proportion of Disease Classes', fontweight='bold')
    ax2.set_ylabel('Cumulative Proportion of Samples', fontweight='bold')
    ax2.set_title('Lorenz Curve of Class Distribution', fontweight='bold', pad=15)
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Calculate Gini coefficient
    n = len(sorted_counts)
    gini = (2 * sum((i + 1) * c for i, c in enumerate(sorted_counts)) /
            (n * sum(sorted_counts))) - (n + 1) / n
    ax2.annotate(f'Gini Coefficient: {gini:.3f}',
                 xy=(0.55, 0.15), xycoords='axes fraction',
                 fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    fig_path = FIGURES_DIR / f"{output_name}_visualization.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Print summary
    print(f"\n  Class Imbalance Summary:")
    print(f"    Most common class: {sorted_classes[-1][0]} ({sorted_classes[-1][1]:,} images)")
    print(f"    Rarest class: {sorted_classes[0][0]} ({sorted_classes[0][1]:,} images)")
    print(f"    Maximum imbalance ratio: {max(imbalance_ratios):.1f}x")
    print(f"    Gini coefficient: {gini:.3f}")

    return imbalance_df


# =============================================================================
# ANALYSIS 5: PATIENT-LEVEL SPLIT VALIDATION
# =============================================================================

def validate_patient_splits(train_df, test_df, output_name='split_validation'):
    """
    Validate train/test split integrity (no patient leakage).

    Outputs:
    - Summary table of splits
    - Leakage check report
    """
    print("\n" + "="*60)
    print("ANALYSIS 5: Patient-Level Split Validation")
    print("="*60)

    # Get patient sets
    train_patients = set(train_df['PatientID'].unique())
    test_patients = set(test_df['PatientID'].unique())

    # Check for leakage
    overlap = train_patients & test_patients
    has_leakage = len(overlap) > 0

    # Calculate split statistics
    total_patients = len(train_patients | test_patients)
    total_images = len(train_df) + len(test_df)

    split_data = {
        'Split': ['Train', 'Test', 'Total'],
        'Patients': [
            len(train_patients),
            len(test_patients),
            total_patients
        ],
        'Images': [
            len(train_df),
            len(test_df),
            total_images
        ],
        'Patient_Percentage': [
            len(train_patients) / total_patients * 100,
            len(test_patients) / total_patients * 100,
            100.0
        ],
        'Image_Percentage': [
            len(train_df) / total_images * 100,
            len(test_df) / total_images * 100,
            100.0
        ],
        'Avg_Images_per_Patient': [
            len(train_df) / len(train_patients),
            len(test_df) / len(test_patients),
            total_images / total_patients
        ]
    }

    split_df = pd.DataFrame(split_data)

    # Add leakage information
    leakage_info = pd.DataFrame({
        'Check': ['Patient Leakage', 'Overlapping Patients'],
        'Result': ['PASS' if not has_leakage else 'FAIL', len(overlap)]
    })

    # Save tables
    split_table_path = TABLES_DIR / f"{output_name}_splits.csv"
    split_df.to_csv(split_table_path, index=False)
    print(f"  Saved: {split_table_path}")

    leakage_table_path = TABLES_DIR / f"{output_name}_leakage.csv"
    leakage_info.to_csv(leakage_table_path, index=False)
    print(f"  Saved: {leakage_table_path}")

    # Print summary
    print("\n  Split Summary:")
    print(split_df.to_string(index=False))
    print(f"\n  Patient Leakage Check: {'PASS - No overlap' if not has_leakage else 'FAIL - Overlap detected!'}")

    if has_leakage:
        print(f"    WARNING: {len(overlap)} patients appear in both splits!")
        print(f"    Overlapping patient IDs (first 10): {list(overlap)[:10]}")

    return split_df, has_leakage


# =============================================================================
# ANALYSIS 6: RARE DISEASE HIGHLIGHT
# =============================================================================

def highlight_rare_diseases(df, rare_threshold=1000, output_name='rare_diseases'):
    """
    Highlight rare diseases, especially Hernia.

    Outputs:
    - Detailed statistics for rare classes
    - Printed textual summary
    """
    print("\n" + "="*60)
    print("ANALYSIS 6: Rare Disease Highlight")
    print("="*60)

    # Calculate statistics for all diseases
    rare_data = []
    for disease in DISEASE_CLASSES:
        count = df[disease].sum()
        patient_count = df[df[disease] == 1]['PatientID'].nunique()

        # Calculate co-occurrence with other diseases
        disease_images = df[df[disease] == 1]
        cooccurrence = disease_images[DISEASE_CLASSES].sum().drop(disease)
        top_cooccur = cooccurrence.nlargest(3)

        rare_data.append({
            'Disease': disease,
            'Image_Count': count,
            'Patient_Count': patient_count,
            'Image_Percentage': count / len(df) * 100,
            'Patient_Percentage': patient_count / df['PatientID'].nunique() * 100,
            'Is_Rare': count < rare_threshold,
            'Avg_Labels_When_Present': disease_images['LabelCount'].mean(),
            'Top_Cooccurrence': ', '.join([f"{d}: {int(c)}" for d, c in top_cooccur.items()])
        })

    rare_df = pd.DataFrame(rare_data)
    rare_df = rare_df.sort_values('Image_Count', ascending=True).reset_index(drop=True)

    # Save table
    table_path = TABLES_DIR / f"{output_name}_table.csv"
    rare_df.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")

    # Print detailed rare disease analysis
    print("\n  " + "-"*50)
    print("  RARE DISEASE ANALYSIS (< 1,000 images)")
    print("  " + "-"*50)

    rare_classes = rare_df[rare_df['Is_Rare']]

    for _, row in rare_classes.iterrows():
        print(f"\n  {row['Disease'].upper()}")
        print(f"    - Absolute count: {row['Image_Count']:,} images")
        print(f"    - Percentage of images: {row['Image_Percentage']:.3f}%")
        print(f"    - Number of patients: {row['Patient_Count']:,}")
        print(f"    - Percentage of patients: {row['Patient_Percentage']:.3f}%")
        print(f"    - Avg labels when present: {row['Avg_Labels_When_Present']:.2f}")
        print(f"    - Top co-occurrences: {row['Top_Cooccurrence']}")

    # Special focus on Hernia
    hernia_stats = rare_df[rare_df['Disease'] == 'Hernia'].iloc[0]

    print("\n  " + "="*50)
    print("  HERNIA: THE MOST CHALLENGING CLASS")
    print("  " + "="*50)
    print(f"""
    Hernia represents the extreme case of class imbalance:

    * Only {hernia_stats['Image_Count']:,} images ({hernia_stats['Image_Percentage']:.3f}% of dataset)
    * Affects {hernia_stats['Patient_Count']:,} unique patients ({hernia_stats['Patient_Percentage']:.3f}%)
    * In test set of ~22,000 images: only ~39 Hernia-positive samples
    * This extreme imbalance makes Hernia detection fundamentally difficult

    Key implications for model training:
    1. Standard loss functions (BCE, CrossEntropy) ignore Hernia
    2. Per-class threshold optimization often results in TP=0
    3. Oversampling is required to force any feature learning
    4. Even with intervention, precision ceiling is ~1.5%
    """)

    return rare_df


# =============================================================================
# ANALYSIS 7: ADDITIONAL THESIS FIGURES
# =============================================================================

def generate_patient_distribution_figure(df, output_name='patient_distribution'):
    """Generate figure showing images per patient distribution."""
    print("\n" + "="*60)
    print("BONUS: Patient Image Distribution")
    print("="*60)

    setup_thesis_style()

    # Calculate images per patient
    images_per_patient = df.groupby('PatientID').size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(images_per_patient, bins=50, color=THESIS_COLORS['primary'],
             edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Number of Images per Patient', fontweight='bold')
    ax1.set_ylabel('Number of Patients', fontweight='bold')
    ax1.set_title('Distribution of Images per Patient', fontweight='bold')
    ax1.axvline(images_per_patient.mean(), color=THESIS_COLORS['rare'],
                linestyle='--', label=f'Mean: {images_per_patient.mean():.1f}')
    ax1.axvline(images_per_patient.median(), color=THESIS_COLORS['secondary'],
                linestyle=':', label=f'Median: {images_per_patient.median():.1f}')
    ax1.legend()

    # Box plot for outliers
    bp = ax2.boxplot(images_per_patient, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor(THESIS_COLORS['primary'])
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Number of Images', fontweight='bold')
    ax2.set_title('Images per Patient (Box Plot)', fontweight='bold')
    ax2.set_xticklabels(['All Patients'])

    plt.tight_layout()

    fig_path = FIGURES_DIR / f"{output_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")


def generate_disease_cooccurrence_heatmap(df, output_name='disease_cooccurrence'):
    """Generate co-occurrence matrix heatmap."""
    print("\n" + "="*60)
    print("BONUS: Disease Co-occurrence Analysis")
    print("="*60)

    setup_thesis_style()

    # Calculate co-occurrence matrix
    cooccur_matrix = np.zeros((len(DISEASE_CLASSES), len(DISEASE_CLASSES)))

    for i, d1 in enumerate(DISEASE_CLASSES):
        for j, d2 in enumerate(DISEASE_CLASSES):
            if i == j:
                cooccur_matrix[i, j] = df[d1].sum()
            else:
                cooccur_matrix[i, j] = ((df[d1] == 1) & (df[d2] == 1)).sum()

    # Normalize by diagonal (self-occurrence)
    norm_matrix = cooccur_matrix / np.diag(cooccur_matrix)[:, np.newaxis] * 100

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(norm_matrix, cmap='Greys', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(DISEASE_CLASSES)))
    ax.set_yticks(np.arange(len(DISEASE_CLASSES)))
    ax.set_xticklabels(DISEASE_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(DISEASE_CLASSES)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Co-occurrence Rate (%)', fontweight='bold')

    # Add text annotations for high values
    for i in range(len(DISEASE_CLASSES)):
        for j in range(len(DISEASE_CLASSES)):
            if norm_matrix[i, j] > 20 or i == j:
                text = ax.text(j, i, f'{norm_matrix[i, j]:.0f}',
                              ha='center', va='center',
                              color='white' if norm_matrix[i, j] > 50 else 'black',
                              fontsize=8)

    ax.set_title('Disease Co-occurrence Matrix\n(% of row disease appearing with column disease)',
                 fontweight='bold', pad=20)

    plt.tight_layout()

    fig_path = FIGURES_DIR / f"{output_name}_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Save matrix as CSV
    cooccur_df = pd.DataFrame(norm_matrix, index=DISEASE_CLASSES, columns=DISEASE_CLASSES)
    table_path = TABLES_DIR / f"{output_name}_matrix.csv"
    cooccur_df.to_csv(table_path)
    print(f"  Saved: {table_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute all analyses."""
    print("\n" + "="*70)
    print("  THESIS DATASET ANALYSIS: ChestX-ray14")
    print("  Chapter 3: Dataset and Problem Definition")
    print("="*70)

    # Load data
    print("\n[Loading Data]")
    try:
        train_df, test_df, full_df = load_split_data()
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please check paths.")
        print(f"Looking for: {TRAIN_CSV} and {TEST_CSV}")
        return

    # Run all analyses
    print("\n[Running Analyses]")

    # 1. Dataset Overview
    overview_df = generate_dataset_overview(full_df)

    # 2. Disease Frequency
    freq_df = generate_disease_frequency_analysis(full_df)

    # 3. Multi-label Characteristics
    multilabel_df = generate_multilabel_analysis(full_df)

    # 4. Class Imbalance
    imbalance_df = generate_imbalance_analysis(full_df)

    # 5. Split Validation
    split_df, has_leakage = validate_patient_splits(train_df, test_df)

    # 6. Rare Disease Highlight
    rare_df = highlight_rare_diseases(full_df)

    # Bonus analyses
    generate_patient_distribution_figure(full_df)
    generate_disease_cooccurrence_heatmap(full_df)

    # Final summary
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n  Output files saved to:")
    print(f"    Tables: {TABLES_DIR}")
    print(f"    Figures: {FIGURES_DIR}")
    print(f"\n  Files generated:")

    for f in sorted(TABLES_DIR.glob("*.csv")):
        print(f"    - {f.name}")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"    - {f.name}")

    print("\n  All figures are 300 DPI, suitable for thesis publication.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
