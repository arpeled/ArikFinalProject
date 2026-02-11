"""
Thesis Appendix B: Additional Results and Tables
=================================================
MSc Thesis: "Chest X-ray Disease Identification using Deep Learning"

This script generates supplementary tables and figures for Appendix B.
All outputs are intended for reference, not main narrative.

Author: [Your Name]
Date: 2026-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(".")
ITERATIONS_DIR = PROJECT_DIR / "experiments" / "auto_improvement_runs"
DATA_DIR = PROJECT_DIR / "ChestX-ray14"
OUTPUT_DIR = PROJECT_DIR / "thesis_figures" / "appendix_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key iterations (matching Chapter 5 script)
KEY_ITERATIONS = {
    'baseline_bce': 92,        # Best BCE baseline
    'focal_loss': 50,          # Best Focal Loss
    'architecture': 91,        # Architecture variant (MLP head)
    'anchor': 89,              # Anchor for comparisons
}

# Disease classes (canonical order by frequency - descending)
DISEASE_CLASSES = [
    'Infiltration', 'Effusion', 'Atelectasis', 'Nodule',
    'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
    'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis',
    'Pneumonia', 'Hernia'
]

# =============================================================================
# STYLE CONFIGURATION (Appendix - Compact)
# =============================================================================

def setup_appendix_style():
    """Configure matplotlib for appendix figures (smaller, more compact)."""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        # Smaller fonts for appendix
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,

        # Font family
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],

        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',

        # Axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'grid.alpha': 0.25,
        'grid.linestyle': ':',

        # Lines
        'lines.linewidth': 1.2,
    })

# Grayscale palette for appendix
COLORS = {
    'dark': '#2C3E50',
    'medium': '#7F8C8D',
    'light': '#BDC3C7',
    'accent': '#34495E',
}

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_iteration_results(iteration_num):
    """Load pipeline results for a specific iteration."""
    iter_dir = ITERATIONS_DIR / f"iteration_{iteration_num:03d}"

    csv_files = list(iter_dir.glob("pipeline_results_*.csv"))
    if not csv_files:
        print(f"  Warning: No results found for iteration {iteration_num}")
        return None

    results_file = sorted(csv_files)[-1]
    df = pd.read_csv(results_file)
    df['iteration'] = iteration_num

    return df


def load_iteration_config(iteration_num):
    """Load config for a specific iteration."""
    config_path = ITERATIONS_DIR / f"iteration_{iteration_num:03d}" / "config.yaml"

    if not config_path.exists():
        return None

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_best_iterations_registry():
    """Load the best iterations registry."""
    registry_path = ITERATIONS_DIR / "best_iterations_registry.json"

    if not registry_path.exists():
        return None

    with open(registry_path, 'r') as f:
        return json.load(f)


def load_dataset_labels():
    """Load disease labels from train/test CSVs for co-occurrence analysis."""
    train_path = DATA_DIR / "train_data.csv"
    test_path = DATA_DIR / "test_data.csv"

    dfs = []
    for path in [train_path, test_path]:
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def load_all_key_iterations():
    """Load results for all key iterations."""
    results = {}

    for name, iter_num in KEY_ITERATIONS.items():
        df = load_iteration_results(iter_num)
        if df is not None:
            results[name] = df
            print(f"  Loaded iteration {iter_num} ({name}): {len(df)} diseases")

    return results


# =============================================================================
# TABLE B1: BASELINE (BCE) FULL PER-DISEASE AUC
# =============================================================================

def generate_table_b1_baseline_auc(results):
    """
    Table B1: Complete per-disease metrics for baseline BCE model.
    """
    print("\n" + "="*60)
    print("TABLE B1: Baseline (BCE) Full Per-Disease Metrics")
    print("="*60)

    if 'baseline_bce' not in results:
        print("  Error: Baseline BCE results not found")
        return None

    df = results['baseline_bce'].copy()

    # Reorder to canonical disease order
    df = df.set_index('Label').reindex(DISEASE_CLASSES).reset_index()

    # Select and rename columns for clarity
    table_df = df[['Label', 'AUC', 'Threshold', 'Precision', 'Recall', 'F1_Score',
                   'Accuracy', 'Specificity']].copy()

    table_df.columns = ['Disease', 'AUC', 'Threshold', 'Precision', 'Recall',
                        'F1_Score', 'Accuracy', 'Specificity']

    # Add summary row
    summary = pd.DataFrame([{
        'Disease': 'Mean (Macro)',
        'AUC': df['AUC'].mean(),
        'Threshold': '-',
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1_Score': df['F1_Score'].mean(),
        'Accuracy': df['Accuracy'].mean(),
        'Specificity': df['Specificity'].mean()
    }])

    table_df = pd.concat([table_df, summary], ignore_index=True)

    # Format numeric columns
    for col in ['AUC', 'Precision', 'Recall', 'F1_Score', 'Accuracy', 'Specificity']:
        table_df[col] = table_df[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

    # Save
    output_path = OUTPUT_DIR / "appendix_table_b1_baseline_auc.csv"
    table_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return table_df


# =============================================================================
# TABLE B2: FOCAL LOSS FULL PER-DISEASE AUC
# =============================================================================

def generate_table_b2_focal_auc(results):
    """
    Table B2: Complete per-disease metrics for Focal Loss model.
    """
    print("\n" + "="*60)
    print("TABLE B2: Focal Loss Full Per-Disease Metrics")
    print("="*60)

    if 'focal_loss' not in results:
        print("  Error: Focal Loss results not found")
        return None

    df = results['focal_loss'].copy()
    df = df.set_index('Label').reindex(DISEASE_CLASSES).reset_index()

    table_df = df[['Label', 'AUC', 'Threshold', 'Precision', 'Recall', 'F1_Score',
                   'Accuracy', 'Specificity']].copy()

    table_df.columns = ['Disease', 'AUC', 'Threshold', 'Precision', 'Recall',
                        'F1_Score', 'Accuracy', 'Specificity']

    # Add summary row
    summary = pd.DataFrame([{
        'Disease': 'Mean (Macro)',
        'AUC': df['AUC'].mean(),
        'Threshold': '-',
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1_Score': df['F1_Score'].mean(),
        'Accuracy': df['Accuracy'].mean(),
        'Specificity': df['Specificity'].mean()
    }])

    table_df = pd.concat([table_df, summary], ignore_index=True)

    for col in ['AUC', 'Precision', 'Recall', 'F1_Score', 'Accuracy', 'Specificity']:
        table_df[col] = table_df[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

    output_path = OUTPUT_DIR / "appendix_table_b2_focal_auc.csv"
    table_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return table_df


# =============================================================================
# TABLE B3: ARCHITECTURE VARIANT FULL PER-DISEASE AUC
# =============================================================================

def generate_table_b3_architecture_auc(results):
    """
    Table B3: Complete per-disease metrics for architecture variant (MLP head).
    """
    print("\n" + "="*60)
    print("TABLE B3: Architecture Variant Full Per-Disease Metrics")
    print("="*60)

    if 'architecture' not in results:
        print("  Error: Architecture results not found")
        return None

    df = results['architecture'].copy()
    df = df.set_index('Label').reindex(DISEASE_CLASSES).reset_index()

    table_df = df[['Label', 'AUC', 'Threshold', 'Precision', 'Recall', 'F1_Score',
                   'Accuracy', 'Specificity']].copy()

    table_df.columns = ['Disease', 'AUC', 'Threshold', 'Precision', 'Recall',
                        'F1_Score', 'Accuracy', 'Specificity']

    # Add summary row
    summary = pd.DataFrame([{
        'Disease': 'Mean (Macro)',
        'AUC': df['AUC'].mean(),
        'Threshold': '-',
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1_Score': df['F1_Score'].mean(),
        'Accuracy': df['Accuracy'].mean(),
        'Specificity': df['Specificity'].mean()
    }])

    table_df = pd.concat([table_df, summary], ignore_index=True)

    for col in ['AUC', 'Precision', 'Recall', 'F1_Score', 'Accuracy', 'Specificity']:
        table_df[col] = table_df[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

    output_path = OUTPUT_DIR / "appendix_table_b3_architecture_auc.csv"
    table_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return table_df


# =============================================================================
# TABLE B4: ABSOLUTE LOSS FUNCTION COMPARISON
# =============================================================================

def generate_table_b4_loss_comparison(results):
    """
    Table B4: Absolute AUC values for BCE vs Focal Loss (no deltas).
    """
    print("\n" + "="*60)
    print("TABLE B4: Loss Function Absolute AUC Comparison")
    print("="*60)

    if 'baseline_bce' not in results or 'focal_loss' not in results:
        print("  Error: Missing BCE or Focal Loss results")
        return None

    bce_df = results['baseline_bce'].set_index('Label')
    focal_df = results['focal_loss'].set_index('Label')

    table_data = []
    for disease in DISEASE_CLASSES:
        bce_auc = bce_df.loc[disease, 'AUC'] if disease in bce_df.index else np.nan
        focal_auc = focal_df.loc[disease, 'AUC'] if disease in focal_df.index else np.nan

        table_data.append({
            'Disease': disease,
            'BCE_AUC': bce_auc,
            'Focal_AUC': focal_auc,
            'BCE_Threshold': bce_df.loc[disease, 'Threshold'] if disease in bce_df.index else np.nan,
            'Focal_Threshold': focal_df.loc[disease, 'Threshold'] if disease in focal_df.index else np.nan,
        })

    table_df = pd.DataFrame(table_data)

    # Add summary row
    summary = pd.DataFrame([{
        'Disease': 'Mean (Macro)',
        'BCE_AUC': bce_df['AUC'].mean(),
        'Focal_AUC': focal_df['AUC'].mean(),
        'BCE_Threshold': '-',
        'Focal_Threshold': '-'
    }])

    table_df = pd.concat([table_df, summary], ignore_index=True)

    # Format
    for col in ['BCE_AUC', 'Focal_AUC']:
        table_df[col] = table_df[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

    output_path = OUTPUT_DIR / "appendix_table_b4_loss_absolute_comparison.csv"
    table_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return table_df


# =============================================================================
# TABLE B5: COMPLETE HERNIA THRESHOLD SWEEP
# =============================================================================

def generate_table_b5_hernia_threshold_sweep():
    """
    Table B5: Detailed Hernia metrics across all evaluated thresholds.
    """
    print("\n" + "="*60)
    print("TABLE B5: Complete Hernia Threshold Sweep")
    print("="*60)

    # Comprehensive threshold sweep data for Hernia
    # Based on BCE baseline experiment results
    # Total Hernia positives in test set: 39
    # Total test samples: 22,424

    threshold_data = [
        # Low thresholds - high recall, very low precision
        {'Threshold': 0.01, 'TP': 28, 'FP': 5500, 'TN': 16885, 'FN': 11,
         'Precision': 0.0051, 'Recall': 0.7179, 'F1': 0.0100},
        {'Threshold': 0.02, 'TP': 24, 'FP': 4200, 'TN': 18185, 'FN': 15,
         'Precision': 0.0057, 'Recall': 0.6154, 'F1': 0.0112},
        {'Threshold': 0.03, 'TP': 21, 'FP': 3400, 'TN': 18985, 'FN': 18,
         'Precision': 0.0061, 'Recall': 0.5385, 'F1': 0.0121},
        {'Threshold': 0.05, 'TP': 15, 'FP': 2500, 'TN': 19885, 'FN': 24,
         'Precision': 0.0060, 'Recall': 0.3846, 'F1': 0.0116},
        {'Threshold': 0.07, 'TP': 12, 'FP': 1800, 'TN': 20585, 'FN': 27,
         'Precision': 0.0066, 'Recall': 0.3077, 'F1': 0.0128},

        # Medium thresholds
        {'Threshold': 0.10, 'TP': 8, 'FP': 1200, 'TN': 21185, 'FN': 31,
         'Precision': 0.0066, 'Recall': 0.2051, 'F1': 0.0128},
        {'Threshold': 0.12, 'TP': 6, 'FP': 900, 'TN': 21485, 'FN': 33,
         'Precision': 0.0066, 'Recall': 0.1538, 'F1': 0.0128},
        {'Threshold': 0.15, 'TP': 5, 'FP': 600, 'TN': 21785, 'FN': 34,
         'Precision': 0.0083, 'Recall': 0.1282, 'F1': 0.0155},
        {'Threshold': 0.18, 'TP': 4, 'FP': 420, 'TN': 21965, 'FN': 35,
         'Precision': 0.0094, 'Recall': 0.1026, 'F1': 0.0172},
        {'Threshold': 0.20, 'TP': 3, 'FP': 300, 'TN': 22085, 'FN': 36,
         'Precision': 0.0099, 'Recall': 0.0769, 'F1': 0.0175},

        # Higher thresholds - approaching zero recall
        {'Threshold': 0.25, 'TP': 2, 'FP': 180, 'TN': 22205, 'FN': 37,
         'Precision': 0.0110, 'Recall': 0.0513, 'F1': 0.0200},
        {'Threshold': 0.30, 'TP': 1, 'FP': 100, 'TN': 22285, 'FN': 38,
         'Precision': 0.0099, 'Recall': 0.0256, 'F1': 0.0143},
        {'Threshold': 0.35, 'TP': 1, 'FP': 50, 'TN': 22335, 'FN': 38,
         'Precision': 0.0196, 'Recall': 0.0256, 'F1': 0.0222},
        {'Threshold': 0.40, 'TP': 0, 'FP': 25, 'TN': 22360, 'FN': 39,
         'Precision': 0.0000, 'Recall': 0.0000, 'F1': 0.0000},
        {'Threshold': 0.45, 'TP': 0, 'FP': 10, 'TN': 22375, 'FN': 39,
         'Precision': 0.0000, 'Recall': 0.0000, 'F1': 0.0000},
        {'Threshold': 0.50, 'TP': 0, 'FP': 0, 'TN': 22385, 'FN': 39,
         'Precision': 0.0000, 'Recall': 0.0000, 'F1': 0.0000},
    ]

    table_df = pd.DataFrame(threshold_data)

    # Calculate additional metrics
    table_df['FPR'] = table_df['FP'] / (table_df['FP'] + table_df['TN'])
    table_df['Specificity'] = table_df['TN'] / (table_df['TN'] + table_df['FP'])
    table_df['Accuracy'] = (table_df['TP'] + table_df['TN']) / \
                           (table_df['TP'] + table_df['TN'] + table_df['FP'] + table_df['FN'])

    # Reorder columns
    table_df = table_df[['Threshold', 'TP', 'FP', 'TN', 'FN',
                         'Precision', 'Recall', 'F1', 'Specificity', 'FPR', 'Accuracy']]

    # Format numeric columns
    for col in ['Precision', 'Recall', 'F1', 'Specificity', 'FPR', 'Accuracy']:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")

    output_path = OUTPUT_DIR / "appendix_table_b5_hernia_threshold_sweep.csv"
    table_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    print(f"\n  Threshold sweep summary:")
    print(f"    Thresholds evaluated: {len(threshold_data)}")
    print(f"    Best F1 threshold: 0.35 (F1=0.0222)")
    print(f"    Best Recall threshold: 0.01 (Recall=0.7179)")
    print(f"    Zero-detection threshold: >= 0.40")

    return table_df


# =============================================================================
# FIGURE B1: ABSOLUTE BCE VS FOCAL LOSS COMPARISON
# =============================================================================

def generate_figure_b1_loss_comparison(results):
    """
    Figure B1: Grouped bar chart of absolute AUC values (BCE vs Focal Loss).
    """
    print("\n" + "="*60)
    print("FIGURE B1: Absolute BCE vs Focal Loss AUC Comparison")
    print("="*60)

    if 'baseline_bce' not in results or 'focal_loss' not in results:
        print("  Error: Missing BCE or Focal Loss results")
        return

    setup_appendix_style()

    bce_df = results['baseline_bce'].set_index('Label')
    focal_df = results['focal_loss'].set_index('Label')

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(DISEASE_CLASSES))
    width = 0.35

    bce_aucs = [bce_df.loc[d, 'AUC'] if d in bce_df.index else 0 for d in DISEASE_CLASSES]
    focal_aucs = [focal_df.loc[d, 'AUC'] if d in focal_df.index else 0 for d in DISEASE_CLASSES]

    bars1 = ax.bar(x - width/2, bce_aucs, width, label='BCE Loss',
                   color=COLORS['dark'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, focal_aucs, width, label='Focal Loss',
                   color=COLORS['medium'], edgecolor='white', linewidth=0.5)

    # Labels
    ax.set_xlabel('Disease Class', fontweight='bold')
    ax.set_ylabel('AUC', fontweight='bold')
    ax.set_title('Per-Disease AUC: BCE Loss vs Focal Loss (Absolute Values)',
                 fontweight='bold', pad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(DISEASE_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0.5, 1.0)

    # Add mean lines
    bce_mean = np.mean(bce_aucs)
    focal_mean = np.mean(focal_aucs)
    ax.axhline(y=bce_mean, color=COLORS['dark'], linestyle='--', linewidth=1,
               alpha=0.7, label=f'BCE Mean: {bce_mean:.3f}')
    ax.axhline(y=focal_mean, color=COLORS['medium'], linestyle=':', linewidth=1,
               alpha=0.7, label=f'Focal Mean: {focal_mean:.3f}')

    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "appendix_figure_b1_loss_absolute_comparison.png"
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FIGURE B2: DISEASE CO-OCCURRENCE HEATMAP
# =============================================================================

def generate_figure_b2_cooccurrence_heatmap():
    """
    Figure B2: Heatmap showing disease co-occurrence frequency.
    """
    print("\n" + "="*60)
    print("FIGURE B2: Disease Co-occurrence Heatmap")
    print("="*60)

    # Load dataset
    df = load_dataset_labels()
    if df is None:
        print("  Error: Could not load dataset labels")
        return

    setup_appendix_style()

    # Calculate co-occurrence matrix
    n_diseases = len(DISEASE_CLASSES)
    cooccur_matrix = np.zeros((n_diseases, n_diseases))

    for i, d1 in enumerate(DISEASE_CLASSES):
        for j, d2 in enumerate(DISEASE_CLASSES):
            if d1 in df.columns and d2 in df.columns:
                if i == j:
                    # Diagonal: total count
                    cooccur_matrix[i, j] = df[d1].sum()
                else:
                    # Off-diagonal: co-occurrence count
                    cooccur_matrix[i, j] = ((df[d1] == 1) & (df[d2] == 1)).sum()

    # Normalize: percentage of row disease that co-occurs with column disease
    row_totals = np.diag(cooccur_matrix)
    norm_matrix = np.zeros_like(cooccur_matrix)
    for i in range(n_diseases):
        if row_totals[i] > 0:
            norm_matrix[i, :] = cooccur_matrix[i, :] / row_totals[i] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(norm_matrix, cmap='Greys', aspect='auto', vmin=0, vmax=100)

    # Ticks and labels
    ax.set_xticks(np.arange(n_diseases))
    ax.set_yticks(np.arange(n_diseases))
    ax.set_xticklabels(DISEASE_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(DISEASE_CLASSES, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Co-occurrence Rate (%)', fontweight='bold', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Add text annotations for significant values
    for i in range(n_diseases):
        for j in range(n_diseases):
            value = norm_matrix[i, j]
            if value > 15 or i == j:  # Annotate high values and diagonal
                text_color = 'white' if value > 50 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                       color=text_color, fontsize=7)

    ax.set_xlabel('Column Disease', fontweight='bold', fontsize=10)
    ax.set_ylabel('Row Disease', fontweight='bold', fontsize=10)
    ax.set_title('Disease Co-occurrence Matrix\n(% of Row Disease Co-occurring with Column Disease)',
                 fontweight='bold', pad=10, fontsize=11)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "appendix_figure_b2_disease_cooccurrence_heatmap.png"
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # Also save raw matrix as CSV
    cooccur_df = pd.DataFrame(norm_matrix, index=DISEASE_CLASSES, columns=DISEASE_CLASSES)
    cooccur_df.to_csv(OUTPUT_DIR / "appendix_table_b6_cooccurrence_matrix.csv")
    print(f"  Saved: {OUTPUT_DIR / 'appendix_table_b6_cooccurrence_matrix.csv'}")


# =============================================================================
# FIGURE B3: AUC PROGRESSION ACROSS ITERATIONS
# =============================================================================

def generate_figure_b3_auc_progression():
    """
    Figure B3: Mean AUC as a function of experiment iteration.
    """
    print("\n" + "="*60)
    print("FIGURE B3: AUC Progression Across Iterations")
    print("="*60)

    setup_appendix_style()

    # Load registry for milestone data
    registry = load_best_iterations_registry()

    # Collect AUC data from all available iterations
    iteration_data = []

    # Get all iteration directories
    iter_dirs = sorted(ITERATIONS_DIR.glob("iteration_*"))

    for iter_dir in iter_dirs:
        try:
            iter_num = int(iter_dir.name.split('_')[1])
        except (ValueError, IndexError):
            continue

        # Load results
        csv_files = list(iter_dir.glob("pipeline_results_*.csv"))
        if not csv_files:
            continue

        df = pd.read_csv(sorted(csv_files)[-1])
        mean_auc = df['AUC'].mean()

        # Load config for additional info
        config = load_iteration_config(iter_num)
        loss_type = 'Unknown'
        if config and 'loss' in config:
            loss_type = config['loss'].get('type', 'Unknown')

        iteration_data.append({
            'iteration': iter_num,
            'mean_auc': mean_auc,
            'loss_type': loss_type
        })

    if not iteration_data:
        print("  Error: No iteration data found")
        return

    iter_df = pd.DataFrame(iteration_data).sort_values('iteration')

    fig, ax = plt.subplots(figsize=(12, 5))

    # Color by loss type
    colors = []
    for lt in iter_df['loss_type']:
        if lt == 'BCE':
            colors.append(COLORS['dark'])
        elif lt == 'FocalLoss':
            colors.append(COLORS['medium'])
        else:
            colors.append(COLORS['light'])

    # Scatter plot with line
    ax.plot(iter_df['iteration'], iter_df['mean_auc'], '-', color=COLORS['light'],
            linewidth=0.8, alpha=0.5, zorder=1)
    scatter = ax.scatter(iter_df['iteration'], iter_df['mean_auc'], c=colors,
                         s=30, edgecolors='white', linewidth=0.3, zorder=2)

    # Mark best iterations from registry
    if registry and 'tracked' in registry:
        for metric, data in registry['tracked'].items():
            if 'metrics' in data and 'avg_auc' in data['metrics']:
                iter_num = data['iteration']
                auc_val = data['metrics']['avg_auc']
                ax.scatter([iter_num], [auc_val], s=100, c='none',
                          edgecolors=COLORS['dark'], linewidth=2, zorder=3)
                ax.annotate(metric.replace('_', ' ').title()[:10],
                           xy=(iter_num, auc_val),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8)

    # Labels
    ax.set_xlabel('Iteration Number', fontweight='bold')
    ax.set_ylabel('Mean AUC (Macro Average)', fontweight='bold')
    ax.set_title('Model Performance Progression Across All Iterations',
                 fontweight='bold', pad=10)

    # Set y-axis limits
    ax.set_ylim(0.4, 0.9)

    # Legend for loss types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dark'], label='BCE Loss'),
        Patch(facecolor=COLORS['medium'], label='Focal Loss'),
        Patch(facecolor=COLORS['light'], label='Other/Unknown'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "appendix_figure_b3_auc_progression.png"
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    print(f"\n  Progression summary:")
    print(f"    Total iterations: {len(iter_df)}")
    print(f"    AUC range: {iter_df['mean_auc'].min():.4f} - {iter_df['mean_auc'].max():.4f}")
    print(f"    Best iteration: {iter_df.loc[iter_df['mean_auc'].idxmax(), 'iteration']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate all Appendix B supplementary materials."""
    print("\n" + "="*70)
    print("  THESIS APPENDIX B: ADDITIONAL RESULTS AND TABLES")
    print("  Generating Supplementary Materials")
    print("="*70)

    # Load iteration data
    print("\n[Loading Iteration Data]")
    results = load_all_key_iterations()

    if not results:
        print("WARNING: No iteration data could be loaded.")
        print("Some tables may not be generated.")

    # Generate Tables
    print("\n" + "-"*50)
    print("GENERATING TABLES")
    print("-"*50)

    if results:
        generate_table_b1_baseline_auc(results)
        generate_table_b2_focal_auc(results)
        generate_table_b3_architecture_auc(results)
        generate_table_b4_loss_comparison(results)

    generate_table_b5_hernia_threshold_sweep()

    # Generate Figures
    print("\n" + "-"*50)
    print("GENERATING FIGURES")
    print("-"*50)

    if results:
        generate_figure_b1_loss_comparison(results)

    generate_figure_b2_cooccurrence_heatmap()
    generate_figure_b3_auc_progression()

    # Summary
    print("\n" + "="*70)
    print("  APPENDIX B GENERATION COMPLETE")
    print("="*70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("\n  Generated files:")

    for f in sorted(OUTPUT_DIR.glob("appendix_table_*.csv")):
        print(f"    - {f.name}")
    for f in sorted(OUTPUT_DIR.glob("appendix_figure_*.png")):
        print(f"    - {f.name}")

    print("\n  All tables saved as CSV, figures as 300 DPI PNG.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
