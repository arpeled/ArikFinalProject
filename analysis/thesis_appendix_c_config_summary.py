"""
Thesis Appendix C: Experiment Configuration Summary
====================================================
MSc Thesis: "Chest X-ray Disease Identification using Deep Learning"

This script extracts and summarizes experimental configuration parameters
across all iterations, producing a consolidated table for Appendix C.

The script discovers configuration files from:
1. auto_improvement_runs/iteration_XXX/config.yaml
2. config_iteration_XXX.yaml (project root)

Author: [Your Name]
Date: 2026-02
"""

import pandas as pd
import numpy as np
import yaml
import re
from pathlib import Path
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(".")
ITERATIONS_DIR = PROJECT_DIR / "experiments" / "auto_improvement_runs"
OUTPUT_DIR = PROJECT_DIR / "thesis_figures" / "appendix_c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMETER EXTRACTION DEFINITIONS
# =============================================================================

# Define which parameters are methodologically relevant for the thesis
# These are organized by category for clarity
PARAMETER_DEFINITIONS = OrderedDict([
    # Metadata
    ('iteration', 'metadata.iteration'),
    ('phase', 'metadata.phase'),
    ('description', 'metadata.description'),
    ('parent_iteration', 'metadata.parent_iteration'),

    # Loss Function
    ('loss_type', 'loss.type'),
    ('loss_gamma', 'loss.gamma'),
    ('loss_alpha', 'loss.alpha'),
    ('use_class_weights', 'loss.use_class_weights'),

    # Model Architecture
    ('architecture', 'model.architecture'),
    ('backbone', 'model.backbone'),
    ('dropout_rate', 'model.dropout_rate'),
    ('head_type', 'model.head.type'),
    ('head_hidden_features', 'model.head.hidden_features'),
    ('head_dropout', 'model.head.dropout'),

    # Training
    ('batch_size', 'training.batch_size'),
    ('learning_rate', 'training.learning_rate'),
    ('num_epochs', 'training.num_epochs'),
    ('freeze_backbone', 'training.freeze_backbone'),
    ('weight_decay', 'training.optimizer.weight_decay'),

    # Scheduler
    ('scheduler_type', 'training.scheduler.type'),
    ('scheduler_patience', 'training.scheduler.patience'),
    ('scheduler_factor', 'training.scheduler.factor'),
    ('scheduler_mode', 'training.scheduler.mode'),

    # Early Stopping
    ('early_stopping_enabled', 'training.early_stopping.enabled'),
    ('early_stopping_patience', 'training.early_stopping.patience'),
    ('early_stopping_monitor', 'training.early_stopping.monitor'),
    ('early_stopping_min_delta', 'training.early_stopping.min_delta'),

    # Augmentation
    ('rare_class_aug_enabled', 'augmentation.rare_class.enabled'),

    # Threshold Optimization
    ('threshold_optimization', 'evaluation.threshold_optimization'),

    # AUC Improvement (if present)
    ('auc_improvement_enabled', 'auc_improvement.enabled'),
    ('auc_improvement_strategy', 'auc_improvement.strategy'),

    # Sampling (if present)
    ('hernia_oversample_enabled', 'sampling.hernia_oversample.enabled'),
    ('hernia_oversample_factor', 'sampling.hernia_oversample.factor'),
])


# =============================================================================
# CONFIGURATION DISCOVERY
# =============================================================================

def discover_config_files():
    """
    Discover all configuration files in the project.

    Returns:
        List of tuples: (iteration_number, config_path, source)
    """
    print("Discovering configuration files...")

    config_files = []

    # Source 1: Iteration directories in auto_improvement_runs/
    if ITERATIONS_DIR.exists():
        for iter_dir in ITERATIONS_DIR.iterdir():
            if iter_dir.is_dir() and iter_dir.name.startswith('iteration_'):
                config_path = iter_dir / 'config.yaml'
                if config_path.exists():
                    # Extract iteration number from directory name
                    match = re.search(r'iteration_(\d+)', iter_dir.name)
                    if match:
                        iter_num = int(match.group(1))
                        config_files.append((iter_num, config_path, 'iteration_dir'))

    # Source 2: Standalone config files in project root
    for config_path in PROJECT_DIR.glob('config_iteration_*.yaml'):
        # Extract iteration number from filename
        match = re.search(r'config_iteration_(\d+)', config_path.name)
        if match:
            iter_num = int(match.group(1))
            # Only add if not already found in iteration directory
            existing_iters = [cf[0] for cf in config_files]
            if iter_num not in existing_iters:
                config_files.append((iter_num, config_path, 'standalone'))

    # Sort by iteration number
    config_files.sort(key=lambda x: x[0])

    print(f"  Found {len(config_files)} configuration files")
    print(f"    - From iteration directories: {sum(1 for cf in config_files if cf[2] == 'iteration_dir')}")
    print(f"    - Standalone configs: {sum(1 for cf in config_files if cf[2] == 'standalone')}")

    return config_files


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def get_nested_value(config, path, default=None):
    """
    Extract a value from a nested dictionary using dot notation.

    Args:
        config: Dictionary to search
        path: Dot-separated path (e.g., 'loss.type')
        default: Default value if path not found

    Returns:
        Value at path or default
    """
    keys = path.split('.')
    value = config

    try:
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    except (KeyError, TypeError):
        return default


def extract_parameters(config):
    """
    Extract all relevant parameters from a configuration dictionary.

    Args:
        config: Parsed YAML configuration dictionary

    Returns:
        Dictionary of extracted parameters
    """
    extracted = {}

    for param_name, config_path in PARAMETER_DEFINITIONS.items():
        value = get_nested_value(config, config_path)

        # Handle special cases
        if value is None:
            extracted[param_name] = '-'
        elif isinstance(value, bool):
            extracted[param_name] = 'Yes' if value else 'No'
        elif isinstance(value, float):
            # Format floats nicely
            if abs(value) < 0.0001:
                extracted[param_name] = f'{value:.2e}'
            elif abs(value) < 1:
                extracted[param_name] = f'{value:.4f}'
            else:
                extracted[param_name] = f'{value:.2f}'
        elif isinstance(value, list):
            # Convert lists to string representation
            extracted[param_name] = str(value)
        else:
            extracted[param_name] = str(value)

    return extracted


def load_and_extract_config(config_path):
    """
    Load a YAML config file and extract parameters.

    Args:
        config_path: Path to YAML file

    Returns:
        Dictionary of extracted parameters or None if failed
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            return None

        return extract_parameters(config)

    except Exception as e:
        print(f"  Warning: Failed to parse {config_path}: {e}")
        return None


# =============================================================================
# SUMMARY TABLE CONSTRUCTION
# =============================================================================

def build_configuration_summary(config_files):
    """
    Build the consolidated configuration summary table.

    Args:
        config_files: List of (iteration_num, config_path, source) tuples

    Returns:
        pandas DataFrame with configuration summary
    """
    print("\nExtracting parameters from configurations...")

    records = []

    for iter_num, config_path, source in config_files:
        params = load_and_extract_config(config_path)

        if params is not None:
            # Add source information
            params['config_source'] = source
            params['config_file'] = str(config_path.name)

            # Ensure iteration number is set (fallback to filename-derived)
            if params.get('iteration', '-') == '-':
                params['iteration'] = str(iter_num)

            records.append(params)

    print(f"  Successfully extracted parameters from {len(records)} configurations")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Reorder columns for readability
    column_order = ['iteration', 'phase', 'config_source'] + \
                   [col for col in PARAMETER_DEFINITIONS.keys() if col not in ['iteration', 'phase']] + \
                   ['config_file', 'description']

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Sort by iteration number
    df['_iter_sort'] = pd.to_numeric(df['iteration'], errors='coerce')
    df = df.sort_values('_iter_sort').drop(columns=['_iter_sort'])
    df = df.reset_index(drop=True)

    return df


def identify_varying_parameters(df):
    """
    Identify which parameters actually varied across iterations.

    Args:
        df: Configuration summary DataFrame

    Returns:
        Tuple of (varying_columns, constant_columns)
    """
    varying = []
    constant = []

    for col in df.columns:
        if col in ['iteration', 'phase', 'config_source', 'config_file', 'description']:
            continue

        unique_values = df[col].dropna().unique()
        # Consider '-' as a missing value for variation detection
        unique_values = [v for v in unique_values if v != '-']

        if len(unique_values) > 1:
            varying.append(col)
        else:
            constant.append(col)

    return varying, constant


def create_compact_summary(df, varying_cols):
    """
    Create a compact version of the summary focusing on varying parameters.

    Args:
        df: Full configuration summary DataFrame
        varying_cols: List of columns that vary across iterations

    Returns:
        Compact DataFrame
    """
    # Always include these columns
    key_cols = ['iteration', 'phase']

    # Add varying columns
    cols_to_include = key_cols + [c for c in varying_cols if c in df.columns]

    compact_df = df[cols_to_include].copy()

    return compact_df


# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def generate_parameter_change_report(df, varying_cols):
    """
    Generate a textual report of parameter changes across iterations.

    Args:
        df: Configuration summary DataFrame
        varying_cols: List of varying parameter names

    Returns:
        String report
    """
    report_lines = [
        "="*70,
        "PARAMETER VARIATION ANALYSIS",
        "="*70,
        "",
        f"Total configurations analyzed: {len(df)}",
        f"Parameters that varied: {len(varying_cols)}",
        "",
        "-"*50,
        "VARYING PARAMETERS:",
        "-"*50,
    ]

    for col in varying_cols:
        unique_values = df[col].value_counts()
        report_lines.append(f"\n{col}:")
        for val, count in unique_values.items():
            report_lines.append(f"    {val}: {count} iterations")

    return "\n".join(report_lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate the experiment configuration summary."""
    print("\n" + "="*70)
    print("  THESIS APPENDIX C: EXPERIMENT CONFIGURATION SUMMARY")
    print("  Extracting Configuration Parameters Across Iterations")
    print("="*70)

    # Step 1: Discover configuration files
    config_files = discover_config_files()

    if not config_files:
        print("ERROR: No configuration files found!")
        return

    # Step 2: Build configuration summary table
    experiment_config_summary = build_configuration_summary(config_files)

    # Step 3: Identify varying parameters
    varying_cols, constant_cols = identify_varying_parameters(experiment_config_summary)

    print(f"\n  Parameter analysis:")
    print(f"    - Varying parameters: {len(varying_cols)}")
    print(f"    - Constant parameters: {len(constant_cols)}")

    # Step 4: Generate reports
    print("\n" + "-"*50)
    print("SAVING OUTPUTS")
    print("-"*50)

    # Save full summary
    full_output_path = OUTPUT_DIR / "appendix_table_c1_experiment_configuration_summary.csv"
    experiment_config_summary.to_csv(full_output_path, index=False)
    print(f"  Saved: {full_output_path}")

    # Save compact summary (varying parameters only)
    compact_df = create_compact_summary(experiment_config_summary, varying_cols)
    compact_output_path = OUTPUT_DIR / "appendix_table_c2_varying_parameters_summary.csv"
    compact_df.to_csv(compact_output_path, index=False)
    print(f"  Saved: {compact_output_path}")

    # Save constant parameters reference
    if constant_cols:
        constant_values = {}
        for col in constant_cols:
            vals = experiment_config_summary[col].dropna().unique()
            vals = [v for v in vals if v != '-']
            if vals:
                constant_values[col] = vals[0]

        constant_df = pd.DataFrame([
            {'Parameter': k, 'Value': v} for k, v in constant_values.items()
        ])
        constant_output_path = OUTPUT_DIR / "appendix_table_c3_constant_parameters.csv"
        constant_df.to_csv(constant_output_path, index=False)
        print(f"  Saved: {constant_output_path}")

    # Generate and print variation report
    report = generate_parameter_change_report(experiment_config_summary, varying_cols)
    print("\n" + report)

    # Save report as text file
    report_path = OUTPUT_DIR / "appendix_c_parameter_variation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {report_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("  APPENDIX C GENERATION COMPLETE")
    print("="*70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"\n  Summary:")
    print(f"    - Iterations analyzed: {len(experiment_config_summary)}")
    print(f"    - Varying parameters: {len(varying_cols)}")
    print(f"    - Constant parameters: {len(constant_cols)}")

    print("\n  Key varying parameters:")
    for col in varying_cols[:10]:  # Show top 10
        print(f"    - {col}")
    if len(varying_cols) > 10:
        print(f"    ... and {len(varying_cols) - 10} more")

    print("\n  Generated files:")
    for f in sorted(OUTPUT_DIR.glob("appendix_*")):
        print(f"    - {f.name}")

    print("="*70 + "\n")

    return experiment_config_summary


if __name__ == "__main__":
    experiment_config_summary = main()
