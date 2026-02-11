"""
Comprehensive Retrospective Analysis Generator
Extracts all configuration and results data from iterations 1-69
for strategic decision-making and course correction.
"""

import os
import json
import pandas as pd
import yaml
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

class RetrospectiveAnalyzer:
    def __init__(self, output_dir="experiments/auto_improvement_runs"):
        self.output_dir = output_dir
        self.iterations_data = []

    def analyze_all_iterations(self):
        """Extract data from all iterations"""
        print("="*80)
        print("COMPREHENSIVE RETROSPECTIVE ANALYSIS")
        print("="*80)
        print()

        # Find all iteration directories
        iteration_dirs = sorted(glob.glob(f"{self.output_dir}/iteration_*"))
        print(f"Found {len(iteration_dirs)} iteration directories")
        print()

        for iter_dir in iteration_dirs:
            try:
                iter_num = int(os.path.basename(iter_dir).split('_')[1])
                print(f"Processing iteration {iter_num}...", end=" ")

                data = self._extract_iteration_data(iter_dir, iter_num)
                if data:
                    self.iterations_data.append(data)
                    print("✓")
                else:
                    print("⚠️ (no data)")

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        print()
        print(f"✅ Successfully processed {len(self.iterations_data)} iterations")
        print()

    def _extract_iteration_data(self, iter_dir, iter_num):
        """Extract all data from a single iteration"""
        data = {'iteration': iter_num}

        # Load iteration summary
        summary_file = f"{iter_dir}/iteration_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            # Extract metrics
            data['avg_f1'] = summary.get('avg_f1', np.nan)
            data['avg_auc'] = summary.get('avg_auc', np.nan)
            data['avg_recall'] = summary.get('avg_recall', np.nan)
            data['avg_precision'] = summary.get('avg_precision', np.nan)
            data['avg_accuracy'] = summary.get('avg_accuracy', np.nan)
            data['train_loss'] = summary.get('train_loss', np.nan)
            data['val_loss'] = summary.get('val_loss', np.nan)
            data['actual_epochs'] = summary.get('actual_epochs', np.nan)
            data['ai_status'] = summary.get('ai_analysis_status', 'unknown')
            data['status'] = summary.get('status', 'completed')
            data['timestamp'] = summary.get('timestamp', '')

        else:
            # Try to reconstruct from results CSV
            results_files = glob.glob(f"{iter_dir}/pipeline_results_*.csv")
            if results_files:
                results_df = pd.read_csv(results_files[0])
                data['avg_f1'] = results_df['F1_Score'].mean()
                data['avg_auc'] = results_df['AUC'].mean()
                data['avg_recall'] = results_df['Recall'].mean()
                data['avg_precision'] = results_df['Precision'].mean()
                data['avg_accuracy'] = results_df['Accuracy'].mean()
            else:
                return None

        # Load configuration
        config_file = f"{iter_dir}/config.yaml"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Extract loss configuration
            loss_config = config.get('loss', {})
            data['loss_type'] = loss_config.get('type', 'unknown')
            data['loss_gamma'] = loss_config.get('gamma', np.nan)
            data['loss_alpha'] = loss_config.get('alpha', np.nan)
            data['use_class_weights'] = loss_config.get('use_class_weights', False)

            # Extract training configuration
            training_config = config.get('training', {})
            data['learning_rate'] = training_config.get('learning_rate', np.nan)
            data['num_epochs'] = training_config.get('num_epochs', np.nan)
            data['batch_size'] = training_config.get('batch_size', np.nan)
            data['optimizer_type'] = training_config.get('optimizer', {}).get('type', 'unknown')
            data['weight_decay'] = training_config.get('optimizer', {}).get('weight_decay', np.nan)

            # Extract early stopping
            es_config = training_config.get('early_stopping', {})
            data['early_stop_enabled'] = es_config.get('enabled', False)
            data['early_stop_patience'] = es_config.get('patience', np.nan)
            data['early_stop_monitor'] = es_config.get('monitor', 'unknown')
            data['early_stop_mode'] = es_config.get('mode', 'unknown')

            # Extract model configuration
            model_config = config.get('model', {})
            data['model_architecture'] = model_config.get('architecture', 'unknown')
            data['dropout_rate'] = model_config.get('dropout_rate', np.nan)

            # Extract data balancing
            data_config = config.get('data', {})
            data['balancing_method'] = data_config.get('balancing', {}).get('method', 'none')
            data['balancing_ratio'] = data_config.get('balancing', {}).get('ratio', np.nan)

            # Extract augmentation info
            aug_config = config.get('augmentation', {}).get('base', {})
            data['has_augmentation'] = 'additional_transforms' in aug_config

        # Load per-disease results
        results_files = glob.glob(f"{iter_dir}/pipeline_results_*.csv")
        if results_files:
            results_df = pd.read_csv(results_files[0])

            # Add per-disease metrics for key diseases
            for disease in ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumonia']:
                if disease in results_df['Label'].values:
                    disease_row = results_df[results_df['Label'] == disease].iloc[0]
                    data[f'{disease}_f1'] = disease_row.get('F1_Score', np.nan)
                    data[f'{disease}_auc'] = disease_row.get('AUC', np.nan)
                    data[f'{disease}_precision'] = disease_row.get('Precision', np.nan)
                    data[f'{disease}_recall'] = disease_row.get('Recall', np.nan)

        return data

    def generate_reports(self):
        """Generate comprehensive analysis reports"""
        if not self.iterations_data:
            print("No data to analyze!")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.iterations_data)

        print("="*80)
        print("GENERATING REPORTS")
        print("="*80)
        print()

        # 1. Full data CSV
        output_file = "retrospective_analysis_full_data.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ 1. Full Data: {output_file}")
        print(f"   - {len(df)} iterations × {len(df.columns)} features")
        print()

        # 2. Configuration changes timeline
        self._generate_config_changes_report(df)

        # 3. Performance trends analysis
        self._generate_performance_trends(df)

        # 4. Best configurations summary
        self._generate_best_configs(df)

        # 5. Critical insights report
        self._generate_insights_report(df)

        # 6. Disease-specific analysis
        self._generate_disease_analysis(df)

        print()
        print("="*80)
        print("✅ ALL REPORTS GENERATED SUCCESSFULLY")
        print("="*80)

    def _generate_config_changes_report(self, df):
        """Track configuration changes over time"""
        config_cols = [
            'loss_type', 'loss_gamma', 'loss_alpha', 'use_class_weights',
            'learning_rate', 'num_epochs', 'optimizer_type', 'weight_decay',
            'dropout_rate', 'balancing_method', 'early_stop_monitor'
        ]

        changes = []
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]

            iter_changes = {'iteration': int(curr['iteration'])}
            changed = False

            for col in config_cols:
                if col in df.columns:
                    prev_val = prev[col]
                    curr_val = curr[col]

                    # Handle NaN comparison
                    if pd.isna(prev_val) and pd.isna(curr_val):
                        continue

                    if prev_val != curr_val:
                        iter_changes[col] = f"{prev_val} → {curr_val}"
                        changed = True

            if changed:
                changes.append(iter_changes)

        if changes:
            changes_df = pd.DataFrame(changes)
            output_file = "retrospective_config_changes_timeline.csv"
            changes_df.to_csv(output_file, index=False)
            print(f"✅ 2. Config Changes Timeline: {output_file}")
            print(f"   - {len(changes)} configuration changes detected")
        else:
            print(f"⚠️  2. Config Changes: No changes detected")
        print()

    def _generate_performance_trends(self, df):
        """Analyze performance trends over time"""
        output_file = "retrospective_performance_trends.txt"

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PERFORMANCE TRENDS ANALYSIS\n")
            f.write("="*80 + "\n\n")

            # Overall trends
            f.write("1. OVERALL METRICS TRENDS\n")
            f.write("-"*80 + "\n")

            for metric in ['avg_f1', 'avg_auc', 'avg_precision', 'avg_recall']:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        best_iter = df.loc[values.idxmax(), 'iteration']
                        worst_iter = df.loc[values.idxmin(), 'iteration']

                        f.write(f"\n{metric.upper()}:\n")
                        f.write(f"  Best:  Iteration {int(best_iter)} = {values.max():.4f}\n")
                        f.write(f"  Worst: Iteration {int(worst_iter)} = {values.min():.4f}\n")
                        f.write(f"  Range: {values.min():.4f} - {values.max():.4f}\n")
                        f.write(f"  Mean:  {values.mean():.4f}\n")
                        f.write(f"  Std:   {values.std():.4f}\n")

                        # Trend in last 10 iterations
                        if len(values) >= 10:
                            last_10 = values.tail(10)
                            first_10 = values.head(10)
                            f.write(f"  First 10 avg: {first_10.mean():.4f}\n")
                            f.write(f"  Last 10 avg:  {last_10.mean():.4f}\n")
                            f.write(f"  Change: {(last_10.mean() - first_10.mean()):.4f}\n")

            # Identify critical periods
            f.write("\n\n2. CRITICAL PERIODS IDENTIFIED\n")
            f.write("-"*80 + "\n")

            if 'avg_f1' in df.columns:
                f1_values = df['avg_f1'].dropna()

                # Find peak
                peak_idx = f1_values.idxmax()
                peak_iter = int(df.loc[peak_idx, 'iteration'])
                peak_value = f1_values.max()

                f.write(f"\n📈 PEAK PERFORMANCE:\n")
                f.write(f"   Iteration {peak_iter}: F1={peak_value:.4f}\n")

                # Find iteration 57 decline
                if len(df) >= 57:
                    iter_57_idx = df[df['iteration'] == 57].index
                    if len(iter_57_idx) > 0:
                        iter_57_idx = iter_57_idx[0]
                        before_57 = df.loc[max(0, iter_57_idx-5):iter_57_idx-1, 'avg_f1'].mean()
                        after_57 = df.loc[iter_57_idx:min(len(df)-1, iter_57_idx+5), 'avg_f1'].mean()

                        f.write(f"\n⚠️  ITERATION 57 DECLINE:\n")
                        f.write(f"   Before (52-56): F1={before_57:.4f}\n")
                        f.write(f"   After (57-62):  F1={after_57:.4f}\n")
                        f.write(f"   Change: {(after_57 - before_57):.4f} ({((after_57/before_57-1)*100):.1f}%)\n")

                # Find best 10 iterations
                best_10 = df.nlargest(10, 'avg_f1')[['iteration', 'avg_f1', 'avg_auc']]
                f.write(f"\n🏆 TOP 10 ITERATIONS BY F1:\n")
                for idx, row in best_10.iterrows():
                    f.write(f"   Iteration {int(row['iteration'])}: F1={row['avg_f1']:.4f}, AUC={row['avg_auc']:.4f}\n")

            # Loss trends
            f.write("\n\n3. TRAINING DYNAMICS\n")
            f.write("-"*80 + "\n")

            if 'train_loss' in df.columns and 'val_loss' in df.columns:
                train_loss = df['train_loss'].dropna()
                val_loss = df['val_loss'].dropna()

                f.write(f"\nTRAIN LOSS:\n")
                f.write(f"  Min:  {train_loss.min():.6f} (Iteration {int(df.loc[train_loss.idxmin(), 'iteration'])})\n")
                f.write(f"  Max:  {train_loss.max():.6f} (Iteration {int(df.loc[train_loss.idxmax(), 'iteration'])})\n")
                f.write(f"  Mean: {train_loss.mean():.6f}\n")

                f.write(f"\nVAL LOSS:\n")
                f.write(f"  Min:  {val_loss.min():.6f} (Iteration {int(df.loc[val_loss.idxmin(), 'iteration'])})\n")
                f.write(f"  Max:  {val_loss.max():.6f} (Iteration {int(df.loc[val_loss.idxmax(), 'iteration'])})\n")
                f.write(f"  Mean: {val_loss.mean():.6f}\n")

            if 'actual_epochs' in df.columns:
                epochs = df['actual_epochs'].dropna()
                f.write(f"\nEPOCHS TRAINED:\n")
                f.write(f"  Min:  {int(epochs.min())} epochs\n")
                f.write(f"  Max:  {int(epochs.max())} epochs\n")
                f.write(f"  Mean: {epochs.mean():.1f} epochs\n")

        print(f"✅ 3. Performance Trends: {output_file}")
        print()

    def _generate_best_configs(self, df):
        """Identify best-performing configurations"""
        output_file = "retrospective_best_configurations.txt"

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BEST CONFIGURATIONS ANALYSIS\n")
            f.write("="*80 + "\n\n")

            # Find best iterations
            if 'avg_f1' in df.columns:
                best_f1_idx = df['avg_f1'].idxmax()
                best_f1 = df.loc[best_f1_idx]

                f.write("1. BEST F1 CONFIGURATION\n")
                f.write("-"*80 + "\n")
                f.write(f"Iteration: {int(best_f1['iteration'])}\n")
                f.write(f"F1 Score: {best_f1['avg_f1']:.4f}\n")
                f.write(f"AUC: {best_f1.get('avg_auc', np.nan):.4f}\n")
                f.write(f"Precision: {best_f1.get('avg_precision', np.nan):.4f}\n")
                f.write(f"Recall: {best_f1.get('avg_recall', np.nan):.4f}\n\n")
                f.write("Configuration:\n")
                f.write(f"  Loss Type: {best_f1.get('loss_type', 'unknown')}\n")
                f.write(f"  Gamma: {best_f1.get('loss_gamma', np.nan)}\n")
                f.write(f"  Alpha: {best_f1.get('loss_alpha', np.nan)}\n")
                f.write(f"  Learning Rate: {best_f1.get('learning_rate', np.nan)}\n")
                f.write(f"  Dropout: {best_f1.get('dropout_rate', np.nan)}\n")
                f.write(f"  Epochs: {best_f1.get('actual_epochs', np.nan)}\n\n")

            if 'avg_auc' in df.columns:
                best_auc_idx = df['avg_auc'].idxmax()
                best_auc = df.loc[best_auc_idx]

                f.write("\n2. BEST AUC CONFIGURATION\n")
                f.write("-"*80 + "\n")
                f.write(f"Iteration: {int(best_auc['iteration'])}\n")
                f.write(f"AUC: {best_auc['avg_auc']:.4f}\n")
                f.write(f"F1 Score: {best_auc.get('avg_f1', np.nan):.4f}\n")
                f.write(f"Precision: {best_auc.get('avg_precision', np.nan):.4f}\n")
                f.write(f"Recall: {best_auc.get('avg_recall', np.nan):.4f}\n\n")
                f.write("Configuration:\n")
                f.write(f"  Loss Type: {best_auc.get('loss_type', 'unknown')}\n")
                f.write(f"  Gamma: {best_auc.get('loss_gamma', np.nan)}\n")
                f.write(f"  Alpha: {best_auc.get('loss_alpha', np.nan)}\n")
                f.write(f"  Learning Rate: {best_auc.get('learning_rate', np.nan)}\n")
                f.write(f"  Dropout: {best_auc.get('dropout_rate', np.nan)}\n")
                f.write(f"  Epochs: {best_auc.get('actual_epochs', np.nan)}\n\n")

            # Configuration patterns analysis
            f.write("\n3. CONFIGURATION PATTERNS\n")
            f.write("-"*80 + "\n\n")

            # Get top 20% performers
            if 'avg_f1' in df.columns:
                top_20_pct = df.nlargest(int(len(df) * 0.2), 'avg_f1')

                f.write("PATTERNS IN TOP 20% PERFORMERS:\n\n")

                # Loss type distribution
                if 'loss_type' in top_20_pct.columns:
                    loss_dist = top_20_pct['loss_type'].value_counts()
                    f.write("Loss Types:\n")
                    for loss_type, count in loss_dist.items():
                        f.write(f"  {loss_type}: {count} iterations ({count/len(top_20_pct)*100:.1f}%)\n")
                    f.write("\n")

                # Gamma range
                if 'loss_gamma' in top_20_pct.columns:
                    gamma_values = top_20_pct['loss_gamma'].dropna()
                    if len(gamma_values) > 0:
                        f.write("Loss Gamma:\n")
                        f.write(f"  Range: {gamma_values.min():.2f} - {gamma_values.max():.2f}\n")
                        f.write(f"  Mean: {gamma_values.mean():.2f}\n")
                        f.write(f"  Median: {gamma_values.median():.2f}\n\n")

                # Learning rate range
                if 'learning_rate' in top_20_pct.columns:
                    lr_values = top_20_pct['learning_rate'].dropna()
                    if len(lr_values) > 0:
                        f.write("Learning Rate:\n")
                        f.write(f"  Range: {lr_values.min():.6f} - {lr_values.max():.6f}\n")
                        f.write(f"  Mean: {lr_values.mean():.6f}\n")
                        f.write(f"  Median: {lr_values.median():.6f}\n\n")

        print(f"✅ 4. Best Configurations: {output_file}")
        print()

    def _generate_insights_report(self, df):
        """Generate critical insights and recommendations"""
        output_file = "retrospective_critical_insights.txt"

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CRITICAL INSIGHTS & STRATEGIC RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            f.write("📊 EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Iterations Analyzed: {len(df)}\n")
            f.write(f"Iteration Range: {int(df['iteration'].min())} to {int(df['iteration'].max())}\n\n")

            if 'avg_f1' in df.columns:
                f1_values = df['avg_f1'].dropna()
                f.write(f"F1 Score:\n")
                f.write(f"  Best: {f1_values.max():.4f} (Iteration {int(df.loc[f1_values.idxmax(), 'iteration'])})\n")
                f.write(f"  Current: {f1_values.iloc[-1]:.4f} (Iteration {int(df.iloc[-1]['iteration'])})\n")
                f.write(f"  Gap from Best: {(f1_values.max() - f1_values.iloc[-1]):.4f} ({((f1_values.max() - f1_values.iloc[-1])/f1_values.max()*100):.1f}%)\n\n")

            # Key observations
            f.write("\n🔍 KEY OBSERVATIONS\n")
            f.write("-"*80 + "\n\n")

            # Observation 1: Iteration 57 decline
            if len(df) >= 57:
                iter_57 = df[df['iteration'] == 57]
                if len(iter_57) > 0 and 'avg_f1' in df.columns:
                    iter_57_f1 = iter_57['avg_f1'].iloc[0]
                    pre_57 = df[df['iteration'] < 57]['avg_f1'].tail(10).mean()
                    post_57 = df[df['iteration'] >= 57]['avg_f1'].head(10).mean()

                    f.write("1. ITERATION 57 DECLINE:\n")
                    f.write(f"   - F1 dropped from {pre_57:.4f} (avg iters 47-56) to {post_57:.4f} (avg iters 57-66)\n")
                    f.write(f"   - Decline: {(post_57 - pre_57):.4f} ({((post_57/pre_57-1)*100):.1f}%)\n")

                    # Check what changed at iteration 57
                    if len(df) > 57:
                        iter_56 = df[df['iteration'] == 56]
                        if len(iter_56) > 0 and len(iter_57) > 0:
                            f.write("   - Configuration changes at iteration 57:\n")
                            for col in ['loss_type', 'loss_gamma', 'loss_alpha', 'learning_rate', 'dropout_rate']:
                                if col in df.columns:
                                    val_56 = iter_56[col].iloc[0]
                                    val_57 = iter_57[col].iloc[0]
                                    if val_56 != val_57:
                                        f.write(f"     • {col}: {val_56} → {val_57}\n")
                    f.write("\n")

            # Observation 2: Precision in first 30 iterations
            if len(df) >= 30 and 'avg_precision' in df.columns:
                first_30_precision = df.head(30)['avg_precision'].mean()
                last_30_precision = df.tail(30)['avg_precision'].mean()

                f.write("2. PRECISION DEGRADATION:\n")
                f.write(f"   - First 30 iterations avg precision: {first_30_precision:.4f}\n")
                f.write(f"   - Last 30 iterations avg precision: {last_30_precision:.4f}\n")
                f.write(f"   - Change: {(last_30_precision - first_30_precision):.4f} ({((last_30_precision/first_30_precision-1)*100):.1f}%)\n")
                f.write(f"   - Best precision: {df['avg_precision'].max():.4f} (Iteration {int(df.loc[df['avg_precision'].idxmax(), 'iteration'])})\n\n")

            # Observation 3: Configuration drift
            f.write("3. CONFIGURATION DRIFT:\n")
            if 'loss_gamma' in df.columns:
                gamma_first_10 = df.head(10)['loss_gamma'].mean()
                gamma_last_10 = df.tail(10)['loss_gamma'].mean()
                f.write(f"   - Loss gamma: {gamma_first_10:.2f} (first 10) → {gamma_last_10:.2f} (last 10)\n")

            if 'learning_rate' in df.columns:
                lr_first_10 = df.head(10)['learning_rate'].mean()
                lr_last_10 = df.tail(10)['learning_rate'].mean()
                f.write(f"   - Learning rate: {lr_first_10:.6f} (first 10) → {lr_last_10:.6f} (last 10)\n")
            f.write("\n")

            # Strategic recommendations
            f.write("\n💡 STRATEGIC RECOMMENDATIONS\n")
            f.write("-"*80 + "\n\n")

            # Find best period
            if 'avg_f1' in df.columns:
                best_iter = int(df.loc[df['avg_f1'].idxmax(), 'iteration'])

                f.write("1. RETURN TO BEST PERFORMING CONFIGURATION:\n")
                f.write(f"   - Reset to iteration {best_iter} configuration\n")
                f.write(f"   - This achieved F1={df['avg_f1'].max():.4f}\n")
                f.write(f"   - Represents {((df['avg_f1'].max() - df['avg_f1'].iloc[-1])/df['avg_f1'].iloc[-1]*100):.1f}% improvement over current\n\n")

                f.write("2. INVESTIGATE ITERATION 57 CHANGES:\n")
                f.write("   - Something changed that caused sustained decline\n")
                f.write("   - Review configuration changes at iteration 57\n")
                f.write("   - Consider reverting those specific changes\n\n")

                f.write("3. FOCUS ON PRECISION RECOVERY:\n")
                f.write("   - Precision was strongest in early iterations\n")
                f.write("   - Review configurations from iterations 1-30\n")
                f.write("   - Consider more conservative thresholds or class weights\n\n")

                f.write("4. CONSIDER FRESH START:\n")
                f.write("   - 69 iterations of incremental changes may have led to local optimum\n")
                f.write("   - Consider starting from best configuration (iter {best_iter})\n")
                f.write("   - Or try completely different approach (new loss function, architecture)\n\n")

                f.write("5. ANALYZE CONFIGURATION PATTERNS:\n")
                f.write("   - Review 'retrospective_best_configurations.txt'\n")
                f.write("   - Look for patterns in top 20% performers\n")
                f.write("   - Identify successful configuration ranges\n\n")

        print(f"✅ 5. Critical Insights: {output_file}")
        print()

    def _generate_disease_analysis(self, df):
        """Analyze per-disease performance over time"""
        output_file = "retrospective_disease_analysis.csv"

        # Disease columns
        disease_cols = [col for col in df.columns if '_f1' in col or '_auc' in col or '_precision' in col or '_recall' in col]

        if disease_cols:
            disease_df = df[['iteration'] + disease_cols]
            disease_df.to_csv(output_file, index=False)
            print(f"✅ 6. Disease-Specific Analysis: {output_file}")
            print(f"   - Per-disease metrics for {len([c for c in disease_cols if '_f1' in c])} diseases")
        else:
            print(f"⚠️  6. Disease-Specific Analysis: No per-disease data found")
        print()

def main():
    analyzer = RetrospectiveAnalyzer()

    # Extract data from all iterations
    analyzer.analyze_all_iterations()

    # Generate comprehensive reports
    analyzer.generate_reports()

    print()
    print("="*80)
    print("GENERATED FILES:")
    print("="*80)
    print()
    print("📊 DATA FILES:")
    print("  1. retrospective_analysis_full_data.csv")
    print("     → All iterations × all features (use for analysis/visualization)")
    print()
    print("  2. retrospective_config_changes_timeline.csv")
    print("     → Timeline of configuration changes")
    print()
    print("  3. retrospective_disease_analysis.csv")
    print("     → Per-disease metrics over time")
    print()
    print("📝 ANALYSIS REPORTS:")
    print("  4. retrospective_performance_trends.txt")
    print("     → Performance trends, peaks, declines")
    print()
    print("  5. retrospective_best_configurations.txt")
    print("     → Best performing configurations")
    print()
    print("  6. retrospective_critical_insights.txt")
    print("     → Strategic insights and recommendations")
    print()
    print("="*80)
    print("✅ RETROSPECTIVE ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review 'retrospective_critical_insights.txt' for strategic recommendations")
    print("  2. Open 'retrospective_analysis_full_data.csv' in Excel/Python for deeper analysis")
    print("  3. Share these files with your instructor or AI tools for guidance")
    print()

if __name__ == "__main__":
    main()
