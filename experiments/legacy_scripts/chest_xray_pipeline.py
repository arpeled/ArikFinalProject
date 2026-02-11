#!/usr/bin/env python3
"""
Chest X-Ray Classification Pipeline
Runs training, testing, and comparison with baseline results in sequence.
"""
import logging
import os
import sys
import time
import datetime
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# Wang et al. baseline results from ChestX-ray8 paper
BASELINE_RESULTS = {
    'Cardiomegaly': {'AUC': 0.919101, 'Threshold': 0.489469, 'Sensitivity': 0.865, 'Specificity': 0.819083, 'Accuracy': 0.820225, 'Precision': 0.112312, 'Recall': 0.863322, 'F1_Score': 0.198765},
    'Emphysema': {'AUC': 0.916227, 'Threshold': 0.445399, 'Sensitivity': 0.866, 'Specificity': 0.819995, 'Accuracy': 0.82103, 'Precision': 0.103055, 'Recall': 0.864245, 'F1_Score': 0.184152},
    'Effusion': {'AUC': 0.888239, 'Threshold': 0.484694, 'Sensitivity': 0.839, 'Specificity': 0.794271, 'Accuracy': 0.799491, 'Precision': 0.354027, 'Recall': 0.838296, 'F1_Score': 0.497818},
    'Hernia': {'AUC': 0.82202, 'Threshold': 0.581222, 'Sensitivity': 0.717, 'Specificity': 0.802346, 'Accuracy': 0.802127, 'Precision': 0.00719748, 'Recall': 0.695652, 'F1_Score': 0.0142476},
    'Infiltration': {'AUC': 0.715871, 'Threshold': 0.476786, 'Sensitivity': 0.666, 'Specificity': 0.645026, 'Accuracy': 0.648673, 'Precision': 0.288003, 'Recall': 0.665575, 'F1_Score': 0.402838},
    'Mass': {'AUC': 0.869582, 'Threshold': 0.585565, 'Sensitivity': 0.758, 'Specificity': 0.825244, 'Accuracy': 0.821655, 'Precision': 0.193864, 'Recall': 0.757089, 'F1_Score': 0.308678},
    'Nodule': {'AUC': 0.792307, 'Threshold': 0.485297, 'Sensitivity': 0.700, 'Specificity': 0.737614, 'Accuracy': 0.735723, 'Precision': 0.136427, 'Recall': 0.703614, 'F1_Score': 0.228502},
    'Atelectasis': {'AUC': 0.818337, 'Threshold': 0.458412, 'Sensitivity': 0.8, 'Specificity': 0.682717, 'Accuracy': 0.694521, 'Precision': 0.2282, 'Recall': 0.799911, 'F1_Score': 0.345336},
    'Pneumothorax': {'AUC': 0.886252, 'Threshold': 0.465026, 'Sensitivity': 0.831, 'Specificity': 0.787684, 'Accuracy': 0.789749, 'Precision': 0.166667, 'Recall': 0.830119, 'F1_Score': 0.277599},
    'Pleural_Thickening': {'AUC': 0.812256, 'Threshold': 0.441709, 'Sensitivity': 0.83, 'Specificity': 0.657618, 'Accuracy': 0.663107, 'Precision': 0.0742729, 'Recall': 0.828691, 'F1_Score': 0.136327},
    'Pneumonia': {'AUC': 0.765119, 'Threshold': 0.51558, 'Sensitivity': 0.73, 'Specificity': 0.696425, 'Accuracy': 0.6968, 'Precision': 0.0292288, 'Recall': 0.726619, 'F1_Score': 0.056197},
    'Fibrosis': {'AUC': 0.826327, 'Threshold': 0.585557, 'Sensitivity': 0.798, 'Specificity': 0.722608, 'Accuracy': 0.723782, 'Precision': 0.0419931, 'Recall': 0.795252, 'F1_Score': 0.0797738},
    'Edema': {'AUC': 0.908537, 'Threshold': 0.413506, 'Sensitivity': 0.911, 'Specificity': 0.746759, 'Accuracy': 0.750156, 'Precision': 0.0714644, 'Recall': 0.908511, 'F1_Score': 0.132596},
    'Consolidation': {'AUC': 0.812462, 'Threshold': 0.546624, 'Sensitivity': 0.774, 'Specificity': 0.722326, 'Accuracy': 0.724462, 'Precision': 0.109814, 'Recall': 0.772632, 'F1_Score': 0.192298}
}

class ChestXRayPipeline:
    def __init__(self, timestamp=None, logger=None):
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger = logger or logging.getLogger('pipeline')
        self.model_file = f"pipeline_model_{self.timestamp}.pth"
        self.results_file = f"pipeline_results_{self.timestamp}.csv"
        self.log_file = f"pipeline_log_{self.timestamp}.txt"
        self.comparison_file = f"baseline_comparison_{self.timestamp}.csv"
        
    def log(self, message):
        """Log message using centralized logger"""
        self.logger.info(message)
    
    def run_training(self):
        """Execute original training script with pipeline parameters"""
        self.log("Starting training phase...")
        
        # Import and run training directly
        from chest_xray_train_pipeline import train_model_pipeline
        
        try:
            train_model_pipeline(
                timestamp=self.timestamp,
                model_file=self.model_file,
                log_file=self.log_file,
                logger=self.logger
            )
            self.log("Training completed successfully")
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def run_testing(self):
        """Execute testing with pipeline parameters"""
        self.log("Starting testing phase...")
        
        # Import and run testing directly
        from chest_xray_test_pipeline import test_model_pipeline
        
        try:
            test_model_pipeline(
                model_file=self.model_file,
                results_file=self.results_file,
                logger=self.logger
            )
            self.log("Testing completed successfully")
        except Exception as e:
            raise Exception(f"Testing failed: {str(e)}")
    
    def compare_with_baseline(self):
        """Compare results with Wang et al. baseline"""
        self.log("Comparing results with baseline...")
        
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file {self.results_file} not found")
        
        # Load our results
        our_results = pd.read_csv(self.results_file)
        
        # Create comparison
        comparison_data = []
        for _, row in our_results.iterrows():
            label = row['Label']
            baseline_metrics = BASELINE_RESULTS.get(label, {})
            
            our_auc = row['AUC']
            baseline_auc = baseline_metrics.get('AUC', np.nan)
            
            auc_improvement = our_auc - baseline_auc if not np.isnan(our_auc) and not np.isnan(baseline_auc) else np.nan
            auc_improvement_pct = (auc_improvement / baseline_auc * 100) if not np.isnan(auc_improvement) and baseline_auc != 0 else np.nan
            
            # Calculate improvements for all metrics
            def calc_improvement(our_val, baseline_val):
                if pd.isna(our_val) or pd.isna(baseline_val):
                    return np.nan, 'Equal'
                improvement = our_val - baseline_val
                better = 'Yes' if improvement > 0 else 'No' if improvement < 0 else 'Equal'
                return improvement, better
            
            auc_imp, auc_better = calc_improvement(our_auc, baseline_auc)
            threshold_imp, threshold_better = calc_improvement(row.get('Threshold', 0.5), baseline_metrics.get('Threshold', np.nan))
            accuracy_imp, accuracy_better = calc_improvement(row.get('Accuracy', np.nan), baseline_metrics.get('Accuracy', np.nan))
            specificity_imp, specificity_better = calc_improvement(row.get('Specificity', np.nan), baseline_metrics.get('Specificity', np.nan))
            recall_imp, recall_better = calc_improvement(row.get('Recall', np.nan), baseline_metrics.get('Recall', np.nan))
            precision_imp, precision_better = calc_improvement(row.get('Precision', np.nan), baseline_metrics.get('Precision', np.nan))
            sensitivity_imp, sensitivity_better = calc_improvement(row.get('Sensitivity', np.nan), baseline_metrics.get('Sensitivity', np.nan))
            f1_imp, f1_better = calc_improvement(row.get('F1_Score', np.nan), baseline_metrics.get('F1_Score', np.nan))
            
            comparison_data.append({
                'Label': label,
                # AUC metrics
                'Baseline_AUC': baseline_auc,
                'Our_AUC': our_auc,
                'AUC_Improvement': auc_imp,
                'Better_AUC': auc_better,
                # Threshold metrics
                'Baseline_Threshold': baseline_metrics.get('Threshold', np.nan),
                'Our_Threshold': row.get('Threshold', 0.5),
                'Threshold_Improvement': threshold_imp,
                'Better_Threshold': threshold_better,
                # Accuracy metrics
                'Baseline_Accuracy': baseline_metrics.get('Accuracy', np.nan),
                'Our_Accuracy': row.get('Accuracy', np.nan),
                'Accuracy_Improvement': accuracy_imp,
                'Better_Accuracy': accuracy_better,
                # Specificity metrics
                'Baseline_Specificity': baseline_metrics.get('Specificity', np.nan),
                'Our_Specificity': row.get('Specificity', np.nan),
                'Specificity_Improvement': specificity_imp,
                'Better_Specificity': specificity_better,
                # Recall metrics
                'Baseline_Recall': baseline_metrics.get('Recall', np.nan),
                'Our_Recall': row.get('Recall', np.nan),
                'Recall_Improvement': recall_imp,
                'Better_Recall': recall_better,
                # Precision metrics
                'Baseline_Precision': baseline_metrics.get('Precision', np.nan),
                'Our_Precision': row.get('Precision', np.nan),
                'Precision_Improvement': precision_imp,
                'Better_Precision': precision_better,
                # Sensitivity metrics
                'Baseline_Sensitivity': baseline_metrics.get('Sensitivity', np.nan),
                'Our_Sensitivity': row.get('Sensitivity', np.nan),
                'Sensitivity_Improvement': sensitivity_imp,
                'Better_Sensitivity': sensitivity_better,
                # F1 Score metrics
                'Baseline_F1_Score': baseline_metrics.get('F1_Score', np.nan),
                'Our_F1_Score': row.get('F1_Score', np.nan),
                'F1_Score_Improvement': f1_imp,
                'Better_F1_Score': f1_better
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.comparison_file, index=False)
        
        # Summary statistics
        valid_improvements = comparison_df['AUC_Improvement'].dropna()
        better_count = (valid_improvements > 0).sum()
        worse_count = (valid_improvements < 0).sum()
        equal_count = (valid_improvements == 0).sum()
        
        avg_improvement = valid_improvements.mean()

        self.log(f"Comparison completed. Results saved to {self.comparison_file}")
        self.log(f"Summary: {better_count} better, {worse_count} worse, {equal_count} equal")

        return comparison_df
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        start_time = time.time()
        self.log("Starting Chest X-Ray Classification Pipeline")
        self.log(f"Timestamp: {self.timestamp}")
        self.log(f"Model file: {self.model_file}")
        self.log(f"Results file: {self.results_file}")
        
        try:
            # Phase 1: Training
            self.run_training()
            
            # Phase 2: Testing
            self.run_testing()
            
            # Phase 3: Comparison
            self.compare_with_baseline()
            
            # Final summary
            total_time = time.time() - start_time
            self.log(f"Pipeline completed successfully in {total_time:.2f} seconds")
            self.log("Generated files:")
            self.log(f"  - Model: {self.model_file}")
            self.log(f"  - Results: {self.results_file}")
            self.log(f"  - Comparison: {self.comparison_file}")
            self.log(f"  - Log: {self.log_file}")
            
            print(f"\\n{'='*60}")
            print("PIPELINE SUMMARY")
            print(f"{'='*60}")
            print(f"Model saved as: {self.model_file}")
            print(f"Results saved as: {self.results_file}")
            print(f"Comparison saved as: {self.comparison_file}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"{'='*60}")
            
            return True
            
        except Exception as e:
            self.log(f"Pipeline failed: {str(e)}")
            print(f"ERROR: {str(e)}")
            return False

if __name__ == "__main__":
    pipeline = ChestXRayPipeline()
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1)