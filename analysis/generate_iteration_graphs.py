"""
Generate Iteration Graphs for Auto-Improvement Pipeline
Visualizes metric progression across iterations for all diseases

Usage:
    python generate_iteration_graphs.py              # Incremental update
    python generate_iteration_graphs.py --init       # Regenerate from scratch
    python generate_iteration_graphs.py --start 50   # Start from iteration 50
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('graph_generator')


class IterationGraphGenerator:
    """Generate and update graphs showing metric progression across iterations"""

    # Metrics to track (order matters for display)
    METRICS = ['AUC', 'Specificity', 'Recall', 'Precision', 'Sensitivity', 'F1_Score']

    # Disease labels (ChestX-ray14 standard order)
    DISEASES = [
        'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
        'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
        'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
    ]

    # Disease groups by prevalence (based on class frequencies)
    DISEASE_GROUPS = {
        'Common (>8%)': [
            'Infiltration',      # 24.51%
            'Effusion',          # 16.41%
            'Atelectasis',       # 14.24%
            'Nodule'             # 7.80%
        ],
        'Moderate (3-8%)': [
            'Mass',              # 7.12%
            'Pneumothorax',      # 6.53%
            'Consolidation',     # 5.75%
            'Pleural_Thickening',# 4.17%
            'Cardiomegaly',      # 3.42%
            'Emphysema'          # 3.10%
        ],
        'Rare (<3%)': [
            'Edema',             # 2.84%
            'Fibrosis',          # 2.08%
            'Pneumonia',         # 1.76%
            'Hernia'             # 0.28%
        ]
    }

    def __init__(
        self,
        output_dir: str = "experiments/auto_improvement_runs",
        graphs_dir: str = "graphs",
        metadata_file: str = "graphs_metadata.json"
    ):
        """
        Initialize graph generator

        Args:
            output_dir: Directory containing iteration results
            graphs_dir: Directory to save generated graphs
            metadata_file: File to track processed iterations
        """
        self.output_dir = output_dir
        self.graphs_dir = graphs_dir
        self.metadata_file = metadata_file

        # Create graphs directory if it doesn't exist
        os.makedirs(graphs_dir, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

        # Data storage: {metric: {disease: {iteration: value}}}
        self.data: Dict[str, Dict[str, Dict[int, float]]] = {}

    def _load_metadata(self) -> Dict:
        """Load metadata about processed iterations"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata: {metadata.get('last_iteration', 0)} iterations processed")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}. Starting fresh.")
                return self._init_metadata()
        else:
            return self._init_metadata()

    def _init_metadata(self) -> Dict:
        """Initialize metadata structure"""
        return {
            'last_iteration': 0,
            'processed_iterations': [],
            'last_update': None,
            'metrics_tracked': self.METRICS,
            'diseases_tracked': self.DISEASES
        }

    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metadata saved: {len(self.metadata['processed_iterations'])} iterations tracked")

    def find_iterations(self, start_iteration: int = 1, max_iterations: Optional[int] = None) -> List[int]:
        """
        Find all available iterations

        Args:
            start_iteration: Iteration to start from
            max_iterations: Maximum number of iterations to process (for testing)

        Returns:
            List of iteration numbers
        """
        iterations = []

        # Look for iteration directories
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                if item.startswith('iteration_') and os.path.isdir(os.path.join(self.output_dir, item)):
                    try:
                        iter_num = int(item.split('_')[1])
                        if iter_num >= start_iteration:
                            iterations.append(iter_num)
                    except (ValueError, IndexError):
                        continue

        iterations.sort()

        # Limit to max_iterations if specified
        if max_iterations is not None and len(iterations) > max_iterations:
            iterations = iterations[:max_iterations]
            logger.info(f"Limited to first {max_iterations} iterations for testing")

        logger.info(f"Found {len(iterations)} iterations (starting from {start_iteration})")
        return iterations

    def load_iteration_data(self, iteration: int) -> Optional[pd.DataFrame]:
        """
        Load pipeline results for a specific iteration

        Args:
            iteration: Iteration number

        Returns:
            DataFrame with results, or None if not found
        """
        iter_dir = os.path.join(self.output_dir, f'iteration_{iteration:03d}')

        if not os.path.exists(iter_dir):
            logger.warning(f"Iteration directory not found: {iter_dir}")
            return None

        # Find pipeline_results CSV file
        csv_files = [f for f in os.listdir(iter_dir) if f.startswith('pipeline_results_') and f.endswith('.csv')]

        if not csv_files:
            logger.warning(f"No pipeline_results CSV found for iteration {iteration}")
            return None

        # Use the most recent file if multiple exist
        csv_file = os.path.join(iter_dir, sorted(csv_files)[-1])

        try:
            df = pd.read_csv(csv_file)
            logger.debug(f"Loaded iteration {iteration}: {len(df)} diseases")
            return df
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
            return None

    def extract_metrics(self, df: pd.DataFrame, iteration: int):
        """
        Extract metrics from DataFrame and store in data structure

        Args:
            df: DataFrame with results
            iteration: Iteration number
        """
        for metric in self.METRICS:
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in iteration {iteration}")
                continue

            if metric not in self.data:
                self.data[metric] = {}

            for _, row in df.iterrows():
                disease = row['Label']
                value = row[metric]

                if disease not in self.data[metric]:
                    self.data[metric][disease] = {}

                self.data[metric][disease][iteration] = value

    def load_all_data(self, iterations: List[int]):
        """
        Load data for all specified iterations

        Args:
            iterations: List of iteration numbers to load
        """
        logger.info(f"Loading data for {len(iterations)} iterations...")

        for iteration in iterations:
            df = self.load_iteration_data(iteration)
            if df is not None:
                self.extract_metrics(df, iteration)
                if iteration not in self.metadata['processed_iterations']:
                    self.metadata['processed_iterations'].append(iteration)

        # Update last iteration
        if iterations:
            self.metadata['last_iteration'] = max(iterations)

        # Sort processed iterations
        self.metadata['processed_iterations'].sort()

        logger.info(f"Loaded data for {len(self.metadata['processed_iterations'])} iterations")

    def generate_metric_graph(self, metric: str, save_path: str):
        """
        Generate graphs for a specific metric, split by disease groups

        Args:
            metric: Metric name (e.g., 'AUC', 'F1_Score')
            save_path: Base path to save the graphs (will append group name)
        """
        if metric not in self.data:
            logger.warning(f"No data available for metric: {metric}")
            return

        # Generate a graph for each disease group
        for group_name, diseases in self.DISEASE_GROUPS.items():
            self._generate_group_graph(metric, group_name, diseases, save_path)

    def _generate_group_graph(self, metric: str, group_name: str, diseases: List[str], base_path: str):
        """
        Generate a graph for a specific disease group

        Args:
            metric: Metric name
            group_name: Name of the disease group
            diseases: List of diseases in this group
            base_path: Base path for saving (will insert group name)
        """
        # Create figure with good size
        plt.figure(figsize=(16, 8))

        # Use a colormap for distinct colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(diseases)))

        # Plot each disease in the group
        for idx, disease in enumerate(diseases):
            if disease not in self.data[metric]:
                continue

            disease_data = self.data[metric][disease]

            # Sort by iteration number
            iterations = sorted(disease_data.keys())
            values = [disease_data[iter_num] for iter_num in iterations]

            # Plot line with markers
            plt.plot(
                iterations,
                values,
                marker='o',
                markersize=5,
                linewidth=2.5,
                label=disease,
                color=colors[idx],
                alpha=0.85
            )

        # Formatting
        plt.xlabel('Iteration', fontsize=14, fontweight='bold')
        plt.ylabel(metric, fontsize=14, fontweight='bold')
        plt.title(f'{metric} Progression: {group_name}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=12, framealpha=0.95)

        # Set y-axis limits based on metric
        if metric in ['AUC', 'Specificity', 'Recall', 'Precision', 'Sensitivity', 'F1_Score']:
            plt.ylim(0, 1.05)

        # Add horizontal line at baseline
        plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Tight layout
        plt.tight_layout()

        # Save figure with group name in filename
        # Convert base_path from "AUC_progression.png" to "AUC_Common_progression.png"
        path_parts = base_path.rsplit('_', 1)
        group_safe = group_name.replace('(', '').replace(')', '').replace('>', '').replace('<', '').replace('%', 'pct').replace(' ', '_')
        save_path = f"{path_parts[0]}_{group_safe}_{path_parts[1]}"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved graph: {save_path}")

        # Close figure to free memory
        plt.close()

    def generate_average_graph(self, save_path: str):
        """
        Generate a graph showing average metrics across all diseases

        Args:
            save_path: Path to save the graph
        """
        logger.info("Generating average metrics graph...")

        # Calculate averages for each metric and iteration
        avg_data = {}

        for metric in self.METRICS:
            if metric not in self.data:
                continue

            avg_data[metric] = {}

            # Get all iterations
            all_iterations = set()
            for disease_data in self.data[metric].values():
                all_iterations.update(disease_data.keys())

            # Calculate average for each iteration
            for iteration in sorted(all_iterations):
                values = []
                for disease in self.DISEASES:
                    if disease in self.data[metric] and iteration in self.data[metric][disease]:
                        values.append(self.data[metric][disease][iteration])

                if values:
                    avg_data[metric][iteration] = np.mean(values)

        # Create figure
        plt.figure(figsize=(16, 10))

        # Use distinct colors for each metric
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.METRICS)))

        # Plot each metric
        for idx, metric in enumerate(self.METRICS):
            if metric not in avg_data:
                continue

            iterations = sorted(avg_data[metric].keys())
            values = [avg_data[metric][iter_num] for iter_num in iterations]

            plt.plot(
                iterations,
                values,
                marker='o',
                markersize=5,
                linewidth=2.5,
                label=f'Avg {metric}',
                color=colors[idx],
                alpha=0.8
            )

        # Formatting
        plt.xlabel('Iteration', fontsize=14, fontweight='bold')
        plt.ylabel('Average Score', fontsize=14, fontweight='bold')
        plt.title('Average Metrics Across Iterations (All Diseases)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=12, framealpha=0.9)
        plt.ylim(0, 1.05)

        # Add horizontal line at 0.5
        plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved average graph: {save_path}")
        plt.close()

    def generate_all_graphs(self):
        """Generate all metric graphs"""
        logger.info("Generating all graphs...")

        # Generate individual metric graphs (3 graphs per metric for each group)
        for metric in self.METRICS:
            save_path = os.path.join(self.graphs_dir, f'{metric}_progression.png')
            self.generate_metric_graph(metric, save_path)

        # Generate average metrics graph
        avg_path = os.path.join(self.graphs_dir, 'Average_Metrics_progression.png')
        self.generate_average_graph(avg_path)

        # Calculate total: 6 metrics × 3 groups + 1 average = 19 graphs
        total_graphs = len(self.METRICS) * len(self.DISEASE_GROUPS) + 1
        logger.info(f"Generated {total_graphs} graphs in {self.graphs_dir}/")

    def run(self, init: bool = False, start_iteration: int = 1, max_iterations: Optional[int] = None):
        """
        Main execution method

        Args:
            init: If True, regenerate all graphs from scratch
            start_iteration: Iteration to start from (only used with init=True)
            max_iterations: Maximum number of iterations to process (for testing)
        """
        logger.info("="*80)
        logger.info("ITERATION GRAPH GENERATOR")
        logger.info("="*80)

        if init:
            logger.info("🔄 INITIALIZATION MODE: Regenerating all graphs from scratch")
            self.metadata = self._init_metadata()
            iterations = self.find_iterations(start_iteration, max_iterations)
        else:
            logger.info("📊 INCREMENTAL MODE: Updating with new iterations")
            last_processed = self.metadata.get('last_iteration', 0)
            logger.info(f"Last processed iteration: {last_processed}")

            # Find new iterations
            all_iterations = self.find_iterations(1, max_iterations)
            iterations = [i for i in all_iterations if i > last_processed]

            if not iterations:
                logger.info("✅ No new iterations found. Graphs are up to date!")
                return

            logger.info(f"Found {len(iterations)} new iterations: {iterations}")

            # Also reload all previous data for complete graphs
            iterations = all_iterations

        if not iterations:
            logger.warning("⚠️  No iterations found to process!")
            return

        # Load data
        self.load_all_data(iterations)

        # Generate graphs
        self.generate_all_graphs()

        # Save metadata
        self._save_metadata()

        logger.info("="*80)
        logger.info(f"✅ COMPLETE: Processed {len(self.metadata['processed_iterations'])} iterations")
        logger.info(f"📁 Graphs saved in: {self.graphs_dir}/")
        logger.info("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate iteration graphs for auto-improvement pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_iteration_graphs.py              # Update with new iterations
  python generate_iteration_graphs.py --init       # Regenerate all graphs
  python generate_iteration_graphs.py --init --start 50  # Regenerate from iteration 50
        """
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='Regenerate all graphs from scratch (ignore previous progress)'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Starting iteration number (default: 1, only used with --init)'
    )

    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum number of iterations to process (for testing)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/auto_improvement_runs',
        help='Directory containing iteration results (default: experiments/auto_improvement_runs)'
    )

    parser.add_argument(
        '--graphs-dir',
        type=str,
        default='graphs',
        help='Directory to save graphs (default: graphs)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create generator
    generator = IterationGraphGenerator(
        output_dir=args.output_dir,
        graphs_dir=args.graphs_dir
    )

    # Run
    try:
        generator.run(init=args.init, start_iteration=args.start, max_iterations=args.max)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()