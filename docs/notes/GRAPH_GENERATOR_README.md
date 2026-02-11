# Iteration Graph Generator

Automatically generates graphs showing metric progression across training iterations for all diseases.

## Features

- **7 Graph Types**: Individual graphs for AUC, Specificity, Recall, Precision, Sensitivity, F1_Score + Average Metrics
- **Incremental Updates**: Only processes new iterations by default
- **Metadata Tracking**: Automatically tracks which iterations have been processed
- **Multi-Disease Visualization**: Each graph shows all 14 diseases with distinct colors

## Quick Start

### First Run (Generate All Graphs)
```bash
uv run generate_iteration_graphs.py --init
```

### Incremental Update (Only New Iterations)
```bash
uv run generate_iteration_graphs.py
```

### Test with Limited Iterations
```bash
uv run generate_iteration_graphs.py --init --max 3
```

## Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--init` | Regenerate all graphs from scratch | `--init` |
| `--start N` | Start from iteration N (only with --init) | `--start 50` |
| `--max N` | Process maximum N iterations (for testing) | `--max 10` |
| `--output-dir DIR` | Directory with iteration results | `--output-dir auto_improvement_runs` |
| `--graphs-dir DIR` | Directory to save graphs | `--graphs-dir graphs` |
| `--verbose` | Enable verbose logging | `--verbose` |

## Usage Examples

### Example 1: First Time Setup
```bash
# Generate graphs for all iterations starting from iteration 1
uv run generate_iteration_graphs.py --init

# Output:
# ✅ Generated 19 graphs (6 metrics × 3 groups + 1 average)
# 📁 Saved in: graphs/
```

### Example 2: After Running More Iterations
```bash
# You've run iterations 1-59, graphs exist for 1-59
# You run 3 more iterations (60, 61, 62)
# Just run:
uv run generate_iteration_graphs.py

# It will automatically:
# - Detect that last processed iteration was 59
# - Find new iterations: 60, 61, 62
# - Update all graphs with the new data
```

### Example 3: Regenerate from Specific Iteration
```bash
# If you want to regenerate graphs starting from iteration 50
uv run generate_iteration_graphs.py --init --start 50
```

### Example 4: Testing with Few Iterations
```bash
# Generate graphs for first 5 iterations only (useful for testing)
uv run generate_iteration_graphs.py --init --max 5
```

## Output Files

### Generated Graphs (in `graphs/` directory)

**19 Total Graphs**: Each metric is split into 3 disease groups for better readability

#### Disease Groups (by prevalence)
1. **Common (>8%)**: Infiltration, Effusion, Atelectasis, Nodule (4 diseases)
2. **Moderate (3-8%)**: Mass, Pneumothorax, Consolidation, Pleural_Thickening, Cardiomegaly, Emphysema (6 diseases)
3. **Rare (<3%)**: Edema, Fibrosis, Pneumonia, Hernia (4 diseases)

#### Graphs per Metric (3 graphs each × 6 metrics = 18 graphs)
- **AUC_Common_8pct_progression.png** - AUC for common diseases
- **AUC_Moderate_3-8pct_progression.png** - AUC for moderate diseases
- **AUC_Rare_3pct_progression.png** - AUC for rare diseases

(Same pattern for: Specificity, Recall, Precision, Sensitivity, F1_Score)

#### Summary Graph (1 graph)
- **Average_Metrics_progression.png** - Average of all metrics across all diseases

### Metadata File
- **graphs_metadata.json** - Tracks processed iterations and configuration

## How It Works

### Incremental Mode (Default)
1. Loads `graphs_metadata.json` to find last processed iteration
2. Scans `auto_improvement_runs/` for new iteration directories
3. Only processes iterations newer than the last one
4. **Reloads all data** to regenerate complete graphs
5. Updates metadata file

### Initialization Mode (`--init`)
1. Ignores existing metadata
2. Processes all iterations from `--start` (default: 1)
3. Generates fresh graphs
4. Creates new metadata file

## Graph Details

### Individual Metric Graphs (Grouped)
Each metric generates **3 separate graphs** for better readability:

**Common Diseases Graph** (4 diseases)
- Infiltration, Effusion, Atelectasis, Nodule
- These are the most prevalent diseases (>8% of samples)

**Moderate Diseases Graph** (6 diseases)
- Mass, Pneumothorax, Consolidation, Pleural_Thickening, Cardiomegaly, Emphysema
- Mid-range prevalence (3-8% of samples)

**Rare Diseases Graph** (4 diseases)
- Edema, Fibrosis, Pneumonia, Hernia
- Least prevalent (<3% of samples)

**Graph Properties:**
- **X-axis**: Iteration number
- **Y-axis**: Metric value (0-1 for all metrics)
- **Lines**: One line per disease within group (4-6 diseases per graph)
- **Colors**: Distinct colors from tab10 colormap
- **Baseline**: Horizontal line at 0.5 for reference
- **Size**: 16×8 inches for clarity

### Average Metrics Graph
- **X-axis**: Iteration number
- **Y-axis**: Average score across all diseases
- **Lines**: One line per metric (6 total)
- **Colors**: Distinct colors from Set2 colormap

## Data Source

The script reads data from:
```
auto_improvement_runs/
├── iteration_001/
│   └── pipeline_results_*.csv
├── iteration_002/
│   └── pipeline_results_*.csv
└── ...
```

## Dependencies

Automatically installed via `uv`:
- pandas
- numpy
- matplotlib

## Troubleshooting

### No iterations found
```bash
# Check that auto_improvement_runs/ exists
ls -la auto_improvement_runs/

# Make sure iteration directories exist
ls auto_improvement_runs/ | grep iteration
```

### Graphs not updating
```bash
# Force regeneration from scratch
uv run generate_iteration_graphs.py --init
```

### Want to reset metadata
```bash
# Delete metadata file and regenerate
rm graphs_metadata.json
uv run generate_iteration_graphs.py --init
```

## Integration with Training Pipeline

After running auto-improvement iterations:

```bash
# Run training iterations
python auto_improvement_loop.py --resume

# Update graphs
uv run generate_iteration_graphs.py

# View graphs in the graphs/ directory
```

## Performance

- **Processing Speed**: ~1 second per 10 iterations
- **Memory Usage**: Minimal (loads one iteration at a time)
- **Graph Generation**: ~0.15 seconds per graph

## Example Workflow

```bash
# Day 1: First run, generate graphs for iterations 1-30
uv run generate_iteration_graphs.py --init
# ✅ Processed 30 iterations

# Day 2: Run more iterations (31-40), update graphs
python auto_improvement_loop.py --resume
uv run generate_iteration_graphs.py
# ✅ Found 10 new iterations, updated all graphs

# Day 3: Run even more (41-62), update graphs
python auto_improvement_loop.py --resume
uv run generate_iteration_graphs.py
# ✅ Found 22 new iterations, updated all graphs
```

---

Generated: 2026-01-08
Script: `generate_iteration_graphs.py`
