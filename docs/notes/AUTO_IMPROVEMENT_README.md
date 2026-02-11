# Auto-Improvement Pipeline for Chest X-Ray Classification

This system automatically iterates through training cycles, using AI (OpenAI GPT-4) to analyze results and suggest hyperparameter improvements.

## Overview

The auto-improvement pipeline consists of:

1. **Configuration Management** - YAML-based hyperparameter storage
2. **Config-Based Training** - Training pipeline that reads from config files
3. **AI Advisor** - OpenAI GPT-4 integration for intelligent optimization suggestions
4. **Auto-Improvement Loop** - Orchestrator that runs multiple iterations automatically
5. **Documentation Generator** - Tracks changes and generates reports

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Auto-Improvement Loop                     │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────────┐
    │ Training│    │  Testing │    │  Comparison │
    └─────────┘    └──────────┘    └─────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                  ┌───────────────┐
                  │  AI Advisor   │
                  │   (GPT-4)     │
                  └───────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  New Config      │
                │  Suggestions     │
                └──────────────────┘
                          │
                          ▼
                  [Repeat Process]
```

## Files

### Core Components

- **`config_baseline.yaml`** - Baseline configuration with all hyperparameters
- **`config_manager.py`** - Configuration loading, saving, and updating
- **`config_based_pipeline.py`** - Training pipeline that uses config files
- **`ai_advisor.py`** - OpenAI integration for intelligent suggestions
- **`auto_improvement_loop.py`** - Main orchestrator

### Existing Pipeline Files (Used by System)

- **`chest_xray_test_pipeline.py`** - Testing logic
- **`dataset.py`** - Dataset and model definitions

## Configuration File Structure

```yaml
metadata:
  config_version: "1.0"
  iteration: 0
  description: "Description of this config"

model:
  architecture: "ModifiedDenseNetWithDropOut"
  num_classes: 14
  dropout_rate: 0.3
  use_additional_features: true

training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 20

  optimizer:
    type: "Adam"
    betas: [0.9, 0.999]

  scheduler:
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 5

  early_stopping:
    enabled: true
    patience: 5
    warmup_epochs: 5

loss:
  type: "FocalLoss"
  gamma: 2.0
  use_class_weights: true

augmentation:
  rare_class:
    enabled: true
    rotation_degrees: 10

data:
  train_csv: "./ChestX-ray14/train_data.csv"
  test_csv: "./ChestX-ray14/test_data.csv"
  images_dir: "./ChestX-ray14/images224"

evaluation:
  threshold: 0.5

hardware:
  device: "auto"  # auto, cuda, mps, cpu
  pin_memory: true
```

## Installation

### Prerequisites

```bash
# Install required packages
pip install openai pyyaml pandas torch torchvision scikit-learn
```

### OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it directly when running the script.

## Usage

### Quick Start

Run the auto-improvement loop with default settings (10 iterations):

```bash
python auto_improvement_loop.py
```

### Advanced Usage

```bash
# Custom number of iterations
python auto_improvement_loop.py --iterations 5

# Custom config file
python auto_improvement_loop.py --config my_custom_config.yaml

# Custom output directory
python auto_improvement_loop.py --output-dir my_experiment

# Pass API key directly
python auto_improvement_loop.py --api-key sk-...

# All options combined
python auto_improvement_loop.py \
    --config config_baseline.yaml \
    --iterations 10 \
    --api-key sk-... \
    --output-dir experiment_20241227
```

### Run Single Iteration (Testing)

To test a configuration without the full loop:

```bash
# Test config-based training
python config_based_pipeline.py

# Test AI advisor
python ai_advisor.py

# Test config manager
python config_manager.py
```

## Output Structure

After running, the system creates:

```
auto_improvement_runs/
├── auto_improvement_YYYYMMDD-HHMMSS.log
├── FINAL_REPORT.md
├── iteration_001/
│   ├── config.yaml
│   ├── pipeline_model_YYYYMMDD-HHMMSS.pth
│   ├── pipeline_results_YYYYMMDD-HHMMSS.csv
│   ├── baseline_comparison_YYYYMMDD-HHMMSS.csv
│   └── ai_analysis_001.txt
├── iteration_002/
│   └── ...
└── iteration_NNN/
    └── ...
```

## What Happens in Each Iteration

1. **Training Phase**
   - Loads configuration from YAML file
   - Trains model with specified hyperparameters
   - Saves trained model (.pth file)

2. **Testing Phase**
   - Loads trained model
   - Evaluates on test set
   - Computes metrics (AUC, F1, Recall, Precision, etc.)
   - Saves results to CSV

3. **Comparison Phase**
   - Compares results with Wang et al. baseline
   - Identifies improvements/regressions
   - Saves comparison to CSV

4. **AI Analysis Phase**
   - Sends config + results + comparison to GPT-4
   - AI analyzes performance issues
   - AI suggests specific hyperparameter changes
   - Saves analysis to text file

5. **Config Update Phase**
   - Applies AI suggestions to create new config
   - Saves new config for next iteration

6. **Repeat**
   - Process repeats with new configuration

## Key Features

### Intelligent Optimization

The AI advisor:
- Understands medical image classification challenges
- Recognizes class imbalance issues
- Suggests specific, actionable changes
- Explains reasoning behind suggestions

### Automated Tracking

Every iteration tracks:
- Configuration used
- Model checkpoint
- Test results
- Baseline comparison
- AI analysis
- Performance metrics

### Final Report

After all iterations, generates:
- Summary table of all iterations
- Best performing iteration
- Improvement trends
- Comparative analysis

## Common AI Suggestions

Based on the current implementation issues, expect suggestions like:

1. **Threshold Optimization**
   - Moving from fixed 0.5 to per-class optimal thresholds
   - Using F1-score or Youden's index for threshold selection

2. **Loss Function Tuning**
   - Adjusting FocalLoss gamma parameter
   - Modifying class weight alpha values

3. **Learning Rate**
   - Reducing for better convergence
   - Adjusting scheduler parameters

4. **Training Duration**
   - Increasing epochs if underfitting
   - Adjusting early stopping for better training

5. **Model Architecture**
   - Adjusting dropout rate
   - Enabling/disabling additional features

6. **Augmentation**
   - Modifying rare class thresholds
   - Adjusting augmentation intensity

## Monitoring Progress

### During Execution

Watch the console/log for:
```
[Iteration N] Phase 1: Training
[Iteration N] Phase 2: Testing
[Iteration N] Phase 3: Baseline Comparison
[Iteration N] Phase 4: AI Analysis
[Iteration N] Completed successfully
  Average AUC: 0.XXXX
  Average F1: 0.XXXX
  Average Recall: 0.XXXX
```

### AI Analysis Output

Each iteration produces an AI analysis like:
```
=== ITERATION N ANALYSIS ===

ANALYSIS:
[Detailed explanation of what's working and what's not]

SUGGESTED_CHANGES:
{
  "training": {
    "learning_rate": 0.0001
  },
  "loss": {
    "gamma": 3.0
  },
  "reasoning": "..."
}
```

### Final Report

Open `FINAL_REPORT.md` to see:
- Iteration-by-iteration metrics
- Best performing configuration
- Overall improvement trends

## Troubleshooting

### "OpenAI API key not provided"

Set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### "Config file not found"

Ensure `config_baseline.yaml` exists in the current directory.

### "CUDA out of memory"

Reduce batch size in config:
```yaml
training:
  batch_size: 32  # Reduce from 64
```

### Training too slow

If using M4 Pro or similar:
```yaml
training:
  num_epochs: 10  # Reduce for testing
```

For testing the system, create a small dataset config first.

## Expected Timeline

For full dataset (89K train, 22K test):
- **Single iteration**: ~2.5 hours (training) + ~15 min (testing) ≈ 2.75 hours
- **10 iterations**: ~27-30 hours
- **AI analysis per iteration**: ~30 seconds

For small dataset (500 train, 200 test):
- **Single iteration**: ~5 minutes
- **10 iterations**: ~50 minutes

## Tips for Best Results

1. **Start with baseline** - Run 1-2 iterations to verify system works
2. **Monitor first iteration** - Check AI suggestions make sense
3. **Review AI reasoning** - Understand why changes are suggested
4. **Use small dataset first** - Test pipeline with small data before full run
5. **Save best model** - Keep track of which iteration performed best
6. **Document manually** - Add your own observations to iteration notes

## Extending the System

### Add New Metrics

Edit `ai_advisor.py` to include additional metrics in prompts.

### Change AI Model

Modify `ai_advisor.py`:
```python
self.ai_advisor = AIAdvisor(api_key=key, model="gpt-4-turbo")
```

### Custom Stopping Criteria

Add logic to `auto_improvement_loop.py`:
```python
if iteration_summary['avg_auc'] > 0.95:
    self.logger.info("Target AUC reached!")
    break
```

### Add New Hyperparameters

1. Add to `config_baseline.yaml`
2. Update `config_based_pipeline.py` to use them
3. AI will automatically consider them in suggestions

## License

Part of the Chest X-Ray Classification Master's Project

## Contact

For questions about this auto-improvement system, refer to the main project README.

## Acknowledgments

- Wang et al. for ChestX-ray8 baseline results
- OpenAI GPT-4 for intelligent optimization suggestions
- PyTorch and torchvision teams
