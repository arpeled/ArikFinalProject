# Chest X-Ray Disease Classification using Deep Learning

Multi-label classification of 14 thoracic diseases from chest X-ray images using a modified DenseNet-121 architecture, with an AI-guided automated hyperparameter optimization pipeline across 150 training iterations.

MSc Thesis Project — Arik Peled

---

## Project Overview

This project implements an end-to-end deep learning pipeline for multi-label chest X-ray classification on the NIH ChestX-ray14 dataset (112,120 images, 30,805 patients). The system classifies 14 thoracic conditions simultaneously:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.

Key contributions:

- **Automated improvement loop**: 150 training iterations guided by an AI advisor (OpenAI API) that analyzes results and suggests configuration changes.
- **Role-based optimization**: Iterations are assigned roles (TRAIN_AUC, RECOVER_F1, ADJUST_THRESHOLDS) following a phased protocol with hard-coded rules the AI cannot override.
- **Per-class threshold optimization**: Each disease receives an individually tuned classification threshold rather than a fixed 0.5 cutoff.
- **Patient-level data splitting**: Train/validation/test splits are performed at the patient level to prevent data leakage.
- **Class imbalance handling**: Weighted BCE loss, rare-class augmentation, and oversampling for underrepresented conditions.

---

## Repository Structure

```
.
├── main.py                      # Primary entry point (see "How to Run")
│
├── core/                        # Core pipeline modules
│   ├── config_based_pipeline.py # Training pipeline (loads YAML config, trains model)
│   ├── dataset.py               # Dataset class, model definitions, augmentation
│   ├── threshold_optimizer.py   # Per-class threshold optimization
│   ├── iteration_baselines.py   # Role-based constants, phases, hard-coded rules
│   ├── config_manager.py        # YAML config loading and versioning
│   ├── chest_xray_test_pipeline.py  # Model evaluation on test set
│   ├── best_iteration_tracker.py    # Multi-metric best-iteration tracking
│   ├── best_model_tracker.py        # Best model tracking with rollback
│   ├── task_manager.py              # Optimization task lifecycle management
│   └── telegram_notifier.py         # Telegram bot progress notifications
│
├── automation/                  # Auto-improvement and AI advisory
│   ├── auto_improvement_loop.py # Main iteration loop (orchestrates everything)
│   ├── ai_advisor.py            # OpenAI integration for config suggestions
│   ├── run_auto_improvement.sh  # Shell script to launch the loop
│   └── run_auto_improvement_uv.sh
│
├── experiments/                 # All experimental outputs (no runnable logic)
│   ├── auto_improvement_runs/   # 150 completed iterations (configs, models, results)
│   ├── configs/                 # YAML configuration files
│   ├── results/                 # Comparison CSVs, evaluation outputs
│   ├── logs/                    # Training logs and run info
│   ├── figures/                 # Generated visualizations
│   ├── notebooks/               # Jupyter notebooks (exploration, prototyping)
│   ├── legacy_scripts/          # Historical script versions (v01–v05)
│   ├── legacy_models/           # Pre-iteration model checkpoints
│   ├── calibration_runs/        # Calibration experiments
│   └── head_upgrade_runs/       # Model head architecture experiments
│
├── analysis/                    # Thesis figures, tables, and statistics
│   ├── thesis_chapter5_figures.py       # Chapter 5 results figures
│   ├── thesis_appendix_b_figures.py     # Appendix B supplementary figures
│   ├── thesis_appendix_c_config_summary.py  # Appendix C config evolution
│   ├── thesis_dataset_analysis.py       # Dataset statistics (Chapter 3)
│   ├── generate_retrospective_analysis.py   # Historical config impact analysis
│   ├── generate_iteration_graphs.py     # Metric progression graphs
│   ├── view_confusion_matrix.py         # Confusion matrix visualization
│   ├── analyze_iterations.py            # Iteration performance trends
│   └── print_stats.py                   # Basic dataset statistics
│
├── docs/                        # Research documentation
│   ├── comprehensive_research_timeline.md
│   ├── final_research_summary.md
│   ├── system_code_overview.md
│   ├── notes/                   # Detailed research notes (fixes, improvements, phases)
│   └── decisions/               # Architectural and design decisions
│
├── ChestX-ray14/                # Dataset directory (NOT included in repo)
│   ├── images224/               # Resized 224x224 X-ray images
│   ├── images/                  # Original full-resolution images
│   ├── labels/                  # Label files
│   ├── Data_Entry_2017_v2020.csv    # Full dataset metadata
│   ├── train_data.csv           # Training split
│   └── test_data.csv            # Test split
│
├── pyproject.toml               # Project dependencies (use with uv)
└── uv.lock                      # Locked dependency versions
```

---

## Dataset

This project uses the **NIH ChestX-ray14** dataset:

- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 14 disease labels (multi-label)
- Patient metadata: age, gender, view position

The dataset is **not included** in this repository due to its size (~45 GB). To set up:

1. Download ChestX-ray14 from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC).
2. Place the data in a `ChestX-ray14/` directory at the project root.
3. Ensure `images224/` contains the resized 224x224 images, and `train_data.csv` / `test_data.csv` are present.

The pipeline expects relative paths: `./ChestX-ray14/train_data.csv`, `./ChestX-ray14/images224/`.

---

## How to Run

### Prerequisites

- Python >= 3.8
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- GPU: Apple Silicon (MPS), NVIDIA (CUDA), or CPU fallback

### Setup

```bash
# Clone and enter the project
cd ArikFinalProject

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Set OpenAI API key (required for AI advisor)
export OPENAI_API_KEY='your-key-here'
```

### Run the auto-improvement loop

```bash
# Start a new run (10 iterations by default)
uv run python main.py --config experiments/configs/config_baseline.yaml --iterations 10

# Resume from the last completed iteration
uv run python main.py --resume --iterations 10
```

### Run a single training iteration

```bash
uv run python main.py --single --config experiments/configs/config_baseline.yaml
```

### Generate thesis figures

```bash
# From project root
uv run python analysis/thesis_chapter5_figures.py
uv run python analysis/thesis_appendix_b_figures.py
uv run python analysis/thesis_dataset_analysis.py
```

---

## Experiments

All experiment outputs are stored under `experiments/`:

- **`auto_improvement_runs/`**: Contains 150 iteration directories (`iteration_001` through `iteration_150`). Each iteration includes its YAML config, trained model weights (`.pth`), test results CSV, confusion matrix JSON, optimized thresholds JSON, AI analysis text, and an iteration summary.
- **`configs/`**: Baseline and per-iteration YAML configuration files.
- **`results/`**: Aggregated comparison CSVs, retrospective analysis, and evaluation summaries.
- **`logs/`**: Training logs and run information files.

Iteration outputs follow the naming pattern `pipeline_results_YYYYMMDD-HHMMSS.csv`.

---

## Reproducibility Notes

### Deterministic data splits

- Train/validation splits use `GroupShuffleSplit` with `random_state=42`, grouped by `Patient ID`.
- This ensures no patient appears in both training and validation sets (preventing data leakage).
- The test set (`test_data.csv`) is fixed and separate from the train/validation split.

### Configuration-driven experiments

- Every iteration's exact configuration is saved as `config.yaml` inside its iteration directory.
- The role-based system (defined in `core/iteration_baselines.py`) enforces hard-coded rules that cannot be overridden by the AI advisor, ensuring methodological consistency.

### Key anchors

- **Iteration 12**: AUC anchor — best AUC performance, used as parent for TRAIN_AUC iterations.
- **Iteration 58**: F1 anchor — best F1 performance, used as parent for RECOVER_F1 iterations.

---

## Notes for Reviewers

### Which script to run

The single entry point is **`main.py`** at the project root. It supports:
- `--resume` to continue from the last iteration
- `--single` to run one training pass
- `--config` to specify a configuration file
- `--iterations` to set the number of iterations

### What not to touch

- **`ChestX-ray14/`**: Read-only dataset directory. Do not modify, move, or restructure.
- **`experiments/auto_improvement_runs/`**: Contains all 150 completed iterations. Do not delete or overwrite.
- **`core/iteration_baselines.py`**: Contains hard-coded rules and anchor constants. Modifying this changes the optimization protocol.

### Architecture at a glance

1. `main.py` invokes `automation/auto_improvement_loop.py`
2. The loop determines the iteration role and phase via `core/iteration_baselines.py`
3. It optionally consults the AI advisor (`automation/ai_advisor.py`) for config suggestions
4. Training is performed by `core/config_based_pipeline.py` using `core/dataset.py`
5. Evaluation and threshold optimization use `core/chest_xray_test_pipeline.py` and `core/threshold_optimizer.py`
6. Results are saved to `experiments/auto_improvement_runs/iteration_NNN/`

### Model architecture

- **Backbone**: DenseNet-121 (ImageNet pretrained)
- **Head**: Custom classifier with dropout, optional multi-layer MLP
- **Input**: 224x224 chest X-ray images + patient metadata (age, gender, view position)
- **Output**: 14 sigmoid outputs (one per disease)
- **Loss**: Weighted Binary Cross-Entropy with Logits

---

## Citation

If referencing this work, please cite the original dataset:

> Wang, X. et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." CVPR 2017.
