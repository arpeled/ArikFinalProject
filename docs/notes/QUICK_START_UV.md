# Quick Start Guide - Auto-Improvement Pipeline (Using UV)

## Why UV?

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:
- **10-100x faster** than pip
- Better dependency resolution
- Built-in virtual environment management
- No need to manually activate venvs

## Prerequisites

- [ ] Python 3.8+ installed
- [ ] UV installed
- [ ] OpenAI API key
- [ ] ChestX-ray14 dataset available

## Step 1: Install UV

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Or via pip
```bash
pip install uv
```

### Verify installation
```bash
uv --version
```

## Step 2: Sync Dependencies

UV will automatically create a virtual environment and install all dependencies:

```bash
cd /Users/arikpeled/PycharmProjects/ArikFinalProject
uv sync
```

This reads `pyproject.toml` and installs:
- openai
- pyyaml
- pandas, numpy
- torch, torchvision
- scikit-learn
- Pillow

**Note:** First sync may take a few minutes to download PyTorch. Subsequent syncs are instant!

## Step 3: Set OpenAI API Key

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence:
```bash
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Step 4: Run Auto-Improvement Loop

### Option A: Interactive Script (Recommended)
```bash
./run_auto_improvement_uv.sh
```

This will:
1. Check UV installation
2. Sync dependencies automatically
3. Ask for configuration (iterations, output dir)
4. Run the pipeline

### Option B: Direct UV Run Command
```bash
uv run auto_improvement_loop.py --iterations 10
```

### Option C: Custom Configuration
```bash
uv run auto_improvement_loop.py \
    --config config_baseline.yaml \
    --iterations 5 \
    --output-dir my_experiment \
    --api-key sk-...
```

## Step 5: Verify Installation (Optional)

```bash
uv run verify_installation.py
```

Expected output: All checks passed ✅

## UV Commands Cheat Sheet

### Install/Update Dependencies
```bash
uv sync
```

### Run Python Scripts
```bash
uv run script_name.py
```

### Run Training Pipeline
```bash
uv run config_based_pipeline.py
```

### Run Specific Iteration
```bash
uv run auto_improvement_loop.py --iterations 1
```

### Add New Dependency
```bash
uv add package-name
```

### Run Python Interactively
```bash
uv run python
```

## Project Structure with UV

```
ArikFinalProject/
├── pyproject.toml                    # UV configuration & dependencies
├── .venv/                            # Auto-created by UV (gitignored)
├── uv.lock                           # Dependency lock file (auto-generated)
├── config_baseline.yaml              # Pipeline configuration
├── auto_improvement_loop.py          # Main script
├── run_auto_improvement_uv.sh        # UV-based startup script
└── ...
```

## Advantages of Using UV

### Speed
- **Dependency resolution**: 10-100x faster than pip
- **Installation**: Parallel downloads and installs
- **Virtual env creation**: Instant

### Convenience
- **Auto venv management**: No need to activate/deactivate
- **Lock file**: `uv.lock` ensures reproducible builds
- **Direct execution**: `uv run` handles everything

### Reliability
- **Better conflict resolution**: Smarter dependency solver
- **Consistent environments**: Lock file guarantees same versions
- **Faster CI/CD**: Much faster in automated pipelines

## Common Tasks

### First Time Setup
```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Navigate to project
cd /Users/arikpeled/PycharmProjects/ArikFinalProject

# 3. Sync dependencies (creates .venv automatically)
uv sync

# 4. Set API key
export OPENAI_API_KEY="sk-..."

# 5. Run pipeline
uv run auto_improvement_loop.py --iterations 2
```

### Daily Usage
```bash
# Just run - UV handles the rest
uv run auto_improvement_loop.py --iterations 10
```

### Updating Dependencies
```bash
# Update all dependencies to latest compatible versions
uv sync --upgrade

# Or update specific package
uv add openai --upgrade
```

### Clean Install
```bash
# Remove virtual environment and reinstall
rm -rf .venv
uv sync
```

## Troubleshooting

### "uv: command not found"
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then restart your terminal or run:
source ~/.zshrc  # or ~/.bashrc
```

### "No such file: pyproject.toml"
Make sure you're in the project directory:
```bash
cd /Users/arikpeled/PycharmProjects/ArikFinalProject
```

### Dependency Conflicts
UV is much better at resolving conflicts than pip, but if issues occur:
```bash
# Clean slate
rm -rf .venv uv.lock
uv sync
```

### CUDA/PyTorch Issues on Apple Silicon
UV will install the correct PyTorch version for your platform. For M4 Pro:
```bash
# UV automatically installs MPS-compatible PyTorch
uv sync
```

### OpenAI API Issues
```bash
# Verify key is set
echo $OPENAI_API_KEY

# Test API connection
uv run python -c "from openai import OpenAI; client = OpenAI(); print('✅ API key works')"
```

## Performance Comparison

### pip (Traditional)
```bash
pip install -r requirements.txt
# ⏱️ ~2-5 minutes (first install)
# ⏱️ ~30-60 seconds (subsequent installs)
```

### UV
```bash
uv sync
# ⚡ ~30-60 seconds (first install)
# ⚡ ~1-2 seconds (subsequent installs)
```

**UV is dramatically faster, especially for repeated setups!**

## Expected Timeline

### Setup Time
- **UV installation**: 10 seconds
- **First `uv sync`**: 30-60 seconds
- **Subsequent syncs**: 1-2 seconds

### Pipeline Execution (Same as pip)
- **Per iteration**: ~2.5-3 hours (full dataset)
- **10 iterations**: ~25-30 hours

## Running Tests Quickly

Want to test the system before a long run?

### Quick Test (5 minutes)
```bash
# Create test config with fewer epochs
cat > config_test.yaml << EOF
$(cat config_baseline.yaml)
training:
  num_epochs: 2
EOF

# Run 1 iteration
uv run auto_improvement_loop.py --iterations 1 --config config_test.yaml
```

## Advanced UV Features

### Run with Different Python Version
```bash
uv run --python 3.11 auto_improvement_loop.py
```

### Install Dev Dependencies
```bash
uv sync --extra dev
```

### Create Standalone Script
```bash
uv run --with openai --with pyyaml my_script.py
```

### Export to requirements.txt (for compatibility)
```bash
uv pip compile pyproject.toml -o requirements.txt
```

## Integration with IDEs

### VS Code
UV-created `.venv` is automatically detected. Just select it as your interpreter:
- `Cmd+Shift+P` → "Python: Select Interpreter" → Choose `.venv`

### PyCharm
1. Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `.venv/bin/python`

### Jupyter
```bash
uv run jupyter notebook
```

## Migration from pip

Already have a `requirements.txt`? UV can handle it:

```bash
# UV reads both pyproject.toml AND requirements.txt
uv sync

# Or import requirements.txt into pyproject.toml
uv add -r requirements.txt
```

## Best Practices

1. **Always use `uv run`** instead of activating venv
2. **Commit `uv.lock`** for reproducible builds
3. **Don't commit `.venv`** (add to .gitignore)
4. **Use `uv sync`** after pulling changes
5. **Keep UV updated**: `uv self update`

## Getting Help

- UV docs: https://github.com/astral-sh/uv
- Project docs: `AUTO_IMPROVEMENT_README.md`
- Quick start (pip): `QUICK_START_GUIDE.md`

---

## Ready to Start?

### Three Commands to Get Running:

```bash
# 1. Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies
uv sync

# 3. Run the pipeline
./run_auto_improvement_uv.sh
```

That's it! UV handles everything else automatically. 🚀

## Why This Matters for Your Thesis

Using UV demonstrates:
- Modern Python development practices
- Efficient workflow management
- Reproducible research environment
- Professional software engineering skills

Your thesis reviewers will appreciate the clean, professional setup! 🎓
