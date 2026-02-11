# Quick Start Guide - Auto-Improvement Pipeline

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (see below)
- [ ] OpenAI API key obtained
- [ ] ChestX-ray14 dataset available

## Step 1: Install Dependencies

```bash
pip install -r requirements_auto_improvement.txt
```

Or install individually:
```bash
pip install openai pyyaml pandas numpy torch torchvision scikit-learn Pillow
```

## Step 2: Verify Installation

```bash
python3 verify_installation.py
```

Expected output: All checks passed ✅

## Step 3: Set OpenAI API Key

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence:
```bash
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Run Auto-Improvement Loop

### Option A: Interactive Script (Easiest)
```bash
./run_auto_improvement.sh
```

### Option B: Direct Python Command
```bash
python3 auto_improvement_loop.py --iterations 10
```

### Option C: Custom Configuration
```bash
python3 auto_improvement_loop.py \
    --config config_baseline.yaml \
    --iterations 5 \
    --output-dir my_experiment \
    --api-key sk-...
```

## Step 5: Monitor Progress

Watch the console output for:
- Training progress (epochs, losses)
- Testing results (AUC, F1, Recall)
- AI analysis and suggestions
- Iteration completion status

## Step 6: Review Results

After completion:

1. **Check the final report:**
   ```bash
   cat auto_improvement_runs/FINAL_REPORT.md
   ```

2. **Review best iteration:**
   - Look for highest AUC in the summary table
   - Find the corresponding model file
   - Review AI analysis for that iteration

3. **Examine iteration details:**
   ```bash
   ls auto_improvement_runs/iteration_001/
   ```

## Expected Output Files

```
auto_improvement_runs/
├── FINAL_REPORT.md                         # Summary of all iterations
├── auto_improvement_YYYYMMDD-HHMMSS.log   # Complete execution log
├── iteration_001/
│   ├── config.yaml                         # Config used
│   ├── pipeline_model_*.pth                # Trained model
│   ├── pipeline_results_*.csv              # Test results
│   ├── baseline_comparison_*.csv           # Comparison with baseline
│   └── ai_analysis_001.txt                 # AI suggestions
└── iteration_NNN/ ...
```

## Troubleshooting

### "OpenAI API key not provided"
```bash
export OPENAI_API_KEY="your-key"
```

### "Config file not found"
Make sure you're in the project directory:
```bash
cd /path/to/ArikFinalProject
```

### "CUDA out of memory"
Edit `config_baseline.yaml`:
```yaml
training:
  batch_size: 32  # Reduce from 64
```

### Dependencies not found
Install in your environment:
```bash
pip install -r requirements_auto_improvement.txt
```

### Want to test quickly?
Create a test config with fewer epochs:
```yaml
training:
  num_epochs: 5  # Instead of 20
```

Then run:
```bash
python3 auto_improvement_loop.py --iterations 2 --config config_test.yaml
```

## What to Expect

### Timeline (Full Dataset)
- **Per iteration**: ~2.5-3 hours
- **10 iterations**: ~25-30 hours
- **AI analysis**: ~30 seconds per iteration

### Timeline (Small Dataset - for testing)
- **Per iteration**: ~5 minutes
- **10 iterations**: ~50 minutes

### Metrics to Watch
1. **AUC** - Primary metric, should trend upward
2. **Recall** - Currently ~0, should improve significantly
3. **F1-Score** - Balance of precision/recall
4. **Precision** - Currently ~0, should improve

### Success Indicators
- ✅ Recall increasing from ~0 to >0.5
- ✅ F1-Score improving each iteration
- ✅ AUC approaching or exceeding baseline (0.82-0.92)
- ✅ AI suggestions making logical sense

## Common First-Time Issues

1. **Model predicting all negatives** (Iteration 1)
   - This is expected! AI will suggest fixes
   - Look for threshold optimization suggestions

2. **High accuracy but low recall**
   - Due to class imbalance
   - AI will suggest loss function tuning

3. **Slow training**
   - First iteration benchmarks your hardware
   - Consider reducing batch size if too slow

## Next Steps After First Run

1. Review `FINAL_REPORT.md`
2. Identify best performing iteration
3. Use that model for your final report
4. Document the improvement journey
5. Consider running more iterations if improvement continues

## Getting Help

- Read `AUTO_IMPROVEMENT_README.md` for detailed documentation
- Check iteration-specific AI analysis files
- Review logs in `auto_improvement_runs/*.log`
- Examine baseline comparison CSVs for detailed metrics

## Tips for Success

1. **Start small** - Run 2-3 iterations first to verify
2. **Monitor first iteration** - Make sure everything works
3. **Review AI suggestions** - Ensure they make sense
4. **Keep baseline config** - Don't overwrite `config_baseline.yaml`
5. **Save best model** - Note which iteration performed best
6. **Document findings** - Track insights for your thesis

---

**Ready to start? Run:**
```bash
./run_auto_improvement.sh
```

Good luck with your Master's project! 🎓
