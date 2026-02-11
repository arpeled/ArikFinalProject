# Warm Start Implementation Guide

## Quick Summary

**Problem**: Each iteration starts from scratch (ImageNet weights), wasting previous learning.

**Solution**: AI-guided intelligent warm start that decides which previous model to start from based on context.

## What We've Created

### 1. Design Document ✅
**File**: `WARM_START_DESIGN.md`
- Complete architecture design
- Decision framework
- AI advisor integration strategy
- Example scenarios

### 2. Core Component ✅
**File**: `best_iteration_tracker.py`
- Tracks multiple "best" iterations
- Monitors F1, AUC, Recall, Precision
- Computes Pareto frontier
- Provides comparison data

## Implementation Roadmap

### Phase 1: Foundation (1-2 hours)

#### Step 1: Integrate Best Iteration Tracker
```python
# In auto_improvement_loop.py __init__

from best_iteration_tracker import BestIterationTracker

self.best_tracker = BestIterationTracker(
    registry_path="auto_improvement_runs/best_iterations_registry.json"
)
```

#### Step 2: Update Tracker After Each Iteration
```python
# In auto_improvement_loop.py, after evaluation

self.best_tracker.update(
    iteration=current_iteration,
    metrics={
        'avg_f1': avg_f1,
        'avg_auc': avg_auc,
        'avg_recall': avg_recall,
        'avg_precision': avg_precision
    },
    config=config,
    model_path=model_path
)

# Log summary
self.logger.info(self.best_tracker.get_summary_report())
```

### Phase 2: AI Advisor Enhancement (2-3 hours)

#### Step 3: Add Warm Start Recommendation Method

**File**: `ai_advisor.py`

```python
def recommend_warm_start(
    self,
    current_iteration: int,
    previous_iteration_data: Dict,
    best_iterations_data: Dict,
    proposed_config: Dict,
    previous_config: Dict
) -> Dict[str, Any]:
    """
    Ask AI to recommend which iteration to warm start from

    Returns:
        {
            'warm_start_from': 'iteration_59' | 'cold_start',
            'reasoning': 'explanation',
            'confidence': 0.85
        }
    """
    # Format data for AI
    context = self._format_warm_start_context(
        current_iteration,
        previous_iteration_data,
        best_iterations_data,
        proposed_config,
        previous_config
    )

    prompt = f"""
You are advising on model weight initialization for iteration {current_iteration}.

PREVIOUS ITERATION:
{self._format_iteration_summary(previous_iteration_data)}

BEST ITERATIONS (Historical Bests):
{self._format_best_iterations(best_iterations_data)}

PROPOSED CONFIGURATION CHANGES:
{self._format_config_diff(previous_config, proposed_config)}

DECISION REQUIRED:
Choose the best starting point:
1. "iteration_XX" - Load weights from a previous iteration
2. "cold_start" - Start from ImageNet pretrained weights

GUIDELINES:
- Small hyperparameter tweaks → warm start from previous or best
- Major architecture/loss changes → cold start
- Stuck in plateau (>10 iterations) → reset to best
- Building on progress → continue from previous
- Rare disease focus → use best rare-disease model

RESPOND WITH JSON:
{{
  "warm_start_from": "iteration_59" or "cold_start",
  "reasoning": "2-3 sentence explanation",
  "confidence": 0.0-1.0,
  "expected_benefit": "faster convergence | better performance | exploration"
}}
"""

    response = self._call_openai_with_retry(prompt)
    recommendation = self._parse_json_from_response(response.choices[0].message.content)

    return recommendation
```

#### Step 4: Add Helper Methods

```python
def _format_iteration_summary(self, iter_data: Dict) -> str:
    """Format iteration data for prompt"""
    return f"""
Iteration {iter_data['iteration']}:
- F1: {iter_data['metrics']['avg_f1']:.4f}
- AUC: {iter_data['metrics']['avg_auc']:.4f}
- Recall: {iter_data['metrics']['avg_recall']:.4f}
- Precision: {iter_data['metrics']['avg_precision']:.4f}
- Epochs trained: {iter_data.get('epochs_trained', 'N/A')}
"""

def _format_best_iterations(self, best_data: Dict) -> str:
    """Format best iterations for prompt"""
    lines = []
    for metric_name, data in best_data['best_iterations'].items():
        if data:
            lines.append(f"- {metric_name}: Iteration {data['iteration']} "
                        f"(value={data['value']:.4f}, "
                        f"{data['iterations_since_current']} iterations ago)")
    return "\n".join(lines)

def _format_config_diff(self, old_config: Dict, new_config: Dict) -> str:
    """Format configuration differences"""
    changes = []

    # Compare loss
    if old_config.get('loss', {}).get('gamma') != new_config.get('loss', {}).get('gamma'):
        changes.append(f"- Loss gamma: {old_config['loss']['gamma']} → {new_config['loss']['gamma']}")

    if old_config.get('loss', {}).get('type') != new_config.get('loss', {}).get('type'):
        changes.append(f"- Loss type: {old_config['loss']['type']} → {new_config['loss']['type']}")

    # Compare training
    if old_config.get('training', {}).get('learning_rate') != new_config.get('training', {}).get('learning_rate'):
        changes.append(f"- Learning rate: {old_config['training']['learning_rate']} → {new_config['training']['learning_rate']}")

    # Compare model
    if old_config.get('model', {}).get('dropout_rate') != new_config.get('model', {}).get('dropout_rate'):
        changes.append(f"- Dropout: {old_config['model']['dropout_rate']} → {new_config['model']['dropout_rate']}")

    return "\n".join(changes) if changes else "No major changes"
```

### Phase 3: Pipeline Integration (2-3 hours)

#### Step 5: Modify Training Pipeline

**File**: `config_based_pipeline.py`

Find the model creation section (around line 527) and add:

```python
# Create model
if model_config['architecture'] == "ModifiedDenseNetWithDropOut":
    model = ModifiedDenseNetWithDropOut(...)
else:
    model = ModifiedDenseNet(...)

model = model.to(device)

# NEW: Intelligent warm start
if iteration > 1 and hasattr(self, 'ai_advisor'):
    # Get data for decision
    previous_iter_data = self._get_iteration_data(iteration - 1)
    best_iterations_data = self.best_tracker.get_comparison_data(iteration)

    # Calculate iterations since current for each best
    for metric_name, best in best_iterations_data['best_iterations'].items():
        if best:
            best['iterations_since_current'] = iteration - best['iteration']

    # Get AI recommendation
    self.logger.info(f"\n🤖 Requesting AI warm start recommendation...")

    try:
        recommendation = self.ai_advisor.recommend_warm_start(
            current_iteration=iteration,
            previous_iteration_data=previous_iter_data,
            best_iterations_data=best_iterations_data,
            proposed_config=config,
            previous_config=previous_iter_data.get('config', {})
        )

        self.logger.info(f"\n📊 AI WARM START DECISION:")
        self.logger.info(f"   Choice: {recommendation['warm_start_from']}")
        self.logger.info(f"   Reasoning: {recommendation['reasoning']}")
        self.logger.info(f"   Confidence: {recommendation['confidence']:.0%}")
        self.logger.info(f"   Expected benefit: {recommendation.get('expected_benefit', 'N/A')}")

        # Apply recommendation
        if recommendation['warm_start_from'] != 'cold_start':
            source_iter_str = recommendation['warm_start_from']
            if source_iter_str.startswith('iteration_'):
                source_iter = int(source_iter_str.split('_')[1])

                # Find model file
                import glob
                pattern = f"auto_improvement_runs/iteration_{source_iter:03d}/pipeline_model_*.pth"
                model_files = glob.glob(pattern)

                if model_files:
                    model_path = model_files[0]
                    self.logger.info(f"   Loading: {model_path}")

                    try:
                        state_dict = torch.load(model_path, map_location=device)
                        model.load_state_dict(state_dict)
                        self.logger.info(f"   🔥 Successfully loaded weights from iteration {source_iter}")
                    except Exception as e:
                        self.logger.warning(f"   ⚠️  Failed to load weights: {e}")
                        self.logger.warning(f"   ⚠️  Falling back to ImageNet weights")
                else:
                    self.logger.warning(f"   ⚠️  Model file not found for iteration {source_iter}")
                    self.logger.warning(f"   ⚠️  Falling back to ImageNet weights")
        else:
            self.logger.info(f"   🆕 Using ImageNet pretrained weights (cold start)")

    except Exception as e:
        self.logger.error(f"   ❌ Error getting warm start recommendation: {e}")
        self.logger.info(f"   🆕 Defaulting to ImageNet weights")

# Continue with training...
```

#### Step 6: Add Helper Method

```python
def _get_iteration_data(self, iteration: int) -> Dict:
    """Load iteration data for warm start decision"""
    iter_dir = f"auto_improvement_runs/iteration_{iteration:03d}"

    # Load summary
    summary_file = f"{iter_dir}/iteration_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            return json.load(f)

    return {}
```

### Phase 4: Integration with Auto-Improvement Loop (1 hour)

#### Step 7: Pass Best Tracker to Pipeline

**File**: `auto_improvement_loop.py`

```python
# In run_iteration method

# Train model
trainer = ConfigBasedPipeline(...)

# NEW: Pass best tracker to pipeline
trainer.best_tracker = self.best_tracker
trainer.ai_advisor = self.ai_advisor

results_df, confusion_matrix, ... = trainer.train_model(...)
```

### Phase 5: Testing (1-2 hours)

#### Step 8: Test the System

```bash
# Test best iteration tracker
uv run best_iteration_tracker.py

# Test with real iteration
python auto_improvement_loop.py --resume
```

Watch for logs:
```
🤖 Requesting AI warm start recommendation...

📊 AI WARM START DECISION:
   Choice: iteration_59
   Reasoning: Previous iteration showed strong AUC (0.791). Proposed changes are incremental...
   Confidence: 85%
   Expected benefit: faster convergence

   Loading: auto_improvement_runs/iteration_059/pipeline_model_20260108-133418.pth
   🔥 Successfully loaded weights from iteration 59
```

## Expected Impact

### Training Time
```
Before:
Iteration 60: 10 epochs × 2h = 20 hours

After (warm start):
Iteration 60: 5 epochs × 2h = 10 hours (50% faster!)
```

### Performance
```
Before:
- Each iteration independent
- No knowledge retention
- Slower convergence

After:
- Builds on previous learning
- Retains successful features
- Faster convergence to better solutions
```

## Monitoring & Debugging

### Check Best Iterations Registry
```bash
cat auto_improvement_runs/best_iterations_registry.json
```

### Check Warm Start Decisions
```bash
grep "WARM START DECISION" auto_improvement_runs/auto_improvement_*.log
```

### Verify Weights Loaded
```bash
grep "Successfully loaded weights" auto_improvement_runs/auto_improvement_*.log
```

## Rollback Plan

If something goes wrong:

1. **Disable warm start**:
```python
# In config_based_pipeline.py
# Comment out the entire warm start block
```

2. **Clear registry**:
```bash
rm auto_improvement_runs/best_iterations_registry.json
```

3. **Resume normal operation**:
```bash
python auto_improvement_loop.py --resume
```

## Future Enhancements

### Phase 6: Advanced Features (Optional)

1. **Disease-specific best models**
   - Track best models for rare diseases
   - Track best models for common diseases
   - Use appropriate starting point based on focus

2. **Ensemble approaches**
   - Combine weights from multiple best iterations
   - Weighted averaging based on metrics

3. **A/B testing**
   - Run some iterations with warm start
   - Run some without
   - Compare results

4. **Automatic rollback**
   - If warm start iteration performs worse
   - Automatically retry with cold start

## Summary

| Component | Status | File |
|-----------|--------|------|
| Design | ✅ Complete | WARM_START_DESIGN.md |
| Best Tracker | ✅ Complete | best_iteration_tracker.py |
| AI Enhancement | ⏳ To implement | ai_advisor.py |
| Pipeline Integration | ⏳ To implement | config_based_pipeline.py |
| Loop Integration | ⏳ To implement | auto_improvement_loop.py |

**Estimated Implementation Time**: 6-10 hours
**Expected Benefit**: 30-50% faster training, better final performance

---

**Key Point**: The AI makes the decision, not hardcoded rules. This allows for intelligent, context-aware choices that adapt to each situation.
