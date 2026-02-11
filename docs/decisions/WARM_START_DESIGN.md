# Intelligent Warm Start Strategy - Design Document

## Problem Statement

Currently, each iteration starts from ImageNet pretrained weights, ignoring all previous learning. This means:
- ❌ Each iteration trains from scratch (10-50 epochs)
- ❌ No benefit from previous successful iterations
- ❌ Potential best models are discarded
- ❌ Training time could be much shorter

## Proposed Solution: AI-Guided Multi-Metric Warm Start

### Core Concept

Let the **AI Advisor decide** which starting point to use based on:
1. **Multiple "best" iterations** (not just one metric)
2. **Context of what changed** (hyperparameters, loss, etc.)
3. **Strategy being tested** (incremental vs exploratory)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Best Iteration Tracker                    │
│                                                               │
│  Tracks multiple "best" iterations:                          │
│  • Best F1-Score        → Iteration 47 (F1=0.2746)          │
│  • Best AUC             → Iteration 50 (AUC=0.8002)          │
│  • Best Recall          → Iteration 31 (Recall=0.5212)       │
│  • Best Precision       → Iteration 48 (Prec=0.3034)         │
│  • Best Rare-Disease F1 → Iteration 56 (Rare_F1=0.15)        │
│  • Best Common-Disease  → Iteration 49 (Common_AUC=0.85)     │
│  • Most Balanced        → Iteration 38 (lowest variance)     │
│  • Pareto Optimal       → [Iteration 47, 50, 48]             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      AI Advisor Analysis                      │
│                                                               │
│  Input:                                                       │
│  • Current iteration: 60                                      │
│  • Previous iteration: 59 (F1=0.269, AUC=0.791)              │
│  • Best iterations (multi-metric)                             │
│  • Proposed changes: reduce gamma 3.0→2.0, alpha 0.75→0.7    │
│                                                               │
│  AI Decision Logic:                                           │
│  "Since we're making incremental loss adjustments and         │
│   iteration 57 (F1=0.2697) is very close to current,         │
│   AND previous iteration 59 had high AUC (0.7913),           │
│   RECOMMEND: warm_start_from iteration 59                     │
│   REASONING: Build on recent progress, small tweaks"          │
│                                                               │
│  Alternative scenarios:                                       │
│  • Major architecture change → cold_start (ImageNet)          │
│  • Testing new loss function → warm_start best_f1 (47)       │
│  • Focusing on rare diseases → warm_start best_rare (56)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Config-Based Pipeline                       │
│                                                               │
│  Loads weights based on AI recommendation:                    │
│                                                               │
│  if warm_start_from == "iteration_59":                        │
│      model.load_state_dict(torch.load(                        │
│          "auto_improvement_runs/iteration_059/model.pth"))    │
│      print(f"🔥 Warm start from iteration 59")               │
│                                                               │
│  elif warm_start_from == "cold_start":                        │
│      # Use ImageNet weights (default behavior)                │
│      print(f"🆕 Cold start from ImageNet")                   │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Metric Best Iteration Tracking

### Metrics to Track

```python
TRACKED_METRICS = {
    # Primary metrics
    'best_f1': {
        'metric': 'avg_f1',
        'mode': 'max',
        'weight': 0.4  # Importance weight
    },
    'best_auc': {
        'metric': 'avg_auc',
        'mode': 'max',
        'weight': 0.3
    },
    'best_recall': {
        'metric': 'avg_recall',
        'mode': 'max',
        'weight': 0.2
    },
    'best_precision': {
        'metric': 'avg_precision',
        'mode': 'max',
        'weight': 0.1
    },

    # Secondary metrics
    'best_rare_disease_f1': {
        'metric': 'rare_disease_f1',  # Hernia, Pneumonia, Fibrosis, Edema
        'mode': 'max',
        'weight': 0.15
    },
    'best_common_disease_auc': {
        'metric': 'common_disease_auc',  # Infiltration, Effusion, etc.
        'mode': 'max',
        'weight': 0.15
    },

    # Composite metrics
    'most_balanced': {
        'metric': 'metric_variance',  # Low variance = balanced
        'mode': 'min',
        'weight': 0.1
    },
    'best_pareto': {
        'metric': 'pareto_rank',  # Multi-objective optimization
        'mode': 'min',
        'weight': 0.2
    }
}
```

### Pareto Optimality

An iteration is Pareto optimal if no other iteration is better in ALL metrics:

```
Example:
Iteration 47: F1=0.275, AUC=0.785, Recall=0.37  ← Pareto optimal (best F1)
Iteration 50: F1=0.272, AUC=0.800, Recall=0.52  ← Pareto optimal (best AUC)
Iteration 48: F1=0.267, AUC=0.783, Recall=0.37  ← Dominated (not Pareto)
```

## AI Advisor Decision Framework

### Information Provided to AI

```json
{
  "current_iteration": 60,
  "previous_iteration": {
    "iteration": 59,
    "metrics": {"avg_f1": 0.269, "avg_auc": 0.791, "avg_recall": 0.446},
    "training": {"epochs_trained": 10, "early_stopped": true}
  },
  "best_iterations": {
    "best_f1": {
      "iteration": 47,
      "value": 0.2746,
      "metrics": {"avg_f1": 0.2746, "avg_auc": 0.7848, "avg_recall": 0.4965},
      "config_diff": ["gamma: 3.0 vs current 2.0", "dropout: 0.3 vs current 0.3"]
    },
    "best_auc": {
      "iteration": 50,
      "value": 0.8002,
      "metrics": {"avg_f1": 0.2718, "avg_auc": 0.8002, "avg_recall": 0.5233}
    },
    "best_recall": {
      "iteration": 31,
      "value": 0.5212,
      "metrics": {"avg_f1": 0.1672, "avg_auc": 0.7055, "avg_recall": 0.5212}
    }
  },
  "proposed_changes": {
    "loss": {"gamma": "3.0 → 2.0", "alpha": "0.75 → 0.7"},
    "reasoning": "Reduce false positives for rare classes"
  },
  "pareto_frontier": [47, 50, 56],
  "recent_trend": {
    "f1_trend": "plateaued around 0.27",
    "auc_trend": "slowly improving",
    "iterations_since_best_f1": 12
  }
}
```

### AI Decision Rules (Examples)

```python
# Rule 1: Incremental changes → Previous iteration
if change_magnitude == "small" and previous_iteration.f1 > 0.25:
    return {
        "warm_start_from": "iteration_59",
        "reasoning": "Incremental improvement, build on recent progress"
    }

# Rule 2: Major changes → Best relevant metric
if "loss.type" changed or "architecture" changed:
    return {
        "warm_start_from": "cold_start",
        "reasoning": "Major structural change, start fresh"
    }

# Rule 3: Targeting specific metric → Use best for that metric
if focusing_on == "recall":
    return {
        "warm_start_from": "iteration_31",  # best_recall
        "reasoning": "Targeting recall, start from best recall iteration"
    }

# Rule 4: Stuck in plateau → Reset to best
if iterations_since_improvement > 10:
    return {
        "warm_start_from": "iteration_47",  # best_f1
        "reasoning": "Plateau detected, reset to best performing iteration"
    }

# Rule 5: Rare disease focus → Use best rare disease iteration
if task.focus == "rare_diseases":
    return {
        "warm_start_from": "iteration_56",
        "reasoning": "Task targets rare diseases, use best rare-disease model"
    }
```

## Implementation Details

### 1. Best Iteration Tracker (`best_iteration_tracker.py`)

```python
class BestIterationTracker:
    """
    Tracks multiple "best" iterations across different metrics
    """

    def __init__(self, registry_path: str = "best_iterations_registry.json"):
        self.registry_path = registry_path
        self.best_iterations = self._load_registry()

    def update(self, iteration: int, metrics: Dict[str, float],
               config: Dict, model_path: str):
        """
        Update best iteration tracking

        Args:
            iteration: Current iteration number
            metrics: All metrics from this iteration
            config: Configuration used
            model_path: Path to saved model
        """
        # Check if this iteration is best for any metric
        for metric_name, tracker_config in TRACKED_METRICS.items():
            metric_value = metrics.get(tracker_config['metric'])

            if self._is_better(metric_value, metric_name):
                self.best_iterations[metric_name] = {
                    'iteration': iteration,
                    'value': metric_value,
                    'metrics': metrics.copy(),
                    'config': config.copy(),
                    'model_path': model_path,
                    'timestamp': datetime.now().isoformat()
                }

        # Update Pareto frontier
        self._update_pareto_frontier(iteration, metrics, config, model_path)

        self._save_registry()

    def get_best_for_metric(self, metric: str) -> Optional[Dict]:
        """Get best iteration for specific metric"""
        return self.best_iterations.get(metric)

    def get_pareto_optimal(self) -> List[Dict]:
        """Get all Pareto optimal iterations"""
        return self.best_iterations.get('pareto_frontier', [])

    def get_comparison_data(self, current_iteration: int) -> Dict:
        """
        Get comparison data for AI advisor

        Returns:
            {
                'current_iteration': 60,
                'best_iterations': {...},
                'pareto_frontier': [...],
                'statistics': {...}
            }
        """
        return {
            'current_iteration': current_iteration,
            'best_iterations': self.best_iterations,
            'pareto_frontier': self.get_pareto_optimal(),
            'statistics': self._calculate_statistics()
        }
```

### 2. Enhanced AI Advisor (`ai_advisor.py`)

Add new method:

```python
def recommend_warm_start(
    self,
    current_iteration: int,
    previous_iteration: Dict,
    best_iterations: Dict,
    proposed_config: Dict,
    current_config: Dict
) -> Dict[str, Any]:
    """
    Recommend which iteration to warm start from

    Returns:
        {
            'warm_start_from': 'iteration_59' | 'cold_start',
            'reasoning': 'Why this choice was made',
            'confidence': 0.85,
            'alternatives': [...]
        }
    """
    # Create prompt with all context
    prompt = f"""
You are advising on model weight initialization for a medical image classification pipeline.

CURRENT SITUATION:
- Current iteration: {current_iteration}
- Previous iteration: {previous_iteration['iteration']}
  - F1: {previous_iteration['metrics']['avg_f1']:.4f}
  - AUC: {previous_iteration['metrics']['avg_auc']:.4f}
  - Recall: {previous_iteration['metrics']['avg_recall']:.4f}

BEST ITERATIONS (Historical):
{self._format_best_iterations(best_iterations)}

PROPOSED CHANGES:
{self._format_config_diff(current_config, proposed_config)}

DECISION REQUIRED:
Choose ONE of the following starting points for iteration {current_iteration}:
1. "warm_start_iteration_XX" - Start from a previous iteration's trained weights
2. "cold_start" - Start fresh from ImageNet pretrained weights

GUIDELINES:
- Small hyperparameter changes (LR, dropout) → warm start from recent/best
- Major changes (architecture, loss type) → cold start
- Stuck in plateau → warm start from best relevant metric
- Exploring new direction → cold start
- Building on progress → warm start from previous or best

RESPOND WITH JSON:
{{
  "warm_start_from": "iteration_XX" or "cold_start",
  "reasoning": "Detailed explanation (2-3 sentences)",
  "confidence": 0.0-1.0,
  "expected_benefit": "faster convergence" | "better performance" | "exploration"
}}
"""

    response = self._call_openai_with_retry(prompt)
    recommendation = self._parse_warm_start_recommendation(response)

    return recommendation
```

### 3. Config-Based Pipeline Integration

```python
def train_model(self, config_path: str, iteration: int):
    """Enhanced training with warm start support"""

    # ... existing setup code ...

    # Create model
    model = self._create_model(model_config)

    # Get warm start recommendation from AI
    if iteration > 1:
        best_iterations = self.best_tracker.get_comparison_data(iteration)
        previous_iteration = self.get_previous_iteration_data(iteration - 1)

        warm_start_recommendation = self.ai_advisor.recommend_warm_start(
            current_iteration=iteration,
            previous_iteration=previous_iteration,
            best_iterations=best_iterations,
            proposed_config=config,
            current_config=previous_iteration['config']
        )

        self.logger.info(f"🤖 AI Warm Start Recommendation:")
        self.logger.info(f"   Decision: {warm_start_recommendation['warm_start_from']}")
        self.logger.info(f"   Reasoning: {warm_start_recommendation['reasoning']}")
        self.logger.info(f"   Confidence: {warm_start_recommendation['confidence']:.2f}")

        # Apply recommendation
        if warm_start_recommendation['warm_start_from'] != 'cold_start':
            source_iter = int(warm_start_recommendation['warm_start_from'].split('_')[1])
            model_path = f"auto_improvement_runs/iteration_{source_iter:03d}/pipeline_model_*.pth"

            # Load weights
            model.load_state_dict(torch.load(model_path))
            self.logger.info(f"🔥 Loaded weights from iteration {source_iter}")
        else:
            self.logger.info(f"🆕 Using ImageNet pretrained weights (cold start)")

    # ... rest of training code ...
```

## Expected Benefits

### 1. Training Efficiency
- ⏱️ **Faster convergence**: 30-50% fewer epochs needed
- 💰 **Cost savings**: Less GPU time
- 🎯 **Better exploration**: Can try more configurations

### 2. Performance Improvements
- 📈 **Incremental gains**: Build on successes
- 🎯 **Avoid pitfalls**: Don't repeat failures
- 🧠 **Knowledge retention**: Keep learned features

### 3. Intelligent Strategy
- 🤖 **AI-driven**: Context-aware decisions
- 📊 **Multi-objective**: Consider all metrics
- 🔄 **Adaptive**: Changes with situation

## Example Decision Flow

```
Iteration 60:
├─ AI analyzes:
│  ├─ Previous (59): F1=0.269, AUC=0.791 (good AUC, decent F1)
│  ├─ Best F1 (47): F1=0.275 (only +0.006 better)
│  ├─ Changes: Small (gamma 3.0→2.0, alpha 0.75→0.7)
│  └─ Context: Trying to reduce false positives
│
├─ AI Decision: "warm_start_from iteration_59"
│  └─ Reasoning: "Recent iteration has strong AUC, proposed changes
│                  are incremental adjustments to loss function.
│                  Starting from iteration 59 allows building on
│                  its learned features while testing if reduced
│                  gamma improves precision. Expected to converge
│                  in 5-7 epochs vs 10+ from scratch."
│
└─ Pipeline loads weights from iteration 59 → Trains → Success!

Iteration 65:
├─ AI analyzes:
│  ├─ Previous (64): F1=0.261 (dropped)
│  ├─ Best F1 (47): F1=0.275 (13 iterations ago)
│  ├─ Changes: Major (switching to WeightedBCE loss)
│  └─ Context: Different loss function
│
├─ AI Decision: "cold_start"
│  └─ Reasoning: "Switching loss functions fundamentally changes
│                  the optimization landscape. Starting from
│                  ImageNet allows the model to learn optimal
│                  features for the new loss from scratch."
│
└─ Pipeline uses ImageNet weights → Trains → Explores new direction!
```

## Visualization & Monitoring

```
Best Iterations Dashboard:
┌──────────────────────────────────────────────────────────┐
│ Best F1:        Iteration 47 (F1=0.2746) [13 iters ago] │
│ Best AUC:       Iteration 50 (AUC=0.8002) [10 iters ago] │
│ Best Recall:    Iteration 31 (Recall=0.5212) [29 ago]   │
│ Best Precision: Iteration 48 (Prec=0.3034) [12 ago]     │
│                                                           │
│ Pareto Frontier: [47, 50, 56]                            │
│                                                           │
│ Warm Start History (last 10 iterations):                 │
│  51: cold_start        (major architecture change)       │
│  52: iteration_50      (build on best AUC)               │
│  53: iteration_52      (incremental improvement)         │
│  54: iteration_47      (reset to best F1)                │
│  55: iteration_54      (small tweaks)                    │
│  56: iteration_55      (continuing progress)             │
│  57: iteration_56      (best rare disease)               │
│  58: cold_start        (new loss function)               │
│  59: iteration_58      (refining new loss)               │
│  60: iteration_59      (small adjustments)               │
└──────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (MVP)
1. ✅ Create `BestIterationTracker` class
2. ✅ Track best F1, AUC, Recall
3. ✅ Basic warm start recommendation (rule-based)

### Phase 2: AI Integration
1. ✅ Enhance AI advisor prompt with best iteration data
2. ✅ Parse AI warm start recommendations
3. ✅ Integrate into pipeline

### Phase 3: Advanced Metrics
1. ✅ Add rare/common disease tracking
2. ✅ Implement Pareto frontier
3. ✅ Add balanced metric tracking

### Phase 4: Monitoring & Visualization
1. ✅ Dashboard for best iterations
2. ✅ Warm start decision history
3. ✅ Performance comparison graphs

---

**Key Innovation**: Let AI make the decision, not hardcoded rules!

The AI advisor has context about:
- What worked before
- What we're trying to achieve
- What changed
- Historical patterns

This allows for nuanced, intelligent decisions that adapt to the specific situation.
