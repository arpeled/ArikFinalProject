# Task-Aware Warm Start System

## Integration with Task Manager

The warm start system now **integrates with your task system** to make intelligent decisions based on:
1. Current task objective (e.g., "improve AUC")
2. Constraint requirements (e.g., "don't harm F1")
3. Historical task performance

## Task-Aware Best Tracking

### Enhanced Tracking Structure

```json
{
  "tracked": {
    "best_f1": {...},      // Global best F1
    "best_auc": {...},     // Global best AUC
    "best_recall": {...}   // Global best Recall
  },
  "task_specific": {
    "task_avg_auc": {
      "iteration": 55,
      "value": 0.7950,
      "metrics": {
        "avg_auc": 0.7950,
        "avg_f1": 0.2680,  // Constraint: maintained > 0.25
        "avg_recall": 0.4420
      },
      "task_info": {
        "target_metric": "avg_auc",
        "target_value": 0.81,
        "constraints": {
          "avg_f1": {"min": 0.25, "operation": "maintain"}
        }
      }
    }
  }
}
```

## Task Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Manager (Existing)                       │
│                                                                  │
│  Current Task:                                                   │
│  • Target: Improve AUC to 0.81                                  │
│  • Constraints: Keep F1 ≥ 0.25 (don't harm)                    │
│  • Strategy: Adjust class weights                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Best Iteration Tracker (Enhanced)                   │
│                                                                  │
│  Tracks:                                                         │
│  • Global best: AUC=0.800 (iter 50)                            │
│  • Task-specific best: AUC=0.795 (iter 55, F1=0.268 ✓)        │
│                         vs AUC=0.800 (iter 50, F1=0.220 ✗)     │
│                                                                  │
│  Iter 55 preferred for task because it meets F1 constraint!     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AI Advisor Decision                           │
│                                                                  │
│  Context:                                                        │
│  • Task: Improve AUC (target 0.81, currently 0.791)            │
│  • Constraint: F1 ≥ 0.25 (currently 0.269)                     │
│  • Best task-specific: Iteration 55 (AUC=0.795, F1=0.268)     │
│  • Best global AUC: Iteration 50 (AUC=0.800, F1=0.220)        │
│                                                                  │
│  Decision: warm_start_from iteration_55                         │
│  Reasoning:                                                      │
│    "Task requires improving AUC while maintaining F1 ≥ 0.25.   │
│     Iteration 55 is task-specific best (AUC=0.795, F1=0.268),  │
│     meeting the F1 constraint. Although iteration 50 has        │
│     higher AUC (0.800), its F1 (0.220) violates constraint.    │
│     Starting from iter 55 provides better foundation for        │
│     constrained optimization."                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Example Scenarios

### Scenario 1: Improve AUC Without Harming F1

```python
# Task definition
task = {
    'name': 'improve_auc_constrained',
    'target_metric': 'avg_auc',
    'target_value': 0.81,
    'constraints': {
        'avg_f1': {'min': 0.25, 'operation': 'maintain'},
        'avg_recall': {'min': 0.40, 'operation': 'maintain'}
    }
}

# Iteration 60 results
metrics = {
    'avg_auc': 0.7913,
    'avg_f1': 0.2690,
    'avg_recall': 0.4462
}

# Best tracker evaluation:
✓ AUC: 0.7913 (improving, was 0.791)
✓ F1: 0.2690 (maintained, > 0.25)
✓ Recall: 0.4462 (maintained, > 0.40)
→ SUCCESS: Task constraints met, recorded as task-best!

# AI recommendation for iteration 61:
"warm_start_from: iteration_60"
"Reasoning: Successfully improved AUC while maintaining F1 and recall.
 Continue from this foundation to push AUC higher toward 0.81 target."
```

### Scenario 2: AUC Improved But F1 Harmed (Rejected)

```python
# Iteration 65 results
metrics = {
    'avg_auc': 0.8050,  # ✓ Improved!
    'avg_f1': 0.2300,   # ✗ Dropped below constraint!
    'avg_recall': 0.4200
}

# Best tracker evaluation:
✓ AUC: 0.8050 (best ever!)
✗ F1: 0.2300 (VIOLATED: < 0.25 minimum)
→ REJECTED: Task constraint violated, NOT recorded as task-best!

# AI recommendation for iteration 66:
"warm_start_from: iteration_60"
"Reasoning: Although iteration 65 achieved highest AUC (0.805),
 it violated the F1 constraint (0.230 < 0.25). Revert to iteration 60
 which maintained balance. Try different approach to improve AUC."
```

### Scenario 3: Rare Disease Task

```python
# Task definition
task = {
    'name': 'improve_rare_disease_recall',
    'target_metric': 'rare_disease_recall',  # Hernia, Pneumonia, Fibrosis, Edema
    'target_value': 0.35,
    'constraints': {
        'avg_f1': {'min': 0.25, 'operation': 'maintain'},
        'common_disease_auc': {'min': 0.78, 'operation': 'maintain'}
    }
}

# Best tracker tracks:
- best_rare_disease_recall (with constraints)
- Ensures common diseases don't suffer
- AI uses rare-disease-specific best model
```

## Enhanced AI Advisor Prompt

```python
def recommend_warm_start_with_task(
    self,
    current_iteration: int,
    task_info: Dict,
    best_iterations: Dict,
    previous_iteration: Dict
) -> Dict:
    """
    Task-aware warm start recommendation
    """

    prompt = f"""
You are advising on model initialization for iteration {current_iteration}.

CURRENT TASK:
{self._format_task_info(task_info)}

TASK-SPECIFIC BEST:
{self._format_task_best(best_iterations, task_info)}

GLOBAL BEST ITERATIONS:
{self._format_global_best(best_iterations)}

PREVIOUS ITERATION:
{self._format_iteration(previous_iteration)}

DECISION REQUIRED:
Choose starting point considering:
1. Task objective: {task_info['target_metric']} → {task_info['target_value']}
2. Constraints: {list(task_info['constraints'].keys())} must be maintained
3. Task-specific best vs global best
4. Trade-offs between metrics

OPTIONS:
a) "iteration_XX" - Task-specific best (satisfies constraints)
b) "iteration_YY" - Global best (may violate constraints)
c) "iteration_ZZ" - Previous iteration (if progressing)
d) "cold_start" - Fresh start if needed

RESPOND WITH JSON:
{{
  "warm_start_from": "iteration_XX" or "cold_start",
  "reasoning": "Why this choice for the task (2-3 sentences)",
  "confidence": 0.0-1.0,
  "expected_outcome": {{
    "{task_info['target_metric']}": "increase to 0.XX",
    "constraints": "maintained within bounds"
  }}
}}
"""
```

## Task-Specific Decision Rules

### Rule 1: Task with Constraints → Use Task-Best

```python
if task.has_constraints and task_specific_best.exists:
    if task_specific_best.meets_constraints:
        return {
            "warm_start_from": f"iteration_{task_specific_best.iteration}",
            "reasoning": f"Task requires optimizing {task.target_metric} "
                        f"while maintaining {task.constraints}. Task-specific "
                        f"best provides proven foundation."
        }
```

### Rule 2: Constraint Violated → Don't Use

```python
if previous_iteration.violates_constraints(task.constraints):
    # Find last iteration that met constraints
    last_valid = find_last_valid_iteration(task.constraints)

    return {
        "warm_start_from": f"iteration_{last_valid}",
        "reasoning": f"Previous iteration violated {violated_metric} constraint. "
                    f"Reverting to last valid iteration that maintained balance."
    }
```

### Rule 3: Progressive Task → Continue

```python
if task.is_progressive and previous_iteration.making_progress:
    if previous_iteration.meets_constraints:
        return {
            "warm_start_from": f"iteration_{previous_iteration.num}",
            "reasoning": f"Making steady progress toward task goal "
                        f"({task.target_metric}: {prev_value} → {target_value}). "
                        f"Constraints maintained. Continue from current position."
        }
```

## Integration with Auto-Improvement Loop

```python
# In auto_improvement_loop.py

def run_iteration(self, iteration: int):
    """Run single iteration with task-aware warm start"""

    # Get current task from task manager
    current_task = self.task_manager.get_current_task()

    # Update best tracker with task info
    self.best_tracker.update(
        iteration=iteration,
        metrics=metrics,
        config=config,
        model_path=model_path,
        task_info=current_task  # ← Task context
    )

    # Get AI recommendation with task awareness
    recommendation = self.ai_advisor.recommend_warm_start(
        current_iteration=iteration + 1,
        task_info=current_task,  # ← AI knows the task!
        best_iterations=self.best_tracker.get_comparison_data(
            iteration + 1,
            task_info=current_task
        ),
        previous_iteration=previous_data
    )

    # Apply recommendation in next iteration
    self.warm_start_config = recommendation
```

## Validation Logic

### Check Constraints Before Recording

```python
def validate_against_constraints(
    metrics: Dict[str, float],
    constraints: Dict[str, Dict]
) -> Tuple[bool, List[str]]:
    """
    Validate metrics against task constraints

    Returns:
        (is_valid, list_of_violations)
    """
    violations = []

    for metric_name, constraint in constraints.items():
        current_value = metrics.get(metric_name)

        if constraint['operation'] == 'maintain':
            min_value = constraint.get('min')
            if current_value < min_value:
                violations.append(
                    f"{metric_name}={current_value:.4f} < {min_value:.4f}"
                )

        elif constraint['operation'] == 'improve':
            target_value = constraint.get('target')
            if current_value < target_value:
                violations.append(
                    f"{metric_name}={current_value:.4f} < target {target_value:.4f}"
                )

    return len(violations) == 0, violations
```

## Example Task Definitions

### Task 1: Balanced Improvement
```python
{
    'name': 'balanced_improvement',
    'target_metric': 'avg_f1',
    'target_value': 0.30,
    'constraints': {
        'avg_auc': {'min': 0.75, 'operation': 'maintain'},
        'avg_recall': {'min': 0.40, 'operation': 'maintain'},
        'avg_precision': {'min': 0.22, 'operation': 'maintain'}
    }
}
```

### Task 2: AUC Focus
```python
{
    'name': 'maximize_auc',
    'target_metric': 'avg_auc',
    'target_value': 0.82,
    'constraints': {
        'avg_f1': {'min': 0.25, 'operation': 'maintain'}
    }
}
```

### Task 3: Rare Disease Focus
```python
{
    'name': 'rare_disease_improvement',
    'target_metric': 'rare_disease_f1',
    'target_value': 0.20,
    'constraints': {
        'avg_f1': {'min': 0.25, 'operation': 'maintain'},
        'common_disease_auc': {'min': 0.78, 'operation': 'maintain'}
    },
    'focus_diseases': ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema']
}
```

## Monitoring Dashboard

```
Task-Aware Progress:
┌──────────────────────────────────────────────────────────────┐
│ Current Task: Improve AUC (target: 0.81)                     │
│ Constraint: F1 ≥ 0.25                                        │
│                                                               │
│ Progress:                                                     │
│ ├─ Iteration 57: AUC=0.7955, F1=0.2697 ✓                   │
│ ├─ Iteration 58: AUC=0.7904, F1=0.2783 ✓ (best task F1!)   │
│ ├─ Iteration 59: AUC=0.7913, F1=0.2690 ✓                   │
│ └─ Iteration 60: AUC=0.7950, F1=0.2680 ✓ (task-best AUC!)  │
│                                                               │
│ Task-Specific Best: Iteration 60                             │
│   • AUC: 0.7950 (target: 0.81, gap: -0.015)                │
│   • F1: 0.2680 (constraint: ≥0.25, margin: +0.018) ✓       │
│                                                               │
│ Warm Start Plan (Iter 61):                                   │
│   Source: Iteration 60 (task-best)                           │
│   Reasoning: "Highest AUC while maintaining F1 constraint.   │
│               Small gap to target (0.015). Continue progress."│
└──────────────────────────────────────────────────────────────┘
```

## Benefits of Task-Aware Warm Start

1. **Constraint Satisfaction** ✅
   - Never use models that violate task constraints
   - Maintain balance while optimizing target

2. **Task-Specific Optimization** 🎯
   - Use best model for specific objective
   - Not just global best

3. **Safe Exploration** 🛡️
   - Can explore without harming critical metrics
   - Automatic rollback if constraints violated

4. **Faster Convergence** ⚡
   - Start from relevant baseline
   - Don't waste time re-learning

5. **Multi-Objective Awareness** 🎭
   - Balance multiple goals
   - Trade-off transparency

## Summary

The warm start system is now **fully integrated with your task system**:

✅ Tracks task-specific best iterations
✅ Validates constraints before recording
✅ AI makes task-aware decisions
✅ Prevents harming constrained metrics
✅ Uses most relevant starting point for task

**Key Innovation**: The system understands "improve AUC without harming F1" and makes intelligent decisions accordingly!
