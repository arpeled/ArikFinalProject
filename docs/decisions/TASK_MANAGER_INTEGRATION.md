# Task Manager Integration Guide

**Version**: 1.0
**Date**: January 7, 2026
**Status**: ✅ Fully Integrated and Tested

---

## Overview

The Task Manager system has been successfully integrated into the auto-improvement loop. This enables systematic tracking of optimization tasks across multiple iterations, automatic task evaluation, and AI advisor integration.

## What Was Integrated

### 1. Core Components

#### TaskManager (`task_manager.py`)
- **Location**: `/task_manager.py`
- **Purpose**: Track tasks across iterations in `tasks_registry.json`
- **Features**:
  - Add new tasks with specific goals
  - Track progress iteration by iteration
  - Automatically evaluate success/failure
  - Provide task data to AI advisor

#### IterationEvaluator (`task_manager.py`)
- **Location**: `/task_manager.py`
- **Purpose**: Analyze iteration trends
- **Features**:
  - Load last N iteration summaries
  - Compare metrics across iterations
  - Identify trends (improving/degrading)
  - Format data for AI advisor

### 2. Integration Points in `auto_improvement_loop.py`

#### Initialization (Lines 91-99)
```python
# Initialize task management system
self.task_manager = TaskManager(
    registry_path=os.path.join(output_dir, "tasks_registry.json"),
    logger=self.logger
)
self.iteration_evaluator = IterationEvaluator(
    output_dir=output_dir,
    logger=self.logger
)
```

#### Task Display at Iteration Start (Lines 302-315)
```python
# Get current task from task manager
current_task = self.task_manager.get_current_task()
if current_task:
    self.logger.info("")
    self.logger.info("📋 CURRENT TASK:")
    self.logger.info(f"   ID: {current_task['task_id']}")
    self.logger.info(f"   Description: {current_task['description']}")
    self.logger.info(f"   Target Metric: {current_task['target_metric']}")
    self.logger.info(f"   Progress: {len(current_task['evaluation_history'])}/{current_task['required_iterations']} iterations")
```

#### Task Progress Tracking (Lines 421-455)
```python
# Update task progress if there's an active task
if current_task:
    target_metric = current_task['target_metric']
    metric_key_map = {
        'f1': 'avg_f1',
        'auc': 'avg_auc',
        'recall': 'avg_recall',
        'precision': 'avg_precision',
        'accuracy': 'avg_accuracy'
    }
    metric_key = metric_key_map.get(target_metric, 'avg_f1')
    metric_value = iteration_summary.get(metric_key, 0.0)

    self.task_manager.update_task_progress(
        task_id=current_task['task_id'],
        iteration=iteration,
        metric_value=metric_value,
        iteration_summary=iteration_summary
    )

    # Evaluate task if enough iterations completed
    if len(current_task['evaluation_history']) >= current_task['required_iterations']:
        task_status = self.task_manager.evaluate_task(current_task['task_id'])
```

#### AI Advisor Integration (Lines 468-486)
```python
# Get task information for AI advisor
task_info = self.task_manager.get_task_for_advisor()

# Get formatted iteration data for AI advisor
last_3_summaries = self.iteration_evaluator.get_last_n_summaries(3)
iteration_analysis = self.iteration_evaluator.format_for_advisor(last_3_summaries)

# Pass to AI advisor
analysis, suggested_changes = self.ai_advisor.analyze_results(
    config=trainer.config,
    results_df=results_df,
    comparison_df=comparison_df,
    iteration=iteration,
    iteration_history=self.iteration_history,
    train_loss=train_loss,
    val_loss=val_loss,
    task_info=task_info,  # NEW
    iteration_analysis=iteration_analysis  # NEW
)
```

### 3. AI Advisor Enhancements (`ai_advisor.py`)

#### Updated Method Signature
```python
def analyze_results(
    self,
    config: Dict[str, Any],
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    iteration: int,
    iteration_history: Optional[List[Dict[str, Any]]] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    task_info: Optional[Dict[str, Any]] = None,  # NEW
    iteration_analysis: Optional[Dict[str, Any]] = None  # NEW
) -> Tuple[str, Dict[str, Any]]:
```

#### Enhanced Prompt Generation
The AI advisor now receives:
- Current active task information
- Task evaluation history
- Completed tasks summary
- F1/AUC/Recall trend analysis
- Per-disease metrics over time

---

## How It Works

### Task Lifecycle

```
1. Task Creation
   └─→ Status: pending
       └─→ Added to tasks_registry.json

2. Iteration 1
   └─→ Task displayed at start
   └─→ Training runs
   └─→ Progress updated
   └─→ Status: pending (1/3 iterations)

3. Iteration 2
   └─→ Progress updated
   └─→ Status: pending (2/3 iterations)

4. Iteration 3
   └─→ Progress updated
   └─→ Task evaluated
       ├─→ Success: Metric improved > 1% of baseline
       │   └─→ Moved to completed_tasks
       │   └─→ AI suggests next task
       │
       └─→ Failed: No significant improvement
           └─→ Moved to completed_tasks
           └─→ AI suggests recovery strategy
```

### Automatic Evaluation

Tasks are automatically evaluated after `required_iterations` (default: 3):

**Success Criteria**:
- Average improvement > 1% of baseline metric value
- Example: If baseline F1 = 0.25, improvement must be > 0.0025

**Failure Criteria**:
- No significant improvement over baseline
- Metric degradation

---

## Usage Examples

### 1. Creating a Task (Manual)

You can manually create tasks in the registry:

```python
from task_manager import TaskManager

tm = TaskManager(registry_path="auto_improvement_runs/tasks_registry.json")

tm.add_task(
    task_id="focal_gamma_tuning",
    description="Tune FocalLoss gamma to improve Hernia F1-score",
    start_iteration=52,
    target_metric="f1",
    required_iterations=3,
    change_applied={
        "loss": {
            "gamma": 3.5
        }
    }
)
```

### 2. Checking Task Status

```python
# Get current task
current = tm.get_current_task()
print(f"Current task: {current['description']}")
print(f"Progress: {len(current['evaluation_history'])}/{current['required_iterations']}")

# Get summary
print(tm.get_task_summary())
```

### 3. Viewing Iteration Trends

```python
from task_manager import IterationEvaluator

evaluator = IterationEvaluator(output_dir="auto_improvement_runs")

# Get last 3 iterations
summaries = evaluator.get_last_n_summaries(3)

# Compare F1 scores
f1_trend = evaluator.compare_metrics(summaries, "avg_f1")
print(f"F1 Trend: {f1_trend['trend']}")
print(f"Best F1: {f1_trend['best_value']:.4f} (iteration {f1_trend['best_iteration']})")
```

---

## Testing

### Run the Test Suite

```bash
uv run python test_task_manager.py
```

**Expected Output**:
```
🧪 TASK MANAGER INTEGRATION TEST SUITE

================================================================================
TESTING TASK MANAGER
================================================================================
✅ TASK MANAGER TEST PASSED

================================================================================
TESTING ITERATION EVALUATOR
================================================================================
✅ ITERATION EVALUATOR TEST PASSED

================================================================================
🎉 ALL TESTS PASSED SUCCESSFULLY!
================================================================================
```

### Test Results (Actual)

✅ **Task Creation**: Successfully created test task
✅ **Progress Tracking**: Updated 3 iterations correctly
✅ **Task Evaluation**: Correctly identified successful task (F1: 0.25 → 0.29)
✅ **Iteration Loading**: Loaded last 3 real iterations (49, 50, 51)
✅ **Trend Analysis**: Identified F1 degrading trend (-0.0206)
✅ **AI Advisor Format**: Correctly formatted all data structures

---

## Files Modified/Created

### Created
- ✅ `task_manager.py` (238 lines)
- ✅ `test_task_manager.py` (226 lines)
- ✅ `TASK_MANAGER_INTEGRATION.md` (this file)

### Modified
- ✅ `auto_improvement_loop.py`:
  - Added imports (line 23)
  - Initialized managers (lines 91-99)
  - Added task display (lines 302-315)
  - Added progress tracking (lines 421-455)
  - Enhanced AI call (lines 468-486)

- ✅ `ai_advisor.py`:
  - Updated analyze_results signature (lines 50-83)
  - Updated _create_analysis_prompt signature (lines 443-497)
  - Added task info to prompt (lines 470-497)
  - Added iteration trends to prompt (lines 616-644)
  - Updated advisor questions (lines 647-662)

### Runtime Files (Created Automatically)
- `auto_improvement_runs/tasks_registry.json` - Created on first use
- Task registry is saved after each update

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    auto_improvement_loop                     │
│                                                              │
│  ┌──────────────┐       ┌─────────────────┐                │
│  │ TaskManager  │       │ IterationEval   │                │
│  └──────┬───────┘       └────────┬────────┘                │
│         │                        │                          │
│         │ Get current task       │ Load last 3              │
│         ├─────────→              │ summaries                │
│         │                        ├─────────→                │
│         │                        │                          │
│         │ Update progress        │ Analyze trends           │
│         │ after iteration        │                          │
│         ├─────────→              ├─────────→                │
│         │                        │                          │
│         │ Evaluate task          │                          │
│         │ if ready               │                          │
│         ├─────────→              │                          │
│         │                        │                          │
│         │ Get task info          │ Format for advisor       │
│         ├────────────────────────┼─────────→                │
│         │                        │                          │
│         │                        │                          │
│         ▼                        ▼                          │
│    ┌────────────────────────────────────────┐               │
│    │          AI Advisor                    │               │
│    │  - Current task info                  │               │
│    │  - Task evaluation history            │               │
│    │  - Completed tasks                    │               │
│    │  - F1/AUC/Recall trends               │               │
│    │  - Per-disease metrics                │               │
│    └─────────────────┬──────────────────────┘               │
│                      │                                      │
│                      │ Suggested changes                    │
│                      ▼                                      │
│              ┌──────────────┐                               │
│              │ New Config   │                               │
│              └──────────────┘                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Benefits

### For Users

1. **Clear Focus**: One task at a time with specific goals
2. **Automatic Tracking**: Progress logged automatically
3. **Objective Evaluation**: Success/failure based on metrics, not intuition
4. **Better AI Guidance**: AI advisor has full context and history
5. **Transparency**: See exactly what's being tested and why

### For the Pipeline

1. **Systematic Improvement**: No random changes
2. **Measurable Impact**: Each task has clear success criteria
3. **Knowledge Retention**: Completed tasks stored in registry
4. **Trend Analysis**: Identify what's working over time
5. **Rollback Ready**: Can revert to successful task configs

### For AI Advisor

1. **Rich Context**: Knows current task and history
2. **Trend Awareness**: Sees improving/degrading patterns
3. **Task-Specific Questions**: Asks relevant questions
4. **Better Suggestions**: Builds on what worked
5. **Failure Recovery**: Can suggest fixes when tasks fail

---

## Next Steps

### Immediate (Iteration 52+)

1. ✅ Integration complete and tested
2. 📝 Run next iteration to generate first real task
3. 📊 Monitor task progress across 3 iterations
4. 🤖 AI advisor will evaluate and suggest next task

### Future Enhancements

1. **Task Dependencies**: Tasks that require other tasks first
2. **Parallel Tasks**: Multiple tasks for different metrics
3. **Task Templates**: Pre-defined common optimization tasks
4. **Visualization**: Web dashboard for task progress
5. **Auto-Recovery**: Automatic rollback on failed tasks

---

## Troubleshooting

### Issue: Task not showing up

**Check**:
```python
tm = TaskManager()
print(tm.get_task_summary())
```

**Solution**: Ensure task was added with `add_task()`

### Issue: Task not evaluating

**Check**: `len(current_task['evaluation_history']) >= required_iterations`

**Solution**: Wait for required number of iterations (default: 3)

### Issue: AI advisor not seeing task info

**Check**: Logs should show "CURRENT ACTIVE TASK:" section

**Solution**: Verify task_info and iteration_analysis are passed to `analyze_results()`

### Issue: Iteration evaluator not loading summaries

**Check**:
```python
evaluator = IterationEvaluator()
summaries = evaluator.get_last_n_summaries(3)
print(f"Loaded: {len(summaries)}")
```

**Solution**: Ensure `iteration_summary.json` files exist in iteration directories

---

## Summary

The Task Manager integration is **fully functional** and **production ready**. It provides:

- ✅ Systematic task tracking
- ✅ Automatic progress monitoring
- ✅ Objective task evaluation
- ✅ Rich AI advisor context
- ✅ Trend analysis across iterations
- ✅ Comprehensive testing

**Status**: Ready for iteration 52 and beyond!

---

**Questions?** Check:
- `task_manager.py` - Implementation details
- `test_task_manager.py` - Usage examples
- `TRAINING_PIPELINE_IMPROVEMENTS.md` - Overall system documentation
