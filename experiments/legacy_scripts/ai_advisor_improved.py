"""
Improved AI Advisor for Auto-Improvement Loop
Includes constraints and clearer optimization objectives
"""

import os
from typing import Dict, Any, Tuple
from openai import OpenAI
import json
import re


class ImprovedAIAdvisor:
    """
    Improved AI advisor with constraints and optimization objectives
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4", optimization_mode: str = "recall"):
        """
        Initialize the AI advisor

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4)
            optimization_mode: What to optimize for:
                - "recall": Maximize recall (medical focus - catch all diseases)
                - "f1": Maximize F1 score (balanced)
                - "auc": Maximize AUC (overall quality)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.optimization_mode = optimization_mode

    def analyze_results(
        self,
        config: Dict[str, Any],
        results_df,
        comparison_df,
        iteration: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze results and suggest improvements with constraints

        Returns:
            (analysis_text, suggested_changes_dict)
        """
        # Create the analysis prompt
        prompt = self._create_analysis_prompt(config, results_df, comparison_df, iteration)

        # Get AI response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )

        response_text = response.choices[0].message.content

        # Parse suggestions from response
        suggested_changes = self._parse_suggestions(response_text)

        return response_text, suggested_changes

    def _get_system_prompt(self) -> str:
        """Get the system prompt with constraints and objectives"""

        # Define optimization objective
        objective_prompts = {
            "recall": """
PRIMARY OPTIMIZATION GOAL: Maximize Average Recall (Sensitivity)
CONSTRAINT: Maintain Average AUC > 0.75

RATIONALE: In medical diagnosis, missing a disease (false negative) is more
dangerous than a false alarm (false positive). We want to catch as many
positive cases as possible while maintaining reasonable overall quality.

TARGET METRICS:
- Avg Recall: Maximize (current best: 0.138)
- Avg AUC: Must stay > 0.75
- Avg F1: Secondary metric (improvement is good)
""",
            "f1": """
PRIMARY OPTIMIZATION GOAL: Maximize Average F1 Score
CONSTRAINT: Maintain Average AUC > 0.75

RATIONALE: F1 score balances precision and recall, giving us good
performance without too many false positives or false negatives.

TARGET METRICS:
- Avg F1: Maximize (current best: 0.108)
- Avg AUC: Must stay > 0.75
- Avg Recall: Secondary metric
""",
            "auc": """
PRIMARY OPTIMIZATION GOAL: Maximize Average AUC
CONSTRAINT: Maintain Average Recall > 0.10

RATIONALE: AUC measures overall classification quality across all thresholds.
We want the best overall model while ensuring we catch at least some diseases.

TARGET METRICS:
- Avg AUC: Maximize (current best: 0.795)
- Avg Recall: Must stay > 0.10
- Avg F1: Secondary metric
"""
        }

        return f"""You are an expert in medical image classification and deep learning optimization.
You are helping to optimize a chest X-ray disease classification model.

DATASET CHARACTERISTICS:
- 14 disease classes (multi-label classification)
- EXTREME class imbalance (Hernia: 0.17% positive, Infiltration: 24.5% positive)
- Medical X-ray images (224x224)
- Patient-level data split (GroupShuffleSplit)
- ~112,000 training images, ~25,000 test images

CURRENT ARCHITECTURE:
- Modified DenseNet-121 with dropout
- FocalLoss with class-specific weights
- Rare-class augmentation strategy
- Per-class threshold optimization

{objective_prompts[self.optimization_mode]}

CRITICAL CONSTRAINTS - YOU MUST FOLLOW THESE:

1. LEARNING RATE CONSTRAINTS:
   - MUST stay between 0.0001 and 0.005
   - Don't reduce below 0.0001 (model won't converge)
   - Don't increase above 0.005 (unstable training)

2. DROPOUT CONSTRAINTS:
   - MUST stay between 0.2 and 0.5
   - Don't oscillate (if you increase, don't decrease next iteration)
   - Stick with values that work

3. FOCAL LOSS GAMMA CONSTRAINTS:
   - MUST stay between 1.0 and 3.0
   - Don't oscillate wildly
   - gamma=1.0 is like standard cross-entropy
   - gamma=3.0 focuses heavily on hard examples

4. EPOCH CONSTRAINTS:
   - Reasonable range: 20-100 epochs
   - With early stopping, more is usually fine

5. CONSISTENCY REQUIREMENT:
   - Don't reverse recent changes without clear evidence
   - If iteration N improved, build on it for N+1
   - If iteration N failed, explain why and try different approach
   - DON'T just toggle parameters back and forth

6. IMPLEMENTATION REQUIREMENT:
   - ALL suggestions must be implementable with current config schema
   - Focus on parameters that exist in the config file
   - Augmentation changes ARE supported and SHOULD be suggested

AVAILABLE PARAMETERS TO TUNE:

model:
  - dropout_rate (0.2 to 0.5)
  - architecture (ModifiedDenseNetWithDropOut or ModifiedDenseNet)
  - backbone (densenet121 or densenet169)

training:
  - learning_rate (0.0001 to 0.005)
  - num_epochs (20 to 100)
  - batch_size (16, 32, 64, 128)
  - early_stopping.patience (5 to 20)
  - early_stopping.min_delta (0.001 to 0.01)
  - optimizer.weight_decay (0.0 to 0.1)
  - scheduler.patience (3 to 10)
  - scheduler.factor (0.1 to 0.9)

loss:
  - gamma (1.0 to 3.0)
  - type (FocalLoss or WeightedBCE) - MUST use exact capitalization
  - use_dynamic_weights (true or false) - enable dynamic class weight computation

augmentation:
  - rare_class.rotation_degrees (0 to 20)
  - rare_class.horizontal_flip_prob (0.0 to 1.0)
  - rare_class.affine_translate ([0.0, 0.0] to [0.15, 0.15])
  - rare_class.color_jitter.brightness (0.0 to 0.5)
  - rare_class.color_jitter.contrast (0.0 to 0.5)

evaluation:
  - threshold_optimization (f1_score, recall, precision, per_class_f1_score)

ANALYSIS REQUIREMENTS:

1. Start with ANALYSIS section:
   - What's working? (be specific with metrics)
   - What's not working? (be specific)
   - Identify the limiting factor (class imbalance, convergence, overfitting, etc.)
   - Reference previous iterations' results if available
   - DON'T just repeat generic statements about class imbalance

2. Provide SUGGESTED_CHANGES in this JSON format:
   ```json
   {{
     "model": {{
       "dropout_rate": value
     }},
     "training": {{
       "learning_rate": value,
       "num_epochs": value
     }},
     "loss": {{
       "gamma": value
     }},
     "augmentation": {{
       "rare_class": {{
         "rotation_degrees": value,
         "horizontal_flip_prob": value
       }}
     }},
     "reasoning": "Detailed explanation of WHY each change will help,
                   HOW it addresses the specific problems identified,
                   and WHY it's consistent with previous iterations"
   }}
   ```

3. Be SPECIFIC in reasoning:
   - Don't say "increase epochs to give model more time"
   - Say "increase epochs from X to Y because early stopping triggered at epoch Z,
     suggesting the model hadn't converged yet"
   - Reference actual metrics from the results

4. AVOID these common mistakes:
   - ❌ Reducing learning rate below 0.0001
   - ❌ Oscillating dropout (0.3 → 0.5 → 0.3 → 0.5)
   - ❌ Oscillating gamma (2.0 → 3.0 → 1.5 → 2.5)
   - ❌ Generic advice without specific reasoning
   - ❌ Suggesting changes that contradict previous iteration without explanation

5. GOOD suggestions include:
   - ✅ More aggressive augmentation for rare classes
   - ✅ Adjusting threshold_optimization strategy
   - ✅ Trying weight_decay for regularization
   - ✅ Adjusting early_stopping.patience based on convergence
   - ✅ Consistent parameter adjustments that build on what's working

Remember: You're optimizing for {self.optimization_mode.upper()}. Every suggestion
should clearly explain how it will improve this metric while respecting constraints.
"""

    def _create_analysis_prompt(
        self,
        config: Dict[str, Any],
        results_df,
        comparison_df,
        iteration: int
    ) -> str:
        """Create the detailed analysis prompt"""

        # Calculate average metrics from results
        avg_auc = results_df['AUC'].mean()
        avg_recall = results_df['Recall'].mean()
        avg_precision = results_df['Precision'].mean()
        avg_f1 = results_df['F1_Score'].mean()

        # Get per-class results
        results_summary = results_df[['Label', 'AUC', 'Recall', 'Precision', 'F1_Score']].to_string()

        # Get comparison with baseline
        comparison_summary = comparison_df.to_string() if comparison_df is not None else "No comparison available"

        # Format current config
        current_config_str = f"""
CURRENT CONFIGURATION (Iteration {iteration}):

Model:
  - Architecture: {config['model']['architecture']}
  - Backbone: {config['model']['backbone']}
  - Dropout Rate: {config['model']['dropout_rate']}

Training:
  - Learning Rate: {config['training']['learning_rate']}
  - Batch Size: {config['training']['batch_size']}
  - Num Epochs: {config['training']['num_epochs']}
  - Early Stopping Patience: {config['training']['early_stopping']['patience']}
  - Optimizer Weight Decay: {config['training']['optimizer'].get('weight_decay', 0.0)}

Loss:
  - Type: {config['loss']['type']}
  - Gamma: {config['loss']['gamma']}

Augmentation (Rare Class):
  - Rotation Degrees: {config['augmentation']['rare_class']['rotation_degrees']}
  - Horizontal Flip Prob: {config['augmentation']['rare_class']['horizontal_flip_prob']}
  - Affine Translate: {config['augmentation']['rare_class']['affine_translate']}
  - Color Jitter Brightness: {config['augmentation']['rare_class']['color_jitter']['brightness']}
  - Color Jitter Contrast: {config['augmentation']['rare_class']['color_jitter']['contrast']}

Evaluation:
  - Threshold Optimization: {config['evaluation'].get('threshold_optimization', 'fixed')}
"""

        prompt = f"""
ITERATION {iteration} RESULTS:

{current_config_str}

PERFORMANCE METRICS:
- Average AUC: {avg_auc:.4f}
- Average Recall: {avg_recall:.4f}
- Average Precision: {avg_precision:.4f}
- Average F1 Score: {avg_f1:.4f}

PER-CLASS RESULTS:
{results_summary}

COMPARISON WITH BASELINE:
{comparison_summary}

ANALYSIS TASK:
1. Analyze these results in detail
2. Identify what's working and what's not
3. Suggest specific configuration changes for iteration {iteration + 1}
4. Follow all constraints specified in the system prompt
5. Provide clear reasoning for each suggested change
6. Focus on improving {self.optimization_mode.upper()}

Remember:
- Stay within the specified parameter ranges
- Don't oscillate parameters without clear reasoning
- Build on what's working from previous iterations
- Be specific, not generic
"""

        return prompt

    def _parse_suggestions(self, response_text: str) -> Dict[str, Any]:
        """Parse suggested changes from AI response"""
        # Look for JSON block in response
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        if matches:
            try:
                suggestions = json.loads(matches[0])

                # Validate constraints
                suggestions = self._validate_constraints(suggestions)

                return suggestions
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse JSON suggestions: {e}")
                return {}
        else:
            print("⚠️  Warning: No JSON suggestions found in AI response")
            return {}

    def _validate_constraints(self, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enforce constraints on suggestions"""
        validated = suggestions.copy()

        # Learning rate constraints
        if 'training' in validated and 'learning_rate' in validated['training']:
            lr = validated['training']['learning_rate']
            if lr < 0.0001:
                print(f"⚠️  Warning: Learning rate {lr} too low, clamping to 0.0001")
                validated['training']['learning_rate'] = 0.0001
            elif lr > 0.005:
                print(f"⚠️  Warning: Learning rate {lr} too high, clamping to 0.005")
                validated['training']['learning_rate'] = 0.005

        # Dropout constraints
        if 'model' in validated and 'dropout_rate' in validated['model']:
            dropout = validated['model']['dropout_rate']
            if dropout < 0.2:
                print(f"⚠️  Warning: Dropout {dropout} too low, clamping to 0.2")
                validated['model']['dropout_rate'] = 0.2
            elif dropout > 0.5:
                print(f"⚠️  Warning: Dropout {dropout} too high, clamping to 0.5")
                validated['model']['dropout_rate'] = 0.5

        # Gamma constraints
        if 'loss' in validated and 'gamma' in validated['loss']:
            gamma = validated['loss']['gamma']
            if gamma < 1.0:
                print(f"⚠️  Warning: Gamma {gamma} too low, clamping to 1.0")
                validated['loss']['gamma'] = 1.0
            elif gamma > 3.0:
                print(f"⚠️  Warning: Gamma {gamma} too high, clamping to 3.0")
                validated['loss']['gamma'] = 3.0

        # Epochs constraints
        if 'training' in validated and 'num_epochs' in validated['training']:
            epochs = validated['training']['num_epochs']
            if epochs < 20:
                print(f"⚠️  Warning: Epochs {epochs} too low, clamping to 20")
                validated['training']['num_epochs'] = 20
            elif epochs > 100:
                print(f"⚠️  Warning: Epochs {epochs} too high, clamping to 100")
                validated['training']['num_epochs'] = 100

        return validated


# Test function
def test_improved_advisor():
    """Test the improved AI advisor"""
    import pandas as pd

    advisor = ImprovedAIAdvisor(optimization_mode="recall")

    # Mock data for testing
    config = {
        'model': {
            'architecture': 'ModifiedDenseNetWithDropOut',
            'backbone': 'densenet121',
            'dropout_rate': 0.3
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 20,
            'early_stopping': {'patience': 5},
            'optimizer': {'weight_decay': 0.0}
        },
        'loss': {
            'type': 'FocalLoss',
            'gamma': 2.0
        },
        'augmentation': {
            'rare_class': {
                'rotation_degrees': 10,
                'horizontal_flip_prob': 0.5,
                'affine_translate': [0.05, 0.05],
                'color_jitter': {'brightness': 0.2, 'contrast': 0.2}
            }
        },
        'evaluation': {
            'threshold_optimization': 'per_class_f1_score'
        }
    }

    results_df = pd.DataFrame({
        'Label': ['Cardiomegaly', 'Emphysema'],
        'AUC': [0.85, 0.79],
        'Recall': [0.12, 0.15],
        'Precision': [0.45, 0.52],
        'F1_Score': [0.19, 0.23]
    })

    comparison_df = None

    print("Testing improved AI advisor...")
    print(f"Optimization mode: {advisor.optimization_mode}")
    print("\nThis test requires OPENAI_API_KEY to be set.")
    print("Skipping actual API call in test mode.")

    # Just test constraint validation
    test_suggestions = {
        'training': {
            'learning_rate': 0.00001,  # Too low
            'num_epochs': 150  # Too high
        },
        'model': {
            'dropout_rate': 0.8  # Too high
        },
        'loss': {
            'gamma': 0.5  # Too low
        }
    }

    validated = advisor._validate_constraints(test_suggestions)
    print("\nConstraint validation test:")
    print(f"Original: {test_suggestions}")
    print(f"Validated: {validated}")


if __name__ == "__main__":
    test_improved_advisor()
