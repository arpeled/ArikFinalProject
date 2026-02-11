"""
AI Advisor for Chest X-Ray Classification Pipeline
Uses OpenAI API to analyze results and suggest configuration improvements

Extended for role-based auto-improvement with dual-lineage strategy:
- Iteration 12 as AUC anchor
- Iteration 58 as F1 anchor
- Hard-coded rules that AI cannot override
"""

import os
import sys

# Add automation/ and core/ directories to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
_core_dir = os.path.join(_project_root, 'core')
for _d in [_this_dir, _core_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)
import json
import pandas as pd
import time
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI
import yaml

# Import role-based constants and phased protocol
try:
    from iteration_baselines import (
        IterationRole,
        OptimizationPhase,
        ITERATION_12_AUC_ANCHOR,
        ITERATION_58_F1_ANCHOR,
        ITERATION_12_EXACT_CONFIG,
        PHASE_CRITERIA,
        TRAIN_AUC_RULES,
        RECOVER_F1_RULES,
        DECISION_RULES,
        validate_ai_suggestion,
        get_parent_for_role,
        determine_current_phase,
        generate_ai_advisory_decision,
        get_phase1_exact_config,
        get_enforced_config_for_phase,
        AIAdvisoryDecision,
        # Phase 5 imports
        PHASE_5_BASELINE_ITERATION,
        PHASE_5_BASELINE_MACRO_AUC,
        PHASE_5_BASELINE_MODEL_PATH,
        PHASE_5_BASELINE_PER_CLASS_AUC,
        PHASE_5_TARGET_DISEASES,
        PHASE_5_STABLE_DISEASES,
        PHASE_5_CONFIG,
        compute_phase5_auc_deltas,
        check_phase5_stop_conditions,
        get_phase5_config
    )
    ROLE_BASED_ADVISOR = True
    PHASED_PROTOCOL_ENABLED = True
except ImportError:
    ROLE_BASED_ADVISOR = False
    PHASED_PROTOCOL_ENABLED = False
    print("Warning: iteration_baselines not available, role-based advisor disabled")

# Try to import Anthropic for fallback
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AIAdvisor:
    """AI-powered advisor for hyperparameter optimization"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2", use_claude_fallback: bool = True):
        """
        Initialize AI Advisor

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use (gpt-5.2, gpt-5.1, gpt-4o, etc.)
            use_claude_fallback: If True, fall back to Claude if OpenAI fails
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.use_claude_fallback = use_claude_fallback

        # Initialize Claude client if available
        self.claude_client = None
        if use_claude_fallback and ANTHROPIC_AVAILABLE:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
            else:
                print("⚠️  ANTHROPIC_API_KEY not set. Claude fallback disabled.")

        # Track what has been tried in previous iterations
        self.tried_strategies = set()

        # Track which model was used for the last analysis
        self.last_model_used = None

        # Track current config phase for system prompt selection
        self._current_config_phase = ''

        # Define systematic improvement strategy queue
        self.improvement_strategies = [
            "threshold_optimization",  # Fix prediction threshold first
            "dynamic_class_weights",   # Then try dynamic weights
            "class_balancing_oversample",  # Then oversample minority classes
            "focal_loss_tuning",       # Then tune focal loss gamma
            "augmentation_increase",   # Then increase augmentation
            "learning_rate_adjust",    # Then adjust learning rate
            "epochs_increase",         # Then increase epochs if needed
            "dropout_tuning",          # Then tune dropout
            "weight_decay",            # Then try weight decay
            "ensemble_techniques"      # Finally, consider ensemble
        ]

    def analyze_results(
        self,
        config: Dict[str, Any],
        results_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        iteration: int,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        task_info: Optional[Dict[str, Any]] = None,
        iteration_analysis: Optional[Dict[str, Any]] = None,
        iteration_role: Optional['IterationRole'] = None,
        parent_iteration: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze results and suggest improvements

        Args:
            config: Current configuration dictionary
            results_df: Test results DataFrame
            comparison_df: Baseline comparison DataFrame
            iteration: Current iteration number
            iteration_history: List of previous iteration summaries (optional)
            train_loss: Final training loss (optional)
            val_loss: Final validation loss (optional)
            task_info: Task registry information (optional)
            iteration_analysis: Detailed iteration analysis from last 3 runs (optional)
            iteration_role: Current iteration role (TRAIN_AUC, RECOVER_F1, etc.)
            parent_iteration: Parent iteration number (for lineage tracking)

        Returns:
            Tuple of (analysis text, suggested config changes)
        """
        # Store current role for use in prompts
        self._current_role = iteration_role
        self._current_parent = parent_iteration

        # Store current config phase for system prompt selection
        self._current_config_phase = config.get('metadata', {}).get('phase', '')

        # Prepare the analysis prompt
        prompt = self._create_analysis_prompt(
            config, results_df, comparison_df, iteration, iteration_history,
            train_loss, val_loss, task_info, iteration_analysis
        )

        # Call OpenAI API with retry logic
        response = self._call_openai_with_retry(prompt)

        # Extract response
        response_text = response.choices[0].message.content

        # Check if response is empty or None
        if not response_text or len(response_text.strip()) == 0:
            print(f"⚠️  WARNING: Received empty response from {self.last_model_used}")
            print(f"   Response object: {response}")
            print(f"   Response choices: {response.choices}")
            print(f"   Attempting Claude fallback...")

            # Try Claude fallback
            if self.claude_client:
                try:
                    response = self._call_claude(prompt)
                    response_text = response.choices[0].message.content
                    if not response_text or len(response_text.strip()) == 0:
                        raise Exception("Claude also returned empty response")
                except Exception as e:
                    print(f"❌ Claude fallback failed: {e}")
                    # Return a minimal analysis
                    response_text = "ERROR: AI analysis failed - both OpenAI and Claude returned empty responses.\nPlease check API keys and model availability."
            else:
                # Return a minimal analysis
                response_text = "ERROR: AI analysis failed - OpenAI returned empty response and Claude fallback not available.\nPlease check API configuration."

        # Prepend model information to analysis
        import datetime
        model_header = "=" * 80 + "\n"
        model_header += "AI ANALYSIS METADATA\n"
        model_header += "=" * 80 + "\n"
        model_header += f"Model Used: {self.last_model_used or 'Unknown'}\n"
        model_header += f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        model_header += f"Iteration: {iteration}\n"
        model_header += "=" * 80 + "\n\n"

        response_text_with_header = model_header + response_text

        # Parse suggested changes
        suggested_changes = self._parse_suggestions(response_text)

        # Track what strategy was suggested
        self._track_suggested_strategy(suggested_changes)

        return response_text_with_header, suggested_changes

    def _track_suggested_strategy(self, suggested_changes: Dict[str, Any]):
        """Track which strategies have been tried"""
        if not suggested_changes or 'reasoning' not in suggested_changes:
            return

        reasoning = suggested_changes.get('reasoning', '').lower()

        # Check which strategies were applied
        if 'threshold' in reasoning or 'threshold_optimization' in suggested_changes.get('evaluation', {}):
            self.tried_strategies.add('threshold_optimization')

        if 'dynamic' in reasoning and 'weight' in reasoning:
            self.tried_strategies.add('dynamic_class_weights')

        if 'balanc' in reasoning or 'oversample' in reasoning or 'data_balancing' in suggested_changes:
            self.tried_strategies.add('class_balancing_oversample')

        if 'gamma' in suggested_changes.get('loss', {}):
            self.tried_strategies.add('focal_loss_tuning')

        if 'augmentation' in suggested_changes:
            self.tried_strategies.add('augmentation_increase')

        if 'learning_rate' in suggested_changes.get('training', {}):
            self.tried_strategies.add('learning_rate_adjust')

        if 'num_epochs' in suggested_changes.get('training', {}):
            self.tried_strategies.add('epochs_increase')

        if 'dropout' in suggested_changes.get('model', {}):
            self.tried_strategies.add('dropout_tuning')

        if 'weight_decay' in suggested_changes.get('training', {}).get('optimizer', {}):
            self.tried_strategies.add('weight_decay')

    def get_next_strategy(self) -> str:
        """
        Get the next untried strategy from the improvement queue

        Returns:
            Strategy name, or 'custom' if all strategies have been tried
        """
        for strategy in self.improvement_strategies:
            if strategy not in self.tried_strategies:
                return strategy

        return 'custom'  # All systematic strategies tried, AI can be creative

    def recommend_warm_start(
        self,
        current_iteration: int,
        previous_iteration_data: Dict[str, Any],
        best_iterations_data: Dict[str, Any],
        proposed_config: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ask AI to recommend which iteration to warm start from

        Args:
            current_iteration: Current iteration number
            previous_iteration_data: Data from previous iteration
            best_iterations_data: Data about all best iterations
            proposed_config: Configuration proposed for current iteration
            previous_config: Configuration from previous iteration

        Returns:
            Dict with keys:
                - warm_start_from: 'iteration_XX' or 'cold_start'
                - reasoning: Explanation (2-3 sentences)
                - confidence: Float 0.0-1.0
                - expected_benefit: String description
        """
        # Format data for AI
        prompt = self._create_warm_start_prompt(
            current_iteration,
            previous_iteration_data,
            best_iterations_data,
            proposed_config,
            previous_config
        )

        try:
            # Call OpenAI API
            response = self._call_openai_with_retry(prompt)
            response_text = response.choices[0].message.content

            # Parse JSON response
            recommendation = self._parse_json_from_response(response_text)

            # Validate response
            if not recommendation or 'warm_start_from' not in recommendation:
                print("⚠️  Failed to parse warm start recommendation, defaulting to cold start")
                recommendation = {
                    'warm_start_from': 'cold_start',
                    'reasoning': 'Failed to parse AI recommendation',
                    'confidence': 0.5,
                    'expected_benefit': 'unknown'
                }

            # Log the decision to file
            self._log_warm_start_decision(
                current_iteration,
                previous_iteration_data,
                best_iterations_data,
                recommendation
            )

            return recommendation

        except Exception as e:
            print(f"⚠️  Error getting warm start recommendation: {e}")
            print("   Defaulting to cold start")

            recommendation = {
                'warm_start_from': 'cold_start',
                'reasoning': f'Error in AI recommendation: {str(e)}',
                'confidence': 0.0,
                'expected_benefit': 'none'
            }

            # Log the error decision
            self._log_warm_start_decision(
                current_iteration,
                previous_iteration_data,
                best_iterations_data,
                recommendation
            )

            return recommendation

    def _call_openai_with_retry(self, prompt: str, max_retries: int = 3):
        """
        Call OpenAI API with exponential backoff retry logic
        Falls back to Claude on any error after retries

        Args:
            prompt: The prompt to send to OpenAI
            max_retries: Maximum number of retry attempts before fallback

        Returns:
            Response object (OpenAI or Claude format)
        """
        import openai

        last_error = None

        for attempt in range(max_retries):
            try:
                # Use max_completion_tokens for GPT-5+ models, max_tokens for older models
                token_param = {}
                if self.model.startswith('gpt-5'):
                    token_param['max_completion_tokens'] = 2000
                else:
                    token_param['max_tokens'] = 2000

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    **token_param
                )
                # Track which model was used
                self.last_model_used = f"OpenAI {self.model}"

                # Log response info for debugging
                print(f"✅ OpenAI API call successful")
                print(f"   Model: {self.model}")
                print(f"   Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
                print(f"   Content length: {len(response.choices[0].message.content) if response.choices and response.choices[0].message.content else 0}")

                return response

            except openai.BadRequestError as e:
                # Check if it's a context length error - fallback immediately
                error_msg = str(e)
                if "context_length_exceeded" in error_msg or "maximum context length" in error_msg:
                    print(f"⚠️  OpenAI context length exceeded. Falling back to Claude...")
                    if self.claude_client:
                        return self._call_claude(prompt)
                    else:
                        print("❌ Claude not available. Trying to reduce prompt...")
                        short_prompt = self._create_short_prompt(prompt)
                        if len(short_prompt) < len(prompt):
                            print(f"Retrying with shortened prompt ({len(short_prompt)} chars vs {len(prompt)})")
                            prompt = short_prompt
                            continue
                        raise

                # Other bad request errors - store and continue to fallback
                last_error = e
                print(f"⚠️  OpenAI BadRequest error: {e}")
                break  # Don't retry bad requests, go to fallback

            except openai.APIConnectionError as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                if attempt < max_retries - 1:
                    print(f"⚠️  Connection error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️  Failed to connect to OpenAI after {max_retries} attempts.")
                    break  # Go to fallback

            except openai.RateLimitError as e:
                error_msg = str(e)
                # Check if it's a token limit error (prompt too large) - fallback immediately
                if "tokens per min" in error_msg or "Requested" in error_msg:
                    print(f"⚠️  OpenAI token limit exceeded (prompt too large). Falling back to Claude...")
                    if self.claude_client:
                        return self._call_claude(prompt)
                    else:
                        print("❌ Claude not available. Trying to reduce prompt...")
                        short_prompt = self._create_short_prompt(prompt)
                        if len(short_prompt) < len(prompt):
                            print(f"Retrying with shortened prompt ({len(short_prompt)} chars vs {len(prompt)})")
                            prompt = short_prompt
                            continue
                        raise

                # Regular rate limit (too many requests) - retry with backoff
                last_error = e
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"⚠️  Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️  Rate limit exceeded after {max_retries} attempts.")
                    break  # Go to fallback

            except Exception as e:
                # Catch-all for any other errors
                last_error = e
                print(f"⚠️  Unexpected error calling OpenAI API: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️  Failed after {max_retries} attempts.")
                    break  # Go to fallback

        # If we get here, all retries failed - try Claude fallback
        if last_error and self.claude_client:
            print(f"🔄 All OpenAI attempts failed. Falling back to Claude...")
            try:
                return self._call_claude(prompt)
            except Exception as claude_error:
                print(f"❌ Claude fallback also failed: {claude_error}")
                print(f"❌ Both providers failed. Original OpenAI error: {last_error}")
                raise last_error
        elif last_error:
            # No Claude available, try shortened prompt as last resort
            print("❌ Claude not available. Trying shortened prompt as last resort...")
            short_prompt = self._create_short_prompt(prompt)
            if len(short_prompt) < len(prompt):
                print(f"Retrying with shortened prompt ({len(short_prompt)} chars vs {len(prompt)})")
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": short_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return response
                except Exception as e:
                    print(f"❌ Shortened prompt also failed: {e}")
                    raise last_error
            else:
                raise last_error

        # Should never reach here, but just in case
        raise Exception("Failed to get response from any provider")

    def _call_claude(self, prompt: str):
        """
        Call Claude API as fallback

        Args:
            prompt: The prompt to send

        Returns:
            Response in OpenAI-compatible format
        """
        print("🤖 Using Claude Sonnet 4 (200k context)...")

        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.7,
                system=self._get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Track which model was used
            self.last_model_used = "Anthropic Claude Sonnet 4 (claude-sonnet-4-20250514)"

            # Convert Claude response to OpenAI format
            class ClaudeResponse:
                def __init__(self, claude_resp):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': claude_resp.content[0].text
                        })()
                    })()]

            return ClaudeResponse(response)

        except Exception as e:
            print(f"❌ Claude API error: {e}")
            raise

    def _create_short_prompt(self, full_prompt: str) -> str:
        """
        Create a shortened version of the prompt by removing verbose sections

        Args:
            full_prompt: Full prompt text

        Returns:
            Shortened prompt
        """
        # Remove detailed iteration history, keep only summary
        lines = full_prompt.split('\n')
        short_lines = []
        skip_section = False

        for line in lines:
            # Skip verbose sections
            if "PREVIOUS ITERATIONS SUMMARY:" in line:
                skip_section = True
                short_lines.append(line)
                short_lines.append("(History truncated to save context - showing only recent metrics)")
                continue
            elif "DETAILED INFO FOR LAST" in line:
                skip_section = True
                continue
            elif line.startswith("CURRENT CONFIGURATION:") or line.startswith("TEST RESULTS:"):
                skip_section = False

            if not skip_section:
                short_lines.append(line)

        return '\n'.join(short_lines)

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the AI advisor.

        STRICT PHASED PROTOCOL:
        - PHASE 1: Exact reproduction of iteration 12 (AUC target)
        - PHASE 2: Threshold calibration only (F1 injection from iter58)
        - SMOTE_HEAD: Feature-space SMOTE experiments (separate from Phase 5)
        - RULE: First reproduce. Then calibrate. Never mix.
        """
        # Check if we're in SMOTE_HEAD phase (takes precedence over Phase 5)
        current_phase = getattr(self, '_current_config_phase', '')
        if current_phase == 'SMOTE_HEAD':
            return self._get_smote_head_system_prompt()

        # Check if phased protocol is enabled
        if PHASED_PROTOCOL_ENABLED:
            return self._get_phased_protocol_system_prompt()
        else:
            return self._get_legacy_system_prompt()

    def _get_smote_head_system_prompt(self) -> str:
        """
        System prompt for SMOTE_HEAD phase - feature-space SMOTE experiments.

        This is a SEPARATE phase from Phase 5 (AUC improvement).
        AI may ONLY suggest changes to smote.sampling_ratio.
        """
        return """You are an AI research advisor analyzing feature-space SMOTE experiments
for chest X-ray classification.

================================================================================
SMOTE_HEAD PHASE - FEATURE-SPACE SMOTE EXPERIMENTS
================================================================================

This phase is SEPARATE from Phase 5 (AUC improvement). You are evaluating
feature-space SMOTE experiments targeting rare classes (especially Hernia).

IMPORTANT CONSTRAINTS:
1. You may ONLY suggest changes to: smote.sampling_ratio (4 → 8)
2. You MUST NOT suggest:
   - Phase 5 / auc_improvement changes
   - Class weights / loss function changes
   - Threshold optimization changes
   - Backbone unfreezing
   - Any other hyperparameter changes

SMOTE EXPERIMENT RULES:
- Maximum 2 SMOTE probes allowed (ratio=4, then ratio=8)
- If Macro AUC drops by >0.003 from baseline → STOP
- If Hernia F1 > 0 and Macro F1 stable → KEEP
- If Hernia F1 still 0 and Macro AUC stable → suggest ratio=8 (if probe 1)
- If 2 probes done or ratio=8 tested → STOP

RESPONSE FORMAT (JSON):
{
  "phase": "SMOTE_HEAD",
  "decision": "KEEP" | "STOP" | "PROBE_AGAIN",
  "reasoning": "...",
  "smote_config": {
    "sampling_ratio": 4 or 8
  },
  "hernia_metrics_summary": "...",
  "macro_metrics_summary": "..."
}

DO NOT mention Phase 5, iteration 84, hard-negative emphasis, or AUC improvement.
This is a SMOTE experiment, not a Phase 5 iteration.
"""

    def _get_phased_protocol_system_prompt(self) -> str:
        """
        Phase 5 system prompt for class-specific AUC improvement.

        POST-ITERATION 84: Phase 1 concluded (not reproducible), now in Phase 5.
        """
        target_diseases_str = ", ".join(PHASE_5_TARGET_DISEASES)
        stable_diseases_str = ", ".join(PHASE_5_STABLE_DISEASES)

        return f"""You are an AI research advisor controlling an automated experiment loop
for multi-label medical image classification.

================================================================================
PHASE 5: CLASS-SPECIFIC AUC IMPROVEMENT (POST-ITERATION 84)
================================================================================

Key Insight (from 84 iterations):
AUC improvement is CLASS-SPECIFIC and achieved by NOISE SUPPRESSION, not recall forcing.

Baseline (Iteration 84):
- Macro AUC: {PHASE_5_BASELINE_MACRO_AUC:.4f}
- Model path: {PHASE_5_BASELINE_MODEL_PATH}

Target Diseases (lowest AUC, highest improvement potential):
{target_diseases_str}

Stable Diseases (must not regress):
{stable_diseases_str}

================================================================================
PHASE 5 RULES (MANDATORY)
================================================================================

GLOBAL RULE:
Do NOT modify backbone, optimizer, learning rate, or loss definition.
Only the auc_improvement block may change between iterations.

Allowed Configuration Changes:
- auc_improvement.enabled: true
- auc_improvement.strategy: hard_negative_emphasis
- auc_improvement.target_classes: [list of diseases]
- auc_improvement.hard_negative_threshold: 0.1 to 0.5
- auc_improvement.hard_negative_weight: 1.0 to 3.0

Forbidden Changes:
❌ Loss gamma / alpha changes
❌ Class re-weighting
❌ Data oversampling
❌ Threshold optimization (F1 is out of scope)
❌ Learning rate changes
❌ Backbone/architecture changes

================================================================================
MANDATORY CHECKS AFTER EACH ITERATION
================================================================================

1. Per-Class AUC Delta Analysis:
   - Compute ΔAUC for each disease vs baseline (iteration 84)
   - Flag improvements >= +0.01 as meaningful

2. Regression Detection:
   - Verify no stable disease loses more than -0.01 AUC
   - If violated → STOP further changes

3. Stability Assessment:
   - Confirm Macro AUC remains within ±0.01 of baseline
   - IGNORE F1, precision, and recall in this phase

4. Focus Validation:
   - Ensure changes only affect target disease group
   - If multiple groups change unintentionally → mark iteration invalid

================================================================================
ALLOWED DECISIONS
================================================================================

You may output exactly ONE of these decisions:
- CONTINUE_TARGETED_AUC_IMPROVEMENT: Continue with adjusted auc_improvement settings
- ADJUST_TARGET_CLASS_LIST: Add/remove diseases from target list
- STOP_PHASE_5_AND_FREEZE: Stop Phase 5 (success or regression detected)

================================================================================
STOP CONDITIONS
================================================================================

STOP Phase 5 when:
1. Two or more target diseases show consistent AUC improvement >= +0.01 (SUCCESS)
2. Any stable disease regresses by more than -0.01 AUC (FAILURE)
3. Macro AUC stagnates for 3 consecutive iterations (STAGNATION)

Upon stopping:
- Freeze model weights
- Transition to calibration phase (outside Phase 5 scope)

================================================================================
IMPROVEMENT TASKS (ONE PER ITERATION)
================================================================================

1. Hard-Negative Emphasis (Primary):
   - Increase penalty for high-scoring negatives in target classes
   - Adjust hard_negative_threshold and hard_negative_weight

2. Target Class Refinement:
   - Add or remove diseases from target list based on AUC response

3. Threshold Sensitivity Audit (Read-Only):
   - Verify ranking improvement is not caused by hidden threshold effects

================================================================================
OUTPUT FORMAT (RESPOND ONLY IN VALID JSON)
================================================================================

{{
  "phase": "PHASE_5",
  "decision": "CONTINUE_TARGETED_AUC_IMPROVEMENT" | "ADJUST_TARGET_CLASS_LIST" | "STOP_PHASE_5_AND_FREEZE",
  "reasoning": "...",
  "target_diseases": ["Pneumonia", "Fibrosis", "Edema"],
  "auc_improvement_config": {{
    "hard_negative_threshold": 0.3,
    "hard_negative_weight": 1.5
  }},
  "per_class_auc_delta": {{"Pneumonia": +0.015, "Fibrosis": -0.002}},
  "stable_diseases_check": "PASSED" | "FAILED",
  "success": true | false
}}

================================================================================
GUIDING PRINCIPLE (MUST BE PRESERVED)
================================================================================

Phase 5 improves RANKING, not DECISIONS.
F1 optimization is explicitly OUT OF SCOPE.

Any iteration that violates this principle must be discarded.
"""

    def _get_legacy_system_prompt(self) -> str:
        """
        Legacy system prompt (kept for backwards compatibility).
        Only used when PHASED_PROTOCOL_ENABLED is False.
        """
        next_strategy = self.get_next_strategy()

        # Build reference iteration info
        if ROLE_BASED_ADVISOR:
            reference_info = f"""
================================================================================
REFERENCE ITERATIONS (ANCHORS)
================================================================================
ITERATION 12 - AUC ANCHOR:
  Macro AUC: {ITERATION_12_AUC_ANCHOR.macro_auc:.4f}
  Macro F1:  {ITERATION_12_AUC_ANCHOR.macro_f1:.4f}
  Role: Best ranking quality (use for TRAIN_AUC parent)

ITERATION 58 - F1 ANCHOR:
  Macro AUC: {ITERATION_58_F1_ANCHOR.macro_auc:.4f}
  Macro F1:  {ITERATION_58_F1_ANCHOR.macro_f1:.4f}
  Role: Best decision quality (use for RECOVER_F1 parent)

DECISION RULES:
  AUC tolerance: {DECISION_RULES.auc_tolerance} (how much AUC drop is acceptable)
  F1 target: {ITERATION_58_F1_ANCHOR.macro_f1 * DECISION_RULES.f1_recovery_threshold:.4f} (95% of iter58)
================================================================================
"""
        else:
            reference_info = ""

        # Prohibited actions (hard-coded rules)
        prohibited_actions = """
================================================================================
PROHIBITED ACTIONS (AI CANNOT OVERRIDE)
================================================================================
1. gamma > 2.5 - PROHIBITED (will be forced to 2.0)
2. use_class_weights=true for TRAIN_AUC role - PROHIBITED (will be forced to false)
3. parent_iteration selection - ENFORCED BY ROLE, not AI suggestion
4. More than 1 major config change per iteration - ONLY FIRST KEPT

ITERATION ROLES (determine parent and config):
- TRAIN_AUC: Focus on AUC, parent=iter12, early stopping on val_macro_auc
- RECOVER_F1: Focus on F1, parent=iter58, early stopping on val_f1
- ADJUST_THRESHOLDS: No training, threshold optimization only
================================================================================
"""

        strategy_guide = {
            "threshold_optimization": """
**FOCUS FOR THIS ITERATION**: Fix prediction thresholds
- Current threshold (0.5) is wrong for imbalanced data
- Suggest changing evaluation.threshold_optimization to 'per_class_f1_score'
- This should dramatically improve F1/Recall/Precision
- Don't change other major parameters yet
""",
            "custom": """
**FOCUS FOR THIS ITERATION**: Custom optimization
- Most systematic strategies have been tried
- You can now suggest combinations or novel approaches
- Still focus on ONE major change per iteration
"""
        }

        strategy_instruction = strategy_guide.get(next_strategy, strategy_guide['custom'])

        # Build role-specific context
        role_context = self._get_current_role_context()

        return f"""You are an AI advisor for improving a chest X-ray multi-label classification model.

{reference_info}
{prohibited_actions}

{role_context}

{strategy_instruction}

OUTPUT FORMAT:
Provide your analysis and suggest changes in JSON format.
"""

    def _create_analysis_prompt(
        self,
        config: Dict[str, Any],
        results_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        iteration: int,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        task_info: Optional[Dict[str, Any]] = None,
        iteration_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create the analysis prompt for the AI advisor

        This prompt includes:
        - Current config
        - Last 3 iteration summaries with full metrics
        - Disease-level breakdowns (F1, AUC, TP, FP, FN, Precision, Recall)
        - Task registry information
        - Threshold data
        """
        prompt_parts = []

        # Header
        prompt_parts.append(f"=== ITERATION {iteration} ANALYSIS REQUEST ===\n")

        # Task Registry Information (if available)
        if task_info:
            current_task = task_info.get('current_task')
            if current_task:
                prompt_parts.append("📋 CURRENT ACTIVE TASK:")
                prompt_parts.append(f"   Task ID: {current_task['task_id']}")
                prompt_parts.append(f"   Description: {current_task['description']}")
                prompt_parts.append(f"   Target Metric: {current_task['target_metric']}")
                prompt_parts.append(f"   Started: Iteration {current_task['start_iteration']}")
                prompt_parts.append(f"   Progress: {len(current_task['evaluation_history'])}/{current_task['required_iterations']} iterations")

                if current_task['evaluation_history']:
                    prompt_parts.append(f"\n   Evaluation History:")
                    for hist in current_task['evaluation_history']:
                        prompt_parts.append(f"      Iteration {hist['iteration']}: {current_task['target_metric']} = {hist['metric_value']:.4f}")

                prompt_parts.append("")
            else:
                prompt_parts.append("📋 NO ACTIVE TASK - Please suggest a new focused task\n")

            # Completed tasks summary
            completed_tasks = task_info.get('completed_tasks', [])
            if completed_tasks:
                prompt_parts.append(f"✅ COMPLETED TASKS ({len(completed_tasks)} total):")
                for task in completed_tasks[-3:]:  # Show last 3
                    status_emoji = "✅" if task['status'] == 'succeeded' else "❌"
                    prompt_parts.append(f"   {status_emoji} {task['task_id']}: {task['status']} - {task['description'][:60]}")
                prompt_parts.append("")

        # Training/Validation Loss Information
        if train_loss is not None and val_loss is not None:
            prompt_parts.append("TRAINING/VALIDATION LOSS:")
            prompt_parts.append(f"  Final Training Loss: {train_loss:.4f}")
            prompt_parts.append(f"  Final Validation Loss: {val_loss:.4f}")

            # Analyze overfitting/underfitting
            loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0
            if loss_ratio > 1.2:
                prompt_parts.append(f"  ⚠️  OVERFITTING DETECTED: Val loss is {((loss_ratio - 1) * 100):.1f}% higher than train loss")
                prompt_parts.append("      Consider: more dropout, weight decay, or data augmentation")
            elif loss_ratio > 1.1:
                prompt_parts.append(f"  ⚠️  Possible overfitting: Val loss is {((loss_ratio - 1) * 100):.1f}% higher than train loss")
            elif train_loss > val_loss * 1.1:
                prompt_parts.append(f"  ⚠️  UNDERFITTING: Train loss is {((train_loss / val_loss - 1) * 100):.1f}% higher than val loss")
                prompt_parts.append("      Consider: more epochs, lower dropout, higher learning rate, or more capacity")
            else:
                prompt_parts.append("  ✓ Balanced training/validation losses")
            prompt_parts.append("")

        # Previous iterations history (if available)
        if iteration_history and len(iteration_history) > 0:
            prompt_parts.append("PREVIOUS ITERATIONS SUMMARY:")
            prompt_parts.append(f"Total previous iterations: {len(iteration_history)}\n")

            # Show summary table of all iterations
            prompt_parts.append("| Iter | Avg AUC | Avg F1 | Avg Recall | Train Loss | Val Loss | Key Config Changes |")
            prompt_parts.append("|------|---------|--------|------------|------------|----------|-------------------|")

            for prev_iter in iteration_history:
                iter_num = prev_iter.get('iteration', '?')
                avg_auc = prev_iter.get('avg_auc', 0.0)
                avg_f1 = prev_iter.get('avg_f1', 0.0)
                avg_recall = prev_iter.get('avg_recall', 0.0)
                train_loss_prev = prev_iter.get('train_loss', 'N/A')
                val_loss_prev = prev_iter.get('val_loss', 'N/A')

                # Format loss values
                train_loss_str = f"{train_loss_prev:.4f}" if isinstance(train_loss_prev, (int, float)) else train_loss_prev
                val_loss_str = f"{val_loss_prev:.4f}" if isinstance(val_loss_prev, (int, float)) else val_loss_prev

                # Extract key config info if available
                config_summary = "N/A"
                if 'suggested_changes' in prev_iter and prev_iter['suggested_changes']:
                    reasoning = prev_iter['suggested_changes'].get('reasoning', '')
                    if reasoning:
                        config_summary = reasoning[:40] + "..." if len(reasoning) > 40 else reasoning

                prompt_parts.append(f"| {iter_num} | {avg_auc:.4f} | {avg_f1:.4f} | {avg_recall:.4f} | {train_loss_str} | {val_loss_str} | {config_summary} |")

            prompt_parts.append("")

            # Detailed info for last 2 iterations (most recent context)
            recent_iters = iteration_history[-2:] if len(iteration_history) >= 2 else iteration_history
            prompt_parts.append(f"DETAILED INFO FOR LAST {len(recent_iters)} ITERATION(S):")

            for prev_iter in recent_iters:
                iter_num = prev_iter.get('iteration', '?')
                avg_auc = prev_iter.get('avg_auc', 0.0)
                avg_f1 = prev_iter.get('avg_f1', 0.0)
                avg_recall = prev_iter.get('avg_recall', 0.0)
                avg_precision = prev_iter.get('avg_precision', 0.0)
                train_loss_prev = prev_iter.get('train_loss')
                val_loss_prev = prev_iter.get('val_loss')

                prompt_parts.append(f"\nIteration {iter_num}:")
                prompt_parts.append(f"  Metrics: AUC={avg_auc:.4f}, F1={avg_f1:.4f}, Recall={avg_recall:.4f}, Precision={avg_precision:.4f}")

                if train_loss_prev is not None and val_loss_prev is not None:
                    loss_ratio = val_loss_prev / train_loss_prev if train_loss_prev > 0 else 1.0
                    overfitting_indicator = " (OVERFITTING)" if loss_ratio > 1.2 else " (balanced)" if loss_ratio < 1.1 else ""
                    prompt_parts.append(f"  Losses: Train={train_loss_prev:.4f}, Val={val_loss_prev:.4f}{overfitting_indicator}")

                if 'suggested_changes' in prev_iter and prev_iter['suggested_changes']:
                    prompt_parts.append(f"  Changes applied: {json.dumps(prev_iter['suggested_changes'], indent=2)}")

            prompt_parts.append("\n" + "="*80)
            prompt_parts.append("CRITICAL: Based on the above history, do NOT repeat the same configurations.")
            prompt_parts.append("If the latest iteration performed WORSE than previous ones, analyze why and suggest corrections.")
            prompt_parts.append("="*80 + "\n")

        # Current configuration
        prompt_parts.append("CURRENT CONFIGURATION:")
        prompt_parts.append("```yaml")
        prompt_parts.append(yaml.dump(config, default_flow_style=False, sort_keys=False))
        prompt_parts.append("```\n")

        # Results summary
        prompt_parts.append("TEST RESULTS:")
        results_summary = self._summarize_results(results_df)
        prompt_parts.append(results_summary)

        # Disease-level detailed breakdown
        prompt_parts.append("\nPER-DISEASE DETAILED METRICS:")
        prompt_parts.append(self._format_disease_metrics(results_df))

        # Threshold information (if available from latest iteration)
        if iteration_history and len(iteration_history) > 0:
            latest_iter = iteration_history[-1]
            if 'thresholds' in latest_iter and latest_iter['thresholds']:
                prompt_parts.append("\nOPTIMIZED THRESHOLDS (from validation set):")
                thresholds_dict = latest_iter['thresholds']
                prompt_parts.append("| Disease              | Threshold | F1 Score | Positive Samples | Status     |")
                prompt_parts.append("|----------------------|-----------|----------|------------------|------------|")
                for disease, thresh_data in thresholds_dict.items():
                    threshold = thresh_data.get('threshold', 0.5)
                    score = thresh_data.get('score', 0.0)
                    pos_samples = thresh_data.get('positive_samples', 0)
                    status = thresh_data.get('status', 'unknown')
                    prompt_parts.append(f"| {disease:<20} | {threshold:>9.3f} | {score:>8.4f} | {pos_samples:>16} | {status:<10} |")
                prompt_parts.append("")

        # Baseline comparison
        prompt_parts.append("\nBASELINE COMPARISON:")
        comparison_summary = self._summarize_comparison(comparison_df)
        prompt_parts.append(comparison_summary)

        # Iteration Analysis (if available)
        if iteration_analysis:
            prompt_parts.append("\n📊 ITERATION TREND ANALYSIS:")

            # F1 comparison
            if 'f1_comparison' in iteration_analysis:
                f1_comp = iteration_analysis['f1_comparison']
                prompt_parts.append(f"\nF1-Score Trend:")
                prompt_parts.append(f"   Best: {f1_comp['best_value']:.4f} (iteration {f1_comp['best_iteration']})")
                prompt_parts.append(f"   Trend: {f1_comp['trend']} ({f1_comp['improvement']:+.4f})")
                prompt_parts.append(f"   Recent values: {[f'{v:.4f}' for v in f1_comp['values']]}")

            # AUC comparison
            if 'auc_comparison' in iteration_analysis:
                auc_comp = iteration_analysis['auc_comparison']
                prompt_parts.append(f"\nAUC Trend:")
                prompt_parts.append(f"   Best: {auc_comp['best_value']:.4f} (iteration {auc_comp['best_iteration']})")
                prompt_parts.append(f"   Trend: {auc_comp['trend']} ({auc_comp['improvement']:+.4f})")
                prompt_parts.append(f"   Recent values: {[f'{v:.4f}' for v in auc_comp['values']]}")

            # Recall comparison
            if 'recall_comparison' in iteration_analysis:
                recall_comp = iteration_analysis['recall_comparison']
                prompt_parts.append(f"\nRecall Trend:")
                prompt_parts.append(f"   Best: {recall_comp['best_value']:.4f} (iteration {recall_comp['best_iteration']})")
                prompt_parts.append(f"   Trend: {recall_comp['trend']} ({recall_comp['improvement']:+.4f})")
                prompt_parts.append(f"   Recent values: {[f'{v:.4f}' for v in recall_comp['values']]}")

            prompt_parts.append("")

        # Specific questions
        prompt_parts.append("\nPLEASE ADDRESS:")
        if task_info and task_info.get('current_task'):
            prompt_parts.append("1. Is the current task showing improvement? Should we continue or change strategy?")
            prompt_parts.append("2. Based on the task evaluation history, what's working and what's not?")
            prompt_parts.append("3. If the task should end, what's the next highest-impact task to focus on?")
        else:
            prompt_parts.append("1. What should be the NEXT focused task to work on?")
            prompt_parts.append("2. Which specific metric should this task target (f1, auc, recall)?")
            prompt_parts.append("3. What config changes are needed for this task?")

        prompt_parts.append("4. Are there any critical issues with current configuration?")
        prompt_parts.append("5. What specific hyperparameter changes would help?")
        if iteration_history and len(iteration_history) > 0:
            prompt_parts.append("6. How does this iteration compare to previous ones? What trends do you see?")
            prompt_parts.append("7. What new approaches should we try that haven't been tested yet?")

        return "\n".join(prompt_parts)

    def _format_disease_metrics(self, results_df: pd.DataFrame) -> str:
        """
        Format per-disease metrics in a detailed table

        Returns:
            Formatted string with disease-level breakdown
        """
        lines = []
        lines.append("| Disease              | F1 Score | AUC    | Precision | Recall | Accuracy | TP    | FP    | FN    | TN    |")
        lines.append("|----------------------|----------|--------|-----------|--------|----------|-------|-------|-------|-------|")

        for _, row in results_df.iterrows():
            disease = row.get('Label', 'Unknown')
            f1 = row.get('F1_Score', 0.0)
            auc = row.get('AUC', 0.0)
            precision = row.get('Precision', 0.0)
            recall = row.get('Recall', 0.0)
            accuracy = row.get('Accuracy', 0.0)

            # Try to get confusion matrix values if available
            tp = row.get('TP', 0)
            fp = row.get('FP', 0)
            fn = row.get('FN', 0)
            tn = row.get('TN', 0)

            lines.append(
                f"| {disease:<20} | {f1:>8.4f} | {auc:>6.4f} | "
                f"{precision:>9.4f} | {recall:>6.4f} | {accuracy:>8.4f} | "
                f"{tp:>5} | {fp:>5} | {fn:>5} | {tn:>5} |"
            )

        return "\n".join(lines)

    def _summarize_results(self, results_df: pd.DataFrame) -> str:
        """Create a summary of test results"""
        summary = []

        # Overall statistics
        avg_auc = results_df['AUC'].mean()
        avg_recall = results_df['Recall'].mean()
        avg_precision = results_df['Precision'].mean()
        avg_f1 = results_df['F1_Score'].mean()

        summary.append(f"Average Metrics:")
        summary.append(f"  AUC: {avg_auc:.4f}")
        summary.append(f"  Recall: {avg_recall:.4f}")
        summary.append(f"  Precision: {avg_precision:.4f}")
        summary.append(f"  F1-Score: {avg_f1:.4f}")
        summary.append("")

        # Per-class results (top 3 best and worst by AUC)
        summary.append("Best Performing Classes (by AUC):")
        top_3 = results_df.nlargest(3, 'AUC')[['Label', 'AUC', 'Recall', 'Precision']]
        for _, row in top_3.iterrows():
            summary.append(f"  {row['Label']}: AUC={row['AUC']:.3f}, Recall={row['Recall']:.3f}, Precision={row['Precision']:.3f}")

        summary.append("\nWorst Performing Classes (by AUC):")
        bottom_3 = results_df.nsmallest(3, 'AUC')[['Label', 'AUC', 'Recall', 'Precision']]
        for _, row in bottom_3.iterrows():
            summary.append(f"  {row['Label']}: AUC={row['AUC']:.3f}, Recall={row['Recall']:.3f}, Precision={row['Precision']:.3f}")

        # Problem detection
        zero_recall_count = (results_df['Recall'] == 0).sum()
        zero_precision_count = (results_df['Precision'] == 0).sum()

        if zero_recall_count > 0 or zero_precision_count > 0:
            summary.append("\n⚠️ CRITICAL ISSUES:")
            summary.append(f"  Classes with zero recall: {zero_recall_count}/14")
            summary.append(f"  Classes with zero precision: {zero_precision_count}/14")
            summary.append("  This indicates the model is predicting almost all samples as negative!")

        return "\n".join(summary)

    def _summarize_comparison(self, comparison_df: pd.DataFrame) -> str:
        """Create a summary of baseline comparison"""
        summary = []

        # Count improvements for all metrics
        metrics = ['AUC', 'F1_Score', 'Recall', 'Accuracy', 'Specificity', 'Precision', 'Sensitivity']

        summary.append(f"Comparison with Wang et al. Baseline:")
        summary.append(f"Classes Better/Worse/Equal:")

        for metric in metrics:
            better_col = f'Better_{metric}'
            if better_col in comparison_df.columns:
                better = (comparison_df[better_col] == 'Yes').sum()
                worse = (comparison_df[better_col] == 'No').sum()
                equal = (comparison_df[better_col] == 'Equal').sum()
                summary.append(f"  {metric:12s}: {better:2d}/{worse:2d}/{equal:2d} (better/worse/equal)")

        # Average improvements for all metrics
        summary.append(f"\nAverage Improvements vs Baseline:")
        for metric in metrics:
            improvement_col = f'{metric}_Improvement'
            if improvement_col in comparison_df.columns:
                avg_improvement = comparison_df[improvement_col].mean()
                summary.append(f"  {metric:12s}: {avg_improvement:+.4f}")

        # Threshold improvements (note: lower threshold change isn't necessarily better/worse)
        if 'Threshold_Improvement' in comparison_df.columns:
            avg_threshold_change = comparison_df['Threshold_Improvement'].mean()
            summary.append(f"  {'Threshold':12s}: {avg_threshold_change:+.4f} (avg change)")

        # Biggest wins and losses (based on F1-Score as it's most critical)
        if len(comparison_df) > 0 and 'F1_Score_Improvement' in comparison_df.columns:
            best_improvement = comparison_df.nlargest(1, 'F1_Score_Improvement').iloc[0]
            worst_improvement = comparison_df.nsmallest(1, 'F1_Score_Improvement').iloc[0]

            summary.append(f"\nBiggest F1-Score Win:  {best_improvement['Label']} ({best_improvement['F1_Score_Improvement']:+.4f})")
            summary.append(f"Biggest F1-Score Loss: {worst_improvement['Label']} ({worst_improvement['F1_Score_Improvement']:+.4f})")

        return "\n".join(summary)

    def _parse_suggestions(self, response_text: str) -> Dict[str, Any]:
        """
        Parse suggested changes from AI response

        Args:
            response_text: Raw response from AI

        Returns:
            Dictionary of suggested changes
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON code block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                suggestions = json.loads(json_str)
            elif "{" in response_text and "}" in response_text:
                # Try to find JSON object
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                suggestions = json.loads(json_str)
            else:
                # No JSON found, return empty suggestions
                suggestions = {"reasoning": "Could not parse suggestions from response"}

            return suggestions

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse AI suggestions: {e}")
            return {"reasoning": "JSON parsing failed", "error": str(e)}

    def validate_and_correct_suggestion(
        self,
        suggestion: Dict[str, Any],
        role: 'IterationRole'
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate AI suggestion against hard-coded rules and correct violations.

        This method enforces the rules that AI cannot override:
        1. gamma > 2.5 → force to 2.0
        2. use_class_weights=true for TRAIN_AUC → force to false
        3. parent_iteration → remove (enforced by role)
        4. Multiple major changes → keep only first

        Args:
            suggestion: AI-provided configuration suggestions
            role: The current iteration role

        Returns:
            Tuple of (corrected_suggestion, list_of_corrections)
        """
        if not ROLE_BASED_ADVISOR:
            return suggestion, []

        # Use the validation function from iteration_baselines
        corrected, corrections = validate_ai_suggestion(suggestion, role)

        # Log corrections
        if corrections:
            print(f"🔧 AI suggestion corrections applied:")
            for correction in corrections:
                print(f"   - {correction}")

        return corrected, corrections

    def _get_current_role_context(self) -> str:
        """
        Build role-specific context string for the current iteration.

        This is called from _get_system_prompt() to include role information.
        Uses self._current_role and self._current_parent set by analyze_results().

        Returns:
            Role context string for the system prompt
        """
        role = getattr(self, '_current_role', None)
        parent = getattr(self, '_current_parent', None)

        if not ROLE_BASED_ADVISOR or role is None:
            return """
================================================================================
CURRENT ITERATION ROLE: UNKNOWN (role-based system not active)
================================================================================
"""

        if role == IterationRole.TRAIN_AUC:
            return f"""
================================================================================
🎯 CURRENT ITERATION ROLE: TRAIN_AUC
================================================================================
Parent iteration: {parent} (AUC anchor)
Primary objective: RECOVER/MAINTAIN AUC (target ≥0.80)
Early stopping: val_macro_auc (THIS IS CORRECT - DO NOT SUGGEST CHANGING IT)

What this means:
- The system detected AUC degradation and is recovering from iteration 12
- Focus your analysis on AUC improvement
- val_macro_auc monitoring is INTENTIONAL, not a bug
- Suggest changes that help AUC without drastically harming F1
================================================================================
"""
        elif role == IterationRole.RECOVER_F1:
            return f"""
================================================================================
🎯 CURRENT ITERATION ROLE: RECOVER_F1
================================================================================
Parent iteration: {parent} (F1 anchor)
Primary objective: IMPROVE F1 (target ≥0.26) while preserving AUC
Early stopping: val_f1 (THIS IS CORRECT)

What this means:
- AUC is already good, but F1 needs improvement
- The system is working from iteration 58's model
- Focus your analysis on F1/precision/recall improvement
- Suggest changes that help F1 without dropping AUC below 0.79
================================================================================
"""
        elif role == IterationRole.ADJUST_THRESHOLDS:
            return """
================================================================================
🎯 CURRENT ITERATION ROLE: ADJUST_THRESHOLDS
================================================================================
Primary objective: Optimize thresholds only (no training)

What this means:
- Model weights are frozen
- Only threshold optimization will be performed
- Suggest threshold optimization strategies, not training changes
================================================================================
"""
        else:
            return f"""
================================================================================
🎯 CURRENT ITERATION ROLE: {role.value if role else 'UNKNOWN'}
================================================================================
"""

    def get_role_specific_prompt_addition(self, role: 'IterationRole') -> str:
        """
        Get additional prompt text specific to the current role.

        Args:
            role: The current iteration role

        Returns:
            Additional prompt text for the role
        """
        if not ROLE_BASED_ADVISOR:
            return ""

        if role == IterationRole.TRAIN_AUC:
            return f"""
================================================================================
CURRENT ROLE: TRAIN_AUC
================================================================================
Your suggestions MUST focus on improving AUC while maintaining model quality.

ENFORCED CONFIGURATION (you cannot change these):
- loss.type: FocalLoss
- loss.gamma: 2.0 (NEVER suggest > 2.5)
- loss.use_class_weights: false
- early_stopping.monitor: val_macro_auc
- early_stopping.mode: max
- Parent iteration: 12 (AUC anchor)

ALLOWED CHANGES (suggest ONE only):
- learning_rate adjustments
- num_epochs adjustments
- dropout_rate adjustments
- augmentation changes
================================================================================
"""
        elif role == IterationRole.RECOVER_F1:
            return f"""
================================================================================
CURRENT ROLE: RECOVER_F1
================================================================================
Your suggestions MUST focus on improving F1 without harming AUC significantly.

ENFORCED CONFIGURATION (you cannot change these):
- learning_rate: max {RECOVER_F1_RULES.max_learning_rate}
- early_stopping.monitor: val_f1
- early_stopping.mode: max
- Parent iteration: 58 (F1 anchor)

ALLOWED CHANGES (suggest ONE only):
- threshold optimization parameters
- augmentation changes
- small dropout adjustments
================================================================================
"""
        elif role == IterationRole.ADJUST_THRESHOLDS:
            return """
================================================================================
CURRENT ROLE: ADJUST_THRESHOLDS
================================================================================
No training will occur. Only threshold optimization.

Your suggestions should focus on:
- Threshold optimization strategy (per_class_f1_score, youden_j, etc.)
- Nothing else - training config will be ignored
================================================================================
"""
        else:
            return ""

    def _create_warm_start_prompt(
        self,
        current_iteration: int,
        previous_iteration_data: Dict[str, Any],
        best_iterations_data: Dict[str, Any],
        proposed_config: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> str:
        """Create prompt for warm start recommendation"""

        # Format previous iteration summary
        prev_summary = f"""
Iteration {previous_iteration_data.get('iteration', 'N/A')}:
- F1: {previous_iteration_data.get('avg_f1', 0.0):.4f}
- AUC: {previous_iteration_data.get('avg_auc', 0.0):.4f}
- Recall: {previous_iteration_data.get('avg_recall', 0.0):.4f}
- Precision: {previous_iteration_data.get('avg_precision', 0.0):.4f}
- Epochs trained: {previous_iteration_data.get('actual_epochs', 'N/A')}
"""

        # Format best iterations summary
        best_summary_lines = []
        best_iters = best_iterations_data.get('best_iterations', {})
        current_iter = current_iteration

        for metric_name, data in best_iters.items():
            if data:
                iters_ago = current_iter - data.get('iteration', current_iter)
                best_summary_lines.append(
                    f"- {metric_name}: Iteration {data.get('iteration', 'N/A')} "
                    f"(value={data.get('value', 0.0):.4f}, {iters_ago} iterations ago)"
                )

        best_summary = "\n".join(best_summary_lines) if best_summary_lines else "No best iterations tracked yet"

        # Format task-specific best if available
        task_best_summary = ""
        task_specific = best_iterations_data.get('task_specific_best', {})
        if task_specific:
            task_best_summary = "\n\nTASK-SPECIFIC BEST ITERATION:"
            for key, data in task_specific.items():
                if data:
                    task_best_summary += f"\n- {key}: Iteration {data.get('iteration', 'N/A')} "
                    task_best_summary += f"(value={data.get('value', 0.0):.4f})"
                    if 'task_info' in data:
                        task_info = data['task_info']
                        task_best_summary += f"\n  Target: {task_info.get('target_metric')} → {task_info.get('target_value')}"
                        task_best_summary += f"\n  Constraints: {list(task_info.get('constraints', {}).keys())}"

        # Format config differences
        changes = []

        # Compare loss params
        prev_loss = previous_config.get('loss', {})
        prop_loss = proposed_config.get('loss', {})
        if prev_loss.get('gamma') != prop_loss.get('gamma'):
            changes.append(f"- Loss gamma: {prev_loss.get('gamma')} → {prop_loss.get('gamma')}")
        if prev_loss.get('type') != prop_loss.get('type'):
            changes.append(f"- Loss type: {prev_loss.get('type')} → {prop_loss.get('type')}")

        # Compare training params
        prev_train = previous_config.get('training', {})
        prop_train = proposed_config.get('training', {})
        if prev_train.get('learning_rate') != prop_train.get('learning_rate'):
            changes.append(f"- Learning rate: {prev_train.get('learning_rate')} → {prop_train.get('learning_rate')}")
        if prev_train.get('num_epochs') != prop_train.get('num_epochs'):
            changes.append(f"- Num epochs: {prev_train.get('num_epochs')} → {prop_train.get('num_epochs')}")

        # Compare model params
        prev_model = previous_config.get('model', {})
        prop_model = proposed_config.get('model', {})
        if prev_model.get('dropout_rate') != prop_model.get('dropout_rate'):
            changes.append(f"- Dropout: {prev_model.get('dropout_rate')} → {prop_model.get('dropout_rate')}")

        config_diff = "\n".join(changes) if changes else "No major configuration changes"

        # Get current task info if available
        task_context = ""
        current_task = best_iterations_data.get('current_task')
        if current_task:
            task_context = f"""

CURRENT TASK:
Target Metric: {current_task.get('target_metric', 'N/A')}
Target Value: {current_task.get('target_value', 'N/A')}
Constraints: {current_task.get('constraints', {})}
"""

        prompt = f"""
You are advising on model weight initialization for iteration {current_iteration}.

PREVIOUS ITERATION:
{prev_summary}

BEST ITERATIONS (Historical Bests):
{best_summary}
{task_best_summary}
{task_context}

PROPOSED CONFIGURATION CHANGES:
{config_diff}

DECISION REQUIRED:
Choose the best starting point for model weights:
1. "iteration_XX" - Load weights from a previous iteration (specify which)
2. "cold_start" - Start from ImageNet pretrained weights

CRITICAL GUIDELINES - MUST COMPARE METRICS:
⚠️  DO NOT use recent iterations just because they're "most recent"!
⚠️  ALWAYS compare metrics: Is previous iteration BETTER than historical bests?
⚠️  If previous iteration has WORSE metrics → use best iteration instead!

Decision Rules (in priority order):
1. **Compare Quality First**:
   - If previous F1 < best F1 by >0.05 → DON'T use previous, use best_f1 iteration
   - If previous AUC < best AUC by >0.05 → DON'T use previous, use best_auc iteration
   - "Most recent" is NOT a good reason if metrics are worse!

2. **Task Constraints**:
   - If task has constraints → ONLY use iterations that satisfy constraints
   - Task-specific best > Global best when constraints exist

3. **Configuration Changes**:
   - Small hyperparameter tweaks (gamma ±1, learning rate ±0.0002) → warm start OK
   - Major changes (loss type, architecture) → cold start safer

4. **Plateau Detection**:
   - If stuck (>10 iterations without improvement) → reset to best

5. **Quality Thresholds** (minimum acceptable):
   - F1 should be ≥ 0.20 (if lower, iteration likely failed)
   - AUC should be ≥ 0.70 (if lower, iteration likely failed)
   - If previous iteration below thresholds → use best instead!

RESPOND WITH VALID JSON ONLY (no additional text):
{{
  "warm_start_from": "iteration_59" or "cold_start",
  "reasoning": "MUST explain metric comparison: Previous vs Best. Example: 'Previous F1=0.127 < Best F1=0.275, so using best_f1 iteration instead.'",
  "confidence": 0.85,
  "expected_benefit": "faster convergence" or "better performance" or "exploration"
}}
"""
        return prompt

    def _parse_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from AI response (same logic as _parse_suggestions)

        Args:
            response_text: Raw response text from AI

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Look for JSON code block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            elif "{" in response_text and "}" in response_text:
                # Try to find JSON object
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                return None
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            return None

    def _log_warm_start_decision(
        self,
        current_iteration: int,
        previous_iteration_data: Dict[str, Any],
        best_iterations_data: Dict[str, Any],
        recommendation: Dict[str, Any]
    ):
        """
        Log warm start decision to file for tracking and analysis

        Args:
            current_iteration: Current iteration number
            previous_iteration_data: Previous iteration metrics
            best_iterations_data: Best iterations data
            recommendation: AI recommendation
        """
        import datetime

        log_file = "experiments/auto_improvement_runs/warm_start_decisions.log"

        # Create log directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Format log entry
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"""
{'='*80}
WARM START DECISION - Iteration {current_iteration}
{'='*80}
Timestamp: {timestamp}
Model Used: {self.last_model_used or 'Unknown'}

PREVIOUS ITERATION ({previous_iteration_data.get('iteration', 'N/A')}):
  F1:        {previous_iteration_data.get('avg_f1', 0.0):.4f}
  AUC:       {previous_iteration_data.get('avg_auc', 0.0):.4f}
  Recall:    {previous_iteration_data.get('avg_recall', 0.0):.4f}
  Precision: {previous_iteration_data.get('avg_precision', 0.0):.4f}

BEST ITERATIONS:
"""

        # Add best iterations info
        best_iters = best_iterations_data.get('best_iterations', {})
        for metric_name, data in best_iters.items():
            if data:
                log_entry += f"  {metric_name}: Iteration {data.get('iteration', 'N/A')} (value={data.get('value', 0.0):.4f})\n"

        # Add task-specific best if available
        task_specific = best_iterations_data.get('task_specific', {})
        if task_specific:
            log_entry += "\nTASK-SPECIFIC BEST:\n"
            for key, data in task_specific.items():
                if data:
                    log_entry += f"  {key}: Iteration {data.get('iteration', 'N/A')} (value={data.get('value', 0.0):.4f})\n"

        # Add AI decision
        log_entry += f"""
AI DECISION:
  Choice:           {recommendation.get('warm_start_from', 'N/A')}
  Reasoning:        {recommendation.get('reasoning', 'N/A')}
  Confidence:       {recommendation.get('confidence', 0.0):.0%}
  Expected Benefit: {recommendation.get('expected_benefit', 'N/A')}

"""

        # Append to log file
        with open(log_file, 'a') as f:
            f.write(log_entry)

        print(f"📝 Warm start decision logged to: {log_file}")

    def save_analysis(self, analysis: str, iteration: int, output_dir: str = "."):
        """
        Save analysis to file

        Args:
            analysis: Analysis text
            iteration: Iteration number
            output_dir: Directory to save analysis
        """
        output_path = os.path.join(output_dir, f"ai_analysis_iteration_{iteration:03d}.txt")
        with open(output_path, 'w') as f:
            f.write(analysis)

        return output_path

    # ========================================================================
    # FEATURE-SPACE SMOTE ADVISORY METHODS
    # ========================================================================

    def analyze_smote_results(
        self,
        current_config: Dict[str, Any],
        hernia_metrics: Dict[str, Any],
        macro_auc: float,
        macro_f1: float,
        baseline_macro_auc: float,
        baseline_macro_f1: float,
        iteration: int,
        smote_probe_count: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze SMOTE experiment results and decide next action.

        This method implements the SMOTE advisory decision logic:
        - Can ONLY change sampling_ratio (4 → 8)
        - Checks Hernia F1 and Macro metrics
        - Maximum 2 SMOTE probes

        Args:
            current_config: Current iteration config
            hernia_metrics: Dict with TP, FP, FN, TN, Precision, Recall, F1, AUC for Hernia
            macro_auc: Current macro AUC
            macro_f1: Current macro F1
            baseline_macro_auc: Baseline macro AUC (from anchor iteration)
            baseline_macro_f1: Baseline macro F1 (from anchor iteration)
            iteration: Current iteration number
            smote_probe_count: How many SMOTE probes have been run (1 or 2)

        Returns:
            Dict with:
                - decision: 'KEEP', 'STOP', or 'PROBE_AGAIN'
                - reasoning: Explanation
                - next_config: Config changes if PROBE_AGAIN (only sampling_ratio)
        """
        self.logger_smote = logging.getLogger('smote_advisor')

        # Extract Hernia metrics
        hernia_f1 = hernia_metrics.get('F1', 0.0)
        hernia_recall = hernia_metrics.get('Recall', 0.0)
        hernia_precision = hernia_metrics.get('Precision', 0.0)
        hernia_auc = hernia_metrics.get('AUC', 0.0)
        hernia_tp = hernia_metrics.get('TP', 0)
        hernia_fn = hernia_metrics.get('FN', 0)

        # Current SMOTE config
        smote_config = current_config.get('smote', {})
        current_ratio = smote_config.get('sampling_ratio', 4)

        # Calculate deltas
        auc_delta = macro_auc - baseline_macro_auc
        f1_delta = macro_f1 - baseline_macro_f1

        # Build advisory log
        advisory_log = f"""
================================================================================
SMOTE ADVISORY ANALYSIS - Iteration {iteration}
================================================================================
SMOTE Configuration:
  Target Class: {smote_config.get('target_class', 'Hernia')}
  Sampling Ratio: {current_ratio}x
  K Neighbors: {smote_config.get('k_neighbors', 3)}
  Probe Count: {smote_probe_count}/2

Hernia-Specific Metrics:
  TP: {hernia_tp}, FP: {hernia_metrics.get('FP', 0)}, FN: {hernia_fn}, TN: {hernia_metrics.get('TN', 0)}
  Precision: {hernia_precision:.4f}
  Recall: {hernia_recall:.4f}
  F1: {hernia_f1:.4f}
  AUC: {hernia_auc:.4f}

Macro Metrics:
  Current Macro AUC: {macro_auc:.4f} (baseline: {baseline_macro_auc:.4f}, delta: {auc_delta:+.4f})
  Current Macro F1:  {macro_f1:.4f} (baseline: {baseline_macro_f1:.4f}, delta: {f1_delta:+.4f})

"""

        # Decision logic (as specified)
        decision = None
        reasoning = ""
        next_config = None

        # Rule 1: If Macro AUC decreases by >0.003 → REJECT and STOP
        if auc_delta < -0.003:
            decision = 'STOP'
            reasoning = f"SMOTE REJECTED: Macro AUC decreased by {abs(auc_delta):.4f} (threshold: 0.003). " \
                       f"Feature-space SMOTE is harming model ranking quality."

        # Rule 2: If Hernia F1 > 0 AND Macro F1 does not decrease by >0.01 → KEEP or STOP
        elif hernia_f1 > 0 and f1_delta >= -0.01:
            decision = 'KEEP'
            reasoning = f"SMOTE SUCCESSFUL: Hernia F1 improved to {hernia_f1:.4f} while Macro F1 " \
                       f"remained stable (delta: {f1_delta:+.4f}). SMOTE ratio {current_ratio}x is effective."

        # Rule 3: If Hernia F1 still == 0 AND Macro AUC is stable AND probes < 2 → propose ratio = 8
        elif hernia_f1 == 0 and auc_delta >= -0.003 and smote_probe_count < 2 and current_ratio < 8:
            decision = 'PROBE_AGAIN'
            reasoning = f"Hernia F1 still zero but Macro AUC is stable (delta: {auc_delta:+.4f}). " \
                       f"Proposing increased sampling_ratio from {current_ratio} to 8 for second probe."
            next_config = {
                'smote': {
                    'sampling_ratio': 8
                }
            }

        # Rule 4: Maximum probes reached or ratio already 8 → STOP
        elif smote_probe_count >= 2 or current_ratio >= 8:
            decision = 'STOP'
            reasoning = f"SMOTE experiment complete. Maximum probes ({smote_probe_count}/2) reached " \
                       f"or ratio already at maximum ({current_ratio}). " \
                       f"Final Hernia F1: {hernia_f1:.4f}, Macro AUC delta: {auc_delta:+.4f}."

        # Default: STOP
        else:
            decision = 'STOP'
            reasoning = f"SMOTE experiment inconclusive. Hernia F1: {hernia_f1:.4f}, " \
                       f"Macro AUC delta: {auc_delta:+.4f}. Stopping to preserve model quality."

        advisory_log += f"""
DECISION: {decision}
REASONING: {reasoning}
"""

        if next_config:
            advisory_log += f"""
NEXT CONFIG CHANGE (if PROBE_AGAIN):
  smote.sampling_ratio: {current_ratio} → 8
"""

        advisory_log += "=" * 80 + "\n"

        # Log to file
        self._log_smote_decision(iteration, advisory_log)

        return {
            'decision': decision,
            'reasoning': reasoning,
            'next_config': next_config,
            'hernia_metrics': hernia_metrics,
            'macro_auc_delta': auc_delta,
            'macro_f1_delta': f1_delta,
            'smote_probe_count': smote_probe_count
        }

    def _log_smote_decision(self, iteration: int, log_content: str):
        """Log SMOTE advisory decision to file."""
        log_dir = "experiments/auto_improvement_runs"
        log_file = os.path.join(log_dir, "smote_advisory_decisions.log")

        os.makedirs(log_dir, exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(log_file, 'a') as f:
            f.write(f"\n[{timestamp}] Iteration {iteration}\n")
            f.write(log_content)
            f.write("\n")

    def generate_smote_probe_config(
        self,
        base_config: Dict[str, Any],
        iteration: int,
        sampling_ratio: int = 4,
        target_class: str = 'Hernia',
        k_neighbors: int = 3,
        parent_iteration: int = 89,
        backbone_source: int = 12
    ) -> Dict[str, Any]:
        """
        Generate a config for a SMOTE probe iteration.

        Args:
            base_config: Base config to extend
            iteration: New iteration number
            sampling_ratio: SMOTE sampling ratio
            target_class: Target class for SMOTE
            k_neighbors: K neighbors for SMOTE
            parent_iteration: Parent iteration (anchor)
            backbone_source: Iteration to load backbone from

        Returns:
            New config dict
        """
        import copy
        new_config = copy.deepcopy(base_config)

        # Update metadata
        new_config['metadata'] = {
            'config_version': '1.0',
            'description': f'SMOTE_PROBE - Feature-space SMOTE for {target_class}, ratio={sampling_ratio}x',
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d'),
            'iteration': iteration,
            'parent_iteration': parent_iteration,
            'backbone_source': backbone_source,
            'phase': 'SMOTE_PROBE',
            'note': f'Feature-space SMOTE targeting {target_class}',
            'anchor_iteration': parent_iteration
        }

        # Ensure HEAD_UPGRADE settings
        new_config['training']['freeze_backbone'] = True
        new_config['loss'] = {'type': 'BCE', 'use_class_weights': False}
        new_config['auc_improvement'] = {'enabled': False}

        # Add SMOTE config
        new_config['smote'] = {
            'enabled': True,
            'mode': 'feature_space',
            'target_class': target_class,
            'sampling_ratio': sampling_ratio,
            'k_neighbors': k_neighbors,
            'seed': 42
        }

        return new_config


import logging


if __name__ == "__main__":
    # Example usage
    print("AI Advisor module loaded successfully")
    print("Make sure to set OPENAI_API_KEY environment variable before using")