Project Summary and Next Steps

Context

This project investigates automated improvement of a multi-label chest X-ray classification model through a controlled, iteration-based training loop. The goal is to improve decision-level performance (F1-score) while maintaining competitive ranking performance (AUROC) relative to a published baseline.

All experiments are executed through an automated pipeline consisting of:
	•	auto_improvement_loop.py
	•	chest_xray_pipeline.py
	•	ai_advisor.py

Each iteration is fully logged, reproducible, and compared against the baseline study using identical metrics.

⸻

What We Have Done So Far

1. Baseline Exploration (Iterations 1–84)
	•	Multiple strategies were explored, including:
	•	Class weighting
	•	Focal loss
	•	Hard-negative mining
	•	Targeted AUC optimization for rare diseases (Phase 5)
	•	These strategies occasionally improved AUROC for specific classes but:
	•	Introduced instability
	•	Often degraded macro F1
	•	Made results difficult to interpret

Key outcome: Phase 5 was fully explored and exhausted.

⸻

2. Identification of a Strong Representation Anchor (Iteration 12)
	•	Iteration 12 consistently produced the strongest and most stable AUROC across classes.
	•	Statistical validation confirmed that its performance was reproducible and not due to noise.

Conclusion: Iteration 12 serves as a reliable representation-level anchor.

⸻

3. Transition to Architecture-Driven Optimization (Iterations 85–89)

HEAD_UPGRADE Phase
	•	Backbone weights from Iteration 12 were frozen.
	•	The linear classification head was replaced with a configurable MLP head.
	•	Only head architecture parameters were varied (hidden size, activation, dropout).
	•	Training used:
	•	Neutral BCE loss
	•	No class weighting
	•	No hard-negative logic
	•	Early stopping based on validation macro AUROC

This phase cleanly isolates the effect of decision-head capacity on ranking performance.

⸻

4. Best Current Result (Iteration 89)

Iteration 89 represents the strongest overall model so far:
	•	Macro AUROC: ~0.819
	•	Macro F1: ~0.295 (higher than the baseline paper)
	•	Improved F1-score in the majority of pathologies
	•	Stable AUROC with reduced variance

Importantly:
	•	All metrics were computed using the same evaluation logic as the baseline study.
	•	Confusion matrices and calibrated thresholds were generated, enabling decision-level analysis not provided in the original paper.

Iteration 89 is now designated as a LOCKED ANCHOR MODEL.

⸻

What We Learned
	1.	Ranking vs. Decision Trade-off
	•	AUROC and F1 cannot be optimized simultaneously during training.
	•	Separating representation learning (AUROC) from calibration (F1) is essential.
	2.	Head Capacity Matters
	•	A non-linear head significantly improves decision-level performance without retraining the backbone.
	3.	Over-Optimization Risks
	•	Continued tuning without constraints leads to metric chasing and weakens scientific claims.

⸻

Plan Going Forward

Guiding Principle

Iteration 89 is the reference model. All further iterations are controlled, reversible probes.

⸻

Allowed Future Iterations (Optional)

If additional iterations are run, they must follow these strict rules:
	•	Always initialize from Iteration 89 weights
	•	Change only one factor per iteration
	•	Maintain neutral training settings (BCE loss, frozen backbone unless explicitly tested)

Permitted experiments:
	•	Unfreeze the last backbone block for 2–3 epochs (single attempt)
	•	Minor head architecture variants (±25% width, activation change)
	•	Robustness checks (different random seed)

⸻

Hard Stop Conditions

The auto-improvement loop must stop if any of the following occur:
	•	Macro AUROC improvement < 0.003 for two consecutive iterations
	•	Macro F1 drops below Iteration 89
	•	Any change reintroduces Phase 5 heuristics

If no improvement is found, Iteration 89 remains the final model.

⸻

Research Contribution Summary
	•	Demonstrated that decision-level performance (F1) can be improved beyond the baseline paper while maintaining competitive AUROC.
	•	Introduced a clean separation between representation learning and decision calibration.
	•	Provided a reproducible, automated framework for controlled model improvement.

Iteration 89 represents a defensible, well-supported final result suitable for reporting in a thesis or publication.