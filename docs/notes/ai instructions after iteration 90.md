AI Advisory Policy – Locked Anchor + Phase 5 Governance

This policy supersedes all previous advisory rules.

⸻

Locked Anchor Declaration

Iteration 89 is designated as a LOCKED ANCHOR MODEL.

Rules:
	1.	Iteration 89 must never be overridden or replaced.
	2.	All future iterations (90+) must:
	•	Start from Iteration 89 weights only
	•	Change exactly ONE factor at a time
	•	Be explicitly marked as exploratory probes (not final models)
	3.	The goal of future iterations is NOT continuous improvement at any cost, but to test whether small, controlled changes can improve AUROC without degrading F1.

⸻

Phase 5 Governance (CRITICAL)

Phase 5 (TARGETED_AUC / class-specific AUC improvement) is CLOSED and HISTORICAL.

Effective immediately:
	•	Phase 5 must not be used as a validity reference.
	•	Phase 5 must not be used for:
	•	Iteration validation
	•	Parent selection
	•	Role enforcement
	•	Loss selection
	•	Heuristic checks

If an iteration declares:

metadata:
  phase: HEAD_UPGRADE

Then the AI advisory MUST:
	•	Ignore all Phase 5 constraints
	•	Ignore TARGETED_AUC rules
	•	Ignore Iteration 84 as a parent
	•	Ignore hard negatives, focal loss, rare-class logic

Any message such as:
	•	“INVALID for Phase 5”
	•	“Revert to Iteration 84”

is explicitly forbidden for Iterations 87+.

⸻

Allowed Decisions for Iterations 90+

The AI advisory may choose only one of the following actions:
	1.	NEXT_PROBE
	•	Propose a single, minimal change relative to Iteration 89
	•	Examples: head width ±25%, activation change, last-block unfreeze (≤3 epochs)
	2.	STOP_AND_CONSOLIDATE
	•	Declare Iteration 89 as the final model

No other decisions are allowed.

⸻

Hard Stop Conditions

The loop must stop immediately if any of the following occur:
	•	Macro AUROC improvement < 0.003 for two consecutive probes
	•	Macro F1 drops below Iteration 89
	•	Any Phase 5 heuristic is reintroduced

If no clear improvement is found, Iteration 89 remains the final model.

⸻

Guiding Principle

Iteration 89 is the reference.
Phase 5 is history.
Everything else is a controlled, reversible probe.