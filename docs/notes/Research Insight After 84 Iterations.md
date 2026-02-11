Research Insight After 84 Iterations

Understanding AUC vs F1 Dynamics in Multi-Label Chest X-Ray Classification

Context

After running 84 automated training iterations on a multi-label chest X-ray classification task, we observed recurring patterns that fundamentally changed how we interpret model improvement. Early iterations suggested that aggressive optimization could yield strong gains in specific metrics, but later iterations revealed deeper structural trade-offs between ranking quality (AUC) and decision quality (F1-score).

This section summarizes the key research insights derived from these experiments.

⸻

Key Empirical Observations
	1.	AUC Improvements Are Class-Specific, Not Global
Across iterations, global Macro AUC remained relatively stable (~0.76–0.77), even when individual diseases showed meaningful improvements. Notably, Hernia, one of the rarest and most challenging classes, exhibited a consistent AUC increase in later iterations, while other classes remained flat.
	2.	F1 and AUC Respond to Different Optimization Forces
Iterations that achieved high F1-scores (e.g., iterations 57–58) did so by aggressively increasing recall through threshold manipulation and loss re-weighting. While effective for F1, these changes introduced substantial noise, increasing false positives and often degrading AUC.
	3.	High AUC Does Not Require High Recall
Iteration 12 achieved the highest Macro AUC (~0.80) despite having near-zero F1 and recall. This demonstrated that strong ranking performance can emerge even when the model is highly conservative and rarely predicts positive labels.
	4.	Later Stable Iterations Represent a True Generalization Regime
Iterations 80–84 consistently converged to similar AUC and F1 values. Although these results did not match the peak AUC observed in iteration 12, their stability across runs suggests they reflect the model’s true generalization capacity under the current pipeline.

⸻

Core Research Insight

Improving AUC and improving F1 are fundamentally different problems.
	•	AUC reflects ranking quality: how well the model orders positive samples above negative ones.
	•	F1 reflects decision quality: how well a fixed threshold balances precision and recall.

Optimizing one metric directly often degrades the other unless the optimization process is explicitly constrained.

⸻

What We Learned from the Hernia Class

The isolated improvement in Hernia AUC provided a crucial clue:
	•	AUC gains were achieved not by forcing more positive predictions, but by reducing noise in the score distribution.
	•	Conservative prediction behavior (fewer high-confidence false positives) led to cleaner ROC curves.
	•	This suggests that noise suppression, rather than signal amplification, is the dominant mechanism for AUC improvement in rare diseases.

This insight is likely transferable to other low-prevalence, high-ambiguity conditions such as Pneumonia, Fibrosis, and Edema.

⸻

Methodological Implications

Based on these findings, future optimization should follow a two-stage paradigm:
	1.	Representation Phase (AUC-focused)
	•	Optimize the model to improve ranking quality.
	•	Avoid threshold tuning, recall forcing, or F1-based early stopping.
	•	Focus on reducing false high scores, especially for hard negatives.
	2.	Calibration Phase (F1-focused)
	•	Freeze model weights.
	•	Optimize decision thresholds and calibration strategies.
	•	Apply class-specific or constrained thresholding to recover F1 without harming AUC.

⸻

Final Takeaway

AUC improvement is inherently local and class-dependent, while F1 improvement is global and decision-dependent.

Treating these objectives as a single optimization problem leads to instability and misleading conclusions. Separating them yields more robust models and clearer scientific understanding.

This insight emerged only after extensive experimentation and is a central contribution of this work.