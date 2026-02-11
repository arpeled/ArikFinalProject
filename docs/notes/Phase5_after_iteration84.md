מצוין. עכשיו, אחרי 84 איטרציות ותובנה מחקרית מגובשת, אפשר סוף-סוף לתכנן שלב הבא “נקי” ומבוקר – לא ניסוי-וטעייה.

אני אתן לך תוכנית עבודה ברורה ל־Phase הבא, מחולקת למטרות, צעדים, ותוצרים (כולל מה נכנס לתזה).

⸻

🔜 השלב הבא: Phase 5 – Class-Aware AUC Improvement

🎯 מטרת־על

לשפר AUC במחלות נוספות (מעבר ל-Hernia)
בלי לפגוע ב-AUC הקיים ובלי לחזור ל-F1-hacking של 57–58.

⸻

עיקרון מנחה (שלא שוברים)

לא משנים Representation גלובלי
לא משנים Loss בסיסי
לא נוגעים ב-thresholds בשלב הזה

זה שלב AUC-only, החלטות יגיעו אחר כך.

⸻

שלב 5.1 — מיפוי ממוקד (חובה לפני קוד)

משימה

לבנות טבלה אחת (אפילו CSV פשוט):

Disease	Current AUC	Trend (last 10 iters)	Variance	Priority


חלוקה לקבוצות
	1.	Stable-High AUC – לא נוגעים (Effusion וכו’)
	2.	Mid AUC, Responsive – כאן עובדים (Pneumothorax, Mass, Edema)
	3.	Low AUC, Noisy – עובדים בזהירות (Pneumonia, Fibrosis)

📌 Hernia משמשת כ-Proof of Concept, לא כיעד.

⸻

שלב 5.2 — AUC Improvement בלי שינוי Loss

שינוי יחיד מותר:

👉 Hard-Negative Emphasis (לא Oversampling)

מה עושים בפועל
ב־training loop (או dataset wrapper):
	•	מחשבים logits על batch
	•	מזהים שליליים עם score גבוה (למשל 0.3–0.6)
	•	נותנים להם weight גבוה יותר ב-loss

כלומר:

loss = base_loss
if sample is negative and score > τ:
    loss *= hard_negative_factor

📌 זה:
	•	משפר ranking
	•	מוריד false-high scores
	•	משפר AUC בלי להעלות FP בהחלטות

⸻

שלב 5.3 — Class-Selective Application

לא מפעילים את זה על כולם.

כלל:
	•	Hernia → ❌ לא (כבר השתפרה)
	•	Effusion → ❌ לא (יציבה)
	•	Pneumonia / Fibrosis / Edema → ✅ כן
	•	Pneumothorax / Mass → ✅ כן (עם factor קטן)

📌 זה מונע פגיעה ב-macro AUC בגלל classes שכבר טובים.

⸻

שלב 5.4 — מדידה נכונה (קריטי)

בשלב הזה לא מסתכלים על F1 בכלל.

מדדים:
	•	AUC per class
	•	ΔAUC per class
	•	Macro AUC grouped (לא ממוצע עיוור)

מומלץ:
	•	bootstrap AUC ×5 למחלות נדירות

⸻

שלב 5.5 — עצירה מבוקרת

עוצרים את Phase 5 כשאחד מתקיים:
	•	✔ 2–3 מחלות נוספות שיפרו AUC באופן עקבי
	•	✔ אין ירידה >0.01 במחלה יציבה
	•	✔ Macro AUC לא נפגע

⸻

🔁 Phase 6 (רק אחרי זה) — Decision Calibration מחדש

רק אחרי ש-AUC שופר:
	•	מקפיאים weights
	•	חוזרים ל-threshold optimization
	•	עם אילוצים, לא חופשי כמו 57–58

לדוגמה:

maximize F1
subject to:
  AUC drop <= 0.01
  FP_rate <= baseline × 1.5


⸻

📄 מה נכנס לתזה מהשלב הזה

אתה מרוויח כאן 3 דברים חזקים מאוד:
	1.	Design Principle
“AUC improvement requires noise suppression, not recall forcing.”
	2.	Ablation Study ברור
	•	baseline
	•		•	hard negatives
	•	per-class effect
	3.	הפרדה מתודולוגית נקייה
	•	Phase AUC
	•	Phase Calibration

זה נראה מאוד טוב אקדמית.

⸻

סיכום חד
	•	✔ כן, אפשר ללמוד מהשיפור ב-Hernia
	•	✔ כן, אפשר להכליל
	•	❌ לא דרך שינויים גלובליים
	•	❌ לא דרך F1
	•	✅ דרך class-aware ranking refinement

⸻
