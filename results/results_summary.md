# Results summary: What Makes Students Skip Classes?

This note reports the actual numbers from `eds_model.py` on 94 complete Google Form responses. Nothing here is made up or rounded past what the script printed. All models predict one thing: whether a student is a **frequent skipper** (survey answer *Often* or *Very often*, 23 students, 24.5%) or not (*Never*, *Rarely* or *Sometimes*, 71 students, 75.5%).

The features are the other six answers: main reason for skipping, whether boring lectures reduce motivation, perceived academic impact, biggest consequence, what would improve attendance, and whether poor sleep leads to skipping morning classes.

---

## 1. Which model did better?

Performance was measured with **repeated stratified 5-fold cross-validation** (5 folds × 10 repeats = 50 scores, `random_state=42`). Stratifying keeps the 23:71 class ratio in every fold. Figures are the mean ± standard deviation. Precision, recall and F1 are for the frequent-skipper class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree (max depth 3, balanced) | 0.537 ± 0.134 | 0.265 ± 0.141 | 0.498 ± 0.285 | 0.329 ± 0.160 | 0.535 ± 0.132 |
| Logistic Regression (balanced) | **0.695 ± 0.092** | **0.412 ± 0.184** | **0.577 ± 0.257** | **0.466 ± 0.184** | **0.709 ± 0.144** |
| Majority-class baseline (always “not frequent”) | 0.756 ± 0.023 | 0.000 | 0.000 | 0.000 | 0.500 |

**Logistic regression is the better model.** It beats the decision tree on every metric that matters for finding frequent skippers, and its ROC-AUC of 0.709 is clearly above the 0.50 chance level. The tree’s ROC-AUC of 0.535 is only slightly above chance, and the large standard deviations (especially recall) show how unstable a shallow tree is on 94 rows.

**Neither model beats the baseline on accuracy, and that is expected.** A dummy rule that always says “not a frequent skipper” is already correct for 71 of 94 students (75.5%). Both models use `class_weight="balanced"`, so they trade some of that easy accuracy in order to catch more of the 23 frequent skippers. Look at recall, F1 and ROC-AUC, not accuracy alone.

### Out-of-fold confusion matrices (one 5-fold pass, each student predicted once)

These are a snapshot, not the headline scores. A single 5-fold split can look luckier or unluckier than the 50-score average above, which is why the repeated-CV table is the one to quote.

**Decision tree**

|  | Predicted not frequent | Predicted frequent |
|---|---:|---:|
| Actually not frequent | 42 | 29 |
| Actually frequent | 7 | 16 |

Accuracy 0.617, precision 0.356, recall 0.696, F1 0.471.

**Logistic regression**

|  | Predicted not frequent | Predicted frequent |
|---|---:|---:|
| Actually not frequent | 56 | 15 |
| Actually frequent | 10 | 13 |

Accuracy 0.734, precision 0.464, recall 0.565, F1 0.510.

In this particular split the tree caught more frequent skippers (16 vs 13) but also raised many more false alarms (29 vs 15). Logistic regression is the more balanced of the two.

---

## 2. Decision-tree splits and important features

The tree below was **fitted on all 94 responses** so it can be drawn and explained. It is not a test-set result. Likert and impact answers are coded 0 (lowest) to 4 (highest), so a split at 2.5 means “Neutral or below” versus “Agree / Positive or above”. The Gini `value` numbers in `decision_tree.png` are class-weighted; the student counts here are the raw ones.

**Gini importances** (only features that were actually used in a split):

1. Poor sleep makes me skip morning classes — 0.249
2. Boring lectures reduce my motivation to attend — 0.241
3. Main reason is “Attendance not strictly enforced” — 0.208
4. “Stricter attendance rules” would improve attendance — 0.152
5. Perceived academic impact of skipping — 0.150

**The five splits, in plain language:**

1. **Root — boring lectures.** First split: Neutral or below, versus Agree / Strongly agree.
2. **If lectures are not clearly “boring” (Neutral or below):** next split is whether the main reason is “attendance not strictly enforced”.
    - Reason is *not* that: **18 students, all not frequent** → predict not frequent. This is a clean leaf.
    - Reason *is* that: **6 students (4 not frequent, 2 frequent)** → the tree predicts frequent only because of class weights. The raw majority in this leaf is still “not frequent”. This split should be treated with caution.
3. **If the student Agrees or Strongly agrees that lectures are boring:** next split is poor sleep (Neutral or below, versus Agree / Strongly agree).
4. **Boring + sleep Neutral or below:** split on perceived impact.
    - Impact Neutral or more negative: **22 students (17 not frequent, 5 frequent)** → not frequent.
    - Impact Positive or Very positive: **10 students, all not frequent** → not frequent.
5. **Boring + sleep Agree or Strongly agree:** split on “stricter attendance rules” as the thing that would improve attendance.
    - Did not pick stricter rules: **30 students, 15 and 15** → the tree predicts frequent because of class weights. This is a tie in the raw data, so it is a weak leaf.
    - Did pick stricter rules: **8 students (7 not frequent, 1 frequent)** → not frequent.

Take-away for a viva: the tree keeps coming back to **boring lectures**, **sleep**, and **what would improve attendance**, which is consistent with the logistic-regression coefficients. But several leaves are small or tied, and the cross-validated AUC is only 0.535, so these splits are a sketch of the data, not a strong predictor.

### Logistic-regression coefficients (standardised features)

Largest absolute coefficients, sign tells the direction:

- Boring lectures **+1.118** and poor sleep **+0.796** — the two strongest, more stable signals for frequent skipping.
- Preparing for other exams or assignments **+0.706**; supportive teachers as the attendance fix **+0.604**.
- Part-time work / family duties **−0.730**, health issues **−0.574**, and “stricter attendance rules” as the fix **−0.572** — associated with *not* being a frequent skipper.

Some other large coefficients belong to very rare answers (“Other” for improve, n = 3; “All of the above” and “Depends on the day” as reasons, n = 1 each). Those should not be treated as real findings.

On the two demonstration profiles in `predict_student()` (models fitted on all 94 rows, not a held-out test): a “lectures feel boring, tired, wants a flexible timetable” student is called a frequent skipper by the tree (probability 75.5%) but not by logistic regression (45.2%). A “does not find lectures boring / stricter rules / sleeps all right” student is called not frequent by both, with probability 0.0%.

---

## 3. Chi-square tests

Each question was tested against the original 5-level skip frequency and against the binary frequent-skipper label. Every table has at least one expected count below 5, so the p-values are only approximate.

**Versus 5-level frequency (Never … Very often)**

| Feature | chi-square | df | p-value | Cramér’s V |
|---|---:|---:|---:|---:|
| improve | 38.02 | 20 | **0.0088** | 0.318 |
| boring | 26.28 | 16 | 0.0502 | 0.264 |
| reason | 41.25 | 36 | 0.2519 | 0.331 |
| sleep | 18.33 | 16 | 0.3051 | 0.221 |
| impact | 12.47 | 16 | 0.7109 | 0.182 |
| conseq | 13.90 | 20 | 0.8356 | 0.192 |

**Versus the binary frequent-skipper label**

| Feature | chi-square | df | p-value | Cramér’s V |
|---|---:|---:|---:|---:|
| improve | 21.18 | 5 | **0.0007** | 0.475 |
| boring | 5.35 | 4 | 0.2537 | 0.238 |
| sleep | 4.17 | 4 | 0.3834 | 0.211 |
| reason | 9.26 | 9 | 0.4136 | 0.314 |
| impact | 2.45 | 4 | 0.6539 | 0.161 |
| conseq | 2.18 | 5 | 0.8235 | 0.152 |

Only **what would improve attendance** is clearly related to skip frequency. The share of frequent skippers in that question is:

- Supportive teachers: 5 of 8 (62.5%)
- Other: 3 of 3 (100%, but only three people)
- Practical / industry-relevant content: 8 of 25 (32.0%)
- Interactive teaching methods: 4 of 31 (12.9%)
- Flexible timetable: 2 of 16 (12.5%)
- Stricter attendance rules: 1 of 11 (9.1%)

**Boring lectures** sits right on the usual 0.05 line against 5-level frequency (p = 0.0502) and is not significant against the binary label (p = 0.2537). The direction is still sensible: none of the 9 students who Disagreed or Strongly disagreed were frequent skippers, compared with 12 of 43 who Agreed (27.9%) and 9 of 27 who Strongly agreed (33.3%). An earlier quick test had flagged both “boring” and “improve”; in this exact run, “improve” is the significant one and “boring” is borderline for the 5-level question.

Sleep, perceived impact, main reason, and biggest consequence show no statistically significant link to skip frequency in these tests.

---

## 4. Top association rules (plain sentences)

Apriori was run on all seven answers, one-hot encoded as items (42 unique items). Thresholds: minimum support 0.10 (at least 10 of 94 students) and minimum confidence 0.50. That produced 69 frequent itemsets and **9 rules**. They are sorted by lift (lift > 1 means the answers appear together more often than if they were independent).

**Top 5 by lift:**

1. Of the 13 students who **Disagree** that poor sleep makes them skip morning classes, **10 (76.9%) say they skip Sometimes**. That pairing occurs about twice as often as chance (support 10/94, confidence 0.769, lift 2.009).
2. Of the 29 students whose main reason is **boring or ineffective lectures**, **16 (55.2%) rate the academic impact of skipping as Neutral** (support 16/94, confidence 0.552, lift 1.572).
3. Of the 24 students who name **difficulty catching up** as the biggest consequence, **13 (54.2%) also rate the impact as Neutral** (support 13/94, confidence 0.542, lift 1.543).
4. Of the 20 students who rate the academic impact of skipping as **Positive**, **14 (70.0%) Agree that boring lectures reduce their motivation to attend** (support 14/94, confidence 0.700, lift 1.530).
5. Of the 27 students who **Strongly agree** that boring lectures reduce motivation, **14 (51.9%) rate the impact of skipping as Neutral** (support 14/94, confidence 0.519, lift 1.477).

A rule that speaks more directly to the classifier’s target, a little further down the list: of the 16 students who skip **Often**, **10 (62.5%) Agree that boring lectures reduce motivation** (support 10/94, confidence 0.625, lift 1.366).

These are patterns in the answers, not proof that one thing causes another.

---

## 5. Limitations (please include these in the report)

- **Only 94 responses.** That is a small sample for machine learning. Cross-validation standard deviations are large, and several tree leaves contain fewer than 10 students.
- **Self-reported opinions, not measured behaviour.** The survey does not contain actual attendance records, internal marks, or GPA, so the model predicts what students *say* about skipping, not what the college’s attendance register would show.
- **Modest predictive power, as expected.** The majority baseline already has 75.5% accuracy. Logistic regression’s ROC-AUC of 0.709 is a real but moderate improvement over chance; the decision tree barely beats chance. It would be misleading to present either model as a highly accurate “skip predictor”.
- **Class imbalance and balanced weights.** Using `class_weight="balanced"` helps the models notice the 23 frequent skippers, but it also means some tree leaves predict “frequent” even when the raw majority in that leaf is “not frequent” (or a 15–15 tie).
- **Sparse tables.** Every chi-square table has expected counts below 5, so those p-values are approximate. Rare options such as “All of the above” (1 student) can produce large logistic-regression coefficients that should be ignored.
- **Weak associations overall.** In the chi-square tests, only “what would improve attendance” is clearly related to skip frequency. “Boring lectures” is borderline. Sleep, impact, reason and consequences are not significant. That is why a classifier trained on these six answers cannot be very accurate, and it is an honest finding, not a failure of the method.

The useful result of this project is not a high accuracy score. It is a cautious picture: students who want more supportive teaching or more practical content, and who agree that lectures are boring, are somewhat more likely to describe themselves as frequent skippers, and several of those answers also travel together in the association rules.
