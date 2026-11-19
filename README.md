# What Makes Students Skip Classes?

EDS course project, Department of Information Science & Engineering, NMAM Institute of Technology, Nitte.

| Name | USN |
|---|---|
| Aneesh M Shet | NN25ISE029 |
| Anuj Bhat Palimar | NN25ISE034 |
| Anvith Shetty | NN25ISE036 |

**Guide:** Dr. Vandana B S

## Problem
Why do students skip classes, and can their survey answers tell us who is likely to skip often?

## Dataset
94 anonymous Google Form responses (October 2026), 7 multiple-choice questions: main reason for skipping, how often they skip, whether boring lectures reduce motivation, whether poor sleep makes them miss morning classes, perceived academic impact, biggest consequence, and what would improve attendance. No names, marks or real attendance records.

## Method
- **Target:** frequent skipper = "Often" or "Very often" (23 students) vs everyone else (71).
- **Models:** Logistic Regression and Decision Tree (max depth 3), both with balanced class weights, compared using repeated stratified 5-fold cross-validation against a majority-class baseline.
- **Chi-square tests** of each answer against skip frequency.
- **Apriori association rules** (min support 0.10, min confidence 0.50).

## Key results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.695 | 0.412 | 0.577 | 0.466 | 0.709 |
| Decision Tree | 0.537 | 0.265 | 0.498 | 0.329 | 0.535 |
| Majority baseline | 0.756 | 0 | 0 | 0 | 0.500 |

Logistic Regression is the better model. Only "what would improve attendance" was clearly significant in the chi-square tests. See `results/results_summary.md` for the full results and association rules.

## How to run
```bash
pip install -r requirements.txt
python src/eds_model.py     # model, evaluation, chi-square, association rules -> outputs/
python src/make_charts.py   # survey charts -> outputs/charts/
```

## Repository structure
```
data/raw/            original Google Form export
data/processed/      cleaned CSV used by the scripts
src/                 eds_model.py, make_charts.py
outputs/             charts, model outputs, lr_model.json
results/             results_summary.md
report/              project report (PDF and DOCX)
```

## Limitations
Small sample (94), self-reported answers, class imbalance, and no real attendance or marks data, so accuracy is modest.
