#!/usr/bin/env python3
"""
EDS Project: What Makes Students Skip Classes?
Frequent-skipper classifier — cross-validation and confusion matrices.

This script is meant to be easy for a student to read and explain in a viva.

What it does
------------
Predict whether a student is a "frequent skipper" (survey answer Often or
Very often) from their other six answers. Compares a shallow Decision Tree
with Logistic Regression, using stratified cross-validation because the
data set is small (94 rows).

Out-of-fold confusion matrices count each student once. The tree picture,
feature importances, logistic-regression coefficients, and the two
predict_student() examples are fitted on all 94 rows so they can be
explained. They are not test-set scores.

Data
----
Reads data/processed/eds_responses.csv from the repository root.
94 Google Form responses. Columns:

    ts       timestamp (not used as a feature)
    reason   main reason for skipping
    freq     how often they skip (the target)
    boring   "Boring lectures reduce my motivation to attend" (Likert)
    impact   perceived academic impact of skipping
    conseq   biggest consequence of skipping
    improve  biggest factor that would improve attendance
    sleep    "Poor sleep makes me skip morning classes" (Likert)

Important limitations (read these before presenting results)
------------------------------------------------------------
* Only 94 self-reported answers. There are no real attendance or marks.
* 23 frequent skippers vs 71 others, so a model that always says
  "not a frequent skipper" is already about 75.5% accurate.
* Modest predictive power is expected. Earlier exploration found that
  only "boring" and "improve" had a clear statistical link to frequency.

How to run
----------
From the repository root:

    python src/eds_model.py

Outputs (written to outputs/): decision_tree.png, confusion_matrices.png,
feature_importance_dt.png, coefficients_lr.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------------------------
# Paths and survey coding
# ---------------------------------------------------------------------------

# This file lives in src/. The repository root is its parent, so
# `python src/eds_model.py` finds the data and the outputs folder
# whether or not the shell was started in that root.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "eds_responses.csv"
OUT_DIR = ROOT / "outputs"

# Full question wording, for printed output and chart titles
QUESTIONS = {
    "reason": "What is the main reason you skip classes?",
    "freq": "How often do you skip classes?",
    "boring": "Boring lectures reduce my motivation to attend.",
    "impact": "Overall, how do you rate the academic impact of skipping classes?",
    "conseq": "What is the biggest consequence of skipping classes?",
    "improve": "What is the biggest factor that would improve your attendance?",
    "sleep": "Poor sleep makes me skip morning classes.",
}

# Ordered categories so "Agree" is higher than "Disagree", and so on.
# OrdinalEncoder will turn these into 0, 1, 2, 3, 4.
LIKERT_ORDER = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]
IMPACT_ORDER = [
    "Very negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very positive",
]
FREQ_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Very often"]

# The two answers we count as "frequent skipper"
FREQUENT_ANSWERS = {"Often", "Very often"}

ORDINAL_FEATURES = ["boring", "impact", "sleep"]
NOMINAL_FEATURES = ["reason", "conseq", "improve"]
FEATURE_COLS = ORDINAL_FEATURES + NOMINAL_FEATURES

# A fixed seed makes the same split and the same tree every time you re-run.
RANDOM_STATE = 42

# Cross-validation: 5 folds, repeated 10 times = 50 scores.
# Repeating is useful when n is only 94, so one unlucky split does not
# dominate the reported numbers.
N_SPLITS = 5
N_REPEATS = 10


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_survey(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and do basic sanity checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Expected data/processed/eds_responses.csv "
            "at the repository root."
        )
    df = pd.read_csv(path)
    expected = ["ts", "reason", "freq", "boring", "impact", "conseq", "improve", "sleep"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if df[FEATURE_COLS + ["freq"]].isna().any().any():
        n_missing = int(df[FEATURE_COLS + ["freq"]].isna().sum().sum())
        raise ValueError(f"Unexpected missing answers ({n_missing} cells).")
    return df


def make_target(freq: pd.Series) -> pd.Series:
    """1 = frequent skipper (Often / Very often), 0 = everyone else."""
    return freq.isin(FREQUENT_ANSWERS).astype(int)


def build_preprocessor() -> ColumnTransformer:
    """
    Encode mixed survey types into numbers the models can use.

    * Likert / impact answers keep their order (ordinal encoding).
    * Multiple-choice reasons, consequences and improvements are
      one-hot encoded (one 0/1 column per option).
    * handle_unknown='ignore' matters in cross-validation: a rare
      option that appears only in a test fold is simply ignored
      instead of crashing the script.
    """
    return ColumnTransformer(
        transformers=[
            (
                "ord",
                OrdinalEncoder(
                    categories=[LIKERT_ORDER, IMPACT_ORDER, LIKERT_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def pretty_feature_name(name: str) -> str:
    """Shorter labels for plots (the raw one-hot names are very long)."""
    prefixes = {
        "boring": "Boring lectures",
        "impact": "Perceived impact",
        "sleep": "Poor sleep (mornings)",
        "reason": "Reason",
        "conseq": "Consequence",
        "improve": "Would improve attendance",
    }
    if name in prefixes:
        return prefixes[name]
    for col, label in prefixes.items():
        key = f"{col}_"
        if name.startswith(key):
            return f"{label}: {name[len(key):]}"
    return name


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """
    Two models, plus a majority-class baseline.

    Decision Tree
        Shallow (max_depth=3) so it stays easy to draw and explain.
        min_samples_leaf=5 stops leaves that only contain one or two
        students, which would overfit a 94-row set.
        class_weight='balanced' gives the 23 frequent skippers more
        weight, otherwise the tree would mostly learn "just guess no".

    Logistic Regression
        Same balanced weights. A StandardScaler sits in front because
        ordinal scores (0-4) and 0/1 dummies are on different scales.
        L2 regularisation (the sklearn default) is kept, which also
        helps when several one-hot columns are correlated.

    Baseline
        Always predicts the majority class ("not a frequent skipper").
        Any useful model should beat this on recall / F1 / AUC, even
        if its raw accuracy is similar or slightly lower.
    """
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)

    return {
        "Decision Tree": Pipeline(
            [("prep", clone(preprocessor)), ("model", tree)]
        ),
        "Logistic Regression": Pipeline(
            [
                ("prep", clone(preprocessor)),
                ("scale", StandardScaler()),
                ("model", logreg),
            ]
        ),
        "Majority baseline": Pipeline(
            [("prep", clone(preprocessor)), ("model", dummy)]
        ),
    }


def scoring_dict() -> dict:
    """Metrics for the positive class (frequent skipper)."""
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Repeated stratified 5-fold CV.

    Stratified means each fold keeps the same 23:71 class ratio.
    We report the mean and standard deviation across all 50 repeats
    so one lucky (or unlucky) fold is not mistaken for the result.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    rows = []
    print("\n" + "=" * 72)
    print("A) CLASSIFIER — frequent skipper vs everyone else")
    print("=" * 72)
    print(
        f"\nTarget: 1 if freq is Often or Very often, else 0."
        f"\nClass counts: frequent skippers = {int(y.sum())} "
        f"({y.mean():.1%}), not frequent = {int((1 - y).sum())} "
        f"({1 - y.mean():.1%})."
        f"\nValidation: Repeated Stratified {N_SPLITS}-fold CV, "
        f"{N_REPEATS} repeats ({N_SPLITS * N_REPEATS} scores), "
        f"random_state={RANDOM_STATE}."
        f"\nMetrics are for the positive class (frequent skipper).\n"
    )

    for name, pipe in models.items():
        with warnings.catch_warnings():
            # DummyClassifier's probability is always 1.0 for class 0,
            # which can trigger a harmless undefined-metric warning.
            warnings.simplefilter("ignore")
            cv_results = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=scoring_dict(),
                n_jobs=1,
            )
        row = {"model": name}
        print(f"{name}")
        print("-" * len(name))
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            scores = cv_results[f"test_{metric}"]
            mean, std = float(scores.mean()), float(scores.std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            print(f"  {metric:10s}  {mean:.3f}  ± {std:.3f}")
        print()
        rows.append(row)
    return pd.DataFrame(rows)


def oof_predictions(pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
    """
    One 5-fold pass of out-of-fold predictions.

    Each student is predicted by a model that did not see that student
    during training. We use a single (not repeated) 5-fold here so the
    confusion matrix counts each person once.
    """
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    try:
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        y_proba = None
    return y_pred, y_proba


def print_oof_confusion(name: str, y: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"Out-of-fold confusion matrix — {name} (single 5-fold, each student once)")
    print(f"                  Pred not frequent    Pred frequent")
    print(f"  Actually not            {tn:5d}              {fp:5d}")
    print(f"  Actually frequent       {fn:5d}              {tp:5d}")
    print(
        f"  accuracy={accuracy_score(y, y_pred):.3f}  "
        f"precision={precision_score(y, y_pred, zero_division=0):.3f}  "
        f"recall={recall_score(y, y_pred, zero_division=0):.3f}  "
        f"F1={f1_score(y, y_pred, zero_division=0):.3f}"
    )
    print()
    return cm


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_confusion_matrices(cms: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    labels = ["Not frequent", "Frequent skipper"]
    for ax, (name, cm) in zip(axes, cms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=11)
    fig.suptitle(
        "Out-of-fold confusion matrices (stratified 5-fold CV)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def plot_decision_tree(pipe: Pipeline, feature_names: list[str], path: Path) -> None:
    tree = pipe.named_steps["model"]
    names = [pretty_feature_name(n) for n in feature_names]
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(
        tree,
        feature_names=names,
        class_names=["Not frequent", "Frequent skipper"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        "Decision tree (max_depth=3, min_samples_leaf=5, class_weight='balanced')\n"
        "Fitted on all 94 responses — use this to explain the splits, "
        "not as a test-set score.",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def plot_tree_importances(pipe: Pipeline, feature_names: list[str], path: Path) -> None:
    tree = pipe.named_steps["model"]
    importances = pd.Series(tree.feature_importances_, index=feature_names)
    importances = importances[importances > 0].sort_values(ascending=True)
    labels = [pretty_feature_name(n) for n in importances.index]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(importances) + 1)))
    ax.barh(labels, importances.values, color="#2a6f97")
    ax.set_xlabel("Gini importance (share of total impurity reduction)")
    ax.set_title("Decision tree — features that were actually used in a split")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")
    return importances.sort_values(ascending=False)


def plot_logreg_coefficients(pipe: Pipeline, feature_names: list[str], path: Path):
    """
    Coefficients are in standardised units (after StandardScaler).
    A positive value raises the log-odds of being a frequent skipper.
    """
    coefs = pd.Series(pipe.named_steps["model"].coef_[0], index=feature_names)
    # Keep the strongest signals so the chart stays readable.
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(15).index)
    top = top.sort_values()
    colors = ["#c1121f" if v < 0 else "#2a6f97" for v in top.values]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh([pretty_feature_name(n) for n in top.index], top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardised features)  →  more likely frequent skipper")
    ax.set_title(
        "Logistic regression — 15 largest |coefficients|\n"
        "Blue: associated with frequent skipping; red: associated with not frequent"
    )
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")
    return coefs.sort_values(key=np.abs, ascending=False)


def likert_threshold_words(feature_name: str, threshold: float) -> str:
    """Translate an encoded split like 'boring <= 2.5' into plain English."""
    # OrdinalEncoder uses 0..4 in the orders declared above.
    if feature_name in ("boring", "sleep", "Poor sleep (mornings)", "Boring lectures"):
        # 2 = Neutral, 3 = Agree, so 2.5 splits Neutral-or-below vs Agree-or-above.
        if abs(threshold - 2.5) < 1e-6:
            return " (i.e. Neutral or below  vs  Agree or Strongly agree)"
    if feature_name in ("impact", "Perceived impact"):
        if abs(threshold - 2.5) < 1e-6:
            return " (i.e. Neutral or more negative  vs  Positive or Very positive)"
    if " <= 0.50" in f"{feature_name} <= {threshold:.2f}" or (
        abs(threshold - 0.5) < 1e-6 and ":" in feature_name
    ):
        return " (0 = no, 1 = yes for this one-hot option)"
    return ""


def describe_tree(pipe: Pipeline, feature_names: list[str], X: pd.DataFrame, y: pd.Series) -> None:
    """Print the tree as if-then rules so it can be quoted in a report."""
    tree = pipe.named_steps["model"]
    t = tree.tree_
    names = [pretty_feature_name(n) for n in feature_names]
    # Raw (unweighted) class counts in each leaf — easier to explain than the
    # class_weight='balanced' masses that plot_tree shows in the 'value' field.
    X_enc = pipe.named_steps["prep"].transform(X)
    leaf_ids = tree.apply(X_enc)
    raw_counts = {}
    for leaf in np.unique(leaf_ids):
        y_leaf = y[leaf_ids == leaf]
        raw_counts[int(leaf)] = (int((y_leaf == 0).sum()), int((y_leaf == 1).sum()))

    print("Decision tree as if-then rules (fitted on all 94 responses):")
    print("Likert / impact answers are coded 0=lowest ... 4=highest.")
    print("A split at 2.5 therefore means 'Neutral or below' vs 'Agree / Positive or above'.")
    print("One-hot features are 0 or 1. The 'value' numbers in the PNG are class-weighted;")
    print("the student counts printed here are the raw numbers.\n")

    def recurse(node: int, indent: str) -> None:
        if t.feature[node] == -2:  # leaf
            n0, n1 = raw_counts[node]
            # Same rule sklearn uses: argmax of (weighted) value; ties go to class 0.
            pred_idx = int(np.argmax(t.value[node][0]))
            pred = "Frequent skipper" if pred_idx == 1 else "Not frequent"
            print(
                f"{indent}→ predict {pred}  "
                f"(leaf has {n0 + n1} students: "
                f"{n0} not frequent, {n1} frequent skippers)"
            )
            return
        feat = names[t.feature[node]]
        thr = t.threshold[node]
        hint = likert_threshold_words(feat, thr)
        print(f"{indent}if {feat} <= {thr:.2f}{hint}:")
        recurse(t.children_left[node], indent + "    ")
        print(f"{indent}else  # {feat} > {thr:.2f}:")
        recurse(t.children_right[node], indent + "    ")

    recurse(0, "")
    print()


# ---------------------------------------------------------------------------
# Example prediction helper
# ---------------------------------------------------------------------------

def predict_student(pipe: Pipeline, answers: dict) -> dict:
    """
    Predict one student from a dict of the six feature answers.

    Returns the predicted label and the model's probability that the
    student is a frequent skipper. The pipeline must already be fitted.
    """
    missing = [c for c in FEATURE_COLS if c not in answers]
    if missing:
        raise ValueError(f"answers is missing: {missing}")
    row = pd.DataFrame([{col: answers[col] for col in FEATURE_COLS}])
    label = int(pipe.predict(row)[0])
    proba = float(pipe.predict_proba(row)[0, 1])
    return {
        "predicted_label": label,
        "predicted_class": "Frequent skipper" if label == 1 else "Not frequent skipper",
        "probability_frequent": proba,
    }


def print_example_predictions(models: dict[str, Pipeline]) -> None:
    """Two made-up students, to show how predict_student() is used."""
    examples = [
        {
            "name": "Student A — lectures feel boring, wants a flexible timetable, tired",
            "answers": {
                "reason": "Tiredness / lack of sleep",
                "boring": "Strongly agree",
                "impact": "Neutral",
                "conseq": "Difficulty catching up",
                "improve": "Flexible timetable",
                "sleep": "Agree",
            },
        },
        {
            "name": "Student B — rarely-skip profile, does not find lectures boring",
            "answers": {
                "reason": "Health issues",
                "boring": "Disagree",
                "impact": "Very negative",
                "conseq": "Lower internal marks / attendance shortage",
                "improve": "Stricter attendance rules",
                "sleep": "Disagree",
            },
        },
    ]
    print("=" * 72)
    print("Example: predict_student() on two made-up answer profiles")
    print("These use the models fitted on all 94 rows, so they are a")
    print("demonstration of the API, not a held-out test result.")
    print("=" * 72 + "\n")
    for ex in examples:
        print(ex["name"])
        for k, v in ex["answers"].items():
            print(f"  {k:8s} {v}")
        for model_name in ["Decision Tree", "Logistic Regression"]:
            result = predict_student(models[model_name], ex["answers"])
            print(
                f"  {model_name:22s}  {result['predicted_class']}"
                f"  (P(frequent) = {result['probability_frequent']:.1%})"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("EDS Project: What Makes Students Skip Classes?")
    print("Cross-validation and confusion matrices  |  94 Google Form responses")
    print(f"Data file: {DATA_PATH}\n")

    df = load_survey()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns. No missing answers in model fields.")
    print("\nSkip frequency counts:")
    for level in FREQ_ORDER:
        n = int((df["freq"] == level).sum())
        flag = "  ← frequent skipper" if level in FREQUENT_ANSWERS else ""
        print(f"  {level:12s} {n:3d}  ({n / len(df):5.1%}){flag}")

    y = make_target(df["freq"])
    X = df[FEATURE_COLS].copy()

    preprocessor = build_preprocessor()
    models = make_models(preprocessor)

    # --- Cross-validated comparison ---
    metrics = evaluate_models(models, X, y)

    print(
        "How to read this: the majority baseline already gets about "
        f"{(y == 0).mean():.1%} accuracy by always saying 'not frequent'.\n"
        "A balanced model is trying to catch the 23 frequent skippers, so its\n"
        "accuracy can be similar or even lower, while recall, F1 and ROC-AUC\n"
        "tell you whether it actually found those students.\n"
    )

    # --- Out-of-fold confusion matrices (one 5-fold, each student once) ---
    cms = {}
    for name in ["Decision Tree", "Logistic Regression"]:
        y_pred, _ = oof_predictions(models[name], X, y)
        cms[name] = print_oof_confusion(name, y, y_pred)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(cms, OUT_DIR / "confusion_matrices.png")

    # --- Fit on all data for the tree picture, importances, coefficients,
    #     and the predict_student() demo. These are explanatory, not test scores. ---
    print("\nFitting both models on all 94 responses for interpretation plots...\n")
    fitted = {}
    for name in ["Decision Tree", "Logistic Regression"]:
        fitted[name] = clone(models[name]).fit(X, y)

    tree_pipe = fitted["Decision Tree"]
    # Feature names after the tree's own preprocessor (no scaler)
    feat_names_tree = list(tree_pipe.named_steps["prep"].get_feature_names_out())
    describe_tree(tree_pipe, feat_names_tree, X, y)
    plot_decision_tree(tree_pipe, feat_names_tree, OUT_DIR / "decision_tree.png")
    importances = plot_tree_importances(
        tree_pipe, feat_names_tree, OUT_DIR / "feature_importance_dt.png"
    )
    print("Decision tree feature importances (only features used in a split):")
    for feat, val in importances.items():
        print(f"  {pretty_feature_name(feat):55s}  {val:.3f}")
    print()

    lr_pipe = fitted["Logistic Regression"]
    feat_names_lr = list(lr_pipe.named_steps["prep"].get_feature_names_out())
    coefs = plot_logreg_coefficients(lr_pipe, feat_names_lr, OUT_DIR / "coefficients_lr.png")
    print("Logistic regression — 10 coefficients with largest absolute value")
    print("(standardised features; positive → more likely a frequent skipper):")
    for feat, val in coefs.head(10).items():
        direction = "more frequent" if val > 0 else "less frequent"
        print(f"  {pretty_feature_name(feat):55s}  {val:+.3f}  ({direction})")
    print()

    print_example_predictions(fitted)

    print("=" * 72)
    print("Done. Figures are in:")
    print(f"  {OUT_DIR}")
    print("=" * 72)

    # metrics is computed for completeness; printed already
    _ = metrics


if __name__ == "__main__":
    main()
