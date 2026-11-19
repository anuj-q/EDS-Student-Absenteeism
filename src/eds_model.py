#!/usr/bin/env python3
"""
EDS Project: What Makes Students Skip Classes?
Frequent-skipper classifier — loading, encoding, and models.

This script is meant to be easy for a student to read and explain in a viva.

What it does
------------
Loads the processed survey and encodes the mixed Likert and multiple-choice
answers. It then defines two classifiers that predict whether a student is a
"frequent skipper" (survey answer Often or Very often) from the other six
answers:

* a shallow decision tree (max_depth=3, min_samples_leaf=5)
* logistic regression with standardised features

A majority-class baseline is defined next to them. Fitting all three on the
full table checks that the encoding and the estimators run. It is not a
test-set score.

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

How to run
----------
From the repository root:

    python src/eds_model.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Paths and survey coding
# ---------------------------------------------------------------------------

# This file lives in src/. The repository root is its parent, so
# `python src/eds_model.py` finds the processed survey wherever the
# shell was started.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "eds_responses.csv"

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


# ---------------------------------------------------------------------------
# Models
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


def main() -> None:
    print("EDS Project: What Makes Students Skip Classes?")
    print("Frequent-skipper classifier  |  94 Google Form responses")
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
    print(
        f"\nClass counts: frequent skippers = {int(y.sum())} "
        f"({y.mean():.1%}), not frequent = {int((1 - y).sum())} "
        f"({1 - y.mean():.1%})."
    )
    print("\nModels:")
    for name in models:
        print(f"  - {name}")

    # Fitting on every row only checks that encoding and the estimators run.
    print("\nFitting on all responses to check encoding and estimators...")
    for name, pipe in models.items():
        fitted = clone(pipe).fit(X, y)
        n_features = len(fitted.named_steps["prep"].get_feature_names_out())
        print(f"  {name}: ok ({n_features} encoded features)")

    print("\nDone. Encoding and model fitting succeeded.")


if __name__ == "__main__":
    main()
