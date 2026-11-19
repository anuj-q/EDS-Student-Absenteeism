"""Survey charts for Chapter 2, drawn directly from eds_responses.csv.
Figure A: 2x2 grid of the four ordered questions. Figure B: the two main multiple-choice questions."""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else "data/processed/eds_responses.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/charts"
import os; os.makedirs(out, exist_ok=True)
d = pd.read_csv(src); n = len(d)
L = ["Strongly\ndisagree", "Disagree", "Neutral", "Agree", "Strongly\nagree"]
Lraw = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
BLUE, RED = "#2a6f97", "#c8553d"
def pct(v): return [f"{x}\n({x / n * 100:.1f}%)" for x in v]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
panels = [
 ("freq", "(a) How often do you skip classes?", ["Never", "Rarely", "Sometimes", "Often", "Very often"], None),
 ("boring", "(b) Boring lectures reduce my motivation", Lraw, L),
 ("impact", "(c) Perceived academic impact of skipping", ["Very negative", "Negative", "Neutral", "Positive", "Very positive"],
  ["Very\nnegative", "Negative", "Neutral", "Positive", "Very\npositive"]),
 ("sleep", "(d) Poor sleep makes me skip morning classes", Lraw, L),
]
for ax, (col, title, order, labels) in zip(axes.flat, panels):
    vc = d[col].value_counts().reindex(order)
    colors = [BLUE] * 3 + [RED] * 2 if col == "freq" else [BLUE] * 5
    bars = ax.bar(labels or order, vc.values, color=colors)
    ax.bar_label(bars, labels=pct(vc.values), padding=2, fontsize=9.5)
    ax.set_ylim(0, vc.max() * 1.32); ax.set_title(title, fontsize=12)
    ax.set_ylabel("Responses", fontsize=10); ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(h_pad=2.0); fig.savefig(f"{out}/fig_ordinal.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), gridspec_kw={"height_ratios": [10, 6]})
for ax, (col, title) in zip(axes, [("reason", "(a) Main reason for skipping classes"),
                                   ("improve", "(b) Biggest factor that would improve attendance")]):
    vc = d[col].value_counts().sort_values()
    bars = ax.barh(vc.index, vc.values, color=BLUE)
    ax.bar_label(bars, labels=[f"{x} ({x / n * 100:.1f}%)" for x in vc.values], padding=3, fontsize=10)
    ax.set_xlim(0, vc.max() * 1.22); ax.set_title(title, fontsize=12, loc="left")
    ax.tick_params(axis="y", labelsize=10.5); ax.set_xlabel("Responses", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(h_pad=1.5); fig.savefig(f"{out}/fig_nominal.png", dpi=200); plt.close(fig)
print("saved", out)
