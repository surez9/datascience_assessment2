# S399630
# Task 2: Average Yellow Cards Comparison

# Q: Did defenders receive significantly more or fewer 
# yellow cards on average at the FIFA World Cup 2026 compared
#  with midfielders?

#Each task follows the required structure:
# analytic question 
# data wrangling 
# sampling 
# descriptive statistics
# 95% confidence interval
# t-test.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.max_columns", 100)

PLAYER_FILE = "dataset/player_stats.csv"
players = pd.read_csv(PLAYER_FILE)

players.shape, players.head(1)


# Data wrangling and preparation 

eligible = players.loc[
    (players["matches_played"] > 0) &
    (players["position"].isin(["DEF", "MID"]))
].copy()

eligible["yellow_cards"] = eligible["yellow_cards"].fillna(0)
eligible["red_cards"] = eligible["red_cards"].fillna(0)

# Stratified sample of n=200, proportional to the DEF/MID population split
def_n = round(len(eligible[eligible.position == "DEF"]) / len(eligible) * 200)
mid_n = 200 - def_n

sample_def = eligible[eligible.position == "DEF"].sample(n=def_n, random_state=4174315)
sample_mid = eligible[eligible.position == "MID"].sample(n=mid_n, random_state=4174315)
sample = pd.concat([sample_def, sample_mid]).sample(frac=1, random_state=4174315)

print("Eligible population:")
print(eligible["position"].value_counts())
print("\nSample:")
print(sample["position"].value_counts())
print("\nSample size:", len(sample))


def mean_ci(x, confidence=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    se = stats.sem(x)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    return mean - margin, mean + margin



# TASK 1: Average Yellow Cards Comparison

# Q: Did defenders receive significantly more or fewer yellow cards
#    on average at the FIFA World Cup 2026 compared with midfielders?

task1_desc = (
    sample.groupby("position")["yellow_cards"]
    .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
    .round(4)
)
task1_desc

for pos in ["DEF", "MID"]:
    x = sample.loc[sample.position == pos, "yellow_cards"]
    lo, hi = mean_ci(x)
    print(f"{pos}: mean={x.mean():.4f}, 95% CI=({lo:.4f}, {hi:.4f})")

def_yc = sample.loc[sample.position == "DEF", "yellow_cards"]
mid_yc = sample.loc[sample.position == "MID", "yellow_cards"]

t_stat, p_value = stats.ttest_ind(def_yc, mid_yc, equal_var=False)

print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Decision: Reject H0. The average yellow-card counts differ significantly.")
else:
    print("Decision: Fail to reject H0. There is no statistically significant difference in average yellow-card counts.")

"""
Using stratified sample, defenders had a lower sample mean
yellow-card count than midfielders. The Welch two-sample t-test produces
p > 0.05, so the difference is not statistically significant.

Therefore, the evidence does not support the claim that defenders received
significantly more or fewer yellow cards than midfielders. The conclusion is
based on the tournament player data after excluding players who never
appeared.
"""

# Diagram: total yellow cards, Defenders vs Midfielders
totals = [def_yc.sum(), mid_yc.sum()]
labels = ["Defenders", "Midfielders"]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.bar(labels, totals, color=["#4C72B0", "#DD8452"], width=0.5)
ax.set_ylim(0, max(totals) * 1.35)
ax.set_ylabel("Total yellow cards (sample, n=200)", fontsize=11)
for i, v in enumerate(totals):
    ax.text(i, v + max(totals) * 0.03, str(int(v)), ha="center", fontsize=16, fontweight="bold")
fig.suptitle("Task 1: Total Yellow Cards \u2014 Defenders vs Midfielders", fontsize=13, fontweight="bold", y=0.99)
verdict = "no significant difference" if p_value >= 0.05 else "significant difference"
ax.set_title(f"Mean-based t-test: {verdict}  (t = {t_stat:.3f}, p = {p_value:.3f})", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"), pad=10)
plt.tight_layout()
plt.savefig("task1_total_yellow_cards.png", dpi=180)
plt.show()


# Yellow-Card Distribution Comparison
# Q: How does the distribution of yellow cards differ between defenders
#    and midfielders, beyond the average?


for pos, x in [("DEF", def_yc), ("MID", mid_yc)]:
    print(f"{pos}: skew={stats.skew(x):.4f}, kurtosis={stats.kurtosis(x):.4f}, "
          f"IQR={np.percentile(x, 75) - np.percentile(x, 25):.4f}")

overall_mean = sample["yellow_cards"].mean()
t3_def, p3_def = stats.ttest_1samp(def_yc, overall_mean)
t3_mid, p3_mid = stats.ttest_1samp(mid_yc, overall_mean)
print(f"One-sample t-test, DEF vs overall mean ({overall_mean:.4f}): t={t3_def:.4f}, p={p3_def:.4f}")
print(f"One-sample t-test, MID vs overall mean ({overall_mean:.4f}): t={t3_mid:.4f}, p={p3_mid:.4f}")

lev_stat, lev_p = stats.levene(def_yc, mid_yc)
print(f"Levene's test for equal variances: stat={lev_stat:.4f}, p={lev_p:.4f}")

"""
Both distributions are heavily right-skewed: nearly every player has zero
cards, with a thin tail of players at one card. Neither group's mean
differs significantly from the pooled outfield-player mean (one-sample
t-tests, both p > 0.05), and Levene's test shows no significant difference
in variance between the two groups (p > 0.05). Defenders and midfielders
therefore show the same overall shape of card distribution, not just the
same average.
"""

# Diagram: card-count distribution (0 vs 1 card), Defenders vs Midfielders
distribution = pd.crosstab(sample["yellow_cards"], sample["position"])[["DEF", "MID"]]

fig, ax = plt.subplots(figsize=(7, 5.5))
distribution.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
for container in ax.containers:
    ax.bar_label(container, fontsize=11, fontweight="bold")
ax.set_xlabel("Yellow cards per player")
ax.set_ylabel("Number of players")
ax.set_xticklabels(distribution.index, rotation=0)
fig.suptitle("Task 3: Card-Count Distribution \u2014 Defenders vs Midfielders", fontsize=13, fontweight="bold", y=0.99)
ax.set_title(f"Levene's test for equal variance: p = {lev_p:.3f} "
             f"({'equal variances' if lev_p >= 0.05 else 'unequal variances'})", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"), pad=10)
plt.tight_layout()
plt.show()


# Average Red Cards Comparison
# Q: Did defenders receive significantly more or fewer red cards
#    on average than midfielders at the FIFA World Cup 2026?


task4_desc = (
    sample.groupby("position")["red_cards"]
    .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
    .round(4)
)
task4_desc

for pos in ["DEF", "MID"]:
    x = sample.loc[sample.position == pos, "red_cards"]
    lo, hi = mean_ci(x)
    print(f"{pos}: mean={x.mean():.4f}, 95% CI=({lo:.4f}, {hi:.4f})")

def_rc = sample.loc[sample.position == "DEF", "red_cards"]
mid_rc = sample.loc[sample.position == "MID", "red_cards"]

t_stat_rc, p_value_rc = stats.ttest_ind(def_rc, mid_rc, equal_var=False)

print(f"t-statistic = {t_stat_rc:.4f}")
print(f"p-value = {p_value_rc:.4f}")

if p_value_rc < 0.05:
    print("Decision: Reject H0. The average red-card counts differ significantly.")
else:
    print("Decision: Fail to reject H0. There is no statistically significant difference in average red-card counts.")

"""
Defenders and midfielders show almost identical average red-card counts in
the sample, and the Welch two-sample t-test produces p > 0.05. The evidence
does not support a significant difference in red cards between the two
positions. Red cards are rare in this dataset, which limits the power of
this test.
"""

# Diagram: total red cards, Defenders vs Midfielders
totals_rc = [def_rc.sum(), mid_rc.sum()]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.bar(labels, totals_rc, color=["#4C72B0", "#DD8452"], width=0.5)
ax.set_ylim(0, max(totals_rc) * 1.35 if max(totals_rc) > 0 else 1)
ax.set_ylabel("Total red cards (sample, n=200)", fontsize=11)
for i, v in enumerate(totals_rc):
    ax.text(i, v + max(totals_rc) * 0.03 + 0.05, str(int(v)), ha="center", fontsize=16, fontweight="bold")
fig.suptitle("Task 4: Total Red Cards \u2014 Defenders vs Midfielders", fontsize=13, fontweight="bold", y=0.99)
verdict_rc = "no significant difference" if p_value_rc >= 0.05 else "significant difference"
ax.set_title(f"Mean-based t-test: {verdict_rc}  (t = {t_stat_rc:.3f}, p = {p_value_rc:.3f})", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"), pad=10)
plt.tight_layout()
plt.show()


# SUMMARY DASHBOARD -- all tasks in one figure

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

panels = [
    (axes[0, 0], totals, "Task 1: Total Yellow Cards", "Total yellow cards", t_stat, p_value),
    (axes[1, 1], totals_rc, "Task 4: Total Red Cards", "Total red cards", t_stat_rc, p_value_rc),
]

for ax, vals, title, ylabel, t_, p_ in panels:
    ax.bar(["DEF", "MID"], vals, color=["#4C72B0", "#DD8452"], width=0.5)
    ax.set_ylim(0, max(vals) * 1.35 if max(vals) > 0 else 1)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.04, str(int(v)), ha="center", fontsize=13, fontweight="bold")
    verdict = "n.s." if p_ >= 0.05 else "sig."
    ax.set_title(f"{title}\nmean-test p = {p_:.3f} ({verdict})", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)

# axes[0, 1] is unused now that Task 2 has moved to a teammate's file
axes[0, 1].axis("off")

ax3 = axes[1, 0]
distribution.plot(kind="bar", ax=ax3, color=["#4C72B0", "#DD8452"], legend=True)
for container in ax3.containers:
    ax3.bar_label(container, fontsize=9)
ax3.set_title(f"Task 3: Card Distribution\nLevene p = {lev_p:.3f}", fontsize=11)
ax3.set_xlabel("Yellow cards per player")
ax3.set_ylabel("Number of players")
ax3.set_xticklabels(distribution.index, rotation=0)

fig.suptitle("Objective 1 Summary: Defenders vs Midfielders \u2014 FIFA World Cup 2026",
             fontsize=15, fontweight="bold", y=1.0)
plt.tight_layout()
plt.show()

print("\nOVERALL CONCLUSION")
print("Position (defender vs midfielder) shows no statistically significant")
print("effect on yellow cards, card-count distribution, or red cards in")
print("this dataset (all p > 0.05).")