# S399630
# Task 1: Average Yellow Cards Comparison

# Q: Did defenders receive significantly more or fewer 
# yellow cards on average at the FIFA World Cup 2026 compared
#  with midfielders?

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

# Stratified sample: 60% DEF and 40% MID, approximately matching the eligible population
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


# Descriptive statistics
task1_desc = (
    sample.groupby("position")["yellow_cards"]
    .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
    .round(4)
)
task1_desc


# 95% confidence intervals for each group mean
def mean_ci(x, confidence=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    se = stats.sem(x)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    return mean - margin, mean + margin

for pos in ["DEF", "MID"]:
    x = sample.loc[sample.position == pos, "yellow_cards"]
    lo, hi = mean_ci(x)
    print(f"{pos}: mean={x.mean():.4f}, 95% CI=({lo:.4f}, {hi:.4f})")


    # Welch two-sample t-test
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
Using the reproducible stratified sample, defenders had a lower sample mean yellow-card count than midfielders.
the Welch two-sample t-test produces p > 0.05, so the 
difference is not statistically significant.

Therefore, the evidence does not support the claim that 
defenders received significantly more or fewer yellow cards 
than midfielders. The conclusion is based on the tournament 
player data after excluding players who never appeared.
"""

