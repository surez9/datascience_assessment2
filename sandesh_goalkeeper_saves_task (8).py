# Sandesh's Task: Goalkeeper Saves
#
# Question: Did the busiest goalkeepers at WC2026 average more or less
# than 15 saves each?
#
# Steps: get the data -> pick the sample -> describe it -> confidence
# interval -> one-sample t-test.

import pandas as pd
import numpy as np
from scipy import stats

ALPHA = 0.05          # 5% significance level
HYPOTHESIZED_MEAN = 15  # the benchmark we are testing against


def ci_mean(sample, confidence=0.95):
    # Works out the 95% confidence interval for a mean
    n = len(sample)
    mean = np.mean(sample)
    sem = stats.sem(sample)               # standard error
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


# Load the data
player_stats = pd.read_csv("dataset/player_stats.csv")

# Keep only goalkeepers who actually played a match
gk_pop = player_stats[
    (player_stats["position"] == "GK") & (player_stats["matches_played"] > 0)
].copy()
print(f"Total goalkeepers who played at least one match: {len(gk_pop)}")

# Pick the 10 busiest goalkeepers (most matches played)
n_sample = 10
gk_sample = gk_pop.sort_values(
    ["matches_played", "minutes_played"], ascending=False
).head(n_sample)

print("\nThe 10 goalkeepers we picked:")
print(gk_sample[["player_name", "team_id", "matches_played", "minutes_played", "saves"]]
      .to_string(index=False))

saves = gk_sample["saves"]

# Basic stats: mean, median, spread, range
print(f"\nBasic stats for saves (n={len(saves)}):")
print(f"Mean   : {saves.mean():.3f}")
print(f"Median : {saves.median():.3f}")
print(f"Std dev: {saves.std(ddof=1):.3f}")
print(f"Min/Max: {saves.min():.3f} / {saves.max():.3f}")
print(f"IQR    : {saves.quantile(0.25):.3f} - {saves.quantile(0.75):.3f}")

# 95% confidence interval for the average number of saves
mean_s, lo_s, hi_s = ci_mean(saves)
print(f"\n95% confidence interval for mean saves: {mean_s:.3f}  [{lo_s:.3f}, {hi_s:.3f}]")

# One-sample t-test: is the average different from 15?
t_stat, p_val = stats.ttest_1samp(saves, HYPOTHESIZED_MEAN)
print(f"\nOne-sample t-test against 15 saves:")
print(f"t = {t_stat:.3f}, df = {len(saves)-1}, p = {p_val:.4f}")

if p_val < ALPHA:
    direction = "more" if mean_s > HYPOTHESIZED_MEAN else "fewer"
    print(f"Result: significantly {direction} than 15 saves on average.")
else:
    print("Result: no significant difference from 15 saves on average.")
